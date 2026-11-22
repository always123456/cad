"""CPU integration: the real model adapters feed the real evaluator.

Unlike ``tests/test_instance_eval.py`` (dependency-injected, torch-free), this
module builds an actual small Mask R-CNN with ``weights="none"``, saves a
checkpoint with the lane's real custom payload schema, and drives
``instance_eval.batch`` through its **default** adapters:

    model.infer.load_checkpoint  (once)
      -> model.infer.load_png             per PNG
      -> model.infer.predict_arrays       per PNG
      -> model.emit.emit_instances        independent masks, emitter="runs"

then seals every prediction with ``evaluator.prepare`` and checks that nothing
was overwritten.  No dataset, manifest, pretrained asset, GT, GPU or network is
touched: the PNGs and the checkpoint are generated in a temporary directory.

Run:  python tests/test_batch_integration.py        (or `-m unittest`)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
sys.path.insert(0, EVAL_ROOT)

from instance_eval import batch as B  # noqa: E402
from instance_eval import contract as C  # noqa: E402
from instance_eval import evaluator as E  # noqa: E402

import numpy  # noqa: E402  the evaluator requires numpy; always available

try:  # torch/torchvision are optional for the evaluator; required for the model.
    import torch
    import torchvision  # noqa: F401  the sibling model builds a Mask R-CNN
    from PIL import Image

    _torch_error = None
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    Image = None
    _torch_error = "%s: %s" % (type(exc).__name__, exc)


def _png(path, height, width, bar):
    """White PNG with one dark bar; ``bar = (y0, y1, x0, x1)`` in pixels."""

    array = numpy.full((height, width, 3), 255, dtype=numpy.uint8)
    y0, y1, x0, x1 = bar
    array[y0:y1, x0:x1] = 0
    Image.fromarray(array, mode="RGB").save(path)
    return {"path": path, "height": height, "width": width}


@unittest.skipIf(
    _torch_error is not None,
    "torch/torchvision unavailable (%s); run under the model owner's environment"
    % _torch_error,
)
class BatchPipelineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="batch_integration_")
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        cls.seal_root = tempfile.mkdtemp(prefix="batch_seal_", dir=C.LANE_ROOT)
        cls.addClassCleanup(shutil.rmtree, cls.seal_root, True)

        torch.manual_seed(20261114)
        try:
            torch.set_num_threads(2)
        except Exception:  # pragma: no cover - build dependent
            pass

        lane_root = B._model_root()
        if lane_root not in sys.path:
            sys.path.insert(0, lane_root)
        from model import model as detector

        cls.detector = detector
        cls.config = detector.DetectorConfig(
            weights="none",
            seed=20261114,
            inference_min_size=96,
            inference_max_size=96,
        )
        model, info = detector.build_model(cls.config)
        model.eval()
        cls.info = info
        cls.checkpoint = os.path.join(cls.work, "synthetic_checkpoint.pt")
        payload = {
            "schema": "cadvec.direct-instance-maskrcnn/1",
            "config": cls.config.to_dict(),
            "model_info": info,
            "weights_source": "none",
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        torch.save(payload, cls.checkpoint)

        # One square and one non-square fixture: native geometry must survive.
        cls.square = _png(os.path.join(cls.work, "square.png"), 96, 96, (20, 70, 40, 44))
        cls.wide = _png(os.path.join(cls.work, "wide.png"), 96, 128, (20, 70, 60, 64))
        cls.records = [
            {"sample_id": "fixture:square", "png_path": cls.square["path"]},
            {"sample_id": "fixture:wide", "png_path": cls.wide["path"]},
        ]

    # ---------------------------------------------------------------- helpers
    def counting_loader(self, calls):
        def load(checkpoint, device):
            calls.append((checkpoint, device))
            return B.default_loader(checkpoint, device)

        return load

    def envelopes(self, predictions_dir):
        out = {}
        for record in self.records:
            path = os.path.join(predictions_dir, C.slug(record["sample_id"]) + ".json")
            out[record["sample_id"]] = C.load_json(path)
        return out

    # ------------------------------------------------------------------ tests
    def test_default_adapters_produce_sealable_documents(self):
        trace: list = []
        predictions = os.path.join(self.work, "preds_default")
        calls: list = []
        run = B.run_batch(
            predictions,
            checkpoint=self.checkpoint,
            device="cpu",
            inputs=[dict(record) for record in self.records],
            sample_set="explicit",
            score_threshold=0.999,
            loader=self.counting_loader(calls),
            trace=trace,
        )

        self.assertEqual(len(calls), 1, "the real checkpoint must load exactly once")
        self.assertEqual(run["checkpoint_loads"], 1)
        self.assertEqual(run["n_predictions"], 2)
        self.assertEqual(run["png_reads"], 2)
        self.assertFalse(run["seal_written"])
        self.assertFalse(run["gt_read"])
        self.assertFalse(run["svg_read"])
        self.assertEqual([name for name, _ in trace].count("prediction_written"), 2)
        self.assertNotIn("svg", " ".join(value for _name, value in trace).lower())

        # The custom checkpoint schema really is the lane's payload.
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(payload["schema"], "cadvec.direct-instance-maskrcnn/1")
        self.assertEqual(payload["weights_source"], "none")

        for sample_id, envelope in self.envelopes(predictions).items():
            self.assertEqual(envelope["schema"], C.SCHEMA_PREDICTION)
            self.assertEqual(envelope["sample_id"], sample_id)
            document = envelope["document"]
            C.validate_prediction_document(document, sample_id)
            self.assertEqual((document["width"], document["height"]), (100, 100))
            source = envelope["source"]
            expected = self.square if sample_id.endswith("square") else self.wide
            self.assertEqual(
                source["native_size"],
                {"height": expected["height"], "width": expected["width"]},
                "native geometry must survive into the envelope",
            )
            self.assertEqual(source["emitter"], "runs")
            self.assertFalse(source["ink_intersection"])
            self.assertFalse(source["connected_components"])
            self.assertTrue(source["dense_diagnostic_only"])
            self.assertTrue(source["no_ground_truth_read"])
            self.assertEqual(source["mask_threshold"], self.config.mask_threshold)
            self.assertEqual(source["checkpoint_sha256"], C.sha256_file(self.checkpoint))
            # A random weights="none" detector may fire inside the letterbox
            # padding, so no test may assume predictions sit inside 0..100.  What
            # must hold is the exact native -> viewBox affine: every emitted
            # coordinate is a pixel coordinate mapped by the canonical transform.
            scale, offset_x, offset_y = C.submission().viewbox_transform(
                expected["width"], expected["height"], C.VIEWBOX
            )
            for primitive in document["args"]:
                for px in primitive[0::2]:
                    native = float(px) * scale + offset_x
                    self.assertAlmostEqual(native, round(native), delta=1e-6)
                    self.assertGreaterEqual(native, -0.5)
                    self.assertLessEqual(native, expected["width"] + 0.5)
                for py in primitive[1::2]:
                    native = float(py) * scale + offset_y
                    self.assertAlmostEqual(native, round(native), delta=1e-6)
                    self.assertGreaterEqual(native, -0.5)
                    self.assertLessEqual(native, expected["height"] + 0.5)
            outside = sum(
                1
                for primitive in document["args"]
                if any(
                    float(value) < -1e-6 or float(value) > 100.0 + 1e-6
                    for value in list(primitive[0::2]) + list(primitive[1::2])
                )
            )
            self.assertEqual(
                source["out_of_viewbox_primitives"],
                outside,
                "the envelope must report, not clamp, letterbox-padding predictions",
            )

        seal_dir = os.path.join(self.seal_root, "default_adapters")
        seal = E.prepare(predictions, seal_dir, inputs=[dict(r) for r in self.records])
        self.assertEqual(seal["counts"]["predictions"], 2)
        self.assertEqual(seal["counts"]["masks"], 0, "JSON predictions need no masks")
        self.assertEqual(seal["dataset"], "tune-fixed4")
        stored = C.load_json(seal["seal_path"])
        self.assertFalse(stored["gt_read"])
        self.assertFalse(stored["png_read"])
        self.assertFalse(stored["svg_read"])

        # Sealed set is complete: a second batch into the same directory refuses.
        with self.assertRaises(C.GuardError) as caught:
            B.run_batch(
                predictions,
                checkpoint=self.checkpoint,
                device="cpu",
                inputs=[dict(record) for record in self.records],
                sample_set="explicit",
                score_threshold=0.999,
            )
        self.assertEqual(caught.exception.code, "predictions_exist")

    def test_zero_injection_path_also_works(self):
        trace: list = []
        predictions = os.path.join(self.work, "preds_zero_injection")
        run = B.run_batch(
            predictions,
            checkpoint=self.checkpoint,
            device="cpu",
            inputs=[dict(record) for record in self.records],
            sample_set="explicit",
            score_threshold=0.999,
            trace=trace,
        )
        self.assertEqual(run["checkpoint_loads"], 1)
        self.assertEqual([name for name, _ in trace].count("png_read"), 2)
        seal = E.prepare(
            predictions,
            os.path.join(self.seal_root, "zero_injection"),
            inputs=[dict(r) for r in self.records],
        )
        self.assertEqual(seal["counts"]["predictions"], 2)
        for document in (item["document"] for item in self.envelopes(predictions).values()):
            C.validate_prediction_document(document, "zero-injection")

    def test_real_emitter_applies_the_letterbox_affine(self):
        height, width = 96, 128  # non-square: viewBox scale 0.96, letterbox x 16 px
        mask = numpy.zeros((height, width), dtype=bool)
        mask[10, 30:90] = True  # columns 30..89 inclusive; row 10
        emitted = B.default_emit(
            [mask],
            [0],
            [0.99],
            height=height,
            width=width,
            dense=False,
        )
        document = emitted["document"]
        C.validate_prediction_document(document, "fixture:affine")
        # 96x128 -> scale 0.96, letterbox_x 16 px; the runs emitter maps pixel
        # index i to i/scale - letterbox_x/scale, so column 30 -> 14.5833... and
        # the LAST occupied column 89 -> 76.0416666..., never 90.
        xs = [float(value) for value in document["args"][0][0::2]]
        ys = [float(value) for value in document["args"][0][1::2]]
        self.assertAlmostEqual(min(xs), (30.0 - 16.0) / 0.96, delta=1e-9)
        self.assertAlmostEqual(max(xs), (89.0 - 16.0) / 0.96, delta=1e-9)
        self.assertAlmostEqual(min(ys), 10.0 / 0.96, delta=1e-9)
        self.assertAlmostEqual(max(ys), 10.0 / 0.96, delta=1e-9)
        self.assertGreaterEqual(min(xs), 0.0)
        self.assertLessEqual(max(xs), 100.0)
        self.assertEqual(document["instanceIds"], [0])
        self.assertEqual(len(emitted["expected_masks"]), 1)


class RealEmitterGeometryTest(unittest.TestCase):
    """``batch.default_emit`` is the real shared emitter; it needs no torch."""

    def test_shared_emitter_keeps_overlaps_and_applies_the_affine(self):
        height, width = 80, 120  # scale 0.8, letterbox x 20 px
        scale, offset_x, offset_y = C.submission().viewbox_transform(
            width, height, (0.0, 0.0, 100.0, 100.0)
        )
        self.assertAlmostEqual(scale, 0.8)
        self.assertAlmostEqual(offset_x, 20.0)
        first = numpy.zeros((height, width), dtype=bool)
        first[10:20, 40:90] = True
        second = numpy.zeros((height, width), dtype=bool)
        second[15:25, 70:118] = True  # overlaps ``first`` and stays in frame
        stuff = numpy.zeros((height, width), dtype=bool)
        stuff[50:60, 40:99] = True  # pixels 40..98 -> viewBox 25..97.5
        emitted = B.default_emit(
            [first, second, stuff],
            [0, 0, 30],
            [0.9, 0.86, 0.7],
            height=height,
            width=width,
            dense=False,
        )
        document = emitted["document"]
        C.validate_prediction_document(document, "fixture:emitter")
        ids = [int(value) for value in document["instanceIds"]]
        order = []
        for value in ids:
            if not order or order[-1] != value:
                order.append(value)
        self.assertEqual(order, [0, 1, -1], "overlapping Thing masks stay distinct; Stuff merges to -1")
        stuff_args = [args for args, value in zip(document["args"], ids) if value == -1]
        self.assertTrue(stuff_args)
        xs = [float(value) for args in stuff_args for value in args[0::2]]
        self.assertGreaterEqual(min(xs), -0.01)
        self.assertLessEqual(max(xs), 100.01)
        self.assertAlmostEqual(min(xs), (40.0 - offset_x) / scale, delta=1e-9)
        self.assertTrue(emitted["diagnostics"]["independent_masks"])

    def test_shared_emitter_reports_content_outside_the_viewbox(self):
        height, width = 80, 200  # scale 0.8, letterbox x 60 px -> window 60..140
        mask = numpy.zeros((height, width), dtype=bool)
        mask[10:20, 150:190] = True  # entirely right of the letterboxed viewBox
        emitted = B.default_emit(
            [mask], [0], [0.9], height=height, width=width, dense=False
        )
        counts = B._outside_viewbox(emitted["document"])
        self.assertGreater(counts, 0)
        xs = [float(value) for value in emitted["document"]["args"][0][0::2]]
        # 80x200 -> scale 0.8, letterbox_x 60 px: the window is pixels 60..140.
        self.assertAlmostEqual(min(xs), (150.0 - 60.0) / 0.8, delta=1e-9)
        self.assertGreater(min(xs), 100.0)


class SiblingApiContractTest(unittest.TestCase):
    """AST check that the adapters still match the sibling model API.

    Runs without torch: it reads the sibling source, never imports it.  This is
    the early warning if ``model.infer``/``model.emit`` rename a keyword that
    ``batch.default_*`` relies on.
    """

    ROOT = os.path.join(B._model_root(), "model")

    @staticmethod
    def _functions(path):
        import ast

        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    def _keywords(self, filename, function):
        path = os.path.join(self.ROOT, filename)
        for node in self._functions(path):
            if node.name == function:
                args = [arg.arg for arg in node.args.args] + [
                    arg.arg for arg in node.args.kwonlyargs
                ]
                return set(args)
        self.fail("%s not found in %s" % (function, filename))

    def test_infer_api_used_by_the_batch_runner(self):
        self.assertIn("image", self._keywords("infer.py", "predict_arrays"))
        self.assertIn("device", self._keywords("infer.py", "predict_arrays"))
        self.assertIn("score_threshold", self._keywords("infer.py", "predict_arrays"))
        self.assertIn("mask_threshold", self._keywords("infer.py", "predict_arrays"))
        self.assertTrue(
            {"checkpoint", "device"} <= self._keywords("infer.py", "load_checkpoint")
        )
        self.assertEqual(self._keywords("infer.py", "load_png"), {"path"})

    def test_emit_api_used_by_the_batch_runner(self):
        names = self._keywords("emit.py", "emit_instances")
        for required in (
            "masks",
            "class_ids",
            "scores",
            "height",
            "width",
            "prediction_indices",
            "emitter",
            "make_dense_diagnostic",
        ):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
