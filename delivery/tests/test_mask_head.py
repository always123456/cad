"""Tests for ``--mask-head``, ``--mask-chunk`` and ``--drop-stuff``.

Two halves, on purpose:

* everything that can be proved without torch runs everywhere (the system Python
  3.14 lane has numpy but no torch), so the CLI's knob, policy and document
  behaviour is always covered;
* the model tests (parameter count, strict load, chunk invariance) are skipped
  when torch is absent and run in the GPU venv (Python 3.12.15 + torch).

The 28x28 head's own path is covered by asserting that *nothing* new appears in
its policy, documents, selection report or receipt.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

from support import args, fake_loader, fake_run, touch  # noqa: F401  (sys.path via support)

import cad_mrcnn_delivery as delivery

try:  # the 3.14 lane has no torch; the model tests skip there
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None

TORCH_REASON = "torch is not installed in this interpreter (system Python 3.14 lane)"

FINE_728 = os.environ.get("CAD_FINE_HEAD_728", "")
FROZEN_28 = delivery.DEFAULT_CHECKPOINT


def _doc(out):
    results = os.path.join(out, "results")
    names = sorted(name for name in os.listdir(results) if name.endswith(".json"))
    assert len(names) == 1, names
    with open(os.path.join(results, names[0]), encoding="utf-8") as handle:
        return json.load(handle)


def _report(out):
    with open(os.path.join(out, "selection_report.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _receipt(out):
    with open(os.path.join(out, "run_receipt.json"), encoding="utf-8") as handle:
        return json.load(handle)


PER_PRIM = ("args", "commands", "layerIds", "lengths", "rgb", "widths", "semanticIds",
            "instanceIds")


def _strip_stuff(doc):
    """Reference document-level stuff strip (cad-parent-stuffcheck01/strip_stuff.py).

    Stuff classes exist only as ``instanceId == -1`` primitive groups, so removing
    them touches no box and no Thing instance: the document --- not the instance
    set --- is what changes.  ``--drop-stuff`` must produce exactly this.
    """

    out = {
        key: [list(row) if isinstance(row, list) else row for row in value]
        if isinstance(value, list) and key not in PER_PRIM
        else value
        for key, value in doc.items()
    }
    keep = [j for j, value in enumerate(doc["instanceIds"]) if int(value) >= 0]
    for key in PER_PRIM:
        out[key] = [doc[key][j] for j in keep]
    return out


class MaskHeadPolicyTests(unittest.TestCase):
    """Knob validation: torch-free, no checkpoint, no PNG."""

    def test_default_head_is_28_and_absent_from_policy(self):
        block = delivery.validate_mask_head()
        self.assertEqual(block["head"], "28")
        self.assertEqual(block["output_size"], 28)
        self.assertIsNone(block["chunk"])
        self.assertFalse(block["chunk_applied"])
        policy = delivery.validate_selection()
        self.assertNotIn("mask_head", policy)
        self.assertEqual(policy["signature"], [0.05, 0.5, 0.5, 300, None])

    def test_explicit_28_is_also_absent_from_policy(self):
        policy = delivery.validate_selection(drop_stuff=False)
        self.assertNotIn("mask_head", policy)
        self.assertNotIn("drop_stuff", policy)
        self.assertEqual(
            sorted(policy),
            sorted(
                [
                    "order",
                    "score_threshold",
                    "mask_threshold",
                    "nms_iou",
                    "max_detections",
                    "per_image_cap",
                    "comparisons",
                    "applied_in",
                    "refine",
                    "decoder_fixed",
                    "signature",
                ]
            ),
        )

    def test_head_112_block(self):
        block = delivery.validate_mask_head("112")
        self.assertEqual(block["head"], "112")
        self.assertEqual(block["output_size"], 112)
        self.assertEqual(block["parameters"], 796452)
        self.assertEqual(block["chunk"], 24)
        self.assertTrue(block["chunk_applied"])
        self.assertEqual(block["pool_size"], 14)

    def test_head_refused_values(self):
        for bad in ("113", "28x28", "1122", "fine", ""):
            with self.assertRaises(delivery.DeliveryRefused) as caught:
                delivery.validate_mask_head(bad)
            self.assertEqual(caught.exception.code, "mask_head")

    def test_chunk_bounds(self):
        for good in (1, 8, 24, 300, 1024):
            self.assertEqual(delivery.validate_mask_head("112", good)["chunk"], good)
        for bad in (0, -1, 1025, "many"):
            with self.assertRaises(delivery.DeliveryRefused) as caught:
                delivery.validate_mask_head("112", bad)
            self.assertEqual(caught.exception.code, "selection_policy")

    def test_chunk_on_28_head_is_recorded_not_applied(self):
        block = delivery.validate_mask_head("28", 8)
        self.assertIsNone(block["chunk"])
        self.assertEqual(block["chunk_requested"], 8)
        self.assertFalse(block["chunk_applied"])

    def test_mask_head_block_carries_the_measured_numbers(self):
        measured = delivery.validate_mask_head("112")["measured"]
        self.assertAlmostEqual(measured["28"]["pq"], 0.120714704164, places=12)
        self.assertAlmostEqual(measured["112_u250"]["pq"], 0.137443379498, places=12)
        self.assertEqual(measured["112_u250"]["tp"], 370)
        self.assertEqual(measured["112_u250"]["fp"], 540)
        self.assertEqual(measured["112_u250"]["fn"], 2674)

    def test_drop_stuff_block(self):
        self.assertIsNone(delivery.validate_drop_stuff(None))
        self.assertIsNone(delivery.validate_drop_stuff(False))
        block = delivery.validate_drop_stuff(True)
        self.assertEqual(block["classes"], [30, 31, 32, 33, 34])
        self.assertIn("after", block["stage"])
        policy = delivery.validate_selection(drop_stuff=True)
        self.assertEqual(policy["drop_stuff"]["classes"], [30, 31, 32, 33, 34])
        self.assertEqual(policy["signature"], [0.05, 0.5, 0.5, 300, None])

    def test_parser_defaults(self):
        parser = delivery.build_parser()
        parsed = parser.parse_args(["--png", "x.png", "--out", "out"])
        self.assertEqual(parsed.mask_head, "28")
        self.assertEqual(parsed.mask_chunk, 24)
        self.assertFalse(parsed.drop_stuff)
        parsed = parser.parse_args(
            ["--png", "x.png", "--out", "out", "--mask-head", "112", "--mask-chunk", "8",
             "--drop-stuff"]
        )
        self.assertEqual((parsed.mask_head, parsed.mask_chunk, parsed.drop_stuff), ("112", 8, True))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--png", "x.png", "--out", "out", "--mask-head", "56"])


class MaskHeadDryRunTests(unittest.TestCase):
    """The dry run validates the head without importing torch or touching a checkpoint."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="maskhead-")
        self.png = touch(os.path.join(self.tmp, "a.png"), b"not-a-real-png")

    def test_dry_run_112_needs_no_model(self):
        before = set(sys.modules)
        summary = delivery.run_delivery(
            args(png=self.png, out=os.path.join(self.tmp, "out"), mask_head="112", dry_run=True)
        )
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["mask_head"]["head"], "112")
        self.assertEqual(summary["policy"]["mask_head"]["chunk"], 24)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out", "run_receipt.json")))
        # No checkpoint load, no model build: the head is a build-time choice and a
        # dry run only validates it.
        self.assertNotIn("cad_fine_head", set(sys.modules) - before)

    def test_dry_run_28_has_no_mask_head_block(self):
        summary = delivery.run_delivery(
            args(png=self.png, out=os.path.join(self.tmp, "out"), dry_run=True)
        )
        self.assertNotIn("mask_head", summary)
        self.assertNotIn("mask_head", summary["policy"])


class MaskHeadDocumentTests(unittest.TestCase):
    """Document-level behaviour on the fake model: the 28 path, drop-stuff, determinism."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="maskdoc-")
        self.png = touch(os.path.join(self.tmp, "a.png"), b"fixture")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"), b"fixture")

    def test_28_head_output_is_untouched(self):
        """Nothing new in policy, document, selection report or receipt for the 28 head."""

        out = os.path.join(self.tmp, "plain")
        fake_run(out, self.checkpoint, png=self.png)
        report = _report(out)
        policy = report["policy"]
        self.assertNotIn("mask_head", policy)
        self.assertNotIn("drop_stuff", policy)
        self.assertEqual(policy["signature"], [0.05, 0.5, 0.5, 300, None])
        self.assertNotIn("mask_head", _receipt(out))
        self.assertNotIn("stuff_drop", _receipt(out))
        self.assertNotIn("stuff_dropped_instances", _receipt(out)["guards"])
        self.assertNotIn("stuff_drop", report)
        self.assertNotIn("stuff_drop", report["images"][0])
        document = _doc(out)
        # the emitter paints the stuff detection (label 33 -> canonical 32) into the
        # instanceId == -1 primitive tail; it never becomes a box
        self.assertEqual(len(document["boxes"]), 2)
        self.assertEqual(sorted(set(document["semanticIds"])), [0, 1, 32])
        self.assertEqual(len(document["boxes"]), len(document["obj_cts"]))

    def test_explicit_28_matches_the_default_bytes(self):
        plain = os.path.join(self.tmp, "plain")
        explicit = os.path.join(self.tmp, "explicit")
        fake_run(plain, self.checkpoint, png=self.png)
        fake_run(explicit, self.checkpoint, png=self.png, mask_head="28", mask_chunk=8,
                 drop_stuff=False)
        self.assertEqual(_doc(plain), _doc(explicit))
        self.assertEqual(_report(plain)["policy"], _report(explicit)["policy"])

    def test_drop_stuff_matches_a_document_level_strip(self):
        """--drop-stuff produces the same document as stripping the stuff tail afterwards."""

        plain = os.path.join(self.tmp, "plain")
        dropped = os.path.join(self.tmp, "dropped")
        fake_run(plain, self.checkpoint, png=self.png)
        fake_run(dropped, self.checkpoint, png=self.png, drop_stuff=True)

        before = _doc(plain)
        after = _doc(dropped)
        self.assertEqual(len(before["boxes"]), 2)
        self.assertEqual(sorted(set(before["semanticIds"])), [0, 1, 32])
        # stuff never survives as a primitive once its detection is dropped
        self.assertEqual(sorted(set(after["semanticIds"])), [0, 1])
        self.assertNotIn(-1, after["instanceIds"])
        self.assertLess(len(after["instanceIds"]), len(before["instanceIds"]))
        # ... and the survivors are exactly the strip_stuff result
        self.assertEqual(_strip_stuff(before), after)
        self.assertEqual([list(map(float, box)) for box in after["boxes"]],
                         [list(map(float, box)) for box in before["boxes"]])
        entry = _report(dropped)["images"][0]
        self.assertEqual(entry["stuff_drop"]["decoded_in"], 4)
        self.assertEqual(entry["stuff_drop"]["dropped"], 1)
        self.assertEqual(entry["stuff_drop"]["dropped_by_class"], {"32": 1})
        self.assertEqual(entry["stuff_drop"]["decoded_kept"], 3)
        self.assertEqual(entry["stuff_drop"]["selected_in"], 2)
        self.assertEqual(entry["stuff_drop"]["kept"], 2)
        self.assertEqual(_receipt(dropped)["guards"]["stuff_dropped_instances"], 1)
        self.assertIn("drop_stuff", _report(dropped)["policy"])

    def test_drop_stuff_frees_the_per_image_cap(self):
        """cap 2: with the stuff detection in the running, it eats a slot; dropped, it does not."""

        capped = os.path.join(self.tmp, "cap-plain")
        capped_dropped = os.path.join(self.tmp, "cap-dropped")
        fake_run(capped, self.checkpoint, png=self.png, per_image_cap=2)
        fake_run(capped_dropped, self.checkpoint, png=self.png, per_image_cap=2, drop_stuff=True)

        # cap 2 without the drop: the 0.60 stuff detection outranks the 0.10 Thing
        plain = _doc(capped)
        self.assertEqual(len(plain["boxes"]), 1)
        self.assertEqual(sorted(set(plain["semanticIds"])), [0, 32])
        self.assertIn(-1, plain["instanceIds"])

        # cap 2 with the drop: the two Things fill the cap and both become boxes
        doc = _doc(capped_dropped)
        self.assertEqual(len(doc["boxes"]), 2)
        self.assertEqual(sorted(int(box[4]) for box in doc["boxes"]), [0, 1])
        self.assertNotIn(-1, doc["instanceIds"])
        self.assertEqual(_report(capped_dropped)["images"][0]["stuff_drop"]["selected_in"], 2)
        self.assertEqual(_report(capped_dropped)["images"][0]["stuff_drop"]["kept"], 2)

    def test_drop_stuff_leaves_the_document_alone_when_no_stuff_can_survive(self):
        out = os.path.join(self.tmp, "clean")
        high = os.path.join(self.tmp, "high")
        fake_run(out, self.checkpoint, png=self.png, score_threshold=0.7)
        fake_run(high, self.checkpoint, png=self.png, score_threshold=0.7, drop_stuff=True)
        self.assertEqual(_doc(out), _doc(high))
        # the one decoded stuff instance is removed, but it could never have been emitted
        self.assertEqual(_report(high)["images"][0]["stuff_drop"]["dropped"], 1)

    def test_two_runs_are_byte_identical(self):
        first = os.path.join(self.tmp, "run1")
        second = os.path.join(self.tmp, "run2")
        for out in (first, second):
            fake_run(out, self.checkpoint, png=self.png, mask_head="112",
                     refine_mask="a6", refine_ink="composite128", refine_ratio=0.0,
                     drop_stuff=True)
        with open(os.path.join(first, "results", "a.json"), "rb") as handle:
            first_bytes = handle.read()
        with open(os.path.join(second, "results", "a.json"), "rb") as handle:
            second_bytes = handle.read()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(_report(first)["policy"], _report(second)["policy"])
        self.assertEqual(_receipt(first)["policy"], _receipt(second)["policy"])

    def test_injected_loader_wins_over_the_head(self):
        """Tests keep their fake loader: --mask-head 112 never silently loads a model."""

        out = os.path.join(self.tmp, "injected")
        summary = fake_run(out, self.checkpoint, png=self.png, mask_head="112")
        self.assertEqual(summary["status"], "delivery_written")
        self.assertEqual(summary["mask_head"]["head"], "112")
        self.assertEqual(summary["mask_head"]["output_size"], 112)
        # the fake loader carries no mask_head report, so nothing claims a chunked
        # model was built
        self.assertFalse(summary["mask_head"]["chunked"])


@unittest.skipUnless(torch is not None, TORCH_REASON)
class FineHeadModelTests(unittest.TestCase):
    """The 112x112 predictor itself: parameter count, chunk invariance, strict load."""

    def _modules(self):
        fine = delivery._fine_head()
        predictor = fine.FineMaskPredictor()
        from torch import nn
        from torchvision.ops import MultiScaleRoIAlign

        pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)
        head = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
        return fine, predictor, pool, head

    def test_parameter_count(self):
        fine, predictor, _pool, _head = self._modules()
        total = sum(p.numel() for p in predictor.parameters())
        self.assertEqual(total, 796452)
        self.assertEqual(fine.PARAMETERS, 796452)
        self.assertEqual(
            [tuple(predictor.conv5_mask.weight.shape), tuple(predictor.conv6_mask.weight.shape),
             tuple(predictor.conv7_mask.weight.shape), tuple(predictor.mask_fcn_logits.weight.shape)],
            [(256, 256, 2, 2), (256, 256, 2, 2), (256, 256, 2, 2), (36, 256, 1, 1)],
        )

    def test_out_via_token_contract(self):
        fine, predictor, pool, head = self._modules()
        features = {"0": torch.randn(1, 256, 16, 16)}
        proposals = [torch.tensor([[4.0, 4.0, 40.0, 40.0]] * 5)]
        stash = fine._MaskStash()
        runner = fine._ChunkedMaskRunner(pool, head, predictor, stash, 3)
        token = stash(features, proposals, [(64, 64)])
        self.assertEqual(tuple(token.shape), (0,))
        out = runner(token)
        self.assertEqual(tuple(out.shape), (5, 36, 112, 112))

    def test_chunk_8_matches_chunk_24_and_the_unchunked_path(self):
        fine, predictor, pool, head = self._modules()
        torch.manual_seed(7)
        features = {"0": torch.randn(1, 256, 16, 16)}
        boxes = torch.tensor(
            [[4.0, 4.0, 40.0, 40.0], [10.0, 12.0, 50.0, 44.0], [2.0, 6.0, 30.0, 60.0]]
            * 8
        )
        proposals = [boxes]
        shapes = [(64, 64)]

        outputs = {}
        with torch.no_grad():
            direct = predictor(head(pool(features, proposals, shapes)))
            for chunk in (8, 24):
                stash = fine._MaskStash()
                runner = fine._ChunkedMaskRunner(pool, head, predictor, stash, chunk)
                outputs[chunk] = runner(stash(features, proposals, shapes))

        self.assertEqual(tuple(direct.shape), (24, 36, 112, 112))
        for chunk, value in outputs.items():
            self.assertEqual(tuple(value.shape), tuple(direct.shape))
            delta = (value - direct).abs().max().item()
            self.assertLess(delta, 1e-4, "chunk %d vs unchunked" % chunk)
            # the decision the pipeline actually uses must be identical
            self.assertTrue(torch.equal(value > 0.0, direct > 0.0))
            self.assertTrue(torch.equal(torch.sigmoid(value) > 0.5, torch.sigmoid(direct) > 0.5))
        self.assertLess((outputs[8] - outputs[24]).abs().max().item(), 1e-5)

    def test_install_and_restore(self):
        fine, _predictor, pool, head = self._modules()
        from types import SimpleNamespace
        from torch import nn

        model = SimpleNamespace(
            roi_heads=SimpleNamespace(
                mask_roi_pool=pool, mask_head=head, mask_predictor=fine.FineMaskPredictor()
            )
        )
        report = fine.install_chunked_mask(model, chunk=6)
        self.assertTrue(report["installed"])
        self.assertEqual(report["chunk"], 6)
        self.assertIsInstance(model.roi_heads.mask_roi_pool, fine._MaskStash)
        self.assertIsInstance(model.roi_heads.mask_predictor, fine._ChunkedMaskRunner)
        restored = fine.install_chunked_mask(model, chunk=0)
        self.assertFalse(restored["installed"])
        self.assertIs(model.roi_heads.mask_roi_pool, pool)
        self.assertIs(model.roi_heads.mask_head, head)


@unittest.skipUnless(torch is not None, TORCH_REASON)
@unittest.skipUnless(os.path.exists(FROZEN_28), "frozen 28x28 checkpoint is not on this machine")
class FineHeadCheckpointTests(unittest.TestCase):
    """Checkpoint pairing and the strict load, on the real weights (CPU)."""

    def test_28_checkpoint_is_refused_by_the_112_head(self):
        loader = delivery.fine_head_loader(4)
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            loader(FROZEN_28, "cpu")
        self.assertEqual(caught.exception.code, "mask_head")
        self.assertIn("112", str(caught.exception))

    @unittest.skipUnless(FINE_728 and os.path.exists(FINE_728), "112x112 checkpoint is not on this machine")
    def test_fine_loader_context(self):
        loader = delivery.fine_head_loader(8)
        context = loader(FINE_728, "cpu")
        self.assertEqual(context["max_detections"], 300)
        head = context["mask_head"]
        self.assertEqual(head["output_size"], 112)
        self.assertEqual(head["parameters"], 796452)
        self.assertEqual(head["chunk"], 8)
        self.assertTrue(head["chunked"])
        self.assertEqual(len(head["checkpoint_mask_tensors"]), 8)
        self.assertEqual(len(head["fresh_keys"]) + len(head["reused_keys"]), 8)
        self.assertIn("strict", head["load"])
        self.assertTrue(hasattr(context["model"].roi_heads.mask_predictor, "stash"))
        self.assertFalse(context["model"].training)


if __name__ == "__main__":
    unittest.main()
