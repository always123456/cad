"""Synthetic tests for the model-independent instance evaluator.

Run:  python tests/test_instance_eval.py       (stdlib unittest; pytest not installed)

Every test is tiny and meaningful: canonical threshold behaviour, one-to-one
duplicate accounting, missed GT, wrong class, empty denominators, the strict
prediction validator, the pre-GT seal and the reference baseline contract.
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
from instance_eval import metrics as M  # noqa: E402


# --------------------------------------------------------------------------- #
# document builders
# --------------------------------------------------------------------------- #
def line(semantic: int, instance: int, x0: float, x1: float, y: float = 10.0):
    """A 1-px-thick horizontal polyline, three of its four points degenerate."""

    return (semantic, instance, [x0, y, x1, y, x1, y, x1, y])


def doc(*segments) -> dict:
    commands: list[int] = []
    args: list[list[float]] = []
    lengths: list[float] = []
    semantic_ids: list[int] = []
    instance_ids: list[int] = []
    rgb: list[list[int]] = []
    layer_ids: list[int] = []
    widths: list[float] = []
    order: list[int] = []
    semantic_of: dict[int, int] = {}
    members: dict[int, list[int]] = {}

    for semantic, instance, points in segments:
        index = len(args)
        commands.append(0)
        args.append([float(value) for value in points])
        lengths.append(2.0)
        semantic_ids.append(int(semantic))
        instance_ids.append(int(instance))
        rgb.append([200, 30, 40])
        layer_ids.append(1)
        widths.append(1.0)
        if 0 <= semantic <= 29 and instance >= 0:
            if instance not in semantic_of:
                semantic_of[instance] = semantic
                order.append(instance)
                members[instance] = []
            members[instance].append(index)

    obj_cts: list[list[float]] = []
    boxes: list[list[float]] = []
    for instance in order:
        xs: list[float] = []
        ys: list[float] = []
        for index in members[instance]:
            flat = args[index]
            xs.extend([flat[0], flat[2], flat[4], flat[6]])
            ys.extend([flat[1], flat[3], flat[5], flat[7]])
        obj_cts.append([(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, 0, instance])
        boxes.append([min(xs), min(ys), max(xs), max(ys), semantic_of[instance]])

    return {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "width": 100,
        "height": 100,
        "obj_cts": obj_cts,
        "boxes": boxes,
        "rgb": rgb,
        "layerIds": layer_ids,
        "widths": widths,
    }


def empty_doc() -> dict:
    return doc()


def dump(path: str, payload) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


# --------------------------------------------------------------------------- #
# canonical threshold behaviour
# --------------------------------------------------------------------------- #
class ThresholdTests(unittest.TestCase):
    """IoU across 0.5 controls TP: same class, strict ``> 0.5``, one-to-one."""

    def test_iou_above_threshold_matches(self):
        gt = doc(line(0, 1, 10.0, 51.0))
        pred = doc(line(0, 1, 10.0, 31.0))  # ~210/410 px overlap -> IoU ~0.51
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (1, 0, 0))
        self.assertAlmostEqual(report["recall"], 1.0, places=9)
        self.assertGreater(report["PQ"], 0.5)
        self.assertTrue(report["consistency"]["ok"])

    def test_iou_below_threshold_misses(self):
        gt = doc(line(0, 1, 10.0, 51.0))
        pred = doc(line(0, 1, 10.0, 30.0))  # ~200/410 px -> IoU ~0.49
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (0, 1, 1))
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["f1"], 0.0)  # 2*TP/(2*TP+FP+FN) with TP=0 is a real 0.0
        self.assertEqual(report["denominators"]["f1"], 2)
        self.assertEqual(report["PQ"], 0.0)
        self.assertIsNone(report["SQ"])

    def test_exact_half_iou_is_a_strict_miss(self):
        """IoU == 0.5 exactly must not match, checked against the canonical masks."""

        data = C.raster_data()
        gt_doc = doc(line(0, 1, 10.0, 12.0))
        pred_doc = doc(line(0, 1, 10.7, 12.7))
        masks = [
            data._segment_masks(document, C.RASTER_SIZE, C.RASTER_SIZE, C.CANVAS_SIZE)[0]["mask"]
            for document in (gt_doc, pred_doc)
        ]
        gt_mask = masks[0].astype(bool)
        pred_mask = masks[1].astype(bool)
        intersection = int((gt_mask & pred_mask).sum())
        union = int((gt_mask | pred_mask).sum())
        self.assertEqual((intersection, union), (14, 28))
        self.assertEqual(intersection * 2, union)  # exactly 0.5, not 0.49 or 0.51

        report = M.instance_report([gt_doc], [pred_doc])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (0, 1, 1))
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["SumIoU"], 0.0)

    def test_threshold_rule_is_strict_and_canonical(self):
        report = M.instance_report([doc(line(0, 1, 10.0, 20.0))], [doc(line(0, 1, 10.0, 20.0))])
        self.assertEqual(report["iou_threshold"], 0.5)
        self.assertEqual(C.IOU_THRESHOLD, 0.5)
        self.assertEqual(report["assignment"], "hungarian")
        self.assertEqual(tuple(report["canvas_size"]), C.CANVAS_SIZE)
        self.assertEqual(tuple(report["raster_size"]), (C.RASTER_SIZE, C.RASTER_SIZE))
        self.assertEqual(report["policy"]["iou_threshold"], 0.5)
        self.assertEqual(report["policy"]["canvas"], [100.0, 100.0])
        self.assertEqual(report["policy"]["raster"], [1000, 1000])


# --------------------------------------------------------------------------- #
# one-to-one accounting
# --------------------------------------------------------------------------- #
class CountingTests(unittest.TestCase):
    def test_duplicate_predictions_are_false_positives(self):
        gt = doc(line(0, 1, 10.0, 20.0))
        pred = doc(line(0, 1, 10.0, 20.0), line(0, 2, 10.0, 20.0))
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (1, 1, 0))
        self.assertAlmostEqual(report["precision"], 0.5, places=9)
        self.assertAlmostEqual(report["recall"], 1.0, places=9)
        self.assertAlmostEqual(report["f1"], 2 * 1.0 * 0.5 / 1.5, places=9)
        self.assertTrue(report["consistency"]["ok"])

    def test_missed_gt_is_false_negative(self):
        gt = doc(line(0, 1, 10.0, 12.0), line(0, 2, 20.0, 22.0))
        pred = doc(line(0, 1, 10.0, 12.0))
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (1, 0, 1))
        self.assertAlmostEqual(report["recall"], 0.5, places=9)
        self.assertAlmostEqual(report["precision"], 1.0, places=9)
        self.assertEqual(report["denominators"]["recall"], 2)

    def test_wrong_class_is_fp_and_fn(self):
        gt = doc(line(0, 1, 10.0, 20.0))
        pred = doc(line(1, 1, 10.0, 20.0))  # identical geometry, wrong class
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (0, 1, 1))
        self.assertEqual(report["per_class"]["0"]["FN"], 1)
        self.assertEqual(report["per_class"]["1"]["FP"], 1)
        self.assertEqual(report["per_class"]["1"]["group"], "thing")
        self.assertEqual(report["thing"]["recall"], 0.0)
        self.assertIsNone(report["stuff"]["recall"])  # stuff denominator zero

    def test_f1_is_zero_when_tp_is_zero(self):
        fp_only = M.instance_report([empty_doc()], [doc(line(0, 1, 10.0, 20.0))])
        self.assertEqual((fp_only["TP"], fp_only["FP"], fp_only["FN"]), (0, 1, 0))
        self.assertIsNone(fp_only["recall"])  # TP+FN == 0
        self.assertEqual(fp_only["precision"], 0.0)
        self.assertEqual(fp_only["f1"], 0.0)  # 2*0/(0+1+0)
        self.assertEqual(fp_only["denominators"]["f1"], 1)

        fn_only = M.instance_report([doc(line(0, 1, 10.0, 20.0))], [empty_doc()])
        self.assertEqual((fn_only["TP"], fn_only["FP"], fn_only["FN"]), (0, 0, 1))
        self.assertEqual(fn_only["recall"], 0.0)
        self.assertIsNone(fn_only["precision"])  # TP+FP == 0
        self.assertEqual(fn_only["f1"], 0.0)
        self.assertEqual(fn_only["denominators"]["f1"], 1)

    def test_f1_denominator_zero_is_none(self):
        report = M.instance_report([empty_doc()], [empty_doc()])
        self.assertEqual(report["denominators"]["f1"], 0)
        self.assertIsNone(report["f1"])

    def test_empty_denominators_are_none(self):
        gt = doc(line(35, -1, 10.0, 20.0), line(2, -1, 30.0, 40.0))  # ignore + unlabelled Thing
        report = M.instance_report([gt], [empty_doc()])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (0, 0, 0))
        for key in ("PQ", "SQ", "RQ", "recall", "precision", "f1"):
            self.assertIsNone(report[key], key)
        self.assertEqual(report["consistency"]["gt_skipped"], 2)
        self.assertEqual(report["consistency"]["gt_keys"], 0)
        self.assertTrue(report["consistency"]["ok"])

    def test_stuff_is_merged_per_class(self):
        gt = doc(line(30, -1, 10.0, 12.0), line(30, -1, 20.0, 22.0))
        pred = doc(line(30, -1, 10.0, 12.0), line(30, -1, 20.0, 22.0))
        report = M.instance_report([gt], [pred])
        self.assertEqual((report["TP"], report["FP"], report["FN"]), (1, 0, 0))
        self.assertEqual(report["stuff"]["TP"], 1)
        self.assertIsNone(report["thing"]["recall"])


# --------------------------------------------------------------------------- #
# strict validation: predictions vs GT
# --------------------------------------------------------------------------- #
class ValidatorTests(unittest.TestCase):
    def test_prediction_validator_rejects_thing_minus_one_gt_allows(self):
        payload = doc(line(0, -1, 10.0, 20.0), line(35, -1, 30.0, 40.0))
        self.assertIs(C.validate_gt_document(payload, "gt"), payload)
        with self.assertRaises(C.GuardError) as caught:
            C.validate_prediction_document(payload, "pred")
        self.assertEqual(caught.exception.code, "prediction_invalid")
        self.assertIn("thing_instance_negative", caught.exception.detail)

    def test_prediction_validator_rejects_stuff_with_instance(self):
        with self.assertRaises(C.GuardError) as caught:
            C.validate_prediction_document(doc(line(30, 5, 10.0, 20.0)), "pred")
        self.assertIn("stuff_instance_not_minus_one", caught.exception.detail)

    def test_gt_validation_rejects_missing_fields(self):
        payload = doc(line(0, 1, 10.0, 20.0))
        del payload["boxes"]
        with self.assertRaises(C.GuardError) as caught:
            C.validate_gt_document(payload, "gt")
        self.assertEqual(caught.exception.code, "gt_invalid")

    def test_envelope_sample_id_must_match(self):
        payload = {"sample_id": "train_final:other", "document": empty_doc()}
        with self.assertRaises(C.GuardError) as caught:
            C.unwrap_prediction(payload, "train_final:I8wfZ")
        self.assertEqual(caught.exception.code, "prediction_sample_id")


# --------------------------------------------------------------------------- #
# two stage plumbing
# --------------------------------------------------------------------------- #
class LaneCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.mkdtemp(prefix="instance_eval_", dir=C.LANE_ROOT)
        self.addCleanup(shutil.rmtree, self.work, True)

    def path(self, *parts: str) -> str:
        return os.path.join(self.work, *parts)

    def make_inputs(self, docs: dict[str, dict]) -> list[dict]:
        png_dir = self.path("png")
        os.makedirs(png_dir, exist_ok=True)
        inputs = []
        for index, sample_id in enumerate(docs):
            png = os.path.join(png_dir, C.slug(sample_id) + ".png")
            dump(png, {})
            svg = os.path.join(png_dir, C.slug(sample_id) + ".svg")
            dump(svg, {})
            inputs.append(
                {
                    "sample_id": sample_id,
                    "group_id": "group-%d" % index,
                    "png_path": png,
                    "svg_path": svg,
                }
            )
        return inputs

    def write_predictions(self, docs: dict[str, dict], directory: str | None = None) -> str:
        directory = directory or self.path("preds")
        os.makedirs(directory, exist_ok=True)
        for sample_id, document in docs.items():
            dump(os.path.join(directory, C.slug(sample_id) + ".json"), document)
        return directory

    def prepare_lane(self, docs: dict[str, dict], **kwargs) -> tuple[str, dict]:
        out = self.path("seal")
        inputs = kwargs.pop("inputs", None) or self.make_inputs(docs)
        predictions = kwargs.pop("predictions", None) or self.write_predictions(docs)
        result = E.prepare(predictions, out, inputs=inputs, **kwargs)
        return out, result

    @staticmethod
    def builder(docs: dict[str, dict]):
        def build(record):
            return {"gt_document": docs[record["sample_id"]]}

        return build


class SealTests(LaneCase):
    def test_prepare_seals_before_gt_and_score_orders(self):
        docs = {sample_id: doc(line(0, 1, 10.0 + index, 20.0 + index)) for index, sample_id in enumerate(C.SAMPLE_IDS)}
        out, result = self.prepare_lane(docs)
        seal = C.load_json(result["seal_path"])
        self.assertFalse(seal["gt_read"])
        self.assertFalse(seal["svg_read"])
        self.assertFalse(seal["png_read"])
        self.assertEqual(seal["sample_ids"], list(C.SAMPLE_IDS))
        self.assertEqual(len(seal["files"]), 4)
        self.assertEqual(seal["counts"]["samples"], 4)

        trace: list[tuple[str, str]] = []
        report = E.score(out, allow_gt=True, gt_builder=self.builder(docs), trace=trace)
        self.assertEqual(report["metrics"]["n_images"], 4)
        self.assertAlmostEqual(report["metrics"]["recall"], 1.0, places=9)
        self.assertAlmostEqual(report["metrics"]["PQ"], 1.0, places=9)
        self.assertTrue(report["success"]["recall_pass"])
        self.assertTrue(report["success"]["gt_leak_free"])
        # No reference baseline supplied: recall alone is not a full pass.
        self.assertFalse(report["success"]["passed"])
        self.assertEqual(report["success"]["status"], "recall_only_reference_unavailable")
        self.assertTrue(os.path.isfile(self.path("seal", E.REPORT_NAME)))

        verified = [index for index, (name, _) in enumerate(trace) if name == "prediction_verified"]
        built = [index for index, (name, _) in enumerate(trace) if name == "gt_built"]
        self.assertEqual(len(verified), 4)
        self.assertEqual(len(built), 4)
        self.assertLess(max(verified), min(built), "GT was built before every prediction was verified")

    def test_prepare_refuses_second_seal(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out, _ = self.prepare_lane(docs)
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(self.path("preds"), out, inputs=self.make_inputs(docs))
        self.assertEqual(caught.exception.code, "seal_exists")

    def test_prepare_refuses_sealed_predictions_dir(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        bad = self.path("eval_pool")
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(bad, self.path("seal"), inputs=self.make_inputs(docs))
        self.assertEqual(caught.exception.code, "sealed_path")

    def test_prepare_refuses_outside_lane(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(
                self.write_predictions(docs),
                os.path.join(tempfile.gettempdir(), "outside_lane_seal"),
                inputs=self.make_inputs(docs),
            )
        self.assertEqual(caught.exception.code, "outside_lane")

    def test_score_refuses_without_grant(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out, _ = self.prepare_lane(docs)
        with self.assertRaises(C.GuardError) as caught:
            E.score(out, gt_builder=self.builder(docs))
        self.assertEqual(caught.exception.code, "gt_not_authorized")
        self.assertFalse(os.path.exists(os.path.join(out, E.REPORT_NAME)))

    def test_score_refuses_stale_prediction(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out, result = self.prepare_lane(docs)
        target = self.path("preds", C.slug(C.SAMPLE_IDS[0]) + ".json")
        dump(target, doc(line(0, 1, 10.0, 20.0), line(0, 2, 30.0, 40.0)))
        with self.assertRaises(C.GuardError) as caught:
            E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertEqual(caught.exception.code, "artifact_changed")
        self.assertFalse(os.path.exists(os.path.join(out, E.REPORT_NAME)))

    def test_score_refuses_missing_prediction(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out, _ = self.prepare_lane(docs)
        os.remove(self.path("preds", C.slug(C.SAMPLE_IDS[0]) + ".json"))
        with self.assertRaises(C.GuardError) as caught:
            E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertIn(caught.exception.code, ("artifact_missing", "artifact_changed"))

    def test_prepare_accepts_bare_and_enveloped_json(self):
        sample_a, sample_b = C.SAMPLE_IDS[0], C.SAMPLE_IDS[1]
        docs = {
            sample_a: doc(line(0, 1, 10.0, 20.0)),
            sample_b: {"sample_id": sample_b, "document": doc(line(0, 1, 10.0, 20.0))},
        }
        predictions = self.path("preds")
        os.makedirs(predictions)
        dump(os.path.join(predictions, C.slug(sample_a) + ".json"), docs[sample_a])
        dump(os.path.join(predictions, sample_b + ".json"), docs[sample_b])
        out = self.path("seal")
        result = E.prepare(predictions, out, inputs=self.make_inputs(docs))
        self.assertEqual(result["counts"]["samples"], 2)
        gt = {sample_id: payload.get("document", payload) for sample_id, payload in docs.items()}
        report = E.score(out, allow_gt=True, gt_builder=self.builder(gt))
        self.assertAlmostEqual(report["metrics"]["recall"], 1.0, places=9)


# --------------------------------------------------------------------------- #
# optional masks
# --------------------------------------------------------------------------- #
class MapTests(LaneCase):
    def _maps(self, semantic, instance, width=100, height=100):
        numpy = C.require_numpy()
        return {
            "semantic": numpy.asarray(semantic, dtype=numpy.uint8),
            "instance": numpy.asarray(instance, dtype=numpy.int32),
            "source_size": numpy.asarray([width, height], dtype=numpy.int32),
        }

    def _fixture(self):
        numpy = C.require_numpy()
        # 200x100 source: the viewBox is letterboxed at x 50..150, so every
        # seeded region must live inside that window to survive the affine.
        semantic = numpy.full((100, 200), 35, dtype=numpy.uint8)
        instance = numpy.full((100, 200), -1, dtype=numpy.int32)
        semantic[10:30, 60:90] = 0
        instance[10:30, 60:90] = 1
        semantic[50:60, 110:130] = 30
        return semantic, instance

    def test_maps_rederivation_matches_and_drift_is_refused(self):
        numpy = C.require_numpy()
        sample_id = C.SAMPLE_IDS[0]
        semantic, instance = self._fixture()
        document, affine = C.document_from_maps(semantic, instance, 200, 100)
        self.assertEqual(set(affine), set(C.AFFINE_FIELDS))
        self.assertEqual(affine["pixels_per_view_unit"], 1.0)
        self.assertEqual(affine["letterbox_x"], 50.0)
        self.assertEqual(affine["offset_x"], -50.0)
        self.assertEqual(affine["offset_y"], 0.0)
        self.assertEqual(
            C.submission().validate_document(document), [], "derived document must be valid"
        )

        docs = {sample_id: document}
        maps_dir = self.path("maps")
        os.makedirs(maps_dir)
        bundle = self._maps(semantic, instance, width=200)
        numpy.savez_compressed(os.path.join(maps_dir, C.slug(sample_id) + ".npz"), **bundle)
        predictions = self.write_predictions(docs)

        out = self.path("seal")
        result = E.prepare(
            predictions, out, inputs=self.make_inputs(docs), maps_dir=maps_dir
        )
        self.assertEqual(result["counts"]["maps"], 1)
        seal = C.load_json(result["seal_path"])
        self.assertTrue(seal["samples"][0]["map_sha256"])

        report = E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertAlmostEqual(report["metrics"]["PQ"], 1.0, places=9)

        tampered = json.loads(json.dumps(document))
        tampered["rgb"][0] = [1, 2, 3]
        predictions_b = self.path("preds_b")
        os.makedirs(predictions_b)
        dump(os.path.join(predictions_b, C.slug(sample_id) + ".json"), tampered)
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(
                predictions_b,
                self.path("seal_b"),
                inputs=self.make_inputs(docs),
                maps_dir=maps_dir,
            )
        self.assertEqual(caught.exception.code, "document_mismatch")
        self.assertIn("rgb", caught.exception.detail)

    def test_maps_reject_thing_without_instance(self):
        numpy = C.require_numpy()
        semantic = numpy.zeros((4, 4), dtype=numpy.uint8)
        instance = numpy.full((4, 4), -1, dtype=numpy.int32)
        with self.assertRaises(C.GuardError) as caught:
            C.validate_prediction_maps(semantic, instance, (4, 4))
        self.assertEqual(caught.exception.code, "map_identity")


# --------------------------------------------------------------------------- #
# reference baseline contract
# --------------------------------------------------------------------------- #
class BaselineTests(LaneCase):
    def _baseline(self, name: str, payload: dict) -> str:
        path = self.path(name)
        dump(path, payload)
        return path

    def test_reference_baseline_comparability(self):
        matching = self._baseline(
            "ok.json",
            {
                "schema": C.SCHEMA_BASELINE,
                "metric": "PQ",
                "value": 0.5,
                "source": "FloorPlanCAD baseline (source recorded by the parent)",
                "normalization": dict(C.REFERENCE_NORMALIZATION),
            },
        )
        record = C.load_reference_baseline(matching)
        self.assertEqual(record["status"], "available")
        self.assertTrue(record["comparable"])
        self.assertEqual(record["value"], 0.5)
        self.assertTrue(record["sha256"])

        scaled = self._baseline(
            "scaled.json",
            {
                "value": 0.5,
                "source": "pixel-scale baseline",
                "normalization": dict(C.REFERENCE_NORMALIZATION, raster=[500, 500]),
            },
        )
        record = C.load_reference_baseline(scaled)
        self.assertEqual(record["status"], "incomparable_normalization")
        self.assertFalse(record["comparable"])
        self.assertIn("value:raster", record["mismatches"])

        missing_scale = self._baseline("noscale.json", {"value": 0.5, "source": "unknown scale"})
        self.assertEqual(C.load_reference_baseline(missing_scale)["status"], "unavailable_normalization")

        missing_value = self._baseline(
            "novalue.json",
            {"source": "TBD", "normalization": dict(C.REFERENCE_NORMALIZATION)},
        )
        self.assertEqual(C.load_reference_baseline(missing_value)["status"], "unavailable_value")

        with self.assertRaises(C.GuardError) as caught:
            C.load_reference_baseline(self._baseline("nosource.json", {"value": 0.5}))
        self.assertEqual(caught.exception.code, "baseline_provenance")

    def test_success_requires_recall_and_reference(self):
        docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out, _ = self.prepare_lane(docs)

        no_reference = E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertFalse(no_reference["success"]["passed"])
        self.assertEqual(no_reference["success"]["pq_reference_status"], "no_reference_baseline")
        self.assertEqual(no_reference["scope"]["role"], "tune")
        self.assertFalse(no_reference["scope"]["heldout"])
        self.assertTrue(no_reference["success"]["recall_pass"])

        good = self._baseline(
            "good.json",
            {
                "value": 0.5,
                "source": "recorded baseline",
                "normalization": dict(C.REFERENCE_NORMALIZATION),
            },
        )
        passed = E.score(
            out,
            allow_gt=True,
            gt_builder=self.builder(docs),
            reference_baseline=good,
            report_path=self.path("report_good.json"),
        )
        self.assertTrue(passed["success"]["passed"])
        self.assertEqual(passed["success"]["status"], "passed")

        strict = self._baseline(
            "strict.json",
            {
                "value": 1.5,
                "source": "unreachable baseline",
                "normalization": dict(C.REFERENCE_NORMALIZATION),
            },
        )
        failed = E.score(
            out,
            allow_gt=True,
            gt_builder=self.builder(docs),
            reference_baseline=strict,
            report_path=self.path("report_strict.json"),
        )
        self.assertFalse(failed["success"]["passed"])
        self.assertEqual(failed["success"]["status"], "pq_below_reference")

    def test_low_recall_fails_even_with_reference(self):
        gt = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0), line(0, 2, 40.0, 50.0))}
        predictions = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        out = self.path("seal")
        E.prepare(
            self.write_predictions(predictions),
            out,
            inputs=self.make_inputs(gt),
        )
        good = self._baseline(
            "good.json",
            {
                "value": 0.1,
                "source": "low baseline",
                "normalization": dict(C.REFERENCE_NORMALIZATION),
            },
        )
        report = E.score(
            out,
            allow_gt=True,
            gt_builder=self.builder(gt),
            reference_baseline=good,
        )
        self.assertAlmostEqual(report["metrics"]["recall"], 0.5, places=9)
        self.assertEqual(report["success"]["status"], "recall_below_target")
        self.assertFalse(report["success"]["passed"])


# --------------------------------------------------------------------------- #
# provider: PNG-only predictions -> official 12-field documents
# --------------------------------------------------------------------------- #
class ProviderTests(LaneCase):
    """A model that emits masks only still yields scorable documents."""

    def _pair(self, semantic, instance, directory=None):
        numpy = C.require_numpy()
        from PIL import Image  # only the prediction side is ever opened here

        directory = directory or self.path("masks")
        os.makedirs(directory, exist_ok=True)
        sample_id = C.SAMPLE_IDS[0]
        semantic_path = os.path.join(directory, C.slug(sample_id) + ".semantic.png")
        instance_path = os.path.join(directory, C.slug(sample_id) + ".instance.png")
        Image.fromarray(numpy.asarray(semantic, dtype=numpy.uint8), mode="L").save(semantic_path)
        Image.fromarray(numpy.asarray(instance, dtype=numpy.uint8), mode="L").save(instance_path)
        return directory, semantic_path, instance_path

    @staticmethod
    def _masks():
        numpy = C.require_numpy()
        semantic = numpy.full((100, 200), 35, dtype=numpy.uint8)
        instance = numpy.full((100, 200), 0, dtype=numpy.uint8)
        semantic[10:30, 60:90] = 0
        instance[10:30, 60:90] = 1
        semantic[50:60, 110:130] = 30
        return semantic, instance

    def test_png_provider_matches_canonical_npz_derivation(self):
        from instance_eval import provider as P

        numpy = C.require_numpy()
        semantic, instance = self._masks()
        directory, semantic_path, instance_path = self._pair(semantic, instance)
        result = P.document_from_mask_files(semantic_path, instance_path)
        expected_instance = numpy.where(
            semantic <= C.THING_MAX, instance.astype(numpy.int32), -1
        )
        expected, expected_affine = C.document_from_maps(
            semantic, expected_instance.astype(numpy.int32), 200, 100
        )
        # instance PNGs cannot carry -1; the provider must rebuild it from the
        # semantic map, so the document is identical to the npz derivation.
        self.assertEqual(
            C.canonical_json(result["document"]), C.canonical_json(expected)
        )
        self.assertEqual(result["affine"], expected_affine)
        self.assertEqual(result["source_size"], [200, 100])
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["thing_instances"], 1)
        self.assertEqual(diagnostics["classes"], [0, 30])
        self.assertEqual(diagnostics["outside_letterbox_pixels"], 0)
        self.assertFalse(diagnostics["empty"])
        provider = P.MaskProvider(directory)
        self.assertEqual(
            C.canonical_json(provider({"sample_id": C.SAMPLE_IDS[0]})["document"]),
            C.canonical_json(expected),
        )
        self.assertEqual(
            C.canonical_json(P.derive_documents(directory, [{"sample_id": C.SAMPLE_IDS[0]}])[C.SAMPLE_IDS[0]]),
            C.canonical_json(expected),
        )

    def test_provider_reports_lost_pixels_outside_letterbox(self):
        from instance_eval import provider as P

        numpy = C.require_numpy()
        semantic, instance = self._masks()
        semantic[40:50, 0:20] = 0  # left of the 50..150 letterbox window
        instance[40:50, 0:20] = 1
        _, semantic_path, instance_path = self._pair(semantic, instance)
        diagnostics = P.document_from_mask_files(semantic_path, instance_path)["diagnostics"]
        self.assertEqual(diagnostics["outside_letterbox_pixels"], 200)

    def test_provider_rejects_colour_and_mismatched_masks(self):
        from instance_eval import provider as P

        numpy = C.require_numpy()
        from PIL import Image

        directory = self.path("masks")
        os.makedirs(directory, exist_ok=True)
        semantic_path = os.path.join(directory, C.slug(C.SAMPLE_IDS[0]) + ".semantic.png")
        instance_path = os.path.join(directory, C.slug(C.SAMPLE_IDS[0]) + ".instance.png")
        colour = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        colour[0, 0] = [10, 20, 30]
        Image.fromarray(colour, mode="RGB").save(semantic_path)
        Image.fromarray(numpy.zeros((8, 8), dtype=numpy.uint8), mode="L").save(instance_path)
        with self.assertRaises(C.GuardError) as caught:
            P.load_mask_pngs(semantic_path, instance_path)
        self.assertEqual(caught.exception.code, "mask_not_index")

        Image.fromarray(numpy.zeros((8, 8), dtype=numpy.uint8), mode="L").save(semantic_path)
        Image.fromarray(numpy.zeros((4, 8), dtype=numpy.uint8), mode="L").save(instance_path)
        with self.assertRaises(C.GuardError) as caught:
            P.load_mask_pngs(semantic_path, instance_path)
        self.assertEqual(caught.exception.code, "mask_shape_mismatch")

    def test_provider_thing_instance_zero_is_a_real_instance(self):
        from instance_eval import provider as P

        numpy = C.require_numpy()
        semantic = numpy.full((20, 20), 35, dtype=numpy.uint8)
        instance = numpy.zeros((20, 20), dtype=numpy.uint8)
        semantic[4:16, 4:16] = 7
        instance[4:16, 4:16] = 0
        _, semantic_path, instance_path = self._pair(semantic, instance)
        document = P.document_from_mask_files(semantic_path, instance_path)["document"]
        self.assertEqual(C.submission().validate_document(document), [])
        self.assertEqual(set(document["instanceIds"]), {0})
        self.assertEqual(set(document["semanticIds"]), {7})


# --------------------------------------------------------------------------- #
# mask-only provider end to end: derive -> seal -> score
# --------------------------------------------------------------------------- #
class MaskPrepareTests(LaneCase):
    def _write_masks(self, sample_ids=None):
        numpy = C.require_numpy()
        from PIL import Image

        sample_ids = sample_ids or [C.SAMPLE_IDS[0], C.SAMPLE_IDS[1]]
        directory = self.path("masks")
        os.makedirs(directory, exist_ok=True)
        for index, sample_id in enumerate(sample_ids):
            semantic = numpy.full((100, 100), 35, dtype=numpy.uint8)
            instance = numpy.zeros((100, 100), dtype=numpy.uint8)
            x0 = 10 + 5 * index
            semantic[10:20, x0:x0 + 15] = 0
            instance[10:20, x0:x0 + 15] = index + 1
            Image.fromarray(semantic, mode="L").save(
                os.path.join(directory, C.slug(sample_id) + ".semantic.png")
            )
            Image.fromarray(instance, mode="L").save(
                os.path.join(directory, C.slug(sample_id) + ".instance.png")
            )
        return directory, sample_ids

    def _derived(self, out, sample_ids):
        return {
            sample_id: C.load_json(os.path.join(out, "predictions", C.slug(sample_id) + ".json"))
            for sample_id in sample_ids
        }

    def test_mask_only_round_trip_seals_pngs_and_scores(self):
        mask_dir, sample_ids = self._write_masks()
        inputs = self.make_inputs({sample_id: {} for sample_id in sample_ids})
        out = self.path("seal")
        result = E.prepare_from_masks(mask_dir, out, inputs=inputs)
        self.assertEqual(result["origin"], "prediction_masks")
        self.assertEqual(result["counts"], {
            "samples": 2,
            "predictions": 2,
            "maps": 0,
            "masks": 2,
            "primitives": result["counts"]["primitives"],
            "thing_instances": 2,
        })
        seal = C.load_json(result["seal_path"])
        self.assertEqual(seal["dataset"], "tune-fixed4")
        self.assertFalse(seal["png_read"], "prediction PNGs are not dataset PNGs")
        kinds = sorted(entry["kind"] for entry in seal["files"].values())
        self.assertEqual(
            kinds,
            sorted(
                ["prediction", "prediction"]
                + ["prediction_mask_instance", "prediction_mask_semantic"] * 2
            ),
        )
        self.assertTrue(seal["samples"][0]["diagnostics"]["primitives"] > 0)

        docs = self._derived(out, sample_ids)
        report = E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertAlmostEqual(report["metrics"]["recall"], 1.0, places=9)
        self.assertAlmostEqual(report["metrics"]["PQ"], 1.0, places=9)
        self.assertEqual(report["metrics"]["n_images"], 2)
        self.assertEqual(report["scope"]["origin"], "prediction_masks")

    def test_score_refuses_tampered_mask_png(self):
        mask_dir, sample_ids = self._write_masks()
        inputs = self.make_inputs({sample_id: {} for sample_id in sample_ids})
        out = self.path("seal")
        E.prepare_from_masks(mask_dir, out, inputs=inputs)
        docs = self._derived(out, sample_ids)
        numpy = C.require_numpy()
        from PIL import Image

        semantic = numpy.full((100, 100), 35, dtype=numpy.uint8)
        semantic[30:40, 30:40] = 0
        Image.fromarray(semantic, mode="L").save(
            os.path.join(mask_dir, C.slug(sample_ids[0]) + ".semantic.png")
        )
        with self.assertRaises(C.GuardError) as caught:
            E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertEqual(caught.exception.code, "artifact_changed")
        self.assertFalse(os.path.exists(os.path.join(out, E.REPORT_NAME)))

    def test_prepare_refuses_json_that_disagrees_with_its_png_masks(self):
        mask_dir, sample_ids = self._write_masks()
        inputs = self.make_inputs({sample_id: {} for sample_id in sample_ids})
        from instance_eval import provider as P

        docs = P.derive_documents(mask_dir, [{"sample_id": sample_id} for sample_id in sample_ids])
        tampered = json.loads(json.dumps(docs[sample_ids[0]]))
        tampered["layerIds"][0] = 9
        docs[sample_ids[0]] = tampered
        predictions = self.write_predictions(docs)
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(
                predictions,
                self.path("seal_drift"),
                inputs=inputs,
                masks_dir=mask_dir,
            )
        self.assertEqual(caught.exception.code, "document_mismatch")
        self.assertIn("layerIds", caught.exception.detail)


# --------------------------------------------------------------------------- #
# prepared documents from any future sample set (FIT / TUNE)
# --------------------------------------------------------------------------- #
class DatasetTests(LaneCase):
    def test_prepare_from_documents_seals_a_labelled_set(self):
        sample_ids = [C.SAMPLE_IDS[0], C.SAMPLE_IDS[1], C.SAMPLE_IDS[2]]
        docs = {
            sample_id: doc(line(0, 1, 10.0 + index, 20.0 + index))
            for index, sample_id in enumerate(sample_ids)
        }
        inputs = self.make_inputs(docs)
        out = self.path("seal")
        result = E.prepare_from_documents(
            docs, out, inputs=inputs, dataset="fit16-diagnostic", origin="staged_documents"
        )
        self.assertEqual(result["dataset"], "fit16-diagnostic")
        self.assertEqual(result["counts"]["samples"], 3)
        seal = C.load_json(result["seal_path"])
        self.assertEqual(seal["origin"], "staged_documents")
        self.assertTrue(all(os.path.isfile(sample["prediction"]) for sample in seal["samples"]))

        report = E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertEqual(report["scope"]["dataset"], "fit16-diagnostic")
        self.assertEqual(report["scope"]["n_images"], 3)
        self.assertEqual(report["scope"]["gt_instances_scored"], 3)
        self.assertTrue(report["scope"]["exposed_samples"])
        self.assertFalse(report["scope"]["heldout"])
        self.assertEqual(report["scope"]["role"], "fit")
        self.assertEqual(report["scope"]["role_source"], "dataset label 'fit16-diagnostic'")
        self.assertIn("capacity FIT memorisation diagnostic", report["scope"]["claim"])
        self.assertIn("not held-out generalisation", report["scope"]["claim"])

    def test_prepare_from_documents_refuses_missing_or_extra(self):
        no_docs = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        inputs = self.make_inputs(no_docs)
        extra = dict(no_docs)
        extra[C.SAMPLE_IDS[3]] = doc(line(0, 1, 10.0, 20.0))
        with self.assertRaises(C.GuardError) as caught:
            E.prepare_from_documents(extra, self.path("seal_extra"), inputs=inputs)
        self.assertEqual(caught.exception.code, "documents_mismatch")
        with self.assertRaises(C.GuardError) as caught:
            E.prepare_from_documents({}, self.path("seal_empty"), inputs=inputs)
        self.assertEqual(caught.exception.code, "documents_missing")

    @unittest.skipUnless(os.path.isfile(C.MANIFEST_PATH), "public manifest not present")
    def test_fit16_is_disjoint_from_the_fixed_four(self):
        fit = C.fit_inputs(count=16)
        ids = [row["sample_id"] for row in fit]
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 16)
        self.assertFalse(set(ids) & set(C.SAMPLE_IDS), "FIT diagnostic overlaps the TUNE four")
        self.assertEqual(len({row["group_id"] for row in fit}), 16)
        fixed = C.select_inputs(count=4)
        self.assertEqual([row["sample_id"] for row in fixed], list(C.SAMPLE_IDS))
        with self.assertRaises(C.GuardError) as caught:
            C.select_inputs(count=1000)
        self.assertEqual(caught.exception.code, "inputs_short")

    @unittest.skipUnless(os.path.isfile(C.MANIFEST_PATH), "public manifest not present")
    def test_fit16_seal_uses_the_same_api(self):
        fit = C.fit_inputs(count=16)
        docs = {
            row["sample_id"]: doc(line(0, 1, 10.0, 20.0))
            for row in fit
        }
        out = self.path("seal_fit")
        result = E.prepare_from_documents(
            docs, out, inputs=fit, dataset="fit16-diagnostic"
        )
        self.assertEqual(result["counts"]["samples"], 16)
        report = E.score(out, allow_gt=True, gt_builder=self.builder(docs))
        self.assertEqual(report["metrics"]["n_images"], 16)
        self.assertEqual(report["scope"]["dataset"], "fit16-diagnostic")


# --------------------------------------------------------------------------- #
# independent numeric expectations (hand computed, not recomputed)
# --------------------------------------------------------------------------- #
class IndependentNumericTests(unittest.TestCase):
    """Exact fractions worked out by hand from the canonical definitions.

    GT  : class 0 lines A(10-30,y10), B(10-30,y40), C(40-60,y60); class 30 line D(10-30,y80)
    PRED: A and B exactly; C's geometry tagged class 1; one extra class 0 line at y90
    TP=3 (A,B,D)  FP=2 (wrong class + extra)  FN=1 (C)  SumIoU=3.0
    PQ = 3/(3+1+0.5) = 2/3   SQ = 1   RQ = 2/3
    recall = 3/4 = 0.75      precision = 3/5 = 0.6   F1 = 2*0.75*0.6/1.35 = 2/3
    thing: TP2 FP2 FN1 -> PQ = 2/3.5 = 0.571428571429, precision = 0.5
    stuff: TP1 FP0 FN0 -> PQ = 1
    class 1: TP0 FP1 FN0 -> recall None (empty denominator), precision 0, PQ 0
    """

    def _case(self):
        gt = doc(
            line(0, 1, 10.0, 30.0, 10.0),
            line(0, 2, 10.0, 30.0, 40.0),
            line(0, 3, 40.0, 60.0, 60.0),
            line(30, -1, 10.0, 30.0, 80.0),
        )
        pred = doc(
            line(0, 1, 10.0, 30.0, 10.0),
            line(0, 2, 10.0, 30.0, 40.0),
            line(1, 1, 40.0, 60.0, 60.0),
            line(30, -1, 10.0, 30.0, 80.0),
            line(0, 4, 40.0, 60.0, 90.0),
        )
        return M.instance_report([gt], [pred])

    def test_hand_computed_counts_and_scores(self):
        report = self._case()
        self.assertEqual(report["TP"], 3)
        self.assertEqual(report["FP"], 2)
        self.assertEqual(report["FN"], 1)
        self.assertEqual(report["SumIoU"], 3.0)
        self.assertAlmostEqual(report["PQ"], 0.666666666667, places=12)
        self.assertAlmostEqual(report["SQ"], 1.0, places=12)
        self.assertAlmostEqual(report["RQ"], 0.666666666667, places=12)
        self.assertAlmostEqual(report["recall"], 0.75, places=12)
        self.assertAlmostEqual(report["precision"], 0.6, places=12)
        self.assertAlmostEqual(report["f1"], 0.666666666667, places=12)

    def test_hand_computed_groups_and_classes(self):
        report = self._case()
        thing = report["thing"]
        self.assertEqual((thing["TP"], thing["FP"], thing["FN"]), (2, 2, 1))
        self.assertAlmostEqual(thing["PQ"], 0.571428571429, places=12)
        self.assertAlmostEqual(thing["recall"], 0.666666666667, places=12)
        self.assertAlmostEqual(thing["precision"], 0.5, places=12)
        stuff = report["stuff"]
        self.assertEqual((stuff["TP"], stuff["FP"], stuff["FN"]), (1, 0, 0))
        self.assertAlmostEqual(stuff["PQ"], 1.0, places=12)

        class_thing = report["per_class"]["0"]
        self.assertEqual((class_thing["TP"], class_thing["FP"], class_thing["FN"]), (2, 1, 1))
        self.assertAlmostEqual(class_thing["PQ"], 0.666666666667, places=12)
        class_wrong = report["per_class"]["1"]
        self.assertEqual((class_wrong["TP"], class_wrong["FP"], class_wrong["FN"]), (0, 1, 0))
        self.assertIsNone(class_wrong["recall"], "TP+FN is zero: recall must be None")
        self.assertEqual(class_wrong["precision"], 0.0)
        self.assertEqual(class_wrong["PQ"], 0.0)
        self.assertAlmostEqual(report["per_class"]["30"]["PQ"], 1.0, places=12)

    def test_identities_from_the_hand_computed_case(self):
        from fractions import Fraction

        report = self._case()
        denominator = Fraction(2 * report["TP"] + report["FP"] + report["FN"], 2)
        self.assertEqual(Fraction(report["TP"]) / denominator, Fraction(2, 3))
        self.assertEqual(Fraction(report["TP"]) / Fraction(report["TP"] + report["FN"]), Fraction(3, 4))
        self.assertEqual(Fraction(report["TP"]) / Fraction(report["TP"] + report["FP"]), Fraction(3, 5))
        # PQ is not the only identity: the group split must reconstitute the sum.
        self.assertEqual(
            report["thing"]["TP"] + report["stuff"]["TP"], report["TP"]
        )
        self.assertEqual(
            report["thing"]["FP"] + report["stuff"]["FP"], report["FP"]
        )
        self.assertEqual(
            report["thing"]["FN"] + report["stuff"]["FN"], report["FN"]
        )
        self.assertTrue(report["consistency"]["ok"])


# --------------------------------------------------------------------------- #
# overlapping instance JSON with an optional winner-take-all map (diagnostic)
# --------------------------------------------------------------------------- #
class DiagnosticMaskTests(LaneCase):
    """Dense maps are diagnostics: overlap cannot survive a per-pixel map."""

    def _overlapping_case(self):
        numpy = C.require_numpy()
        from PIL import Image

        sample_id = C.SAMPLE_IDS[0]
        # Two independent instances on the same pixels: legal in JSON, impossible
        # in a single winner-take-all mask.
        documents = {sample_id: doc(line(0, 1, 10.0, 30.0), line(0, 2, 10.0, 30.0))}
        mask_dir = self.path("masks")
        os.makedirs(mask_dir, exist_ok=True)
        semantic = numpy.full((100, 100), 35, dtype=numpy.uint8)
        instance = numpy.zeros((100, 100), dtype=numpy.uint8)
        semantic[10, 10:30] = 0
        instance[10, 10:30] = 1  # the winner keeps instance 1 only
        Image.fromarray(semantic, mode="L").save(
            os.path.join(mask_dir, C.slug(sample_id) + ".semantic.png")
        )
        Image.fromarray(instance, mode="L").save(
            os.path.join(mask_dir, C.slug(sample_id) + ".instance.png")
        )
        return documents, mask_dir, sample_id

    def test_diagnostic_masks_tolerate_overlap_drift(self):
        documents, mask_dir, sample_id = self._overlapping_case()
        predictions = self.write_predictions(documents)
        inputs = self.make_inputs(documents)
        out = self.path("seal")

        with self.assertRaises(C.GuardError) as caught:
            E.prepare(
                predictions,
                out,
                inputs=inputs,
                masks_dir=mask_dir,
                mask_role="authoritative",
            )
        self.assertEqual(caught.exception.code, "document_mismatch")

        result = E.prepare(
            predictions, out, inputs=inputs, masks_dir=mask_dir, mask_role="diagnostic"
        )
        seal = C.load_json(result["seal_path"])
        sample = seal["samples"][0]
        self.assertEqual(sample["mask_role"], "diagnostic")
        self.assertTrue(sample["mask_drift"], "expected recorded drift, not a refusal")
        self.assertIn("instanceIds", sample["mask_drift"])
        self.assertEqual(seal["counts"]["masks"], 1)
        self.assertTrue(any("diagnostic winner map" in note for note in seal["caveats"]))

        report = E.score(out, allow_gt=True, gt_builder=self.builder(documents))
        self.assertEqual(report["metrics"]["n_images"], 1)
        self.assertEqual((report["metrics"]["TP"], report["metrics"]["FP"]), (2, 0))
        self.assertEqual(report["metrics"]["recall"], 1.0)
        self.assertTrue(any(sample_id in note for note in report["attachment_notes"]))
        self.assertTrue(report["success"]["gt_leak_free"])


class ExplicitInputsTests(LaneCase):
    """The model owns its sample list and order; the evaluator assumes neither."""

    def test_explicit_inputs_keep_the_model_order(self):
        sample_ids = ["train_final:5Mnth", "train_final:M5Xaz", "train_final:WP1Nr"]
        documents = {
            sample_id: doc(line(0, index + 1, 10.0 + index, 20.0 + index))
            for index, sample_id in enumerate(sample_ids)
        }
        inputs = list(reversed(self.make_inputs(documents)))
        predictions = self.write_predictions(documents)
        out = self.path("seal")
        result = E.prepare(
            predictions,
            out,
            inputs=inputs,
            dataset="fit16-diagnostic",
            origin="model_json",
        )
        self.assertEqual(result["sample_ids"], [item["sample_id"] for item in inputs])
        self.assertEqual(result["dataset"], "fit16-diagnostic")

        report = E.score(out, allow_gt=True, gt_builder=self.builder(documents))
        self.assertEqual(report["metrics"]["n_images"], 3)
        self.assertEqual([item["sample_id"] for item in report["metrics"]["per_image"]], result["sample_ids"])
        self.assertEqual(report["scope"]["role"], "fit")

    def test_explicit_inputs_refuse_duplicate_ids(self):
        documents = {C.SAMPLE_IDS[0]: doc(line(0, 1, 10.0, 20.0))}
        inputs = self.make_inputs(documents)
        inputs = inputs + [dict(inputs[0])]
        predictions = self.write_predictions(documents)
        with self.assertRaises(C.GuardError) as caught:
            E.prepare(predictions, self.path("seal"), inputs=inputs)
        self.assertEqual(caught.exception.code, "inputs_duplicate")


# --------------------------------------------------------------------------- #
# PNG-only batch runner: one checkpoint load, independent masks, no seal/GT
# --------------------------------------------------------------------------- #
class BatchTests(LaneCase):
    """The producer stage: PNGs in, prediction envelopes out, nothing else."""

    @staticmethod
    def _image(size=(100, 100)):
        numpy = C.require_numpy()
        return numpy.full((size[1], size[0], 3), 255, dtype=numpy.uint8)

    def png(self, sample_id: str, size=(100, 100)) -> str:
        from PIL import Image

        directory = self.path("png")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, C.slug(sample_id) + ".png")
        Image.fromarray(self._image(size), mode="RGB").save(path)
        return path

    def records(self, sample_ids) -> list[dict]:
        return [
            {"sample_id": sample_id, "group_id": "g-%d" % index, "png_path": self.png(sample_id)}
            for index, sample_id in enumerate(sample_ids)
        ]

    def checkpoint(self) -> str:
        path = self.path("direct_instance.pt")
        with open(path, "wb") as handle:
            handle.write(b"synthetic-checkpoint-bytes")
        return path

    @staticmethod
    def png_loader(path: str):
        numpy = C.require_numpy()
        from PIL import Image

        with Image.open(path) as image:
            array = numpy.asarray(image.convert("RGB")).astype(numpy.float32) / 255.0
        height, width = array.shape[:2]
        return array.transpose(2, 0, 1), numpy.zeros((height, width), dtype=bool), {
            "path": path,
            "width": width,
            "height": height,
        }

    @staticmethod
    def detections(plan):
        """plan: list of (class_id_canonical, score, bool mask) per detection."""

        numpy = C.require_numpy()

        def predict(_context, _image):
            labels = [int(item[0]) + 1 for item in plan]
            scores = [float(item[1]) for item in plan]
            masks = [numpy.asarray(item[2], dtype=bool) for item in plan]
            return {
                "labels": numpy.asarray(labels, dtype=numpy.int64),
                "scores": numpy.asarray(scores, dtype=numpy.float32),
                "mask_probabilities": numpy.asarray(masks, dtype=numpy.float32),
                "mask_threshold": 0.5,
                "score_threshold": 0.05,
            }

        return predict

    def loader(self, calls):
        def load(path, device):
            calls.append((path, device))
            return {"model": object(), "device": device, "max_detections": 300}

        return load

    def bar(self, y=10, x0=10, x1=30, size=(100, 100)):
        numpy = C.require_numpy()
        mask = numpy.zeros((size[1], size[0]), dtype=bool)
        mask[y, x0:x1] = True
        return mask

    def test_all_predictions_complete_before_seal_with_one_load(self):
        sample_ids = list(C.SAMPLE_IDS[:2])
        records = self.records(sample_ids)
        plan = [(0, 0.9, self.bar())]
        load_calls: list = []
        trace: list = []
        run = B.run_batch(
            self.path("predictions"),
            checkpoint=self.checkpoint(),
            inputs=records,
            sample_set="explicit",
            loader=self.loader(load_calls),
            predict=self.detections(plan),
            emit=B.default_emit,  # the real shared emitter, no torch needed
            png_loader=self.png_loader,
            trace=trace,
        )

        self.assertEqual(run["status"], "predictions_written")
        self.assertEqual(len(load_calls), 1, "the checkpoint must load exactly once")
        self.assertEqual(run["checkpoint_loads"], 1)
        self.assertEqual(run["n_predictions"], 2)
        self.assertEqual(run["sample_ids"], sample_ids)
        self.assertFalse(run["seal_written"])
        self.assertFalse(run["gt_read"])
        self.assertFalse(run["svg_read"])

        events = [name for name, _ in trace]
        self.assertEqual(events[0], "checkpoint_loaded")
        self.assertEqual(events.count("prediction_written"), 2)
        self.assertLess(events.index("predicted"), events.index("prediction_written"))
        self.assertNotIn("seal", " ".join(events))
        self.assertFalse(
            any(E.SEAL_NAME in files for _root, _dirs, files in os.walk(self.work)),
            "the runner must not seal anything",
        )

        emitted: dict[str, dict] = {}
        for sample_id in sample_ids:
            path = os.path.join(run["predictions_dir"], C.slug(sample_id) + ".json")
            payload = C.load_json(path)
            self.assertEqual(payload["schema"], C.SCHEMA_PREDICTION)
            document, _extra = C.unwrap_prediction(payload, sample_id)
            C.validate_prediction_document(document, sample_id)
            self.assertEqual(payload["source"]["emitter"], "runs")
            self.assertTrue(payload["source"]["no_ground_truth_read"])
            self.assertEqual(payload["source"]["png_path"], records[0]["png_path"] if sample_id == sample_ids[0] else records[1]["png_path"])
            emitted[sample_id] = document

        # The complete set seals, then scores against the GT of the same shapes.
        seal = E.prepare(run["predictions_dir"], self.path("seal"), inputs=records)
        self.assertEqual(seal["counts"]["predictions"], 2)
        self.assertEqual(seal["counts"]["masks"], 0)
        report = E.score(self.path("seal"), allow_gt=True, gt_builder=self.builder(emitted))
        self.assertEqual(report["metrics"]["TP"], 2)
        self.assertEqual(report["metrics"]["recall"], 1.0)

    def test_overlapping_instances_survive_the_batch(self):
        numpy = C.require_numpy()
        sample_ids = [C.SAMPLE_IDS[0]]
        mask = self.bar()
        records = self.records(sample_ids)
        run = B.run_batch(
            self.path("predictions"),
            checkpoint=self.checkpoint(),
            inputs=records,
            sample_set="explicit",
            loader=self.loader([]),
            predict=self.detections([(0, 0.9, mask), (0, 0.8, mask.copy())]),
            emit=B.default_emit,
            png_loader=self.png_loader,
        )
        path = os.path.join(run["predictions_dir"], C.slug(sample_ids[0]) + ".json")
        document, _ = C.unwrap_prediction(C.load_json(path), sample_ids[0])
        instances = sorted(int(value) for value in document["instanceIds"])
        self.assertEqual(len(instances), 2, "both overlapping masks must be emitted")
        self.assertEqual(len(set(instances)), 2)
        self.assertTrue(all(value >= 0 for value in instances))
        self.assertEqual(set(int(value) for value in document["semanticIds"]), {0})

        E.prepare(run["predictions_dir"], self.path("seal"), inputs=records)
        report = E.score(
            self.path("seal"), allow_gt=True, gt_builder=self.builder({sample_ids[0]: document})
        )
        self.assertEqual((report["metrics"]["TP"], report["metrics"]["FP"]), (2, 0))
        self.assertEqual(report["metrics"]["recall"], 1.0)

    def test_stuff_classes_merge_into_one_minus_one_instance(self):
        records = self.records([C.SAMPLE_IDS[0]])
        run = B.run_batch(
            self.path("predictions"),
            checkpoint=self.checkpoint(),
            inputs=records,
            sample_set="explicit",
            loader=self.loader([]),
            predict=self.detections(
                [(30, 0.9, self.bar(y=10)), (30, 0.7, self.bar(y=40))]
            ),
            emit=B.default_emit,
            png_loader=self.png_loader,
        )
        path = os.path.join(run["predictions_dir"], C.slug(C.SAMPLE_IDS[0]) + ".json")
        document, _ = C.unwrap_prediction(C.load_json(path), C.SAMPLE_IDS[0])
        self.assertEqual(set(int(value) for value in document["semanticIds"]), {30})
        self.assertEqual(set(int(value) for value in document["instanceIds"]), {-1})
        self.assertEqual(run["predictions"][0]["instances"], 1, "one merged Stuff segment")

    def test_existing_outputs_are_refused_before_any_load(self):
        records = self.records([C.SAMPLE_IDS[0]])
        target = self.path("predictions")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "leftover.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        load_calls: list = []
        with self.assertRaises(C.GuardError) as caught:
            B.run_batch(
                target,
                checkpoint=self.checkpoint(),
                inputs=records,
                sample_set="explicit",
                loader=self.loader(load_calls),
                predict=self.detections([(0, 0.9, self.bar())]),
                png_loader=self.png_loader,
            )
        self.assertEqual(caught.exception.code, "predictions_exist")
        self.assertEqual(load_calls, [], "refusal must precede the checkpoint load")

    def test_inputs_json_keeps_the_model_order(self):
        sample_ids = [C.SAMPLE_IDS[2], C.SAMPLE_IDS[0], C.SAMPLE_IDS[1]]
        records = self.records(sample_ids)
        spec = self.path("parent_capacity16_inputs.json")
        C.write_json(spec, {"schema": C.SCHEMA_INPUTS, "samples": records})
        run = B.run_batch(
            self.path("predictions"),
            checkpoint=self.checkpoint(),
            inputs=spec,
            sample_set="explicit",
            loader=self.loader([]),
            predict=self.detections([(0, 0.9, self.bar())]),
            emit=B.default_emit,
            png_loader=self.png_loader,
        )
        self.assertEqual(run["sample_ids"], sample_ids)
        self.assertEqual(
            sorted(os.listdir(run["predictions_dir"])),
            sorted(C.slug(sample_id) + ".json" for sample_id in sample_ids),
        )

    def test_non_png_and_sealed_inputs_are_refused(self):
        png = self.png(C.SAMPLE_IDS[0])
        svg = self.path("png", "drawing.svg")
        with open(svg, "w", encoding="utf-8") as handle:
            handle.write("<svg/>")
        with self.assertRaises(C.GuardError) as caught:
            B.run_batch(
                self.path("predictions"),
                checkpoint=self.checkpoint(),
                inputs=[{"sample_id": C.SAMPLE_IDS[0], "png_path": svg}],
                sample_set="explicit",
                loader=self.loader([]),
                predict=self.detections([(0, 0.9, self.bar())]),
                png_loader=self.png_loader,
            )
        self.assertEqual(caught.exception.code, "batch_png_required")

        with self.assertRaises(C.GuardError) as caught:
            B.run_batch(
                self.path("predictions"),
                checkpoint=self.checkpoint(),
                inputs=[
                    {"sample_id": C.SAMPLE_IDS[0], "png_path": os.path.join(self.work, "test_final", "a.png")}
                ],
                sample_set="explicit",
                loader=self.loader([]),
                predict=self.detections([(0, 0.9, self.bar())]),
                png_loader=self.png_loader,
            )
        self.assertEqual(caught.exception.code, "sealed_path")
        self.assertTrue(os.path.isfile(png))

    def test_letterboxed_png_lands_inside_the_viewbox(self):
        """200x100 source: pixel 60..140 must become viewBox 10..90, not 60..140."""

        records = [
            {
                "sample_id": C.SAMPLE_IDS[0],
                "png_path": self.png(C.SAMPLE_IDS[0], size=(200, 100)),
            }
        ]
        run = B.run_batch(
            self.path("predictions"),
            checkpoint=self.checkpoint(),
            inputs=records,
            sample_set="explicit",
            loader=self.loader([]),
            predict=self.detections([(0, 0.9, self.bar(y=10, x0=60, x1=140, size=(200, 100)))]),
            emit=B.default_emit,
            png_loader=self.png_loader,
        )
        path = os.path.join(run["predictions_dir"], C.slug(C.SAMPLE_IDS[0]) + ".json")
        document, _ = C.unwrap_prediction(C.load_json(path), C.SAMPLE_IDS[0])
        xs = [float(value) for value in document["args"][0][0::2]]
        self.assertGreaterEqual(min(xs), 0.0)
        self.assertLessEqual(max(xs), 100.0, "letterbox content must stay on canvas")
        self.assertAlmostEqual(min(xs), 10.0, delta=1.0)
        self.assertAlmostEqual(max(xs), 90.0, delta=1.0)
        self.assertEqual(run["predictions"][0]["out_of_viewbox_primitives"], 0)
        self.assertEqual(
            C.load_json(path)["source"]["out_of_viewbox_primitives"], 0
        )

    def test_checkpoint_is_required(self):
        records = self.records([C.SAMPLE_IDS[0]])
        with self.assertRaises(C.GuardError) as caught:
            B.run_batch(
                self.path("predictions"),
                inputs=records,
                sample_set="explicit",
                loader=self.loader([]),
                predict=self.detections([(0, 0.9, self.bar())]),
                png_loader=self.png_loader,
            )
        self.assertEqual(caught.exception.code, "checkpoint_missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
