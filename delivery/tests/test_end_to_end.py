"""End-to-end delivery runs with fake model access: real selection, real emitter, real contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

from tests.support import DEFAULT_SPECS, args, delivery, detections, fake_loader, fake_run, touch

DOCUMENT_FIELDS = tuple(sorted(delivery.DOCUMENT_FIELDS))


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class DeliveryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-e2e-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"), b"weights")
        self.png = touch(os.path.join(self.tmp, "drawing.png"), b"\x89PNG\r\n\x1a\n")
        self.out = os.path.join(self.tmp, "out")

    # helpers --------------------------------------------------------------- #
    def run_png(self, out=None, **overrides):
        return fake_run(out or self.out, self.checkpoint, png=self.png, **overrides)

    def result(self, stem="drawing", out=None):
        return load(os.path.join(out or self.out, "results", stem + ".json"))

    def receipt(self, out=None):
        return load(os.path.join(out or self.out, "run_receipt.json"))

    def selection(self, out=None):
        return load(os.path.join(out or self.out, "selection_report.json"))


class WrittenDeliveryTest(DeliveryCase):
    def test_single_png_writes_the_contract_document(self):
        summary = self.run_png()
        self.assertEqual(summary["status"], "delivery_written")
        self.assertEqual(summary["n_results"], 1)
        path = os.path.join(self.out, "results", "drawing.json")
        self.assertTrue(os.path.isfile(path))
        document = load(path)
        self.assertEqual(tuple(sorted(document)), DOCUMENT_FIELDS)
        self.assertEqual(document["width"], 100)
        self.assertEqual(document["height"], 100)
        with open(path, "r", encoding="utf-8") as handle:
            self.assertNotIn('"score"', handle.read())
        self.assertEqual(delivery.document_report(document), [])
        violations = delivery._contract().validate_prediction_document(document, "drawing")
        self.assertIs(violations, document)

    def test_stuff_merges_to_instance_minus_one_and_things_get_ids(self):
        self.run_png()
        document = self.result()
        pairs = list(zip(document["semanticIds"], document["instanceIds"]))
        stuff = [pair for pair in pairs if pair[0] in (30, 31, 32, 33, 34)]
        things = [pair for pair in pairs if pair[0] < 30]
        self.assertTrue(stuff)
        self.assertTrue(all(instance == -1 for _semantic, instance in stuff))
        self.assertEqual({semantic for semantic, _instance in stuff}, {32})
        self.assertTrue(things)
        self.assertTrue(all(instance >= 0 for _semantic, instance in things))
        self.assertEqual(sorted({instance for _semantic, instance in things}), [0, 1])
        self.assertEqual(len(document["args"]), len(document["commands"]))
        for field in ("args", "lengths", "widths", "semanticIds", "instanceIds", "rgb", "layerIds"):
            self.assertEqual(len(document[field]), len(document["commands"]))
        # obj_cts and boxes carry one row per Thing instance, not per primitive
        self.assertEqual(len(document["obj_cts"]), len({instance for _s, instance in things}))
        self.assertEqual(len(document["boxes"]), len({instance for _s, instance in things}))

    def test_default_selection_counts(self):
        self.run_png()
        report = self.selection()
        entry = report["images"][0]
        # 4 candidates; the duplicate same-class pair collapses under NMS; nothing
        # is below 0.05; nothing hits either cap.
        self.assertEqual(entry["candidates"], 4)
        self.assertEqual(entry["after_nms"], 3)
        self.assertEqual(entry["after_threshold"], 3)
        self.assertEqual(entry["after_cap"], 3)
        self.assertEqual(entry["selected"], 3)
        self.assertEqual(entry["emitted"], 3)  # 2 things + 1 merged Stuff instance
        self.assertAlmostEqual(entry["lowest_surviving_score"], 0.10, places=6)
        self.assertEqual(report["totals"]["images"], 1)
        self.assertEqual(report["policy"]["signature"], [0.05, 0.5, 0.5, 300, None])

    def test_score_threshold_applies_after_decode(self):
        self.run_png(score_threshold=0.5)
        entry = self.selection()["images"][0]
        self.assertEqual(entry["candidates"], 4)
        self.assertEqual(entry["after_nms"], 3)
        self.assertEqual(entry["after_threshold"], 2)
        self.assertEqual(entry["emitted"], 2)
        self.assertAlmostEqual(entry["lowest_surviving_score"], 0.60, places=6)
        self.assertNotIn(1, self.result()["semanticIds"])  # the 0.10 instance is gone

    def test_per_image_cap_keeps_the_top_n_by_score(self):
        self.run_png(per_image_cap=1)
        entry = self.selection()["images"][0]
        self.assertEqual(entry["after_threshold"], 3)
        self.assertEqual(entry["after_cap"], 3)
        self.assertEqual(entry["selected"], 1)
        self.assertAlmostEqual(entry["lowest_surviving_score"], 0.90, places=6)
        self.assertEqual(set(self.result()["semanticIds"]), {0})

    def test_global_cap_truncates_in_decode_order(self):
        self.run_png(max_detections=2)
        entry = self.selection()["images"][0]
        self.assertEqual(entry["after_cap"], 2)
        self.assertEqual(entry["selected"], 2)
        document = self.result()
        self.assertIn(0, document["semanticIds"])
        self.assertIn(32, document["semanticIds"])

    def test_two_images_from_inputs_json(self):
        second = touch(os.path.join(self.tmp, "pngs", "second.png"), b"\x89PNG\r\n\x1a\n")
        inputs = os.path.join(self.tmp, "inputs.json")
        with open(inputs, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": "cadvec.direct_instance.inputs.v1",
                    "samples": [
                        {"sample_id": "a", "group_id": "g-1", "png_path": self.png},
                        {"sample_id": "b", "group_id": "g-2", "png_path": second},
                    ],
                },
                handle,
            )
        summary = fake_run(self.out, self.checkpoint, inputs=inputs)
        self.assertEqual(summary["n_results"], 2)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "results", "drawing.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "results", "second.json")))
        index = load(os.path.join(self.out, "results_index.json"))
        self.assertEqual(set(index["images"]), {"drawing", "second"})
        self.assertEqual(index["images"]["second"]["sample_id"], "b")
        self.assertEqual(self.selection()["totals"]["images"], 2)
        receipt = self.receipt()
        self.assertEqual(receipt["inputs"]["count"], 2)
        self.assertEqual(len(receipt["timings"]["per_image"]), 2)

    def test_square_flags_are_reported(self):
        summary = self.run_png()
        entry = self.selection()["images"][0]
        self.assertTrue(entry["square"])
        self.assertEqual(entry["width"], 32)
        self.assertEqual(entry["height"], 32)
        self.assertEqual(self.selection()["totals"]["non_square_images"], 0)
        self.assertTrue(self.receipt()["guards"]["all_images_square"])
        self.assertTrue(self.receipt()["timings"]["per_image"][0]["square"])
        self.assertEqual(summary["non_square_images"], 0)

    def test_genuine_aspect_mismatch_is_padded_and_delivered(self):
        """A real aspect mismatch is delivered, not refused.

        submission.viewbox_transform(width, height) letterboxes: scale is
        min(w/100, h/100) and the short side is centred with an offset, so a mask
        on the short side's edge maps to a negative coordinate and the document
        guard would refuse the image.  The CLI therefore pads every non-square
        raster to max(width, height) square (right/bottom) before the frozen
        emitter, so a 32x24 raster is emitted as 32x32 with no offset.  A missing
        document scores zero; a slightly mis-scaled one still scores something.
        """

        import numpy as np

        def png_loader(*_args, **_kwargs):
            image = np.zeros((3, 24, 32), dtype=np.float32)
            image[:, 4:12, 4:12] = 1.0
            return image, np.zeros((24, 32), dtype=bool), {"path": "fake", "width": 32, "height": 24}

        def predict(_context, _image):
            mask = np.zeros((24, 32), dtype=np.float32)
            mask[2:8, 2:8] = 1.0
            return {
                "boxes": np.zeros((1, 4), dtype=np.float32),
                "labels": np.array([1], dtype=np.int64),
                "scores": np.array([0.9], dtype=np.float32),
                "mask_probabilities": mask[None, :, :],
                "score_threshold": 0.05,
                "mask_threshold": 0.5,
            }

        namespace = args(png=self.png, out=self.out, checkpoint=self.checkpoint, device="cpu")
        summary = delivery.run_delivery(
            namespace,
            loader=fake_loader,
            predict=predict,
            png_loader=png_loader,
        )
        self.assertEqual(summary["status"], "delivery_written")
        self.assertEqual(summary["non_square_images"], 1)

        document = load(os.path.join(self.out, "results", "drawing.json"))
        self.assertEqual(len(document), 12)
        values = [item for argument in document["args"] for item in argument]
        self.assertGreaterEqual(min(values), 0.0)
        self.assertLessEqual(max(values), 100.0)

        canvas = self.selection()["images"][0]["emit_canvas"]
        self.assertTrue(canvas["padded_to_square"])
        self.assertEqual((canvas["width"], canvas["height"]), (32, 32))
        self.assertEqual(canvas["crop_residual_px"], 8)
        self.assertTrue(canvas["unusual_aspect"])

    def test_results_are_deterministic_across_runs(self):
        first = os.path.join(self.tmp, "out-a")
        second = os.path.join(self.tmp, "out-b")
        self.run_png(out=first)
        self.run_png(out=second)
        with open(os.path.join(first, "results", "drawing.json"), "rb") as handle:
            a = handle.read()
        with open(os.path.join(second, "results", "drawing.json"), "rb") as handle:
            b = handle.read()
        self.assertEqual(a, b)
        report_a = self.selection(out=first)
        report_b = self.selection(out=second)
        self.assertEqual(report_a["totals"], report_b["totals"])
        self.assertEqual(report_a["images"], report_b["images"])


class ReceiptTest(DeliveryCase):
    def test_receipt_records_decode_checkpoint_and_environment(self):
        self.run_png()
        receipt = self.receipt()
        self.assertEqual(receipt["schema"], "cad-png-delivery-mrcnn01.receipt/1")
        self.assertEqual(receipt["status"], "delivery_written")
        self.assertEqual(receipt["checkpoint"]["sha256"], delivery.sha256_file(self.checkpoint))
        self.assertEqual(receipt["checkpoint"]["bytes"], len(b"weights"))
        self.assertEqual(receipt["device"], "cpu")
        decode = receipt["decode"]
        self.assertEqual(decode["roi_score_thresh"], 0.05)
        self.assertEqual(decode["roi_nms_thresh"], 0.5)
        self.assertEqual(decode["roi_detections_per_img"], 300)
        self.assertEqual(decode["device"], "cpu")
        self.assertEqual(receipt["selection_signature"], [0.05, 0.5, 0.5, 300, None])
        self.assertEqual(receipt["policy"]["mask_threshold"], 0.5)
        self.assertEqual(receipt["inputs"]["count"], 1)
        self.assertTrue(receipt["inputs"]["image_only_path"].endswith("image_only_inputs.json"))
        self.assertTrue(os.path.isfile(receipt["inputs"]["image_only_path"]))
        self.assertFalse(receipt["guards"]["ground_truth_read"])
        self.assertFalse(receipt["guards"]["svg_read"])
        self.assertFalse(receipt["guards"]["score_field_in_document"])
        self.assertEqual(receipt["lane"]["batch_max_detections"], 300)
        self.assertFalse(receipt["lane"]["gt_read_reported_by_driver"])
        sources = receipt["lane"]["sources"]
        for name in (
            "run_batch_predict.py",
            "instance_eval/batch.py",
            "instance_eval/contract.py",
            "model/infer.py",
            "model/emit.py",
            "src/cadvec/recognition_submission.py",
            "src/cadvec/png_raster_vectorize.py",
        ):
            self.assertTrue(sources[name]["exists"], name)
            self.assertEqual(len(sources[name]["sha256"]), 64, name)
        self.assertEqual(len(sources["cad_mrcnn_delivery.py"]["sha256"]), 64)
        environment = receipt["environment"]
        # The receipt must record the interpreter that produced it, not a pinned version:
        # the same code runs under the delivery runtime and under the scoring runtime.
        self.assertEqual(environment["python"], "%d.%d.%d" % sys.version_info[:3])
        for key in ("torch", "torchvision", "numpy", "pillow", "torch_cuda", "cuda_devices", "cudnn"):
            self.assertIn(key, environment)

    def test_decode_is_untouched_by_the_selection_flags(self):
        self.run_png(score_threshold=0.97, max_detections=5)
        receipt = self.receipt()
        self.assertEqual(receipt["decode"]["roi_score_thresh"], 0.05)
        self.assertEqual(receipt["decode"]["roi_detections_per_img"], 300)
        self.assertEqual(receipt["policy"]["score_threshold"], 0.97)
        self.assertEqual(receipt["policy"]["max_detections"], 5)
        self.assertEqual(receipt["guards"]["decode_settings_from_flags"], False)

    def test_receipt_records_per_image_timings(self):
        self.run_png()
        totals = self.receipt()["timings"]["totals"]
        self.assertEqual(totals["images"], 1)
        for key in ("wall_s", "read_s", "predict_s", "select_s", "emit_s", "per_image_mean_s", "per_image_median_s", "per_image_max_s", "images_per_s"):
            self.assertIsNotNone(totals[key], key)
        entry = self.receipt()["timings"]["per_image"][0]
        self.assertEqual(entry["stem"], "drawing")
        self.assertGreaterEqual(entry["total_s"], 0.0)

    def test_lane_sources_match_the_recorded_hashes(self):
        self.run_png()
        sources = self.receipt()["lane"]["sources"]
        expected = {
            "run_batch_predict.py": "6652c03332317f9fa845d6486a69e0a53b3d9857f8b66d024571431c3a4ade60",
            "instance_eval/batch.py": "184919042227e22991ae1742127e3cef8005b025f97521c7e32b1439220b67e9",
            "instance_eval/contract.py": "5b91a80914676bc43eefcbaef9b1fa4060da577b33441b2d8404c6e7657223c4",
            "model/infer.py": "23c9fc24980f9f0d94a72abd28337a282af5523b8017adce12b0a641ef74017f",
            "model/emit.py": "14f4458658da414cde296560fa8a912ba120b84e9c53ba91bcfdcbd3ddd340f6",
        }
        for name, digest in expected.items():
            self.assertEqual(sources[name]["sha256"], digest, name)

    def test_selection_report_can_be_placed_elsewhere_too(self):
        extra = os.path.join(self.tmp, "reports", "selection.json")
        self.run_png(selection_report=extra)
        self.assertTrue(os.path.isfile(extra))
        self.assertEqual(load(extra), self.selection())
        self.assertIn(os.path.abspath(extra), [os.path.abspath(p) for p in self.receipt()["outputs"]["selection_reports"]])


class RefusalTest(DeliveryCase):
    def test_dry_run_writes_nothing(self):
        summary = self.run_png(dry_run=True)
        self.assertEqual(summary["status"], "dry_run")
        self.assertFalse(os.path.exists(self.out))
        payload = summary["image_only_inputs"]
        self.assertEqual(payload["count"], 1)
        self.assertEqual(set(payload["samples"][0]), {"sample_id", "group_id", "png_path"})

    def test_second_run_into_the_same_out_is_refused(self):
        self.run_png()
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            self.run_png()
        self.assertEqual(ctx.exception.code, "out_not_empty")

    def test_no_detections_is_reported_not_crashed(self):
        def empty(_context, _image):
            return detections()

        try:
            summary = self.run_png(predict=empty)
        except delivery.DeliveryRefused as exc:
            self.assertEqual(exc.code, "document_violation")
        else:
            self.assertEqual(summary["status"], "delivery_written")
            document = self.result()
            self.assertEqual(document["commands"], [])


if __name__ == "__main__":
    unittest.main()
