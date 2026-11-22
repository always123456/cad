"""The image-only inputs file, and proof that no ground truth or SVG is read."""

from __future__ import annotations

import builtins
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.support import delivery, fake_run, touch

def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


TOP_LEVEL_KEYS = {
    "schema",
    "created_utc",
    "dataset",
    "role",
    "image_only",
    "field_note",
    "derived_from",
    "ignored_row_keys",
    "count",
    "samples",
}


def lane_style_inputs(tmp, count=2):
    """A lane-shaped inputs file: extra row keys, svg paths, planted ground truth."""

    rows = []
    markers = []
    for index in range(count):
        png = touch(os.path.join(tmp, "pngs", "drawing%d.png" % index))
        svg = touch(os.path.join(tmp, "svg", "drawing%d.svg" % index), "<svg/>")
        markers.append(svg)
        rows.append(
            {
                "sample_id": "train_final:drawing%d" % index,
                "group_id": "g-%04d" % index,
                "record_index": index,
                "position": index,
                "png_path": png,
                "svg_path": svg,
                "ground_truth": {"instances": [{"semanticId": 1}]},
            }
        )
    path = os.path.join(tmp, "inputs.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "cadvec.direct_instance.inputs.v1",
                "dataset": "fixture",
                "samples": rows,
            },
            handle,
        )
    return path, markers


class ImageOnlyPayloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-inputs-")

    def test_payload_is_structurally_image_only(self):
        path, _markers = lane_style_inputs(self.tmp)
        payload = delivery.image_only_inputs(inputs=path)
        self.assertEqual(set(payload), TOP_LEVEL_KEYS)
        self.assertTrue(payload["image_only"])
        self.assertEqual(payload["schema"], "cadvec.direct_instance.inputs.v1")
        self.assertEqual(payload["count"], 2)
        for row in payload["samples"]:
            self.assertEqual(set(row), {"sample_id", "group_id", "png_path"})
        self.assertEqual(
            payload["ignored_row_keys"], ["ground_truth", "position", "record_index", "svg_path"]
        )
        self.assertEqual(payload["derived_from"]["inputs_sha256"], delivery.sha256_file(path))

    def test_samples_are_absolute_png_paths(self):
        path, _markers = lane_style_inputs(self.tmp, count=1)
        payload = delivery.image_only_inputs(inputs=path)
        row = payload["samples"][0]
        self.assertTrue(os.path.isabs(row["png_path"]))
        self.assertTrue(row["png_path"].lower().endswith(".png"))

    def test_bare_list_inputs_accepted(self):
        png = touch(os.path.join(self.tmp, "a.png"))
        path = os.path.join(self.tmp, "list.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"png_path": png}], handle)
        payload = delivery.image_only_inputs(inputs=path)
        self.assertEqual(payload["samples"][0]["sample_id"], "a")
        self.assertIsNone(payload["samples"][0]["group_id"])

    def test_single_png_uses_its_stem(self):
        png = touch(os.path.join(self.tmp, "board.png"))
        payload = delivery.image_only_inputs(png=png)
        self.assertEqual(payload["samples"][0]["sample_id"], "board")
        self.assertEqual(payload["count"], 1)

    def test_sample_id_and_group_id_overrides(self):
        png = touch(os.path.join(self.tmp, "board.png"))
        payload = delivery.image_only_inputs(png=png, sample_id="case-7", group_id="g-7")
        self.assertEqual(payload["samples"][0]["sample_id"], "case-7")
        self.assertEqual(payload["samples"][0]["group_id"], "g-7")

    def test_schema_mismatch_refused(self):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "something.else", "samples": []}, handle)
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "inputs_schema")

    def test_empty_samples_refused(self):
        path = os.path.join(self.tmp, "empty.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"samples": []}, handle)
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "inputs_shape")

    def test_duplicate_sample_ids_refused(self):
        png = touch(os.path.join(self.tmp, "a.png"))
        other = touch(os.path.join(self.tmp, "b.png"))
        path = os.path.join(self.tmp, "dup.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {"sample_id": "same", "png_path": png},
                    {"sample_id": "same", "png_path": other},
                ],
                handle,
            )
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "inputs_duplicate")

    def test_duplicate_results_stem_refused(self):
        first = touch(os.path.join(self.tmp, "one", "a.png"))
        second = touch(os.path.join(self.tmp, "two", "a.png"))
        path = os.path.join(self.tmp, "stem.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {"sample_id": "x", "png_path": first},
                    {"sample_id": "y", "png_path": second},
                ],
                handle,
            )
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "results_stem_collision")

    def test_non_png_row_refused(self):
        svg = touch(os.path.join(self.tmp, "a.svg"), "<svg/>")
        path = os.path.join(self.tmp, "svg.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"png_path": svg}], handle)
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "png_required")

    def test_missing_png_row_refused(self):
        path = os.path.join(self.tmp, "missing.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"png_path": os.path.join(self.tmp, "nope.png")}], handle)
        with self.assertRaises(delivery.DeliveryRefused) as ctx:
            delivery.image_only_inputs(inputs=path)
        self.assertEqual(ctx.exception.code, "png_missing")


class NoGroundTruthReadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-noread-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"), b"weights")
        self.out = os.path.join(self.tmp, "out")

    def test_run_never_opens_svg_or_ground_truth(self):
        path, markers = lane_style_inputs(self.tmp)
        for marker in markers:
            self.assertTrue(os.path.isfile(marker))

        opened = []
        real_open = builtins.open

        def recorder(file, mode="r", *rest, **kwargs):
            opened.append((str(file), str(mode)))
            return real_open(file, mode, *rest, **kwargs)

        with mock.patch("builtins.open", side_effect=recorder):
            summary = fake_run(self.out, self.checkpoint, inputs=path)

        self.assertEqual(summary["status"], "delivery_written")
        reads = [path_ for path_, mode in opened if "w" not in mode and "a" not in mode]
        self.assertFalse([item for item in reads if item.lower().endswith(".svg")])
        self.assertFalse([item for item in reads if "ground_truth" in item.lower()])
        self.assertTrue([item for item in reads if item.lower().endswith(".json")])

        receipt = load_json(os.path.join(self.out, "run_receipt.json"))
        self.assertFalse(receipt["guards"]["ground_truth_read"])
        self.assertFalse(receipt["guards"]["svg_read"])
        self.assertFalse(receipt["lane"]["gt_read_reported_by_driver"])
        self.assertFalse(receipt["lane"]["svg_read_reported_by_driver"])
        self.assertTrue(receipt["inputs"]["image_only_path"].endswith("image_only_inputs.json"))
        self.assertIn("svg_path", receipt["inputs"]["ignored_row_keys"])
        self.assertIn("ground_truth", receipt["inputs"]["ignored_row_keys"])

    def test_written_inputs_file_is_image_only(self):
        path, _markers = lane_style_inputs(self.tmp)
        fake_run(self.out, self.checkpoint, inputs=path)
        written = os.path.join(self.out, "inputs", "image_only_inputs.json")
        payload = load_json(written)
        self.assertEqual(set(payload), TOP_LEVEL_KEYS)
        for row in payload["samples"]:
            self.assertEqual(set(row), {"sample_id", "group_id", "png_path"})
        self.assertTrue(payload["image_only"])
        self.assertEqual(payload["count"], 2)

    def test_open_audit_records_roles_and_no_violations(self):
        path, _markers = lane_style_inputs(self.tmp, count=1)
        fake_run(self.out, self.checkpoint, inputs=path)
        receipt = load_json(os.path.join(self.out, "run_receipt.json"))
        audit = receipt["open_audit"]
        self.assertTrue(audit["installed"])
        self.assertEqual(audit["violations"], [])
        self.assertEqual(audit["roles"].get("checkpoint"), 1)
        self.assertEqual(audit["roles"].get("inputs"), 1)
        self.assertEqual(audit["roles"].get("png"), 1)

    def test_svg_read_would_be_caught(self):
        """The audit is not decorative: a planted SVG read is reported as a violation."""

        png = touch(os.path.join(self.tmp, "a.png"))
        svg = touch(os.path.join(self.tmp, "planted.svg"), "<svg/>")
        audit = delivery.OpenAudit(roles={png: "png"}).install()
        with open(svg, "r", encoding="utf-8") as handle:
            handle.read()
        report = audit.report()
        self.assertTrue(report["violations"])
        self.assertEqual(report["violations"][0]["role"], "violation:svg")
        self.assertEqual(report["violations"][0]["path"], svg)


if __name__ == "__main__":
    unittest.main()
