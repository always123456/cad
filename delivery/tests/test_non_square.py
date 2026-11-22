"""Non-square rasters: padded to the square canvas, never refused.

The scorer rasterises the submitted ``args`` through the 100 x 100 viewBox onto
its own 1000 px canvas; it never looks at the PNG's aspect ratio.  So the PNG's
dimensions matter only to the model, and any W x H raster is mapped through
``side = max(W, H)`` with offset 0 and scale ``100 / side``: the drawing is
treated as a square ``side x side`` canvas cropped at the right/bottom.  The CLI
zero-pads the mask right/bottom to that square before calling the frozen emitter
(the lane's own emitter would letterbox instead, pushing edge content negative).

The dataset's non-square PNGs really are ±1 pixel crops of a 100 x 100 viewBox
rendered at 10x (1000x999, 999x1000), but the rule is general: a badly non-square
raster is still delivered, with the padding and an unusual-aspect note recorded in
the receipt.  A missing document scores zero; a mis-scaled one still scores
something.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from support import args, fake_loader, touch

import cad_mrcnn_delivery as delivery

# The pre-change output for the three-instance 24x25 fixture below, captured
# before the ±1 px special case was generalised.  It pins that the ±1 px path is
# byte-identical after the change.
GOLDEN_SHA256 = "c117061df651de6522223d14131c3f71bb5a4a849ae0939532f27ff3a9307215"

# Three masks on a 24 wide x 25 tall raster: the top-left corner, the middle and
# the bottom edge.
GOLDEN_MASKS = ((0, 6, 0, 6), (8, 14, 8, 16), (22, 25, 18, 24))


class SquareCanvasMasksTest(unittest.TestCase):
    def test_square_raster_is_untouched(self):
        import numpy as np

        mask = np.ones((8, 8), dtype=bool)
        masks, side, padded = delivery.square_canvas_masks([mask], 8, 8)
        self.assertEqual(side, None)
        self.assertFalse(padded)
        self.assertIs(masks[0], mask)

    def test_crop_residual_is_padded_right_and_bottom(self):
        import numpy as np

        mask = np.zeros((25, 24), dtype=bool)
        mask[0:6, 0:6] = True
        masks, side, padded = delivery.square_canvas_masks([mask], 24, 25)
        self.assertTrue(padded)
        self.assertEqual(side, 25)
        self.assertEqual(masks[0].shape, (25, 25))
        self.assertEqual(int(masks[0].sum()), 36)
        self.assertTrue(masks[0][0, 0])
        self.assertFalse(masks[0][:, 24].any())  # the pad column is empty

    def test_genuine_aspect_mismatch_is_padded_not_refused(self):
        """A 24x32 raster is padded to 32x32, not left to the letterbox affine."""

        import numpy as np

        mask = np.zeros((32, 24), dtype=bool)
        mask[0:6, 0:6] = True
        masks, side, padded = delivery.square_canvas_masks([mask], 24, 32)
        self.assertTrue(padded)
        self.assertEqual(side, 32)
        self.assertEqual(masks[0].shape, (32, 32))
        self.assertEqual(int(masks[0].sum()), 36)
        self.assertFalse(masks[0][:, 24:].any())  # right pad column
        self.assertFalse(masks[0][:, 0].sum() != 6)

    def test_empty_mask_list_is_still_padded(self):
        masks, side, padded = delivery.square_canvas_masks([], 1000, 750)
        self.assertTrue(padded)
        self.assertEqual(side, 1000)
        self.assertEqual(masks, [])

    def test_canvas_padding_record(self):
        square = delivery.canvas_padding_record(1000, 1000)
        self.assertEqual(square["crop_residual_px"], 0)
        self.assertEqual(square["canvas_padding"], {"left": 0, "top": 0, "right": 0, "bottom": 0})
        self.assertFalse(square["unusual_aspect"])
        self.assertIsNone(square["note"])

        residual = delivery.canvas_padding_record(1000, 999)
        self.assertEqual(residual["crop_residual_px"], 1)
        self.assertEqual(residual["canvas_padding"]["bottom"], 1)
        self.assertEqual(residual["canvas_padding"]["right"], 0)
        self.assertFalse(residual["unusual_aspect"])
        self.assertIsNone(residual["note"])

        wide = delivery.canvas_padding_record(1000, 750)
        self.assertEqual(wide["side"], 1000)
        self.assertEqual(wide["crop_residual_px"], 250)
        self.assertEqual(wide["canvas_padding"], {"left": 0, "top": 0, "right": 0, "bottom": 250})
        self.assertTrue(wide["unusual_aspect"])
        self.assertIn("unusual aspect ratio", wide["note"])

        tall = delivery.canvas_padding_record(750, 1000)
        self.assertEqual(tall["canvas_padding"], {"left": 0, "top": 0, "right": 250, "bottom": 0})
        self.assertTrue(tall["unusual_aspect"])


class _DeliveryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-nonsquare-")
        self.png = touch(os.path.join(self.tmp, "drawing.png"))
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"))
        self.out = os.path.join(self.tmp, "out")

    def selection(self, out=None):
        out = out or self.out
        with open(os.path.join(out, "selection_report.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def receipt(self, out=None):
        out = out or self.out
        with open(os.path.join(out, "run_receipt.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def document(self, out=None):
        out = out or self.out
        with open(os.path.join(out, "results", "drawing.json"), encoding="utf-8") as handle:
            return json.load(handle)


class CropResidualDeliveryTest(_DeliveryCase):
    """A 24x25 raster (1 px short of square) with content on the top-left edge."""

    def run_png(self, out=None):
        return delivery.run_delivery(
            args(png=self.png, out=out or self.out, checkpoint=self.checkpoint, device="cpu"),
            loader=fake_loader,
            predict=_edge_predict,
            png_loader=_crop_png_loader,
        )

    def test_crop_residual_is_delivered_within_the_viewbox(self):
        summary = self.run_png()
        self.assertEqual(summary["status"], "delivery_written")
        self.assertEqual(summary["non_square_images"], 1)

        document = self.document()
        self.assertEqual(len(document), 12)
        self.assertNotIn("score", document)
        values = [item for argument in document["args"] for item in argument]
        self.assertGreaterEqual(min(values), 0.0)
        self.assertLessEqual(max(values), 100.0)

        entry = self.selection()["images"][0]
        self.assertFalse(entry["square"])
        self.assertTrue(entry["emit_canvas"]["padded_to_square"])
        self.assertEqual(entry["emit_canvas"]["width"], 25)
        self.assertEqual(entry["emit_canvas"]["height"], 25)
        self.assertEqual(entry["emit_canvas"]["crop_residual_px"], 1)
        self.assertEqual(
            entry["emit_canvas"]["canvas_padding"], {"left": 0, "top": 0, "right": 1, "bottom": 0}
        )
        self.assertFalse(entry["emit_canvas"]["unusual_aspect"])
        self.assertIsNone(entry["emit_canvas"]["note"])

        block = self.receipt()["non_square"]
        self.assertEqual(block["non_square_images"], 1)
        self.assertEqual(block["padded_images"], 1)
        self.assertEqual(block["unusual_aspect_images"], 0)
        self.assertEqual(block["crop_residual_px_max"], 1)
        self.assertFalse(block["lane_files_modified"])
        self.assertEqual(len(block["images"]), 1)
        self.assertEqual(block["images"][0]["sample_id"], "drawing")
        self.assertEqual(block["images"][0]["crop_residual_px"], 1)

    def test_padded_coordinates_are_the_exact_inverse_of_the_render(self):
        """Padded 25 px canvas -> scale 0.25 units/px, no centring offset.

        Pixel 0 maps to 0.0 (never the -0.05 the letterbox produced) and pixel 5,
        the far edge of the 6 px block, maps to 20.0.
        """

        self.run_png()
        document = self.document()
        values = sorted({item for argument in document["args"] for item in argument})
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 20.0)


class CropResidualGoldenTest(_DeliveryCase):
    """The ±1 px path is byte-identical to the pre-change output."""

    def test_crop_residual_bytes_match_the_golden_output(self):
        summary = delivery.run_delivery(
            args(png=self.png, out=self.out, checkpoint=self.checkpoint, device="cpu"),
            loader=fake_loader,
            predict=_three_predict,
            png_loader=_three_png_loader,
        )
        self.assertEqual(summary["status"], "delivery_written")
        with open(os.path.join(self.out, "results", "drawing.json"), "rb") as handle:
            raw = handle.read()
        self.assertEqual(len(json.loads(raw)["args"]), 15)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), GOLDEN_SHA256)


class OddAspectDeliveryTest(_DeliveryCase):
    """1000x750 and 750x1000: delivered, counted and reported, never refused."""

    # (row0, row1, col0, col1) as fractions of the raster, so the same drawing
    # lands in the same relative place on either orientation.
    SPECS = (
        (0.01, 0.06, 0.01, 0.06),  # top-left corner, touching pixel (0, 0)
        (0.40, 0.46, 0.60, 0.66),
        (0.98, 1.00, 0.98, 1.00),  # bottom-right corner
        (0.50, 0.55, 0.10, 0.15),
        (0.90, 1.00, 0.10, 0.20),  # touches the last row
    )

    def _run(self, width, height, out):
        boxes = _pixel_boxes(self.SPECS, width, height)
        masks = [_mask_from([box], width, height) for box in boxes]

        def predict(_context, _image):
            import numpy as np

            return {
                "boxes": np.tile(np.array([[0, 0, 1, 1]], dtype=np.float32), (len(masks), 1)),
                "labels": np.array([1] * len(masks), dtype=np.int64),
                "scores": np.linspace(0.99, 0.90, len(masks)).astype(np.float32),
                "mask_probabilities": np.stack([m.astype(np.float32) for m in masks]),
                "score_threshold": 0.05,
                "mask_threshold": 0.5,
            }

        def png_loader(*_args, **_kwargs):
            import numpy as np

            ink = np.zeros((height, width), dtype=bool)
            for box in boxes:
                ink[box] = True
            image = np.zeros((3, height, width), dtype=np.float32)
            image[:, ink] = 1.0
            return image, ink, {"path": "fake", "width": width, "height": height}

        return delivery.run_delivery(
            args(png=self.png, out=out, checkpoint=self.checkpoint, device="cpu"),
            loader=fake_loader,
            predict=predict,
            png_loader=png_loader,
        )

    def test_odd_aspect_is_delivered_not_refused(self):
        square_out = os.path.join(self.tmp, "square")
        square = self._run(1000, 1000, square_out)
        self.assertEqual(square["status"], "delivery_written")
        square_document = self.document(square_out)
        square_instances = len(square_document["obj_cts"])
        self.assertEqual(square_instances, len(self.SPECS))
        self.assertFalse(self.selection(square_out)["images"][0]["emit_canvas"]["padded_to_square"])

        for width, height in ((1000, 750), (750, 1000)):
            with self.subTest(width=width, height=height):
                out = os.path.join(self.tmp, "%dx%d" % (width, height))
                summary = self._run(width, height, out)
                self.assertEqual(summary["status"], "delivery_written")
                self.assertEqual(summary["non_square_images"], 1)

                document = self.document(out)
                self.assertEqual(len(document), 12)
                self.assertNotIn("score", document)
                self.assertEqual(len(document["obj_cts"]), len(self.SPECS))
                self.assertEqual(len(document["obj_cts"]), square_instances)
                self.assertEqual(len(summary["results"]), 1)

                values = [item for argument in document["args"] for item in argument]
                self.assertGreaterEqual(min(values), 0.0)
                self.assertLessEqual(max(values), 100.0)

                side = max(width, height)
                pad = {
                    "left": 0,
                    "top": 0,
                    "right": side - width,
                    "bottom": side - height,
                }
                entry = self.selection(out)["images"][0]
                canvas = entry["emit_canvas"]
                self.assertFalse(entry["square"])
                self.assertTrue(canvas["padded_to_square"])
                self.assertEqual((canvas["width"], canvas["height"]), (side, side))
                self.assertEqual(canvas["crop_residual_px"], abs(width - height))
                self.assertEqual(canvas["canvas_padding"], pad)
                self.assertTrue(canvas["unusual_aspect"])
                self.assertIn("unusual aspect ratio", canvas["note"])

                block = self.receipt(out)["non_square"]
                self.assertEqual(block["non_square_images"], 1)
                self.assertEqual(block["padded_images"], 1)
                self.assertEqual(block["unusual_aspect_images"], 1)
                self.assertEqual(block["crop_residual_px_max"], abs(width - height))
                self.assertFalse(block["lane_files_modified"])
                self.assertEqual(len(block["images"]), 1)
                image = block["images"][0]
                self.assertEqual((image["native_width"], image["native_height"]), (width, height))
                self.assertEqual((image["emit_width"], image["emit_height"]), (side, side))
                self.assertEqual(image["canvas_padding"], pad)
                self.assertTrue(image["unusual_aspect"])
                self.assertIn("unusual aspect ratio", image["note"])

    def test_odd_aspect_coordinates_use_the_whole_long_side(self):
        """1000x750 -> side 1000, scale 0.1 units/px, content stays in the top 3/4."""

        out = os.path.join(self.tmp, "wide")
        self._run(1000, 750, out)
        document = self.document(out)
        ys = sorted({argument[i] for argument in document["args"] for i in (1, 3, 5, 7)})
        self.assertGreaterEqual(min(ys), 0.0)
        self.assertLess(min(ys), 1.0)  # the drawing starts at its first row (7 px -> 0.7)
        self.assertLessEqual(max(ys), 75.0)  # 750 px * 0.1; the bottom pad carries no ink


def _pixel_boxes(specs, width, height):
    boxes = []
    for row0, row1, col0, col1 in specs:
        y0, y1 = int(row0 * height), max(int(row0 * height) + 1, int(row1 * height))
        x0, x1 = int(col0 * width), max(int(col0 * width) + 1, int(col1 * width))
        y0, y1 = max(0, min(y0, height - 1)), min(y1, height)
        x0, x1 = max(0, min(x0, width - 1)), min(x1, width)
        boxes.append((slice(y0, y1), slice(x0, x1)))
    return boxes


def _mask_from(boxes, width, height):
    import numpy as np

    mask = np.zeros((height, width), dtype=bool)
    for rows, cols in boxes:
        mask[rows, cols] = True
    return mask


def _crop_png_loader(*_args, **_kwargs):
    import numpy as np

    image = np.zeros((3, 25, 24), dtype=np.float32)
    image[:, 0:6, 0:6] = 1.0
    return image, np.zeros((25, 24), dtype=bool), {"path": "fake", "width": 24, "height": 25}


def _edge_predict(_context, _image):
    import numpy as np

    mask = np.zeros((25, 24), dtype=np.float32)
    mask[0:6, 0:6] = 1.0
    return {
        "boxes": np.zeros((1, 4), dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "scores": np.array([0.9], dtype=np.float32),
        "mask_probabilities": mask[None, :, :],
        "score_threshold": 0.05,
        "mask_threshold": 0.5,
    }


def _three_png_loader(*_args, **_kwargs):
    import numpy as np

    boxes = [(slice(r0, r1), slice(c0, c1)) for r0, r1, c0, c1 in GOLDEN_MASKS]
    mask = _mask_from(boxes, 24, 25)
    image = np.zeros((3, 25, 24), dtype=np.float32)
    image[:, mask] = 1.0
    return image, mask, {"path": "fake", "width": 24, "height": 25}


def _three_predict(_context, _image):
    import numpy as np

    masks = []
    for r0, r1, c0, c1 in GOLDEN_MASKS:
        mask = np.zeros((25, 24), dtype=np.float32)
        mask[r0:r1, c0:c1] = 1.0
        masks.append(mask)
    return {
        "boxes": np.tile(np.array([[0, 0, 10, 10]], dtype=np.float32), (len(masks), 1)),
        "labels": np.array([1] * len(masks), dtype=np.int64),
        "scores": np.array([0.99, 0.95, 0.90], dtype=np.float32),
        "mask_probabilities": np.stack(masks),
        "score_threshold": 0.05,
        "mask_threshold": 0.5,
    }


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
