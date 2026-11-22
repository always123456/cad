"""Unit and wiring tests for the ground-truth-free ink refinement.

The rule under test is the measured one (arm A6): an ink component of the original
PNG is kept whole when at least half of it lies inside the predicted mask, and a
mask is never emptied by the refinement -- the frozen emitter skips empty masks and
the emitted instance count must not move.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import cad_mrcnn_delivery as delivery  # noqa: E402
import refine  # noqa: E402
import support  # noqa: E402


class RuleTests(unittest.TestCase):
    """The component rule itself, on hand-built rasters."""

    def test_component_fully_inside_the_mask_is_kept_whole(self):
        ink = np.zeros((8, 8), dtype=bool)
        ink[3:6, 3:6] = True  # one 9 px component
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:8, 2:8] = True
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="a6", stats=stats)
        self.assertEqual(int(refined.sum()), 9)
        np.testing.assert_array_equal(refined, ink)
        self.assertEqual(stats["kept_components"], 1)
        self.assertEqual(stats["components"], 1)
        self.assertEqual(stats["changed"], 1)
        self.assertEqual(stats["a6_to_a1"], 0)
        self.assertEqual(stats["dropped_instances"], 0)

    def test_component_at_exactly_half_coverage_is_kept(self):
        ink = np.zeros((8, 8), dtype=bool)
        ink[0:4, 0:4] = True  # 16 px component
        mask = np.zeros((8, 8), dtype=bool)
        mask[0:2, 0:4] = True  # 8 px = 50.0 %
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="a6", stats=stats)
        self.assertEqual(stats["kept_components"], 1)
        self.assertEqual(int(refined.sum()), 16)

    def test_component_just_under_half_coverage_is_dropped(self):
        # The labeller only sees the mask bbox widened by 2 px, so a component that
        # reaches past that window is judged on its visible part.  The mask bbox here
        # spans the whole component (a stray background pixel at (9, 12) widens it),
        # which makes the 49 of 100 px an honest 49 %.
        ink = np.zeros((16, 16), dtype=bool)
        ink[0:10, 0:10] = True  # 100 px component
        mask = np.zeros((16, 16), dtype=bool)
        mask[0:7, 0:7] = True  # 49 px = 49.0 %
        mask[9, 12] = True     # background, not ink: widens the bbox, adds nothing inside
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="a6", stats=stats)
        self.assertEqual(stats["kept_components"], 0)
        self.assertEqual(stats["instances_without_component"], 1)
        # nothing kept -> the mask & ink fallback, 49 px, not the 100 px component
        self.assertEqual(int(refined.sum()), 49)
        self.assertEqual(stats["a6_to_a1"], 1)
        self.assertEqual(stats["a1_to_original"], 0)

    def test_dropped_components_fall_back_to_mask_and_ink(self):
        ink = np.zeros((12, 12), dtype=bool)
        ink[0:6, 0:6] = True
        ink[6:12, 6:12] = True
        mask = np.zeros((12, 12), dtype=bool)
        mask[0:2, 0:6] = True   # 12 of 36 px of component 1
        mask[6:7, 6:12] = True  # 6 of 36 px of component 2
        mask[0:2, 8:10] = True  # background, not ink
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="a6", stats=stats)
        self.assertEqual(stats["a6_to_a1"], 1)
        self.assertEqual(stats["a1_to_original"], 0)
        np.testing.assert_array_equal(refined, mask & ink)
        self.assertEqual(int(refined.sum()), 18)

    def test_exhausted_fallbacks_return_the_original_mask(self):
        ink = np.zeros((10, 10), dtype=bool)
        ink[0:3, 0:3] = True
        mask = np.zeros((10, 10), dtype=bool)
        mask[6:9, 6:9] = True  # no ink under the mask at all
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="a6", stats=stats)
        np.testing.assert_array_equal(refined, mask)
        self.assertEqual(stats["a6_to_a1"], 1)
        self.assertEqual(stats["a1_to_original"], 1)
        self.assertEqual(stats["dropped_instances"], 0)
        self.assertEqual(stats["changed"], 0)

    def test_a1_rule_is_mask_and_ink_with_its_own_fallback(self):
        ink = np.zeros((8, 8), dtype=bool)
        ink[2:6, 2:6] = True
        mask = np.zeros((8, 8), dtype=bool)
        mask[4:8, 4:8] = True
        refined = refine.refine_mask(mask, ink, rule="a1")
        self.assertEqual(int(refined.sum()), 4)
        np.testing.assert_array_equal(refined, mask & ink)

        blank_mask = np.zeros((8, 8), dtype=bool)
        blank_mask[6:8, 0:2] = True
        stats = refine.new_stats()
        fallback = refine.refine_mask(blank_mask, ink, rule="a1", stats=stats)
        np.testing.assert_array_equal(fallback, blank_mask)
        self.assertEqual(stats["a1_rule_empty"], 1)
        self.assertEqual(stats["dropped_instances"], 0)

    def test_none_rule_leaves_the_mask_untouched(self):
        ink = np.zeros((8, 8), dtype=bool)
        ink[0:4, 0:4] = True
        mask = np.zeros((8, 8), dtype=bool)
        mask[4:8, 4:8] = True
        stats = refine.new_stats()
        refined = refine.refine_mask(mask, ink, rule="none", stats=stats)
        np.testing.assert_array_equal(refined, mask)
        self.assertEqual(stats, refine.new_stats())

    def test_refine_masks_never_changes_the_instance_count(self):
        ink = np.zeros((16, 16), dtype=bool)
        ink[0:4, 0:4] = True
        ink[10:14, 10:14] = True
        masks = []
        for box in ((0, 4, 0, 4), (0, 3, 0, 3), (12, 16, 12, 16), (7, 9, 7, 9), (5, 6, 5, 6)):
            mask = np.zeros((16, 16), dtype=bool)
            mask[box[0]:box[1], box[2]:box[3]] = True
            masks.append(mask)
        stats = refine.new_stats()
        refined = refine.refine_masks(masks, ink, rule="a6", stats=stats)
        self.assertEqual(len(refined), len(masks))
        self.assertEqual(sum(1 for m in refined if m.any()), sum(1 for m in masks if m.any()))
        self.assertEqual(stats["instances"], len(masks))
        self.assertEqual(stats["dropped_instances"], 0)

    def test_eight_connected_components_join_on_the_diagonal(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True  # diagonal neighbour: one 8-connected component
        mask[5, 0] = True  # separate
        labels, count = refine.label8(mask)
        self.assertEqual(count, 2)
        self.assertEqual(labels[1, 1], labels[2, 2])
        self.assertNotEqual(labels[1, 1], labels[5, 0])
        pure_labels, pure_count = refine.label8(mask, force_numpy=True)
        self.assertEqual(pure_count, count)
        np.testing.assert_array_equal(pure_labels, labels)

    def test_pure_numpy_labeller_matches_scipy_on_a_busy_mask(self):
        mask = np.zeros((23, 17), dtype=bool)
        mask[2:9, 3:8] = True
        mask[9:14, 7:12] = True  # touches the first block diagonally/edge
        mask[18:22, 0:4] = True
        mask[0, 16] = True
        fast_labels, fast_count = refine.label8(mask)
        slow_labels, slow_count = refine.label8(mask, force_numpy=True)
        self.assertEqual(fast_count, slow_count)
        np.testing.assert_array_equal(fast_labels, slow_labels)


class InkTests(unittest.TestCase):
    """The ink definition: any nonzero colour channel, alpha ignored."""

    def test_nonzero_ink_survives_a_black_background(self):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[1, 1] = (0, 178, 178, 255)  # teal stroke
        rgba[2, 2] = (0, 0, 0, 255)      # black on a black background: no ink
        ink = refine.ink_from_rgba(rgba, "nonzero")
        self.assertEqual(int(ink.sum()), 1)
        self.assertTrue(ink[1, 1])
        self.assertFalse(ink[2, 2])

    def test_composite128_is_thinner_than_nonzero(self):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[0, 0] = (0, 178, 178, 255)  # dark stroke: ink under both rules
        rgba[0, 1] = (250, 250, 250, 255)  # near white: nonzero says ink, composite128 does not
        rgba[1, 0] = (0, 178, 178, 0)    # fully transparent colour: white after compositing
        nonzero = refine.ink_from_rgba(rgba, "nonzero")
        composite = refine.ink_from_rgba(rgba, "composite128")
        self.assertTrue(nonzero[0, 0] and composite[0, 0])
        self.assertTrue(nonzero[0, 1])
        self.assertFalse(composite[0, 1])
        self.assertTrue(nonzero[1, 0])
        self.assertFalse(composite[1, 0])
        self.assertLess(int(composite.sum()), int(nonzero.sum()))

    def test_ink_from_png_keeps_the_native_raster(self):
        from PIL import Image

        rgba = np.zeros((13, 20, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[5:8, 2:9] = (0, 178, 0, 255)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stroke.png")
            Image.fromarray(rgba, mode="RGBA").save(path)
            ink, (width, height) = refine.ink_from_png(path, "nonzero")
        self.assertEqual((width, height), (20, 13))
        self.assertEqual(ink.shape, (13, 20))
        self.assertEqual(int(ink.sum()), 21)
        self.assertEqual(int(ink[5:8, 2:9].sum()), 21)


class RasterTests(unittest.TestCase):
    """A 999-row raster is refined in place; coordinates stay top-left, truncated."""

    def test_refinement_truncates_on_a_999_row_raster(self):
        ink = np.zeros((999, 1000), dtype=bool)
        ink[995:999, 100:200] = True
        mask = np.zeros((999, 1000), dtype=bool)
        mask[990:999, 90:210] = True
        refined = refine.refine_mask(mask, ink, rule="a6")
        self.assertEqual(refined.shape, (999, 1000))
        self.assertEqual(np.flatnonzero(refined.any(axis=1)).tolist(), [995, 996, 997, 998])

        # The lane's transform maps a pixel's top-left corner, and Pillow truncates:
        # row 998 -> int(998 * 100 / 999) = 99, inside the viewBox.  Rounding (or
        # adding 0.5) puts the same row at 100 and the document guard refuses it.
        exact = 998 * (100.0 / 999.0)
        self.assertEqual(int(exact), 99)
        self.assertEqual(int(exact + 0.5), 100)
        self.assertLess(exact, 100.0)


class WiringTests(unittest.TestCase):
    """The CLI stage: after selection, before the square-canvas pad and the emitter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.checkpoint = support.touch(os.path.join(self.root, "checkpoint.pt"))
        self.png = support.touch(os.path.join(self.root, "sample.png"))

    def tearDown(self):
        self.tmp.cleanup()

    def out(self, name):
        return os.path.join(self.root, name)

    def run_png(self, out, png_loader=None, **overrides):
        base = {
            "png": self.png,
            "out": self.out(out),
            "checkpoint": self.checkpoint,
        }
        base.update(overrides)
        return delivery.run_delivery(
            support.args(**base),
            loader=support.fake_loader,
            predict=support.default_predict,
            png_loader=png_loader or support.flat_png_loader,
        )

    def receipt(self, summary):
        with open(summary["run_receipt"], encoding="utf-8") as handle:
            return json.load(handle)

    def documents(self, summary):
        loaded = []
        for path in summary["results"]:
            with open(os.path.join(summary["out"], path), encoding="utf-8") as handle:
                loaded.append(json.load(handle))
        return loaded

    def test_default_rule_is_a6_and_the_receipt_records_it(self):
        summary = self.run_png("a6")
        receipt = self.receipt(summary)
        self.assertEqual(receipt["refine_mask"]["rule"], "a6")
        self.assertEqual(receipt["refine_mask"]["ink"], "nonzero")
        self.assertEqual(receipt["refine_mask"]["component_coverage"], 0.5)
        self.assertEqual(receipt["policy"]["refine"]["rule"], "a6")
        # the selection signature stays the five-number one the receipts pin
        self.assertEqual(receipt["selection_signature"], [0.05, 0.5, 0.5, 300, None])
        self.assertEqual(receipt["refine_mask"]["instances_selected"], 3)
        self.assertEqual(receipt["refine_mask"]["instances_refined"], 3)
        self.assertTrue(receipt["guards"]["refine_kept_instance_count"])
        self.assertEqual(receipt["guards"]["refine_dropped_instances"], 0)
        self.assertEqual(receipt["guards"]["refine_rule"], "a6")
        fallbacks = receipt["refine_mask"]["fallback_counts"]
        self.assertEqual(set(fallbacks), {
            "a6_no_kept_component", "a6_to_a1", "a1_to_original", "a1_rule_empty", "dropped_instances",
        })
        self.assertEqual(receipt["refine_mask"]["ink_sources"], {"loader": 1})
        self.assertTrue(receipt["refine_mask"]["images_refined"] == 1)
        with open(os.path.join(summary["out"], "selection_report.json"), encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["refine_mask"]["rule"], "a6")
        self.assertEqual(report["images"][0]["refine"]["rule"], "a6")

    def test_none_reproduces_the_unrefined_documents(self):
        off = self.run_png("off", refine_mask="none")
        on = self.run_png("on", refine_mask="a6")
        self.assertEqual(self.documents(off), self.documents(on))
        # the fixture ink is exactly the largest predicted mask, so a6 keeps it as is
        self.assertEqual(self.receipt(off)["refine_mask"]["instances_refined"], 0)
        self.assertEqual(self.receipt(on)["refine_mask"]["instances_changed"], 0)
        self.assertEqual(
            self.receipt(off)["refine_mask"]["instances_nonempty_before"],
            self.receipt(on)["refine_mask"]["instances_nonempty_after"],
        )

    def test_refinement_reads_the_declared_png_and_keeps_the_instance_count(self):
        from PIL import Image

        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[6:10, 4:12] = (0, 178, 178, 255)  # a thin bar inside the 4:12 x 4:12 mask
        png = os.path.join(self.root, "real.png")
        Image.fromarray(rgba, mode="RGBA").save(png)

        off = self.run_png("off", png=png, refine_mask="none")
        on = self.run_png("on", png=png, refine_mask="a6")
        receipt = self.receipt(on)
        self.assertEqual(receipt["refine_mask"]["ink_sources"], {"png": 1})
        self.assertEqual(receipt["refine_mask"]["ink_pixels"], 32)
        self.assertGreaterEqual(receipt["refine_mask"]["instances_changed"], 1)
        self.assertEqual(
            receipt["refine_mask"]["instances_nonempty_before"],
            receipt["refine_mask"]["instances_nonempty_after"],
        )
        self.assertEqual(
            self.receipt(off)["refine_mask"]["per_image"][0]["instances_nonempty_after"],
            receipt["refine_mask"]["per_image"][0]["instances_nonempty_after"],
        )
        self.assertEqual(len(self.documents(off)), len(self.documents(on)))
        self.assertNotEqual(self.documents(off)[0], self.documents(on)[0])

    def test_refinement_runs_on_the_native_raster_before_the_square_pad(self):
        width, height = 1000, 999
        ink = np.zeros((height, width), dtype=bool)
        ink[995:999, 100:200] = True
        mask = np.zeros((height, width), dtype=bool)
        mask[990:999, 90:210] = True

        def png_loader(_path):
            image = np.zeros((3, height, width), dtype=np.float32)
            image[:, 995:999, 100:200] = 1.0
            return image, ink, {"width": width, "height": height}

        policy = delivery.validate_selection(score_threshold=0.99, per_image_cap=12)
        run = delivery.DeliveryRun(policy, png_impl=png_loader)
        run.png_loader(os.path.join(self.root, "absent.png"))
        emitted = run.emit([mask], [0], [0.995], height=height, width=width)
        record = run.images[0]
        self.assertEqual(record["refine"]["native"], [width, height])
        self.assertEqual(record["refine"]["source"], "loader")
        self.assertEqual(record["refine"]["instances_nonempty_before"], 1)
        self.assertEqual(record["refine"]["instances_nonempty_after"], 1)
        self.assertEqual(record["refine"]["stats"]["kept_components"], 1)
        self.assertEqual(
            record["emit_canvas"],
            {
                "width": 1000,
                "height": 1000,
                "padded_to_square": True,
                "crop_residual_px": 1,
                "canvas_padding": {"left": 0, "top": 0, "right": 0, "bottom": 1},
                "unusual_aspect": False,
                "note": None,
            },
        )
        self.assertEqual(int(emitted["diagnostics"]["emitted_instances"]), 1)
        # the pad adds the missing bottom row; the refined content does not move
        padded, side, was_padded = delivery.square_canvas_masks(
            [refine.refine_mask(mask, ink, rule="a6")], width, height
        )
        self.assertTrue(was_padded)
        self.assertEqual(side, 1000)
        self.assertEqual(padded[0].shape, (1000, 1000))
        self.assertEqual(np.flatnonzero(padded[0].any(axis=1)).tolist(), [995, 996, 997, 998])

    def test_a1_is_selectable_and_needs_no_components(self):
        summary = self.run_png("a1", refine_mask="a1")
        receipt = self.receipt(summary)
        self.assertEqual(receipt["refine_mask"]["rule"], "a1")
        self.assertEqual(receipt["refine_mask"]["rule"], receipt["policy"]["refine"]["rule"])
        self.assertEqual(receipt["refine_mask"]["components_in_instance_crops"], 0)
        self.assertEqual(receipt["refine_mask"]["ink_sources"], {"loader": 1})

    def test_invalid_refine_knobs_are_refused_before_any_path_work(self):
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.validate_selection(refine_mask="a9")
        self.assertEqual(caught.exception.code, "refine_policy")
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.validate_selection(refine_ink="bright")
        self.assertEqual(caught.exception.code, "refine_policy")
        import contextlib
        import io

        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            delivery.main(
                ["--png", self.png, "--out", os.path.join(self.root, "bad"), "--refine-mask", "a2"]
            )
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(os.path.isdir(os.path.join(self.root, "bad")))


class RatioTests(unittest.TestCase):
    """``--refine-ratio``: the component-inclusion ratio, inclusive, 0.0..1.0.

    One field, five 16 px ink components, each with a different covered share:
    F 16/16 (100 %), D 14/16 (87.5 %), C 8/16 (50 %), E 7/16 (43.75 %), G 1/16
    (6.25 %).  The mask bbox widened by 2 px contains all five whole, so every
    fraction is honest rather than truncated by the crop.
    """

    def field(self):
        ink = np.zeros((20, 20), dtype=bool)
        ink[2:4, 2:10] = True    # C  16 px
        ink[6:8, 2:10] = True    # F  16 px
        ink[10:12, 2:10] = True  # D  16 px
        ink[13:15, 2:10] = True  # G  16 px
        ink[16:18, 2:10] = True  # E  16 px
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:4, 2:6] = True    # C: 8 of 16  -> 50.0 %
        mask[6:8, 2:10] = True   # F: 16 of 16 -> 100 %
        mask[10:12, 2:9] = True  # D: 14 of 16 -> 87.5 %
        mask[13, 2] = True       # G: 1 of 16  -> 6.25 %
        mask[16:17, 2:9] = True  # E: 7 of 16  -> 43.75 %
        return ink, mask

    def refined(self, ratio, ink, mask, stats=None):
        return refine.refine_mask(mask, ink, rule="a6", stats=stats, coverage=ratio)

    def test_ratio_0_keeps_a_single_inside_pixel_and_a_wholly_inside_component(self):
        ink, mask = self.field()
        stats = refine.new_stats()
        refined = self.refined(0.0, ink, mask, stats)
        # every present component is kept, so the whole ink field comes back
        np.testing.assert_array_equal(refined, ink)
        self.assertEqual(int(refined.sum()), 80)
        self.assertEqual(stats["kept_components"], 5)
        self.assertEqual(stats["components"], 5)
        self.assertEqual(stats["a6_to_a1"], 0)
        self.assertEqual(stats["instances_without_component"], 0)
        self.assertEqual(stats["changed"], 1)
        self.assertEqual(stats["dropped_instances"], 0)

    def test_ratio_0_5_keeps_only_components_at_least_half_covered(self):
        ink, mask = self.field()
        stats = refine.new_stats()
        refined = self.refined(0.5, ink, mask, stats)
        expected = np.zeros_like(ink)
        expected[2:4, 2:10] = True   # C, exactly 50 %, inclusive
        expected[6:8, 2:10] = True   # F
        expected[10:12, 2:10] = True  # D
        np.testing.assert_array_equal(refined, expected)
        self.assertEqual(int(refined.sum()), 48)
        self.assertEqual(stats["kept_components"], 3)
        self.assertEqual(stats["components"], 5)
        self.assertEqual(stats["a6_to_a1"], 0)

    def test_ratio_1_0_keeps_only_fully_inside_components(self):
        ink, mask = self.field()
        stats = refine.new_stats()
        refined = self.refined(1.0, ink, mask, stats)
        expected = np.zeros_like(ink)
        expected[6:8, 2:10] = True  # F is the only fully covered one
        np.testing.assert_array_equal(refined, expected)
        self.assertEqual(int(refined.sum()), 16)
        self.assertEqual(stats["kept_components"], 1)
        self.assertEqual(stats["a6_to_a1"], 0)

    def test_default_ratio_is_the_shipped_half(self):
        ink, mask = self.field()
        np.testing.assert_array_equal(
            refine.refine_mask(mask, ink, rule="a6"),
            self.refined(refine.COMPONENT_COVERAGE, ink, mask),
        )
        self.assertEqual(refine.validate_coverage(None), 0.5)
        self.assertEqual(refine.validate_coverage(0), 0.0)
        self.assertEqual(refine.validate_coverage(0.0), 0.0)  # falsy, still a value
        self.assertEqual(refine.validate_coverage(1), 1.0)

    def test_bad_ratios_are_refused_before_the_mask_is_read(self):
        for bad in (1.5, -0.001, float("nan"), float("inf"), float("-inf"), True, "0.5", [0.5]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    refine.validate_coverage(bad)
                # mask/ink are None: a validator that ran after the arrays would raise
                # TypeError from np.asarray first, so ValueError proves the ordering
                with self.assertRaises(ValueError):
                    refine.refine_mask(None, None, rule="a6", coverage=bad)
                with self.assertRaises(ValueError):
                    refine.refine_masks([], None, rule="a6", coverage=bad)

    def test_fallbacks_and_instance_count_hold_at_every_ratio(self):
        ink, mask = self.field()
        empty_under_mask = np.zeros_like(mask)
        empty_under_mask[19, 19] = True  # no ink at all under this one
        masks = [mask, ink & ~mask, empty_under_mask, np.zeros_like(mask)]
        nonempty = sum(1 for item in masks if item.any())
        for ratio in (0.0, 0.01, 0.25, 0.5, 0.75, 0.9375, 0.99, 1.0):
            with self.subTest(ratio=ratio):
                stats = refine.new_stats()
                refined = refine.refine_masks(masks, ink, rule="a6", stats=stats, coverage=ratio)
                self.assertEqual(len(refined), len(masks))
                self.assertEqual(sum(1 for item in refined if item.any()), nonempty)
                self.assertEqual(stats["instances"], len(masks))
                self.assertEqual(stats["dropped_instances"], 0)
                # an instance with no component kept never comes back empty
                self.assertTrue(refined[2].any())
                np.testing.assert_array_equal(refined[2], empty_under_mask)
                # the empty instance stays empty, as the sealed emitter expects
                self.assertFalse(refined[3].any())
        # ratio 1.0 keeps no component for this field, so the a6 -> a1 fallback fires
        stats = refine.new_stats()
        fallback = refine.refine_mask(mask, np.where(mask, False, True), rule="a6",
                                      stats=stats, coverage=1.0)
        self.assertEqual(stats["a6_to_a1"], 1)
        np.testing.assert_array_equal(fallback, mask)


class RatioWiringTests(unittest.TestCase):
    """``--refine-ratio`` through the CLI, the receipt and the guards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.checkpoint = support.touch(os.path.join(self.root, "checkpoint.pt"))
        self.png = support.touch(os.path.join(self.root, "sample.png"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_png(self, out, **overrides):
        base = {"png": self.png, "out": os.path.join(self.root, out), "checkpoint": self.checkpoint}
        base.update(overrides)
        summary = delivery.run_delivery(
            support.args(**base),
            loader=support.fake_loader,
            predict=support.default_predict,
            png_loader=support.flat_png_loader,
        )
        with open(summary["run_receipt"], encoding="utf-8") as handle:
            return summary, json.load(handle)

    def test_receipt_records_the_ratio_actually_used(self):
        summary, receipt = self.run_png("zero", refine_mask="a6", refine_ratio=0.0)
        self.assertEqual(receipt["policy"]["refine"]["component_coverage"], 0.0)
        self.assertEqual(receipt["refine_mask"]["component_coverage"], 0.0)
        self.assertEqual(receipt["refine_mask"]["component_coverage_observed"], [0.0])
        self.assertTrue(receipt["refine_mask"]["component_coverage_consistent"])
        self.assertEqual(receipt["guards"]["refine_component_coverage"], 0.0)
        self.assertTrue(receipt["guards"]["refine_component_coverage_consistent"])
        self.assertEqual(summary["refine_mask"]["component_coverage"], 0.0)
        self.assertEqual(receipt["refine_mask"]["component_coverage_default"], 0.5)
        self.assertEqual(receipt["refine_mask"]["component_coverage_range"], [0.0, 1.0])
        self.assertIn(">=", receipt["refine_mask"]["component_coverage_rule"])

    def test_default_run_still_records_the_half_ratio(self):
        _summary, receipt = self.run_png("default")
        self.assertEqual(receipt["refine_mask"]["component_coverage"], 0.5)
        self.assertEqual(receipt["guards"]["refine_component_coverage"], 0.5)
        self.assertTrue(receipt["refine_mask"]["component_coverage_consistent"])

    def test_ratio_0_never_keeps_less_ink_than_the_half_rule(self):
        from PIL import Image

        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[5:12, 3:14] = (0, 178, 178, 255)
        rgba[24:26, 26:30] = (0, 178, 178, 255)  # tiny, far component: 1 px under the mask
        png = os.path.join(self.root, "ink.png")
        Image.fromarray(rgba, mode="RGBA").save(png)

        _half, half = self.run_png("half", png=png, refine_mask="a6", refine_ratio=0.5)
        _zero, zero = self.run_png("zero", png=png, refine_mask="a6", refine_ratio=0.0)
        self.assertGreaterEqual(
            zero["refine_mask"]["kept_components"], half["refine_mask"]["kept_components"]
        )
        self.assertEqual(
            zero["refine_mask"]["instances_nonempty_before"],
            zero["refine_mask"]["instances_nonempty_after"],
        )

    def test_a1_and_none_ignore_the_ratio_but_still_record_it(self):
        _s, receipt = self.run_png("a1", refine_mask="a1", refine_ratio=0.0)
        self.assertEqual(receipt["refine_mask"]["rule"], "a1")
        self.assertEqual(receipt["refine_mask"]["component_coverage"], 0.0)
        self.assertEqual(receipt["refine_mask"]["components_in_instance_crops"], 0)

    def test_bad_ratio_is_refused_before_the_output_directory_exists(self):
        for bad in ("1.5", "-0.5", "nan", "inf"):
            with self.subTest(bad=bad):
                out = os.path.join(self.root, "refused-" + bad.replace("-", "m").replace(".", "d"))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    code = delivery.main(["--png", self.png, "--out", out, "--refine-ratio", bad])
                self.assertEqual(code, 2)
                self.assertFalse(os.path.isdir(out))
                self.assertFalse(os.path.exists(out))
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.validate_selection(refine_ratio=1.000001)
        self.assertEqual(caught.exception.code, "refine_policy")
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.validate_selection(refine_ratio=float("nan"))
        self.assertEqual(caught.exception.code, "refine_policy")

    def test_ratio_is_on_the_command_line_and_in_the_policy(self):
        parser = delivery.build_parser()
        args = parser.parse_args(["--png", self.png, "--out", os.path.join(self.root, "x")])
        self.assertEqual(args.refine_ratio, 0.5)
        args = parser.parse_args(
            ["--png", self.png, "--out", os.path.join(self.root, "x"), "--refine-ratio", "0"]
        )
        self.assertEqual(args.refine_ratio, 0.0)
        policy = delivery.validate_selection(refine_ratio=1.0)
        self.assertEqual(policy["refine"]["component_coverage"], 1.0)
        self.assertEqual(policy["refine"]["component_coverage_range"], [0.0, 1.0])
        # the frozen five-number selection signature is untouched
        self.assertEqual(policy["signature"], [0.05, 0.5, 0.5, 300, None])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
