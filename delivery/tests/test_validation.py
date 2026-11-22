"""Fail-closed knob validation, and proof that it runs before any model or data access."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.support import args, delivery, touch

REFUSE = delivery.DeliveryRefused


class SelectionKnobTest(unittest.TestCase):
    def policy(self, **knobs):
        return delivery.validate_selection(**knobs)

    def assertRefused(self, code, **knobs):
        with self.assertRaises(REFUSE) as ctx:
            self.policy(**knobs)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    # -- defaults ----------------------------------------------------------- #
    def test_defaults_are_the_measured_decode_settings(self):
        policy = self.policy()
        self.assertEqual(policy["score_threshold"], 0.05)
        self.assertEqual(policy["mask_threshold"], 0.5)
        self.assertEqual(policy["nms_iou"], 0.5)
        self.assertEqual(policy["max_detections"], 300)
        self.assertIsNone(policy["per_image_cap"])
        self.assertEqual(policy["signature"], [0.05, 0.5, 0.5, 300, None])
        self.assertEqual(policy["decoder_fixed"]["box_detections_per_img"], 300)
        self.assertIn("strict", policy["comparisons"]["score"])
        self.assertTrue(policy["order"].startswith("candidates -> same-class mask IoU NMS"))

    def test_flags_are_additive(self):
        policy = self.policy(score_threshold=0.97, per_image_cap=15)
        self.assertEqual(policy["score_threshold"], 0.97)
        self.assertEqual(policy["per_image_cap"], 15)
        self.assertEqual(policy["max_detections"], 300)
        self.assertEqual(policy["nms_iou"], 0.5)

    # -- score threshold ---------------------------------------------------- #
    def test_score_threshold_bounds(self):
        self.assertEqual(self.policy(score_threshold=0.0001)["score_threshold"], 0.0001)
        self.assertEqual(self.policy(score_threshold=0.999)["score_threshold"], 0.999)
        for bad in (0, 0.0, -0.1, 1, 1.0, 1.5, 2):
            with self.subTest(bad=bad):
                self.assertRefused("selection_policy", score_threshold=bad)

    def test_score_threshold_rejects_non_finite_and_bool(self):
        for bad in (float("nan"), float("inf"), float("-inf"), True, False, "0.5", None.__class__):
            with self.subTest(bad=bad):
                self.assertRefused("selection_policy", score_threshold=bad)

    # -- nms ---------------------------------------------------------------- #
    def test_nms_iou_bounds(self):
        self.assertEqual(self.policy(nms_iou=0.3)["nms_iou"], 0.3)
        self.assertEqual(self.policy(nms_iou=1.0)["nms_iou"], 1.0)
        for bad in (0, 0.0, -0.5, 1.0001, 2, True, float("nan")):
            with self.subTest(bad=bad):
                self.assertRefused("selection_policy", nms_iou=bad)

    # -- caps --------------------------------------------------------------- #
    def test_max_detections_bounds(self):
        self.assertEqual(self.policy(max_detections=1)["max_detections"], 1)
        self.assertEqual(self.policy(max_detections=300)["max_detections"], 300)
        for bad in (0, -1, 301, 1000, 3.0, True, "300"):
            with self.subTest(bad=bad):
                self.assertRefused("selection_policy", max_detections=bad)

    def test_per_image_cap_bounds(self):
        policy = self.policy(per_image_cap=7)
        self.assertEqual(policy["per_image_cap"], 7)
        for bad in (0, -2, 301, 300 + 1):
            with self.subTest(bad=bad):
                self.assertRefused("selection_policy", per_image_cap=bad)

    def test_per_image_cap_cannot_exceed_max_detections(self):
        self.assertRefused("selection_policy", max_detections=10, per_image_cap=11)
        self.assertEqual(self.policy(max_detections=10, per_image_cap=10)["per_image_cap"], 10)

    def test_mask_threshold_choices(self):
        self.assertEqual(self.policy(mask_threshold=0.5)["mask_threshold"], 0.5)
        for choice in (0.45, 0.40, 0.35):
            with self.subTest(choice=choice):
                self.assertEqual(self.policy(mask_threshold=choice)["mask_threshold"], choice)
        refused = self.assertRefused("selection_policy", mask_threshold=0.42)
        self.assertIn("binarises", refused.detail)
        self.assertIn("0.45", refused.detail)

    def test_mask_threshold_is_in_the_signature(self):
        # the second element of the five-number selection signature is the
        # binarisation value: changing it must be visible in the receipt signature
        self.assertEqual(self.policy(mask_threshold=0.45)["signature"][1], 0.45)
        self.assertEqual(self.policy()["signature"][1], 0.5)

    def test_mask_threshold_still_bounds_checked(self):
        self.assertRefused("selection_policy", mask_threshold=0)
        self.assertRefused("selection_policy", mask_threshold=1)


class DeliveryPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-path-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"))
        self.png = touch(os.path.join(self.tmp, "drawing.png"))
        self.out = os.path.join(self.tmp, "out")

    def check(self, **overrides):
        base = dict(out=self.out, checkpoint=self.checkpoint, png=self.png, inputs=None, device="cuda:0")
        base.update(overrides)
        return delivery.validate_delivery_paths(**base)

    def test_accepts_png_or_inputs(self):
        plan = self.check()
        self.assertEqual(plan["device"], "cuda:0")
        inputs = touch(os.path.join(self.tmp, "in.json"), "{\"samples\": []}")
        plan = self.check(png=None, inputs=inputs)
        self.assertEqual(plan["inputs"], os.path.abspath(inputs))

    def test_refuses_both_or_neither(self):
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=self.png, inputs="x.json")
        self.assertEqual(ctx.exception.code, "inputs_choice")
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=None, inputs=None)
        self.assertEqual(ctx.exception.code, "inputs_choice")

    def test_refuses_bad_device(self):
        for bad in ("gpu:0", "cuda:x", "cuda:0,1", "", "CPU0"):
            with self.subTest(bad=bad):
                with self.assertRaises(REFUSE) as ctx:
                    self.check(device=bad)
                self.assertEqual(ctx.exception.code, "device")
        for good in ("cpu", "cuda", "cuda:0", "cuda:3"):
            self.assertEqual(self.check(device=good)["device"], good)

    def test_refuses_missing_checkpoint(self):
        with self.assertRaises(REFUSE) as ctx:
            self.check(checkpoint=os.path.join(self.tmp, "nope.pt"))
        self.assertEqual(ctx.exception.code, "checkpoint_missing")

    def test_refuses_missing_or_non_png(self):
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=os.path.join(self.tmp, "nope.png"))
        self.assertEqual(ctx.exception.code, "png_missing")
        svg = touch(os.path.join(self.tmp, "drawing.svg"))
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=svg)
        self.assertEqual(ctx.exception.code, "png_required")

    def test_refuses_non_empty_out(self):
        os.makedirs(self.out)
        touch(os.path.join(self.out, "keep.txt"))
        with self.assertRaises(REFUSE) as ctx:
            self.check()
        self.assertEqual(ctx.exception.code, "out_not_empty")

    def test_refuses_out_that_is_a_file(self):
        target = touch(os.path.join(self.tmp, "outfile"))
        with self.assertRaises(REFUSE) as ctx:
            self.check(out=target)
        self.assertEqual(ctx.exception.code, "out_not_directory")

    def test_refuses_sealed_paths(self):
        sealed = touch(os.path.join(self.tmp, "test_final", "x.png"))
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=sealed)
        self.assertEqual(ctx.exception.code, "sealed_path")


class OrderingTest(unittest.TestCase):
    """A refusal must happen before any lane import, checkpoint or PNG access."""

    def test_knob_refusal_precedes_every_import(self):
        tmp = tempfile.mkdtemp(prefix="delivery-order-")
        buffer = io.StringIO()
        with mock.patch.object(delivery, "_contract", side_effect=AssertionError("lane imported")):
            with mock.patch.object(delivery, "_batch", side_effect=AssertionError("lane imported")):
                with contextlib.redirect_stdout(buffer):
                    exit_code = delivery.main(
                        [
                            "--png",
                            os.path.join(tmp, "missing.png"),
                            "--out",
                            os.path.join(tmp, "out"),
                            "--score-threshold",
                            "1.0",
                        ]
                    )
        self.assertEqual(exit_code, 2)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["code"], "selection_policy")
        self.assertIn("score-threshold", payload["detail"])

    def test_path_refusal_happens_before_model_access(self):
        tmp = tempfile.mkdtemp(prefix="delivery-order-")
        checkpoint = touch(os.path.join(tmp, "checkpoint.pt"))
        buffer = io.StringIO()
        with mock.patch.object(delivery, "_contract", side_effect=AssertionError("lane imported")):
            with mock.patch.object(delivery, "_batch", side_effect=AssertionError("lane imported")):
                with contextlib.redirect_stdout(buffer):
                    exit_code = delivery.main(
                        [
                            "--png",
                            os.path.join(tmp, "missing.png"),
                            "--out",
                            os.path.join(tmp, "out"),
                            "--checkpoint",
                            checkpoint,
                        ]
                    )
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(buffer.getvalue())["code"], "png_missing")

    def test_preflight_refusal_leaves_out_untouched(self):
        tmp = tempfile.mkdtemp(prefix="delivery-order-")
        out = os.path.join(tmp, "out")
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            exit_code = delivery.main(
                ["--png", os.path.join(tmp, "a.png"), "--out", out, "--score-threshold", "0"]
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(buffer.getvalue())["code"], "selection_policy")
        self.assertFalse(os.path.exists(out))

    def test_argparse_requires_a_source_and_an_out(self):
        for argv in ([], ["--png", "x.png"], ["--out", "o"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        delivery.main(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                delivery.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
