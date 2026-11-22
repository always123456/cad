"""Both mask-paste flags: validation, the prob-mode byte identity, and the logits mechanism.

The cheap tests (validation, the untouched prob path, the receipt records) run on
the CPU with no torch at all.  The mechanism tests patch the installed
torchvision, so they skip when torch/torchvision is not importable.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

from tests.support import args, delivery, fake_loader, flat_png_loader, touch

REFUSE = delivery.DeliveryRefused


def read_documents(out):
    results = os.path.join(out, "results")
    documents = {}
    for name in sorted(os.listdir(results)):
        with open(os.path.join(results, name), "rb") as handle:
            documents[name] = handle.read()
    return documents


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def flat_predict(_context, _image):
    """The fixture 32x32 image with one hard-on instance and one duplicate."""

    from tests.support import DEFAULT_SPECS, detections

    return detections(*DEFAULT_SPECS)


def soft_predict(value):
    """One instance per score, its mask a flat ``value`` over a 6x6 block."""

    import numpy as np

    def predict(_context, _image):
        mask = np.zeros((1, 32, 32), dtype=np.float32)
        mask[:, 10:16, 10:16] = float(value)
        return {
            "boxes": np.array([[10.0, 10.0, 16.0, 16.0]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "scores": np.array([0.9], dtype=np.float32),
            "mask_probabilities": mask,
            "score_threshold": 0.05,
            "mask_threshold": 0.5,
        }

    return predict


def _torch_available():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the interpreter
        return False


TORCH = _torch_available()


class MaskPastePolicyTest(unittest.TestCase):
    def policy(self, **knobs):
        return delivery.validate_selection(**knobs)

    def assertRefused(self, code, **knobs):
        with self.assertRaises(REFUSE) as ctx:
            self.policy(**knobs)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_default_installs_nothing(self):
        self.assertIsNone(self.policy().get("mask_paste"))
        self.assertIsNone(self.policy(mask_paste="prob").get("mask_paste"))

    def test_logits_block(self):
        block = self.policy(mask_paste="logits")["mask_paste"]
        self.assertEqual(block["mode"], "logits")
        self.assertEqual(block["logit_pad"], delivery.DEFAULT_MASK_LOGIT_PAD)
        self.assertEqual(block["logit_zero_probability"], 0.5)
        self.assertIn("logit >= 0", block["boundary"])
        self.assertIn("predicted-class", " ".join(block["unchanged"]))
        self.assertIn("int64", " ".join(block["unchanged"]))

    def test_custom_pad(self):
        block = self.policy(mask_paste="logits", mask_logit_pad=-20)["mask_paste"]
        self.assertEqual(block["logit_pad"], -20.0)
        self.assertIsInstance(block["logit_pad"], float)

    def test_the_paste_does_not_move_the_selection_signature(self):
        # the paste is a decode setting outside the five selection numbers; the
        # threshold is the second number of the signature, so it does move it
        self.assertEqual(self.policy(mask_paste="logits")["signature"],
                         [0.05, 0.5, 0.5, 300, None])
        self.assertEqual(
            self.policy(mask_paste="logits", mask_threshold=0.35)["signature"][1], 0.35
        )

    def test_mode_refusals(self):
        for bad in ("logit", "LOGITS", "prob ", 0, 1.0, True, []):
            with self.subTest(bad=bad):
                refused = self.assertRefused("selection_policy", mask_paste=bad)
                self.assertIn("--mask-paste", refused.detail)

    def test_pad_only_with_logits(self):
        refused = self.assertRefused("selection_policy", mask_logit_pad=-12.0)
        self.assertIn("--mask-logit-pad only applies", refused.detail)
        self.assertRefused("selection_policy", mask_paste="prob", mask_logit_pad=-12.0)

    def test_pad_must_be_a_very_negative_finite_logit(self):
        for bad in (0, 0.0, -0.5, 1.0, float("nan"), float("inf"), True, "x"):
            with self.subTest(bad=bad):
                refused = self.assertRefused(
                    "selection_policy", mask_paste="logits", mask_logit_pad=bad
                )
                self.assertIn("--mask-logit-pad", refused.detail)


class MaskThresholdChoicesTest(unittest.TestCase):
    def policy(self, **knobs):
        return delivery.validate_selection(**knobs)

    def assertRefused(self, code, **knobs):
        with self.assertRaises(REFUSE) as ctx:
            self.policy(**knobs)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_measured_choices_are_accepted(self):
        for choice in delivery.MASK_THRESHOLD_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(self.policy(mask_threshold=choice)["mask_threshold"], choice)

    def test_other_values_are_refused(self):
        for bad in (0.42, 0.48, 0.3, 0.51, 0.9):
            with self.subTest(bad=bad):
                refused = self.assertRefused("selection_policy", mask_threshold=bad)
                self.assertIn("binarises", refused.detail)
                self.assertIn("0.45", refused.detail)

    def test_bounds_still_apply(self):
        self.assertRefused("selection_policy", mask_threshold=0)
        self.assertRefused("selection_policy", mask_threshold=1)


class ProbPasteByteIdentityTest(unittest.TestCase):
    """``--mask-paste prob`` must be the run that mrcnn05 already shipped."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="paste-prob-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"))
        self.png = touch(os.path.join(self.tmp, "drawing.png"))

    def run_delivery(self, name, **overrides):
        out = os.path.join(self.tmp, name)
        summary = delivery.run_delivery(
            args(png=self.png, out=out, checkpoint=self.checkpoint, refine_mask="none", **overrides),
            loader=fake_loader,
            predict=flat_predict,
            png_loader=flat_png_loader,
        )
        return out, summary

    def test_prob_documents_and_receipt_are_the_default_run(self):
        base_out, _ = self.run_delivery("base")
        prob_out, _ = self.run_delivery("prob", mask_paste="prob")
        self.assertEqual(read_documents(base_out), read_documents(prob_out))
        base_receipt = read_json(os.path.join(base_out, "run_receipt.json"))
        prob_receipt = read_json(os.path.join(prob_out, "run_receipt.json"))
        for key in ("mask_paste", "mask_threshold"):
            self.assertNotIn(key, base_receipt)
            self.assertNotIn(key, prob_receipt)
        self.assertEqual(base_receipt["policy"], prob_receipt["policy"])
        self.assertEqual(base_receipt["guards"], prob_receipt["guards"])
        self.assertFalse(base_receipt["guards"]["decode_settings_from_flags"])

    def test_prob_is_the_untouched_predict(self):
        # structural guarantee: with the default paste the wrapper is never built,
        # so the run calls the injected predict and nothing else.
        seen = []

        def spy(context, image):
            seen.append(1)
            return flat_predict(context, image)

        with mock.patch.object(delivery.mask_paste, "wrap_predict") as wrap:
            delivery.run_delivery(
                args(png=self.png, out=os.path.join(self.tmp, "spy"),
                     checkpoint=self.checkpoint, refine_mask="none"),
                loader=fake_loader,
                predict=spy,
                png_loader=flat_png_loader,
            )
        wrap.assert_not_called()
        self.assertEqual(len(seen), 1)


class MaskThresholdReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="paste-threshold-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"))
        self.png = touch(os.path.join(self.tmp, "drawing.png"))

    def run_delivery(self, name, predict=None, **overrides):
        out = os.path.join(self.tmp, name)
        delivery.run_delivery(
            args(png=self.png, out=out, checkpoint=self.checkpoint, refine_mask="none", **overrides),
            loader=fake_loader,
            predict=predict or flat_predict,
            png_loader=flat_png_loader,
        )
        return out

    def test_the_receipt_records_the_value(self):
        out = self.run_delivery("t45", mask_threshold=0.45)
        receipt = read_json(os.path.join(out, "run_receipt.json"))
        self.assertEqual(receipt["policy"]["mask_threshold"], 0.45)
        self.assertEqual(receipt["mask_threshold"]["value"], 0.45)
        self.assertEqual(receipt["mask_threshold"]["default"], 0.5)
        self.assertEqual(receipt["mask_threshold"]["choices"], [0.5, 0.45, 0.4, 0.35])
        self.assertTrue(receipt["guards"]["decode_settings_from_flags"])
        self.assertNotIn("mask_paste", receipt)

    def test_the_threshold_reaches_the_decoder_binarisation(self):
        # a 0.47 mask is dropped at 0.5 and kept at 0.45 by the lane's own
        # binarisation -- proof the value travels through run_batch, not the CLI
        at_default = self.run_delivery("d", predict=soft_predict(0.47))
        at_45 = self.run_delivery("d45", predict=soft_predict(0.47), mask_threshold=0.45)

        def primitives(out):
            index = read_json(os.path.join(out, "results_index.json"))
            target = os.path.join(out, index["images"]["drawing"]["result"])
            return len(read_json(target)["semanticIds"])

        self.assertEqual(primitives(at_default), 0)
        self.assertGreater(primitives(at_45), 0)


class LogitPasteMechanismTest(unittest.TestCase):
    """The patch itself: padding, the boundary, and the sigmoid round trip."""

    def test_importable_without_torch(self):
        # the CLI imports mask_paste unconditionally, so the module must not need
        # torch at import time
        import mask_paste

        self.assertEqual(mask_paste.DEFAULT_MASK_PASTE, "prob")
        self.assertEqual(mask_paste.MASK_PASTE_MODES, ("prob", "logits"))

    def test_wrap_predict_puts_the_boundary_on_logit_zero(self):
        import mask_paste
        import numpy as np

        logits = np.array([[-2.0, -0.001, 0.0, 0.001, 2.0]], dtype=np.float32)
        wrapped = mask_paste.wrap_predict(lambda _c, _i: {"mask_probabilities": logits}, -12.0)
        with mock.patch.object(mask_paste, "logit_paste", lambda pad: contextlib.nullcontext()):
            result = wrapped(None, None)
        values = result["mask_probabilities"]
        self.assertLess(values[0, 0], 0.5)
        self.assertLess(values[0, 1], 0.5)
        self.assertEqual(float(values[0, 2]), 0.5)
        self.assertGreater(values[0, 3], 0.5)
        self.assertGreater(values[0, 4], 0.5)
        self.assertEqual(result["mask_paste"], "logits")
        self.assertEqual(result["mask_logit_pad"], -12.0)

    def test_wrap_predict_leaves_the_rest_alone(self):
        import mask_paste
        import numpy as np

        base = {
            "boxes": np.zeros((1, 4), dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
            "scores": np.array([0.9], dtype=np.float32),
            "mask_probabilities": np.zeros((1, 4, 4), dtype=np.float32),
            "score_threshold": 0.05,
            "mask_threshold": 0.5,
            "extra": "kept",
        }
        with mock.patch.object(mask_paste, "logit_paste", lambda pad: contextlib.nullcontext()):
            result = mask_paste.wrap_predict(lambda _c, _i: base) (None, None)
        for key in ("boxes", "labels", "scores", "score_threshold", "mask_threshold", "extra"):
            self.assertIn(key, result)
        self.assertIs(result["labels"], base["labels"])
        self.assertIsNot(result["mask_probabilities"], base["mask_probabilities"])

    @unittest.skipUnless(TORCH, "torch/torchvision not importable here")
    def test_patch_installs_and_restores(self):
        import torch
        from torchvision.models.detection import roi_heads

        import mask_paste

        before = {
            "maskrcnn_inference": roi_heads.maskrcnn_inference,
            "expand_masks": roi_heads.expand_masks,
            "paste_mask_in_image": roi_heads.paste_mask_in_image,
        }
        with mask_paste.logit_paste(-12.0) as ctx:
            module = ctx._module
            self.assertIs(module, roi_heads)
            self.assertIs(module.maskrcnn_inference, mask_paste._maskrcnn_inference_logits)
            self.assertIsNot(module.expand_masks, before["expand_masks"])
            self.assertIsNot(module.paste_mask_in_image, before["paste_mask_in_image"])
            padded, scale = module.expand_masks(torch.zeros(1, 1, 112, 112), 1)
            self.assertEqual(float(padded[0, 0, 0, 0]), -12.0)
            self.assertEqual(float(padded[0, 0, 1, 1]), 0.0)
            self.assertAlmostEqual(scale, 114.0 / 112.0, places=12)
        for name, function in before.items():
            self.assertIs(getattr(roi_heads, name), function)

    @unittest.skipUnless(TORCH, "torch/torchvision not importable here")
    def test_the_canvas_outside_the_box_is_background(self):
        """torchvision fills the canvas with 0.0; in logits space that is p 0.5."""

        import torch
        from torchvision.models.detection import roi_heads

        import mask_paste

        mask = torch.zeros(8, 8)
        boxes = torch.tensor([[2.0, 2.0, 5.0, 5.0]])
        inside = roi_heads.paste_masks_in_image(mask.unsqueeze(0).unsqueeze(0), boxes, (10, 10))
        with mask_paste.logit_paste(-12.0):
            pasted = roi_heads.paste_masks_in_image(mask.unsqueeze(0).unsqueeze(0), boxes, (10, 10))

        self.assertEqual(float(inside.max()), 0.0)
        self.assertEqual(float(pasted.min()), -12.0)
        self.assertEqual(float(pasted[:, :, 6:, :].max()), -12.0)
        # inside the pasted box the geometry is torchvision's own
        self.assertEqual(
            tuple(pasted[:, :, 0, :].shape), tuple(inside[:, :, 0, :].shape)
        )

    @unittest.skipUnless(TORCH, "torch/torchvision not importable here")
    def test_the_box_interior_is_torchvisions_own_resample(self):
        import torch
        from torchvision.models.detection import roi_heads

        import mask_paste

        mask = torch.linspace(-3.0, 3.0, 112 * 112).reshape(1, 1, 112, 112)
        box = torch.tensor([10, 20, 300, 400])

        expected = roi_heads.paste_mask_in_image(mask[0, 0], box, 500, 600)
        with mask_paste.logit_paste(-12.0):
            actual = roi_heads.paste_mask_in_image(mask[0, 0], box, 500, 600)
        torch.testing.assert_close(
            actual[20:401, 10:301], expected[20:401, 10:301], rtol=0, atol=0
        )
        self.assertEqual(float(actual.min()), -12.0)
        self.assertEqual(float(expected[0, 0]), 0.0)
        self.assertEqual(float(actual[0, 0]), -12.0)

    @unittest.skipUnless(TORCH, "torch/torchvision not importable here")
    def test_logits_inference_skips_the_sigmoid(self):
        import torch

        import mask_paste

        logits = torch.tensor(
            [
                [[[2.0, -3.0], [0.0, 0.5]], [[9.0, 9.0], [9.0, 9.0]], [[1.0, 1.0], [1.0, 1.0]]],
                [[[7.0, 7.0], [7.0, 7.0]], [[8.0, 8.0], [8.0, 8.0]], [[6.0, 6.0], [6.0, 6.0]]],
            ]
        )
        labels = [torch.tensor([2, 0])]
        selected = mask_paste._maskrcnn_inference_logits(logits, labels)[0]
        # class 2 for the first box, class 0 for the second: the raw logits, no sigmoid
        self.assertEqual(float(selected[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(selected[1, 0, 1, 1]), 7.0)
        self.assertEqual(tuple(selected.shape), (2, 1, 2, 2))

    @unittest.skipUnless(TORCH, "torch/torchvision not importable here")
    def test_the_probability_field_is_the_sigmoid_of_the_pasted_logits(self):
        """The decision is exactly ``pasted logit >= 0``, i.e. the same 0.5 test.

        This is the property that makes the variant legal: whatever bilinear does
        to the field, the lane's ``p >= mask_threshold`` on the returned
        probabilities is the same predicate as ``logit >= 0`` on the pasted logits.
        """

        import numpy as np
        import torch
        from torchvision.models.detection import roi_heads

        import mask_paste

        size = 112
        logits = torch.full((1, 1, size, size), -0.847)
        logits[0, 0, 56, 8:104] = 0.405  # a 1-cell stroke, p = 0.60
        boxes = torch.tensor([[24.0, 24.0, 136.0, 136.0]])
        shape = (1000, 1000)

        probabilities = roi_heads.paste_masks_in_image(logits.sigmoid(), boxes, shape)
        with mask_paste.logit_paste(-12.0):
            pasted = roi_heads.paste_masks_in_image(logits, boxes, shape)

        converted = mask_paste.sigmoid(pasted.numpy())
        np.testing.assert_allclose(converted, pasted.sigmoid().numpy(), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(converted >= 0.5, pasted.numpy() >= 0.0)
        # sanity on the fixture itself: the stroke is a real peak in both fields
        self.assertGreaterEqual(float(probabilities.max()), 0.5)
        self.assertGreaterEqual(float(pasted.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
