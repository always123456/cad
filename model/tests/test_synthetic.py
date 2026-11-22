"""CPU synthetic tests for the direct-instance Mask R-CNN lane.

Run from ``outputs/recognition-direct-instance-20261114-01``::

    python -m unittest model.tests.test_synthetic -v

No COCO download, dataset, GT annotation or GPU is used.  Images/documents are
built in memory; the real torchvision model is constructed with
``weights=None, weights_backbone=None``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from model import contract, data, emit, infer, model as detector, partial, train


def _config(**overrides):
    values = {
        "weights": "none",
        "seed": 20261108,
        "inference_min_size": 96,
        "inference_max_size": 96,
    }
    values.update(overrides)
    return detector.DetectorConfig(**values)


def _image(height=96, width=96):
    return torch.ones(3, height, width, dtype=torch.float32)


def _single_target(height=96, width=96, *, class_id=3, instance=0, box=(30, 30, 50, 60)):
    mask = torch.zeros(height, width, dtype=torch.bool)
    x1, y1, x2, y2 = [int(v) for v in box]
    mask[y1:y2, x1:x2] = True
    target = {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([class_id + 1], dtype=torch.int64),
        "masks": mask[None],
        "keys": [(class_id, instance)],
        "loss_domain": mask.clone(),
        "blank_negative": ~mask,
    }
    return target, mask


def _synthetic_doc(primitives, width=100, height=100):
    commands, args, lengths, semantic, instance = [], [], [], [], []
    for command, points, semantic_id, instance_id in primitives:
        flat = []
        for x, y in points:
            flat.extend((float(x), float(y)))
        commands.append(command)
        args.append(flat)
        lengths.append(0.0)
        semantic.append(semantic_id)
        instance.append(instance_id)
    return {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic,
        "instanceIds": instance,
        "width": width,
        "height": height,
    }


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _config()
        cls.model, cls.info = detector.build_model(cls.config)
        cls.model.transform.min_size = (96,)
        cls.model.transform.max_size = 96
        cls.model.training_transform = ((96,), 96)
        cls.model.eval()

    def test_build_is_36_class_pretrained_topology_with_frozen_bn(self):
        self.assertEqual(self.info["head"]["num_classes"], 36)
        self.assertEqual(int(self.model.roi_heads.box_predictor.cls_score.out_features), 36)
        self.assertEqual(self.info["mask_roi_pool_output_size"], 14)
        self.assertEqual(self.config.mask_roi_output_size, 28)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                self.assertFalse(module.training)

    def test_training_transform_drift_is_rejected(self):
        saved_min, saved_max = self.model.transform.min_size, self.model.transform.max_size
        try:
            self.model.transform.min_size = (1000,)
            self.model.transform.max_size = 1000
            with self.assertRaises(RuntimeError):
                detector.assert_training_transform(self.model, self.config)
        finally:
            self.model.transform.min_size = saved_min
            self.model.transform.max_size = saved_max
        detector.assert_training_transform(self.model, self.config)

    def test_class_mapping_uses_keys_not_dense_labels(self):
        target, _ = _single_target()
        prepared = detector.prepare_detector_targets(target, device=torch.device("cpu"))
        self.assertEqual(prepared["labels"].tolist(), [4])
        target["labels"] = torch.tensor([7], dtype=torch.int64)  # dense diagnostic noise
        prepared = detector.prepare_detector_targets(target, device=torch.device("cpu"))
        self.assertEqual(prepared["labels"].tolist(), [4])

    def test_two_images_two_positives_backward_and_bn_running_stats_frozen(self):
        before = {
            name: buffer.detach().clone()
            for name, buffer in self.model.named_buffers()
            if "running_" in name
        }
        image_a = _image()
        target_a, _ = _single_target(class_id=3, instance=0)
        image_b = _image()
        target_b, _ = _single_target(class_id=4, instance=1, box=(60, 20, 85, 50))
        prepared = [
            detector.prepare_detector_targets(target_a, device=torch.device("cpu")),
            detector.prepare_detector_targets(target_b, device=torch.device("cpu")),
        ]
        result = detector.training_losses(self.model, [image_a, image_b], prepared)
        self.assertGreaterEqual(result["counts"]["roi_positive"], 2)
        self.assertGreater(result["counts"]["mask_positive_cells"], 0)
        total = detector.total_loss(result["losses"])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(self.model.roi_heads.box_predictor.cls_score.weight.grad)
        for name, buffer in self.model.named_buffers():
            if name in before:
                self.assertTrue(torch.equal(buffer, before[name]), name)
        self.model.zero_grad(set_to_none=True)

    def test_empty_gt_forward_backward_is_safe(self):
        height = width = 96
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, height, width), dtype=torch.bool),
            "keys": [],
            "loss_domain": torch.ones(height, width, dtype=torch.bool),
            "blank_negative": torch.ones(height, width, dtype=torch.bool),
        }
        prepared = detector.prepare_detector_targets(target, device=torch.device("cpu"))
        result = detector.training_losses(self.model, [_image()], [prepared])
        self.assertEqual(result["counts"]["target_count"], 0)
        self.assertEqual(result["counts"]["mask_positive_cells"], 0)
        total = detector.total_loss(result["losses"])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.model.zero_grad(set_to_none=True)

    def test_negative_licensing_requires_every_touched_pixel(self):
        negative = torch.ones(1, 1, 10, 10, dtype=torch.bool)
        boxes = torch.tensor([[1.2, 1.2, 5.8, 5.8]], dtype=torch.float32)
        integral = partial.integral_image(negative[0, 0])
        self.assertTrue(bool(partial.licensed_negative_boxes(boxes, integral, 10, 10)[0]))
        negative[0, 0, 2, 2] = False
        integral = partial.integral_image(negative[0, 0])
        self.assertFalse(bool(partial.licensed_negative_boxes(boxes, integral, 10, 10)[0]))
        outside = torch.tensor([[-0.5, 1.0, 4.0, 4.0]], dtype=torch.float32)
        self.assertFalse(bool(partial.licensed_negative_boxes(outside, integral, 10, 10)[0]))

    def test_unknown_cell_is_outside_mask_bce_and_has_no_gradient(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 36, 7, 7, requires_grad=True)
        gt = torch.zeros(1, 8, 8)
        gt[0, 2:6, 2:6] = 1.0
        domain = torch.ones(8, 8, dtype=torch.bool)
        domain[3, 3] = False  # unknown pixel
        proposal = [torch.tensor([[1.0, 1.0, 7.0, 7.0]])]
        matched = [torch.tensor([0])]
        loss_a, counts_a = partial.mask_domain_bce(
            logits, [gt], [domain], proposal, matched, torch.tensor([1]), output_size=7
        )
        self.assertLess(counts_a["licensed_cells"], 49)
        loss_a.backward()
        grad_a = logits.grad.detach().clone()
        self.assertTrue(torch.isfinite(grad_a).all())
        # Change the GT only in the unknown cell: loss and gradients are identical.
        gt_changed = gt.clone()
        gt_changed[0, 3, 3] = 1.0
        logits_b = logits.detach().clone().requires_grad_(True)
        loss_b, counts_b = partial.mask_domain_bce(
            logits_b, [gt_changed], [domain], proposal, matched, torch.tensor([1]), output_size=7
        )
        loss_b.backward()
        self.assertTrue(torch.allclose(loss_a.detach(), loss_b.detach()))
        self.assertTrue(torch.allclose(grad_a, logits_b.grad))
        self.assertEqual(counts_a["licensed_cells"], counts_b["licensed_cells"])

    def test_mask_domain_bce_two_images_batch_columns(self):
        torch.manual_seed(11)
        logits = torch.randn(2, 36, 6, 6, requires_grad=True)
        gt_a = torch.zeros(1, 16, 16)
        gt_a[0, 2:9, 2:9] = 1.0
        gt_b = torch.zeros(1, 16, 16)
        gt_b[0, 8:14, 8:14] = 1.0
        domain = torch.ones(16, 16, dtype=torch.bool)
        proposals = [torch.tensor([[1.0, 1.0, 14.0, 14.0]]), torch.tensor([[2.0, 2.0, 13.0, 13.0]])]
        matched = [torch.tensor([0]), torch.tensor([0])]
        loss, counts = partial.mask_domain_bce(
            logits,
            [gt_a, gt_b],
            [domain, domain],
            proposals,
            matched,
            torch.tensor([1, 2]),
            output_size=6,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(counts["licensed_cells"], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mask_target_aligned_defaults_serialize_and_legacy_payload_loads(self):
        default = detector.DetectorConfig()
        self.assertIs(default.mask_target_aligned, True)
        self.assertEqual(default.mask_target_sampling_ratio, 2)
        legacy_payload = default.to_dict()
        legacy_payload.pop("mask_target_aligned")
        legacy_payload.pop("mask_target_sampling_ratio")
        legacy_payload.pop("mask_target_projection")
        restored = detector.DetectorConfig.from_dict(legacy_payload)
        self.assertIs(restored.mask_target_aligned, True)
        self.assertEqual(restored.mask_target_sampling_ratio, 2)
        self.assertEqual(restored.mask_target_projection, "legacy")

        variant = _config(
            mask_target_aligned=False,
            mask_target_sampling_ratio=-1,
            mask_target_projection="paste_consistent",
        )
        roundtrip = detector.DetectorConfig.from_dict(variant.to_dict())
        self.assertIs(roundtrip.mask_target_aligned, False)
        self.assertEqual(roundtrip.mask_target_sampling_ratio, -1)
        self.assertEqual(roundtrip.mask_target_projection, "paste_consistent")
        with self.assertRaises(ValueError):
            detector.DetectorConfig(mask_target_sampling_ratio=0)
        with self.assertRaises(ValueError):
            detector.DetectorConfig(mask_target_aligned="yes")

    def test_mask_domain_default_roi_align_matches_legacy_explicit_arguments(self):
        torch.manual_seed(23)
        gt_a = torch.zeros(1, 16, 16)
        gt_a[0, 3:10, 2:8] = 1.0
        gt_b = torch.zeros(1, 16, 16)
        gt_b[0, 6:13, 5:12] = 1.0  # overlaps gt_a
        domain = torch.ones(16, 16, dtype=torch.bool)
        proposals = [
            torch.tensor([[1.0, 1.0, 14.0, 14.0]]),
            torch.tensor([[2.0, 2.0, 15.0, 15.0]]),
        ]
        matched = [torch.tensor([0]), torch.tensor([0])]
        base_logits = torch.randn(2, 36, 6, 6)

        def run(**kwargs):
            logits = base_logits.clone().requires_grad_(True)
            loss, counts = partial.mask_domain_bce(
                logits,
                [gt_a, gt_b],
                [domain, domain],
                proposals,
                matched,
                torch.tensor([1, 2]),
                output_size=6,
                **kwargs,
            )
            loss.backward()
            return loss.detach(), logits.grad.detach(), counts

        loss_default, grad_default, counts_default = run()
        loss_legacy, grad_legacy, counts_legacy = run(
            aligned=True, sampling_ratio=2
        )
        self.assertTrue(torch.equal(loss_default, loss_legacy))
        self.assertTrue(torch.equal(grad_default, grad_legacy))
        self.assertEqual(counts_default["licensed_cells"], counts_legacy["licensed_cells"])
        self.assertEqual(counts_default["positive_cells"], counts_legacy["positive_cells"])
        self.assertGreater(counts_default["positive_cells"], 0)
        self.assertIs(counts_default["target_aligned"], True)
        self.assertEqual(counts_default["target_sampling_ratio"], 2)

        loss_variant, grad_variant, counts_variant = run(
            aligned=False, sampling_ratio=-1
        )
        self.assertTrue(torch.isfinite(loss_variant))
        self.assertTrue(torch.isfinite(grad_variant).all())
        self.assertIs(counts_variant["target_aligned"], False)
        self.assertEqual(counts_variant["target_sampling_ratio"], -1)

    def test_all_unknown_domain_variant_is_zero_with_zero_gradient(self):
        logits = torch.randn(1, 36, 5, 5, requires_grad=True)
        gt = torch.ones(1, 8, 8)
        domain = torch.zeros(8, 8, dtype=torch.bool)
        for aligned, ratio in ((True, 2), (False, -1)):
            loss, counts = partial.mask_domain_bce(
                logits,
                [gt],
                [domain],
                [torch.tensor([[1.0, 1.0, 7.0, 7.0]])],
                [torch.tensor([0])],
                torch.tensor([1]),
                output_size=5,
                aligned=aligned,
                sampling_ratio=ratio,
            )
            self.assertEqual(float(loss), 0.0)
            self.assertEqual(counts["licensed_cells"], 0)
            self.assertEqual(counts["positive_cells"], 0)
            self.assertIs(counts["target_aligned"], aligned)
            self.assertEqual(counts["target_sampling_ratio"], ratio)

    def test_paste_consistent_rois_match_independent_float32_formula(self):
        boxes = torch.tensor(
            [
                [10.2, 5.7, 50.9, 40.1],
                [0.0, 0.0, 1.0, 1.0],
                [100.5, 20.25, 101.5, 21.75],
                [-2.25, 3.5, 8.75, 12.25],
            ],
            dtype=torch.float32,
        )
        output_size = 28
        channels = output_size + 2
        rois = partial.paste_consistent_rois(boxes, output_size=output_size)
        expanded = partial.expand_boxes(boxes, channels / float(output_size))
        integer = expanded.to(torch.int64)
        for index in range(int(boxes.shape[0])):
            x0, y0, x2, y2 = [int(value) for value in integer[index].tolist()]
            width = x2 - x0 + 1
            height = y2 - y0 + 1
            self.assertGreaterEqual(width, 2)
            self.assertGreaterEqual(height, 2)
            width_f = torch.tensor(float(width), dtype=torch.float32)
            height_f = torch.tensor(float(height), dtype=torch.float32)
            channels_f = torch.tensor(float(channels), dtype=torch.float32)
            size_f = torch.tensor(float(output_size), dtype=torch.float32)
            start_x = torch.tensor(float(x0)) + width_f / channels_f - 0.5
            start_y = torch.tensor(float(y0)) + height_f / channels_f - 0.5
            extent_x = size_f * width_f / channels_f
            extent_y = size_f * height_f / channels_f
            expected = torch.stack(
                (start_x, start_y, start_x + extent_x, start_y + extent_y)
            )
            self.assertTrue(
                torch.equal(rois[index], expected),
                (index, rois[index].tolist(), expected.tolist()),
            )
            # ROI cell centres are exactly the installed paste cell centres.
            for cell in (0, 7, output_size - 1):
                roi_center = float(
                    rois[index, 0]
                    + (cell + 0.5)
                    * (rois[index, 2] - rois[index, 0])
                    / output_size
                )
                paste_center = x0 + (cell + 1.5) * width / channels - 0.5
                self.assertAlmostEqual(roi_center, paste_center, places=5)

    def test_zero_extent_box_is_invalid_under_actual_expand_operator(self):
        _rois, valid, widths, heights = partial._paste_consistent_rois(
            torch.tensor([[5.0, 5.0, 5.0, 5.0]], dtype=torch.float32),
            output_size=28,
        )
        self.assertFalse(bool(valid[0]))
        self.assertEqual(int(widths[0]), 1)
        self.assertEqual(int(heights[0]), 1)
        _rois, valid, widths, heights = partial._paste_consistent_rois(
            torch.tensor([[-0.6, -0.6, 0.4, 0.4]], dtype=torch.float32),
            output_size=28,
        )
        self.assertFalse(bool(valid[0]))
        self.assertEqual(int(widths[0]), 1)
        self.assertEqual(int(heights[0]), 1)

    def test_paste_consistent_mixed_batch_excludes_only_narrow_rows(self):
        torch.manual_seed(31)
        logits = torch.zeros(4, 36, 28, 28, requires_grad=True)
        gt_a = torch.zeros(2, 24, 24)
        gt_a[0, 1:15, 1:12] = 1.0
        gt_a[1, 8:22, 10:23] = 1.0
        gt_b = torch.zeros(2, 24, 24)
        gt_b[0, 1:9, 1:9] = 1.0
        gt_b[1, 4:20, 3:18] = 1.0
        domain = torch.ones(24, 24, dtype=torch.bool)
        proposals = [
            torch.tensor(
                [[0.0, 0.0, 20.0, 20.0], [0.2, 0.2, 0.4, 0.4]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 40.0, 0.9]],
                dtype=torch.float32,
            ),
        ]
        matched = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        labels = torch.tensor([1, 2, 3, 4])
        loss, counts = partial.mask_domain_bce(
            logits,
            [gt_a, gt_b],
            [domain, domain],
            proposals,
            matched,
            labels,
            output_size=28,
            projection="paste_consistent",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(counts["original_positive_proposals"], 4)
        self.assertEqual(counts["excluded_narrow_proposals"], 2)
        self.assertEqual(counts["effective_supervised_proposals"], 2)
        self.assertGreater(counts["licensed_cells"], 0)
        loss.backward()
        grad = logits.grad
        self.assertTrue(torch.isfinite(grad).all())
        # Rows 1 and 3 are the excluded narrow boxes, in the exact original
        # proposal order.  Their gradient is exactly zero.
        self.assertTrue(torch.equal(grad[1], torch.zeros_like(grad[1])))
        self.assertTrue(torch.equal(grad[3], torch.zeros_like(grad[3])))
        self.assertGreater(float(grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(grad[2, 3].abs().sum()), 0.0)
        self.assertEqual(float(grad[0, 2:].abs().sum()), 0.0)
        self.assertEqual(float(grad[2, :3].abs().sum()), 0.0)

    def test_paste_consistent_all_invalid_is_graph_connected_zero(self):
        torch.manual_seed(37)
        logits = torch.randn(2, 36, 28, 28, requires_grad=True)
        domain = torch.ones(24, 24, dtype=torch.bool)
        gt = torch.ones(2, 24, 24)
        proposals = [
            torch.tensor(
                [[0.2, 0.2, 0.4, 0.4], [0.0, 0.0, 40.0, 0.9]],
                dtype=torch.float32,
            )
        ]
        loss, counts = partial.mask_domain_bce(
            logits,
            [gt],
            [domain],
            proposals,
            [torch.tensor([0, 1])],
            torch.tensor([1, 2]),
            output_size=28,
            projection="paste_consistent",
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(loss.requires_grad, True)
        self.assertEqual(counts["original_positive_proposals"], 2)
        self.assertEqual(counts["excluded_narrow_proposals"], 2)
        self.assertEqual(counts["effective_supervised_proposals"], 0)
        self.assertEqual(counts["licensed_cells"], 0)
        self.assertEqual(counts["positive_cells"], 0)
        self.assertEqual(counts["soft_positive_mass"], 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_paste_consistent_degenerate_box_raises(self):
        with self.assertRaises(ValueError):
            partial.paste_consistent_rois(
                torch.tensor([[0.0, 0.0, 0.0, 4.0]], dtype=torch.float32),
                output_size=28,
            )
        with self.assertRaises(ValueError):
            partial.paste_consistent_rois(
                torch.tensor([[4.0, 4.0, 3.0, 8.0]], dtype=torch.float32),
                output_size=28,
            )

    def test_paste_consistent_projection_effective_info_and_overlap(self):
        torch.manual_seed(29)
        logits = torch.randn(2, 36, 7, 7, requires_grad=True)
        gt_a = torch.zeros(1, 24, 24)
        gt_a[0, 3:15, 4:13] = 1.0
        gt_b = torch.zeros(1, 24, 24)
        gt_b[0, 8:20, 7:18] = 1.0  # independent class, overlaps gt_a
        domain = torch.ones(24, 24, dtype=torch.bool)
        proposals = [
            torch.tensor([[1.25, 1.75, 22.5, 22.25]]),
            torch.tensor([[3.5, 4.25, 23.0, 21.5]]),
        ]
        matched = [torch.tensor([0]), torch.tensor([0])]
        loss, counts = partial.mask_domain_bce(
            logits,
            [gt_a, gt_b],
            [domain, domain],
            proposals,
            matched,
            torch.tensor([1, 2]),
            output_size=7,
            projection="paste_consistent",
            aligned=True,
            sampling_ratio=2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(counts["target_projection"], "paste_consistent")
        self.assertIs(counts["target_aligned"], False)
        self.assertEqual(counts["target_sampling_ratio"], -1)
        self.assertIs(counts["target_aligned_requested"], True)
        self.assertEqual(counts["target_sampling_ratio_requested"], 2)
        self.assertGreater(counts["positive_cells"], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        unknown_logits = torch.randn(1, 36, 5, 5, requires_grad=True)
        unknown_loss, unknown_counts = partial.mask_domain_bce(
            unknown_logits,
            [torch.ones(1, 8, 8)],
            [torch.zeros(8, 8, dtype=torch.bool)],
            [torch.tensor([[1.0, 1.0, 7.0, 7.0]])],
            [torch.tensor([0])],
            torch.tensor([1]),
            output_size=5,
            projection="paste_consistent",
        )
        self.assertEqual(float(unknown_loss.detach()), 0.0)
        self.assertEqual(unknown_counts["licensed_cells"], 0)
        self.assertEqual(unknown_counts["target_projection"], "paste_consistent")

    def test_variant_config_model_empty_and_overlap_forward_backward(self):
        variant = _config(mask_target_aligned=False, mask_target_sampling_ratio=-1)
        model, info = detector.build_model(variant)
        model.transform.min_size = (96,)
        model.transform.max_size = 96
        model.training_transform = ((96,), 96)
        model.eval()
        self.assertEqual(info["mask_target_aligned"], False)
        self.assertEqual(info["mask_target_sampling_ratio"], -1)
        self.assertIs(model.mask_target_aligned, False)
        self.assertEqual(model.mask_target_sampling_ratio, -1)

        empty = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, 96, 96), dtype=torch.bool),
            "keys": [],
            "loss_domain": torch.ones(96, 96, dtype=torch.bool),
            "blank_negative": torch.ones(96, 96, dtype=torch.bool),
        }
        prepared = detector.prepare_detector_targets(empty, device=torch.device("cpu"))
        result = detector.training_losses(model, [_image()], [prepared])
        self.assertEqual(result["counts"]["target_count"], 0)
        self.assertEqual(result["counts"]["mask_target_aligned"], False)
        self.assertEqual(result["counts"]["mask_target_sampling_ratio"], -1)
        total = detector.total_loss(result["losses"])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        model.zero_grad(set_to_none=True)

        target_a, _ = _single_target(class_id=3, instance=0, box=(15, 15, 65, 65))
        target_b, _ = _single_target(class_id=4, instance=1, box=(35, 30, 80, 75))
        prepared = [
            detector.prepare_detector_targets(target_a, device=torch.device("cpu")),
            detector.prepare_detector_targets(target_b, device=torch.device("cpu")),
        ]
        result = detector.training_losses(
            model, [_image(), _image()], prepared
        )
        self.assertGreaterEqual(result["counts"]["roi_positive"], 2)
        self.assertGreater(result["counts"]["mask_positive_cells"], 0)
        self.assertEqual(result["counts"]["mask_target_aligned"], False)
        self.assertEqual(result["counts"]["mask_target_sampling_ratio"], -1)
        total = detector.total_loss(result["losses"])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(
            torch.isfinite(model.roi_heads.box_predictor.cls_score.weight.grad).all()
        )
        model.zero_grad(set_to_none=True)


    def test_paste_consistent_config_model_empty_and_overlap_backward(self):
        config = _config(mask_target_projection="paste_consistent")
        model, info = detector.build_model(config)
        model.transform.min_size = (96,)
        model.transform.max_size = 96
        model.training_transform = ((96,), 96)
        model.eval()
        self.assertEqual(info["mask_target_projection"], "paste_consistent")
        self.assertEqual(model.mask_target_projection, "paste_consistent")

        empty = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, 96, 96), dtype=torch.bool),
            "keys": [],
            "loss_domain": torch.ones(96, 96, dtype=torch.bool),
            "blank_negative": torch.ones(96, 96, dtype=torch.bool),
        }
        prepared = detector.prepare_detector_targets(empty, device=torch.device("cpu"))
        result = detector.training_losses(model, [_image()], [prepared])
        self.assertEqual(result["counts"]["target_count"], 0)
        self.assertEqual(
            result["counts"]["mask_target_projection"], "paste_consistent"
        )
        self.assertEqual(result["counts"]["mask_target_aligned"], False)
        self.assertEqual(result["counts"]["mask_target_sampling_ratio"], -1)
        self.assertEqual(result["counts"]["mask_target_aligned_requested"], True)
        self.assertEqual(
            result["counts"]["mask_target_sampling_ratio_requested"], 2
        )
        total = detector.total_loss(result["losses"])
        total.backward()
        model.zero_grad(set_to_none=True)

        target_a, _ = _single_target(class_id=3, instance=0, box=(15, 15, 65, 65))
        target_b, _ = _single_target(class_id=4, instance=1, box=(40, 35, 85, 80))
        prepared = [
            detector.prepare_detector_targets(target_a, device=torch.device("cpu")),
            detector.prepare_detector_targets(target_b, device=torch.device("cpu")),
        ]
        result = detector.training_losses(model, [_image(), _image()], prepared)
        self.assertGreaterEqual(result["counts"]["roi_positive"], 2)
        self.assertGreater(result["counts"]["mask_positive_cells"], 0)
        self.assertEqual(
            result["counts"]["mask_target_projection"], "paste_consistent"
        )
        self.assertEqual(
            result["counts"]["mask_original_positive_proposals"],
            result["counts"]["mask_positive_proposals"],
        )
        self.assertEqual(result["counts"]["mask_excluded_narrow_proposals"], 0)
        self.assertEqual(
            result["counts"]["mask_effective_supervised_proposals"],
            result["counts"]["mask_positive_proposals"],
        )
        total = detector.total_loss(result["losses"])
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(
            torch.isfinite(model.roi_heads.box_predictor.cls_score.weight.grad).all()
        )
        model.zero_grad(set_to_none=True)


class EmissionTest(unittest.TestCase):
    def test_non_square_emit_uses_official_letterbox_affine(self):
        height, width = 60, 40
        first = np.zeros((height, width), dtype=bool)
        first[10:50, 12:18] = True
        second = np.zeros((height, width), dtype=bool)
        second[25:31, 5:35] = True
        result = emit.emit_instances(
            [first, second], [3, 4], [0.9, 0.8], height=height, width=width
        )
        self.assertEqual(len(result["instances"]), 2)
        self.assertEqual(result["document"]["width"], 100)
        self.assertEqual(result["document"]["height"], 100)
        # viewbox_transform(40, 60) -> scale 0.4, letterbox_y 10; pixel -> view
        # is (pixel - offset) / scale.  Boxes are source-run coordinate extents.
        boxes = result["document"]["boxes"]
        self.assertEqual([entry[4] for entry in boxes], [3, 4])
        self.assertAlmostEqual(boxes[0][0], 12 / 0.4, places=6)
        self.assertAlmostEqual(boxes[0][1], 0.0, places=6)
        self.assertAlmostEqual(boxes[0][2], 17 / 0.4, places=6)
        self.assertAlmostEqual(boxes[0][3], (49 - 10) / 0.4, places=6)
        self.assertAlmostEqual(boxes[1][0], 5 / 0.4, places=6)
        self.assertAlmostEqual(boxes[1][1], (25 - 10) / 0.4, places=6)
        self.assertAlmostEqual(boxes[1][2], 34 / 0.4, places=6)
        self.assertAlmostEqual(boxes[1][3], (30 - 10) / 0.4, places=6)
        roundtrip = emit.roundtrip_iou(
            result["expected_masks"], result["document"], height=height, width=width
        )
        self.assertFalse(roundtrip["native_inverse_exact"])
        self.assertEqual(roundtrip["expected_keys"], 2)

    def test_square_overlap_roundtrip_preserves_both_masks(self):
        height = width = 40
        first = np.zeros((height, width), dtype=bool)
        first[5:35, 15:18] = True
        second = np.zeros((height, width), dtype=bool)
        second[18:21, 5:35] = True
        result = emit.emit_instances(
            [first, second], [3, 4], [0.9, 0.8], height=height, width=width
        )
        self.assertTrue(bool((first & second).any()))
        roundtrip = emit.roundtrip_iou(
            result["expected_masks"], result["document"], height=height, width=width
        )
        self.assertTrue(roundtrip["native_inverse_exact"])
        self.assertEqual(roundtrip["expected_keys"], 2)
        self.assertAlmostEqual(roundtrip["min_iou"], 1.0, places=6)
        self.assertEqual(roundtrip["total_lost_pixels"], 0)
        self.assertEqual(roundtrip["total_added_pixels"], 0)
        self.assertGreater(result["diagnostics"]["dense_diagnostic"]["conflict_pixels"], 0)

    def test_shifted_equal_count_is_not_iou_one(self):
        height, width = 40, 40
        source = np.zeros((height, width), dtype=bool)
        source[10:30, 10:14] = True
        result = emit.emit_instances([source], [3], [0.9], height=height, width=width)
        shifted = np.zeros_like(source)
        shifted[10:30, 11:15] = True
        roundtrip = emit.roundtrip_iou(
            [((3, 0), shifted)], result["document"], height=height, width=width
        )
        self.assertLess(roundtrip["min_iou"], 1.0)
        self.assertGreater(roundtrip["total_lost_pixels"], 0)
        self.assertGreater(roundtrip["total_added_pixels"], 0)

    def test_stuff_merges_per_class_with_instance_minus_one(self):
        height, width = 30, 30
        masks = [np.zeros((height, width), dtype=bool) for _ in range(3)]
        masks[0][2:8, 2:8] = True
        masks[1][10:16, 10:16] = True
        masks[2][2:8, 20:26] = True
        result = emit.emit_instances(
            masks, [30, 30, 31], [0.9, 0.8, 0.7], height=height, width=width
        )
        stuff = [item for item in result["instances"] if item.class_id == 30]
        self.assertEqual(len(stuff), 1)
        self.assertEqual(stuff[0].instance_id, -1)
        self.assertEqual(stuff[0].pixels, 72)


class TargetsAndInferenceTest(unittest.TestCase):
    def test_sibling_targets_build_crop_prepare(self):
        height = width = 64
        image = np.ones((3, height, width), dtype=np.float32)
        document = _synthetic_doc(
            [
                (0, [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (30.0, 20.0)], 3, 0),
                (0, [(40.0, 40.0), (50.0, 55.0), (45.0, 58.0), (40.0, 55.0)], 7, 2),
            ]
        )
        targets = contract.load_targets()
        target = targets.build_targets(image, document)
        self.assertEqual(len(target["keys"]), 2)
        cropped = targets.crop_targets(image, target, origin=(8, 8), size=48)
        prepared = detector.prepare_detector_targets(cropped, device=torch.device("cpu"))
        self.assertEqual(prepared["labels"].tolist(), [4, 8])
        self.assertEqual(prepared["masks"].shape, (2, 48, 48))
        self.assertEqual(prepared["boxes"].shape, (2, 4))
        self.assertTrue(set(prepared["loss_domain"].unique().tolist()).issubset({False, True}))
        self.assertFalse(bool((prepared["blank_negative"] & prepared["masks"].any(0)).any()))

    def test_crop_for_update_keys_reach_prepare_and_real_loss(self):
        """Regression for guard01: the real crop adapter must keep keys.

        The in-memory source exercises ``data.crop_for_update`` itself (not the
        sibling ``crop_targets`` shortcut), including an overlapping pair and an
        empty record.  The same prepared target is then consumed by the real
        weights-none torchvision model forward/backward.
        """

        class MemoryFitSource:
            def __init__(self, entries):
                self.targets = contract.load_targets()
                self.records = tuple(
                    data.FitRecord(
                        record_index=i,
                        sample_id="memory:%d" % i,
                        group_id="memory-group-%d" % i,
                        split="train_final",
                        source_role="fit",
                        png_name="memory-%d.png" % i,
                        svg_name="memory-%d.svg" % i,
                        png_digest="",
                    )
                    for i in range(len(entries))
                )
                self._entries = tuple(entries)

            def get(self, index):
                return self._entries[index]

        targets = contract.load_targets()
        image = np.ones((3, 128, 128), dtype=np.float32)
        document = _synthetic_doc(
            [
                (0, [(10.0, 10.0), (32.0, 10.0), (32.0, 15.0), (10.0, 15.0)], 3, 0),
                (0, [(27.0, 5.0), (32.0, 5.0), (32.0, 38.0), (27.0, 38.0)], 7, 2),
            ]
        )
        image_target = targets.build_targets(image, document)
        empty_image = np.ones((3, 96, 96), dtype=np.float32)
        empty_target = targets.build_targets(empty_image, _synthetic_doc([]))
        source = MemoryFitSource(
            [({"image": image}, image_target), ({"image": empty_image}, empty_target)]
        )

        crop = data.crop_for_update(
            source, 0, seed=20261108, update=0, jitter=0, size=96
        )
        target = crop["target"]
        self.assertEqual(list(target["keys"]), [(3, 0), (7, 2)])
        self.assertEqual(target["labels"].tolist(), [4, 8])
        self.assertEqual(target["boxes"].shape, (2, 4))
        self.assertEqual(target["masks"].shape, (2, 96, 96))
        self.assertTrue(bool((target["masks"][0] & target["masks"][1]).any()))

        prepared = detector.prepare_detector_targets(target, device=torch.device("cpu"))
        self.assertEqual(prepared["labels"].tolist(), [4, 8])
        self.assertEqual(prepared["boxes"].shape, (2, 4))

        config = _config()
        model, _ = detector.build_model(config)
        model.transform.min_size = (96,)
        model.transform.max_size = 96
        model.training_transform = ((96,), 96)
        result = detector.training_losses(model, [crop["image"]], [prepared])
        self.assertTrue(torch.isfinite(detector.total_loss(result["losses"])))
        detector.total_loss(result["losses"]).backward()
        model.zero_grad(set_to_none=True)

        empty_crop = data.crop_for_update(
            source, 1, seed=20261108, update=0, jitter=0, size=96
        )
        self.assertEqual(empty_crop["target"]["keys"], ())
        self.assertEqual(empty_crop["target"]["labels"].numel(), 0)
        self.assertEqual(empty_crop["target"]["boxes"].shape, (0, 4))
        empty_prepared = detector.prepare_detector_targets(
            empty_crop["target"], device=torch.device("cpu")
        )
        empty_result = detector.training_losses(model, [empty_crop["image"]], [empty_prepared])
        self.assertEqual(empty_result["counts"]["target_count"], 0)
        self.assertTrue(torch.isfinite(detector.total_loss(empty_result["losses"])))
        detector.total_loss(empty_result["losses"]).backward()
        model.zero_grad(set_to_none=True)

    def test_run_train_actual_crop_adapter_seam(self):
        """Exercise ``train.run_train`` itself, not only the guard shortcut."""

        class MemoryFitSource:
            def __init__(self, entries):
                self.targets = contract.load_targets()
                self.records = tuple(
                    data.FitRecord(
                        record_index=i,
                        sample_id="train-memory:%d" % i,
                        group_id="train-memory-group-%d" % i,
                        split="train_final",
                        source_role="fit",
                        png_name="train-memory-%d.png" % i,
                        svg_name="train-memory-%d.svg" % i,
                        png_digest="",
                    )
                    for i in range(len(entries))
                )
                self._entries = tuple(entries)
                self.manifest_sha256 = "memory-manifest"
                self.records_sha256 = "memory-records"

            def __len__(self):
                return len(self._entries)

            def get(self, index):
                return self._entries[index]

        targets = contract.load_targets()
        image = np.ones((3, 384, 384), dtype=np.float32)
        document = _synthetic_doc(
            [
                (0, [(10.0, 10.0), (42.0, 10.0), (42.0, 15.0), (10.0, 15.0)], 3, 0),
                (0, [(34.0, 5.0), (39.0, 5.0), (39.0, 55.0), (34.0, 55.0)], 7, 2),
            ]
        )
        image_target = targets.build_targets(image, document)
        empty_image = np.ones((3, 384, 384), dtype=np.float32)
        empty_target = targets.build_targets(empty_image, _synthetic_doc([]))
        source = MemoryFitSource(
            [({"image": image}, image_target), ({"image": empty_image}, empty_target)]
        )

        from model import train as trainer

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            args = argparse.Namespace(
                updates=2,
                accumulate=1,
                weights="none",
                allow_download=False,
                weights_file=None,
                manifest=str(temp_path / "unused-manifest.json"),
                output_dir=str(temp_path / "out"),
                crops=2,
                cache_size=1,
                seed=20261108,
                jitter=0,
                save_every=0,
                log_every=1,
                device="cpu",
                mask_target_mode="soft",
            )
            with mock.patch.object(trainer, "_crop_source", lambda *a, **k: source):
                self.assertEqual(trainer.run_train(args), 0)
            log_path = temp_path / "out" / "training_log.jsonl"
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["counts"]["target_count"], 2)
            self.assertEqual(rows[1]["counts"]["target_count"], 0)
            self.assertTrue((temp_path / "out" / "checkpoint.pt").is_file())
            report = json.loads((temp_path / "out" / "train_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["selected_records"], 2)

    def test_checkpoint_load_infer_and_no_gt(self):
        config = _config()
        model, info = detector.build_model(config)
        model.transform.min_size = (96,)
        model.transform.max_size = 96
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            checkpoint = temp / "checkpoint.pt"
            payload = {
                "schema": "cadvec.direct-instance-maskrcnn/1",
                "config": config.to_dict(),
                "model_info": info,
                "weights_source": "none",
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            torch.save(payload, checkpoint)
            png = temp / "synthetic.png"
            array = np.full((96, 96, 3), 255, dtype=np.uint8)
            array[20:70, 40:44] = 0
            Image.fromarray(array, mode="RGB").save(png)
            prediction = infer.predict_document(
                checkpoint,
                png,
                device_name="cpu",
                score_threshold=0.0,
                max_detections=3,
                compute_roundtrip=True,
            )
            self.assertIn("document", prediction)
            self.assertEqual(prediction["document"]["width"], 100)
            self.assertEqual(prediction["document"]["height"], 100)
            self.assertTrue(prediction["diagnostics"]["no_ground_truth_read"])
            self.assertFalse(prediction["diagnostics"]["canonical_1000px_scoring_measured"])
            if prediction["instances"]:
                self.assertTrue(all(item["class_id"] <= 34 for item in prediction["instances"]))
            if prediction["roundtrip"] is not None:
                self.assertLessEqual(prediction["roundtrip"]["min_iou"], 1.0 + 1e-6)

    def test_checkpoint_legacy_missing_projection_and_new_paste_mode_load(self):
        config = _config()
        model, info = detector.build_model(config)
        state = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        legacy_config = config.to_dict()
        legacy_config.pop("mask_target_projection")
        paste_config = config.to_dict()
        paste_config["mask_target_projection"] = "paste_consistent"
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            legacy_path = temp / "legacy.pt"
            paste_path = temp / "paste.pt"
            common = {
                "schema": "cadvec.direct-instance-maskrcnn/1",
                "model_info": info,
                "weights_source": "none",
                "model_state": state,
            }
            torch.save(dict(common, config=legacy_config), legacy_path)
            torch.save(dict(common, config=paste_config), paste_path)

            legacy_model, legacy_loaded, _, _ = infer.load_checkpoint(
                legacy_path, device=torch.device("cpu")
            )
            self.assertEqual(
                legacy_loaded.mask_target_projection, "legacy"
            )
            self.assertEqual(legacy_model.mask_target_projection, "legacy")
            self.assertFalse(legacy_model.training)

            paste_model, paste_loaded, _, _ = infer.load_checkpoint(
                paste_path, device=torch.device("cpu")
            )
            self.assertEqual(
                paste_loaded.mask_target_projection, "paste_consistent"
            )
            self.assertEqual(paste_model.mask_target_projection, "paste_consistent")
            self.assertFalse(paste_model.training)


class CliContractTest(unittest.TestCase):
    def test_train_cli_target_projection_defaults_and_variant(self):
        parser = train.build_parser()
        default = parser.parse_args(
            ["train", "--output-dir", "unused"]
        )
        self.assertEqual(default.mask_target_projection, "legacy")
        self.assertEqual(default.mask_target_aligned, True)
        self.assertEqual(default.mask_target_sampling_ratio, 2)
        variant = parser.parse_args(
            [
                "train",
                "--output-dir",
                "unused",
                "--mask-target-projection",
                "paste_consistent",
                "--no-mask-target-aligned",
                "--mask-target-sampling-ratio",
                "-1",
            ]
        )
        self.assertEqual(variant.mask_target_projection, "paste_consistent")
        self.assertEqual(variant.mask_target_aligned, False)
        self.assertEqual(variant.mask_target_sampling_ratio, -1)

    def test_capacity_list_length_and_refusal(self):
        self.assertEqual(len(data.CANONICAL_FIT16), 16)
        self.assertEqual(len(set(data.CANONICAL_FIT16)), 16)
        self.assertTrue(all(value.startswith("train_final:") for value in data.CANONICAL_FIT16))

    def test_output_dir_refuses_existing_checkpoint(self):
        from model import train as trainer

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "checkpoint.pt").write_bytes(b"existing")
            with self.assertRaises(trainer.TrainError):
                trainer._prepare_output_dir(path, files=("checkpoint.pt",))


if __name__ == "__main__":
    unittest.main()
