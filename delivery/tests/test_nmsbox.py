"""Box-selection rule (``nmsbox.select_by_box_nms``) and its CLI wiring.

The rule is the deliverable readout of the wide pool: raw score >= thr, greedy
same-class box IoU suppression (>= tau, inclusive) in score-descending order with
ties in decode order, then the caps; classes >= 30 are never kept.  These tests
pin the boundary behaviour the offline re-measurement depends on.
"""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

from tests.support import delivery, detections, fake_run, touch

nmsbox = delivery._nmsbox()


def select(boxes, labels, scores, *, thr=0.97, tau=0.10, cap=300, per_image_cap=None):
    return nmsbox.select_by_box_nms(
        boxes,
        labels,
        scores,
        score_threshold=thr,
        box_iou=tau,
        max_detections=cap,
        per_image_cap=per_image_cap,
    )


def square(x0, y0, side, offset=0):
    """A square box, shifted right by ``offset`` pixels."""

    return [float(x0 + offset), float(y0), float(x0 + offset + side), float(y0 + side)]


class BoxIouMatrixTest(unittest.TestCase):
    def test_self_iou_is_one_and_matrix_symmetric(self):
        boxes = [[0, 0, 10, 10], [5, 0, 15, 10], [0, 0, 10, 5], [20, 20, 30, 30]]
        matrix = nmsbox.box_iou_matrix(boxes)
        self.assertEqual(matrix.shape, (4, 4))
        for index in range(4):
            self.assertAlmostEqual(matrix[index][index], 1.0)
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_half_overlap_is_one_third(self):
        # 100 + 100 - 50 intersection = 150 union
        matrix = nmsbox.box_iou_matrix([[0, 0, 10, 10], [5, 0, 15, 10]])
        self.assertAlmostEqual(matrix[0][1], 50.0 / 150.0)

    def test_disjoint_and_empty(self):
        matrix = nmsbox.box_iou_matrix([[0, 0, 10, 10], [20, 20, 30, 30]])
        self.assertEqual(matrix[0][1], 0.0)
        self.assertEqual(nmsbox.box_iou_matrix([]).shape, (0, 0))

    def test_degenerate_box_has_zero_iou(self):
        matrix = nmsbox.box_iou_matrix([[0, 0, 0, 0], [0, 0, 10, 10]])
        self.assertEqual(matrix[0][1], 0.0)


class SelectByBoxNmsTest(unittest.TestCase):
    def test_same_class_overlap_suppressed(self):
        result = select([square(0, 0, 10), square(0, 0, 10)], [1, 1], [0.99, 0.98])
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["nms_suppressed"], 1)

    def test_cross_class_overlap_kept(self):
        result = select([square(0, 0, 10), square(0, 0, 10)], [1, 2], [0.99, 0.98])
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["nms_suppressed"], 0)

    def test_iou_equal_to_tau_is_suppressed(self):
        # same-class boxes with IoU exactly 0.10: [0,10] against [0,1] (areas 100 / 10)
        boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 1.0]]
        self.assertAlmostEqual(nmsbox.box_iou_matrix(boxes)[0][1], 0.10)
        result = select(boxes, [1, 1], [0.99, 0.98], tau=0.10)
        self.assertEqual(result["indices"], [0])

    def test_iou_below_tau_is_kept(self):
        boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 0.9]]  # IoU 0.09 < 0.10
        result = select(boxes, [1, 1], [0.99, 0.98], tau=0.10)
        self.assertEqual(result["indices"], [0, 1])

    def test_score_threshold_is_inclusive(self):
        result = select([square(0, 0, 10), square(50, 0, 10)], [1, 1], [0.97, 0.9699], thr=0.97)
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["threshold_dropped"], 1)

    def test_greedy_suppression_follows_score_not_index(self):
        # index 0 is the weaker duplicate; index 1 outranks it and must survive.
        result = select([square(0, 0, 10), square(0, 0, 10)], [1, 1], [0.90, 0.96], thr=0.85)
        self.assertEqual(result["indices"], [1])

    def test_stuff_is_never_kept(self):
        boxes = [square(0, 0, 10), square(50, 0, 10), square(100, 0, 10), square(150, 0, 10)]
        labels = [30, 32, 34, 35]
        result = select(boxes, labels, [0.999, 0.999, 0.999, 0.999])
        self.assertEqual(result["indices"], [])
        self.assertEqual(result["counts"]["threshold_dropped"], 4)

    def test_stuff_does_not_suppress_a_thing(self):
        result = select([square(0, 0, 10), square(0, 0, 10)], [30, 1], [0.999, 0.99])
        self.assertEqual(result["indices"], [1])

    def test_per_image_cap_keeps_top_scores(self):
        boxes = [square(0, 0, 10, offset=20 * index) for index in range(45)]
        scores = [0.90 + 0.002 * index for index in range(45)]
        result = select(boxes, [1] * 45, scores, thr=0.80, per_image_cap=40)
        self.assertEqual(len(result["indices"]), 40)
        self.assertEqual(result["counts"]["after_per_image_cap"], 40)
        self.assertEqual(result["counts"]["per_image_cap_dropped"], 5)
        # the five weakest are the ones dropped
        dropped = {0, 1, 2, 3, 4}
        self.assertFalse(dropped & set(result["indices"]))
        self.assertAlmostEqual(result["lowest_selected_score"], scores[5])

    def test_per_image_cap_20_on_the_old_default(self):
        boxes = [square(0, 0, 10, offset=20 * index) for index in range(30)]
        result = select(boxes, [1] * 30, [0.99] * 30, cap=300, per_image_cap=20)
        self.assertEqual(len(result["indices"]), 20)
        self.assertEqual(result["counts"]["selected"], 20)

    def test_global_cap_applies_after_nms(self):
        # 30 non-overlapping candidates, cap 25: NMS first (nothing suppressed),
        # then the global cap, then the per-image cap.
        boxes = [square(0, 0, 10, offset=20 * index) for index in range(30)]
        scores = [0.99 - 0.001 * index for index in range(30)]
        result = select(boxes, [1] * 30, scores, thr=0.90, cap=25, per_image_cap=20)
        self.assertEqual(result["counts"]["after_nms"], 30)
        self.assertEqual(result["counts"]["after_cap"], 25)
        self.assertEqual(result["counts"]["after_per_image_cap"], 20)
        self.assertEqual(result["indices"], list(range(20)))

    def test_ties_keep_decode_order(self):
        boxes = [square(0, 0, 10, offset=20 * index) for index in range(3)]
        result = select(boxes, [1, 1, 1], [0.99, 0.99, 0.99])
        self.assertEqual(result["indices"], [0, 1, 2])

    def test_counts_are_consistent(self):
        boxes = [square(0, 0, 10), square(0, 0, 10), square(80, 0, 10)]
        result = select(boxes, [1, 1, 1], [0.99, 0.98, 0.10], thr=0.97)
        counts = result["counts"]
        self.assertEqual(counts["candidates"], 3)
        self.assertEqual(counts["after_threshold"], 2)
        self.assertEqual(counts["after_nms"], 1)
        self.assertEqual(counts["nms_suppressed"], 1)
        self.assertEqual(counts["threshold_dropped"], 1)
        self.assertEqual(counts["selected"], 1)
        self.assertEqual(result["indices"], [0])

    def test_empty_image(self):
        result = select([], [], [])
        self.assertEqual(result["indices"], [])
        self.assertIsNone(result["lowest_selected_score"])
        self.assertEqual(result["counts"]["selected"], 0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            select([square(0, 0, 10)], [1, 1], [0.99, 0.98])


class SelectInstancesDispatchTest(unittest.TestCase):
    def masks(self, count, size=32):
        mask = np.zeros((size, size), dtype=bool)
        mask[4:12, 4:12] = True
        return [mask for _ in range(count)]

    def test_plain_rule_ignores_boxes(self):
        masks = self.masks(2)
        plain = delivery.select_instances(
            masks, [1, 1], [0.99, 0.98], score_threshold=0.05, nms_iou=0.5, max_detections=300
        )
        with_boxes = delivery.select_instances(
            masks,
            [1, 1],
            [0.99, 0.98],
            score_threshold=0.05,
            nms_iou=0.5,
            max_detections=300,
            boxes=[[0, 0, 1, 1], [10, 10, 20, 20]],
        )
        self.assertEqual(plain, with_boxes)
        self.assertEqual(plain["indices"], [0])  # identical masks -> mask NMS

    def test_nmsbox_dispatches_to_the_box_rule(self):
        masks = self.masks(2)
        result = delivery.select_instances(
            masks,
            [1, 1],
            [0.99, 0.98],
            score_threshold=0.05,
            nms_iou=0.5,
            max_detections=300,
            boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
            select_rule="nmsbox",
            nms_box_iou=0.10,
            nms_box_thr=0.97,
        )
        # disjoint boxes: the mask rule would have suppressed, the box rule keeps both
        self.assertEqual(result["indices"], [0, 1])

    def test_nmsbox_without_boxes_is_refused(self):
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.select_instances(
                self.masks(1), [1], [0.99], score_threshold=0.05, nms_iou=0.5,
                max_detections=300, select_rule="nmsbox",
            )
        self.assertEqual(caught.exception.code, "box_nms_boxes")

    def test_unknown_rule_is_refused(self):
        with self.assertRaises(delivery.DeliveryRefused) as caught:
            delivery.select_instances(
                self.masks(1), [1], [0.99], score_threshold=0.05, nms_iou=0.5,
                max_detections=300, boxes=[[0, 0, 10, 10]], select_rule="maskbox",
            )
        self.assertEqual(caught.exception.code, "selection_policy")

    def test_policy_block_only_for_the_non_default_rule(self):
        plain = delivery.validate_selection()
        self.assertNotIn("box_nms", plain)
        self.assertEqual(plain["order"], delivery.SELECTION_ORDER)
        self.assertEqual(plain["signature"], [0.05, 0.5, 0.5, 300, None])

        block = delivery.validate_selection(select_rule="nmsbox")
        self.assertEqual(block["box_nms"]["rule"], "nmsbox")
        self.assertEqual(block["box_nms"]["box_iou"], 0.10)
        self.assertEqual(block["box_nms"]["box_threshold"], 0.97)
        self.assertEqual(block["box_nms"]["stuff_dropped_from"], 30)
        self.assertEqual(block["order"], block["box_nms"]["order"])
        # the five-number signature is untouched by the rule
        self.assertEqual(delivery.selection_signature(block), (0.05, 0.5, 0.5, 300, None))

    def test_rule_knobs_are_validated(self):
        block = delivery.validate_selection(select_rule="nmsbox", nms_box_iou=0.05, nms_box_thr=0.90)
        self.assertEqual(block["box_nms"]["box_iou"], 0.05)
        self.assertEqual(block["box_nms"]["box_threshold"], 0.90)
        for bad in (0.0, -0.1):
            with self.assertRaises(delivery.DeliveryRefused):
                delivery.validate_selection(select_rule="nmsbox", nms_box_iou=bad)
        with self.assertRaises(delivery.DeliveryRefused):
            delivery.validate_selection(select_rule="nmsbox", nms_box_thr=1.0)


class NmsBoxRunTest(unittest.TestCase):
    """The wiring: predict's boxes reach emit's selection stage, aligned with masks."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")
        self.checkpoint = touch(os.path.join(self.tmp.name, "weights.pt"), b"weights")
        self.png = touch(os.path.join(self.tmp.name, "drawing.png"), b"png")

    def predict(self, _context, _image):
        # label 31 -> canonical 30 (Stuff, dropped), then two same-class duplicates
        result = detections(
            (1, 0.99, (4, 12, 4, 12)),
            (31, 0.995, (16, 24, 16, 24)),
            (1, 0.98, (4, 12, 4, 12)),
        )
        result["boxes"] = np.array(
            [[4, 4, 12, 12], [16, 16, 24, 24], [4, 4, 12, 12]], dtype=np.float32
        )
        return result

    def test_boxes_survive_predict_into_the_selection_stage(self):
        summary = fake_run(
            self.out,
            self.checkpoint,
            png=self.png,
            predict=self.predict,
            drop_stuff=True,
            select_rule="nmsbox",
            nms_box_iou=0.10,
            nms_box_thr=0.97,
            per_image_cap=40,
        )
        self.assertEqual(summary["status"], "delivery_written")
        with open(os.path.join(self.out, "selection_report.json"), encoding="utf-8") as handle:
            report = json.load(handle)
        image = report["images"][0]
        # 3 decoded, the Stuff instance dropped, the duplicate suppressed, 1 kept
        self.assertEqual(report["policy"]["box_nms"]["rule"], "nmsbox")
        self.assertEqual(image["candidates"], 2)  # post-drop candidates
        self.assertEqual(image["after_threshold"], 2)
        self.assertEqual(image["after_nms"], 1)
        self.assertEqual(image["selected"], 1)
        self.assertEqual(image["emitted"], 1)
        self.assertEqual(image["stuff_drop"]["dropped"], 1)

        with open(os.path.join(self.out, "run_receipt.json"), encoding="utf-8") as handle:
            receipt = json.load(handle)
        self.assertEqual(receipt["guards"]["select_rule"], "nmsbox")
        self.assertEqual(receipt["selection_rule"]["box_iou"], 0.10)
        self.assertEqual(receipt["selection_rule"]["box_threshold"], 0.97)
        self.assertEqual(receipt["selection_signature"], [0.05, 0.5, 0.5, 300, 40])

    def test_plain_receipt_has_no_selection_rule_key(self):
        fake_run(self.out, self.checkpoint, png=self.png)
        with open(os.path.join(self.out, "run_receipt.json"), encoding="utf-8") as handle:
            receipt = json.load(handle)
        self.assertNotIn("selection_rule", receipt)
        self.assertNotIn("select_rule", receipt["guards"])


if __name__ == "__main__":
    unittest.main()
