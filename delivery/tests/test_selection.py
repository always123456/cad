"""Selection arithmetic on synthetic score / mask arrays. No torch, no PNG, no checkpoint."""

from __future__ import annotations

import unittest

import numpy as np

from tests.support import delivery


def block(height=12, width=12, x0=0, y0=0, x1=4, y1=4):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def reference_select(masks, class_ids, scores, score_threshold, nms_iou, max_detections, per_image_cap=None):
    """Independent O(n^2) reference: full-mask AND, no bbox restriction."""

    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    areas = [int(mask.sum()) for mask in masks]
    keep = []
    for index, mask in enumerate(masks):
        drop = False
        for other in keep:
            if class_ids[other] != class_ids[index]:
                continue
            intersection = int(np.logical_and(mask, masks[other]).sum())
            union = areas[index] + areas[other] - intersection
            if union <= 0:
                continue
            if intersection / union > nms_iou:
                drop = True
                break
        if not drop:
            keep.append(index)
    above = [i for i in keep if scores[i] > score_threshold]
    capped = above[:max_detections]
    chosen = capped
    if per_image_cap is not None and len(capped) > per_image_cap:
        best = sorted(capped, key=lambda i: (-scores[i], i))[:per_image_cap]
        chosen = [i for i in capped if i in set(best)]
    return chosen


class SelectionArithmeticTest(unittest.TestCase):
    def select(self, masks, labels, scores, **policy):
        base = dict(score_threshold=0.5, nms_iou=0.5, max_detections=300, per_image_cap=None)
        base.update(policy)
        return delivery.select_instances(masks, labels, scores, **base)

    # -- NMS ---------------------------------------------------------------- #
    def test_same_class_duplicate_is_suppressed(self):
        masks = [block(), block()]
        result = self.select(masks, [1, 1], [0.9, 0.8])
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["after_nms"], 1)
        self.assertEqual(result["counts"]["nms_suppressed"], 1)

    def test_different_class_overlap_is_kept(self):
        result = self.select([block(), block()], [1, 2], [0.9, 0.8])
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["nms_suppressed"], 0)

    def test_iou_equal_to_threshold_is_kept(self):
        big = block(x0=0, y0=0, x1=2, y1=2)  # 4 px
        half = block(x0=0, y0=0, x1=2, y1=1)  # 2 px inside big, iou = 2/4
        result = self.select([big, half], [1, 1], [0.9, 0.8], nms_iou=0.5)
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["nms_suppressed"], 0)

    def test_iou_above_threshold_is_suppressed(self):
        big = block(x0=0, y0=0, x1=2, y1=2)  # 4 px
        three = big.copy()
        three[1, 1] = False  # 3 px, iou = 3/4
        result = self.select([big, three], [1, 1], [0.9, 0.8], nms_iou=0.5)
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["nms_suppressed"], 1)

    def test_iou_above_threshold_kept_when_nms_raised(self):
        big = block(x0=0, y0=0, x1=2, y1=2)
        three = big.copy()
        three[1, 1] = False
        result = self.select([big, three], [1, 1], [0.9, 0.8], nms_iou=0.75)
        self.assertEqual(result["indices"], [0, 1])

    def test_nms_keeps_first_in_decode_order(self):
        # The later, higher-scoring duplicate is dropped: decode order decides.
        result = self.select([block(), block()], [3, 3], [0.6, 0.99])
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["lowest_selected_score"], 0.6)

    def test_nms_iou_one_disables_suppression(self):
        result = self.select([block(), block()], [1, 1], [0.9, 0.8], nms_iou=1.0)
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["nms_suppressed"], 0)

    def test_empty_mask_neither_suppresses_nor_is_suppressed(self):
        empty = np.zeros((12, 12), dtype=bool)
        result = self.select([empty, block()], [1, 1], [0.9, 0.8])
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["nms_suppressed"], 0)

    def test_two_empty_masks_do_not_divide_by_zero(self):
        empty = np.zeros((12, 12), dtype=bool)
        result = self.select([empty, empty.copy()], [1, 1], [0.9, 0.8])
        self.assertEqual(result["indices"], [0, 1])

    # -- threshold ---------------------------------------------------------- #
    def test_threshold_is_strict(self):
        result = self.select([block(), block(x0=6, y0=6, x1=10, y1=10)], [1, 1], [0.5, 0.5000001])
        self.assertEqual(result["indices"], [1])
        self.assertEqual(result["counts"]["threshold_dropped"], 1)

    def test_threshold_keeps_everything_above(self):
        masks = [block(x0=0, y0=0, x1=2, y1=2), block(x0=4, y0=4, x1=6, y1=6)]
        result = self.select(masks, [1, 1], [0.51, 0.99], score_threshold=0.5)
        self.assertEqual(result["indices"], [0, 1])

    # -- caps --------------------------------------------------------------- #
    def test_global_cap_keeps_decode_order_not_score_order(self):
        masks = [
            block(x0=0, y0=0, x1=2, y1=2),
            block(x0=4, y0=0, x1=6, y1=2),
            block(x0=8, y0=0, x1=10, y1=2),
        ]
        result = self.select(masks, [1, 1, 1], [0.9, 0.6, 0.8], max_detections=2)
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["counts"]["cap_dropped"], 1)

    def test_threshold_runs_before_the_global_cap(self):
        masks = [
            block(x0=0, y0=0, x1=2, y1=2),
            block(x0=4, y0=0, x1=6, y1=2),
            block(x0=8, y0=0, x1=10, y1=2),
        ]
        result = self.select(masks, [1, 1, 1], [0.3, 0.9, 0.8], max_detections=2)
        self.assertEqual(result["indices"], [1, 2])
        self.assertEqual(result["counts"]["threshold_dropped"], 1)
        self.assertEqual(result["counts"]["cap_dropped"], 0)

    def test_per_image_cap_keeps_top_n_by_score(self):
        masks = [
            block(x0=0, y0=0, x1=2, y1=2),
            block(x0=4, y0=0, x1=6, y1=2),
            block(x0=8, y0=0, x1=10, y1=2),
        ]
        result = self.select(masks, [1, 1, 1], [0.9, 0.6, 0.8], per_image_cap=2)
        self.assertEqual(result["indices"], [0, 2])
        self.assertEqual(result["counts"]["per_image_cap_dropped"], 1)
        self.assertEqual(result["lowest_selected_score"], 0.8)

    def test_per_image_cap_ties_keep_decode_order(self):
        masks = [
            block(x0=0, y0=0, x1=2, y1=2),
            block(x0=4, y0=0, x1=6, y1=2),
            block(x0=8, y0=0, x1=10, y1=2),
        ]
        result = self.select(masks, [1, 1, 1], [0.9, 0.8, 0.8], per_image_cap=2)
        self.assertEqual(result["indices"], [0, 1])

    def test_per_image_cap_larger_than_candidates_is_a_noop(self):
        result = self.select([block()], [1], [0.9], per_image_cap=5)
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["per_image_cap_dropped"], 0)

    def test_cap_order_is_global_then_per_image(self):
        masks = [
            block(x0=0, y0=0, x1=2, y1=2),
            block(x0=4, y0=0, x1=6, y1=2),
            block(x0=8, y0=0, x1=10, y1=2),
        ]
        result = self.select(masks, [1, 1, 1], [0.9, 0.85, 0.8], max_detections=2, per_image_cap=1)
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["counts"]["cap_dropped"], 1)
        self.assertEqual(result["counts"]["per_image_cap_dropped"], 1)

    # -- degenerate --------------------------------------------------------- #
    def test_empty_input(self):
        result = self.select([], [], [])
        self.assertEqual(result["indices"], [])
        self.assertIsNone(result["lowest_selected_score"])
        self.assertEqual(
            result["counts"],
            {
                "candidates": 0,
                "after_nms": 0,
                "after_threshold": 0,
                "after_cap": 0,
                "after_per_image_cap": 0,
                "selected": 0,
                "nms_suppressed": 0,
                "threshold_dropped": 0,
                "cap_dropped": 0,
                "per_image_cap_dropped": 0,
            },
        )

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            self.select([block()], [1, 1], [0.9])

    def test_one_dimensional_mask_raises(self):
        with self.assertRaises(ValueError):
            self.select([np.zeros(5, dtype=bool)], [1], [0.9])

    def test_counts_are_internally_consistent(self):
        rng = np.random.default_rng(7)
        masks = []
        for _ in range(24):
            mask = np.zeros((16, 16), dtype=bool)
            y, x = rng.integers(0, 10, size=2)
            mask[y : y + rng.integers(2, 6), x : x + rng.integers(2, 6)] = True
            masks.append(mask)
        labels = [int(value) for value in rng.integers(1, 4, size=len(masks))]
        scores = [float(value) for value in rng.random(len(masks))]
        result = self.select(masks, labels, scores, score_threshold=0.3, max_detections=10, per_image_cap=5)
        counts = result["counts"]
        self.assertEqual(counts["candidates"], counts["after_nms"] + counts["nms_suppressed"])
        self.assertEqual(counts["after_nms"], counts["after_threshold"] + counts["threshold_dropped"])
        self.assertEqual(counts["after_threshold"], counts["after_cap"] + counts["cap_dropped"])
        self.assertEqual(counts["after_cap"], counts["selected"] + counts["per_image_cap_dropped"])
        self.assertEqual(counts["selected"], len(result["indices"]))


class SelectionEquivalenceTest(unittest.TestCase):
    """The bbox-restricted AND must reproduce a full-mask reference exactly."""

    def test_roi_intersection_equals_full_and(self):
        rng = np.random.default_rng(11)
        for _ in range(60):
            a = np.zeros((20, 20), dtype=bool)
            b = np.zeros((20, 20), dtype=bool)
            a[rng.integers(0, 12) : rng.integers(13, 20), rng.integers(0, 12) : rng.integers(13, 20)] = True
            b[rng.integers(0, 12) : rng.integers(13, 20), rng.integers(0, 12) : rng.integers(13, 20)] = True
            span_a = delivery._mask_span(a)
            span_b = delivery._mask_span(b)
            if span_a is None or span_b is None:
                continue
            self.assertEqual(
                delivery._roi_intersection(a, span_a, b, span_b),
                int(np.logical_and(a, b).sum()),
            )

    def test_selection_matches_full_mask_reference(self):
        rng = np.random.default_rng(20261108)
        for trial in range(20):
            count = int(rng.integers(0, 14))
            masks = []
            for _ in range(count):
                mask = np.zeros((18, 18), dtype=bool)
                y, x = int(rng.integers(0, 12)), int(rng.integers(0, 12))
                mask[y : y + int(rng.integers(1, 7)), x : x + int(rng.integers(1, 7))] = True
                masks.append(mask)
            labels = [int(value) for value in rng.integers(1, 4, size=count)]
            scores = [round(float(value), 3) for value in rng.random(count)]
            score_threshold = float(rng.choice([0.2, 0.5, 0.9]))
            nms_iou = float(rng.choice([0.3, 0.5, 0.9]))
            max_detections = int(rng.choice([1, 3, 300]))
            per_image_cap = rng.choice([None, 1, 2])
            expected = reference_select(
                masks, labels, scores, score_threshold, nms_iou, max_detections, per_image_cap
            )
            actual = delivery.select_instances(
                masks,
                labels,
                scores,
                score_threshold=score_threshold,
                nms_iou=nms_iou,
                max_detections=max_detections,
                per_image_cap=None if per_image_cap is None else int(per_image_cap),
            )
            self.assertEqual(actual["indices"], expected, "trial %d" % trial)


if __name__ == "__main__":
    unittest.main()
