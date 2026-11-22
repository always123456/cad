"""Same-class box IoU selection ("wide" readout) — numpy only, no lane imports.

This is the stage-2 rule of the wide-pool readout, re-measured offline on the 387
held-out records in feature space (``delivery/v2lib.py`` ``eval_keep``): the grid
optimum is ``nmsbox(score >= 0.97, tau 0.10, cap 40)`` = 0.166234 micro PQ
(plain 0.99/cap20 on the same pool 0.147478; 0.80/0.20/cap20 0.119681 — the wide
pool's 0.50..0.80 candidates flood that arm), canonical estimate ~0.1676:

    candidates (the delivery decode: mask threshold 0.40, score >= 0.50, <= 300)
      -> Things only (class < 30)
      -> score >= threshold (0.97)
      -> greedy in score-descending order (ties keep decode order): a kept box
         suppresses every *same-class* box with IoU >= tau (0.10, inclusive)
      -> global max-detections cap (survivor order, i.e. score-descending)
      -> per-image cap (top-N by score, ties keep decode order)

It deliberately shares nothing with ``exp3b_nms`` (that module imports the lost
``v2lib``) or with the deleted feature extractor: the whole rule is
``box_iou_matrix`` plus one greedy pass, so it can be unit-tested on synthetic
boxes and shipped as-is.

The mask-IoU stage of ``select_instances`` is *replaced* by this rule, not added
to it: the measurement never applied both.
"""

from __future__ import annotations

import numpy as np

# Canonical vocabulary: classes 30..34 are the Stuff tail, 35 is the ignore
# class.  A candidate at or above this id is never kept.
STUFF_CLASS_START = 30

BOX_SELECT_ORDER = (
    "candidates -> Things only (class < 30) -> score >= nms_box_thr (inclusive) "
    "-> greedy in score-descending order, ties keep decode order; a kept box suppresses "
    "same-class boxes with IoU >= nms_box_iou (inclusive) -> global max-detections cap "
    "(survivor order kept) -> per-image cap (top-N by score, ties keep decode order)"
)


def box_iou_matrix(boxes):
    """Pairwise IoU of ``xyxy`` boxes; shape (N, N), zero where the union is empty."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    count = boxes.shape[0]
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)

    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    top = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    right = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)

    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = widths * heights
    union = areas[:, None] + areas[None, :] - intersection

    iou = np.zeros_like(intersection)
    np.divide(intersection, union, out=iou, where=union > 0.0)
    return iou


def select_by_box_nms(
    boxes,
    class_ids,
    scores,
    *,
    score_threshold,
    box_iou,
    max_detections,
    per_image_cap=None,
):
    """Apply the wide readout to one image's decoded instances.

    ``boxes`` are ``xyxy`` native-pixel boxes in decode order, ``class_ids`` the
    canonical ids (0..34) and ``scores`` the raw detection confidences.  Returns
    the same shape as ``select_instances``: ``{"indices", "counts",
    "lowest_selected_score"}``, with ``indices`` in score-descending order (the
    order the greedy pass kept them in).
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = [int(value) for value in class_ids]
    values = [float(value) for value in scores]
    if not (len(boxes) == len(labels) == len(values)):
        raise ValueError(
            "box-selection input mismatch: %d boxes, %d class ids, %d scores"
            % (len(boxes), len(labels), len(values))
        )

    candidates = len(values)
    things = np.asarray([label < STUFF_CLASS_START for label in labels], dtype=bool)
    eligible = [
        int(index)
        for index in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
        if things[index] and values[index] >= score_threshold
    ]

    overlap = box_iou_matrix(boxes)
    same_class = np.asarray(labels, dtype=np.int64)[:, None] == np.asarray(labels, dtype=np.int64)[None, :]
    alive = np.ones(candidates, dtype=bool)
    kept = []
    for index in eligible:
        if not alive[index]:
            continue
        kept.append(index)
        alive[index] = False
        alive &= ~(same_class[index] & (overlap[index] >= box_iou))

    capped = kept[: int(max_detections)]
    per_image_dropped = 0
    selected = capped
    if per_image_cap is not None and len(capped) > per_image_cap:
        top = set(sorted(capped, key=lambda index: (-values[index], index))[: int(per_image_cap)])
        selected = [index for index in capped if index in top]
        per_image_dropped = len(capped) - len(selected)

    lowest = min((values[index] for index in selected), default=None)
    counts = {
        "candidates": candidates,
        "after_nms": len(kept),
        "after_threshold": len(eligible),
        "after_cap": len(capped),
        "after_per_image_cap": len(selected),
        "selected": len(selected),
        "nms_suppressed": len(eligible) - len(kept),
        "threshold_dropped": candidates - len(eligible),
        "cap_dropped": len(kept) - len(capped),
        "per_image_cap_dropped": per_image_dropped,
    }
    return {"indices": selected, "counts": counts, "lowest_selected_score": lowest}
