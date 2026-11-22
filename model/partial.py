"""Partial-label loss seams for the direct-instance Mask R-CNN route.

Pure-ish helpers kept separate from the detector build so the partial-label
restriction is auditable and unit-testable without a full model.

The only negative pixels ever licensed are those in
``target['blank_negative']`` (equivalently ``loss_domain & ~union(target masks)``).
An anchor or ROI box is a negative only when its whole pixel rectangle lies in
that map.  Boxes that touch ink of an unknown/skipped primitive, cross a known
positive, or leave the image are labelled -1 and removed by the standard
sampler before any loss sees them (torchvision ``fastrcnn_loss`` does not accept
-1 itself).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision.models.detection.roi_heads import expand_boxes
    from torchvision.ops import roi_align
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchvision roi_align/expand_boxes are required: %s" % exc
    ) from exc


def integral_image(mask: Any) -> np.ndarray:
    """Summed-area table with a zero border; shape (H+1, W+1)."""

    array = mask
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("integral_image expects HW")
    table = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = array.cumsum(axis=0).cumsum(axis=1)
    return table


def _clip_floor(value: float, upper: int) -> int:
    return max(0, min(int(upper), int(math.floor(value + 1e-6))))


def _clip_ceil(value: float, upper: int) -> int:
    return max(0, min(int(upper), int(math.ceil(value - 1e-6))))


def licensed_negative_boxes(
    boxes: torch.Tensor,
    integral: np.ndarray,
    width: int,
    height: int,
) -> torch.Tensor:
    """Boolean per box: fully inside the image and every touched pixel negative.

    ``integral`` is the summed-area table of the *negative* map, not of the raw
    image, so a licensed box is one whose integer rectangle is entirely inside
    ``blank_negative``.  The decisive test is therefore
    ``sum == rectangle_area`` (every pixel negative), never ``sum == 0``.
    Fractional boxes are expanded to every pixel they touch; boxes partially
    outside the image are rejected.
    """

    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    if tuple(integral.shape) != (height + 1, width + 1):
        raise ValueError("integral shape %r does not match %dx%d" % (integral.shape, width, height))
    raw = boxes.detach().cpu().numpy().astype(np.float64, copy=False)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("boxes must be Nx4")
    x1, y1, x2, y2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    ok = np.isfinite(raw).all(axis=1)
    ok &= (x2 > x1) & (y2 > y1)
    ok &= (x1 >= -1e-4) & (y1 >= -1e-4)
    ok &= (x2 <= width + 1e-4) & (y2 <= height + 1e-4)
    ix1 = np.array([_clip_floor(v, width) for v in x1], dtype=np.int64)
    iy1 = np.array([_clip_floor(v, height) for v in y1], dtype=np.int64)
    ix2 = np.array([_clip_ceil(v, width) for v in x2], dtype=np.int64)
    iy2 = np.array([_clip_ceil(v, height) for v in y2], dtype=np.int64)
    ok &= (ix2 > ix1) & (iy2 > iy1)
    safe_ix1 = np.clip(ix1, 0, width)
    safe_iy1 = np.clip(iy1, 0, height)
    safe_ix2 = np.clip(ix2, 0, width)
    safe_iy2 = np.clip(iy2, 0, height)
    area = (
        integral[safe_iy2, safe_ix2]
        - integral[safe_iy1, safe_ix2]
        - integral[safe_iy2, safe_ix1]
        + integral[safe_iy1, safe_ix1]
    )
    rectangle_area = (safe_ix2 - safe_ix1) * (safe_iy2 - safe_iy1)
    # Every pixel of the touched rectangle must be a licensed negative.
    ok &= area == rectangle_area
    return torch.from_numpy(ok).to(device=boxes.device, dtype=torch.bool)


def assign_partial_rpn_labels(
    rpn: Any,
    anchors: Sequence[torch.Tensor],
    gt_boxes: Sequence[torch.Tensor],
    negative_maps: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """RPN labels with only provably ink-free ``BELOW_LOW_THRESHOLD`` anchors as 0.

    Between-threshold anchors and unlicensed below-threshold anchors stay -1.
    Matched anchors (> low threshold) are positive regardless of ink, which is
    correct: their regression target is a known official box.
    """

    matcher = rpn.proposal_matcher
    labels_out: List[torch.Tensor] = []
    matched_out: List[torch.Tensor] = []
    for anchor, gt, negative in zip(anchors, gt_boxes, negative_maps):
        height, width = int(negative.shape[0]), int(negative.shape[1])
        integral = integral_image(negative)
        count = int(anchor.shape[0])
        labels = torch.full((count,), -1.0, dtype=torch.float32, device=anchor.device)
        if gt.numel() == 0:
            licensed = licensed_negative_boxes(anchor, integral, width, height)
            labels[licensed] = 0.0
            matched_out.append(torch.zeros((count, 4), dtype=anchor.dtype, device=anchor.device))
            labels_out.append(labels)
            continue
        ious = rpn.box_similarity(gt, anchor)
        matched = matcher(ious)
        labels[matched >= 0] = 1.0
        below = matched == int(matcher.BELOW_LOW_THRESHOLD)
        if bool(below.any()):
            licensed = licensed_negative_boxes(
                anchor[below], integral, width, height
            )
            below_indices = torch.where(below)[0]
            labels[below_indices[licensed]] = 0.0
        matched_out.append(gt[matched.clamp(min=0)])
        labels_out.append(labels)
    return labels_out, matched_out


def filter_roi_negatives(
    labels: Sequence[torch.Tensor],
    proposals: Sequence[torch.Tensor],
    negative_maps: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Turn unlicensed below-threshold ROI labels from 0 into -1.

    Input labels come from ``roi_heads.assign_targets_to_proposals`` where every
    below-low proposal is 0.  Only proposals whose full rectangle is in
    ``blank_negative`` remain 0; the rest are ignored by the sampler.
    """

    filtered: List[torch.Tensor] = []
    for label, proposal, negative in zip(labels, proposals, negative_maps):
        label = label.clone()
        height, width = int(negative.shape[0]), int(negative.shape[1])
        integral = integral_image(negative)
        negative_indices = torch.where(label == 0)[0]
        if negative_indices.numel():
            licensed = licensed_negative_boxes(
                proposal[negative_indices], integral, width, height
            )
            label[negative_indices[~licensed]] = -1.0
        filtered.append(label)
    return filtered


def _sample_indices(
    labels: torch.Tensor,
    batch_size_per_image: int,
    positive_fraction: float,
) -> torch.Tensor:
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    num_pos = min(int(positive.numel()), int(batch_size_per_image * positive_fraction))
    num_neg = min(int(negative.numel()), int(batch_size_per_image - num_pos))
    if num_pos:
        permutation = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        positive = positive[permutation]
    else:
        positive = positive[:0]
    if num_neg:
        permutation = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        negative = negative[permutation]
    else:
        negative = negative[:0]
    return torch.cat((positive, negative), dim=0)


def rpn_losses(
    objectness: torch.Tensor,
    pred_bbox_deltas: torch.Tensor,
    labels: Sequence[torch.Tensor],
    regression_targets: Sequence[torch.Tensor],
    *,
    batch_size_per_image: int = 256,
    positive_fraction: float = 0.5,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Standard objectness CE + smooth-L1, restricted to sampled licensed labels."""

    sampled: List[torch.Tensor] = []
    offset = 0
    counts = {
        "anchors": int(sum(int(value.numel()) for value in labels)),
        "positive": int(sum(int((value >= 1).sum().item()) for value in labels)),
        "negative": int(sum(int((value == 0).sum().item()) for value in labels)),
        "ignored": int(sum(int((value < 0).sum().item()) for value in labels)),
    }
    for value in labels:
        indices = _sample_indices(value, batch_size_per_image, positive_fraction)
        sampled.append(indices + offset)
        offset += int(value.numel())
    flat_indices = torch.cat(sampled, dim=0) if sampled else torch.empty(0, dtype=torch.long)
    flat_labels = torch.cat(labels, dim=0) if labels else torch.empty(0)
    counts["sampled"] = int(flat_indices.numel())

    objectness_flat = objectness.reshape(-1)
    deltas_flat = pred_bbox_deltas.reshape(-1, 4)
    regression_flat = torch.cat(regression_targets, dim=0)

    if flat_indices.numel() == 0:
        zero = objectness_flat.sum() * 0.0
        return (
            {"loss_objectness": zero, "loss_rpn_box_reg": deltas_flat.sum() * 0.0},
            counts,
        )

    sampled_labels = flat_labels[flat_indices].to(dtype=objectness_flat.dtype)
    loss_objectness = F.binary_cross_entropy_with_logits(
        objectness_flat[flat_indices], sampled_labels, reduction="mean"
    )
    positive_indices = flat_indices[sampled_labels >= 1]
    counts["sampled_positive"] = int(positive_indices.numel())
    counts["sampled_negative"] = int(flat_indices.numel() - positive_indices.numel())
    if positive_indices.numel():
        loss_box = (
            F.smooth_l1_loss(
                deltas_flat[positive_indices],
                regression_flat[positive_indices],
                beta=1 / 9,
                reduction="sum",
            )
            / flat_indices.numel()
        )
    else:
        loss_box = deltas_flat.sum() * 0.0
    return {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_box}, counts


def _paste_consistent_rois(
    boxes: torch.Tensor, *, output_size: int, padding: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched paste-consistent ROIs plus per-box validity and integer extents."""

    if boxes.ndim != 2 or int(boxes.shape[1]) != 4:
        raise ValueError("boxes must be Nx4")
    if int(output_size) < 1:
        raise ValueError("output_size must be positive")
    boxes32 = boxes.to(dtype=torch.float32)
    if int(boxes.shape[0]) == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
        integer_axis = torch.zeros((0,), dtype=torch.int64, device=boxes.device)
        return boxes32, empty, integer_axis, integer_axis
    if not bool(torch.isfinite(boxes32).all()):
        raise ValueError("boxes contain non-finite values")
    if bool((boxes32[:, 2] < boxes32[:, 0]).any()) or bool(
        (boxes32[:, 3] < boxes32[:, 1]).any()
    ):
        raise ValueError("boxes must have x2 >= x1 and y2 >= y1")
    channels = int(output_size) + 2 * int(padding)
    scale = channels / float(output_size)
    expanded = expand_boxes(boxes32, scale)
    integer = expanded.to(torch.int64)
    widths = integer[:, 2] - integer[:, 0] + 1
    heights = integer[:, 3] - integer[:, 1] + 1
    valid = (widths >= 2) & (heights >= 2)
    width_f = widths.to(dtype=torch.float32)
    height_f = heights.to(dtype=torch.float32)
    x1 = integer[:, 0].to(dtype=torch.float32) + width_f / channels - 0.5
    y1 = integer[:, 1].to(dtype=torch.float32) + height_f / channels - 0.5
    extent_x = float(output_size) * width_f / channels
    extent_y = float(output_size) * height_f / channels
    rois = torch.stack((x1, y1, x1 + extent_x, y1 + extent_y), dim=1)
    return rois, valid, widths, heights


def paste_consistent_rois(
    boxes: torch.Tensor, *, output_size: int, padding: int = 1
) -> torch.Tensor:
    """Return ROIAlign boxes whose cell centres invert the installed paste.

    This is the production batched form of the read-only diagnostic equation.
    It uses the installed torchvision ``expand_boxes`` float32 arithmetic and
    ``Tensor.to(torch.int64)`` truncation toward zero, not Python doubles, so
    values near integer boundaries match inference.  With pad ``p`` and mask
    size ``M`` the paste operator writes an integer extent
    ``W = int(expanded_x2) - int(expanded_x1) + 1`` into ``M + 2p`` cells.  An
    ``aligned=False`` ROI from ``a`` to ``a + M*W/(M+2p)``, with
    ``a = int(expanded_x1) + W/(M+2p) - 0.5``, has cell centres at
    ``int(expanded_x1) + (j+1.5)*W/(M+2p) - 0.5``, exactly the paste centres.

    ``W >= 2`` (equivalently extent ``>= 1`` for ``M=28``) is required by the
    installed ``roi_align(aligned=False)`` extent check.  Degenerate boxes raise
    instead of silently switching convention.
    """

    rois, valid, widths, heights = _paste_consistent_rois(
        boxes, output_size=output_size, padding=padding
    )
    if bool((~valid).any()):
        index = int(torch.nonzero(~valid, as_tuple=False)[0, 0].item())
        raise ValueError(
            "paste-consistent projection has transformed extent below 1 "
            "(box index %d, box %s, paste W=%d H=%d); extend the box"
            % (
                index,
                boxes.to(dtype=torch.float32)[index].detach().cpu().tolist(),
                int(widths[index]),
                int(heights[index]),
            )
        )
    return rois


def mask_domain_bce(
    mask_logits: torch.Tensor,
    gt_masks: Sequence[torch.Tensor],
    domain_maps: Sequence[torch.Tensor],
    proposals: Sequence[torch.Tensor],
    matched_indices: Sequence[torch.Tensor],
    positive_labels: torch.Tensor,
    *,
    output_size: int,
    target_mode: str = "soft",
    aligned: bool = True,
    sampling_ratio: int = 2,
    projection: str = "legacy",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mask BCE over licensed domain cells only.

    For each positive proposal the target is the official GT mask of its matched
    instance, projected through ROIAlign.  A cell is in the denominator when
    the sampled projected-domain occupancy is ``>= 0.999``; this is a sampled
    licensing test, not an exact full-pixel-footprint test under adaptive
    sampling.  Unknown ink therefore never enters the loss.  At a crossing, the
    instance's own positive target wins (target=1) while the other instance's
    known pixels are licensed negatives (target=0).

    ``target_mode='soft'`` keeps the ROI-projected occupancy fraction as the
    BCE target.  This is required for one-pixel official strokes: a hard
    ``>0.5`` target on a loose proposal ROI erases the stroke entirely (a
    1px x 240px line inside an 80px ROI produced 0 hard-positive cells and
    10.0 soft mass).  Tie-tight GT ROIs keep both modes, but sampling includes
    loose proposals, so the soft target is the faithful objective.  Counts of
    both are always reported.

    ``projection='legacy'`` keeps the original ABI and uses ``aligned`` /
    ``sampling_ratio`` (default ``True, 2``) for BOTH the GT target and the
    licensed-domain projection.  ``projection='paste_consistent'`` derives both
    ROIs from the exact integer paste geometry and forces the effective
    ``aligned=False, sampling_ratio=-1``; the raw legacy arguments are ignored
    in that mode and reported separately so no mixed convention can occur.
    """

    if target_mode not in ("soft", "hard"):
        raise ValueError("target_mode must be 'soft' or 'hard'")
    if not isinstance(aligned, bool):
        raise ValueError("aligned must be a bool")
    if sampling_ratio not in (-1, 2):
        raise ValueError("sampling_ratio must be -1 or 2")
    if projection not in ("legacy", "paste_consistent"):
        raise ValueError("projection must be 'legacy' or 'paste_consistent'")
    requested_aligned = aligned
    requested_sampling_ratio = sampling_ratio
    if projection == "paste_consistent":
        effective_aligned = False
        effective_sampling_ratio = -1
    else:
        effective_aligned = requested_aligned
        effective_sampling_ratio = requested_sampling_ratio
    total = int(positive_labels.numel())
    if total == 0:
        return mask_logits.sum() * 0.0, {
            "original_positive_proposals": 0,
            "excluded_narrow_proposals": 0,
            "effective_supervised_proposals": 0,
            "licensed_cells": 0,
            "positive_cells": 0,
            "negative_cells": 0,
            "soft_positive_mass": 0.0,
            "hard_positive_cells": 0,
            "target_mode": target_mode,
            "target_projection": projection,
            "target_aligned": effective_aligned,
            "target_sampling_ratio": effective_sampling_ratio,
            "target_aligned_requested": requested_aligned,
            "target_sampling_ratio_requested": requested_sampling_ratio,
        }
    if int(mask_logits.shape[0]) != total:
        raise ValueError("mask logits rows %d != positive labels %d" % (mask_logits.shape[0], total))

    targets: List[torch.Tensor] = []
    licensed_cells: List[torch.Tensor] = []
    row_valid_parts: List[torch.Tensor] = []
    excluded_narrow = 0
    start = 0
    for image_index, (proposal, matched) in enumerate(zip(proposals, matched_indices)):
        count = int(proposal.shape[0])
        if count == 0:
            continue
        # The GT projection and domain projection always use the SAME derived
        # ROI.  For paste_consistent, boxes whose integer paste extent cannot
        # satisfy roi_align(aligned=False) are excluded at this boundary only;
        # proposals/matched masks/labels/logit rows keep their exact order.
        if projection == "paste_consistent":
            roi_boxes, valid, _, _ = _paste_consistent_rois(
                proposal, output_size=output_size
            )
            valid = valid.to(dtype=torch.bool, device=proposal.device)
            excluded_narrow += int(count - int(valid.sum().item()))
        else:
            roi_boxes = proposal
            valid = torch.ones(
                (count,), dtype=torch.bool, device=proposal.device
            )
        row_valid_parts.append(valid)
        dtype = torch.float32 if target_mode == "soft" else torch.bool
        full_target = torch.zeros(
            (count, output_size, output_size), dtype=dtype, device=mask_logits.device
        )
        full_licensed = torch.zeros(
            (count, output_size, output_size), dtype=torch.bool, device=mask_logits.device
        )
        supervised_count = int(valid.sum().item())
        if supervised_count:
            proposal_use = proposal[valid]
            matched_use = matched[valid]
            boxes_use = roi_boxes[valid]
            # ``selected`` is stacked along batch K, so each ROI addresses its
            # own batch row 0..K-1.  The domain map has batch 1, so every domain
            # ROI addresses row 0.  Using ``image_index`` for both is wrong for
            # image>0.
            selected_batch = torch.arange(
                supervised_count, dtype=proposal.dtype, device=proposal.device
            )[:, None]
            gt_rois = torch.cat((selected_batch, boxes_use), dim=1)
            domain_batch = torch.zeros(
                (supervised_count, 1), dtype=proposal.dtype, device=proposal.device
            )
            domain_rois = torch.cat((domain_batch, boxes_use), dim=1)
            gt = gt_masks[image_index]
            if gt.numel() == 0:
                raise ValueError("positive proposal without a GT mask")
            selected = gt[matched_use]
            projected = roi_align(
                selected[:, None].float(),
                gt_rois,
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=effective_sampling_ratio,
                aligned=effective_aligned,
            )[:, 0]
            projected = projected.clamp(0.0, 1.0)
            domain = domain_maps[image_index]
            projected_domain = roi_align(
                domain[None, None].float(),
                domain_rois,
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=effective_sampling_ratio,
                aligned=effective_aligned,
            )[:, 0]
            if target_mode == "soft":
                full_target[valid] = projected
            else:
                full_target[valid] = projected > 0.5
            full_licensed[valid] = projected_domain >= 0.999
        targets.append(full_target)
        licensed_cells.append(full_licensed)
        start += count
    if start != total:
        raise ValueError("positive proposals %d != positive labels %d" % (start, total))

    target_tensor = torch.cat(targets, dim=0)
    row_valid = torch.cat(row_valid_parts, dim=0)
    licensed_tensor = torch.cat(licensed_cells, dim=0) & row_valid[:, None, None]
    flat_count = int(licensed_tensor.sum().item())
    hard_tensor = target_tensor > 0.5
    counts = {
        "original_positive_proposals": total,
        "excluded_narrow_proposals": excluded_narrow,
        "effective_supervised_proposals": int(row_valid.sum().item()),
        "licensed_cells": flat_count,
        "positive_cells": int((hard_tensor & licensed_tensor).sum().item()),
        "negative_cells": int((~hard_tensor & licensed_tensor).sum().item()),
        "soft_positive_mass": float(
            target_tensor[licensed_tensor].sum().item()
        ),
        "hard_positive_cells": int((hard_tensor & licensed_tensor).sum().item()),
        "target_mode": target_mode,
        "target_projection": projection,
        "target_aligned": effective_aligned,
        "target_sampling_ratio": effective_sampling_ratio,
        "target_aligned_requested": requested_aligned,
        "target_sampling_ratio_requested": requested_sampling_ratio,
    }
    if flat_count == 0:
        return mask_logits.sum() * 0.0, counts

    rows = torch.arange(total, device=mask_logits.device)
    per_instance = mask_logits[rows, positive_labels.long()]
    selected_logits = per_instance[licensed_tensor].float()
    selected_targets = target_tensor[licensed_tensor].float()
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="sum"
    ) / float(flat_count)
    return loss, counts
