"""PNG-only raster U-Net prediction and fit/tune internal evaluation.

Run ``python -m cadvec.recognition_raster_infer --help``. Predictions use
semantic_cc8_v1: semantic argmax on image-only ink support, ignore35 elsewhere.
Support reuses the data policy: alpha composite over white, then min RGB < 254.
Unknown GT semantic pixels are excluded independently of instance validity.
Eight-connectivity can merge touching objects and split disconnected parts of
one object; embeddings/centers are deliberately not used by this baseline.

Any positive PNG raster size is accepted without resizing.  Pixel -> viewBox
coordinates reuse the calibrated nominal ``xMidYMid meet`` mapping of
``recognition_submission``: ``s = min(W, H) / 100``,
``tx = (W - 100*s)/2``, ``ty = (H - 100*s)/2`` and
``(x_svg, y_svg) = ((x_px - tx)/s, (y_px - ty)/s)``.  That mapping is an
explicit unconfirmed assumption: inference receives no SVG metadata.

Inputs are bounded before decoding by the frontend resource defaults
(``max_pixels=40_000_000``, ``max_file_bytes=64 MiB`` from
``png_frontend.FrontendConfig``).  That is an input guard only; it is not a
GPU memory guarantee for the raster model.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import png_frontend as frontend
from . import png_raster_vectorize as vectorizer
from . import recognition_raster_data as data
from . import recognition_submission as submission
from . import recognition_raster_train as train

try:
    from scipy.ndimage import label as _scipy_label
except ImportError:  # no dependency installation required
    _scipy_label = None


# A tune diagnostic describes a frozen prefix; it never feeds checkpoint or
# decoder selection.  Four samples keep it cheap and prevent metric-driven
# iteration over the tune role.
TUNE_DIAGNOSTIC_SAMPLE_CAP = 4
KNOWN_SUPPORT_METRIC_VERSION = "known_support_pq_v2_shared_domain"

KNOWN_SUPPORT_POLICY = {
    "name": "known-support-only",
    "metric_version": KNOWN_SUPPORT_METRIC_VERSION,
    "semantic_metrics": "semantic_valid only",
    "shared_evaluable_domain": (
        "union of valid Thing support and valid Stuff support"
    ),
    "thing_support": (
        "GT segment mask AND semantic_valid AND same class AND instance_valid; "
        "unknown/missing instance yields no GT instance and no instance negatives"
    ),
    "stuff_support": "semantic_valid AND same class, merged per class",
    "gt_clipping": (
        "each GT segment clipped to its class-specific valid support; identity "
        "preserved; empty segments dropped; no connected component recomputation"
    ),
    "prediction_clipping": (
        "every predicted segment clipped to the shared evaluable domain; predicted "
        "class and instance identity preserved; empty segments dropped; no connected "
        "component recomputation"
    ),
    "matching": "strict same-class IoU > 0.5 with Hungarian assignment",
    "wrong_class_predictions": (
        "retained on the shared domain and counted as known false positives"
    ),
    "unknown_predictions": "outside shared evaluable domain; never counted as FP",
    "instance_only_pixels": (
        "no class is invented for valid-instance pixels without semantic class; "
        "reported explicitly as unscorable instance-only coverage"
    ),
    "independent_instance_caveat": (
        "instance-only pixels are not independently scorable PQ support because "
        "no class is available; only semantic-valid instances enter known support"
    ),
    "aggregation": "dataset-micro",
    "official_compatibility": "unconfirmed",
    "checkpoint_selection_use": False,
    "decoder_selection_use": False,
}


class RasterInferError(ValueError):
    """Invalid inference input or model output."""


def _components(mask: Any) -> Any:
    """Label 8-connected components in deterministic row-major order."""
    mask = np.asarray(mask, dtype=bool)
    if _scipy_label is not None:
        labels, _ = _scipy_label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        return labels
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    component = 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        component += 1
        labels[y, x] = component
        pending = [(int(y), int(x))]
        while pending:
            cy, cx = pending.pop()
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = component
                        pending.append((ny, nx))
    return labels


def instance_ids(semantic: Any) -> Any:
    """Separate every Thing component; leave Stuff and ignore35 at -1."""
    semantic = np.asarray(semantic)
    if semantic.ndim != 2 or not np.issubdtype(semantic.dtype, np.integer):
        raise RasterInferError("semantic must be a 2D integer array")
    if semantic.size == 0 or semantic.min() < 0 or semantic.max() > 35:
        raise RasterInferError("semantic must contain only class ids 0..35 (35 is ignore)")
    instances = np.full(semantic.shape, -1, dtype=np.int32)
    offset = 0
    for class_id in range(30):
        mask = semantic == class_id
        if not mask.any():
            continue
        labels = _components(mask)
        instances[mask] = labels[mask] + offset
        offset += int(labels.max())
    return instances


def _is_thing(class_id: int) -> bool:
    return 0 <= class_id <= data.THING_MAX - 1


def _is_stuff(class_id: int) -> bool:
    return data.STUFF_MIN - 1 <= class_id <= data.STUFF_MAX - 1


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _pq_block(tp: int, fp: int, fn: int, sum_iou: float) -> Dict[str, Any]:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "SumIoU": round(float(sum_iou), 12),
        "PQ": _safe_ratio(sum_iou, denominator),
        "SQ": _safe_ratio(sum_iou, tp),
        "RQ": _safe_ratio(tp, denominator),
    }


def _sample_arrays(sample: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Return semantic, semantic_valid, instance and instance_valid arrays.

    Semantic and instance validity stay independent.  A known support pixel
    additionally requires a real class (``0..34``); a known instance identity
    requires a positive local id (``0`` is reserved, ``-1`` is UNKNOWN).
    """
    if not isinstance(sample, Mapping):
        raise RasterInferError("known-support sample must be a mapping")
    if "semantic" not in sample:
        raise RasterInferError("known-support sample has no semantic array")
    semantic = np.asarray(sample["semantic"])
    if semantic.ndim != 2 or not np.issubdtype(semantic.dtype, np.integer):
        raise RasterInferError("sample semantic must be a 2D integer array")
    shape = semantic.shape

    if "semantic_valid" in sample:
        semantic_valid = np.asarray(sample["semantic_valid"], dtype=bool)
    else:
        semantic_valid = semantic >= 0
    if semantic_valid.shape != shape:
        raise RasterInferError("semantic_valid shape does not match semantic")
    semantic_valid = (
        semantic_valid & (semantic >= 0) & (semantic < data.IGNORE_CLASS)
    )

    if "instance" in sample:
        instance = np.asarray(sample["instance"])
        if instance.shape != shape:
            raise RasterInferError("instance shape does not match semantic")
    else:
        instance = np.full(shape, -1, dtype=np.int64)
    if "instance_valid" in sample:
        instance_valid = np.asarray(sample["instance_valid"], dtype=bool)
    else:
        instance_valid = instance > 0
    if instance_valid.shape != shape:
        raise RasterInferError("instance_valid shape does not match semantic")
    instance_valid = instance_valid & (instance > 0)
    return semantic, semantic_valid, instance, instance_valid


def _segment_entries(
    segments: Any, shape: Tuple[int, int], label: str
) -> List[Dict[str, Any]]:
    """Validate segment records without touching their identity or topology."""
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        raise RasterInferError(f"{label} segments must be a sequence")
    entries: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise RasterInferError(f"{label} segment {index} must be a mapping")
        class_id = segment.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)):
            raise RasterInferError(f"{label} segment {index} has no integer class_id")
        mask = np.asarray(segment.get("mask"), dtype=bool)
        if mask.shape != shape:
            raise RasterInferError(
                f"{label} segment {index} mask shape {mask.shape} != sample {shape}"
            )
        instance_id = segment.get("instance_id", -1)
        if isinstance(instance_id, bool) or not isinstance(
            instance_id, (int, np.integer)
        ):
            raise RasterInferError(
                f"{label} segment {index} has no integer instance_id"
            )
        entries.append(
            {
                "class_id": int(class_id),
                "instance_id": int(instance_id),
                "mask": mask,
            }
        )
    return entries


def _known_support(
    semantic: Any,
    semantic_valid: Any,
    instance_valid: Any,
    gt_segments: Sequence[Mapping[str, Any]],
) -> Dict[int, Any]:
    """Build class-specific known support from GT segments plus validity.

    Thing support is limited to GT segment pixels with known class membership
    and valid instance identity.  Stuff support is every semantic-valid pixel of
    the class, merged per class.
    """
    shape = semantic.shape
    zeros = np.zeros(shape, dtype=bool)
    thing_masks: Dict[int, Any] = {}
    for segment in gt_segments:
        class_id = int(segment["class_id"])
        if not _is_thing(class_id):
            continue
        mask = segment["mask"]
        if class_id not in thing_masks:
            thing_masks[class_id] = mask.copy()
        else:
            thing_masks[class_id] = np.logical_or(thing_masks[class_id], mask)

    supports: Dict[int, Any] = {}
    for class_id, mask in thing_masks.items():
        supports[class_id] = (
            mask & semantic_valid & (semantic == class_id) & instance_valid
        )

    stuff_classes = np.unique(
        semantic[
            semantic_valid
            & (semantic >= data.STUFF_MIN - 1)
            & (semantic <= data.STUFF_MAX - 1)
        ]
    )
    for value in stuff_classes:
        class_id = int(value)
        supports[class_id] = semantic_valid & (semantic == class_id)
    return supports


def _clip_segments(
    segments: Sequence[Mapping[str, Any]],
    support: Any,
    shape: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Any, Any]:
    """Clip segments to a shared domain or to per-class supports.

    A boolean ``support`` array is the shared evaluable domain: every segment is
    clipped to it regardless of class, so wrong-class predictions survive with
    their predicted class and remain false positives.  A mapping keeps the
    class-specific GT clipping policy.  Segment identity is preserved and empty
    segments are dropped; components are never recomputed.
    """
    zeros = np.zeros(shape, dtype=bool)
    if isinstance(support, Mapping):
        class_supports: Optional[Mapping[int, Any]] = support
        shared: Optional[Any] = None
    else:
        class_supports = None
        shared = np.asarray(support, dtype=bool)
        if shared.shape != shape:
            raise RasterInferError(
                f"shared support shape {shared.shape} != sample {shape}"
            )
    kept: List[Dict[str, Any]] = []
    union_before = zeros
    union_after = zeros
    for segment in segments:
        mask = segment["mask"]
        union_before = np.logical_or(union_before, mask)
        if shared is not None:
            domain = shared
        else:
            domain = class_supports.get(int(segment["class_id"]))
        clipped = mask & domain if domain is not None else zeros
        if not bool(clipped.any()):
            continue
        union_after = np.logical_or(union_after, clipped)
        kept.append(
            {
                "class_id": int(segment["class_id"]),
                "instance_id": int(segment["instance_id"]),
                "mask": clipped,
            }
        )
    counts = {
        "before": int(len(segments)),
        "after": int(len(kept)),
        "dropped_empty": int(len(segments) - len(kept)),
        "pixels_before": int(union_before.sum()),
        "pixels_after": int(union_after.sum()),
    }
    return kept, counts, union_before, union_after


def _sample_identifier(sample: Mapping[str, Any], index: int) -> str:
    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("sample_id")
        if isinstance(value, str) and value:
            return value
    value = sample.get("sample_id")
    if isinstance(value, str) and value:
        return value
    return f"image-{index}"


def _score_known_support_image(
    sample: Mapping[str, Any],
    predicted_segments: Sequence[Mapping[str, Any]],
    index: int = 0,
) -> Tuple[Dict[str, Any], Dict[int, List[float]]]:
    """Score one image and return its metric block plus per-class totals."""
    semantic, semantic_valid, _instance, instance_valid = _sample_arrays(sample)
    shape = semantic.shape
    if "segments" not in sample:
        raise RasterInferError(
            "known-support scoring requires dense GT segments on the sample"
        )
    gt_segments = _segment_entries(sample["segments"], shape, "GT")
    pred_segments = _segment_entries(predicted_segments, shape, "prediction")

    supports = _known_support(
        semantic, semantic_valid, instance_valid, gt_segments
    )
    known = np.zeros(shape, dtype=bool)
    thing_support = np.zeros(shape, dtype=bool)
    stuff_support = np.zeros(shape, dtype=bool)
    for class_id, mask in supports.items():
        known = np.logical_or(known, mask)
        if _is_thing(class_id):
            thing_support = np.logical_or(thing_support, mask)
        elif _is_stuff(class_id):
            stuff_support = np.logical_or(stuff_support, mask)

    gt_kept, gt_counts, gt_before, gt_after = _clip_segments(
        gt_segments, supports, shape
    )
    pred_kept, pred_counts, pred_before, pred_after = _clip_segments(
        pred_segments, known, shape
    )

    totals = data._score_image(gt_kept, pred_kept)
    block = _pq_block(
        sum(row[0] for row in totals.values()),
        sum(row[1] for row in totals.values()),
        sum(row[2] for row in totals.values()),
        sum(row[3] for row in totals.values()),
    )
    block["per_class"] = {
        str(class_id): _pq_block(*totals[class_id])
        for class_id in sorted(totals)
    }
    support = {
        "semantic_valid_pixels": int(semantic_valid.sum()),
        "instance_valid_pixels": int(instance_valid.sum()),
        "known_support_pixels": int(known.sum()),
        "shared_evaluable_pixels": int(known.sum()),
        "thing_support_pixels": int(thing_support.sum()),
        "stuff_support_pixels": int(stuff_support.sum()),
        "instance_only_pixels": int(
            np.logical_and(instance_valid, np.logical_not(semantic_valid)).sum()
        ),
        "unscorable_instance_only_pixels": int(
            np.logical_and(instance_valid, np.logical_not(semantic_valid)).sum()
        ),
        "unscorable_semantic_valid_pixels": int(
            np.logical_and(semantic_valid, np.logical_not(known)).sum()
        ),
        "gt_segment_pixels_before_clip": gt_counts["pixels_before"],
        "gt_segment_pixels_after_clip": gt_counts["pixels_after"],
        "pred_segment_pixels_before_clip": pred_counts["pixels_before"],
        "pred_segment_pixels_after_clip": pred_counts["pixels_after"],
        "predicted_pixels_outside_known_support": int(
            np.logical_and(pred_before, np.logical_not(pred_after)).sum()
        ),
        "gt_segments_before_clip": gt_counts["before"],
        "gt_segments_after_clip": gt_counts["after"],
        "gt_segments_dropped_empty": gt_counts["dropped_empty"],
        "pred_segments_before_clip": pred_counts["before"],
        "pred_segments_after_clip": pred_counts["after"],
        "pred_segments_dropped_empty": pred_counts["dropped_empty"],
        "known_support_coverage": _safe_ratio(
            int(known.sum()), int(semantic_valid.sum())
        ),
        "unscorable_instance_only_coverage": _safe_ratio(
            int(
                np.logical_and(
                    instance_valid, np.logical_not(semantic_valid)
                ).sum()
            ),
            int(instance_valid.sum()),
        ),
        "gt_segment_pixel_retention": _safe_ratio(
            gt_counts["pixels_after"], gt_counts["pixels_before"]
        ),
        "pred_segment_pixel_retention": _safe_ratio(
            pred_counts["pixels_after"], pred_counts["pixels_before"]
        ),
    }
    block["sample_id"] = _sample_identifier(sample, index)
    block["support"] = support
    return block, totals


def score_known_support(
    samples: Sequence[Mapping[str, Any]],
    predicted_segments: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Diagnostic PQ on known support only; unknown predictions are not FP.

    Semantic accuracy elsewhere must still use ``semantic_valid`` alone.  This
    scorer exists for descriptive tune diagnostics and is never a checkpoint or
    decoder selection signal.

    GT segments are clipped to their class-specific valid support.  Predictions
    are clipped to the shared evaluable domain (the union of valid Thing and
    Stuff support) while retaining their predicted class, so a wholly wrong-class
    prediction survives as an FP and the unmatched GT segment remains an FN.
    Wrong-class spill from a same-class segment can lower its IoU.  Pixels with a
    valid instance identity but no semantic class are not assigned a class; they
    are reported as unscorable instance-only coverage instead.
    """
    if len(samples) != len(predicted_segments):
        raise RasterInferError(
            f"sample/prediction count mismatch: samples={len(samples)} "
            f"predictions={len(predicted_segments)}"
        )
    totals: Dict[int, List[float]] = {}
    support_keys = (
        "semantic_valid_pixels",
        "instance_valid_pixels",
        "known_support_pixels",
        "shared_evaluable_pixels",
        "thing_support_pixels",
        "stuff_support_pixels",
        "instance_only_pixels",
        "unscorable_instance_only_pixels",
        "unscorable_semantic_valid_pixels",
        "gt_segment_pixels_before_clip",
        "gt_segment_pixels_after_clip",
        "pred_segment_pixels_before_clip",
        "pred_segment_pixels_after_clip",
        "predicted_pixels_outside_known_support",
        "gt_segments_before_clip",
        "gt_segments_after_clip",
        "gt_segments_dropped_empty",
        "pred_segments_before_clip",
        "pred_segments_after_clip",
        "pred_segments_dropped_empty",
    )
    support_totals = {key: 0 for key in support_keys}
    per_image: List[Dict[str, Any]] = []
    for index, (sample, prediction) in enumerate(
        zip(samples, predicted_segments)
    ):
        block, image_totals = _score_known_support_image(sample, prediction, index)
        for class_id, (tp, fp, fn, sum_iou) in image_totals.items():
            row = totals.setdefault(class_id, [0, 0, 0, 0.0])
            row[0] += tp
            row[1] += fp
            row[2] += fn
            row[3] += sum_iou
        for key in support_keys:
            support_totals[key] += block["support"][key]
        per_image.append(block)

    aggregate = _pq_block(
        sum(row[0] for row in totals.values()),
        sum(row[1] for row in totals.values()),
        sum(row[2] for row in totals.values()),
        sum(row[3] for row in totals.values()),
    )
    aggregate["per_class"] = {
        str(class_id): _pq_block(*totals[class_id])
        for class_id in sorted(totals)
    }
    support_totals["known_support_coverage"] = _safe_ratio(
        support_totals["known_support_pixels"],
        support_totals["semantic_valid_pixels"],
    )
    support_totals["unscorable_instance_only_coverage"] = _safe_ratio(
        support_totals["unscorable_instance_only_pixels"],
        support_totals["instance_valid_pixels"],
    )
    support_totals["gt_segment_pixel_retention"] = _safe_ratio(
        support_totals["gt_segment_pixels_after_clip"],
        support_totals["gt_segment_pixels_before_clip"],
    )
    support_totals["pred_segment_pixel_retention"] = _safe_ratio(
        support_totals["pred_segment_pixels_after_clip"],
        support_totals["pred_segment_pixels_before_clip"],
    )
    result = {
        "name": "known_support_pq",
        "metric_version": KNOWN_SUPPORT_METRIC_VERSION,
        "official_compatibility": "unconfirmed",
        "checkpoint_selection_use": False,
        "decoder_selection_use": False,
        "policy": dict(KNOWN_SUPPORT_POLICY),
        **aggregate,
        "n_images": len(per_image),
        "support": support_totals,
        "per_image": per_image,
    }
    return result


def _select_records(
    records: Sequence[Mapping[str, Any]], role: str, max_samples: Optional[int]
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Deduplicate by sample_id and enforce the tune diagnostic prefix.

    Selection happens before any checkpoint or sample asset is opened.  Frozen
    list order is preserved; conflicting duplicate ids are rejected.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise RasterInferError("role sample listing must be a sequence")
    deduplicated: List[Mapping[str, Any]] = []
    seen: Dict[str, Tuple[Any, Any]] = {}
    duplicates = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise RasterInferError("role sample record must be a mapping")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RasterInferError("role sample record has no sample_id")
        png_path = record.get("png_path")
        svg_path = record.get("svg_path")
        if not isinstance(png_path, str) or not png_path:
            raise RasterInferError(f"{sample_id}: png_path missing")
        if not isinstance(svg_path, str) or not svg_path:
            raise RasterInferError(f"{sample_id}: svg_path missing")
        identity = (png_path, svg_path)
        if sample_id in seen:
            if seen[sample_id] != identity:
                raise RasterInferError(
                    f"duplicate sample_id {sample_id!r} has conflicting assets"
                )
            duplicates += 1
            continue
        seen[sample_id] = identity
        deduplicated.append(record)

    limit = max_samples
    if role == "tune" and (
        limit is None or limit > TUNE_DIAGNOSTIC_SAMPLE_CAP
    ):
        limit = TUNE_DIAGNOSTIC_SAMPLE_CAP
    selected = deduplicated if limit is None else deduplicated[:limit]
    metadata = {
        "role": role,
        "frozen_order": True,
        "single_pass": True,
        "tune_sample_cap": TUNE_DIAGNOSTIC_SAMPLE_CAP,
        "cap_applied": role == "tune",
        "max_samples_requested": max_samples,
        "applied_sample_limit": limit,
        "records_listed": len(records),
        "records_after_deduplication": len(deduplicated),
        "duplicates_dropped": duplicates,
        "selected_sample_ids": [record["sample_id"] for record in selected],
    }
    return list(selected), metadata


# Input guard only: these are the frontend resource defaults, not a guarantee
# that the raster model fits in GPU memory for an image within the limits.
_INPUT_LIMITS = frontend.FrontendConfig()
INPUT_MAX_PIXELS = _INPUT_LIMITS.max_pixels
INPUT_MAX_FILE_BYTES = _INPUT_LIMITS.max_file_bytes
_INPUT_LIMITS_SOURCE = "png_frontend.FrontendConfig defaults"
_PNG_HEADER_BYTES = 24  # signature(8) + chunk length(4) + "IHDR"(4) + 8 dims


def _check_png_input(png_path: str) -> None:
    """Reject oversized PNG files or rasters before decode/compositing.

    The byte check uses ``os.stat`` when available; the pixel check parses the
    IHDR width/height from the first 24 bytes.  Errors that prevent reading the
    header are left to ``Image.open`` so non-PNG diagnostics stay unchanged.
    Limits mirror the frontend defaults (``max_pixels=40M``,
    ``max_file_bytes=64M``) and bound input only, never GPU memory.
    """

    try:
        file_bytes = os.path.getsize(png_path)
    except OSError:
        file_bytes = None
    if file_bytes is not None and file_bytes > INPUT_MAX_FILE_BYTES:
        raise RasterInferError(
            f"PNG file is {file_bytes} bytes, above max_file_bytes="
            f"{INPUT_MAX_FILE_BYTES}; nothing was decoded"
        )
    try:
        with open(png_path, "rb") as handle:
            header = handle.read(_PNG_HEADER_BYTES)
    except OSError:
        return
    if len(header) < _PNG_HEADER_BYTES:
        raise RasterInferError(
            f"PNG header is {len(header)} bytes, shorter than "
            f"{_PNG_HEADER_BYTES}; nothing was decoded"
        )
    if header[:8] != frontend.PNG_SIGNATURE:
        raise RasterInferError("prediction input must be PNG")
    if header[12:16] != b"IHDR":
        raise RasterInferError("PNG has no IHDR header; nothing was decoded")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise RasterInferError(
            f"PNG dimensions must be positive; got {width}x{height}"
        )
    pixels = width * height
    if pixels > INPUT_MAX_PIXELS:
        raise RasterInferError(
            f"image is {width}x{height} = {pixels} pixels, above max_pixels="
            f"{INPUT_MAX_PIXELS}; no resize or crop is applied"
        )


def _read_png(png_path: str) -> Any:
    """Return composited RGB CHW float32; any positive raster size, no resize."""

    _check_png_input(png_path)
    with Image.open(png_path) as source:
        if source.format != "PNG":
            raise RasterInferError("prediction input must be PNG")
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", source.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, rgba).convert("RGB")
        image = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(image, (2, 0, 1)).copy()


def _viewbox_affine(width: int, height: int) -> Dict[str, float]:
    """Nominal shared viewBox mapping for a ``width x height`` PNG raster.

    Reuses ``recognition_submission.viewbox_transform`` (uniform scale, centred
    letterbox).  Returns the vectorizer affine ``x_view = x_px * scale_x +
    offset_x`` plus the pixel-space calibration terms for provenance.
    """

    try:
        scale, letterbox_x, letterbox_y = submission.viewbox_transform(
            int(width), int(height), submission.VIEWBOX
        )
    except submission.SubmissionError as exc:
        raise RasterInferError(f"unsupported PNG size {width}x{height}: {exc}") from exc
    inverse = 1.0 / scale
    return {
        "pixels_per_view_unit": scale,
        "letterbox_x": letterbox_x,
        "letterbox_y": letterbox_y,
        "scale_x": inverse,
        "scale_y": inverse,
        "offset_x": -letterbox_x * inverse,
        "offset_y": -letterbox_y * inverse,
    }


def _predict(png_path: str, model: Any, device: str, tol_px: float) -> Dict[str, Any]:
    import torch

    image = _read_png(png_path)
    height, width = int(image.shape[1]), int(image.shape[2])
    affine = _viewbox_affine(width, height)
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.from_numpy(image).unsqueeze(0).to(device))
        logits = outputs["semantic_logits"]
        if tuple(logits.shape) != (1, 35, height, width):
            raise RasterInferError(f"unexpected semantic logits shape: {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise RasterInferError("semantic logits must be finite")
        semantic = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    # Fixed semantic_cc8_v1 decoder: support is PNG-only and must be applied
    # before components so blank pixels never bridge otherwise separate objects.
    support = data._image_ink_support(image)
    semantic[~support] = data.IGNORE_CLASS
    instances = instance_ids(semantic)
    document = vectorizer.vectorize_masks(
        semantic, instances,
        scale_x=affine["scale_x"], scale_y=affine["scale_y"],
        offset_x=affine["offset_x"], offset_y=affine["offset_y"],
        viewbox=submission.VIEWBOX,
        tol_px=tol_px,
    )
    return {
        "document": document,
        "semantic": semantic,
        "instance": instances,
        "metadata": {
            "png_path": str(Path(png_path).absolute()),
            "source_size": [width, height],
            "affine": {
                "scale_x": affine["scale_x"],
                "scale_y": affine["scale_y"],
                "offset_x": affine["offset_x"],
                "offset_y": affine["offset_y"],
            },
            "viewbox": [0.0, 0.0, 100.0, 100.0],
            "viewbox_mapping": (
                "uniform xMidYMid meet: s=min(W,H)/100, "
                "tx=(W-100s)/2, ty=(H-100s)/2, svg=(px-t)/s"
            ),
            "viewbox_mapping_status": "unconfirmed assumption; no SVG metadata at inference",
            "input_guard": {
                "max_pixels": INPUT_MAX_PIXELS,
                "max_file_bytes": INPUT_MAX_FILE_BYTES,
                "source": _INPUT_LIMITS_SOURCE,
                "scope": "input guard only; not a GPU memory guarantee",
            },
            "decoder": "semantic_cc8_v1",
            "ink_support_policy": dict(data.INK_SUPPORT_POLICY),
            "ink_support_pixels": int(support.sum()),
            "unknown_policy": "argmax 0..34 on PNG ink support; outside support is ignore35 before CC8",
            "instance_policy": "Thing: per-class 8-connected components; Stuff: merged class, id -1",
            "instance_limitations": "touching objects may merge; disconnected object parts may split",
            "prediction_scope": "PNG only; no GT semantic/instance/validity masks",
            "scorer_compatibility": "internal/unconfirmed",
        },
    }


def predict_png(
    png_path: str, checkpoint_path: str, *, device: str = "cpu", tol_px: float = 0.5,
) -> Dict[str, Any]:
    """Load only the PNG and checkpoint; return document, masks and provenance.

    No SVG, manifest, labels, or ground-truth-derived support enters prediction.
    """
    loaded = train.load_checkpoint(checkpoint_path, device=device)
    result = _predict(png_path, loaded["model"], device, tol_px)
    result["metadata"]["checkpoint_path"] = str(Path(checkpoint_path).absolute())
    return result


def evaluate_role(
    manifest_path: str, checkpoint_path: str, *, role: str = "tune", scale: int = 1,
    max_samples: Optional[int] = None, device: str = "cpu", tol_px: float = 0.5,
) -> Dict[str, Any]:
    """Single descriptive pass over a fit/tune prefix.

    GT is loaded only after its PNG prediction.  Tune is capped to at most four
    unique samples in frozen order, deduplicated before any asset read.  The
    diagnostic reports ``known_support_pq``: predictions outside the shared
    evaluable domain are never false positives, and wrong-class predictions
    inside it remain false positives.  Semantic accuracy uses
    ``semantic_valid`` only.
    Checkpoint and decoder selection never consume this report.
    """
    if role not in ("fit", "tune"):
        raise RasterInferError(f"role must be fit or tune; got {role!r}")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1
    ):
        raise RasterInferError("max_samples must be a positive integer or None")
    records = data.load_role_samples(manifest_path, role, scale)
    selected, selection = _select_records(records, role, max_samples)
    loaded = train.load_checkpoint(checkpoint_path, device=device)
    samples, predicted_segments, per_image = [], [], []
    total_correct = total_valid = 0
    for record in selected:
        prediction = _predict(record["png_path"], loaded["model"], device, tol_px)
        target = data.build_sample(record["png_path"], record["svg_path"], record["sample_id"])
        truth = np.asarray(target["semantic"])
        if truth.shape != prediction["semantic"].shape:
            raise RasterInferError("target/prediction semantic shape mismatch")
        valid = (
            np.asarray(target["semantic_valid"], dtype=bool)
            & (truth >= 0)
            & (truth < data.IGNORE_CLASS)
        )
        valid_count = int(valid.sum())
        correct = int((prediction["semantic"][valid] == truth[valid]).sum())

        pred_segments = data.rasterize_document(
            prediction["document"],
            raster_size=data.RASTER_SIZE,
            canvas_size=data.CANVAS_SIZE,
        )
        metric = score_known_support([target], [pred_segments])
        per_image.append({
            "sample_id": record["sample_id"],
            "known_support_pq": {
                key: metric[key]
                for key in (
                    "name", "metric_version", "TP", "FP", "FN", "SumIoU",
                    "PQ", "SQ", "RQ", "per_class", "support",
                )
            },
            "semantic_correct_pixels": correct,
            "semantic_valid_pixels": valid_count,
            "semantic_pixel_accuracy": correct / valid_count if valid_count else None,
        })
        samples.append(target)
        predicted_segments.append(pred_segments)
        total_correct += correct
        total_valid += valid_count
    return {
        "role": role,
        "scale": scale,
        "sample_count": len(per_image),
        "metric_name": "known_support_pq",
        "metric_version": KNOWN_SUPPORT_METRIC_VERSION,
        "official_compatibility": "unconfirmed",
        "checkpoint_selection_use": False,
        "decoder_selection_use": False,
        "sample_selection": selection,
        "scorer_compatibility": "internal/unconfirmed",
        "scorer_policy": dict(data.SCORER_POLICY),
        "known_support_policy": dict(KNOWN_SUPPORT_POLICY),
        "per_image": per_image,
        "aggregate": {
            "known_support_pq": score_known_support(samples, predicted_segments),
            "semantic_correct_pixels": total_correct,
            "semantic_valid_pixels": total_valid,
            "semantic_pixel_accuracy": total_correct / total_valid if total_valid else None,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="PNG-only prediction")
    predict.add_argument("--png", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        help="single descriptive fit/tune known-support diagnostic",
        description=(
            "Score one frozen prefix with known_support_pq. Tune is capped to "
            f"{TUNE_DIAGNOSTIC_SAMPLE_CAP} unique samples; the report is never "
            "used for checkpoint or decoder selection."
        ),
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--role", choices=("fit", "tune"), default="tune")
    evaluate.add_argument("--scale", type=int, required=True)
    evaluate.add_argument(
        "--max-samples", type=int,
        help=f"fit prefix limit; tune is always capped at {TUNE_DIAGNOSTIC_SAMPLE_CAP}",
    )
    for command in (predict, evaluate):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--tol-px", type=float, default=0.5)
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "predict":
        result = predict_png(args.png, args.checkpoint, device=args.device, tol_px=args.tol_px)
        report = result["document"]
    else:
        report = evaluate_role(
            args.manifest, args.checkpoint, role=args.role, scale=args.scale,
            max_samples=args.max_samples, device=args.device, tol_px=args.tol_px,
        )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
