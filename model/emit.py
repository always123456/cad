"""Independent-mask document emission for direct Mask R-CNN instances.

The official document must preserve overlapping instances.  Therefore each
selected predicted mask is vectorised on its own (class raster + unique Thing
instance id) with the canonical ``vectorize_masks(..., emitter="runs")`` and the
12-field documents are merged.  A score-resolved dense map is produced only as
a diagnostic and never becomes the official output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import contract


STUFF_CLASS_START = 30
IGNORE_CLASS = 35
VIEWBOX = (0.0, 0.0, 100.0, 100.0)
DOCUMENT_FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
    "obj_cts",
    "boxes",
    "rgb",
    "layerIds",
    "widths",
)


@dataclass(frozen=True)
class EmittedInstance:
    prediction_index: int
    class_id: int
    instance_id: int
    score: float
    pixels: int
    bbox: Tuple[float, float, float, float]


def _empty_document(width: int, height: int) -> Dict[str, Any]:
    return {
        "commands": [],
        "args": [],
        "lengths": [],
        "semanticIds": [],
        "instanceIds": [],
        "width": int(width),
        "height": int(height),
        "obj_cts": [],
        "boxes": [],
        "rgb": [],
        "layerIds": [],
        "widths": [],
    }


def _single_instance_document(
    mask: np.ndarray,
    class_id: int,
    instance_id: int,
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    emitter: str,
    vectorize: Any,
) -> Dict[str, Any]:
    height, width = mask.shape
    semantic = np.full((height, width), IGNORE_CLASS, dtype=np.int16)
    instance = np.full((height, width), -1, dtype=np.int16)
    semantic[mask] = int(class_id)
    if class_id < STUFF_CLASS_START and instance_id >= 0:
        instance[mask] = int(instance_id)
    document = vectorize.vectorize_masks(
        semantic,
        instance,
        1.0 / scale,
        1.0 / scale,
        -offset_x / scale,
        -offset_y / scale,
        VIEWBOX,
        emitter=emitter,
    )
    return document


def _remap_instance(document: Dict[str, Any], global_id: int) -> None:
    document["instanceIds"] = [
        int(global_id) if int(value) >= 0 else -1 for value in document["instanceIds"]
    ]
    for entry in document["obj_cts"]:
        if int(entry[3]) >= 0:
            entry[3] = int(global_id)


def _append_document(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for field in DOCUMENT_FIELDS:
        if field in ("width", "height"):
            continue
        target[field].extend(source[field])


def emit_instances(
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    scores: Sequence[float],
    *,
    height: int,
    width: int,
    prediction_indices: Sequence[int] | None = None,
    emitter: str = "runs",
    instance_ids: Sequence[int] | None = None,
    make_dense_diagnostic: bool = True,
) -> Dict[str, Any]:
    """Vectorise selected independent masks and merge them into one document.

    Thing classes 0..29 get globally unique non-negative ids in selection order.
    Stuff classes 30..34 merge per class into a single instance -1 mask.
    """

    if emitter not in ("runs", "skeleton"):
        raise ValueError("emitter must be 'runs' or 'skeleton'")
    if not (len(masks) == len(class_ids) == len(scores)):
        raise ValueError("masks/class_ids/scores length mismatch")
    if height <= 0 or width <= 0:
        raise ValueError("invalid native size")
    modules = contract.project_modules()
    vectorize = modules["vectorize"]
    submission = modules["submission"]
    scale, offset_x, offset_y = submission.viewbox_transform(int(width), int(height), VIEWBOX)

    output = _empty_document(100, 100)
    emitted: List[EmittedInstance] = []
    expected_masks: List[Tuple[Tuple[int, int], np.ndarray]] = []
    used_ids: List[int] = []
    next_id = 0
    if instance_ids is not None:
        if len(instance_ids) != len(masks):
            raise ValueError("instance_ids length mismatch")
        next_id = max([int(v) for v in instance_ids if int(v) >= 0], default=-1) + 1

    thing_order = [
        index for index, class_id in enumerate(class_ids) if int(class_id) < STUFF_CLASS_START
    ]
    stuff_groups: Dict[int, List[int]] = {}
    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        if class_id < 0 or class_id > 34:
            raise ValueError("class id %d outside canonical 0..34" % class_id)
        if class_id >= STUFF_CLASS_START:
            stuff_groups.setdefault(class_id, []).append(index)

    for index in thing_order:
        mask = np.asarray(masks[index], dtype=bool)
        if mask.ndim != 2 or mask.shape != (height, width):
            raise ValueError("mask shape does not match native H/W")
        if not bool(mask.any()):
            continue
        global_id = next_id
        next_id += 1
        document = _single_instance_document(
            mask,
            int(class_ids[index]),
            global_id,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
            emitter=emitter,
            vectorize=vectorize,
        )
        _remap_instance(document, global_id)
        _append_document(output, document)
        expected_masks.append(((int(class_ids[index]), int(global_id)), mask))
        rows, cols = np.where(mask)
        emitted.append(
            EmittedInstance(
                prediction_index=int(
                    prediction_indices[index] if prediction_indices is not None else index
                ),
                class_id=int(class_ids[index]),
                instance_id=int(global_id),
                score=float(scores[index]),
                pixels=int(mask.sum()),
                bbox=(
                    float(cols.min()),
                    float(rows.min()),
                    float(cols.max() + 1),
                    float(rows.max() + 1),
                ),
            )
        )
        used_ids.append(global_id)

    for class_id in sorted(stuff_groups):
        group = stuff_groups[class_id]
        union = np.zeros((height, width), dtype=bool)
        for index in group:
            mask = np.asarray(masks[index], dtype=bool)
            if mask.ndim != 2 or mask.shape != (height, width):
                raise ValueError("mask shape does not match native H/W")
            union |= mask
        if not bool(union.any()):
            continue
        document = _single_instance_document(
            union,
            int(class_id),
            -1,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
            emitter=emitter,
            vectorize=vectorize,
        )
        _append_document(output, document)
        expected_masks.append(((int(class_id), -1), union))
        rows, cols = np.where(union)
        best = max(group, key=lambda value: float(scores[value]))
        emitted.append(
            EmittedInstance(
                prediction_index=int(
                    prediction_indices[best] if prediction_indices is not None else best
                ),
                class_id=int(class_id),
                instance_id=-1,
                score=float(scores[best]),
                pixels=int(union.sum()),
                bbox=(
                    float(cols.min()),
                    float(rows.min()),
                    float(cols.max() + 1),
                    float(rows.max() + 1),
                ),
            )
        )

    output["width"] = 100
    output["height"] = 100
    errors = submission.validate_document(output)
    if errors:
        raise ValueError("merged document invalid: " + ";".join(errors))

    diagnostics: Dict[str, Any] = {
        "emitter": emitter,
        "native_size": {"height": int(height), "width": int(width)},
        "emitted_instances": len(emitted),
        "selected_instances": len(masks),
        "thing_instances": len(used_ids),
        "stuff_classes": sorted(int(k) for k in stuff_groups),
        "global_instance_ids": used_ids,
        "independent_masks": True,
        "dense_diagnostic_only": True,
    }
    dense = None
    if make_dense_diagnostic:
        dense = _dense_diagnostic(
            masks, class_ids, scores, height=height, width=width
        )
        diagnostics["dense_diagnostic"] = {
            "conflict_pixels": dense["conflict_pixels"],
            "covered_pixels": dense["covered_pixels"],
            "policy": "score-winner diagnostic; never the official document",
        }
    return {
        "document": output,
        "instances": emitted,
        "diagnostics": diagnostics,
        "dense_diagnostic": dense,
        "expected_masks": expected_masks,
    }


def _dense_diagnostic(
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    scores: Sequence[float],
    *,
    height: int,
    width: int,
) -> Dict[str, Any]:
    best_score = np.full((height, width), -np.inf, dtype=np.float32)
    semantic = np.full((height, width), IGNORE_CLASS, dtype=np.int16)
    coverage = np.zeros((height, width), dtype=np.int32)
    for mask, class_id, score in zip(masks, class_ids, scores):
        mask = np.asarray(mask, dtype=bool)
        coverage += mask.astype(np.int32)
        better = mask & (float(score) > best_score)
        best_score[better] = float(score)
        semantic[better] = int(class_id)
    return {
        "semantic": semantic,
        "covered_pixels": int((coverage > 0).sum()),
        "conflict_pixels": int((coverage > 1).sum()),
    }


def roundtrip_iou(
    expected: Sequence[Tuple[Tuple[int, int], np.ndarray]],
    document: Dict[str, Any],
    *,
    height: int,
    width: int,
) -> Dict[str, Any]:
    """Rasterise the official document and compute per-key IoU against sources.

    ``expected`` is the exact independent-mask key list used for emission:
    one ``(class, global_instance)`` per Thing and one union mask per Stuff
    class.  Same-count displacement is visible as IoU < 1; lost and added
    pixels are reported separately.  This is a native roundtrip only and says
    nothing about the canonical 1000px scoring domain.
    """

    modules = contract.project_modules()
    raster = modules["raster_data"]
    segments = raster.rasterize_document(document, raster_size=(int(width), int(height)))
    by_key = {(int(s["class_id"]), int(s["instance_id"])): np.asarray(s["mask"], dtype=bool) for s in segments}
    report = []
    for key, source in expected:
        source = np.asarray(source, dtype=bool)
        target = by_key.get(key)
        if target is None:
            intersection = 0
            union = int(source.sum())
            lost = int(source.sum())
            added = 0
        else:
            intersection = int((source & target).sum())
            union = int((source | target).sum())
            lost = int((source & ~target).sum())
            added = int((~source & target).sum())
        report.append(
            {
                "class_id": int(key[0]),
                "instance_id": int(key[1]),
                "source_pixels": int(source.sum()),
                "raster_pixels": int(target.sum()) if target is not None else 0,
                "intersection": intersection,
                "union": union,
                "iou": float(intersection / union) if union else 1.0,
                "lost_pixels": lost,
                "added_pixels": added,
            }
        )
    return {
        "keys": len(by_key),
        "expected_keys": len(expected),
        "segments": report,
        "min_iou": float(min([entry["iou"] for entry in report], default=1.0)),
        "mean_iou": float(
            sum(entry["iou"] for entry in report) / len(report) if report else 1.0
        ),
        "total_lost_pixels": int(sum(entry["lost_pixels"] for entry in report)),
        "total_added_pixels": int(sum(entry["added_pixels"] for entry in report)),
        "domain": "native_canonical_rasterizer",
        "native_inverse_exact": bool(int(height) == int(width)),
        "note": (
            "canonical rasterize_document stretches viewBox by H/W; it is the exact "
            "inverse of the official letterbox affine only for square native images"
        ),
    }
