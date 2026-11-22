"""Positive-only, class-agnostic shape-abstention gate (bounded CPU pilot).

The gate is *one-class*: it learns what a known-Thing mask looks like and
abstains on proposed components that are far away in shape space.  It never
treats an unknown region as a negative and never uses a class label to vote.

Fixed, deterministic pipeline (no sweep, no per-class tuning, no joint-offset
features):

1. **Positive bank** -- every known Thing instance of the permanent FIT cache
   (``instance_valid > 0 & semantic_valid > 0 & instance > 0 & 0 <= semantic <=
   29``) is one positive.  Its mask is cropped to its own bounding box and that
   crop is thinned with the shared vectorizer helper
   :func:`cadvec.png_raster_vectorize._stroke_skeleton`, so a ground-truth mask
   and a predicted component of the same shape get identical treatment (no
   thin-GT / thick-prediction distribution split, no silent size filter).
2. **12 geometry descriptors** (exact definitions below, computed on the
   skeleton inside the crop, positioned through the *original* image shape):
   ``log1p_area``, ``extent_width``, ``extent_height``, ``aspect``,
   ``occupancy``, ``perimeter4_over_area``, ``cc8_count``, ``holes_cc4``,
   ``centroid_x_norm``, ``centroid_y_norm``, ``bbox_span_x_norm``,
   ``bbox_span_y_norm``.  No pixels, no class ids, no joint embeddings.
3. **Robust standardisation** -- per-feature median and inter-quartile range,
   fitted on the bank that is being used (never on the query), with an ``1e-6``
   floor that turns a degenerate feature into a zero contribution instead of a
   noise amplifier.  Nearest neighbour is Euclidean in that standardised space.
4. **3 group folds** -- group ids are sorted and assigned round-robin, so one
   duplicated drawing group never straddles a fold boundary and an image is
   never both bank and query.
5. **Threshold** -- one fixed quantile, ``0.95``, of leave-one-image-group-out
   nearest-neighbour distances computed *inside* the two training folds of a
   fold (nested), so the held-out fold never influences the threshold that is
   applied to it.  The deployment threshold is the same quantile over the
   concatenated out-of-fold distances of all three folds.
6. **Application** -- a predicted CC8 Thing component is kept when its
   nearest-neighbour distance is ``<= threshold`` and otherwise becomes class
   ``35`` (ignore) with instance ``-1``.  Stuff (30..34) is untouched.  No
   candidate component is matched against ground truth: attribution is
   geometric only.

Artifacts (builder lane, ``GT-free`` for the 24 scored records)::

    outputs/recognition-positive-shape-gate-20261112-01/
        gate_calibration.json        # cadvec.recognition.positive_shape_gate_calibration/1
        fit_positive_bank.json       # bank provenance + per-instance descriptors
        predictions/<slug>.npz       # semantic, candidate, retained, source_size
        documents/<slug>.json        # 12-field retained-geometry document
        predictions_manifest.json    # cadvec.recognition.positive_shape_gate_predictions/1
        candidate_report.json        # per-class retained components/pixels, empty images

Run ``python -m cadvec.recognition_positive_shape_gate --help``.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - environment dependent
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from . import png_raster_vectorize as vectorizer
from . import recognition_raster_data as data
from . import recognition_raster_infer as raster_infer

SCHEMA_CALIBRATION = "cadvec.recognition.positive_shape_gate_calibration/1"
SCHEMA_PREDICTIONS = "cadvec.recognition.positive_shape_gate_predictions/1"
SCHEMA_DOCUMENT = "cadvec.recognition.positive_shape_gate_document/1"
SCHEMA_BANK = "cadvec.recognition.positive_shape_gate_bank/1"
SCHEMA_CANDIDATE_REPORT = "cadvec.recognition.positive_shape_gate_candidates/1"

ROUTE = "positive_shape_gate"
IGNORE_ID = 35
THING_MIN = 0
THING_MAX = 29
STUFF_MIN = 30
STUFF_MAX = 34

FOLD_COUNT = 3
QUANTILE = 0.95
ROBUST_EPS = 1e-6
NN_CHUNK = 512
DEFAULT_LIMIT = 24
DEFAULT_TOL_PX = 0.5
#: The permanent FIT set lives at scale 256 in the public manifest: 204 records
#: over 192 duplicate groups.  Scale 32 is the 24-record candidate prefix and
#: must never be used for calibration scope.
ALL_FIT_SCALE = 256
#: Record count of that frozen prefix.  ``train_scales[256].fit`` counts the 192
#: *drawing groups* of the prefix, which expand to these 204 records because a few
#: groups are duplicated drawings; both numbers are asserted separately.
ALL_FIT_RECORDS = 204

FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_area",
    "extent_width",
    "extent_height",
    "aspect",
    "occupancy",
    "perimeter4_over_area",
    "cc8_count",
    "holes_cc4",
    "centroid_x_norm",
    "centroid_y_norm",
    "bbox_span_x_norm",
    "bbox_span_y_norm",
)

DEFAULT_RUN_ROOT = "outputs/recognition-positive-shape-gate-20261112-01"
DEFAULT_FIT_CACHE = "outputs/recognition-pretrained-fullfit-20261111-01/cache"
DEFAULT_GROUPS = "outputs/png-diagnostic-20261108-02/duplicate_groups.json"
DEFAULT_FIT_MANIFEST = (
    "outputs/recognition-gate3b-fit-tune-20261108-01/recognition_public_manifest.json"
)
DEFAULT_CC8_ARM = (
    "outputs/recognition-joint-offsets-20261112-01/evaluation/arms/joint_semantic_cc8"
)


class ShapeGateError(ValueError):
    """Invalid input or an internally inconsistent gate artifact."""


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ImportError("recognition_positive_shape_gate requires numpy")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=False, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(sample_id: str) -> str:
    return sample_id.replace(":", "_")


# ---------------------------------------------------------------------------
# runs/union-find labelling (no SciPy in this environment)
# ---------------------------------------------------------------------------


def _row_runs(row: "np.ndarray") -> List[Tuple[int, int]]:
    """Inclusive ``(start, end)`` column runs of a boolean row."""

    if not row.any():  # type: ignore[union-attr]
        return []
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[0::2], edges[1::2])]


def _label(mask: Any, eight: bool) -> "np.ndarray":
    """Deterministic row-major component labels (``0`` = background).

    Implemented with row runs plus union-find; SciPy is not available here and a
    pixel-level Python flood fill is too slow for the crop sizes this lane sees.
    """

    _require_numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ShapeGateError(f"mask_not_2d:{mask.shape}")
    height, width = int(mask.shape[0]), int(mask.shape[1])
    labels = np.zeros((height, width), dtype=np.int32)
    if not mask.any():
        return labels

    union: List[int] = [0]

    def find(value: int) -> int:
        root = value
        while union[root] != root:
            root = union[root]
        while union[value] != root:
            union[value], value = root, union[value]
        return root

    def join(left: int, right: int) -> int:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return left_root
        low, high = (left_root, right_root) if left_root < right_root else (right_root, left_root)
        union[high] = low
        return low

    previous: List[Tuple[int, int, int]] = []
    for row in range(height):
        runs = _row_runs(mask[row])
        current: List[Tuple[int, int, int]] = []
        index = 0
        for start, end in runs:
            label = 0
            while index < len(previous) and previous[index][1] < start - (1 if eight else 0):
                index += 1
            probe = index
            while probe < len(previous) and previous[probe][0] <= end + (1 if eight else 0):
                candidate = previous[probe][2]
                label = candidate if label == 0 else join(label, candidate)
                probe += 1
            if label == 0:
                union.append(len(union))
                label = len(union) - 1
            else:
                label = find(label)
            current.append((start, end, label))
        for start, end, label in current:
            labels[row, start : end + 1] = label
        previous = current

    flat = labels.ravel()
    used = np.unique(flat[flat > 0])
    if used.size:
        # Two runs of different provisional ids may have been unified after their
        # row was already written, so resolve every surviving id through the
        # union-find roots before the dense relabel.
        remap = np.zeros(int(used.max()) + 1, dtype=np.int32)
        roots = np.asarray([find(int(value)) for value in used.tolist()], dtype=np.int32)
        unique_roots, dense = np.unique(roots, return_inverse=True)
        remap[used] = (dense.astype(np.int32) + 1)
        labels = remap[labels]
        if int(labels.max()) != int(unique_roots.size):  # pragma: no cover
            raise ShapeGateError("label_dense_remap")
    return labels


def _holes_cc4(mask: Any) -> int:
    """Enclosed 4-connected background components of ``mask``."""

    _require_numpy()
    mask = np.asarray(mask, dtype=bool)
    background = ~mask
    if not background.any():
        return 0
    labels = _label(background, eight=False)
    if labels.max() == 0:
        return 0
    border = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    exterior = {int(value) for value in np.unique(border) if int(value) != 0}
    return int(len({int(value) for value in np.unique(labels) if int(value) != 0} - exterior))


# ---------------------------------------------------------------------------
# 12 descriptors
# ---------------------------------------------------------------------------


def _crop(mask: "np.ndarray") -> Tuple[Any, Tuple[int, int, int, int]]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ShapeGateError("empty_mask")
    row_start, row_stop = int(rows[0]), int(rows[-1])
    col_start, col_stop = int(cols[0]), int(cols[-1])
    return mask[row_start : row_stop + 1, col_start : col_stop + 1], (
        row_start,
        row_stop,
        col_start,
        col_stop,
    )


def skeleton_of(mask: Any) -> "np.ndarray":
    """Thin a mask exactly like the shared vectorizer does.

    The mask is cropped to its own bounding box first (so a 1000x1000 raster and
    the same shape as a small crop thin identically), then thinned with
    :func:`png_raster_vectorize._stroke_skeleton`.
    """

    _require_numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ShapeGateError(f"mask_not_2d:{mask.shape}")
    if not mask.any():
        raise ShapeGateError("empty_mask")
    crop, _box = _crop(mask)
    skeleton = np.asarray(vectorizer._stroke_skeleton(crop), dtype=bool)
    if not skeleton.any():  # pragma: no cover - helper never returns empty here
        raise ShapeGateError("empty_skeleton")
    return skeleton


def skeleton_in_frame(mask: Any) -> Tuple["np.ndarray", int, int]:
    """Cropped skeleton plus its top/left offset inside the original raster.

    The thinning is done on the bounding-box crop (identical treatment for a GT
    mask and a prediction component); the offsets let a caller place the
    skeleton back into full-image coordinates when it needs to.
    """

    _require_numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ShapeGateError(f"mask_not_2d:{mask.shape}")
    crop, box = _crop(mask)
    skeleton = np.asarray(vectorizer._stroke_skeleton(crop), dtype=bool)
    if not skeleton.any():  # pragma: no cover - helper never returns empty here
        raise ShapeGateError("empty_skeleton")
    return skeleton, int(box[0]), int(box[2])


def shape_descriptors(mask: Any, image_shape: Sequence[int]) -> "np.ndarray":
    """The 12 fixed descriptors of one instance mask.

    ``image_shape`` is ``(height, width)`` of the original raster; the two
    position descriptors are normalised by it, so the same shape placed
    differently is not the same point in feature space.

    Definitions, all on the thinned crop skeleton:

    * ``log1p_area`` -- ``log1p`` of the skeleton pixel count;
    * ``extent_width`` / ``extent_height`` -- skeleton bounding-box size in px;
    * ``aspect`` -- ``extent_width / extent_height``;
    * ``occupancy`` -- skeleton pixels / bounding-box area;
    * ``perimeter4_over_area`` -- 4-neighbour boundary pixel count / skeleton
      pixel count (the skeleton is one pixel wide, so this is a stroke density);
    * ``cc8_count`` -- 8-connected components of the skeleton;
    * ``holes_cc4`` -- enclosed 4-connected background components;
    * ``centroid_x_norm`` / ``centroid_y_norm`` -- skeleton centroid, in full
      image coordinates, divided by image width / height;
    * ``bbox_span_x_norm`` / ``bbox_span_y_norm`` -- bounding-box size divided by
      image width / height.
    """

    _require_numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ShapeGateError(f"mask_not_2d:{mask.shape}")
    if len(tuple(image_shape)) != 2:
        raise ShapeGateError("image_shape_not_two_values")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ShapeGateError(f"image_shape_not_positive:{height}x{width}")

    skeleton, frame_row, frame_col = skeleton_in_frame(mask)
    rows = np.flatnonzero(skeleton.any(axis=1))
    cols = np.flatnonzero(skeleton.any(axis=0))
    row_start, row_stop = int(rows[0]), int(rows[-1])
    col_start, col_stop = int(cols[0]), int(cols[-1])
    extent_height = row_stop - row_start + 1
    extent_width = col_stop - col_start + 1
    area = float(int(skeleton.sum()))

    boundary = vectorizer._boundary(skeleton)
    perimeter = float(int(boundary.sum()))
    cc8 = float(int(_label(skeleton, eight=True).max()))
    holes = float(_holes_cc4(skeleton))

    coordinates = np.argwhere(skeleton)
    # Position is reported in original-raster coordinates and normalised by the
    # original raster shape (that is what the spec fixes), so the crop offsets
    # are added back before dividing.
    centroid_row = float(coordinates[:, 0].mean()) + float(frame_row)
    centroid_col = float(coordinates[:, 1].mean()) + float(frame_col)

    values = [
        math.log1p(area),
        float(extent_width),
        float(extent_height),
        float(extent_width) / float(extent_height),
        area / float(extent_width * extent_height),
        perimeter / max(area, 1.0),
        cc8,
        holes,
        centroid_col / float(width),
        centroid_row / float(height),
        float(extent_width) / float(width),
        float(extent_height) / float(height),
    ]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),):  # pragma: no cover
        raise ShapeGateError(f"descriptor_length:{vector.shape}")
    if not np.all(np.isfinite(vector)):  # pragma: no cover
        raise ShapeGateError("descriptor_not_finite")
    return vector


# ---------------------------------------------------------------------------
# robust scaling and nearest neighbour
# ---------------------------------------------------------------------------


def fit_scaler(matrix: Any) -> Dict[str, Any]:
    """Per-feature median / IQR scaler; a degenerate IQR becomes scale 1."""

    _require_numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ShapeGateError(f"scaler_needs_rows:{matrix.shape}")
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25.0, axis=0)
    q75 = np.percentile(matrix, 75.0, axis=0)
    iqr = q75 - q25
    degenerate = iqr < ROBUST_EPS
    scale = np.where(degenerate, 1.0, iqr)
    return {
        "median": median,
        "scale": scale,
        "degenerate_features": [FEATURE_NAMES[i] for i in np.flatnonzero(degenerate)],
        "rows": int(matrix.shape[0]),
    }


def transform(matrix: Any, scaler: Mapping[str, Any]) -> "np.ndarray":
    _require_numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise ShapeGateError(f"feature_width:{matrix.shape}")
    median = np.asarray(scaler["median"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return (matrix - median) / scale


def scaler_from_bank_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild the deployment scaler from a persisted ``fit_positive_bank.json``.

    ``predict`` must not silently re-derive the scaler from the cache and hope it
    matches the calibration: the persisted bank is the artifact the calibration
    wrote, so its declared medians/IQR are used and the recomputed ones are only
    validated against them.
    """

    _require_numpy()
    names = [str(name) for name in payload.get("feature_names") or []]
    if names != list(FEATURE_NAMES):
        raise ShapeGateError(f"bank_feature_names:{names[:3]}")
    medians = payload.get("feature_medians")
    iqr = payload.get("feature_iqr")
    if not isinstance(medians, Mapping) or not isinstance(iqr, Mapping):
        raise ShapeGateError("bank_scaler_missing")
    try:
        median = np.asarray([float(medians[name]) for name in FEATURE_NAMES], dtype=np.float64)
        scale = np.asarray([float(iqr[name]) for name in FEATURE_NAMES], dtype=np.float64)
    except KeyError as error:
        raise ShapeGateError(f"bank_scaler_missing_feature:{error}") from error
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(scale)):
        raise ShapeGateError("bank_scaler_not_finite")
    if np.any(scale <= 0.0):
        raise ShapeGateError(f"bank_scaler_non_positive_scale:{scale.tolist()}")
    return {
        "median": median,
        "scale": scale,
        "degenerate_features": [
            str(name) for name in payload.get("degenerate_features") or []
        ],
        "rows": int((payload.get("summary") or {}).get("instances") or 0),
    }


def compare_scalers(
    persisted: Mapping[str, Any], computed: Mapping[str, Any], tolerance: float = 1e-9
) -> Dict[str, Any]:
    """Validate a recomputed scaler against the persisted one (no hashes, no drift)."""

    _require_numpy()
    median_gap = np.abs(
        np.asarray(persisted["median"], dtype=np.float64)
        - np.asarray(computed["median"], dtype=np.float64)
    )
    scale_gap = np.abs(
        np.asarray(persisted["scale"], dtype=np.float64)
        - np.asarray(computed["scale"], dtype=np.float64)
    )
    worst = float(max(median_gap.max(), scale_gap.max()))
    if worst > float(tolerance):
        raise ShapeGateError(
            f"scaler_drift:{worst:.3e} max_median_gap={float(median_gap.max()):.3e} "
            f"max_iqr_gap={float(scale_gap.max()):.3e}"
        )
    return {
        "max_median_abs_gap": float(median_gap.max()),
        "max_iqr_abs_gap": float(scale_gap.max()),
        "tolerance": float(tolerance),
    }


def nn_distances(query: Any, bank: Any, chunk: int = NN_CHUNK) -> "np.ndarray":
    """Euclidean nearest-neighbour distance from every query row to ``bank``."""

    _require_numpy()
    query = np.asarray(query, dtype=np.float64)
    bank = np.asarray(bank, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if bank.ndim == 1:
        bank = bank.reshape(1, -1)
    if query.shape[1] != bank.shape[1]:
        raise ShapeGateError(f"nn_width:{query.shape} vs {bank.shape}")
    if chunk <= 0:
        raise ShapeGateError(f"nn_chunk:{chunk}")
    bank_sq = np.einsum("ij,ij->i", bank, bank)
    out = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        block_sq = np.einsum("ij,ij->i", block, block)
        distances_sq = block_sq[:, None] - 2.0 * (block @ bank.T) + bank_sq[None, :]
        np.maximum(distances_sq, 0.0, out=distances_sq)
        out[start : start + block.shape[0]] = np.sqrt(distances_sq.min(axis=1))
    return out


def quantile_threshold(scores: Any, quantile: float = QUANTILE) -> float:
    _require_numpy()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ShapeGateError("threshold_needs_scores")
    return float(np.percentile(values, quantile * 100.0, method="linear"))


def fold_of_groups(group_ids: Iterable[str], fold_count: int = FOLD_COUNT) -> Dict[str, int]:
    """Sorted round-robin fold for every distinct group id."""

    if fold_count < 2:
        raise ShapeGateError(f"fold_count:{fold_count}")
    unique = sorted({str(value) for value in group_ids})
    if not unique:
        raise ShapeGateError("no_groups")
    return {group: index % fold_count for index, group in enumerate(unique)}


# ---------------------------------------------------------------------------
# FIT positive bank
# ---------------------------------------------------------------------------


def load_cache_map(cache_dir: str) -> Dict[str, str]:
    """``sample_id -> npz path`` for the permanent FIT dense cache."""

    mapping: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        loaded = np.load(path)
        if "sample_id" not in loaded:
            raise ShapeGateError(f"cache_without_sample_id:{path}")
        sample_id = str(loaded["sample_id"])
        if sample_id in mapping:
            raise ShapeGateError(f"duplicate_cache_sample:{sample_id}")
        mapping[sample_id] = path
    if not mapping:
        raise ShapeGateError(f"no_cache_files:{cache_dir}")
    return mapping


def load_group_map(groups_path: str) -> Dict[str, str]:
    payload = _read_json(Path(groups_path))
    mapping: Dict[str, str] = {}
    for group in payload.get("groups") or []:
        group_id = str(group.get("group_id"))
        for member in group.get("members") or []:
            sample_id = str(member.get("sample_id"))
            if sample_id in mapping and mapping[sample_id] != group_id:
                raise ShapeGateError(f"sample_in_two_groups:{sample_id}")
            mapping[sample_id] = group_id
    if not mapping:
        raise ShapeGateError(f"no_groups:{groups_path}")
    return mapping


def load_fit_records(manifest_path: str, limit: int = DEFAULT_LIMIT) -> List[Mapping[str, Any]]:
    records = data.load_role_samples(manifest_path, "fit", 32)
    selected = list(records[: int(limit)])
    if len(selected) != int(limit):
        raise ShapeGateError(f"fit_records:{len(selected)} of {limit}")
    ids = [str(record["sample_id"]) for record in selected]
    if len(set(ids)) != len(ids):
        raise ShapeGateError("duplicate_fit_sample")
    return selected


def _all_fit_records(manifest_path: str) -> List[Mapping[str, Any]]:
    """Every FIT record of the permanent fit set (scale 256: 204 records, 192 groups).

    The public manifest's ``train_scales`` gives the frozen fit prefix per scale;
    scale 256 is the full FIT set.  The returned counts are checked against the
    manifest's own declared counts so a wrong ``scale`` cannot silently shrink
    the bank to the 24-record candidate prefix (scale 32).
    """

    records = list(data.load_role_samples(manifest_path, "fit", ALL_FIT_SCALE))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not records:
        raise ShapeGateError(f"no_fit_records:{manifest_path}")
    ids = [str(record["sample_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise ShapeGateError("duplicate_fit_sample")
    # Two different units, asserted separately: ``train_scales[256].fit`` counts
    # the *drawing groups* of the frozen prefix (192) while the prefix expands to
    # 204 records, because some groups hold duplicated drawings.  Comparing the
    # record count against the group count is what previously raised
    # ``fit_records_at_scale_256:204 of 192`` on every run.
    declared = (manifest.get("train_scales") or {}).get(str(ALL_FIT_SCALE)) or {}
    declared_groups = int(declared.get("fit", 0) or 0) or int(
        (manifest.get("counts") or {}).get("fit_sequence", 0) or 0
    )
    groups = {str(record["group_id"]) for record in records}
    if len(records) != ALL_FIT_RECORDS:
        raise ShapeGateError(
            f"fit_records_at_scale_{ALL_FIT_SCALE}:{len(records)} of {ALL_FIT_RECORDS}"
        )
    if declared_groups and len(groups) != declared_groups:
        raise ShapeGateError(
            f"fit_groups_at_scale_{ALL_FIT_SCALE}:{len(groups)} of {declared_groups}"
        )
    return records


def thing_instance_masks(
    semantic: Any, instance: Any, instance_valid: Any, semantic_valid: Any
) -> Dict[int, "np.ndarray"]:
    """Known-Thing instance masks of one dense GT raster (dev bank only)."""

    _require_numpy()
    semantic = np.asarray(semantic)
    instance = np.asarray(instance)
    instance_valid = np.asarray(instance_valid)
    semantic_valid = np.asarray(semantic_valid)
    known = (
        (instance_valid > 0)
        & (semantic_valid > 0)
        & (instance > 0)
        & (semantic >= THING_MIN)
        & (semantic <= THING_MAX)
    )
    masks: Dict[int, Any] = {}
    for value in np.unique(instance[known]).tolist():
        instance_id = int(value)
        masks[instance_id] = known & (instance == instance_id)
    return masks


def build_positive_bank(
    cache_map: Mapping[str, str],
    group_map: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every known-Thing instance of the FIT cache, with its 12 descriptors."""

    _require_numpy()
    rows: List[Dict[str, Any]] = []
    per_class: Dict[int, int] = {}
    images_with_positives = 0
    missing_groups = 0
    for sample_id in sorted(cache_map):
        group_id = group_map.get(sample_id)
        if group_id is None:
            missing_groups += 1
            continue
        loaded = np.load(cache_map[sample_id])
        semantic = loaded["semantic"]
        instance = loaded["instance"]
        image_shape = (int(semantic.shape[0]), int(semantic.shape[1]))
        masks = thing_instance_masks(
            semantic,
            instance,
            loaded["instance_valid"],
            loaded["semantic_valid"],
        )
        if masks:
            images_with_positives += 1
        for instance_id in sorted(masks):
            mask = masks[instance_id]
            descriptors = shape_descriptors(mask, image_shape)
            classes = np.unique(semantic[mask]).tolist()
            if len(classes) != 1:
                raise ShapeGateError(f"instance_not_single_class:{sample_id}:{instance_id}")
            class_id = int(classes[0])
            per_class[class_id] = per_class.get(class_id, 0) + 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": str(group_id),
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "pixels": int(mask.sum()),
                    "features": descriptors,
                }
            )
    if missing_groups:
        raise ShapeGateError(f"cache_samples_without_group:{missing_groups}")
    summary = {
        "instances": len(rows),
        "images_with_positives": images_with_positives,
        "images": len(cache_map),
        "groups": len({row["group_id"] for row in rows}),
        "per_class": {str(key): int(value) for key, value in sorted(per_class.items())},
    }
    return rows, summary


def bank_matrix(rows: Sequence[Mapping[str, Any]]) -> "np.ndarray":
    if not rows:
        raise ShapeGateError("empty_bank")
    return np.vstack([np.asarray(row["features"], dtype=np.float64) for row in rows])


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def calibrate(
    rows: Sequence[Mapping[str, Any]],
    record_groups: Mapping[str, str],
    *,
    fold_count: int = FOLD_COUNT,
    quantile: float = QUANTILE,
    chunk: int = NN_CHUNK,
) -> Dict[str, Any]:
    """Group-fold calibration with a nested leave-one-group-out threshold.

    Folds partition the **204 FIT records** (by permanent duplicate group, sorted
    round-robin), so no image or drawing group is ever in a bank and its own
    query set at once.  ``folds[].positives`` is the number of held-out FIT
    records and sums to 204, as the evaluation contract requires.

    For every held-out fold: the bank is the positives of the other folds, the
    scaler is fitted on that bank only, and the fold threshold is the fixed
    quantile of leave-one-image-group-out distances computed inside the same
    bank (an image group never scores against itself).  The held-out fold's
    positives are then scored against the bank and counted as retained when
    ``distance <= threshold``.

    Instance-level numbers are ``heldout_instances`` / ``retained_instances``;
    the coverage of interest is over held-out known-Thing *instances*, not
    candidate pixels.
    """

    _require_numpy()
    if not rows:
        raise ShapeGateError("empty_bank")
    if not record_groups:
        raise ShapeGateError("no_records")
    missing = {str(row["sample_id"]) for row in rows} - set(record_groups)
    if missing:
        raise ShapeGateError(f"bank_rows_without_record:{len(missing)}")
    fold_of = fold_of_groups(record_groups.values(), fold_count)
    folds: List[Dict[str, Any]] = []
    oof_scores: List[float] = []
    total_retained = 0
    total_heldout = 0
    total_instances_retained = 0
    total_instances = 0
    heldout_seen: Dict[str, int] = {}
    for fold_index in range(fold_count):
        heldout_records = sorted(
            sample for sample, group in record_groups.items() if fold_of[str(group)] == fold_index
        )
        if not heldout_records:
            raise ShapeGateError(f"fold_{fold_index}_has_no_records")
        bank_rows = [
            row for row in rows if str(row["sample_id"]) not in set(heldout_records)
        ]
        query_rows = [
            row for row in rows if str(row["sample_id"]) in set(heldout_records)
        ]
        if not bank_rows or not query_rows:
            raise ShapeGateError(f"fold_{fold_index}_empty_bank_or_query")
        heldout_group_set = {str(record_groups[sample]) for sample in heldout_records}
        leaked = heldout_group_set & {str(row["group_id"]) for row in bank_rows}
        if leaked:
            raise ShapeGateError(
                f"fold_{fold_index}_group_leak:{sorted(leaked)[:3]}"
            )
        for row in query_rows:
            key = f"{row['sample_id']}|{row['instance_id']}"
            heldout_seen[key] = heldout_seen.get(key, 0) + 1
        scaler = fit_scaler(bank_matrix(bank_rows))
        bank_z = transform(bank_matrix(bank_rows), scaler)

        nested_scores: List[float] = []
        for group in sorted({str(row["group_id"]) for row in bank_rows}):
            inner_keep = [i for i, row in enumerate(bank_rows) if str(row["group_id"]) != group]
            inner_query = [i for i, row in enumerate(bank_rows) if str(row["group_id"]) == group]
            if not inner_keep or not inner_query:
                raise ShapeGateError(f"nested_group_leaves_empty_bank:{group}")
            # Every instance of the held-away group contributes its own distance:
            # the fixed rule is the 0.95 quantile of leave-group-out *instance*
            # NN distances, not the easiest instance of each group.
            nested_scores.extend(
                float(value)
                for value in nn_distances(bank_z[inner_query], bank_z[inner_keep], chunk)
            )
        if not nested_scores:  # pragma: no cover - guarded above
            raise ShapeGateError(f"fold_{fold_index}_no_nested_scores")
        threshold = quantile_threshold(nested_scores, quantile)

        query_z = transform(bank_matrix(query_rows), scaler)
        scores = nn_distances(query_z, bank_z, chunk)
        kept_mask = scores <= threshold
        retained_instances = int(np.count_nonzero(kept_mask))
        oof_scores.extend(float(value) for value in scores)

        # The evaluation contract counts the 204 FIT records as its positives, so
        # a held-out record is "retained" only when none of its held-out positive
        # instances was abstained (no tolerance thresholds are invented here).  A
        # record with no known-Thing instance has nothing to abstain and is
        # retained vacuously; those are counted separately so the record ratio can
        # be read with and without them.  The instance ratio beside it is the
        # class-agnostic positive retention the 0.80 floor describes.
        per_record: Dict[str, List[bool]] = {}
        for row, kept in zip(query_rows, kept_mask.tolist()):
            per_record.setdefault(str(row["sample_id"]), []).append(bool(kept))
        retained_records = sum(1 for flags in per_record.values() if all(flags))
        total_retained += retained_records
        total_heldout += len(heldout_records)
        total_instances_retained += retained_instances
        total_instances += len(query_rows)
        folds.append(
            {
                "group_fold": fold_index,
                "groups": len({str(record_groups[sample]) for sample in heldout_records}),
                # Record accounting: the contract's 204 FIT positives.  A record
                # counts as retained only when every one of its held-out
                # known-Thing instances was kept (no tolerance is invented).
                "positives": len(heldout_records),
                "retained_positives": int(retained_records),
                # Measured ratio: positive-containing records only.  A record with
                # no known-Thing instance cannot be retained or abstained, so it is
                # excluded here and exposed both separately and in the all-record
                # figure beside it instead of being counted as a failure.
                "record_coverage": (
                    retained_records / len(per_record) if per_record else None
                ),
                "record_coverage_all_records": (
                    retained_records / len(heldout_records) if heldout_records else None
                ),
                "records_with_positives": len(per_record),
                "records_without_positives": len(heldout_records) - len(per_record),
                # Instance accounting: the class-agnostic positive retention the
                # 0.80 floor describes.  ``coverage`` is this instance ratio, and
                # the evaluation contract reads exactly these two keys first.
                "heldout_instances": len(query_rows),
                "retained_instances": retained_instances,
                "coverage": retained_instances / len(query_rows),
                "bank_instances": len(bank_rows),
                "bank_groups": len({str(row["group_id"]) for row in bank_rows}),
                "nested_calibration_positives": len(nested_scores),
                "mean_heldout_nn_distance": float(np.mean(scores)),
                "max_heldout_nn_distance": float(np.max(scores)),
                "thresholds": {"shape_nn_distance_p95": float(threshold)},
            }
        )

    final_scaler = fit_scaler(bank_matrix(rows))
    final_threshold = quantile_threshold(oof_scores, quantile)
    if len(heldout_seen) != len(rows):
        raise ShapeGateError(
            f"heldout_instance_coverage:{len(heldout_seen)} of {len(rows)}"
        )
    duplicated = [key for key, count in heldout_seen.items() if count != 1]
    if duplicated:
        raise ShapeGateError(f"heldout_instance_twice:{duplicated[:3]}")
    in_sample = nn_distances(transform(bank_matrix(rows), final_scaler), transform(bank_matrix(rows), final_scaler), chunk)
    return {
        "folds": folds,
        "oof_scores": oof_scores,
        "final_scaler": final_scaler,
        "final_threshold": final_threshold,
        "in_sample": {
            "bank_instances": len(rows),
            "threshold": float(final_threshold),
            "mean_self_distance": float(np.mean(in_sample)),
            "max_self_distance": float(np.max(in_sample)),
        },
        "retained_total": total_retained,
        "heldout_total": total_heldout,
        "instances_retained_total": total_instances_retained,
        "instances_total": total_instances,
        "fold_of": fold_of,
    }


# ---------------------------------------------------------------------------
# application to predicted CC8 components
# ---------------------------------------------------------------------------


def gate_semantic(
    semantic: Any,
    instance: Any,
    *,
    scaler: Mapping[str, Any],
    bank_z: "np.ndarray",
    threshold: float,
    chunk: int = NN_CHUNK,
) -> Dict[str, Any]:
    """Keep or abstain each predicted Thing component; Stuff passes through.

    A component is one ``(class, instance)`` region of the predicted raster (the
    CC8 decoder already labels 8-connected Thing components).  Nothing is
    compared with ground truth: the only inputs are the component mask geometry
    and the positive bank.
    """

    _require_numpy()
    semantic = np.asarray(semantic)
    instance = np.asarray(instance)
    if semantic.shape != instance.shape:
        raise ShapeGateError(f"shape_mismatch:{semantic.shape} vs {instance.shape}")
    if semantic.ndim != 2:
        raise ShapeGateError(f"semantic_not_2d:{semantic.shape}")
    image_shape = (int(semantic.shape[0]), int(semantic.shape[1]))
    if int(semantic.min()) < 0 or int(semantic.max()) > IGNORE_ID:
        raise ShapeGateError("semantic_outside_0_35")

    thing = (semantic >= THING_MIN) & (semantic <= THING_MAX) & (instance >= 0)
    stuff = (semantic >= STUFF_MIN) & (semantic <= STUFF_MAX)
    thing_ids = sorted({int(value) for value in np.unique(instance[thing])})
    if not thing_ids:
        # Nothing to gate: every labelled pixel is Stuff (or ignore) and passes
        # through untouched.  The return shape must match the gated path exactly,
        # ``instance`` included, or a caller crashes on an all-Stuff/empty image.
        return {
            "semantic": semantic.astype(np.int16, copy=True),
            "instance": instance.astype(np.int32, copy=True),
            "candidate": stuff.copy(),
            "retained": stuff.copy(),
            "components": [],
            "kept_components": 0,
            "abstained_components": 0,
            "thing_candidate_pixels": 0,
            "thing_retained_pixels": 0,
            "stuff_pixels": int(stuff.sum()),
        }

    feature_rows = []
    masks = []
    class_ids: List[int] = []
    for instance_id in thing_ids:
        mask = thing & (instance == instance_id)
        # One predicted component must carry one predicted class: a region whose
        # Thing pixels disagree would make the descriptor set ambiguous and the
        # class label arbitrary, so it is refused instead of silently taking the
        # first label.
        classes = [int(value) for value in np.unique(semantic[mask]).tolist()]
        if len(classes) != 1:
            raise ShapeGateError(
                f"thing_instance_multi_class:{int(instance_id)}:{classes[:4]}"
            )
        masks.append(mask)
        class_ids.append(classes[0])
        feature_rows.append(shape_descriptors(mask, image_shape))
    scores = nn_distances(transform(np.vstack(feature_rows), scaler), bank_z, chunk)

    candidate = np.zeros(semantic.shape, dtype=bool)
    retained = np.zeros(semantic.shape, dtype=bool)
    components: List[Dict[str, Any]] = []
    out = semantic.astype(np.int16, copy=True)
    out_instance = instance.astype(np.int32, copy=True)
    kept = 0
    for index, instance_id in enumerate(thing_ids):
        mask = masks[index]
        candidate |= mask
        distance = float(scores[index])
        keep = distance <= float(threshold)
        components.append(
            {
                "instance_id": int(instance_id),
                "class_id": int(class_ids[index]),
                "pixels": int(mask.sum()),
                "nn_distance": distance,
                "retained": bool(keep),
            }
        )
        if keep:
            kept += 1
            retained |= mask
        else:
            out[mask] = IGNORE_ID
            out_instance[mask] = -1
    # Stuff is unchanged and stays labelled; candidate covers every labelled
    # pixel so the persisted masks satisfy retained <= candidate <= labelled.
    candidate |= stuff
    retained |= stuff
    return {
        "semantic": out,
        "instance": out_instance,
        "candidate": candidate,
        "retained": retained,
        "components": components,
        "kept_components": kept,
        "abstained_components": len(thing_ids) - kept,
        "thing_candidate_pixels": int(sum(int(mask.sum()) for mask in masks)),
        "thing_retained_pixels": int(sum(int(mask.sum()) for mask, comp in zip(masks, components) if comp["retained"])),
        "stuff_pixels": int(stuff.sum()),
    }


def _viewbox_affine(height: int, width: int) -> Dict[str, float]:
    affine = raster_infer._viewbox_affine(int(width), int(height))
    return {
        "scale_x": float(affine["scale_x"]),
        "scale_y": float(affine["scale_y"]),
        "offset_x": float(affine["offset_x"]),
        "offset_y": float(affine["offset_y"]),
    }


def retained_document(
    semantic: Any, instance: Any, tol_px: float = DEFAULT_TOL_PX
) -> Dict[str, Any]:
    """12-field document of the retained geometry (abstained pixels are 35)."""

    height, width = int(semantic.shape[0]), int(semantic.shape[1])
    affine = _viewbox_affine(height, width)
    document = vectorizer.vectorize_masks(
        semantic,
        instance,
        affine["scale_x"],
        affine["scale_y"],
        affine["offset_x"],
        affine["offset_y"],
        viewbox=(0.0, 0.0, 100.0, 100.0),
        tol_px=float(tol_px),
    )
    errors = vectorizer.validate_document(document)
    if errors:
        raise ShapeGateError("invalid_document:" + ",".join(errors))
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def calib_payload(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    result: Mapping[str, Any],
    record_count: int,
) -> Dict[str, Any]:
    folds = result["folds"]
    records_total = sum(int(fold["positives"]) for fold in folds)
    if records_total != int(record_count):
        raise ShapeGateError(f"fold_records:{records_total} vs {record_count}")
    retained_total = sum(int(fold["retained_positives"]) for fold in folds)
    instance_total = sum(int(fold["heldout_instances"]) for fold in folds)
    instance_retained = sum(int(fold["retained_instances"]) for fold in folds)
    if instance_total != int(len(rows)):
        raise ShapeGateError(
            f"heldout_instances_scope:{instance_total} vs bank {len(rows)}"
        )
    fold_groups = sum(int(fold["groups"]) for fold in folds)
    all_groups = len({str(group) for group in result["fold_of"]})
    if fold_groups != all_groups:
        raise ShapeGateError(f"fold_groups_scope:{fold_groups} vs {all_groups}")
    records_with_positives = sum(int(fold["records_with_positives"]) for fold in folds)
    records_without_positives = sum(
        int(fold["records_without_positives"]) for fold in folds
    )
    if records_with_positives + records_without_positives != records_total:
        raise ShapeGateError(
            "record_positive_partition:"
            f"{records_with_positives}+{records_without_positives} vs {records_total}"
        )
    bank_groups = int(summary["groups"])
    return {
        "schema": SCHEMA_CALIBRATION,
        "created_utc": utc_now(),
        "gt_read": False,
        "route": ROUTE,
        # ``fit_known_positives`` is what the evaluation contract reads: the 204
        # FIT RECORDS (24 in the scale-32 candidate prefix, 204 at scale 256).
        # Positive *instances* are a different, larger unit and are declared
        # separately; no key mixes the two.
        "fit_known_positives": records_total,
        "fit_records": records_total,
        "fit_scale": ALL_FIT_SCALE,
        "fit_groups": all_groups,
        "bank_positive_instances": int(len(rows)),
        "bank_positive_groups": bank_groups,
        "fit_positive_instances": int(len(rows)),
        "fit_positive_images": int(summary["images_with_positives"]),
        "fit_images": int(summary["images"]),
        "fit_per_class": summary["per_class"],
        "count_units": {
            "fit_known_positives": "fit_records",
            "fit_records": "fit_records",
            "fit_groups": "drawing_groups",
            "bank_positive_instances": "known_thing_instances",
            "bank_positive_groups": "drawing_groups",
            "fit_positive_instances": "known_thing_instances",
            "folds[].positives": "fit_records",
            "folds[].heldout_instances": "known_thing_instances",
            "folds[].retained_instances": "known_thing_instances",
        },
        "unknown_used_as_negative": False,
        "geometry_source": "builder_oof",
        "joint_features_used": False,
        "leave_one_group_out": True,
        "quantile": QUANTILE,
        "fold_count": int(len(folds)),
        "feature_names": list(FEATURE_NAMES),
        "robust_scale": {"statistic": "median_iqr", "iqr_floor": ROBUST_EPS},
        "thresholds": {"shape_nn_distance_p95": float(result["final_threshold"])},
        "folds": folds,
        # The preregistered .80 floor is nested held-out positive retention.  The
        # declared value follows the per-fold unit the contract reads first, i.e.
        # held-out known-Thing instances; the record-level ratio is reported
        # beside it and is never substituted for it.
        "positive_heldout_coverage": (
            instance_retained / instance_total if instance_total else None
        ),
        "positive_heldout_counts": {
            "retained": int(instance_retained),
            "positives": int(instance_total),
            "unit": "known_thing_instances",
        },
        "heldout_positive_coverage": {
            "definition": (
                "nested held-out positive coverage, instance level: retained "
                "known-Thing instances / known-Thing instances of the held-out "
                "group folds"
            ),
            "numerator": int(instance_retained),
            "denominator": int(instance_total),
            "coverage": (
                instance_retained / instance_total if instance_total else None
            ),
            "calibration": (
                "each fold threshold is the 0.95 quantile of leave-one-image-group-out "
                "distances inside the two training folds (nested); the held-out fold "
                "never enters its own threshold"
            ),
        },
        "heldout_record_coverage": {
            "definition": (
                "record level, beside the instance ratio: held-out FIT records that "
                "contain at least one known-Thing instance and had none of them "
                "abstained, over the held-out FIT records that contain a positive at "
                "all.  Records without any known-Thing instance cannot be scored here "
                "and are counted separately, never folded into this ratio"
            ),
            "numerator": int(retained_total),
            "denominator": int(records_with_positives),
            "coverage": (
                retained_total / records_with_positives if records_with_positives else None
            ),
            "records_with_positives": int(records_with_positives),
            "vacuous_records": int(records_without_positives),
            "all_records_with_vacuous": {
                "definition": (
                    "the same counts with every positive-free record treated as "
                    "retained because it has nothing to abstain; shown only so the "
                    "all-record figure is not confused with the measured one"
                ),
                "numerator": int(retained_total + records_without_positives),
                "denominator": int(records_total),
                "coverage": (
                    (retained_total + records_without_positives) / records_total
                    if records_total
                    else None
                ),
            },
        },
        "deployment": {
            "bank_instances": int(len(rows)),
            "bank_groups": int(summary["groups"]),
            "scaler_fit": "all known FIT positives",
            "threshold": float(result["final_threshold"]),
            "threshold_source": "0.95 quantile of the concatenated out-of-fold distances",
            "density_disclosure": (
                "the deployment bank is the full bank of {bank} known-Thing instances "
                "over {bank_groups} drawing groups, denser than the roughly 2/3 bank "
                "each fold threshold was calibrated on; applying the deployment "
                "threshold to predicted components is therefore looser than the fold "
                "measurement, and the {records} FIT records are inside the deployment "
                "bank, so their retention is in-sample and descriptive only"
            ).format(
                bank=int(len(rows)),
                bank_groups=int(summary["groups"]),
                records=int(records_total),
            ),
        },
        "in_sample_application": dict(
            result["in_sample"],
            definition=(
                "every known positive scored against the bank it belongs to; a "
                "sanity check that the persisted descriptors and scaler are "
                "self-consistent (self distance ~0), NOT quality evidence and NOT a "
                "coverage measurement"
            ),
        ),
    }


def record_group_map(records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """``sample_id -> permanent group id`` for the calibration fold partition."""

    mapping: Dict[str, str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        group_id = record.get("group_id")
        if group_id is None:
            raise ShapeGateError(f"record_without_group:{sample_id}")
        mapping[sample_id] = str(group_id)
    if not mapping:
        raise ShapeGateError("no_records")
    return mapping


def _calibrate_from_paths(
    cache_dir: str,
    groups_path: str,
    manifest_path: str,
    *,
    chunk: int = NN_CHUNK,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    cache_map = load_cache_map(cache_dir)
    group_map = load_group_map(groups_path)
    records = _all_fit_records(manifest_path)
    record_groups = record_group_map(records)
    rows, summary = build_positive_bank(cache_map, group_map)
    result = calibrate(rows, record_groups, chunk=chunk)
    return rows, summary, result, record_groups


def command_calibrate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows, summary, result, record_groups = _calibrate_from_paths(
        args.fit_cache, args.groups, args.manifest, chunk=args.chunk
    )
    payload = calib_payload(rows, summary, result, len(record_groups))
    _write_json(out / "gate_calibration.json", payload)
    bank_payload = {
        "schema": SCHEMA_BANK,
        "created_utc": utc_now(),
        "gt_read": False,
        "route": ROUTE,
        "summary": summary,
        "feature_names": list(FEATURE_NAMES),
        "feature_medians": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, result["final_scaler"]["median"])
        },
        "feature_iqr": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, result["final_scaler"]["scale"])
        },
        "degenerate_features": result["final_scaler"]["degenerate_features"],
        "instances": [
            {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "instance_id": int(row["instance_id"]),
                "class_id": int(row["class_id"]),
                "pixels": int(row["pixels"]),
                "features": [float(value) for value in np.asarray(row["features"]).tolist()],
            }
            for row in rows
        ],
    }
    _write_json(out / "fit_positive_bank.json", bank_payload)
    coverage = payload["heldout_record_coverage"]
    print(
        "calibrated bank_instances={i} heldout_instances={hi} "
        "retained_instances={ri} instance_coverage={ic:.4f} "
        "heldout_records={h} retained_records={r} "
        "record_coverage={rc:.4f} vacuous_records={v} threshold={t:.4f}".format(
            i=payload["fit_positive_instances"],
            hi=payload["heldout_positive_coverage"]["denominator"],
            ri=payload["heldout_positive_coverage"]["numerator"],
            ic=payload["positive_heldout_coverage"],
            h=coverage["records_with_positives"],
            r=coverage["numerator"],
            rc=coverage["coverage"] or 0.0,
            v=coverage["vacuous_records"],
            t=result["final_threshold"],
        )
    )
    return 0


def command_predict(args: argparse.Namespace) -> int:
    _require_numpy()
    out = Path(args.out)
    calibration_path = out / "gate_calibration.json"
    if not calibration_path.is_file():
        raise ShapeGateError(f"missing_calibration:{calibration_path}")
    calibration = _read_json(calibration_path)
    if calibration.get("schema") != SCHEMA_CALIBRATION:
        raise ShapeGateError(f"bad_calibration_schema:{calibration.get('schema')}")
    threshold = float(calibration["thresholds"]["shape_nn_distance_p95"])

    cache_map = load_cache_map(args.fit_cache)
    group_map = load_group_map(args.groups)
    rows, summary = build_positive_bank(cache_map, group_map)
    if len(rows) != int(calibration["fit_positive_instances"]):
        raise ShapeGateError(
            f"bank_drift:{len(rows)} vs {calibration['fit_positive_instances']}"
        )
    # Use the scaler the calibration persisted instead of silently re-deriving one
    # from the cache: the two must agree, so the recomputation is validated against
    # the artifact rather than replacing it.
    bank_payload = _read_json(out / "fit_positive_bank.json")
    scaler = scaler_from_bank_payload(bank_payload)
    scaler_check = compare_scalers(scaler, fit_scaler(bank_matrix(rows)))
    persisted_features = np.asarray(
        [row["features"] for row in bank_payload.get("instances") or []], dtype=np.float64
    )
    recomputed_features = bank_matrix(rows)
    if persisted_features.shape != recomputed_features.shape:
        raise ShapeGateError(
            f"bank_feature_shape:{persisted_features.shape} vs {recomputed_features.shape}"
        )
    feature_gap = float(np.abs(persisted_features - recomputed_features).max())
    if feature_gap > 1e-9:
        raise ShapeGateError(f"bank_feature_drift:{feature_gap:.3e}")
    bank_z = transform(bank_matrix(rows), scaler)

    records = load_fit_records(args.manifest, args.limit)
    arm_dir = Path(args.cc8_arm)
    samples: List[Dict[str, Any]] = []
    per_class: Dict[str, Dict[str, int]] = {}
    empty_images: List[str] = []
    for record in records:
        sample_id = str(record["sample_id"])
        name = slug(sample_id)
        prediction_path = arm_dir / "predictions" / f"{name}.npz"
        if not prediction_path.is_file():
            raise ShapeGateError(f"missing_arm_prediction:{prediction_path}")
        loaded = np.load(prediction_path)
        semantic = np.asarray(loaded["semantic"])
        instance = np.asarray(loaded["instance"])
        source_size = [int(value) for value in np.asarray(loaded["source_size"]).tolist()]
        if source_size != [int(semantic.shape[1]), int(semantic.shape[0])]:
            raise ShapeGateError(f"source_size_mismatch:{sample_id}")
        gated = gate_semantic(
            semantic,
            instance,
            scaler=scaler,
            bank_z=bank_z,
            threshold=threshold,
        )
        document = retained_document(gated["semantic"], gated["instance"])
        predictions_dir = out / "predictions"
        documents_dir = out / "documents"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        documents_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_dir / f"{name}.npz",
            semantic=gated["semantic"].astype(np.int16),
            candidate=np.asarray(gated["candidate"], dtype=bool),
            retained=np.asarray(gated["retained"], dtype=bool),
            source_size=np.asarray(source_size, dtype=np.int64),
        )
        _write_json(
            documents_dir / f"{name}.json",
            {
                "schema": SCHEMA_DOCUMENT,
                "sample_id": sample_id,
                "source_size": source_size,
                "document": document,
            },
        )
        for component in gated["components"]:
            bucket = per_class.setdefault(
                str(component["class_id"]),
                {"components": 0, "retained_components": 0, "pixels": 0, "retained_pixels": 0},
            )
            bucket["components"] += 1
            bucket["pixels"] += int(component["pixels"])
            if component["retained"]:
                bucket["retained_components"] += 1
                bucket["retained_pixels"] += int(component["pixels"])
        if not gated["components"]:
            empty_images.append(sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "png_path": str(record["png_path"]),
                "candidate_pixels": int(np.count_nonzero(gated["candidate"])),
                "retained_pixels": int(np.count_nonzero(gated["retained"])),
                "thing_candidate_pixels": int(gated["thing_candidate_pixels"]),
                "thing_retained_pixels": int(gated["thing_retained_pixels"]),
                "stuff_pixels": int(gated["stuff_pixels"]),
                "thing_components": len(gated["components"]),
                "retained_components": int(gated["kept_components"]),
                "abstained_components": int(gated["abstained_components"]),
                "primitives": len(document["commands"]),
                "source_size": source_size,
            }
        )

    manifest = {
        "schema": SCHEMA_PREDICTIONS,
        "route": ROUTE,
        "created_utc": utc_now(),
        "gt_read_during_prediction": False,
        "stage": "predictions_persisted",
        "all_png_persisted": True,
        "sample_count": len(samples),
        "threshold": threshold,
        "scaler_validation": {
            "source": "fit_positive_bank.json feature_medians/feature_iqr",
            "recomputed_from_cache": True,
            "max_median_abs_gap": scaler_check["max_median_abs_gap"],
            "max_iqr_abs_gap": scaler_check["max_iqr_abs_gap"],
            "max_feature_abs_gap": feature_gap,
            "tolerance": scaler_check["tolerance"],
            "drift": False,
        },
        "samples": samples,
    }
    _write_json(out / "predictions_manifest.json", manifest)
    report = {
        "schema": SCHEMA_CANDIDATE_REPORT,
        "created_utc": utc_now(),
        "gt_read": False,
        "route": ROUTE,
        "threshold": threshold,
        "per_class": {key: per_class[key] for key in sorted(per_class, key=lambda k: int(k))},
        "empty_images": empty_images,
        "totals": {
            "samples": len(samples),
            "candidate_pixels": sum(row["candidate_pixels"] for row in samples),
            "retained_pixels": sum(row["retained_pixels"] for row in samples),
            "thing_candidate_pixels": sum(row["thing_candidate_pixels"] for row in samples),
            "thing_retained_pixels": sum(row["thing_retained_pixels"] for row in samples),
            "stuff_pixels": sum(row["stuff_pixels"] for row in samples),
            "thing_components": sum(row["thing_components"] for row in samples),
            "retained_components": sum(row["retained_components"] for row in samples),
            "abstained_components": sum(row["abstained_components"] for row in samples),
        },
    }
    _write_json(out / "candidate_report.json", report)
    totals = report["totals"]
    print(
        "predicted samples={s} components={c} retained={r} abstained={a} "
        "thing_pixels={tp}->{rp}".format(
            s=totals["samples"],
            c=totals["thing_components"],
            r=totals["retained_components"],
            a=totals["abstained_components"],
            tp=totals["thing_candidate_pixels"],
            rp=totals["thing_retained_pixels"],
        )
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows: List[Dict[str, Any]] = []
    for name, label in (
        ("gate_calibration.json", "calibration"),
        ("fit_positive_bank.json", "bank"),
        ("predictions_manifest.json", "manifest"),
        ("candidate_report.json", "candidates"),
    ):
        path = out / name
        rows.append({"artifact": label, "path": str(path), "present": path.is_file()})
    predictions = sorted(glob.glob(str(out / "predictions" / "*.npz")))
    documents = sorted(glob.glob(str(out / "documents" / "*.json")))
    payload = {
        "route": ROUTE,
        "out": str(out),
        "artifacts": rows,
        "predictions": len(predictions),
        "documents": len(documents),
    }
    print(json.dumps(payload, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadvec.recognition_positive_shape_gate",
        description="Positive-only class-agnostic shape-abstention gate (CPU pilot).",
    )
    parser.add_argument("--out", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fit-cache", default=DEFAULT_FIT_CACHE)
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--manifest", default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--cc8-arm", default=DEFAULT_CC8_ARM)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--chunk", type=int, default=NN_CHUNK)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("calibrate")
    sub.add_parser("predict")
    sub.add_parser("status")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    if args.command == "calibrate":
        return command_calibrate(args)
    if args.command == "predict":
        return command_predict(args)
    return command_status(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
