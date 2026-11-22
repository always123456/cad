"""Frozen-semantic instance **offset** lane: per-pixel vectors to the centroid.

This is an *additive* pilot lane and an independent competitor of
``recognition_instance_affinity`` (which merges whole connected components).
Nothing here changes the semantic model, the semantic loss, the rasterizer, the
decoder of any other lane or the shared cache:

* the pure-fit semantic checkpoint is loaded read-only, every parameter is
  frozen and the wrapper is always run in ``eval`` mode (batch norm uses
  running stats), exactly like the affinity lane;
* a small ``Conv1x1(256 -> 64) -> ReLU -> Conv1x1(64 -> 2)`` head sits on the
  ``net.classifier[3]`` ReLU output (stride 8, 256 channels) and is the only
  trainable tensor (16 578 trainable parameters, measured);
* supervision is one 2-D vector per annotated *Thing instance pixel* pointing
  at the centroid of its own ground-truth instance, expressed in pixels divided
  by :data:`OFFSET_NORM` (256).  The centroid is always the **full native**
  centroid of that ``(class, instance)`` group, computed on the whole cached
  record and then translated into the crop frame, so a truncated instance whose
  cropped centroid is biased still has an honest target.  SmoothL1 (beta
  ``2/256``) is applied on ``mask`` pixels only and each crop is weighted by its
  supervised pixel mass;
* unknown pixels are never negatives: the mask is ``instance_valid &
  instance >= 0 & semantic_valid & 0 <= semantic < 30`` and the two validity
  masks stay independent (``instance_valid`` and ``semantic_valid`` are ANDed,
  never derived from one another);
* decoding is *not* whole-component merging.  Every predicted Thing pixel casts
  one vote at ``(row + off_row * 256, col + off_col * 256)``; votes are grouped
  per predicted class with the pre-registered, non-swept schedule
  :data:`VOTE_BIN_PX` = 8 px bins, then :data:`MEANSHIFT_ITERATIONS` = 3
  mean-shift iterations with :data:`MEANSHIFT_RADIUS_PX` = 16 px, centres closer
  than :data:`VOTE_MERGE_PX` are merged, and every pixel is assigned to the
  nearest surviving centre.  Clusters thinner than :data:`MIN_CLUSTER_PIXELS`
  are dissolved and their pixels fall back to their **baseline connected
  component** id (no pixel is ever dropped and no pixel is ever relabelled), so
  a single pixel can never become a one-pixel instance;
* Stuff (30..34) and unknown keep instance ``-1`` and their pixel classes stay
  bit-identical; ids are globally unique across classes because they are
  assigned in one deterministic sweep over ``(class, cluster)``.

The cache contract is the reviewed full-fit cache (``image`` uint8,
``semantic`` class indices ``0..34`` with ``-1`` unknown, ``semantic_valid``,
``instance`` raw local ids with ``-1`` unknown, ``instance_valid``).  There is no
independent ``instance_class`` array, so the class source is the semantic map;
that is documented rather than papered over.

Native tiling: a head output is already *relative to its own pixel*, so the
global vote of a tile pixel is ``pixel_global + offset * 256``.  Overlapping
tiles average the **offsets** (origin independent) and never add a tile origin a
second time.  This lane never reads ground truth at inference and never reads
SVG styles anywhere.

Units: :class:`OffsetEmbedder` produces head **units** (pixels / 256), which is
what :func:`decode_offsets` consumes and scales exactly once; ``field["pixels"]``
is already ``units * 256`` and must never be handed to the decoder.

CLI::

    python -m cadvec.recognition_instance_offsets prepare --help
    python -m cadvec.recognition_instance_offsets smoke   --help
    python -m cadvec.recognition_instance_offsets fit     --help   # parent-scheduled
    python -m cadvec.recognition_instance_offsets guard   --help   # parent-scheduled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - environment dependent
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - environment dependent
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from . import recognition_instance_affinity as affinity
from . import recognition_pretrained_fullfit as fullfit
from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_infer as raster_infer
from . import recognition_raster_model as raster_model
from . import recognition_raster_train as train

OFFSETS_SCHEMA = "cadvec.recognition.instance_offsets/1"
PREPARE_SCHEMA = "cadvec.recognition.instance_offsets_prepare/1"
FIT_SCHEMA = "cadvec.recognition.instance_offsets_fit/1"
SMOKE_SCHEMA = "cadvec.recognition.instance_offsets_smoke/1"
CHECKPOINT_SCHEMA = "cadvec.recognition.instance_offsets_checkpoint/1"
GUARD_SCHEMA = "cadvec.recognition.instance_offsets_guard/1"
DECODE_STATS_SCHEMA = "cadvec.recognition.instance_offsets_decode_stats/1"
# Stable decode-statistic key set (see _decode_stats); consumers rely on it.
DECODE_STATS_KEYS = (
    "schema",
    "class",
    "vote_bin_px",
    "meanshift_iterations",
    "meanshift_radius_px",
    "vote_merge_px",
    "min_cluster_pixels",
    "offset_norm",
    "instance_id_base",
    "ground_truth_used",
    "thing_pixels",
    "assigned_thing_pixels",
    "pixels_deleted",
    "clusters",
    "dissolved_clusters",
    "dissolved_pixels",
    "fallback_groups",
    "fallback_pixels",
    "split_components",
    "max_component_split",
    "seeds_total",
    "votes_total",
    "stuff_pixels",
    "unknown_pixels",
    "pixel_classes_bitidentical",
    "stuff_instance_unchanged",
    "unknown_instance_unchanged",
    "ids_unique",
    "id_class_collision",
    "tile_rows",
    "tile_columns",
    "per_class",
)
OFFSETS_VERSION = 1

# Feature geometry (fixed by the frozen semantic model).
HEAD_IN_CHANNELS = affinity.HEAD_IN_CHANNELS  # 256
HEAD_HIDDEN_CHANNELS = 64
OFFSET_CHANNELS = 2
FEATURE_STRIDE = affinity.FEATURE_STRIDE  # 8
NUM_CLASSES = pilot.NUM_CLASSES  # 35
NUM_THING_CLASSES = raster_model.NUM_THING_CLASSES  # 30
IGNORE_CLASS = pilot.NUM_CLASSES  # 35, raw "ignore" sentinel
KEY_STRIDE = 1 << 15  # (class << 15) | instance

# Crops / tiling: mirrored from the frozen full-fit lane.
TILE_SIZE = fullfit.TILE_SIZE  # 256
TILE_STRIDE = fullfit.TILE_STRIDE  # 128
CROP_SIZE = fullfit.CROP_SIZE  # 256
CROPS_PER_RECORD = fullfit.CROPS_PER_RECORD  # 8
EPOCHS = fullfit.EPOCHS  # 5
ACCUMULATION_STEPS = fullfit.ACCUMULATION_STEPS  # 8
BASE_SEED = fullfit.BASE_SEED  # 20261108
EPOCH_SEED_STRIDE = fullfit.EPOCH_SEED_STRIDE  # 1000003
HEAD_INIT_SEED = 20261108

# Offset geometry.
OFFSET_NORM = float(CROP_SIZE)  # head units are pixels / 256
SMOOTHL1_BETA = 2.0 / OFFSET_NORM  # 2 px transition, in head units
MIN_GROUP_PIXELS = 8  # a crop group needs this many pixels to be supervised
CROPS_WITH_SUPERVISION_MIN = 1

# Training.
LR = 1e-3
WEIGHT_DECAY = 0.0
MIN_ELIGIBLE_CROPS_PER_EPOCH = 200

# Pre-registered, non-swept vote-clustering schedule.  These are settled
# defaults, not knobs: 8 px bins, 3 mean-shift iterations at a 16 px radius,
# 8 px merge, and a 16 px minimum cluster.  Nothing here is tuned or swept.
VOTE_BIN_PX = 8.0
MEANSHIFT_ITERATIONS = 3
MEANSHIFT_RADIUS_PX = 16.0
VOTE_MERGE_PX = 8.0
MIN_CLUSTER_PIXELS = 16
CENTER_CHUNK = 64
# Point-block size for the streamed distance reductions: both the point axis and
# the centre axis are blocked, so the largest temporary is
# POINT_CHUNK x CENTER_CHUNK x 2 float64 (8 MiB here), never N x centres.
POINT_CHUNK = 8192

# GPU guard: memory that must remain free *after* the measured accumulation peak
# (equivalently free_before >= peak + 1024 MiB), so a neighbouring job still fits.
RESERVE_AFTER_PEAK_MIB = 1024.0

DEFAULT_MANIFEST = os.path.join(
    "outputs", "recognition-gate3b-fit-tune-20261108-01", "recognition_public_manifest.json"
)
DEFAULT_CACHE_DIR = os.path.join(
    "outputs", "recognition-pretrained-fullfit-20261111-01", "cache"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    "outputs", "recognition-instance-offsets-20261111-01"
)
DEFAULT_SEMANTIC_CHECKPOINT = os.path.join(
    "outputs", "recognition-fullfit-e30-20261111-01", "checkpoint_fullfit.pt"
)


class InstanceOffsetsError(pilot.PilotError):
    """Raised for contract violations of the instance-offset lane."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover - environment dependent
        raise InstanceOffsetsError("numpy is required for the instance-offset lane")
    return np


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - environment dependent
        raise InstanceOffsetsError("torch is required for the instance-offset lane")
    return torch


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InstanceOffsetsError(f"{name} must be an int; got {value!r}")
    if value < minimum:
        raise InstanceOffsetsError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InstanceOffsetsError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise InstanceOffsetsError(f"{name} must be finite; got {value!r}")
    return number


def offsets_to_pixels(units: Any) -> Any:
    """Head units (pixels / 256) -> native pixels."""

    return units * OFFSET_NORM


def pixels_to_offsets(pixels: Any) -> Any:
    """Native pixels -> head units (pixels / 256)."""

    return pixels / OFFSET_NORM


# ---------------------------------------------------------------------------
# frozen semantic wrapper + offset head
# ---------------------------------------------------------------------------


def semantic_parameter_digest(
    model: Any,
) -> Tuple[Tuple[str, Tuple[int, ...], float, float], ...]:
    """Re-exported from the affinity lane: same frozen-reference digest."""

    return affinity.semantic_parameter_digest(model)


def freeze_semantic_model(model: Any) -> Dict[str, Any]:
    """Re-exported from the affinity lane: every semantic parameter frozen."""

    return affinity.freeze_semantic_model(model)


def build_offset_head(
    *,
    in_channels: int = HEAD_IN_CHANNELS,
    hidden_channels: int = HEAD_HIDDEN_CHANNELS,
    out_channels: int = OFFSET_CHANNELS,
    seed: int = HEAD_INIT_SEED,
    device: str = "cpu",
) -> Any:
    """Deterministic ``Conv1x1(256->64) -> ReLU -> Conv1x1(64->2)`` head.

    Initialisation uses an explicit ``torch.Generator`` seeded with ``seed``, so
    the head never depends on the global RNG state.  The fit lane therefore sets
    the epoch seed *after* the head exists without changing a single weight, and
    a re-run always starts from the same head.
    """

    torch = _require_torch()
    in_channels = _positive_int("in_channels", in_channels)
    hidden_channels = _positive_int("hidden_channels", hidden_channels)
    out_channels = _positive_int("out_channels", out_channels)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    hidden = torch.nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1)
    output = torch.nn.Conv2d(int(hidden_channels), int(out_channels), kernel_size=1)
    for layer in (hidden, output):
        fan_in = float(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1])
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound, generator=generator)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound, generator=generator)
    module = torch.nn.Sequential(
        hidden, torch.nn.ReLU(inplace=False), output
    )
    module.to(device)
    return module


class OffsetEmbedder(affinity.PilotEmbedder):
    """Affinity's read-only feature tap with the offset head in front.

    ``feature_map`` / ``crop_embeddings`` are inherited unchanged: the semantic
    wrapper runs under ``no_grad``, the head runs outside it, and the head output
    is upsampled bilinearly to the crop resolution with ``align_corners=False``
    (the very interpolation the semantic wrapper uses).  Only ``dim`` changes:
    it is :data:`OFFSET_CHANNELS`, so ``crop_embeddings`` returns ``B x 2 x H x W``
    offset vectors in head units.
    """

    def __init__(
        self,
        model: Any,
        *,
        head: Any = None,
        device: str = "cpu",
        seed: int = HEAD_INIT_SEED,
    ) -> None:
        super().__init__(
            model,
            dim=OFFSET_CHANNELS,
            in_channels=HEAD_IN_CHANNELS,
            head=head if head is not None else build_offset_head(seed=seed, device=device),
            device=device,
        )

    # -- tiled native offsets -------------------------------------------
    def native_offsets(
        self,
        image: Any,
        *,
        tile_size: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
    ) -> Dict[str, Any]:
        """Tiled offset field at source resolution, overlap averaged.

        Mirrors :meth:`PilotEmbedder.native_embeddings` geometry (same tile
        positions, white padding to bottom/right, same weight average), but the
        accumulated quantity is the head output *in head units*.  Because a head
        output is relative to its own pixel, averaging is origin independent; the
        decoder adds the global pixel index once, and the tile origin is never
        added to the offset itself.
        """

        torch = _require_torch()
        numpy = _require_numpy()
        if isinstance(image, torch.Tensor):
            source = image.detach().to("cpu", dtype=torch.float32)
        else:
            source = torch.from_numpy(
                numpy.ascontiguousarray(numpy.asarray(image, dtype=numpy.float32))
            )
        if source.ndim != 3 or int(source.shape[0]) != 3:
            raise InstanceOffsetsError(
                f"image must be 3 x H x W (unit range); got {tuple(source.shape)}"
            )
        if not bool(torch.isfinite(source).all()):
            raise InstanceOffsetsError("non_finite_image")
        height, width = int(source.shape[-2]), int(source.shape[-1])
        if height <= 0 or width <= 0:
            raise InstanceOffsetsError("empty image")
        tile_size = _positive_int("tile_size", tile_size)
        stride = _positive_int("stride", stride)
        if stride > tile_size:
            raise InstanceOffsetsError("stride must not exceed tile size")
        padded_height = max(int(tile_size), height)
        padded_width = max(int(tile_size), width)
        padded = source
        if padded_height != height or padded_width != width:
            padded = F.pad(
                source.unsqueeze(0),
                (0, padded_width - width, 0, padded_height - height),
                mode="constant",
                value=1.0,
            ).squeeze(0)
        rows = affinity._tile_positions(padded_height, tile_size, stride)
        columns = affinity._tile_positions(padded_width, tile_size, stride)

        self.model.eval()
        accum: Optional[Any] = None
        weights: Optional[Any] = None
        with torch.no_grad():
            for row in rows:
                for column in columns:
                    window = padded[
                        :, row : row + tile_size, column : column + tile_size
                    ]
                    features = self.feature_map(window.unsqueeze(0).to(self.device))
                    tile = F.interpolate(
                        self.head(features["features"]),
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    ).detach().to("cpu", dtype=torch.float32)
                    if accum is None:
                        accum = torch.zeros(
                            (OFFSET_CHANNELS, padded_height, padded_width),
                            dtype=torch.float32,
                        )
                        weights = torch.zeros(
                            (1, padded_height, padded_width), dtype=torch.float32
                        )
                    accum[:,
                          row : row + tile_size,
                          column : column + tile_size] += tile[0]
                    weights[:,
                            row : row + tile_size,
                            column : column + tile_size] += 1.0
        assert accum is not None and weights is not None
        units = (accum / weights)[:, :height, :width].contiguous().numpy()
        units = units.astype(numpy.float32)
        return {
            "units": units,
            "pixels": (units * numpy.float32(OFFSET_NORM)).astype(numpy.float32),
            "tile_rows": len(rows),
            "tile_columns": len(columns),
            "tile_size": int(tile_size),
            "stride": int(stride),
        }


def load_embedder(
    semantic_checkpoint: str,
    *,
    head_checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[OffsetEmbedder, Dict[str, Any]]:
    """Frozen semantic model + (optionally) a strictly loaded offset head."""

    model, payload = pilot.load_checkpoint_model(semantic_checkpoint, device=device)
    embedder = OffsetEmbedder(model, device=device)
    report: Dict[str, Any] = {
        "semantic_checkpoint": os.path.abspath(semantic_checkpoint),
        "semantic_schema": payload.get("schema"),
        "semantic_epoch": payload.get("epoch"),
        "semantic_digest": _digest_hex(embedder.model),
        "frozen": embedder.frozen_report,
        "head_loaded": False,
    }
    if head_checkpoint is not None:
        head_payload = load_offset_checkpoint(head_checkpoint, device=device)
        _assert_semantic_reference(head_payload, semantic_checkpoint)
        recorded = str(head_payload.get("semantic", {}).get("digest") or "")
        if recorded and recorded != report["semantic_digest"]:
            raise InstanceOffsetsError(
                "semantic_digest_mismatch: the head was trained on a different "
                "frozen semantic checkpoint"
            )
        embedder.head.load_state_dict(head_payload["head_state_dict"], strict=True)
        for parameter in embedder.head.parameters():
            parameter.requires_grad_(True)
        embedder.head.to(device)
        report["head_loaded"] = True
        report["head_checkpoint"] = os.path.abspath(head_checkpoint)
        report["head_epoch"] = head_payload.get("epoch")
    return embedder, report


def _digest_hex(model: Any) -> str:
    digest = semantic_parameter_digest(model)
    material = "|".join(
        f"{name}:{shape}:{total!r}:{absolute!r}"
        for name, shape, total, absolute in digest
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# crop windows with explicit native origin (mirror of SupportCropSampler)
# ---------------------------------------------------------------------------


def extract_window(
    entry: Mapping[str, Any], center: Sequence[int], *, size: int = CROP_SIZE
) -> Dict[str, Any]:
    """Cut one window *and report its native origin*.

    Byte-for-byte mirror of ``train.SupportCropSampler._extract`` (image padding
    1.0, labels ``-1``, masks ``False`` outside the source rectangle); the only
    addition is ``top`` / ``left``, which is what the offset targets need to
    translate a native centroid into the crop frame.
    """

    numpy = _require_numpy()
    image = numpy.asarray(entry["image"], dtype=numpy.float32)
    semantic = numpy.asarray(entry["semantic"], dtype=numpy.int16)
    semantic_valid = numpy.asarray(entry["semantic_valid"], dtype=bool)
    instance = numpy.asarray(entry["instance"], dtype=numpy.int16)
    instance_valid = numpy.asarray(entry["instance_valid"], dtype=bool)
    height, width = int(semantic.shape[0]), int(semantic.shape[1])
    size = _positive_int("size", int(size))
    half = size // 2
    top = int(center[0]) - half
    left = int(center[1]) - half
    bottom, right = top + size, left + size
    src_top, src_left = max(0, top), max(0, left)
    src_bottom, src_right = min(height, bottom), min(width, right)
    pad_top, pad_left = src_top - top, src_left - left

    out_image = numpy.full(
        (int(image.shape[0]), size, size), train.CROP_IMAGE_PAD_VALUE, dtype=numpy.float32
    )
    out_semantic = numpy.full((size, size), -1, dtype=numpy.int16)
    out_semantic_valid = numpy.zeros((size, size), dtype=bool)
    out_instance = numpy.full((size, size), -1, dtype=numpy.int16)
    out_instance_valid = numpy.zeros((size, size), dtype=bool)
    window_height = max(0, src_bottom - src_top)
    window_width = max(0, src_right - src_left)
    if window_height and window_width:
        target = (
            slice(pad_top, pad_top + window_height),
            slice(pad_left, pad_left + window_width),
        )
        source = (slice(src_top, src_bottom), slice(src_left, src_right))
        out_image[:, target[0], target[1]] = image[:, source[0], source[1]]
        out_semantic[target] = semantic[source]
        out_semantic_valid[target] = semantic_valid[source]
        out_instance[target] = instance[source]
        out_instance_valid[target] = instance_valid[source]
    padded = (src_top, src_left, src_bottom, src_right) != (top, left, bottom, right)
    return {
        "image": out_image,
        "semantic": out_semantic,
        "semantic_valid": out_semantic_valid,
        "instance": out_instance,
        "instance_valid": out_instance_valid,
        "sample_id": str(entry.get("sample_id") or ""),
        "source_width": int(entry.get("source_width") or width),
        "source_height": int(entry.get("source_height") or height),
        "top": int(top),
        "left": int(left),
        "padded": bool(padded),
        "valid_semantic_pixels": int(out_semantic_valid.sum()),
    }


def select_window(
    sampler: Any, entry: Mapping[str, Any], anchor: Mapping[str, Any]
) -> Dict[str, Any]:
    """Reproduce ``SupportCropSampler.crop``'s candidate loop with origins.

    The sampler keeps the best window and stops at the first candidate that
    reaches ``min_valid``; this function replays exactly that decision and
    returns the window *plus* ``top``/``left``.  ``crop`` is still the
    authoritative extraction: callers compare the two (see
    :func:`crop_offset_targets`'s ``window`` argument) and any mismatch raises
    :class:`InstanceOffsetsError`.
    """

    size = int(getattr(sampler, "crop_size", CROP_SIZE))
    min_valid = int(getattr(sampler, "min_valid", 0))
    best: Optional[Dict[str, Any]] = None
    best_valid = -1
    for center in anchor["candidates"]:
        window = extract_window(entry, center, size=size)
        valid = int(window["valid_semantic_pixels"])
        if valid > best_valid:
            best, best_valid = window, valid
        if valid >= min_valid:
            break
    if best is None:  # pragma: no cover - candidates is never empty
        raise InstanceOffsetsError("anchor has no candidate window")
    return best


def _assert_window_matches(authoritative: Mapping[str, Any], mirrored: Mapping[str, Any]) -> None:
    numpy = _require_numpy()
    for key in ("image", "semantic", "semantic_valid", "instance", "instance_valid"):
        if not numpy.array_equal(
            numpy.asarray(authoritative[key]), numpy.asarray(mirrored[key])
        ):
            raise InstanceOffsetsError(
                f"window_mirror_mismatch: {key} differs from the sampler crop"
            )


# ---------------------------------------------------------------------------
# ground-truth native centroids and crop-frame offset targets
# ---------------------------------------------------------------------------


def native_group_centroids(
    instance: Any,
    instance_valid: Any,
    semantic: Any,
    semantic_valid: Any,
    *,
    num_thing_classes: int = NUM_THING_CLASSES,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Native centroid per ``(class, instance)`` of one whole cached record.

    Computed on the **full** map, before any cropping, so truncation can never
    bias a target.  A group participates only when ``instance_valid`` says so
    and the class is a known Thing; ``-1`` ids are ignored and ``0`` is legal.
    """

    numpy = _require_numpy()
    inst = numpy.asarray(instance, dtype=numpy.int64)
    iv = numpy.asarray(instance_valid, dtype=bool)
    sem = numpy.asarray(semantic, dtype=numpy.int64)
    sv = numpy.asarray(semantic_valid, dtype=bool)
    if inst.shape != iv.shape or inst.shape != sem.shape or inst.shape != sv.shape:
        raise InstanceOffsetsError("instance/semantic arrays must share a shape")
    mask = (
        iv
        & (inst >= 0)
        & sv
        & (sem >= 0)
        & (sem < int(num_thing_classes))
    )
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not bool(mask.any()):
        return result
    keys = sem[mask] * numpy.int64(KEY_STRIDE) + inst[mask]
    ys, xs = numpy.nonzero(mask)
    unique, inverse, counts = numpy.unique(keys, return_inverse=True, return_counts=True)
    sum_y = numpy.bincount(inverse, weights=ys.astype(numpy.float64), minlength=unique.size)
    sum_x = numpy.bincount(inverse, weights=xs.astype(numpy.float64), minlength=unique.size)
    for position, key in enumerate(unique.tolist()):
        cls = int(key // KEY_STRIDE)
        identifier = int(key % KEY_STRIDE)
        result[(cls, identifier)] = {
            "class": cls,
            "instance": identifier,
            "pixels": int(counts[position]),
            "row": float(sum_y[position] / counts[position]),
            "column": float(sum_x[position] / counts[position]),
        }
    return result


def crop_offset_targets(
    window: Mapping[str, Any],
    centroids: Mapping[Tuple[int, int], Mapping[str, Any]],
    *,
    authoritative: Optional[Mapping[str, Any]] = None,
    min_group_pixels: int = MIN_GROUP_PIXELS,
    norm: float = OFFSET_NORM,
    num_thing_classes: int = NUM_THING_CLASSES,
) -> Dict[str, Any]:
    """Offset targets for one crop window (crop frame, head units).

    ``offsets[0]`` is the row offset, ``offsets[1]`` the column offset, both in
    ``pixel / 256``.  ``mask`` is the supervision mask and never derived from
    anything but the independent validities and the known Thing class:

        ``instance_valid & instance >= 0 & semantic_valid & 0 <= semantic < 30``

    and, per crop, a group must own at least ``min_group_pixels`` pixels inside
    the window.  The target itself is the **native** centroid of that group
    translated into the crop frame (``row - top``, ``column - left``), so a
    truncated instance is still aimed at its true centre.
    """

    numpy = _require_numpy()
    if authoritative is not None:
        _assert_window_matches(authoritative, window)
    size = _positive_int("crop size", int(window["semantic"].shape[0]))
    top = int(window.get("top", 0))
    left = int(window.get("left", 0))
    inst = numpy.asarray(window["instance"], dtype=numpy.int64)
    iv = numpy.asarray(window["instance_valid"], dtype=bool)
    sem = numpy.asarray(window["semantic"], dtype=numpy.int64)
    sv = numpy.asarray(window["semantic_valid"], dtype=bool)
    if inst.shape != (size, size):
        raise InstanceOffsetsError("window instance map is not the crop size")
    base = iv & (inst >= 0) & sv & (sem >= 0) & (sem < int(num_thing_classes))
    offsets = numpy.zeros((OFFSET_CHANNELS, size, size), dtype=numpy.float32)
    mask = numpy.zeros((size, size), dtype=bool)
    groups: List[Dict[str, Any]] = []
    pixels = 0
    max_abs = 0.0
    if bool(base.any()):
        keys = sem[base] * numpy.int64(KEY_STRIDE) + inst[base]
        unique, inverse, counts = numpy.unique(
            keys, return_inverse=True, return_counts=True
        )
        ys, xs = numpy.nonzero(base)
        for position, key in enumerate(unique.tolist()):
            if int(counts[position]) < int(min_group_pixels):
                continue
            cls = int(key // KEY_STRIDE)
            identifier = int(key % KEY_STRIDE)
            centroid = centroids.get((cls, identifier))
            if centroid is None:
                continue
            selected = inverse == position
            rows = ys[selected]
            columns = xs[selected]
            # Native centroid translated into the crop frame: the *whole-image*
            # centroid minus the pixel's own native coordinate.
            target_row = (float(centroid["row"]) - (top + rows.astype(numpy.float64))) / float(norm)
            target_column = (
                float(centroid["column"]) - (left + columns.astype(numpy.float64))
            ) / float(norm)
            offsets[0][rows, columns] = target_row.astype(numpy.float32)
            offsets[1][rows, columns] = target_column.astype(numpy.float32)
            mask[rows, columns] = True
            pixels += int(counts[position])
            if target_row.size:
                max_abs = max(
                    max_abs,
                    float(numpy.abs(target_row).max()),
                    float(numpy.abs(target_column).max()),
                )
            groups.append(
                {
                    "class": cls,
                    "instance": identifier,
                    "crop_pixels": int(counts[position]),
                    "native_pixels": int(centroid["pixels"]),
                    "centroid_row": float(centroid["row"]),
                    "centroid_column": float(centroid["column"]),
                    "centroid_inside_crop": bool(
                        0 <= float(centroid["row"]) - top < size
                        and 0 <= float(centroid["column"]) - left < size
                    ),
                }
            )
    return {
        "offsets": offsets,
        "mask": mask,
        "pixels": int(pixels),
        "groups": groups,
        "group_count": len(groups),
        "max_abs_unit": float(max_abs),
        "top": int(top),
        "left": int(left),
    }


def crop_is_supervised(
    targets: Mapping[str, Any],
    *,
    min_pixels: int = MIN_GROUP_PIXELS,
) -> Tuple[bool, Dict[str, Any]]:
    """Cheap gate evaluated *before* any forward pass."""

    pixels = int(targets.get("pixels", 0))
    usable = pixels >= int(min_pixels) and int(targets.get("group_count", 0)) >= 1
    return usable, {
        "supervised_pixels": pixels,
        "groups": int(targets.get("group_count", 0)),
        "max_abs_unit": float(targets.get("max_abs_unit", 0.0)),
    }


def offset_loss(head_output: Any, targets: Mapping[str, Any], *, beta: float = SMOOTHL1_BETA) -> Any:
    """Masked SmoothL1 on head units; mean over (pixels x 2).

    ``head_output`` is ``B x 2 x H x W`` in head units (pixels / 256).  Nothing is
    computed for masked-out pixels and unknown pixels never become negatives.
    """

    torch = _require_torch()
    if not torch.is_tensor(head_output) or head_output.ndim != 4:
        raise InstanceOffsetsError("head output must be a B x 2 x H x W tensor")
    if int(head_output.shape[1]) != OFFSET_CHANNELS:
        raise InstanceOffsetsError(
            f"head output must have {OFFSET_CHANNELS} channels; got {int(head_output.shape[1])}"
        )
    numpy = _require_numpy()
    target = torch.from_numpy(
        numpy.array(targets["offsets"], dtype=numpy.float32, copy=True)
    ).to(head_output.device)
    mask = torch.from_numpy(
        numpy.array(targets["mask"], dtype=bool, copy=True)
    ).to(head_output.device)
    if tuple(target.shape[-2:]) != tuple(head_output.shape[-2:]):
        raise InstanceOffsetsError("target/logit size mismatch")
    if not bool(mask.any()):
        raise InstanceOffsetsError("offset_loss called without supervised pixels")
    element = F.smooth_l1_loss(head_output, target.unsqueeze(0), beta=float(beta), reduction="none")
    weights = mask.unsqueeze(0).unsqueeze(0).to(element.dtype)
    denominator = weights.sum().clamp(min=1.0) * float(OFFSET_CHANNELS)
    return (element * weights).sum() / denominator


# ---------------------------------------------------------------------------
# label-free vote clustering decoder (no ground truth, no whole-CC merge)
# ---------------------------------------------------------------------------


def _bin_seeds(points: Any, *, bin_px: float) -> Tuple[Any, Any]:
    """Deterministic bin seeds: mean endpoint per ``bin_px`` cell."""

    numpy = _require_numpy()
    keys = numpy.floor(points / float(bin_px)).astype(numpy.int64)
    order = numpy.lexsort((keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_points = points[order]
    boundaries = numpy.ones(sorted_keys.shape[0], dtype=bool)
    boundaries[1:] = (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)
    starts = numpy.flatnonzero(boundaries)
    stops = numpy.append(starts[1:], sorted_keys.shape[0])
    centres = numpy.stack(
        [sorted_points[start:stop].mean(axis=0) for start, stop in zip(starts, stops)]
    )
    return centres.astype(numpy.float64), sorted_keys[starts]


def _squared_distances(points_block: Any, centres_block: Any) -> Any:
    """``(centres, points)`` squared distances for one bounded block pair.

    The only place a distance temporary is materialised.  Both axes are blocked
    by the callers, so the allocation is ``CENTER_CHUNK x POINT_CHUNK`` — the
    full ``N x centres`` matrix is never built.
    """

    numpy = _require_numpy()
    delta = points_block[None, :, :] - centres_block[:, None, :]
    return numpy.einsum("ijk,ijk->ij", delta, delta)


def _mean_shift(
    points: Any,
    centres: Any,
    *,
    radius_px: float,
    iterations: int,
) -> Tuple[Any, int]:
    """Deterministic mean-shift over vote endpoints (fixed iteration count).

    Streamed: for every centre block the neighbour mask is accumulated over
    point blocks in ascending order, so the neighbour set and the float64 sums
    are the same as a single full reduction while the temporary stays bounded.
    """

    numpy = _require_numpy()
    moved = centres.astype(numpy.float64)
    radius_squared = float(radius_px) ** 2
    point_total = int(points.shape[0])
    for _ in range(int(iterations)):
        updated = moved.copy()
        for start in range(0, moved.shape[0], CENTER_CHUNK):
            stop = min(moved.shape[0], start + CENTER_CHUNK)
            block = moved[start:stop]
            counts = numpy.zeros(stop - start, dtype=numpy.int64)
            sums = numpy.zeros((stop - start, 2), dtype=numpy.float64)
            for point_start in range(0, point_total, POINT_CHUNK):
                point_stop = min(point_total, point_start + POINT_CHUNK)
                points_block = points[point_start:point_stop]
                squared = _squared_distances(points_block, block)
                nearest = squared <= radius_squared
                counts += nearest.sum(axis=1)
                sums += nearest.astype(numpy.float64) @ points_block
            for offset in range(stop - start):
                if counts[offset] > 0:
                    updated[start + offset] = sums[offset] / float(counts[offset])
        if numpy.allclose(updated, moved):
            moved = updated
            break
        moved = updated
    return moved, int(iterations)


def _merge_centres(centres: Any, *, merge_px: float) -> Tuple[Any, Any]:
    """Merge centres closer than ``merge_px``; returns (centres, member_map)."""

    numpy = _require_numpy()
    count = int(centres.shape[0])
    if count == 0:
        return centres, numpy.zeros((0,), dtype=numpy.int64)
    parent = list(range(count))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    threshold = float(merge_px) ** 2
    for left in range(count):
        delta = centres[left + 1 :] - centres[left]
        if delta.size == 0:
            break
        squared = numpy.einsum("ij,ij->i", delta, delta)
        for position in numpy.flatnonzero(squared < threshold).tolist():
            right = left + 1 + int(position)
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                if left_root < right_root:
                    parent[right_root] = left_root
                else:
                    parent[left_root] = right_root
    roots = numpy.asarray([find(item) for item in range(count)], dtype=numpy.int64)
    keep = numpy.unique(roots)
    merged = numpy.stack([centres[roots == root].mean(axis=0) for root in keep.tolist()])
    remap = {int(root): index for index, root in enumerate(keep.tolist())}
    mapping = numpy.asarray([remap[int(root)] for root in roots], dtype=numpy.int64)
    return merged.astype(numpy.float64), mapping


def _assign_nearest(points: Any, centres: Any) -> Any:
    """Nearest-centre assignment; ties resolve to the lower centre index.

    Streamed over point blocks: the temporary is ``CENTER_CHUNK x`` a point
    block, never ``N x centres``.  Strict ``<`` keeps the documented tie rule
    identical to a full-block reduction.
    """

    numpy = _require_numpy()
    if centres.shape[0] == 0:
        raise InstanceOffsetsError("cannot assign pixels without centres")
    point_total = int(points.shape[0])
    best = numpy.full(point_total, -1, dtype=numpy.int64)
    best_distance = numpy.full(point_total, numpy.inf, dtype=numpy.float64)
    for point_start in range(0, point_total, POINT_CHUNK):
        point_stop = min(point_total, point_start + POINT_CHUNK)
        points_block = points[point_start:point_stop]
        block_count = point_stop - point_start
        for start in range(0, centres.shape[0], CENTER_CHUNK):
            stop = min(centres.shape[0], start + CENTER_CHUNK)
            squared = _squared_distances(points_block, centres[start:stop])
            local = numpy.argmin(squared, axis=0)
            distance = squared[local, numpy.arange(block_count)]
            better = distance < best_distance[point_start:point_stop]
            best[point_start:point_stop][better] = start + local[better]
            best_distance[point_start:point_stop][better] = distance[better]
    return best


def cluster_votes(
    endpoints: Any,
    *,
    bin_px: float = VOTE_BIN_PX,
    iterations: int = MEANSHIFT_ITERATIONS,
    radius_px: float = MEANSHIFT_RADIUS_PX,
    merge_px: float = VOTE_MERGE_PX,
    min_cluster_pixels: int = MIN_CLUSTER_PIXELS,
) -> Dict[str, Any]:
    """Deterministic label-free clustering of one class' vote endpoints.

    ``endpoints`` is ``N x 2`` of ``(row, column)`` in native pixels.  The
    schedule is exactly the pre-registered one — 8 px bins, 3 mean-shift
    iterations of radius 16 px, merge under 8 px — and is *not* swept.  Clusters
    thinner than ``min_cluster_pixels`` are returned as ``dissolved`` (their
    pixels are never dropped by the caller; they fall back to their CC id).
    """

    numpy = _require_numpy()
    points = numpy.asarray(endpoints, dtype=numpy.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise InstanceOffsetsError("endpoints must be N x 2")
    bin_px = _finite(bin_px, "bin_px")
    radius_px = _finite(radius_px, "radius_px")
    merge_px = _finite(merge_px, "merge_px")
    if bin_px <= 0 or radius_px <= 0 or merge_px < 0:
        raise InstanceOffsetsError("vote geometry must be positive")
    iterations = _positive_int("iterations", int(iterations))
    min_cluster_pixels = _positive_int("min_cluster_pixels", int(min_cluster_pixels))
    count = int(points.shape[0])
    if count == 0:
        return {
            "points": 0,
            "seeds": 0,
            "centres": numpy.zeros((0, 2), dtype=numpy.float64),
            "assignment": numpy.zeros((0,), dtype=numpy.int64),
            "sizes": numpy.zeros((0,), dtype=numpy.int64),
            "retained": [],
            "dissolved": [],
            "dissolved_pixels": 0,
        }
    seeds, _keys = _bin_seeds(points, bin_px=bin_px)
    shifted, used_iterations = _mean_shift(
        points, seeds, radius_px=radius_px, iterations=iterations
    )
    centres, mapping = _merge_centres(shifted, merge_px=merge_px)
    assignment = _assign_nearest(points, centres)
    sizes = numpy.bincount(assignment, minlength=int(centres.shape[0]))
    retained: List[int] = []
    dissolved: List[int] = []
    for index in range(int(centres.shape[0])):
        if int(sizes[index]) >= int(min_cluster_pixels):
            retained.append(index)
        else:
            dissolved.append(index)
    order = sorted(
        retained,
        key=lambda index: (float(centres[index][0]), float(centres[index][1]), index),
    )
    final = numpy.full((count,), -1, dtype=numpy.int64)
    kept_order: List[int] = []
    for slot, index in enumerate(order):
        final[assignment == index] = slot
        kept_order.append(index)
    return {
        "points": count,
        "seeds": int(seeds.shape[0]),
        "centres": centres,
        "kept_centres": centres[kept_order] if kept_order else numpy.zeros((0, 2)),
        "assignment": assignment,
        "final": final,
        "sizes": sizes,
        "retained": kept_order,
        "dissolved": dissolved,
        "dissolved_pixels": int(sizes[dissolved].sum()) if dissolved else 0,
        "iterations": used_iterations,
    }


def _decode_stats(**overrides: Any) -> Dict[str, Any]:
    """Stable decode-statistics record; the key set never varies.

    Consumers (the paired harness, this lane's artifacts) index these keys
    directly, so every decode returns exactly :data:`DECODE_STATS_KEYS`, not a
    path-dependent subset.  ``tile_rows`` / ``tile_columns`` exist on every
    record and stay ``None`` unless a tiled decoder filled them in.
    """

    stats: Dict[str, Any] = {
        "schema": DECODE_STATS_SCHEMA,
        "class": "label-free per-class vote clustering",
        "vote_bin_px": float(VOTE_BIN_PX),
        "meanshift_iterations": int(MEANSHIFT_ITERATIONS),
        "meanshift_radius_px": float(MEANSHIFT_RADIUS_PX),
        "vote_merge_px": float(VOTE_MERGE_PX),
        "min_cluster_pixels": int(MIN_CLUSTER_PIXELS),
        "offset_norm": float(OFFSET_NORM),
        "instance_id_base": 0,
        "ground_truth_used": False,
        "thing_pixels": 0,
        "assigned_thing_pixels": 0,
        "pixels_deleted": 0,
        "clusters": 0,
        "dissolved_clusters": 0,
        "dissolved_pixels": 0,
        "fallback_groups": 0,
        "fallback_pixels": 0,
        "split_components": 0,
        "max_component_split": 1,
        "seeds_total": 0,
        "votes_total": 0,
        "stuff_pixels": 0,
        "unknown_pixels": 0,
        "pixel_classes_bitidentical": True,
        "stuff_instance_unchanged": True,
        "unknown_instance_unchanged": True,
        "ids_unique": True,
        "id_class_collision": False,
        "tile_rows": None,
        "tile_columns": None,
        "per_class": {},
    }
    unknown = sorted(set(overrides) - set(stats))
    if unknown:
        raise InstanceOffsetsError(f"unknown decode stats keys: {unknown}")
    stats.update(overrides)
    if set(stats) != set(DECODE_STATS_KEYS):
        raise InstanceOffsetsError("decode stats key set drifted")
    return stats


def decode_offsets(
    semantic: Any,
    offsets: Any,
    *,
    semantic_valid: Any = None,
    bin_px: float = VOTE_BIN_PX,
    iterations: int = MEANSHIFT_ITERATIONS,
    radius_px: float = MEANSHIFT_RADIUS_PX,
    merge_px: float = VOTE_MERGE_PX,
    min_cluster_pixels: int = MIN_CLUSTER_PIXELS,
    norm: float = OFFSET_NORM,
    num_thing_classes: int = NUM_THING_CLASSES,
) -> Dict[str, Any]:
    """Vote-cluster predicted Thing pixels into instances; Stuff stays ``-1``.

    * ``semantic`` is a 2-D class-index map (``-1`` unknown, ``0..34`` classes,
      raw ``35`` accepted and normalised to ``-1``);
    * ``offsets`` is ``2 x H x W`` **head units** — pixels divided by ``norm``
      (256).  This function multiplies by ``norm`` to reach native pixels, so a
      caller holding :meth:`OffsetEmbedder.native_offsets` output must pass
      ``field["units"]``; passing ``field["pixels"]`` (already ``units * 256``)
      would scale every vote 256x and destroy the clustering;
    * every Thing pixel receives a non-negative id, ids are unique across
      classes, nothing is deleted, and Stuff / unknown keep ``-1``;
    * pixels of a dissolved thin cluster fall back to their baseline CC8 id, so
      one-pixel instances cannot appear;
    * no ground truth is read.
    """

    numpy = _require_numpy()
    values = numpy.asarray(semantic)
    if values.ndim != 2 or not numpy.issubdtype(values.dtype, numpy.integer):
        raise InstanceOffsetsError("semantic must be a 2D integer array")
    snapshot = values.copy()
    if values.size and (int(values.min()) < -1 or int(values.max()) > NUM_CLASSES):
        raise InstanceOffsetsError("semantic must hold class indices -1..35")
    values = numpy.where(values == IGNORE_CLASS, -1, values).astype(numpy.int64)
    field = numpy.asarray(offsets, dtype=numpy.float32)
    if field.ndim != 3 or int(field.shape[0]) != OFFSET_CHANNELS:
        raise InstanceOffsetsError(f"offsets must be 2 x H x W; got {field.shape}")
    if tuple(field.shape[1:]) != values.shape:
        raise InstanceOffsetsError("offsets must match the semantic map")
    if not bool(numpy.isfinite(field).all()):
        raise InstanceOffsetsError("non_finite_offsets")
    if semantic_valid is None:
        known = values >= 0
    else:
        mask = numpy.asarray(semantic_valid, dtype=bool)
        if mask.shape != values.shape:
            raise InstanceOffsetsError("semantic_valid shape does not match")
        known = mask & (values >= 0)
    thing = known & (values >= 0) & (values < int(num_thing_classes))

    out = numpy.full(values.shape, -1, dtype=numpy.int32)
    if not bool(thing.any()):
        return {
            "instance": out,
            "semantic": values.astype(numpy.int16),
            "stats": _decode_stats(),
        }

    # Baseline CC8 over the *predicted* classes, used only as a fallback.
    raw = numpy.where(thing, values, IGNORE_CLASS)
    baseline = numpy.asarray(raster_infer.instance_ids(raw), dtype=numpy.int64)
    if bool((baseline[thing] < 0).any()):
        raise InstanceOffsetsError("baseline lost Thing pixels")

    next_id = 0
    per_class: Dict[str, Any] = {}
    total_clusters = 0
    total_dissolved = 0
    total_dissolved_pixels = 0
    total_fallback_pixels = 0
    total_fallback_groups = 0
    total_points = 0
    total_seeds = 0

    for class_id in range(int(num_thing_classes)):
        class_mask = thing & (values == int(class_id))
        count = int(class_mask.sum())
        if count == 0:
            continue
        ys, xs = numpy.nonzero(class_mask)
        endpoints = numpy.stack(
            [
                ys.astype(numpy.float64) + field[0][class_mask].astype(numpy.float64) * float(norm),
                xs.astype(numpy.float64) + field[1][class_mask].astype(numpy.float64) * float(norm),
            ],
            axis=1,
        )
        cluster = cluster_votes(
            endpoints,
            bin_px=bin_px,
            iterations=iterations,
            radius_px=radius_px,
            merge_px=merge_px,
            min_cluster_pixels=min_cluster_pixels,
        )
        total_points += int(cluster["points"])
        total_seeds += int(cluster["seeds"])
        kept = list(cluster["retained"])
        final_assignment = cluster["final"]
        class_report: Dict[str, Any] = {
            "pixels": count,
            "seeds": int(cluster["seeds"]),
            "clusters": len(kept),
            "dissolved_clusters": len(cluster["dissolved"]),
            "dissolved_pixels": int(cluster["dissolved_pixels"]),
            "fallback_groups": 0,
            "fallback_pixels": 0,
        }
        for slot in range(len(kept)):
            selector = final_assignment == slot
            instance_id = next_id
            next_id += 1
            out[ys[selector], xs[selector]] = numpy.int32(instance_id)
            total_clusters += 1
        unresolved = final_assignment < 0
        if bool(unresolved.any()):
            total_dissolved += len(cluster["dissolved"])
            total_dissolved_pixels += int(cluster["dissolved_pixels"])
            fallback_labels = baseline[ys[unresolved], xs[unresolved]]
            for label in numpy.unique(fallback_labels).tolist():
                selector = unresolved & (baseline[ys, xs] == int(label))
                instance_id = next_id
                next_id += 1
                out[ys[selector], xs[selector]] = numpy.int32(instance_id)
                total_fallback_groups += 1
                total_fallback_pixels += int(selector.sum())
                class_report["fallback_groups"] += 1
                class_report["fallback_pixels"] += int(selector.sum())
        per_class[str(class_id)] = class_report

    # --- invariant checks (the decode contract) -------------------------
    thing_pixels = int(thing.sum())
    assigned = int((out[thing] >= 0).sum())
    stuff_pixels = int(((values >= int(num_thing_classes)) & (values <= 34)).sum())
    unknown_pixels = int((values < 0).sum())
    normalised_snapshot = numpy.where(
        snapshot == IGNORE_CLASS, -1, snapshot
    ).astype(numpy.int16)
    pixel_classes_ok = bool(
        numpy.array_equal(values.astype(numpy.int16), normalised_snapshot)
        and values.shape == snapshot.shape
    )
    unique_ids = numpy.unique(out[thing])
    # Ids are unique by construction; verify no id spans two classes.
    id_class: Dict[int, int] = {}
    class_conflict = False
    for class_id in numpy.unique(values[thing]).tolist():
        for identifier in numpy.unique(out[thing & (values == int(class_id))]).tolist():
            identifier = int(identifier)
            if identifier in id_class and id_class[identifier] != int(class_id):
                class_conflict = True
            id_class[identifier] = int(class_id)
    if int(unique_ids.size) != int(len(id_class)):
        class_conflict = True
    # How many baseline CCs did the vote split?
    split_components = 0
    max_split = 1
    for label in numpy.unique(baseline[thing]).tolist():
        selector = thing & (baseline == int(label))
        pieces = int(numpy.unique(out[selector]).size)
        if pieces > 1:
            split_components += 1
            max_split = max(max_split, pieces)

    stats = _decode_stats(
        vote_bin_px=float(bin_px),
        meanshift_iterations=int(iterations),
        meanshift_radius_px=float(radius_px),
        vote_merge_px=float(merge_px),
        min_cluster_pixels=int(min_cluster_pixels),
        offset_norm=float(norm),
        thing_pixels=thing_pixels,
        assigned_thing_pixels=assigned,
        pixels_deleted=int(thing_pixels - assigned),
        clusters=int(total_clusters),
        dissolved_clusters=int(total_dissolved),
        dissolved_pixels=int(total_dissolved_pixels),
        fallback_groups=int(total_fallback_groups),
        fallback_pixels=int(total_fallback_pixels),
        split_components=int(split_components),
        max_component_split=int(max_split),
        seeds_total=int(total_seeds),
        votes_total=int(total_points),
        stuff_pixels=stuff_pixels,
        unknown_pixels=unknown_pixels,
        pixel_classes_bitidentical=bool(pixel_classes_ok),
        stuff_instance_unchanged=bool(
            (out[(values >= int(num_thing_classes)) & (values <= 34)] == -1).all()
        ),
        unknown_instance_unchanged=bool((out[values < 0] == -1).all()),
        ids_unique=bool(int(unique_ids.size) == len(id_class)),
        id_class_collision=bool(class_conflict),
        per_class=per_class,
    )
    if stats["pixels_deleted"] != 0:
        raise InstanceOffsetsError("decoder_invariant_violated: Thing pixels lost")
    if not stats["pixel_classes_bitidentical"]:
        raise InstanceOffsetsError("decoder_invariant_violated: pixel classes changed")
    if class_conflict:
        raise InstanceOffsetsError("decoder_invariant_violated: id spans classes")
    return {"instance": out, "semantic": values.astype(numpy.int16), "stats": stats}


# ---------------------------------------------------------------------------
# decoder object (parent-facing API)
# ---------------------------------------------------------------------------


class OffsetDecoder:
    """Frozen semantic + offset head + the fixed vote-clustering decoder."""

    def __init__(
        self,
        embedder: OffsetEmbedder,
        *,
        report: Optional[Mapping[str, Any]] = None,
        tile_size: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
    ) -> None:
        self.embedder = embedder
        self.report = dict(report or {})
        self.tile_size = _positive_int("tile_size", tile_size)
        self.stride = _positive_int("stride", stride)
        self.device = str(embedder.device)

    def offset_field(
        self, image: Any, *, tile_size: Optional[int] = None, stride: Optional[int] = None
    ) -> Dict[str, Any]:
        """Native offset field in head units and pixels (unit-range image)."""

        return self.embedder.native_offsets(
            image,
            tile_size=int(tile_size or self.tile_size),
            stride=int(stride or self.stride),
        )

    def decode(
        self,
        image: Any,
        semantic: Any,
        *,
        semantic_valid: Any = None,
        tile_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decode one image at source resolution.  ``image`` is NOT divided.

        The field is produced in head units and handed to :func:`decode_offsets`
        as units; the single ``* 256`` happens inside the decoder, never here.
        """

        field = self.offset_field(image, tile_size=tile_size, stride=stride)
        result = decode_offsets(
            semantic, field["units"], semantic_valid=semantic_valid
        )
        result["stats"]["tile_rows"] = int(field["tile_rows"])
        result["stats"]["tile_columns"] = int(field["tile_columns"])
        return result

    def close(self) -> None:
        self.embedder.close()


def load_decoder(
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    head_checkpoint: Optional[str] = None,
    *,
    device: str = "cpu",
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> OffsetDecoder:
    """Load the frozen semantic model (+ optional offset head) as a decoder.

    ``head_checkpoint=None`` gives a decoder with a freshly seeded head, which
    is only useful for smoke tests and for the paired lane's "no head" control.
    """

    if head_checkpoint is None and isinstance(semantic_checkpoint, str) and not semantic_checkpoint:
        raise InstanceOffsetsError("semantic checkpoint path is required")
    embedder, report = load_embedder(
        semantic_checkpoint, head_checkpoint=head_checkpoint, device=device
    )
    return OffsetDecoder(embedder, report=report, tile_size=tile_size, stride=stride)


def decode_from_image(
    decoder: OffsetDecoder,
    image: Any,
    semantic: Any,
    *,
    semantic_valid: Any = None,
) -> Dict[str, Any]:
    """Decode ``(image, saved_semantic_index)`` into instances.

    ``image`` is a ``3 x H x W`` array/tensor already in unit range — it is used
    as given and never divided by 255 again.  ``semantic`` is the saved
    class-index map (``-1..34``; raw ``35`` is accepted as ignore).  Returns
    ``{"instance": int32 HxW, "semantic": int16 HxW, "stats": {...}}`` where
    every Thing pixel has a non-negative globally unique id, Stuff/unknown keep
    ``-1`` and pixel classes are unchanged.  ``stats`` always carries
    :data:`DECODE_STATS_KEYS` (schema :data:`DECODE_STATS_SCHEMA`); the offset
    field is consumed in head units, so the ``* 256`` happens exactly once.
    """

    if not isinstance(decoder, OffsetDecoder):
        raise InstanceOffsetsError("decode_from_image needs an OffsetDecoder")
    return decoder.decode(image, semantic, semantic_valid=semantic_valid)


# ---------------------------------------------------------------------------
# supervision planning (CPU, measured on the real cache)
# ---------------------------------------------------------------------------


def _targets_for_crop(
    entry: Mapping[str, Any],
    sampler: Any,
    anchor: Mapping[str, Any],
    centroids: Mapping[Tuple[int, int], Mapping[str, Any]],
    *,
    min_group_pixels: int = MIN_GROUP_PIXELS,
) -> Dict[str, Any]:
    """Authoritative sampler crop + origin-mirrored window + offset targets."""

    authoritative = sampler.crop(entry, anchor)
    window = select_window(sampler, entry, anchor)
    targets = crop_offset_targets(
        window,
        centroids,
        authoritative=authoritative,
        min_group_pixels=min_group_pixels,
    )
    targets["crop"] = authoritative
    targets["crop_index"] = int(authoritative.get("crop_index", 0))
    targets["crop_kind"] = str(authoritative.get("crop_kind", ""))
    return targets


def plan_epochs(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_group_pixels: int = MIN_GROUP_PIXELS,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
    accumulation_steps: int = ACCUMULATION_STEPS,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk every scheduled crop of every epoch and count real supervision."""

    numpy = _require_numpy()
    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    selected = list(records)
    if max_records is not None:
        selected = selected[: _positive_int("max_records", int(max_records))]
    per_epoch: List[Dict[str, Any]] = []
    class_groups: Dict[str, int] = {}
    unique_groups: set = set()
    instance_pixels_total = 0
    far_pixels = 0
    far_crops = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        source, sampler = fullfit.make_epoch_source(
            selected,
            epoch=epoch,
            cache_dir=cache_dir,
            base_seed=base_seed,
        )
        del source  # only the epoch-seeded sampler is needed here
        base = train.SampleSource(selected, cache_dir=cache_dir)
        eligible = 0
        groups = 0
        pixels = 0
        samples: List[str] = []
        for record_index in range(len(selected)):
            entry = base.get(record_index)
            centroids = native_group_centroids(
                entry["instance"],
                entry["instance_valid"],
                entry["semantic"],
                entry["semantic_valid"],
            )
            instance_pixels_total += int(
                numpy.asarray(
                    entry["instance_valid"], dtype=bool
                )[numpy.asarray(entry["instance"], dtype=numpy.int64) >= 0].sum()
            )
            anchors = sampler.build_anchors(entry, record_index=record_index)
            for crop_index in range(int(sampler.crops_per_record)):
                targets = _targets_for_crop(
                    entry,
                    sampler,
                    anchors[crop_index],
                    centroids,
                    min_group_pixels=min_group_pixels,
                )
                usable, report = crop_is_supervised(targets, min_pixels=min_group_pixels)
                if not usable:
                    continue
                eligible += 1
                groups += int(report["groups"])
                pixels += int(report["supervised_pixels"])
                sample_id = str(targets["crop"].get("sample_id", ""))
                if sample_id not in samples:
                    samples.append(sample_id)
                outside = 0
                for item in targets["groups"]:
                    key = str(item["class"])
                    class_groups[key] = int(class_groups.get(key, 0)) + 1
                    unique_groups.add((sample_id, item["class"], item["instance"]))
                    if not item["centroid_inside_crop"]:
                        outside += int(item["crop_pixels"])
                if outside:
                    far_crops += 1
                    far_pixels += outside
        per_epoch.append(
            {
                "epoch": epoch,
                "seed": fullfit.epoch_seed(base_seed, epoch),
                "scheduled_crops": int(len(selected) * int(sampler.crops_per_record)),
                "eligible_crops": int(eligible),
                "skipped_crops": int(
                    len(selected) * int(sampler.crops_per_record) - eligible
                ),
                "eligible_groups": int(groups),
                "supervised_pixels": int(pixels),
                "records_with_eligible_crop": len(samples),
                "optimizer_steps": int(math.ceil(eligible / float(accumulation_steps))),
                "below_min_eligible": bool(eligible < int(min_eligible_crops)),
            }
        )
    stop_cheap = any(item["below_min_eligible"] for item in per_epoch)
    return {
        "records": len(selected),
        "epochs": epochs,
        "per_epoch": per_epoch,
        "stop_cheap": bool(stop_cheap),
        "min_eligible_crops_per_epoch": int(min_eligible_crops),
        "unique_supervised_groups": len(unique_groups),
        "eligible_groups_by_class": {
            key: int(class_groups[key]) for key in sorted(class_groups, key=int)
        },
        "cache_instance_valid_pixels": int(instance_pixels_total),
        "truncated_group_crops": int(far_crops),
        "truncated_group_pixels": int(far_pixels),
        "seconds": round(time.monotonic() - started, 3),
    }


def plan_policy(
    *,
    manifest_path: str,
    cache_dir: Optional[str],
    records: int,
    epochs: int,
    base_seed: int,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """The disclosed, non-swept policy of this lane."""

    return {
        "schema": PREPARE_SCHEMA,
        "version": OFFSETS_VERSION,
        "kind": "frozen_semantic_instance_offsets_prepare",
        "role": "fit",
        "manifest": os.path.abspath(manifest_path),
        "cache_dir": os.path.abspath(cache_dir) if cache_dir else None,
        "records": int(records),
        "crops_per_record": int(CROPS_PER_RECORD),
        "crop_size": int(CROP_SIZE),
        "tile_size": int(TILE_SIZE),
        "tile_stride": int(TILE_STRIDE),
        "semantic_frozen": True,
        "semantic_eval_always": True,
        "difference_from_affinity_lane": (
            "affinity merges whole CC fragments; this lane predicts one vector per "
            "Thing pixel and vote-clusters endpoints, so one connected component "
            "may be split into several instances"
        ),
        "head": {
            "module": "Conv2d(256, 64, 1) -> ReLU -> Conv2d(64, 2, 1)",
            "tap": "net.classifier[3] (ReLU, stride 8, 256 channels)",
            "trainable_parameters": int(
                HEAD_IN_CHANNELS * HEAD_HIDDEN_CHANNELS
                + HEAD_HIDDEN_CHANNELS
                + HEAD_HIDDEN_CHANNELS * OFFSET_CHANNELS
                + OFFSET_CHANNELS
            ),
            "init_seed": int(HEAD_INIT_SEED),
            "init_rng": "explicit torch.Generator, independent of the global RNG",
            "center_logit_head": (
                "omitted: the label contract forbids negatives (unknown is never a "
                "negative), so a centre classifier has no supervision signal; the "
                "offset vector already encodes the centre (zero offset == centroid)"
            ),
        },
        "loss": {
            "function": "masked SmoothL1 on head units (pixels / 256)",
            "beta": SMOOTHL1_BETA,
            "reduction": "sum over masked pixels / (masked pixels x 2)",
            "crop_weighting": "supervised pixel mass within one accumulation window",
            "target": (
                "per-pixel difference: target[c, r, c] = (native GT (class, instance) "
                "centroid[c] - (top + r_or_c)) / 256; a vector per pixel, never the "
                "centroid coordinate, and never the cropped instance centroid"
            ),
            "target_norm": OFFSET_NORM,
            "image_scale": "unit range supplied by SampleSource.get (rescaled exactly once, upstream); this module never divides by 255 again",
            "mask": "instance_valid & instance>=0 & semantic_valid & 0<=semantic<30",
            "min_group_pixels_in_crop": int(MIN_GROUP_PIXELS),
            "instance_class_array_in_cache": False,
            "independent_masks": "instance_valid and semantic_valid are ANDed, never derived",
            "unknown_negatives": False,
            "stuff_negatives": False,
            "zero_supervision": "skipped before any forward pass",
        },
        "training": {
            "epochs": int(epochs),
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "accumulation_steps": int(ACCUMULATION_STEPS),
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_only": True,
        },
        "decoder": {
            "class": "label-free per-class vote clustering (no whole-CC merge)",
            "vote": "endpoint = pixel + offset_pixels (native); tile origin is never added twice",
            "offset_units_at_decode": (
                "decode_offsets takes head units (pixels / 256) and applies the "
                "single *256; callers pass field['units'], never field['pixels']"
            ),
            "stats_schema": DECODE_STATS_SCHEMA,
            "stats_keys_stable": list(DECODE_STATS_KEYS),
            "distance_memory": (
                "mean-shift and nearest-centre assignment block both the point axis "
                f"(POINT_CHUNK {POINT_CHUNK}) and the centre axis (CENTER_CHUNK "
                f"{CENTER_CHUNK}); the largest temporary is "
                f"{POINT_CHUNK * CENTER_CHUNK * 2 * 8 // (1024 * 1024)} MiB"
            ),
            "bin_px": VOTE_BIN_PX,
            "meanshift_iterations": MEANSHIFT_ITERATIONS,
            "meanshift_radius_px": MEANSHIFT_RADIUS_PX,
            "merge_px": VOTE_MERGE_PX,
            "min_cluster_pixels": MIN_CLUSTER_PIXELS,
            "pre_registered": True,
            "thresholds_swept": False,
            "settled_defaults": True,
            "settled_default_note": (
                "bin 8, iterations 3, radius 16, merge 8, min cluster 16 are fixed "
                "decisions of this lane, not tuning knobs; they must not be changed"
            ),
            "fallback": "baseline CC8 id for pixels of a dissolved thin cluster",
            "pixels_deleted": False,
            "stuff": "unchanged (-1 instance, pixel classes bit-identical)",
            "ids": "globally unique across classes, assigned in one (class, cluster) sweep",
            "ground_truth": "never read",
        },
        "plan": dict(plan),
    }


def prepare(
    manifest_path: str = DEFAULT_MANIFEST,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
    max_records: Optional[int] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """CPU-only preparation: role-checked fit records + measured supervision."""

    train.require_command_role("fit", "fit")
    records = fullfit.load_fit_records(manifest_path)
    plan = plan_epochs(
        records,
        cache_dir=cache_dir,
        epochs=epochs,
        base_seed=base_seed,
        min_eligible_crops=min_eligible_crops,
        max_records=max_records,
    )
    policy = plan_policy(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        records=len(records) if max_records is None else min(len(records), int(max_records)),
        epochs=epochs,
        base_seed=base_seed,
        plan=plan,
    )
    if write:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "prepare.json"), policy)
        _write_json(
            os.path.join(output_dir, "policy.json"),
            {key: value for key, value in policy.items() if key != "plan"},
        )
    return policy


# ---------------------------------------------------------------------------
# checkpointing (final epoch only, strict semantic reference)
# ---------------------------------------------------------------------------


def _semantic_contract(
    semantic_checkpoint: str, payload: Mapping[str, Any], *, digest: str
) -> Dict[str, Any]:
    return {
        "path": os.path.abspath(semantic_checkpoint),
        "schema": payload.get("schema"),
        "role": payload.get("role"),
        "epoch": payload.get("epoch"),
        "epochs": payload.get("epochs"),
        "num_classes": NUM_CLASSES,
        "digest": digest,
        "frozen": True,
        "eval_always": True,
        "batch_norm_running_stats": True,
        "trainable": False,
    }


def save_offset_checkpoint(
    path: str,
    embedder: OffsetEmbedder,
    *,
    epoch: int,
    epochs: int,
    semantic_checkpoint: str,
    semantic_payload: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Final-epoch-only head checkpoint with an explicit frozen reference."""

    torch = _require_torch()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": OFFSETS_VERSION,
        "kind": "instance_offset_head",
        "final_epoch_only": True,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "offset_channels": OFFSET_CHANNELS,
        "hidden_channels": HEAD_HIDDEN_CHANNELS,
        "in_channels": HEAD_IN_CHANNELS,
        "feature_stride": FEATURE_STRIDE,
        "offset_norm": OFFSET_NORM,
        "head_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in embedder.head.state_dict().items()
        },
        "head_trainable_parameters": int(
            sum(p.numel() for p in embedder.head.parameters() if p.requires_grad)
        ),
        "semantic": _semantic_contract(
            semantic_checkpoint, semantic_payload, digest=_digest_hex(embedder.model)
        ),
        "decoder": {
            "vote_bin_px": VOTE_BIN_PX,
            "meanshift_iterations": MEANSHIFT_ITERATIONS,
            "meanshift_radius_px": MEANSHIFT_RADIUS_PX,
            "vote_merge_px": VOTE_MERGE_PX,
            "min_cluster_pixels": MIN_CLUSTER_PIXELS,
            "labels": "class indices 0..34, -1 unknown (raw 35 accepted as ignore)",
            "instances": "globally unique ids >= 0 on every Thing pixel, -1 otherwise",
            "ground_truth_at_inference": False,
        },
        "seeds": {
            "base_seed": int(BASE_SEED),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_init_seed": int(HEAD_INIT_SEED),
            "epoch_seeds": {
                str(index): fullfit.epoch_seed(BASE_SEED, index)
                for index in range(1, int(epochs) + 1)
            },
        },
        "contract": {
            "loaders_strict": True,
            "model_promotion": False,
            "pixel_metric": "PQ-style instance metrics only after a paired evaluation",
        },
        "metrics": dict(metrics or {}),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    return {
        "path": os.path.abspath(path),
        "schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "final_epoch_only": True,
    }


def load_offset_checkpoint(path: str, *, device: str = "cpu") -> Dict[str, Any]:
    torch = _require_torch()
    if not os.path.isfile(path):
        raise InstanceOffsetsError(f"checkpoint_not_found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise InstanceOffsetsError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("kind") != "instance_offset_head":
        raise InstanceOffsetsError(f"checkpoint_kind_mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise InstanceOffsetsError("checkpoint_must_be_final_epoch_only")
    head_state = payload.get("head_state_dict")
    if not isinstance(head_state, Mapping) or not head_state:
        raise InstanceOffsetsError("checkpoint_missing_head_state")
    if int(payload.get("offset_channels", -1)) != OFFSET_CHANNELS:
        raise InstanceOffsetsError("checkpoint_offset_channels_mismatch")
    if int(payload.get("in_channels", -1)) != HEAD_IN_CHANNELS:
        raise InstanceOffsetsError("checkpoint_in_channels_mismatch")
    if int(payload.get("hidden_channels", -1)) != HEAD_HIDDEN_CHANNELS:
        raise InstanceOffsetsError("checkpoint_hidden_channels_mismatch")
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping) or not semantic.get("path"):
        raise InstanceOffsetsError("checkpoint_missing_semantic_reference")
    if semantic.get("frozen") is not True or semantic.get("eval_always") is not True:
        raise InstanceOffsetsError("checkpoint_semantic_contract_not_frozen")
    result = dict(payload)
    result["head_state_dict"] = {
        key: value.to(device) for key, value in head_state.items()
    }
    result["semantic_reference_exists"] = bool(os.path.isfile(semantic["path"]))
    return result


def _assert_semantic_reference(
    head_payload: Mapping[str, Any], semantic_checkpoint: str
) -> None:
    reference = head_payload.get("semantic") or {}
    reference_path = reference.get("path")
    if not reference_path:
        raise InstanceOffsetsError("head_checkpoint_missing_semantic_reference")
    if os.path.normcase(os.path.abspath(reference_path)) != os.path.normcase(
        os.path.abspath(semantic_checkpoint)
    ):
        raise InstanceOffsetsError(
            "semantic_checkpoint_mismatch: head was trained against "
            f"{os.path.abspath(reference_path)}"
        )


# ---------------------------------------------------------------------------
# head-only training (parent-scheduled; CPU default)
# ---------------------------------------------------------------------------


def _microbatch_loss(embedder: OffsetEmbedder, targets: Mapping[str, Any]) -> Any:
    torch = _require_torch()
    numpy = _require_numpy()
    image = torch.from_numpy(
        numpy.ascontiguousarray(numpy.asarray(targets["crop"]["image"], dtype=numpy.float32))
    ).unsqueeze(0)
    head_output = embedder.crop_embeddings(image)
    loss = offset_loss(head_output, targets)
    if not bool(torch.isfinite(loss)):
        raise InstanceOffsetsError("non_finite_offset_loss")
    return loss


def _assert_finite_gradient(parameters: Sequence[Any]) -> None:
    torch = _require_torch()
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise InstanceOffsetsError("non_finite_head_gradient")


def _torch_device(device: Any) -> Any:
    """Normalise a device argument to ``torch.device``.

    ``"cuda"``, ``"cuda:0"`` and ``torch.device("cuda:1")`` all normalise, so a
    guard configured for ``cuda:0`` is not silently skipped as "not cuda".
    """

    torch = _require_torch()
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(str(device))
    except Exception as error:  # pragma: no cover - torch's message is enough
        raise InstanceOffsetsError(f"invalid device {device!r}: {error}") from error


def is_cuda_device(device: Any) -> bool:
    """True for any CUDA device spelling, including an explicit index."""

    return _torch_device(device).type == "cuda"


def head_snapshot(head: Any) -> Dict[str, Any]:
    """Bit-exact CPU copy of a head's parameters and buffers."""

    return {
        key: value.detach().to("cpu").clone()
        for key, value in head.state_dict().items()
    }


def restore_head(head: Any, snapshot: Mapping[str, Any]) -> bool:
    """Restore ``head`` from :func:`head_snapshot`; returns bit-exact equality."""

    torch = _require_torch()
    head.load_state_dict(dict(snapshot), strict=True)
    current = head.state_dict()
    return all(
        torch.equal(current[key].detach().to("cpu"), snapshot[key])
        for key in snapshot
    )


def guard_verdict(
    free_before_mib: float,
    peak_mib: float,
    *,
    reserve_after_peak_mib: float = RESERVE_AFTER_PEAK_MIB,
    min_free_mib: float = 512.0,
) -> Dict[str, Any]:
    """Pure acceptance arithmetic for the GPU guard (no torch, no device).

    ``free_before - peak`` is the memory left *after* the accumulation window;
    it must cover ``max(reserve_after_peak_mib, min_free_mib)``.  That is the same
    as ``free_before >= peak + reserve``: the peak is charged **once**.  The
    earlier form (``free_before - peak >= peak + reserve``) charged it twice and
    demanded ``2 x peak + reserve`` — wrong, and corrected here.
    """

    free_before = _finite(free_before_mib, "free_before_mib")
    peak = _finite(peak_mib, "peak_mib")
    reserve = _finite(reserve_after_peak_mib, "reserve_after_peak_mib")
    floor = _finite(min_free_mib, "min_free_mib")
    if peak < 0.0 or reserve < 0.0 or floor < 0.0:
        raise InstanceOffsetsError("guard inputs must be non-negative")
    remaining = float(free_before) - float(peak)
    required = max(float(reserve), float(floor))
    return {
        "status": "ok" if remaining >= required else "insufficient_headroom",
        "free_before_mib": round(float(free_before), 3),
        "peak_mib": round(float(peak), 3),
        "remaining_after_peak_mib": round(remaining, 3),
        "required_remaining_mib": round(required, 3),
        "reserve_after_peak_mib": float(reserve),
        "min_free_mib": float(floor),
        "shortfall_mib": round(max(0.0, required - remaining), 3),
        "equivalent": "free_before_mib >= peak_mib + required_remaining_mib",
    }


def cuda_accum_guard(
    embedder: OffsetEmbedder,
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    device: str = "cuda",
    accumulation_steps: int = ACCUMULATION_STEPS,
    base_seed: int = BASE_SEED,
    epoch: int = 1,
    min_free_mib: float = 512.0,
    reserve_after_peak_mib: float = RESERVE_AFTER_PEAK_MIB,
) -> Dict[str, Any]:
    """Measure the **actual** peak of ``accumulation_steps`` pending graphs.

    One forward is not a guard: ``accumulation_steps`` graphs stay alive until
    the step, so this builds that many real supervised crops, backpropagates
    them all, steps once and reports ``torch.cuda.max_memory_allocated`` /
    ``reserved`` plus free memory before and after.

    The measurement is **pure with respect to the head**: the head state is
    snapshotted before the run and restored bit-exactly in a ``finally``, and
    every gradient is dropped, so a guard run never becomes a warm start for the
    fit that follows it.  Gradients are still built and stepped, because that is
    what allocates the real accumulation window.

    ``status`` is ``ok`` only when the memory left after the measured peak covers
    :data:`RESERVE_AFTER_PEAK_MIB`: ``free_before - peak >= 1024 MiB``, i.e.
    ``free_before >= peak + 1024 MiB``.  The peak is charged once.  See
    :func:`guard_verdict` for the arithmetic (pure, testable without a device).
    """

    torch = _require_torch()
    numpy = _require_numpy()
    target = _torch_device(device)
    if target.type != "cuda":
        raise InstanceOffsetsError(
            f"cuda_accum_guard requires a CUDA device; got {device!r}"
        )
    if not torch.cuda.is_available():
        raise InstanceOffsetsError("cuda_accum_guard requires an available CUDA device")
    index = int(target.index) if target.index is not None else int(
        torch.cuda.current_device()
    )
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    free_before, total = torch.cuda.mem_get_info(index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)
    source, sampler = fullfit.make_epoch_source(
        records, epoch=int(epoch), cache_dir=cache_dir, base_seed=base_seed
    )
    base = train.SampleSource(list(records), cache_dir=cache_dir)
    head_state = head_snapshot(embedder.head)
    head_training = bool(embedder.head.training)
    model_training = bool(embedder.model.training)
    embedder.model.eval()
    embedder.head.to(target)
    embedder.head.train()
    started = time.monotonic()
    pending: List[Tuple[Any, int]] = []
    pending_pixels = 0
    losses: List[float] = []
    groups_per_crop: List[int] = []
    steps = 0
    parameters = [p for p in embedder.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    try:
        try:
            for record_index in range(len(records)):
                entry = base.get(record_index)
                centroids = native_group_centroids(
                    entry["instance"],
                    entry["instance_valid"],
                    entry["semantic"],
                    entry["semantic_valid"],
                )
                anchors = sampler.build_anchors(entry, record_index=record_index)
                for crop_index in range(int(sampler.crops_per_record)):
                    targets = _targets_for_crop(
                        entry, sampler, anchors[crop_index], centroids
                    )
                    usable, report = crop_is_supervised(targets)
                    if not usable:
                        continue
                    loss = _microbatch_loss(embedder, targets)
                    pending.append((loss, int(targets["pixels"])))
                    pending_pixels += int(targets["pixels"])
                    groups_per_crop.append(int(report["groups"]))
                    losses.append(float(loss.detach().item()))
                    if len(pending) >= accumulation_steps:
                        raise _AccumulationReached
        except _AccumulationReached:
            pass
        if len(pending) < accumulation_steps:
            raise InstanceOffsetsError(
                f"guard found only {len(pending)} supervised crops, "
                f"need {accumulation_steps}"
            )
        total_pixels = max(1, pending_pixels)
        for loss, pixels in pending:
            (loss * (float(pixels) / float(total_pixels))).backward()
        _assert_finite_gradient(parameters)
        optimizer.step()
        steps += 1
        peak_allocated = torch.cuda.max_memory_allocated(index)
        peak_reserved = torch.cuda.max_memory_reserved(index)
        free_after, _total_after = torch.cuda.mem_get_info(index)
    finally:
        # Purity: the guard never leaves a trained weight or a live gradient.
        restored = restore_head(embedder.head, head_state)
        for parameter in embedder.head.parameters():
            parameter.grad = None
        optimizer.zero_grad(set_to_none=True)
        embedder.head.train(head_training)
        embedder.model.train(model_training)
        del pending
        torch.cuda.empty_cache()
    peak_mib = peak_allocated / (1024.0**2)
    verdict = guard_verdict(
        free_before / (1024.0**2),
        peak_mib,
        reserve_after_peak_mib=reserve_after_peak_mib,
        min_free_mib=min_free_mib,
    )
    remaining = verdict["remaining_after_peak_mib"]
    status = verdict["status"]
    return {
        "schema": GUARD_SCHEMA,
        "device": torch.cuda.get_device_name(index),
        "cuda": True,
        "device_index": index,
        "head_state_restored": bool(restored),
        "gradients_cleared": True,
        "head_training_after": bool(embedder.head.training),
        "total_mib": round(total / (1024.0**2), 1),
        "free_before_mib": round(free_before / (1024.0**2), 1),
        "free_after_mib": round(free_after / (1024.0**2), 1),
        "pending_graphs": len(groups_per_crop),
        "groups_per_crop": groups_per_crop,
        "losses": [round(value, 6) for value in losses],
        "peak_allocated_mib": round(peak_mib, 1),
        "peak_reserved_mib": round(peak_reserved / (1024.0**2), 1),
        "headroom_mib": round(remaining, 1),
        "required_headroom_mib": round(verdict["required_remaining_mib"], 1),
        "reserve_after_peak_mib": float(reserve_after_peak_mib),
        "min_free_mib": float(min_free_mib),
        "acceptance": (
            "free_before_mib - peak_allocated_mib >= max(reserve_after_peak_mib, "
            "min_free_mib)  ==  free_before_mib >= peak_allocated_mib + reserve"
        ),
        "acceptance_note": (
            "the reserve is memory still free AFTER the peak; the peak is charged "
            "once, not twice"
        ),
        "optimizer_steps": steps,
        "seconds": round(time.monotonic() - started, 3),
        "status": status,
        "semantic_requires_grad": int(
            sum(1 for p in embedder.model.parameters() if p.requires_grad)
        ),
        "note": "full accumulation window measured; one forward is not a guard",
    }


class _AccumulationReached(Exception):
    """Internal control-flow marker for the guard loop."""


def train_offset_head(
    embedder: OffsetEmbedder,
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    output_dir: str,
    semantic_checkpoint: str,
    semantic_payload: Mapping[str, Any],
    epochs: int = EPOCHS,
    device: str = "cpu",
    accumulation_steps: int = ACCUMULATION_STEPS,
    base_seed: int = BASE_SEED,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    role: str = "fit",
    guard: bool = True,
    min_free_mib: float = 512.0,
) -> Dict[str, Any]:
    """Bounded head-only fit: pixel-mass-weighted SmoothL1, frozen semantics."""

    torch = _require_torch()
    train.require_command_role("fit", role)
    if not records:
        raise InstanceOffsetsError("no fit records")
    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    lr = _finite(lr, "lr")
    weight_decay = _finite(weight_decay, "weight_decay")
    if lr <= 0.0 or weight_decay < 0.0:
        raise InstanceOffsetsError("lr must be positive and weight_decay >= 0")
    freeze_semantic_model(embedder.model)
    digest_before = _digest_hex(embedder.model)
    parameters = [p for p in embedder.head.parameters() if p.requires_grad]
    if not parameters:
        raise InstanceOffsetsError("head has no trainable parameters")
    guard_report: Optional[Dict[str, Any]] = None
    guard_purity: Optional[Dict[str, Any]] = None
    head_init_state = head_snapshot(embedder.head)
    if guard and is_cuda_device(device):
        guard_report = cuda_accum_guard(
            embedder,
            records,
            cache_dir=cache_dir,
            device=device,
            accumulation_steps=accumulation_steps,
            base_seed=base_seed,
            min_free_mib=min_free_mib,
        )
        # The guard steps a throwaway optimizer; put the head back exactly where
        # the fit must start from and drop every gradient it produced.
        restored = restore_head(embedder.head, head_init_state)
        for parameter in embedder.head.parameters():
            parameter.grad = None
        guard_purity = {
            "head_state_restored_bit_exact": bool(restored),
            "gradients_cleared": all(
                parameter.grad is None for parameter in embedder.head.parameters()
            ),
            "guard_never_warm_starts_the_fit": bool(restored),
        }
        if not restored:
            raise InstanceOffsetsError(
                "cuda_guard_impure: guard changed the head and it could not be restored"
            )
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "cuda_accum8_guard.json"), guard_report)
        if guard_report["status"] != "ok":
            raise InstanceOffsetsError(
                f"cuda_guard_failed: {guard_report['status']} "
                f"(headroom {guard_report['headroom_mib']} MiB)"
            )
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    started = time.monotonic()
    epoch_reports: List[Dict[str, Any]] = []
    final_metrics: Dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        seed = fullfit.epoch_seed(base_seed, epoch)
        train.seed_everything(seed)
        source, sampler = fullfit.make_epoch_source(
            records,
            epoch=epoch,
            cache_dir=cache_dir,
            base_seed=base_seed,
        )
        base = train.SampleSource(list(records), cache_dir=cache_dir)
        embedder.model.eval()
        embedder.head.train()
        optimizer.zero_grad(set_to_none=True)
        scheduled = 0
        supervised = 0
        skipped = 0
        steps = 0
        pixel_total = 0
        group_total = 0
        weighted_loss = 0.0
        pending: List[Tuple[Any, int]] = []
        pending_pixels = 0
        max_abs_unit = 0.0

        def flush() -> None:
            nonlocal steps, pixel_total, group_total, weighted_loss
            nonlocal pending, pending_pixels
            if not pending:
                return
            total = max(1, pending_pixels)
            for loss, pixels in pending:
                if not bool(torch.isfinite(loss)):
                    raise InstanceOffsetsError("non_finite_offset_loss_before_backward")
                (loss * (float(pixels) / float(total))).backward()
            _assert_finite_gradient(parameters)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            pixel_total += pending_pixels
            weighted_loss += sum(
                float(loss.detach().item()) * pixels for loss, pixels in pending
            )
            pending = []
            pending_pixels = 0

        for record_index in range(len(records)):
            entry = base.get(record_index)
            centroids = native_group_centroids(
                entry["instance"],
                entry["instance_valid"],
                entry["semantic"],
                entry["semantic_valid"],
            )
            anchors = sampler.build_anchors(entry, record_index=record_index)
            for crop_index in range(int(sampler.crops_per_record)):
                scheduled += 1
                targets = _targets_for_crop(
                    entry, sampler, anchors[crop_index], centroids
                )
                usable, report = crop_is_supervised(targets)
                if not usable:
                    skipped += 1
                    continue
                loss = _microbatch_loss(embedder, targets)
                supervised += 1
                group_total += int(report["groups"])
                pending.append((loss, int(targets["pixels"])))
                pending_pixels += int(targets["pixels"])
                max_abs_unit = max(max_abs_unit, float(targets["max_abs_unit"]))
                if len(pending) >= accumulation_steps:
                    flush()
        flush()
        embedder.model.eval()
        if _digest_hex(embedder.model) != digest_before:
            raise InstanceOffsetsError(
                f"semantic parameters changed during epoch {epoch}"
            )
        epoch_report = {
            "epoch": int(epoch),
            "seed": int(seed),
            "scheduled_crops": int(scheduled),
            "supervised_crops": int(supervised),
            "skipped_crops": int(skipped),
            "optimizer_steps": int(steps),
            "groups": int(group_total),
            "supervised_pixels": int(pixel_total),
            "loss_pixel_weighted": (
                (weighted_loss / float(pixel_total)) if pixel_total else None
            ),
            "max_abs_target_unit": float(max_abs_unit),
            "accumulation_steps": int(accumulation_steps),
            "semantic_parameters_unchanged": True,
        }
        epoch_reports.append(epoch_report)
        final_metrics = epoch_report
        del source
    moved = any(
        not torch.equal(
            head_init_state[key].to(embedder.head.state_dict()[key].device),
            embedder.head.state_dict()[key],
        )
        for key in head_init_state
    )
    result = {
        "schema": FIT_SCHEMA,
        "version": OFFSETS_VERSION,
        "kind": "instance_offsets_fit",
        "final_epoch_only": True,
        "device": str(device),
        "epochs": int(epochs),
        "per_epoch": epoch_reports,
        "final": final_metrics,
        "head_changed_from_init": bool(moved),
        "head_trainable_parameters": int(
            sum(p.numel() for p in embedder.head.parameters() if p.requires_grad)
        ),
        "finite_checks": {
            "loss_per_microbatch_aborts_on_nonfinite": True,
            "loss_before_backward_aborts_on_nonfinite": True,
            "grad_before_step_aborts_on_nonfinite": True,
            "nonfinite_detected": False,
        },
        "semantic_parameters_frozen": True,
        "semantic_digest": digest_before,
        "loss": {
            "beta": SMOOTHL1_BETA,
            "offset_norm": OFFSET_NORM,
            "pixel_mass_weighted": True,
            "min_group_pixels": int(MIN_GROUP_PIXELS),
        },
        "seeds": {
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_init_seed": int(HEAD_INIT_SEED),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "accumulation_steps": int(accumulation_steps),
        },
        "guard": guard_report,
        "guard_purity": guard_purity,
        "seconds": round(time.monotonic() - started, 3),
    }
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint_instance_offsets.pt")
    result["checkpoint"] = save_offset_checkpoint(
        checkpoint_path,
        embedder,
        epoch=epochs,
        epochs=epochs,
        semantic_checkpoint=semantic_checkpoint,
        semantic_payload=semantic_payload,
        metrics={"final": final_metrics, "epochs": epoch_reports},
    )
    _write_json(os.path.join(output_dir, "fit.json"), result)
    return result


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    *,
    size: int = CROP_SIZE,
    device: str = "cpu",
) -> Dict[str, Any]:
    """CPU load of the frozen checkpoint + head/decoder shape and invariant check."""

    torch = _require_torch()
    numpy = _require_numpy()
    size = _positive_int("size", size)
    embedder, report = load_embedder(semantic_checkpoint, device=device)
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    outputs = embedder.feature_map(image)
    head_output = embedder.crop_embeddings(image)
    field = embedder.native_offsets(image[0].numpy())
    semantic_map = numpy.full((size, size), -1, dtype=numpy.int16)
    semantic_map[size // 4 : size // 2, size // 4 : size // 2] = 0
    decoded = decode_offsets(semantic_map, field["units"])
    result = {
        "schema": SMOKE_SCHEMA,
        "version": OFFSETS_VERSION,
        "device": device,
        "image_shape": [1, 3, size, size],
        "feature_shape": [int(v) for v in outputs["features"].shape],
        "head_output_shape": [int(v) for v in head_output.shape],
        "native_offsets_shape": [int(v) for v in field["pixels"].shape],
        "tile_rows": int(field["tile_rows"]),
        "tile_columns": int(field["tile_columns"]),
        "head_parameters": int(sum(p.numel() for p in embedder.head.parameters())),
        "semantic_parameters_requires_grad": int(
            sum(1 for p in embedder.model.parameters() if p.requires_grad)
        ),
        "finite_head_output": bool(numpy.isfinite(head_output.detach().numpy()).all()),
        "finite_offsets": bool(numpy.isfinite(field["pixels"]).all()),
        "decode_thing_pixels": int(decoded["stats"]["thing_pixels"]),
        "decode_assigned_pixels": int(decoded["stats"]["assigned_thing_pixels"]),
        "decode_clusters": int(decoded["stats"]["clusters"]),
        "report": report,
    }
    if tuple(result["feature_shape"][-2:]) != (
        size // FEATURE_STRIDE,
        size // FEATURE_STRIDE,
    ):
        raise InstanceOffsetsError("unexpected feature stride")
    if tuple(result["head_output_shape"]) != (1, OFFSET_CHANNELS, size, size):
        raise InstanceOffsetsError("unexpected head output shape")
    if result["semantic_parameters_requires_grad"]:
        raise InstanceOffsetsError("semantic parameters are not frozen")
    if result["decode_thing_pixels"] != result["decode_assigned_pixels"]:
        raise InstanceOffsetsError("decoder lost pixels in smoke")
    embedder.close()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_instance_offsets",
        description=(
            "Frozen-semantic instance-offset pilot: CPU prepare, CPU smoke, an "
            "explicitly scheduled head-only fit and the CUDA accumulation guard."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="CPU supervision plan")
    prepare_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    prepare_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--epochs", type=int, default=EPOCHS)
    prepare_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    prepare_parser.add_argument(
        "--min-eligible-crops", type=int, default=MIN_ELIGIBLE_CROPS_PER_EPOCH
    )
    prepare_parser.add_argument("--max-records", type=int, default=None)

    smoke_parser = subparsers.add_parser("smoke", help="CPU checkpoint smoke")
    smoke_parser.add_argument("--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT)
    smoke_parser.add_argument("--size", type=int, default=CROP_SIZE)
    smoke_parser.add_argument("--device", default="cpu")

    fit_parser = subparsers.add_parser("fit", help="head-only fit (parent-scheduled)")
    fit_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    fit_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    fit_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    fit_parser.add_argument("--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT)
    fit_parser.add_argument("--role", default="fit")
    fit_parser.add_argument("--device", default="cpu")
    fit_parser.add_argument("--epochs", type=int, default=EPOCHS)
    fit_parser.add_argument(
        "--accumulation-steps", type=int, default=ACCUMULATION_STEPS
    )
    fit_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    fit_parser.add_argument("--lr", type=float, default=LR)
    fit_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    fit_parser.add_argument("--no-guard", action="store_true")
    fit_parser.add_argument("--min-free-mib", type=float, default=512.0)

    guard_parser = subparsers.add_parser(
        "guard", help="measure the real 8-accumulation CUDA peak"
    )
    guard_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    guard_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    guard_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    guard_parser.add_argument("--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT)
    guard_parser.add_argument("--device", default="cuda")
    guard_parser.add_argument(
        "--accumulation-steps", type=int, default=ACCUMULATION_STEPS
    )
    guard_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    guard_parser.add_argument("--min-free-mib", type=float, default=512.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare(
            args.manifest,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            base_seed=args.base_seed,
            min_eligible_crops=args.min_eligible_crops,
            max_records=args.max_records,
        )
        print(json.dumps(payload["plan"], indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        print(
            json.dumps(
                smoke(args.semantic_checkpoint, size=args.size, device=args.device),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit":
        train.require_command_role("fit", args.role)
        records = fullfit.load_fit_records(args.manifest)
        model, payload = pilot.load_checkpoint_model(
            args.semantic_checkpoint, device=args.device
        )
        embedder = OffsetEmbedder(model, device=args.device)
        result = train_offset_head(
            embedder,
            records,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            semantic_checkpoint=args.semantic_checkpoint,
            semantic_payload=payload,
            epochs=args.epochs,
            device=args.device,
            accumulation_steps=args.accumulation_steps,
            base_seed=args.base_seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            role=args.role,
            guard=not args.no_guard,
            min_free_mib=args.min_free_mib,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "guard":
        train.require_command_role("fit", "fit")
        records = fullfit.load_fit_records(args.manifest)
        model, _payload = pilot.load_checkpoint_model(
            args.semantic_checkpoint, device=args.device
        )
        embedder = OffsetEmbedder(model, device=args.device)
        report = cuda_accum_guard(
            embedder,
            records,
            cache_dir=args.cache_dir,
            device=args.device,
            accumulation_steps=args.accumulation_steps,
            base_seed=args.base_seed,
            min_free_mib=args.min_free_mib,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        _write_json(os.path.join(args.output_dir, "cuda_accum8_guard.json"), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DECODE_STATS_KEYS",
    "DECODE_STATS_SCHEMA",
    "FIT_SCHEMA",
    "InstanceOffsetsError",
    "MIN_CLUSTER_PIXELS",
    "MEANSHIFT_ITERATIONS",
    "MEANSHIFT_RADIUS_PX",
    "OFFSETS_SCHEMA",
    "OFFSET_CHANNELS",
    "OFFSET_NORM",
    "OffsetDecoder",
    "OffsetEmbedder",
    "PREPARE_SCHEMA",
    "SMOKE_SCHEMA",
    "VOTE_BIN_PX",
    "VOTE_MERGE_PX",
    "build_offset_head",
    "cluster_votes",
    "crop_is_supervised",
    "crop_offset_targets",
    "cuda_accum_guard",
    "decode_from_image",
    "decode_offsets",
    "extract_window",
    "guard_verdict",
    "head_snapshot",
    "is_cuda_device",
    "load_decoder",
    "load_embedder",
    "load_offset_checkpoint",
    "native_group_centroids",
    "offset_loss",
    "offsets_to_pixels",
    "pixels_to_offsets",
    "plan_epochs",
    "prepare",
    "restore_head",
    "save_offset_checkpoint",
    "select_window",
    "semantic_parameter_digest",
    "smoke",
    "train_offset_head",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
