#!/usr/bin/env python3
"""Full-fit pretrained DeepLabV3 lane with tiled inference.

This module is the larger, independent companion of the small 24-record pilot.
It keeps the same model and masked cross-entropy, but trains on every fit
record, varies the support-crop seed per epoch, accumulates gradients over a
fixed number of supervised crops, and provides a source-resolution tiled
inference API that matches the 256 px training window.

Scope rules enforced here:

* the data reader resolves the fit role before touching any asset;
* only fit PNG/SVG pairs are used; tune and protected roles are refused;
* labels are never fed to the model and unknown semantic pixels are ignored;
* instance labels are carried for diagnostics only and are never fused;
* no resize: training crops and inference tiles are windowed from source.

Nothing in this module downloads data or weights unless ``fit`` is asked to
load official pretrained weights. ``prepare`` only reads fit metadata/assets
and builds the on-disk array cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_train as train

try:  # pragma: no cover - exercised through the project venv
    import numpy as numpy
except Exception:  # pragma: no cover
    numpy = None  # type: ignore


FULLFIT_SCHEMA = "cadvec.recognition.pretrained_fullfit/1"
PREPARE_SCHEMA = "cadvec.recognition.pretrained_fullfit_prepare/1"
FIT_SCHEMA = "cadvec.recognition.pretrained_fullfit_fit/1"
TILED_SCHEMA = "cadvec.recognition.pretrained_fullfit_tiled/1"

# Fixed plan: no sweep, no tuning of these values.
TILE_SIZE = 256
TILE_STRIDE = 128
CROP_SIZE = 256
CROPS_PER_RECORD = 8
CROP_MIN_VALID = 32
ACCUMULATION_STEPS = 8
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 0.01
TRAIN_SCALE = 256
BASE_SEED = 20261108
EPOCH_SEED_STRIDE = 1000003
DISK_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB for the array cache
BACKGROUND_FRACTION = 0.0
CROP_JITTER = 0

TILING_POLICY = {
    "mode": "tiled_source_resolution",
    "tile_size": TILE_SIZE,
    "stride": TILE_STRIDE,
    "overlap": TILE_SIZE - TILE_STRIDE,
    "combine": "arithmetic mean of overlapping logits",
    "padding": "right/bottom only, image value 1.0 (white), labels never padded",
    "resize": None,
    "context": "inference tile equals the 256 px training crop",
}

TRAIN_POLICY = {
    "role": "fit",
    "records": "all fit records at the requested group scale",
    "objective": "masked semantic cross-entropy, known 0..34 only",
    "unknown": "excluded from the loss; never a negative or background target",
    "instance_labels": "untouched; not used by the loss or the model input",
    "class_weights": None,
    "optimizer": "AdamW",
    "weight_decay": WEIGHT_DECAY,
    "learning_rate": LR,
    "accumulation": {
        "micro_batch": 1,
        "crops_per_step": ACCUMULATION_STEPS,
        "normalization": (
            "exact weighted mean over the crops in the step: "
            "sum(crop_ce * known_crop) / sum(known_crop); partial final step "
            "uses its own known-pixel mass"
        ),
    },
    "crops": {
        "size": CROP_SIZE,
        "per_record_per_epoch": CROPS_PER_RECORD,
        "stride": None,
        "resize": None,
        "epoch_seed": "base + epoch * 1000003 (disclosed per epoch)",
        "background_fraction": BACKGROUND_FRACTION,
        "jitter": CROP_JITTER,
        "min_valid": CROP_MIN_VALID,
        "zero_supervision": "skipped before forward; no optimizer step",
    },
    "selection": "final epoch only",
    "amp": "not used",
    "tune_evaluation": "not performed by this lane",
}


class FullFitError(pilot.PilotError):
    """Raised for full-fit lane contract violations."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    global numpy
    if numpy is None:
        try:
            import numpy as numpy_module
        except Exception as exc:  # pragma: no cover - environment guard
            raise FullFitError("numpy is required for the full-fit lane") from exc
        numpy = numpy_module
    return numpy


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def epoch_seed(base_seed: int = BASE_SEED, epoch: int = 1) -> int:
    """Deterministic, disclosed seed for one epoch's crop sampler."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise FullFitError(f"epoch must be a positive int; got {epoch!r}")
    return int(base_seed) + EPOCH_SEED_STRIDE * int(epoch)


def load_fit_records(
    manifest_path: str, *, train_scale: int = TRAIN_SCALE
) -> List[Dict[str, Any]]:
    """Resolve *fit* records through the existing reader (role checked first)."""

    records = train.role_sample_records(
        manifest_path, "fit", train_scale=int(train_scale)
    )
    if not records:
        raise FullFitError("fit role resolved to zero records")
    for record in records:
        if record.get("role") != "fit":
            raise train.RoleAccessError(
                f"fit reader returned a non-fit record: {record.get('role')!r}"
            )
    return records


class EpochCropSource:
    """Sequential crop source that keeps exactly one full record in memory.

    ``train.SampleSource`` reloads the cached record for every crop.  Eight
    crops per record would mean eight npz reads; this wrapper caches the last
    record and its anchors instead, which keeps RAM bounded to one sample.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        cache_dir: Optional[str],
        sampler: "train.SupportCropSampler",
    ) -> None:
        self.records = list(records)
        self.sampler = sampler
        self.crops_per_record = int(sampler.crops_per_record)
        self._base = train.SampleSource(self.records, cache_dir=cache_dir)
        self._cached_index = -1
        self._entry: Optional[Dict[str, Any]] = None
        self._anchors: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_record

    def get(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index, crop_index = divmod(int(index), self.crops_per_record)
        if record_index != self._cached_index:
            entry = self._base.get(record_index)
            self._anchors = self.sampler.build_anchors(
                entry, record_index=record_index
            )
            self._entry = entry
            self._cached_index = record_index
        if self._entry is None or crop_index >= len(self._anchors):
            raise FullFitError("crop anchors are incomplete for this record")
        return self.sampler.crop(self._entry, self._anchors[crop_index])


def make_epoch_source(
    records: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    cache_dir: Optional[str],
    base_seed: int = BASE_SEED,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    crop_min_valid: int = CROP_MIN_VALID,
) -> Tuple[EpochCropSource, "train.SupportCropSampler"]:
    """Build a deterministic epoch-varying crop source."""

    sampler = train.SupportCropSampler(
        crop_size=int(crop_size),
        crops_per_record=int(crops_per_record),
        background_fraction=BACKGROUND_FRACTION,
        jitter=CROP_JITTER,
        min_valid=int(crop_min_valid),
        seed=epoch_seed(base_seed, epoch),
    )
    return (
        EpochCropSource(records, cache_dir=cache_dir, sampler=sampler),
        sampler,
    )


# ---------------------------------------------------------------------------
# labels (semantic only; instance is never touched)
# ---------------------------------------------------------------------------


def _entry_arrays(entry: Mapping[str, Any]) -> Tuple[Any, Any]:
    numpy_module = _require_numpy()
    semantic = numpy_module.asarray(entry["semantic"])
    valid = numpy_module.asarray(entry["semantic_valid"], dtype=bool)
    if semantic.shape != valid.shape:
        raise FullFitError(
            f"semantic/semantic_valid shape mismatch: {semantic.shape} vs {valid.shape}"
        )
    return semantic, valid


def entry_known_pixels(entry: Mapping[str, Any]) -> int:
    """Count semantic pixels with a known class 0..34."""

    semantic, valid = _entry_arrays(entry)
    known = valid & (semantic >= 0) & (semantic < pilot.NUM_CLASSES)
    return int(known.sum())


def _entry_target(entry: Mapping[str, Any]) -> Any:
    """Semantic target with everything not known-and-valid mapped to -1."""

    numpy_module = _require_numpy()
    semantic, valid = _entry_arrays(entry)
    known = valid & (semantic >= 0) & (semantic < pilot.NUM_CLASSES)
    return numpy_module.where(
        known, semantic, numpy_module.full_like(semantic, pilot.IGNORE_INDEX)
    )


def _entry_skip_reason(entry: Mapping[str, Any]) -> Optional[str]:
    if bool(entry.get("skip_training")):
        return "zero_supervision_crop"
    if entry_known_pixels(entry) == 0:
        return "no_known_semantic_pixels"
    return None


def class_coverage(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pixel coverage per semantic class over a sequence of entries."""

    numpy_module = _require_numpy()
    counts = numpy_module.zeros(pilot.NUM_CLASSES, dtype=numpy_module.int64)
    unknown = 0
    total = 0
    for entry in entries:
        semantic, valid = _entry_arrays(entry)
        known = valid & (semantic >= 0) & (semantic < pilot.NUM_CLASSES)
        values = semantic[known].astype(numpy_module.int64, copy=False)
        if values.size:
            counts += numpy_module.bincount(
                values, minlength=pilot.NUM_CLASSES
            ).astype(numpy_module.int64)
        total += int(semantic.size)
        unknown += int(semantic.size - known.sum())
    present = [
        {"class_id": index, "pixels": int(count)}
        for index, count in enumerate(counts.tolist())
        if count
    ]
    missing = [index for index, count in enumerate(counts.tolist()) if not count]
    return {
        "total_pixels": int(total),
        "known_pixels": int(counts.sum()),
        "unknown_pixels": int(unknown),
        "present_classes": present,
        "present_class_ids": [item["class_id"] for item in present],
        "missing_class_ids": missing,
        "class_pixel_counts": {
            str(index): int(count) for index, count in enumerate(counts.tolist())
        },
    }


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------


def iter_group_batches(
    source: Any, accumulation_steps: int = ACCUMULATION_STEPS
) -> Iterator[Dict[str, Any]]:
    """Yield groups of at most ``accumulation_steps`` supervised crops.

    Zero-supervision entries are dropped before any model call and do not
    consume a slot in the group.  The final partial group is yielded as-is.
    """

    if isinstance(accumulation_steps, bool) or int(accumulation_steps) < 1:
        raise FullFitError("accumulation_steps must be a positive int")
    steps = int(accumulation_steps)
    pending: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    produced = False
    for index in range(len(source)):
        entry = source.get(index)
        reason = _entry_skip_reason(entry)
        if reason is not None:
            skipped.append(
                {
                    "index": int(index),
                    "sample_id": str(entry.get("sample_id") or ""),
                    "crop_index": int(entry.get("crop_index", 0)),
                    "reason": reason,
                }
            )
            continue
        pending.append(entry)
        if len(pending) == steps:
            produced = True
            yield {"entries": pending, "skipped": skipped}
            pending = []
            skipped = []
    if pending or skipped or not produced:
        yield {"entries": pending, "skipped": skipped}


def _image_tensor(entry: Mapping[str, Any], device: str) -> Any:
    torch = pilot._require_torch()
    numpy_module = _require_numpy()
    image = numpy_module.asarray(entry["image"], dtype=numpy_module.float32)
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise FullFitError(f"image must be 3 x H x W; got {image.shape}")
    tensor = torch.from_numpy(numpy_module.ascontiguousarray(image)).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _target_tensor(entry: Mapping[str, Any], device: str) -> Any:
    torch = pilot._require_torch()
    numpy_module = _require_numpy()
    target = numpy_module.asarray(_entry_target(entry), dtype=numpy_module.int64)
    tensor = torch.from_numpy(numpy_module.ascontiguousarray(target)).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.long)


def accumulated_step(
    model: Any,
    entries: Sequence[Mapping[str, Any]],
    optimizer: Any,
    *,
    device: str = "cpu",
) -> Dict[str, Any]:
    """One optimizer step over at most one logical batch of crops.

    Every crop is forwarded and backwarded on its own, so no graph is held
    across crops.  Each crop's mean CE is scaled by ``known_crop /
    total_known`` which makes the accumulated gradient exactly the gradient of
    the pixel-weighted mean CE over the logical batch.
    """

    if not entries:
        return {
            "loss": None,
            "known_pixels": 0,
            "supervised_crops": 0,
            "optimizer_step": False,
        }
    counts = [entry_known_pixels(entry) for entry in entries]
    total_known = int(sum(counts))
    if total_known <= 0:
        return {
            "loss": None,
            "known_pixels": 0,
            "supervised_crops": len(entries),
            "optimizer_step": False,
        }
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for entry, count in zip(entries, counts):
        images = _image_tensor(entry, device)
        logits = model(images)["semantic_logits"]
        expected = (
            1,
            pilot.NUM_CLASSES,
            int(images.shape[-2]),
            int(images.shape[-1]),
        )
        if tuple(logits.shape) != expected:
            raise FullFitError(
                f"unexpected logits shape: {tuple(logits.shape)} vs {expected}"
            )
        target = _target_tensor(entry, device)
        loss, stats = pilot.masked_semantic_ce(logits, target)
        if int(stats["known_pixels"]) != int(count):
            raise FullFitError(
                "known-pixel count mismatch between target build and loss: "
                f"{stats['known_pixels']} vs {count}"
            )
        (loss * (float(count) / float(total_known))).backward()
        weighted_loss += float(loss.detach().item()) * int(count)
    optimizer.step()
    return {
        "loss": weighted_loss / float(total_known),
        "known_pixels": int(total_known),
        "supervised_crops": int(len(entries)),
        "optimizer_step": True,
    }


def train_epoch_accum(
    model: Any,
    source: Any,
    optimizer: Any,
    *,
    device: str = "cpu",
    accumulation_steps: int = ACCUMULATION_STEPS,
    epoch: int = 1,
) -> Dict[str, Any]:
    """One deterministic epoch with gradient accumulation."""

    torch = pilot._require_torch()
    model.train()
    frozen_bn = pilot.freeze_batch_norm(model)
    started = time.monotonic()
    scheduled_crops = 0
    supervised_crops = 0
    skipped_crops = 0
    skipped_batches = 0
    zero_supervision_crops = 0
    optimizer_steps = 0
    known_total = 0
    weighted_loss = 0.0

    for group in iter_group_batches(source, accumulation_steps):
        scheduled_crops += len(group["entries"]) + len(group["skipped"])
        skipped_crops += len(group["skipped"])
        zero_supervision_crops += sum(
            1
            for record in group["skipped"]
            if record["reason"] == "zero_supervision_crop"
        )
        if not group["entries"]:
            skipped_batches += 1
            continue
        result = accumulated_step(
            model, group["entries"], optimizer, device=device
        )
        if not result["optimizer_step"]:
            skipped_batches += 1
            continue
        optimizer_steps += 1
        supervised_crops += int(result["supervised_crops"])
        known_total += int(result["known_pixels"])
        weighted_loss += float(result["loss"]) * int(result["known_pixels"])

    elapsed = time.monotonic() - started
    return {
        "epoch": int(epoch),
        "scheduled_crops": int(scheduled_crops),
        "supervised_crops": int(supervised_crops),
        "skipped_crops": int(skipped_crops),
        "skipped_batches": int(skipped_batches),
        "zero_supervision_crops": int(zero_supervision_crops),
        "optimizer_steps": int(optimizer_steps),
        "samples_trained": int(supervised_crops),
        "known_pixels": int(known_total),
        "loss": (weighted_loss / known_total) if known_total else None,
        "micro_batch_size": 1,
        "accumulation_steps": int(accumulation_steps),
        "batch_norm_frozen_modules": int(frozen_bn),
        "seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# tiled source-resolution inference
# ---------------------------------------------------------------------------


def _tile_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= 0 or tile <= 0 or stride <= 0:
        raise FullFitError("tile geometry must be positive")
    if stride > tile:
        raise FullFitError("stride must not exceed tile size")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def predict_tiled_image(
    model: Any,
    image: Any,
    *,
    device: str = "cpu",
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> Dict[str, Any]:
    """Predict logits for one source-resolution image with overlap averaging.

    ``image`` is 3 x H x W in [0, 1] (numpy array or torch tensor).  No resize
    is applied: the window is exactly ``tile_size`` and the bottom/right edge
    is padded with white (1.0) only when the image is smaller than a tile.
    Labels are not accepted by this API and are never decoded.
    """

    torch = pilot._require_torch()
    numpy_module = _require_numpy()
    if isinstance(image, torch.Tensor):
        source = image.detach().to("cpu", dtype=torch.float32)
    else:
        source = torch.from_numpy(
            numpy_module.ascontiguousarray(
                numpy_module.asarray(image, dtype=numpy_module.float32)
            )
        )
    if source.ndim == 2:
        source = source.unsqueeze(0)
    if source.ndim != 3 or int(source.shape[0]) != 3:
        raise FullFitError(f"image must be 3 x H x W; got {tuple(source.shape)}")
    if not bool(torch.isfinite(source).all()):
        raise FullFitError("non_finite_image")
    height, width = int(source.shape[-2]), int(source.shape[-1])
    if height <= 0 or width <= 0:
        raise FullFitError("empty image")
    padded_height = max(int(tile_size), height)
    padded_width = max(int(tile_size), width)
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    padded = source
    if pad_bottom or pad_right:
        padded = torch.nn.functional.pad(
            source.unsqueeze(0),
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=1.0,
        ).squeeze(0)

    rows = _tile_positions(padded_height, int(tile_size), int(stride))
    columns = _tile_positions(padded_width, int(tile_size), int(stride))
    was_training = bool(getattr(model, "training", False))
    model.eval()
    accum = None
    weights = None
    try:
        with torch.no_grad():
            for row in rows:
                for column in columns:
                    window = padded[
                        :,
                        row : row + int(tile_size),
                        column : column + int(tile_size),
                    ]
                    window = window.unsqueeze(0).to(device=device)
                    logits = model(window)["semantic_logits"]
                    expected = (
                        1,
                        pilot.NUM_CLASSES,
                        int(tile_size),
                        int(tile_size),
                    )
                    if tuple(logits.shape) != expected:
                        raise FullFitError(
                            "unexpected tile logits shape: "
                            f"{tuple(logits.shape)} vs {expected}"
                        )
                    if accum is None:
                        accum = torch.zeros(
                            (1, pilot.NUM_CLASSES, padded_height, padded_width),
                            dtype=logits.dtype,
                            device=logits.device,
                        )
                        weights = torch.zeros(
                            (1, 1, padded_height, padded_width),
                            dtype=logits.dtype,
                            device=logits.device,
                        )
                    accum[
                        :, :, row : row + int(tile_size), column : column + int(tile_size)
                    ] += logits
                    weights[
                        :, :, row : row + int(tile_size), column : column + int(tile_size)
                    ] += 1.0
    finally:
        if was_training:
            model.train()

    if accum is None or weights is None:  # pragma: no cover - defensive
        raise FullFitError("no tiles were produced")
    averaged = accum / weights
    logits_out = averaged[:, :, :height, :width]
    return {
        "schema": TILED_SCHEMA,
        "semantic_logits": logits_out,
        "tile_count": int(len(rows) * len(columns)),
        "tile_rows": rows,
        "tile_columns": columns,
        "tile_size": int(tile_size),
        "stride": int(stride),
        "source_size": [int(height), int(width)],
        "padded_size": [int(padded_height), int(padded_width)],
        "padding": [int(pad_bottom), int(pad_right)],
        "policy": dict(TILING_POLICY),
    }


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def check_disk_cap(
    used_bytes: int, cap_bytes: int = DISK_CAP_BYTES
) -> Dict[str, Any]:
    """Report cache growth against the configured disk cap."""

    used = int(used_bytes)
    cap = int(cap_bytes)
    if used < 0 or cap <= 0:
        raise FullFitError("disk byte counts must be positive")
    return {
        "used_bytes": used,
        "cap_bytes": cap,
        "remaining_bytes": cap - used,
        "within_cap": used <= cap,
    }


def _supervision_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    crop_min_valid: int = CROP_MIN_VALID,
) -> List[Dict[str, Any]]:
    """Exact per-epoch supervised/skipped counts using cached arrays."""

    plan: List[Dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        source, sampler = make_epoch_source(
            records,
            epoch=epoch,
            cache_dir=cache_dir,
            base_seed=base_seed,
            crop_size=crop_size,
            crops_per_record=crops_per_record,
            crop_min_valid=crop_min_valid,
        )
        scheduled = 0
        supervised = 0
        skipped = 0
        zero_supervision = 0
        for index in range(len(source)):
            entry = source.get(index)
            scheduled += 1
            reason = _entry_skip_reason(entry)
            if reason is None:
                supervised += 1
            else:
                skipped += 1
                if reason == "zero_supervision_crop":
                    zero_supervision += 1
        plan.append(
            {
                "epoch": int(epoch),
                "epoch_seed": int(epoch_seed(base_seed, epoch)),
                "scheduled_crops": int(scheduled),
                "supervised_crops": int(supervised),
                "skipped_crops": int(skipped),
                "zero_supervision_crops": int(zero_supervision),
                "planned_optimizer_steps": int(
                    math.ceil(supervised / float(ACCUMULATION_STEPS))
                ),
            }
        )
    return plan


def prepare_fullfit(
    manifest_path: str,
    *,
    out: str,
    role: str,
    train_scale: int = TRAIN_SCALE,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Build the full-fit array cache and report coverage/step plan."""

    train.require_command_role("fit", role)
    if isinstance(limit, bool) or (limit is not None and int(limit) < 1):
        raise FullFitError(f"limit must be a positive int; got {limit!r}")
    os.makedirs(out, exist_ok=True)
    records = load_fit_records(manifest_path, train_scale=int(train_scale))
    partial = False
    if limit is not None and int(limit) < len(records):
        records = records[: int(limit)]
        partial = True
    cache = cache_dir or os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    numpy_module = _require_numpy()
    base = train.SampleSource(records, cache_dir=cache)

    class_counts = numpy_module.zeros(pilot.NUM_CLASSES, dtype=numpy_module.int64)
    total_pixels = 0
    unknown_pixels = 0
    groups: Counter = Counter()
    dimensions: Counter = Counter()
    cache_bytes = 0
    cache_files = 0
    started = time.monotonic()

    for index, record in enumerate(records):
        entry = base.get(index)
        semantic, valid = _entry_arrays(entry)
        known = valid & (semantic >= 0) & (semantic < pilot.NUM_CLASSES)
        values = semantic[known].astype(numpy_module.int64, copy=False)
        if values.size:
            class_counts += numpy_module.bincount(
                values, minlength=pilot.NUM_CLASSES
            ).astype(numpy_module.int64)
        total_pixels += int(semantic.size)
        unknown_pixels += int(semantic.size - known.sum())
        groups[str(record.get("group_id") or "")] += 1
        dimensions[
            (
                int(entry.get("source_height") or semantic.shape[0]),
                int(entry.get("source_width") or semantic.shape[1]),
            )
        ] += 1
        path = base._cache_path(index)
        if path and os.path.isfile(path):
            cache_bytes += int(os.path.getsize(path))
            cache_files += 1
        cap = check_disk_cap(cache_bytes)
        if progress and (index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "prepared": index + 1,
                        "records": len(records),
                        "cache_bytes": cache_bytes,
                        "seconds": round(time.monotonic() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not cap["within_cap"]:
            payload = {
                "schema": PREPARE_SCHEMA,
                "status": "blocked",
                "reason": "cache_disk_cap_exceeded",
                "role": "fit",
                "data": {
                    "records": len(records),
                    "groups": len(groups),
                    "cache_dir": os.path.abspath(cache),
                    "cache_files": cache_files,
                    "cache_bytes": cache_bytes,
                    "disk_cap_bytes": DISK_CAP_BYTES,
                },
                "policy": dict(TRAIN_POLICY),
            }
            _write_json(os.path.join(out, "fullfit_prepare.json"), payload)
            return payload

    present = [
        {"class_id": index, "pixels": int(count)}
        for index, count in enumerate(class_counts.tolist())
        if count
    ]
    missing = [
        index for index, count in enumerate(class_counts.tolist()) if not count
    ]
    plan = _supervision_plan(
        records,
        cache_dir=cache,
        epochs=EPOCHS,
        base_seed=BASE_SEED,
        crop_size=CROP_SIZE,
        crops_per_record=CROPS_PER_RECORD,
        crop_min_valid=CROP_MIN_VALID,
    )
    total_steps = sum(item["planned_optimizer_steps"] for item in plan)
    image_sizes = [
        {"height": height, "width": width, "records": int(count)}
        for (height, width), count in sorted(dimensions.items())
    ]
    payload = {
        "schema": PREPARE_SCHEMA,
        "status": "ready",
        "role": "fit",
        "partial": bool(partial),
        "config": {
            "train_scale": int(train_scale),
            "crop_size": CROP_SIZE,
            "crops_per_record": CROPS_PER_RECORD,
            "crop_min_valid": CROP_MIN_VALID,
            "epochs": EPOCHS,
            "accumulation_steps": ACCUMULATION_STEPS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "base_seed": BASE_SEED,
            "epoch_seed_stride": EPOCH_SEED_STRIDE,
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
        },
        "data": {
            "records": len(records),
            "groups": len(groups),
            "group_counts": dict(sorted(groups.items())),
            "image_sizes": image_sizes,
            "cache_dir": os.path.abspath(cache),
            "cache_files": cache_files,
            "cache_bytes": cache_bytes,
            "disk_cap_bytes": DISK_CAP_BYTES,
            "disk_cap_status": check_disk_cap(cache_bytes),
            "manifest": os.path.abspath(manifest_path),
            "seconds": round(time.monotonic() - started, 3),
        },
        "coverage": {
            "total_pixels": int(total_pixels),
            "known_pixels": int(class_counts.sum()),
            "unknown_pixels": int(unknown_pixels),
            "present_classes": present,
            "present_class_ids": [item["class_id"] for item in present],
            "missing_class_ids": missing,
            "class_pixel_counts": {
                str(index): int(count)
                for index, count in enumerate(class_counts.tolist())
            },
        },
        "supervision": {
            "epochs": plan,
            "planned_optimizer_steps_total": int(total_steps),
            "scheduled_crops_per_epoch": len(records) * CROPS_PER_RECORD,
            "effective_batch_crops": ACCUMULATION_STEPS,
            "micro_batch_size": 1,
        },
        "estimates": {
            "note": (
                "estimates only; validate on the parent GPU. Prior 24-record "
                "pilot measured about 0.19 s per 256 px crop including cache reads."
            ),
            "seconds_per_crop_assumed": 0.19,
            "estimated_epoch_seconds": round(
                len(records) * CROPS_PER_RECORD * 0.19, 1
            ),
            "estimated_total_seconds": round(
                EPOCHS * len(records) * CROPS_PER_RECORD * 0.19, 1
            ),
            "estimated_peak_allocated_mb": [1000, 1600],
            "first_epoch_note": "cache is already warm after prepare",
        },
        "commands": {
            "prepare": (
                "python -m cadvec.recognition_pretrained_fullfit prepare "
                f"--manifest \"{os.path.abspath(manifest_path)}\" --role fit "
                f"--out \"{os.path.abspath(out)}\""
            ),
            "fit": (
                "python -m cadvec.recognition_pretrained_fullfit fit "
                f"--manifest \"{os.path.abspath(manifest_path)}\" --role fit "
                f"--out \"{os.path.abspath(out)}\" --device cuda"
            ),
        },
        "policy": dict(TRAIN_POLICY),
        "tiling": dict(TILING_POLICY),
    }
    _write_json(os.path.join(out, "fullfit_prepare.json"), payload)
    return payload


# ---------------------------------------------------------------------------
# future fit entry point (parent schedules the GPU; never started here)
# ---------------------------------------------------------------------------


def _merge_sampler_stats(
    totals: Dict[str, Any], stats: Mapping[str, Any]
) -> None:
    for key, value in stats.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            totals[key] = int(totals.get(key, 0)) + int(value)
        elif isinstance(value, Mapping):
            target = totals.setdefault(key, {})
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, int):
                    target[str(inner_key)] = int(target.get(str(inner_key), 0)) + int(
                        inner_value
                    )


def run_fit(
    manifest_path: str,
    *,
    role: str,
    out: str,
    device: str = "auto",
    epochs: int = EPOCHS,
    lr: float = LR,
    base_seed: int = BASE_SEED,
    train_scale: int = TRAIN_SCALE,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    crop_min_valid: int = CROP_MIN_VALID,
    max_fit_samples: Optional[int] = None,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    weights_cache: Optional[str] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Train on every fit record with the fixed plan (parent-scheduled)."""

    train.require_command_role("fit", role)
    if isinstance(epochs, bool) or int(epochs) < 1:
        raise FullFitError(f"epochs must be a positive int; got {epochs!r}")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0:
        raise FullFitError(f"lr must be positive; got {lr!r}")
    if max_fit_samples is not None and (
        isinstance(max_fit_samples, bool) or int(max_fit_samples) < 1
    ):
        raise FullFitError("max_fit_samples must be a positive int")

    torch = pilot._require_torch()
    os.makedirs(out, exist_ok=True)
    records = load_fit_records(manifest_path, train_scale=int(train_scale))
    if max_fit_samples is not None and int(max_fit_samples) < len(records):
        records = records[: int(max_fit_samples)]
    target_device = train.resolve_device(device)
    model, provenance = pilot.build_pilot_model(
        pretrained=bool(pretrained),
        weights_path=weights_path,
        weights_cache=weights_cache,
        progress=progress,
    )
    model.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY
    )
    cache = os.path.join(out, "cache")
    history: List[Dict[str, Any]] = []
    crop_totals: Dict[str, Any] = {}
    crop_stats_per_epoch: List[Dict[str, Any]] = []
    started = time.monotonic()

    for epoch in range(1, int(epochs) + 1):
        train.seed_everything(epoch_seed(base_seed, epoch))
        source, sampler = make_epoch_source(
            records,
            epoch=epoch,
            cache_dir=cache,
            base_seed=base_seed,
            crop_size=int(crop_size),
            crops_per_record=int(crops_per_record),
            crop_min_valid=int(crop_min_valid),
        )
        metrics = train_epoch_accum(
            model,
            source,
            optimizer,
            device=target_device,
            accumulation_steps=ACCUMULATION_STEPS,
            epoch=epoch,
        )
        sampler_stats = sampler.stats()
        _merge_sampler_stats(crop_totals, sampler_stats)
        crop_stats_per_epoch.append(
            {"epoch": epoch, "epoch_seed": epoch_seed(base_seed, epoch)}
        )
        history.append(metrics)
        if progress:
            print(json.dumps(metrics, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    final_metrics = history[-1]
    config = {
        "train_scale": int(train_scale),
        "crop_size": int(crop_size),
        "crops_per_record": int(crops_per_record),
        "crop_min_valid": int(crop_min_valid),
        "epochs": int(epochs),
        "micro_batch_size": 1,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_crops": ACCUMULATION_STEPS,
        "optimizer": "AdamW",
        "lr": float(lr),
        "weight_decay": WEIGHT_DECAY,
        "base_seed": int(base_seed),
        "epoch_seed_stride": EPOCH_SEED_STRIDE,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "class_weights": None,
        "amp": "not used",
        "device": target_device,
    }
    peak_mb = None
    if target_device == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
    checkpoint = os.path.join(out, "checkpoint_fullfit.pt")
    pilot.save_checkpoint(
        checkpoint,
        model,
        epoch=int(epochs),
        epochs=int(epochs),
        config=config,
        provenance=provenance,
        train_metrics=final_metrics,
    )
    metrics_payload = {
        "schema": FIT_SCHEMA,
        "status": "ok",
        "role": "fit",
        "config": config,
        "history": history,
        "selection": "final_epoch_only",
        "tune_evaluation": "not performed (explicit evaluation is outside this lane)",
    }
    _write_json(os.path.join(out, "metrics.json"), metrics_payload)
    fit_manifest = {
        "schema": FIT_SCHEMA,
        "status": "ok",
        "role": "fit",
        "config": config,
        "data": {
            "fit_records": len(records),
            "groups": len({str(r.get("group_id") or "") for r in records}),
            "cache_dir": os.path.abspath(cache),
        },
        "policy": dict(TRAIN_POLICY),
        "tiling": dict(TILING_POLICY),
        "crop_sampling": crop_stats_per_epoch,
        "crop_stats": crop_totals,
        "history": history,
        "actual": {
            "seconds": round(elapsed, 3),
            "optimizer_steps": int(sum(h["optimizer_steps"] for h in history)),
            "skipped_crops": int(sum(h["skipped_crops"] for h in history)),
            "peak_cuda_memory_mb": peak_mb,
        },
        "provenance": provenance,
        "final": {
            "checkpoint": os.path.abspath(checkpoint),
            "epoch": int(epochs),
        },
        "selection": "final_epoch_only",
        "tune_evaluation": "not performed",
    }
    _write_json(os.path.join(out, "fit_manifest.json"), fit_manifest)
    return fit_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-fit pretrained DeepLabV3 lane: prepare the fit cache, then "
            "train on every fit record with fixed crop/accumulation settings."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="build the fit array cache and coverage/step plan"
    )
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--role", required=True)
    prepare.add_argument("--train-scale", type=int, default=TRAIN_SCALE)
    prepare.add_argument("--cache-dir", default=None)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--progress", action="store_true")

    fit = sub.add_parser(
        "fit", help="train on every fit record (parent-scheduled GPU)"
    )
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--role", required=True)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--epochs", type=int, default=EPOCHS)
    fit.add_argument("--lr", type=float, default=LR)
    fit.add_argument("--seed", type=int, default=BASE_SEED)
    fit.add_argument("--train-scale", type=int, default=TRAIN_SCALE)
    fit.add_argument("--crop-size", type=int, default=CROP_SIZE)
    fit.add_argument("--crops-per-record", type=int, default=CROPS_PER_RECORD)
    fit.add_argument("--crop-min-valid", type=int, default=CROP_MIN_VALID)
    fit.add_argument("--max-fit-samples", type=int, default=None)
    fit.add_argument("--no-pretrained", action="store_true")
    fit.add_argument("--weights-path", default=None)
    fit.add_argument("--weights-cache", default=None)
    fit.add_argument("--progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            payload = prepare_fullfit(
                args.manifest,
                out=args.out,
                role=args.role,
                train_scale=args.train_scale,
                cache_dir=args.cache_dir,
                limit=args.limit,
                progress=args.progress,
            )
        elif args.command == "fit":
            payload = run_fit(
                args.manifest,
                out=args.out,
                role=args.role,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                base_seed=args.seed,
                train_scale=args.train_scale,
                crop_size=args.crop_size,
                crops_per_record=args.crops_per_record,
                crop_min_valid=args.crop_min_valid,
                max_fit_samples=args.max_fit_samples,
                pretrained=not args.no_pretrained,
                weights_path=args.weights_path,
                weights_cache=args.weights_cache,
                progress=args.progress,
            )
        else:  # pragma: no cover - argparse enforces the choices
            raise FullFitError(f"unknown command: {args.command!r}")
    except (FullFitError, train.RoleAccessError, train.RasterTrainError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in ("ready", "ok") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
