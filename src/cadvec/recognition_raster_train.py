"""Fit/tune-only train and evaluation runner for the raster U-Net route.

The module is a *runner*: it owns no labels, no model code and no scorer.  It
composes three existing, already-tested layers:

* :mod:`cadvec.recognition_raster_data` for the role-filtered sample listing and
  the immutable ground-truth rasters (independent semantic/instance masks);
* :mod:`cadvec.recognition_raster_model` for the full-resolution U-Net and the
  masked semantic / instance-embedding losses;
* :mod:`cadvec.recognition_contract` for role names and error types.

Hard invariants enforced here:

* the ``fit`` command accepts exactly ``--role fit``; the ``evaluate`` command
  accepts exactly ``--role tune``.  ``eval``, ``reserve``, ``test`` and ``dev``
  are refused *before* the manifest is opened;
* no evaluation happens during training.  Fit records fit-only losses per
  epoch, then performs exactly one tune evaluation on the final checkpoint for
  checkpoint selection;
* sample geometry is never resized: the model sees the full source PNG
  resolution (1000 px square in the competition data) with batch size 1;
* loss masks stay strictly independent: unknown labels are excluded, never
  turned into negatives or background;
* augmentations are disabled by default and the policy is recorded in the
  run manifest.

PyTorch is optional at import time.  When it is absent, ``fit``/``evaluate``
return a ``blocked`` result *without reading any data*; nothing is downloaded
or installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # optional at import time, mirrors recognition_raster_data
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore

from . import recognition_raster_data as raster_data
from . import recognition_raster_model as raster_model
from .recognition_contract import (
    ROLE_EVAL,
    ROLE_FIT,
    ROLE_RESERVE,
    ROLE_TUNE,
    RecognitionError,
    RoleAccessError,
    SEED,
)

SCHEMA = "cadvec.recognition.raster_train/1"
CHECKPOINT_SCHEMA = "cadvec.recognition.raster_checkpoint/1"
METRICS_SCHEMA = "cadvec.recognition.raster_train_metrics/1"
TUNE_EVAL_SCHEMA = "cadvec.recognition.raster_tune_eval/1"
ALIGNMENT_SCHEMA = "cadvec.recognition.raster_alignment/1"

DEFAULT_TRAIN_SCALE = 32
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_INSTANCE_WEIGHT = 1.0

#: Fit-only support-anchored crop sampling is opt-in; 0 keeps full frames.
DEFAULT_CROP_SIZE = 0
DEFAULT_CROPS_PER_RECORD = 1
DEFAULT_CROP_BACKGROUND_FRACTION = 0.0
DEFAULT_CROP_JITTER = 0
DEFAULT_CROP_MIN_VALID = 0

#: Padding is always the "unknown" value; the image is padded with white so
#: padded pixels look like the source background instead of synthetic ink.
CROP_IMAGE_PAD_VALUE = 1.0

#: Bounded, deterministic correction against zero-valid support crops: a crop
#: whose window still contains no valid pixel may try at most this many
#: candidate centers, and at most this many support pixels per crop index.
CROP_MAX_CANDIDATES = 12
CROP_SUPPORT_PROBE_LIMIT = 16

#: Only ``fit`` may train and only ``tune`` may be evaluated.
COMMAND_ROLES = {"fit": ROLE_FIT, "evaluate": ROLE_TUNE}

#: Anything else is refused by name so errors are explicit, not accidental.
FORBIDDEN_ROLES = (ROLE_EVAL, ROLE_RESERVE, "test", "dev", "test_final", "train_final")

PQ_UNAVAILABLE_REASON = "vectorizer_coordinate_alignment_unverified"
PQ_NO_VECTORIZER_REASON = "vectorizer_not_available"

AUGMENTATION_POLICY = {
    "enabled": False,
    "rationale": (
        "random small augmentations are only permitted when the identical "
        "transform is applied to the image and every target mask; none are "
        "implemented in this runner yet"
    ),
    "implemented": [],
}

TRAIN_POLICY = {
    "batch_size_cap": DEFAULT_BATCH_SIZE,
    "full_source_resolution": True,
    "resize": None,
    "loss_masks": "independent semantic/instance masks; unknown excluded, never negative",
    "unknown_as_negative": False,
    "checkpoint_selection": "single_tune_eval_on_final_epoch",
    "tune_eval_per_epoch": False,
    "augmentation": AUGMENTATION_POLICY,
    "fit_crop_sampling": (
        "opt-in via --crop-size > 0; disabled by default; anchors use fit "
        "semantic_valid targets only; tune evaluation stays full-frame"
    ),
    "fit_crop_zero_supervision": (
        "fit crops with no valid semantic and no valid instance pixel are "
        "skipped before forward/backward; labels stay unknown and no synthetic "
        "negative or replacement crop is created"
    ),
    "amp": "cuda_only_when_requested",
    "optimizer": "AdamW",
}


class RasterTrainError(RecognitionError, ValueError):
    code = "raster_train_error"


# ---------------------------------------------------------------------------
# dependency / role guards


def dependency_status() -> Dict[str, Any]:
    """Combined torch + raster-data dependency report; installs nothing."""

    status = dict(raster_model.dependency_status())
    status["numpy"] = np is not None
    status["pillow"] = getattr(raster_data, "Image", None) is not None
    status["data_ready"] = status["numpy"] and status["pillow"]
    status["ready"] = bool(status["available"] and status["data_ready"])
    return status


def require_command_role(command: str, role: str) -> str:
    """Return the frozen role for *command* or raise before any file access."""

    if not isinstance(command, str) or command not in COMMAND_ROLES:
        raise RasterTrainError(f"unknown command: {command!r}")
    expected = COMMAND_ROLES[command]
    if not isinstance(role, str) or role != expected:
        raise RoleAccessError(
            f"{command} requires --role {expected} exactly; refused role {role!r}"
        )
    return expected


def _require_role(role: str) -> str:
    if role not in (ROLE_FIT, ROLE_TUNE):
        raise RoleAccessError(
            f"raster runner only accepts {ROLE_FIT!r} or {ROLE_TUNE!r}; got {role!r}"
        )
    return role


def _require_torch() -> Any:
    raster_model.require_torch()
    import torch  # imported only after require_torch() succeeded

    return torch


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover - environment dependent
        raise RasterTrainError("numpy is required for the raster runner")
    return np


def runner_smoke() -> Dict[str, Any]:
    """Dependency/interface smoke test; never reads a manifest or a sample."""

    status = dependency_status()
    if not status["ready"]:
        missing = [
            name
            for name in ("torch", "numpy", "pillow")
            if not bool(status.get(name))
        ]
        return {
            "schema": SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "missing_dependency",
            "missing": missing,
            "dependency_status": status,
            "trained": False,
            "no_eval_access": True,
        }
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "ready",
        "dependency_status": status,
        "trained": False,
        "no_eval_access": True,
    }


# ---------------------------------------------------------------------------
# role-filtered data adapters
#
# These two functions are the only bridge to the data layer.  Keeping them
# isolated means a future loader signature change lands here and nowhere else.


def role_sample_records(
    manifest_path: str,
    role: str,
    *,
    train_scale: int = DEFAULT_TRAIN_SCALE,
) -> List[Dict[str, Any]]:
    """Metadata-only fit/tune records; the role is checked first."""

    _require_role(role)
    return raster_data.load_role_samples(manifest_path, role, train_scale)


def build_role_sample(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one full-resolution sample through the data layer."""

    role = record.get("role")
    if role not in (ROLE_FIT, ROLE_TUNE):
        raise RoleAccessError(f"sample record has a non fit/tune role: {role!r}")
    return raster_data.build_sample(
        str(record["png_path"]), str(record["svg_path"]), str(record["sample_id"])
    )


def _as_sample_arrays(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only what training needs; drop segments/GT documents early."""

    numpy = _require_numpy()
    metadata = sample.get("metadata") or {}
    return {
        "image": numpy.asarray(sample["image"], dtype=numpy.float32),
        "semantic": numpy.asarray(sample["semantic"], dtype=numpy.int16),
        "semantic_valid": numpy.asarray(sample["semantic_valid"], dtype=bool),
        "instance": numpy.asarray(sample["instance"], dtype=numpy.int16),
        "instance_valid": numpy.asarray(sample["instance_valid"], dtype=bool),
        "sample_id": str(metadata.get("sample_id") or ""),
        "source_width": int(metadata.get("source_width") or 0),
        "source_height": int(metadata.get("source_height") or 0),
    }


class SampleSource:
    """Lazy, optionally disk-cached fit sample provider.

    Holding every 1000x1000 sample in RAM would exceed the 6 GB budget, so the
    first access builds the sample from the data layer and the result is cached
    as a plain ``.npz`` (uint8 image, int16 labels, uint8 masks).  Later epochs
    read the cache instead of re-rasterising the SVG.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        cache_dir: Optional[str] = None,
        crop_sampler: Optional["SupportCropSampler"] = None,
    ) -> None:
        self.records = list(records)
        self.cache_dir = cache_dir
        self.crop_sampler = crop_sampler
        self.crops_per_record = (
            int(crop_sampler.crops_per_record) if crop_sampler is not None else 1
        )
        self._anchor_cache: Dict[int, List[Dict[str, Any]]] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_record

    def _cache_path(self, index: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        record = self.records[index]
        digest = hashlib.sha256(
            f"{record.get('role')}\x00{record.get('sample_id')}\x00"
            f"{record.get('png_path')}\x00{record.get('svg_path')}".encode("utf-8")
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.npz")

    def get(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index, crop_index = divmod(int(index), self.crops_per_record)
        entry = self._load_record(record_index)
        if self.crop_sampler is None:
            return entry
        anchors = self._anchor_cache.get(record_index)
        if anchors is None:
            anchors = self.crop_sampler.build_anchors(entry, record_index=record_index)
            self._anchor_cache[record_index] = anchors
        return self.crop_sampler.crop(entry, anchors[crop_index])

    def _load_record(self, record_index: int) -> Dict[str, Any]:
        numpy = _require_numpy()
        path = self._cache_path(record_index)
        if path and os.path.isfile(path):
            with numpy.load(path, allow_pickle=False) as cached:
                return {
                    "image": cached["image"].astype(numpy.float32) / 255.0,
                    "semantic": cached["semantic"].astype(numpy.int16),
                    "semantic_valid": cached["semantic_valid"].astype(bool),
                    "instance": cached["instance"].astype(numpy.int16),
                    "instance_valid": cached["instance_valid"].astype(bool),
                    "sample_id": str(cached["sample_id"]),
                    "source_width": int(cached["source_width"]),
                    "source_height": int(cached["source_height"]),
                }
        arrays = _as_sample_arrays(build_role_sample(self.records[record_index]))
        if path:
            numpy.savez(
                path,
                image=numpy.clip(numpy.rint(arrays["image"] * 255.0), 0, 255).astype(
                    numpy.uint8
                ),
                semantic=arrays["semantic"],
                semantic_valid=arrays["semantic_valid"].astype(numpy.uint8),
                instance=arrays["instance"],
                instance_valid=arrays["instance_valid"].astype(numpy.uint8),
                sample_id=numpy.asarray(arrays["sample_id"]),
                source_width=numpy.asarray(arrays["source_width"]),
                source_height=numpy.asarray(arrays["source_height"]),
            )
        return arrays


# ---------------------------------------------------------------------------
# fit-only support-anchored crop sampling
#
# The sampler is a *view* over a full-resolution fit sample: it never resizes,
# never writes to the source arrays and never fabricates labels.  Window
# coordinates come from the fit ground-truth semantic mask only; the model
# still receives the image tensor alone.


def normalize_crop_config(
    crop_size: int = DEFAULT_CROP_SIZE,
    crops_per_record: int = DEFAULT_CROPS_PER_RECORD,
    background_fraction: float = DEFAULT_CROP_BACKGROUND_FRACTION,
    jitter: int = DEFAULT_CROP_JITTER,
    min_valid: int = DEFAULT_CROP_MIN_VALID,
) -> Dict[str, Any]:
    """Validate crop options; ``crop_size <= 0`` means disabled."""

    def _int_option(name: str, value: Any, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RasterTrainError(f"{name} must be an int; got {value!r}")
        if value < minimum:
            raise RasterTrainError(f"{name} must be >= {minimum}; got {value}")
        return int(value)

    size = _int_option("--crop-size", crop_size, 0)
    per_record = _int_option("--crops-per-record", crops_per_record, 1)
    jitter_px = _int_option("--crop-jitter", jitter, 0)
    min_valid_px = _int_option("--crop-min-valid", min_valid, 0)
    if isinstance(background_fraction, bool) or not isinstance(
        background_fraction, (int, float)
    ):
        raise RasterTrainError(
            f"--crop-background-fraction must be a number; got {background_fraction!r}"
        )
    fraction = float(background_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise RasterTrainError(
            f"--crop-background-fraction must be within [0, 1]; got {fraction}"
        )
    return {
        "enabled": size > 0,
        "crop_size": size,
        "crops_per_record": per_record,
        "background_fraction": fraction,
        "jitter": jitter_px,
        "min_valid": min_valid_px,
    }


def _disabled_crop_policy() -> Dict[str, Any]:
    return {
        **normalize_crop_config(),
        "reason": "disabled_default_full_frame",
        "resize": None,
        "tune_eval": "full frames unchanged",
        "sampling_correction": None,
    }


class SupportCropSampler:
    """Deterministic, class-stratified fit crop sampler.

    Anchors are drawn from ``semantic_valid & semantic >= 0`` pixels of the
    *fit* ground truth: crops are round-robin across the classes present in each
    record, and all randomness is derived from hashes of
    ``(seed, sample_id, record_index, crop_index)`` so repeated runs and
    repeated epochs produce the same crops.  An optional background fraction
    anchors some crops on pixels without semantic support; those crops keep
    ``-1``/``False`` labels and never become negative examples.
    """

    def __init__(
        self,
        *,
        crop_size: int,
        crops_per_record: int = DEFAULT_CROPS_PER_RECORD,
        background_fraction: float = DEFAULT_CROP_BACKGROUND_FRACTION,
        jitter: int = DEFAULT_CROP_JITTER,
        min_valid: int = DEFAULT_CROP_MIN_VALID,
        seed: int = SEED,
    ) -> None:
        config = normalize_crop_config(
            crop_size=crop_size,
            crops_per_record=crops_per_record,
            background_fraction=background_fraction,
            jitter=jitter,
            min_valid=min_valid,
        )
        if not config["enabled"]:
            raise RasterTrainError("SupportCropSampler requires --crop-size >= 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RasterTrainError(f"crop seed must be an int; got {seed!r}")
        self.seed = int(seed)
        self.crop_size = config["crop_size"]
        self.crops_per_record = config["crops_per_record"]
        self.background_fraction = config["background_fraction"]
        self.jitter = config["jitter"]
        self.min_valid = config["min_valid"]
        self._stats: Dict[str, Any] = {
            "crops_total": 0,
            "records_seen": 0,
            "support_records": 0,
            "no_support_records": 0,
            "support_crops": 0,
            "background_crops": 0,
            "no_support_fallback_crops": 0,
            "padded_crops": 0,
            "below_min_valid_crops": 0,
            "min_valid_satisfied_crops": 0,
            "support_guard_failed_crops": 0,
            "zero_supervision_crops": 0,
            "anchor_probes": 0,
            "acceptance_attempts": 0,
            "class_counts": {},
            "unknown_pixels": 0,
            "valid_semantic_pixels": 0,
            "valid_instance_pixels": 0,
        }

    # -- deterministic RNG ------------------------------------------------

    def _rng(self, *parts: Any) -> Any:
        numpy = _require_numpy()
        material = "\x00".join(str(part) for part in (self.seed,) + parts)
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return numpy.random.default_rng(
            int.from_bytes(digest[:8], "little", signed=False)
        )

    # -- anchors ----------------------------------------------------------

    def build_anchors(
        self, entry: Mapping[str, Any], *, record_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Return exactly ``crops_per_record`` anchors built from fit targets."""

        numpy = _require_numpy()
        semantic = numpy.asarray(entry["semantic"])
        semantic_valid = numpy.asarray(entry["semantic_valid"], dtype=bool)
        valid = semantic_valid & (semantic >= 0)
        height, width = int(semantic.shape[0]), int(semantic.shape[1])
        if height < 1 or width < 1:
            raise RasterTrainError("crop sampler received an empty semantic mask")
        sample_id = str(entry.get("sample_id") or f"record-{record_index}")
        self._stats["records_seen"] += 1

        background_count = min(
            self.crops_per_record,
            int(round(self.crops_per_record * self.background_fraction)),
        )
        support_count = self.crops_per_record - background_count
        present = (
            [int(value) for value in numpy.unique(semantic[valid]).tolist()]
            if bool(valid.any())
            else []
        )
        if present:
            self._stats["support_records"] += 1
        else:
            self._stats["no_support_records"] += 1

        anchors: List[Dict[str, Any]] = []
        for crop_index in range(support_count):
            rng = self._rng(sample_id, record_index, crop_index, "anchor")
            if present:
                class_id: Optional[int] = present[crop_index % len(present)]
                pixels = numpy.flatnonzero(valid & (semantic == class_id))
                chosen = int(pixels[int(rng.integers(pixels.size))])
                row, column = divmod(chosen, width)
                kind = "support"
            else:
                class_id = None
                row = int(rng.integers(height))
                column = int(rng.integers(width))
                kind = "no_support_fallback"
            anchors.append(
                self._anchor(
                    rng,
                    crop_index,
                    kind,
                    class_id,
                    row,
                    column,
                    height,
                    width,
                    valid=valid,
                    semantic=semantic,
                    record_has_support=bool(present),
                )
            )

        free = numpy.flatnonzero(~valid)
        for offset in range(background_count):
            crop_index = support_count + offset
            rng = self._rng(sample_id, record_index, crop_index, "anchor")
            if free.size:
                chosen = int(free[int(rng.integers(free.size))])
                row, column = divmod(chosen, width)
            else:
                row = int(rng.integers(height))
                column = int(rng.integers(width))
            anchors.append(
                self._anchor(
                    rng,
                    crop_index,
                    "background",
                    None,
                    row,
                    column,
                    height,
                    width,
                    valid=valid,
                    semantic=semantic,
                    record_has_support=bool(present),
                )
            )
        return anchors

    def _anchor(
        self,
        rng: Any,
        crop_index: int,
        kind: str,
        class_id: Optional[int],
        row: int,
        column: int,
        height: int,
        width: int,
        *,
        valid: Optional[Any] = None,
        semantic: Optional[Any] = None,
        record_has_support: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self.jitter > 0:
            jitter_y = int(rng.integers(-self.jitter, self.jitter + 1))
            jitter_x = int(rng.integers(-self.jitter, self.jitter + 1))
        else:
            jitter_y = 0
            jitter_x = 0
        primary = (
            min(max(int(row) + jitter_y, 0), height - 1),
            min(max(int(column) + jitter_x, 0), width - 1),
        )
        candidates = [primary]
        unjittered = (int(row), int(column))
        if unjittered != primary:
            candidates.append(unjittered)
        center = (height // 2, width // 2)
        if center not in candidates:
            candidates.append(center)

        corrected = False
        estimated: Optional[int] = None
        probes = 0
        if (
            kind == "support"
            and self.min_valid > 0
            and valid is not None
            and semantic is not None
        ):
            candidates, primary, estimated, probes = self._support_candidates(
                valid,
                semantic,
                class_id,
                candidates,
                rng,
            )
            corrected = True
        elif valid is not None:
            estimated = self._window_count(valid, primary)
        satisfied = bool(self.min_valid <= 0 or (estimated or 0) >= self.min_valid)
        return {
            "crop_index": int(crop_index),
            "kind": kind,
            "class_id": class_id,
            "center": primary,
            "candidates": candidates,
            "record_has_support": (
                bool(record_has_support) if record_has_support is not None else None
            ),
            "estimated_valid_pixels": estimated,
            "min_valid_satisfied": satisfied,
            "candidate_probes": probes,
            "corrected": corrected,
        }

    def _window_count(self, valid: Any, center: Sequence[int]) -> int:
        """Valid-pixel count of the exact window ``_extract`` would cut."""

        height, width = int(valid.shape[0]), int(valid.shape[1])
        half = int(self.crop_size) // 2
        top = max(0, min(height, int(center[0]) - half))
        left = max(0, min(width, int(center[1]) - half))
        bottom = max(top, min(height, int(center[0]) - half + int(self.crop_size)))
        right = max(left, min(width, int(center[1]) - half + int(self.crop_size)))
        return int(valid[top:bottom, left:right].sum())

    def _support_candidates(
        self,
        valid: Any,
        semantic: Any,
        class_id: Optional[int],
        base: Sequence[Sequence[int]],
        rng: Any,
    ) -> tuple:
        """Bounded deterministic search for a window meeting ``min_valid``.

        The chosen center is tried first; the rest of the list stays as a
        bounded fallback so ``crop`` can still keep the best window.  Only
        support pixels of the crop's own class, ordered by a seeded permutation,
        are probed: never a label from another crop's class and never a pixel
        without semantic support.
        """

        numpy = _require_numpy()
        width = int(semantic.shape[1])
        if class_id is not None:
            pool = numpy.flatnonzero(valid & (semantic == class_id))
        else:
            pool = numpy.flatnonzero(valid)
        probes: List[tuple] = [tuple(int(value) for value in base[0])]
        if pool.size:
            take = min(int(pool.size), CROP_SUPPORT_PROBE_LIMIT)
            order = rng.permutation(int(pool.size))[:take].tolist()
            for position in order:
                pixel = int(pool[position])
                candidate = (pixel // width, pixel % width)
                if candidate not in probes:
                    probes.append(candidate)
                if len(probes) >= CROP_MAX_CANDIDATES:
                    break

        best = probes[0]
        best_count = -1
        evaluated = 0
        for candidate in probes:
            evaluated += 1
            count = self._window_count(valid, candidate)
            if count > best_count:
                best, best_count = candidate, count
            if count >= self.min_valid:
                best, best_count = candidate, count
                break
        self._stats["anchor_probes"] += evaluated

        ordered: List[Sequence[int]] = [best]
        for candidate in list(base) + probes:
            candidate = tuple(int(value) for value in candidate)
            if candidate not in ordered:
                ordered.append(candidate)
        return [tuple(center) for center in ordered[:CROP_MAX_CANDIDATES]], best, best_count, evaluated

    # -- extraction -------------------------------------------------------

    def crop(
        self, entry: Mapping[str, Any], anchor: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Crop one fixed-size window; masks and image share the window."""

        best: Optional[Dict[str, Any]] = None
        best_valid = -1
        best_padded = False
        attempts = 0
        for center in anchor["candidates"]:
            attempts += 1
            cropped, padded, valid_pixels = self._extract(entry, center)
            if valid_pixels > best_valid:
                best = cropped
                best_valid = valid_pixels
                best_padded = padded
            if valid_pixels >= self.min_valid:
                break
        assert best is not None  # candidates is never empty

        stats = self._stats
        stats["crops_total"] += 1
        stats["acceptance_attempts"] += attempts
        kind = str(anchor["kind"])
        if kind == "support":
            stats["support_crops"] += 1
            if best_valid >= self.min_valid:
                stats["min_valid_satisfied_crops"] += 1
            else:
                stats["support_guard_failed_crops"] += 1
        elif kind == "background":
            stats["background_crops"] += 1
        else:
            stats["no_support_fallback_crops"] += 1
        class_id = anchor.get("class_id")
        if class_id is not None:
            counts = stats["class_counts"]
            counts[str(int(class_id))] = int(counts.get(str(int(class_id)), 0)) + 1
        if best_padded:
            stats["padded_crops"] += 1
        if best_valid < self.min_valid:
            stats["below_min_valid_crops"] += 1
        semantic_valid_pixels = int(best["semantic_valid"].sum())
        instance_valid_pixels = int(best["instance_valid"].sum())
        zero_supervision = semantic_valid_pixels == 0 and instance_valid_pixels == 0
        if zero_supervision:
            stats["zero_supervision_crops"] += 1
        stats["unknown_pixels"] += int((~best["semantic_valid"]).sum())
        stats["valid_semantic_pixels"] += semantic_valid_pixels
        stats["valid_instance_pixels"] += instance_valid_pixels
        best["crop_index"] = int(anchor.get("crop_index", 0))
        best["crop_kind"] = kind
        best["skip_training"] = bool(zero_supervision)
        best["zero_supervision"] = bool(zero_supervision)
        best["valid_semantic_pixels"] = semantic_valid_pixels
        best["valid_instance_pixels"] = instance_valid_pixels
        best["min_valid_satisfied"] = bool(
            self.min_valid <= 0 or best_valid >= self.min_valid
        )
        return best

    def _extract(
        self, entry: Mapping[str, Any], center: Sequence[int]
    ) -> tuple:
        numpy = _require_numpy()
        image = numpy.asarray(entry["image"], dtype=numpy.float32)
        semantic = numpy.asarray(entry["semantic"], dtype=numpy.int16)
        semantic_valid = numpy.asarray(entry["semantic_valid"], dtype=bool)
        instance = numpy.asarray(entry["instance"], dtype=numpy.int16)
        instance_valid = numpy.asarray(entry["instance_valid"], dtype=bool)
        height, width = int(semantic.shape[0]), int(semantic.shape[1])

        size = int(self.crop_size)
        half = size // 2
        top = int(center[0]) - half
        left = int(center[1]) - half
        bottom = top + size
        right = left + size
        src_top = max(0, top)
        src_left = max(0, left)
        src_bottom = min(height, bottom)
        src_right = min(width, right)
        pad_top = src_top - top
        pad_left = src_left - left

        out_image = numpy.full(
            (int(image.shape[0]), size, size), CROP_IMAGE_PAD_VALUE, dtype=numpy.float32
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
            source = (
                slice(src_top, src_bottom),
                slice(src_left, src_right),
            )
            out_image[:, target[0], target[1]] = image[:, source[0], source[1]]
            out_semantic[target] = semantic[source]
            out_semantic_valid[target] = semantic_valid[source]
            out_instance[target] = instance[source]
            out_instance_valid[target] = instance_valid[source]

        padded = (src_top, src_left, src_bottom, src_right) != (
            top,
            left,
            bottom,
            right,
        )
        cropped = {
            "image": out_image,
            "semantic": out_semantic,
            "semantic_valid": out_semantic_valid,
            "instance": out_instance,
            "instance_valid": out_instance_valid,
            "sample_id": str(entry.get("sample_id") or ""),
            "source_width": int(entry.get("source_width") or width),
            "source_height": int(entry.get("source_height") or height),
        }
        return cropped, bool(padded), int(out_semantic_valid.sum())

    # -- reporting --------------------------------------------------------

    def policy(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "role": ROLE_FIT,
            "crop_size": self.crop_size,
            "crops_per_record": self.crops_per_record,
            "background_fraction": self.background_fraction,
            "jitter_pixels": self.jitter,
            "min_valid_semantic_pixels": self.min_valid,
            "anchor_source": "fit semantic/semantic_valid targets only",
            "anchor_strategy": (
                "deterministic class-stratified round-robin over classes present "
                "in each fit record; seeded by (seed, sample_id, record, crop)"
            ),
            "background_policy": (
                "background crops anchor on pixels without semantic support; "
                "their labels stay unknown (-1/False) and are never negatives"
            ),
            "sampling_correction": {
                "applies_to": ROLE_FIT,
                "zero_supervision_policy": (
                    "a crop with no valid semantic pixel and no valid instance "
                    "pixel is marked skip_training and dropped before "
                    "forward/backward; labels stay unknown, no synthetic "
                    "negative and no replacement crop are created"
                ),
                "support_guard": (
                    "for records with semantic support, up to "
                    f"{CROP_MAX_CANDIDATES} deterministic candidate windows "
                    f"(anchor, jittered/un-jittered, up to "
                    f"{CROP_SUPPORT_PROBE_LIMIT} class support probes) are tried "
                    "until >= min_valid semantic pixels are inside"
                ),
                "bounded_candidates": CROP_MAX_CANDIDATES,
                "support_probe_limit": CROP_SUPPORT_PROBE_LIMIT,
                "impossible_guard_policy": (
                    "keep the maximum-valid window, keep true labels, and report "
                    "support_guard_failed_crops; never fabricate labels"
                ),
                "exposure_schedule": (
                    "crop count and index per record remain fixed; a skipped "
                    "crop is not replaced, reordered or oversampled"
                ),
                "optimizer_steps": (
                    "reported per epoch; skipped crops consume no optimizer step"
                ),
            },
            "window_policy": (
                "image, semantic, semantic_valid, instance and instance_valid "
                "share one identical window"
            ),
            "padding": {
                "mode": "constant_unknown",
                "image": CROP_IMAGE_PAD_VALUE,
                "semantic": -1,
                "instance": -1,
                "semantic_valid": False,
                "instance_valid": False,
                "reason": "source smaller than the crop or window crossing a border",
            },
            "resize": None,
            "full_source_resolution": True,
            "unknown_as_negative": False,
            "synthetic_negatives": False,
            "model_input": (
                "image tensor only; anchor coordinates, class ids and all other "
                "label features never enter the model input"
            ),
            "tune_eval": "full frames unchanged",
        }

    def stats(self) -> Dict[str, Any]:
        result = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in self._stats.items()
        }
        result["class_counts"] = {
            str(key): int(value)
            for key, value in sorted(
                result["class_counts"].items(), key=lambda item: int(item[0])
            )
        }
        return result


# ---------------------------------------------------------------------------
# JSON helpers


def _write_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# model / tensor adapters


def seed_everything(seed: int) -> Dict[str, Any]:
    """Deterministic seeding across python, numpy and torch."""

    random.seed(int(seed))
    numpy = _require_numpy()
    numpy.random.seed(int(seed) % (2**32))
    torch = _require_torch()
    torch.manual_seed(int(seed))
    applied: Dict[str, Any] = {"seed": int(seed), "torch_deterministic": False}
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        applied["torch_deterministic"] = True
    except Exception:  # pragma: no cover - older torch
        applied["torch_deterministic"] = False
    return applied


def resolve_device(device: str) -> str:
    torch = _require_torch()
    if device not in ("cpu", "cuda", "auto"):
        raise RasterTrainError(f"--device must be cpu, cuda or auto; got {device!r}")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RasterTrainError("cuda_unavailable")
    return device


def _model_config(**overrides: Any) -> Dict[str, Any]:
    allowed = set(raster_model.RasterModelConfig().to_dict())
    unknown = set(overrides) - allowed
    if unknown:
        raise RasterTrainError(f"unknown model config keys: {sorted(unknown)}")
    config = raster_model.RasterModelConfig(**overrides)
    return config.to_dict()


def build_model(**overrides: Any) -> Any:
    """Thin adapter over :func:`recognition_raster_model.build_raster_model`."""

    return raster_model.build_raster_model(_model_config(**overrides))


def _batch_tensors(entries: Sequence[Mapping[str, Any]], device: str) -> Dict[str, Any]:
    numpy = _require_numpy()
    torch = _require_torch()
    shapes = {(int(entry["image"].shape[1]), int(entry["image"].shape[2])) for entry in entries}
    if len(shapes) != 1:
        raise RasterTrainError(
            f"cannot batch samples with different source sizes: {sorted(shapes)}"
        )
    images = numpy.stack([entry["image"] for entry in entries])
    semantic = numpy.stack([entry["semantic"] for entry in entries]).astype(numpy.int64)
    semantic_valid = numpy.stack([entry["semantic_valid"] for entry in entries])
    instance = numpy.stack([entry["instance"] for entry in entries]).astype(numpy.int64)
    instance_valid = numpy.stack([entry["instance_valid"] for entry in entries])
    return {
        "images": torch.from_numpy(images).to(device=device, dtype=torch.float32),
        "semantic": torch.from_numpy(semantic).to(device),
        "semantic_valid": torch.from_numpy(semantic_valid).to(device),
        "instance": torch.from_numpy(instance).to(device),
        "instance_valid": torch.from_numpy(instance_valid).to(device),
    }


def _autocast_context(torch: Any, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    except AttributeError:  # pragma: no cover - older torch
        return torch.cuda.amp.autocast()


def _make_scaler(torch: Any, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=True)


def _float_outputs(outputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Losses run in fp32 even when the forward pass used fp16 autocast."""

    return {
        "semantic_logits": outputs["semantic_logits"].float(),
        "embeddings": outputs["embeddings"].float(),
        "center_logits": outputs.get("center_logits"),
        "output_size": outputs.get("output_size"),
    }


# ---------------------------------------------------------------------------
# one epoch of fit (fit role only)


def fit_semantic_class_weighting(source: SampleSource) -> Dict[str, Any]:
    """Count selected fit pixels once; never consult tune or unknown labels."""

    numpy = _require_numpy()
    counts = numpy.zeros(int(raster_model.NUM_CLASSES), dtype=numpy.int64)
    for index in range(len(source)):
        entry = source.get(index)
        targets = numpy.asarray(entry["semantic"])
        keep = numpy.asarray(entry["semantic_valid"], dtype=bool) & (targets >= 0) & (targets < len(counts))
        counts += numpy.bincount(targets[keep].astype(numpy.int64), minlength=len(counts))
    present = counts > 0
    total = int(counts.sum())
    classes = int(present.sum())
    if total == 0:
        raise RasterTrainError("semantic_class_weighting_empty_known_fit_support")
    weights = numpy.ones(len(counts), dtype=numpy.float64)
    weights[present] = numpy.sqrt(total / (classes * counts[present]))
    weights[present] /= weights[present].mean()
    return {
        "method": "sqrt-inverse-frequency",
        "source_role": ROLE_FIT,
        "counts": counts.tolist(),
        "valid_pixels": total,
        "present_classes": classes,
        "formula": "sqrt(N/(K*n[c])); present weights normalized to arithmetic mean 1; absent weights 1",
        "weights": weights.tolist(),
        "policy": "selected fit records only; semantic_valid and targets 0..34; unknown excluded; computed once",
    }


def train_epoch(
    model: Any,
    source: SampleSource,
    optimizer: Any,
    *,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    instance_weight: float = DEFAULT_INSTANCE_WEIGHT,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    amp: bool = False,
    semantic_class_weight: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one pass over fit samples.  No tune/eval sample is ever touched."""

    torch = _require_torch()
    if batch_size < 1:
        raise RasterTrainError("batch_size must be >= 1")
    scaler = _make_scaler(torch, amp)
    model.train()
    totals = {
        "entries_seen": 0,
        "sample_count": 0,
        "valid_semantic_pixels": 0,
        "semantic_loss_sum": 0.0,
        "instance_loss_sum": 0.0,
        "loss_sum": 0.0,
        "batch_count": 0,
        "skipped_crops": 0,
        "skipped_batches": 0,
        "zero_supervision_crops": 0,
    }
    started = time.monotonic()
    indices = list(range(len(source)))
    scheduled_batches = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        scheduled_batches += 1
        entries = [source.get(index) for index in chunk]
        totals["entries_seen"] += len(entries)
        totals["zero_supervision_crops"] += sum(
            1 for entry in entries if entry.get("zero_supervision")
        )
        # Zero-supervision crops carry no label signal at all: they are dropped
        # before the forward pass so they consume no optimizer step and do not
        # dilute the masked loss averages.  Nothing is replaced or oversampled.
        trainable = [
            entry for entry in entries if not bool(entry.get("skip_training", False))
        ]
        totals["skipped_crops"] += len(entries) - len(trainable)
        if not trainable:
            totals["skipped_batches"] += 1
            continue
        entries = trainable
        tensors = _batch_tensors(entries, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(torch, amp):
            raw_outputs = model(tensors["images"])
        parts = raster_model.combined_raster_loss(
            _float_outputs(raw_outputs),
            semantic_targets=tensors["semantic"],
            semantic_valid=tensors["semantic_valid"],
            instance_targets=tensors["instance"],
            instance_valid=tensors["instance_valid"],
            semantic_weight=float(semantic_weight),
            instance_weight=float(instance_weight),
            semantic_class_weight=semantic_class_weight,
        )
        loss = parts["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        valid_pixels = int(tensors["semantic_valid"].sum().item())
        totals["sample_count"] += len(entries)
        totals["valid_semantic_pixels"] += valid_pixels
        totals["semantic_loss_sum"] += float(parts["semantic"].detach()) * max(valid_pixels, 1)
        totals["instance_loss_sum"] += float(parts["instance"].detach())
        totals["loss_sum"] += float(loss.detach()) * max(valid_pixels, 1)
        totals["batch_count"] += 1

    weight = max(totals["valid_semantic_pixels"], 1)
    result = {
        "entries": totals["entries_seen"],
        "samples": totals["sample_count"],
        "batches": totals["batch_count"],
        "optimizer_steps": totals["batch_count"],
        "scheduled_batches": scheduled_batches,
        "skipped_crops": totals["skipped_crops"],
        "skipped_batches": totals["skipped_batches"],
        "zero_supervision_crops": totals["zero_supervision_crops"],
        "skipped_fraction": round(
            totals["skipped_crops"] / max(totals["entries_seen"], 1), 6
        ),
        "valid_semantic_pixels": totals["valid_semantic_pixels"],
        "loss": round(totals["loss_sum"] / weight, 8),
        "semantic": round(totals["semantic_loss_sum"] / weight, 8),
        "instance": round(
            totals["instance_loss_sum"] / max(totals["batch_count"], 1), 8
        ),
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    return result


# ---------------------------------------------------------------------------
# checkpoint IO


def save_checkpoint(
    path: str,
    model: Any,
    *,
    model_config: Mapping[str, Any],
    epoch: int,
    epochs: int,
    train_scale: int,
    seed: int,
    train_metrics: Mapping[str, Any],
    device: str,
) -> str:
    torch = _require_torch()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "model_config": dict(model_config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": int(epoch),
        "epochs": int(epochs),
        "role": ROLE_FIT,
        "stage": ROLE_FIT,
        "train_scale": int(train_scale),
        "seed": int(seed),
        "device": device,
        "semantic_weight": float(train_metrics.get("semantic_weight", DEFAULT_SEMANTIC_WEIGHT)),
        "instance_weight": float(train_metrics.get("instance_weight", DEFAULT_INSTANCE_WEIGHT)),
        "metrics": dict(train_metrics),
        "label_space": {
            "num_classes": int(raster_model.NUM_CLASSES),
            "thing_classes": [0, int(raster_model.NUM_THING_CLASSES) - 1],
            "unknown": raster_model.NO_INSTANCE,
            "ignore_index": int(raster_model.IGNORE_INDEX),
        },
        "created_unix": round(time.time(), 3),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`.

    Only ``schema``, ``model_config`` and ``state_dict`` are trusted; the rest is
    returned as provenance.  The payload holds no pickled custom classes.
    """

    torch = _require_torch()
    if not os.path.isfile(path):
        raise RasterTrainError(f"checkpoint_not_found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6 signature
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RasterTrainError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("role") not in (ROLE_FIT, None):
        raise RoleAccessError(
            f"checkpoint role must be {ROLE_FIT!r}; got {payload.get('role')!r}"
        )
    model = build_model(**dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return {"model": model, "payload": payload, "path": path}


# ---------------------------------------------------------------------------
# masked semantic evaluation (tune only)


def evaluate_samples(
    model: Any,
    source: SampleSource,
    *,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Masked semantic loss/accuracy over tune samples; no tuning decisions."""

    numpy = _require_numpy()
    torch = _require_torch()
    model.eval()
    num_classes = int(raster_model.NUM_CLASSES)
    confusion = numpy.zeros((num_classes, num_classes), dtype=numpy.int64)
    loss_sum = 0.0
    valid_pixels = 0
    sample_count = 0
    unknown_pixels = 0
    instance_pixels = 0

    indices = list(range(len(source)))
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            entries = [source.get(index) for index in chunk]
            tensors = _batch_tensors(entries, device)
            raw_outputs = model(tensors["images"])
            outputs = _float_outputs(raw_outputs)
            loss = raster_model.masked_semantic_loss(
                outputs["semantic_logits"], tensors["semantic"], tensors["semantic_valid"]
            )
            predictions = outputs["semantic_logits"].argmax(dim=1)
            keep = tensors["semantic_valid"] & (tensors["semantic"] >= 0)
            if bool(keep.any()):
                true_ids = tensors["semantic"][keep].to(dtype=torch.long).cpu().numpy()
                pred_ids = predictions[keep].to(dtype=torch.long).cpu().numpy()
                flat = true_ids * num_classes + pred_ids
                counts = numpy.bincount(flat, minlength=num_classes * num_classes)
                confusion += counts.reshape(num_classes, num_classes)
                valid_pixels += int(true_ids.size)
                loss_sum += float(loss.detach()) * int(true_ids.size)
            unknown_pixels += int((~tensors["semantic_valid"]).sum().item())
            instance_pixels += int(tensors["instance_valid"].sum().item())
            sample_count += len(entries)

    diagonal = int(numpy.trace(confusion))
    accuracy = (diagonal / valid_pixels) if valid_pixels else None
    per_class: Dict[str, Dict[str, Any]] = {}
    for class_index in range(num_classes):
        true_total = int(confusion[class_index].sum())
        predicted_total = int(confusion[:, class_index].sum())
        if true_total == 0 and predicted_total == 0:
            continue
        correct = int(confusion[class_index, class_index])
        per_class[str(class_index)] = {
            "raw_semantic": class_index + 1,
            "true_pixels": true_total,
            "predicted_pixels": predicted_total,
            "correct_pixels": correct,
            "accuracy": round(correct / true_total, 8) if true_total else None,
        }
    return {
        "samples": sample_count,
        "valid_pixels": valid_pixels,
        "unknown_pixels": unknown_pixels,
        "instance_valid_pixels": instance_pixels,
        "loss": round(loss_sum / valid_pixels, 8) if valid_pixels else None,
        "accuracy": round(accuracy, 8) if accuracy is not None else None,
        "correct_pixels": diagonal,
        "per_class": per_class,
    }


def pq_status(alignment_report: Optional[str] = None) -> Dict[str, Any]:
    """PQ gate: unavailable until coordinate alignment *and* a vectorizer exist.

    There is no prediction vectorizer in this repository yet, so even a
    verified alignment report cannot produce PQ documents.  The result says
    exactly which condition is missing instead of emitting a guessed score.
    """

    base: Dict[str, Any] = {
        "available": False,
        "metric": "PQ",
        "official_compatibility": "unconfirmed",
        "scorer_policy": dict(raster_data.SCORER_POLICY),
    }
    if not alignment_report:
        return {
            **base,
            "reason": PQ_UNAVAILABLE_REASON,
            "alignment": None,
        }
    try:
        report = _read_json(alignment_report)
    except (OSError, ValueError) as exc:
        return {
            **base,
            "reason": PQ_UNAVAILABLE_REASON,
            "alignment": {"status": "report_unreadable", "detail": str(exc)},
        }
    verified = (
        isinstance(report, Mapping)
        and report.get("schema") == ALIGNMENT_SCHEMA
        and report.get("status") == "verified"
    )
    if not verified:
        return {
            **base,
            "reason": PQ_UNAVAILABLE_REASON,
            "alignment": {
                "status": (report.get("status") if isinstance(report, Mapping) else None),
                "report": os.path.abspath(alignment_report),
            },
        }
    return {
        **base,
        "reason": PQ_NO_VECTORIZER_REASON,
        "alignment": {
            "status": "verified",
            "report": os.path.abspath(alignment_report),
        },
    }


# ---------------------------------------------------------------------------
# fit orchestration


def _limit(records: Sequence[Mapping[str, Any]], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    items = [dict(record) for record in records]
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise RasterTrainError("max_samples must be an int")
        if max_samples < 1:
            raise RasterTrainError("max_samples must be >= 1")
        items = items[:max_samples]
    return items


def _blocked(reason: str, **extra: Any) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "blocked",
        "reason": reason,
        "trained": False,
        "no_eval_access": True,
        **extra,
    }


def run_fit(
    manifest_path: str,
    *,
    role: str,
    epochs: int,
    out: str,
    train_scale: int = DEFAULT_TRAIN_SCALE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    instance_weight: float = DEFAULT_INSTANCE_WEIGHT,
    device: str = "cpu",
    seed: int = SEED,
    base_channels: int = 24,
    depth: int = 3,
    embedding_dim: int = raster_model.EMBED_DIM,
    amp: str = "auto",
    max_samples: Optional[int] = None,
    cache: bool = True,
    tune_once: bool = True,
    pq_alignment_report: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    semantic_class_weighting: str = "none",
    crop_size: int = DEFAULT_CROP_SIZE,
    crops_per_record: int = DEFAULT_CROPS_PER_RECORD,
    crop_background_fraction: float = DEFAULT_CROP_BACKGROUND_FRACTION,
    crop_jitter: int = DEFAULT_CROP_JITTER,
    crop_min_valid: int = DEFAULT_CROP_MIN_VALID,
) -> Dict[str, Any]:
    """Train on the fit role, then one tune evaluation of the final checkpoint."""

    require_command_role("fit", role)
    crop_config = normalize_crop_config(
        crop_size=crop_size,
        crops_per_record=crops_per_record,
        background_fraction=crop_background_fraction,
        jitter=crop_jitter,
        min_valid=crop_min_valid,
    )
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise RasterTrainError("--epochs must be an int >= 1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise RasterTrainError("--batch-size must be an int >= 1")
    if amp not in ("auto", "off"):
        raise RasterTrainError("--amp must be auto or off")
    if semantic_class_weighting not in ("none", "sqrt-inverse-frequency"):
        raise RasterTrainError("--semantic-class-weighting must be none or sqrt-inverse-frequency")

    status = dependency_status()
    if not status["ready"]:
        missing = [name for name in ("torch", "numpy", "pillow") if not status.get(name)]
        return _blocked(
            "missing_dependency",
            missing=missing,
            dependency_status=status,
            out=out,
            message="install PyTorch/numpy/Pillow locally; this runner never downloads",
        )

    model_config = _model_config(
        base_channels=int(base_channels),
        depth=int(depth),
        embedding_dim=int(embedding_dim),
    )
    init_payload = None
    if init_checkpoint is not None:
        torch = _require_torch()
        if not os.path.isfile(init_checkpoint):
            raise RasterTrainError(f"checkpoint_not_found: {init_checkpoint}")
        try:
            init_payload = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - torch < 2.6 signature
            init_payload = torch.load(init_checkpoint, map_location="cpu")
        if not isinstance(init_payload, Mapping) or init_payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RasterTrainError(f"checkpoint_schema_mismatch: {init_checkpoint}")
        if init_payload.get("role") not in (ROLE_FIT, None):
            raise RoleAccessError(f"checkpoint role must be {ROLE_FIT!r}; got {init_payload.get('role')!r}")
        if init_payload.get("model_config") != model_config:
            raise RasterTrainError("checkpoint_model_config_mismatch")

    records = _limit(
        role_sample_records(manifest_path, ROLE_FIT, train_scale=train_scale), max_samples
    )
    if not records:
        return _blocked("no_fit_samples", out=out, role=ROLE_FIT, train_scale=train_scale)

    torch = _require_torch()
    resolved_device = resolve_device(device)
    seed_info = seed_everything(seed)
    use_amp = bool(amp == "auto" and resolved_device.startswith("cuda"))
    amp_note = (
        "enabled_fp16_cuda"
        if use_amp
        else ("disabled_requested" if amp == "off" else "disabled_non_cuda_device")
    )

    os.makedirs(out, exist_ok=True)
    cache_dir = os.path.join(out, "cache") if cache else None
    crop_sampler = (
        SupportCropSampler(
            crop_size=int(crop_config["crop_size"]),
            crops_per_record=int(crop_config["crops_per_record"]),
            background_fraction=float(crop_config["background_fraction"]),
            jitter=int(crop_config["jitter"]),
            min_valid=int(crop_config["min_valid"]),
            seed=int(seed),
        )
        if crop_config["enabled"]
        else None
    )
    source = SampleSource(records, cache_dir=cache_dir, crop_sampler=crop_sampler)
    full_source = SampleSource(records, cache_dir=cache_dir) if crop_sampler else source
    weighting = (
        fit_semantic_class_weighting(full_source)
        if semantic_class_weighting == "sqrt-inverse-frequency"
        else None
    )
    semantic_weight_tensor = (
        torch.tensor(weighting["weights"], dtype=torch.float32, device=resolved_device)
        if weighting is not None else None
    )

    model = raster_model.build_raster_model(model_config).to(resolved_device)
    if init_payload is not None:
        model.load_state_dict(init_payload["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    metrics_path = os.path.join(out, "metrics.json")
    history: List[Dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_metrics = train_epoch(
            model,
            source,
            optimizer,
            device=resolved_device,
            batch_size=int(batch_size),
            semantic_weight=float(semantic_weight),
            instance_weight=float(instance_weight),
            grad_clip=float(grad_clip),
            amp=use_amp,
            semantic_class_weight=semantic_weight_tensor,
        )
        checkpoint_path = os.path.join(out, f"checkpoint_epoch_{epoch:04d}.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            model_config=model_config,
            epoch=epoch,
            epochs=epochs,
            train_scale=int(train_scale),
            seed=int(seed),
            train_metrics={
                **epoch_metrics,
                "semantic_weight": float(semantic_weight),
                "instance_weight": float(instance_weight),
            },
            device=resolved_device,
        )
        entry = {
            "epoch": epoch,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            **epoch_metrics,
        }
        history.append(entry)
        _write_json(
            metrics_path,
            {
                "schema": METRICS_SCHEMA,
                "version": 1,
                "role": ROLE_FIT,
                "train_scale": int(train_scale),
                "epochs_requested": int(epochs),
                "epochs_completed": epoch,
                "init_checkpoint": os.path.abspath(init_checkpoint) if init_checkpoint else None,
                "init_checkpoint_epoch": init_payload.get("epoch") if init_payload is not None else None,
                "semantic_class_weighting": weighting,
                "history": history,
            },
        )

    final_path = history[-1]["checkpoint"]
    selection: Dict[str, Any] = {
        "rule": "single_tune_eval_on_final_epoch" if tune_once else "fit_only_final_epoch",
        "selected_checkpoint": final_path,
        "selected_epoch": history[-1]["epoch"],
        "tune_eval": None,
    }
    tune_eval: Dict[str, Any] = {"status": "skipped", "reason": "disabled_by_flag"}
    if tune_once:
        try:
            tune_records = _limit(
                role_sample_records(manifest_path, ROLE_TUNE, train_scale=train_scale),
                max_samples,
            )
            if not tune_records:
                tune_eval = {"status": "blocked", "reason": "no_tune_samples"}
            else:
                tune_eval = run_tune_eval(
                    tune_records,
                    final_path,
                    device=resolved_device,
                    train_scale=int(train_scale),
                    pq_alignment_report=pq_alignment_report,
                )
        except RecognitionError as exc:
            tune_eval = {
                "status": "blocked",
                "reason": getattr(exc, "code", type(exc).__name__),
                "detail": str(exc),
            }
    selection["tune_eval"] = {
        "status": tune_eval.get("status"),
        "semantic_accuracy": (tune_eval.get("semantic") or {}).get("accuracy"),
        "semantic_loss": (tune_eval.get("semantic") or {}).get("loss"),
    }

    selected_copy = os.path.join(out, "checkpoint_selected.pt")
    if history:
        shutil.copyfile(final_path, selected_copy)
        selection["selected_checkpoint_copy"] = selected_copy

    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "status": "complete",
        "role": ROLE_FIT,
        "train_scale": int(train_scale),
        "seed": int(seed),
        "seed_info": seed_info,
        "device": resolved_device,
        "amp": {"requested": amp, "applied": use_amp, "note": amp_note},
        "optimizer": {
            "name": "AdamW",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
        },
        "loss": {
            "semantic_weight": float(semantic_weight),
            "instance_weight": float(instance_weight),
            "mask_policy": TRAIN_POLICY["loss_masks"],
            "unknown_as_negative": False,
        },
        "model_config": model_config,
        "batch_size": int(batch_size),
        "fit_samples": len(records),
        "fit_entries": len(source),
        "crop_sampling": (
            crop_sampler.policy() if crop_sampler is not None else _disabled_crop_policy()
        ),
        "crop_counts": (
            crop_sampler.stats() if crop_sampler is not None else None
        ),
        "optimizer_schedule": {
            "batch_size": int(batch_size),
            "fit_entries": len(source),
            "scheduled_batches_per_epoch": (
                len(source) + int(batch_size) - 1
            ) // int(batch_size),
            "optimizer_steps_per_epoch": [
                int(item["optimizer_steps"]) for item in history
            ],
            "skipped_crops_per_epoch": [
                int(item["skipped_crops"]) for item in history
            ],
            "skipped_crops_total": sum(
                int(item["skipped_crops"]) for item in history
            ),
            "expected_optimizer_steps_note": (
                "scheduled_batches_per_epoch minus batches whose crops are all "
                "zero-supervision; measured in optimizer_steps_per_epoch"
            ),
            "exposure_schedule": (
                "one fixed entry per record and crop index; a zero-supervision "
                "crop is skipped, never replaced, reordered or oversampled"
            ),
        },
        "label_based_sampling": {
            "applies_to": ROLE_FIT,
            "source": "fit semantic/semantic_valid targets only",
            "tune_eval": "full frames unchanged",
            "model_input": "image tensor only; no label coordinates or features",
            "zero_supervision_skipped": True,
        },
        "fit_groups": sorted({str(record.get("group_id")) for record in records}),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "init_checkpoint": os.path.abspath(init_checkpoint) if init_checkpoint else None,
        "init_checkpoint_epoch": init_payload.get("epoch") if init_payload is not None else None,
        "semantic_class_weighting": weighting,
        "history": history,
        "checkpoint_selection": selection,
        "policy": TRAIN_POLICY,
        "cache": {"enabled": bool(cache), "dir": cache_dir},
        "elapsed_s": round(time.monotonic() - started, 3),
        "no_eval_access": True,
    }
    _write_json(os.path.join(out, "train_manifest.json"), manifest)
    _write_json(os.path.join(out, "tune_eval.json"), tune_eval)

    peak_gpu_mb = None
    if resolved_device.startswith("cuda"):
        peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 3)
    return {
        **manifest,
        "metrics_path": metrics_path,
        "tune_eval_path": os.path.join(out, "tune_eval.json"),
        "resource_peak": {"peak_gpu_memory_mb": peak_gpu_mb},
        "trained": True,
    }


# ---------------------------------------------------------------------------
# tune evaluation (no tuning decisions, no fit access)


def run_tune_eval(
    records: Sequence[Mapping[str, Any]],
    checkpoint: str,
    *,
    device: str = "cpu",
    train_scale: int = DEFAULT_TRAIN_SCALE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: Optional[int] = None,
    pq_alignment_report: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one checkpoint on tune samples with masked semantic metrics."""

    limited = _limit(records, max_samples)
    if not limited:
        return {
            "schema": TUNE_EVAL_SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "no_tune_samples",
            "role": ROLE_TUNE,
        }
    if any(record.get("role") != ROLE_TUNE for record in limited):
        raise RoleAccessError("tune evaluation received a non-tune sample record")

    resolved_device = resolve_device(device)
    loaded = load_checkpoint(checkpoint, device=resolved_device)
    payload = loaded["payload"]
    source = SampleSource(limited)
    metrics = evaluate_samples(
        loaded["model"], source, device=resolved_device, batch_size=batch_size
    )
    return {
        "schema": TUNE_EVAL_SCHEMA,
        "version": 1,
        "status": "complete",
        "role": ROLE_TUNE,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_train_scale": payload.get("train_scale"),
        "train_scale": int(train_scale),
        "checkpoint_epoch": payload.get("epoch"),
        "device": resolved_device,
        "samples": metrics["samples"],
        "sample_ids": [record["sample_id"] for record in limited],
        "semantic": {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "valid_pixels": metrics["valid_pixels"],
            "correct_pixels": metrics["correct_pixels"],
            "unknown_pixels": metrics["unknown_pixels"],
            "per_class": metrics["per_class"],
            "scored_on": "semantic_valid_mask_only",
        },
        "instance": {
            "status": "not_scored_no_clustering",
            "reason": "instance ids must be produced by a label-free grouping step",
            "valid_pixels": metrics["instance_valid_pixels"],
        },
        "pq": pq_status(pq_alignment_report),
        "no_eval_access": True,
        "tuning": {"performed": False, "checkpoint_selection_use": "single_eval_only"},
    }


def run_evaluate(
    manifest_path: str,
    *,
    role: str,
    checkpoint: str,
    out: str,
    train_scale: int = DEFAULT_TRAIN_SCALE,
    device: str = "cpu",
    max_samples: Optional[int] = None,
    pq_alignment_report: Optional[str] = None,
) -> Dict[str, Any]:
    """CLI ``evaluate``: tune role only, one checkpoint, masked metrics."""

    require_command_role("evaluate", role)
    status = dependency_status()
    if not status["ready"]:
        missing = [name for name in ("torch", "numpy", "pillow") if not status.get(name)]
        return _blocked(
            "missing_dependency",
            missing=missing,
            dependency_status=status,
            out=out,
            message="install PyTorch/numpy/Pillow locally; this runner never downloads",
        )
    records = role_sample_records(manifest_path, ROLE_TUNE, train_scale=train_scale)
    result = run_tune_eval(
        records,
        checkpoint,
        device=device,
        train_scale=train_scale,
        max_samples=max_samples,
        pq_alignment_report=pq_alignment_report,
    )
    target = out
    if not str(out).lower().endswith(".json"):
        target = os.path.join(str(out), "tune_metrics.json")
    _write_json(target, result)
    return {**result, "out": target}


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_raster_train",
        description="Fit/tune-only raster U-Net runner (fit on fit, evaluate on tune).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="dependency smoke test; reads no data")

    fit = sub.add_parser("fit", help="train on fit role, then one tune evaluation")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--role", required=True, help="must be exactly 'fit'")
    fit.add_argument("--epochs", type=int, required=True)
    fit.add_argument("--out", required=True, help="run directory")
    fit.add_argument("--init-checkpoint", default=None, help="fit checkpoint weights for warm-start")
    fit.add_argument("--semantic-class-weighting", choices=("none", "sqrt-inverse-frequency"), default="none")
    fit.add_argument("--train-scale", type=int, default=DEFAULT_TRAIN_SCALE)
    fit.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    fit.add_argument("--lr", type=float, default=DEFAULT_LR)
    fit.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    fit.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    fit.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT)
    fit.add_argument("--instance-weight", type=float, default=DEFAULT_INSTANCE_WEIGHT)
    fit.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--base-channels", type=int, default=24)
    fit.add_argument("--depth", type=int, default=3)
    fit.add_argument("--embedding-dim", type=int, default=raster_model.EMBED_DIM)
    fit.add_argument("--amp", choices=("auto", "off"), default="auto")
    fit.add_argument("--max-samples", type=int, default=None)
    fit.add_argument("--no-cache", action="store_true", help="disable the label cache")
    fit.add_argument(
        "--skip-tune-eval",
        action="store_true",
        help="skip the single post-training tune evaluation",
    )
    fit.add_argument("--pq-alignment-report", default=None)
    fit.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help="fit-only square crop side in source pixels; 0 keeps full frames (default)",
    )
    fit.add_argument(
        "--crops-per-record",
        type=int,
        default=DEFAULT_CROPS_PER_RECORD,
        help="fit crops sampled per record when --crop-size > 0",
    )
    fit.add_argument(
        "--crop-background-fraction",
        type=float,
        default=DEFAULT_CROP_BACKGROUND_FRACTION,
        help="fraction of crops anchored away from semantic support (default 0)",
    )
    fit.add_argument(
        "--crop-jitter",
        type=int,
        default=DEFAULT_CROP_JITTER,
        help="maximum deterministic anchor jitter in pixels (default 0)",
    )
    fit.add_argument(
        "--crop-min-valid",
        type=int,
        default=DEFAULT_CROP_MIN_VALID,
        help="minimum valid semantic pixels per crop; 0 disables the check (default)",
    )

    evaluate = sub.add_parser("evaluate", help="masked semantic metrics on tune")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--role", required=True, help="must be exactly 'tune'")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True, help="JSON file or directory")
    evaluate.add_argument("--train-scale", type=int, default=DEFAULT_TRAIN_SCALE)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--pq-alignment-report", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = runner_smoke()
        elif args.command == "fit":
            result = run_fit(
                args.manifest,
                role=args.role,
                epochs=args.epochs,
                out=args.out,
                train_scale=args.train_scale,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                semantic_weight=args.semantic_weight,
                instance_weight=args.instance_weight,
                device=args.device,
                seed=args.seed,
                base_channels=args.base_channels,
                depth=args.depth,
                embedding_dim=args.embedding_dim,
                amp=args.amp,
                max_samples=args.max_samples,
                cache=not args.no_cache,
                tune_once=not args.skip_tune_eval,
                pq_alignment_report=args.pq_alignment_report,
                init_checkpoint=args.init_checkpoint,
                semantic_class_weighting=args.semantic_class_weighting,
                crop_size=args.crop_size,
                crops_per_record=args.crops_per_record,
                crop_background_fraction=args.crop_background_fraction,
                crop_jitter=args.crop_jitter,
                crop_min_valid=args.crop_min_valid,
            )
        else:
            result = run_evaluate(
                args.manifest,
                role=args.role,
                checkpoint=args.checkpoint,
                out=args.out,
                train_scale=args.train_scale,
                device=args.device,
                max_samples=args.max_samples,
                pq_alignment_report=args.pq_alignment_report,
            )
    except (RecognitionError, OSError, ValueError) as exc:
        print(f"error: {getattr(exc, 'code', type(exc).__name__)}: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("status") in ("ready", "complete", "partial", "ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUGMENTATION_POLICY",
    "CHECKPOINT_SCHEMA",
    "COMMAND_ROLES",
    "CROP_IMAGE_PAD_VALUE",
    "CROP_MAX_CANDIDATES",
    "CROP_SUPPORT_PROBE_LIMIT",
    "DEFAULT_CROP_BACKGROUND_FRACTION",
    "DEFAULT_CROP_JITTER",
    "DEFAULT_CROP_MIN_VALID",
    "DEFAULT_CROP_SIZE",
    "DEFAULT_CROPS_PER_RECORD",
    "PQ_UNAVAILABLE_REASON",
    "RasterTrainError",
    "SCHEMA",
    "SampleSource",
    "SupportCropSampler",
    "TRAIN_POLICY",
    "build_model",
    "build_parser",
    "build_role_sample",
    "dependency_status",
    "evaluate_samples",
    "load_checkpoint",
    "main",
    "normalize_crop_config",
    "pq_status",
    "require_command_role",
    "role_sample_records",
    "run_evaluate",
    "run_fit",
    "run_tune_eval",
    "runner_smoke",
    "save_checkpoint",
    "seed_everything",
    "train_epoch",
]
