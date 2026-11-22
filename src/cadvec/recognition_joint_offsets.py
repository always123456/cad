"""Joint semantic + instance-offset pilot (384 px, CPU-only readiness lane).

Why this lane exists
--------------------
The earlier instance-offset lane
(:mod:`cadvec.recognition_instance_offsets`) kept the semantic model frozen at
the pure-semantic FIT checkpoint and trained only a small offset head.  If that
semantic map is already wrong the offset head cannot repair it: every instance
claim is conditioned on a class the trunk chose without ever seeing the offset
objective.  This pilot tests the one materially different hypothesis available
without new data or tuning:

    fine-tune the semantic trunk and the offset head jointly, in one shared
    forward pass, using instance-context crops.

Nothing here is trained on real data by this module's test suite.  The live FIT
run is parent-scheduled; this module only prepares measured readiness, the
model/loss/decoder implementation, and recorded tests.

Hard contract
-------------
* crop and tile geometry are **384** px, tile stride **192**; offset targets are
  normalised by **256**, never by the crop size;
* the offset head is freshly initialised (Conv1x1 256->64->2, seed 20261108)
  *before* any other random construction and does not depend on the global RNG;
* the trunk and the classifier stay trainable, but every BatchNorm keeps its
  running statistics fixed (module in eval after ``model.train()``);
* one forward pass produces the semantic logits and the tapped
  ``net.classifier[3]`` features; the tap is neither detached nor recomputed and
  no ``torch.no_grad`` wraps training;
* the two losses are kept branch-local: semantic CE sum over known pixels and
  SmoothL1 sum over ``2 * instance pixels``, each divided by its own denominator
  for the current eight-crop accumulation window.  The denominators are counted
  from the target masks before any forward; each crop is then forwarded, scaled,
  and backwarded immediately, so only one graph is retained.  A crop with one
  empty branch is still forwarded, a crop with both branches empty is skipped
  *before* the forward, and missing labels are never used as negatives;
* inference recomputes the semantic map from the PNG only (384/192 tiled,
  averaged logits) together with the offset unit field, then reuses the settled
  vote-clustering decoder.  No ground truth and no saved semantic identity gate
  is read.

Run ``python -m cadvec.recognition_joint_offsets prepare`` for the CPU-only
readiness report.  The default subcommand is ``prepare``; nothing in this module
changes the CLI of any existing lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import png_raster_vectorize as vectorizer
from . import recognition_instance_offsets as offsets
from . import recognition_pretrained_fullfit as fullfit
from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_data as raster_data
from . import recognition_raster_infer as raster_infer
from . import recognition_raster_train as train
from . import recognition_submission as submission

# ---------------------------------------------------------------------------
# constants (settled, not knobs)
# ---------------------------------------------------------------------------

VERSION = 1
CHECKPOINT_SCHEMA = "cadvec.recognition.joint_offsets_checkpoint/1"
PLAN_SCHEMA = "cadvec.recognition.joint_offsets_plan/1"
FIT_SCHEMA = "cadvec.recognition.joint_offsets_fit/1"
GUARD_SCHEMA = "cadvec.recognition.joint_offsets_guard/1"
READINESS_SCHEMA = "cadvec.recognition.joint_offsets_readiness/1"

# Training geometry: 384 px crops, 384 px tiles, 192 px stride.
CROP_SIZE = 384
TILE_SIZE = 384
TILE_STRIDE = 192
CROPS_PER_RECORD = 8
INSTANCE_CROPS_PER_RECORD = 4
INSTANCE_JITTER = 64
CROP_MIN_VALID = fullfit.CROP_MIN_VALID  # 32, the sampler's own gate

# Offset normalisation is the *offset* scale, independent of the crop size.
OFFSET_NORM = 256.0
SMOOTHL1_BETA = 2.0 / OFFSET_NORM
MIN_GROUP_PIXELS = 1  # branch = instance_valid & instance>=0 & known Thing only

# Training.
EPOCHS = 10
ACCUMULATION_STEPS = 8
BASE_SEED = 20261108
EPOCH_SEED_STRIDE = 1000003
HEAD_INIT_SEED = 20261108
TRUNK_LR = 1e-4
TRUNK_WEIGHT_DECAY = 0.01
OFFSET_LR = 1e-3
OFFSET_WEIGHT_DECAY = 0.0
NUM_CLASSES = pilot.NUM_CLASSES  # 35
NUM_THING_CLASSES = 30

# Settled decoder schedule (reused, never swept).
DECODER_POLICY = {
    "bin_px": 8.0,
    "iterations": 3,
    "radius_px": 16.0,
    "merge_px": 8.0,
    "min_cluster_pixels": 16,
    "norm": OFFSET_NORM,
}
RESERVE_AFTER_PEAK_MIB = 1024.0
MIN_FREE_MIB = 512.0

# Input guard mirrors the PNG frontend defaults; accumulation is additionally
# bounded so a single 35-channel logit accumulator cannot silently exhaust RAM.
MAX_INPUT_PIXELS = raster_infer.INPUT_MAX_PIXELS
MAX_INPUT_BYTES = raster_infer.INPUT_MAX_FILE_BYTES
MAX_LOGIT_ACCUM_BYTES = 2 * 1024**3

DEFAULT_MANIFEST = offsets.DEFAULT_MANIFEST
DEFAULT_CACHE_DIR = offsets.DEFAULT_CACHE_DIR
DEFAULT_SEMANTIC_CHECKPOINT = offsets.DEFAULT_SEMANTIC_CHECKPOINT
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "recognition-joint-offsets-20261112-01"
)


class JointOffsetsError(pilot.PilotError):
    """Raised for contract violations of the joint semantic + offset lane."""


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover - environment dependent
        raise JointOffsetsError("numpy is required for the joint lane")
    return np


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - environment dependent
        raise JointOffsetsError("torch is required for the joint lane")
    return torch


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JointOffsetsError(f"{name} must be an int; got {value!r}")
    if value < minimum:
        raise JointOffsetsError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _finite(name: str, value: Any, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointOffsetsError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise JointOffsetsError(f"{name} must be finite; got {value!r}")
    if minimum is not None and number < minimum:
        raise JointOffsetsError(f"{name} must be >= {minimum}; got {number}")
    return number


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def epoch_seed(base_seed: int = BASE_SEED, epoch: int = 1) -> int:
    """Deterministic epoch seed; identical formula to the FIT lanes."""

    return int(base_seed) + int(EPOCH_SEED_STRIDE) * int(epoch)


def deterministic_rng(*parts: Any) -> Any:
    """Deterministic NumPy generator from arbitrary parts.

    Uses the same ``sha256("\\x00".join(parts))`` recipe as the raster
    sampler, so crop choice is reproducible without relying on global state.
    """

    numpy = _require_numpy()
    material = "\x00".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()[:8]
    return numpy.random.default_rng(int.from_bytes(digest, "little"))


def seed_everything(seed: int) -> Dict[str, Any]:
    """Seed python/numpy/torch globals for one epoch of crop selection."""

    import random

    torch = _require_torch()
    numpy = _require_numpy()
    random.seed(int(seed))
    numpy.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    return {
        "python": int(seed),
        "numpy": int(seed) % (2**32),
        "torch": int(seed),
    }


def model_digest(model: Any) -> str:
    """Stable digest of every semantic parameter (name, shape, value)."""

    digest = offsets.semantic_parameter_digest(model)
    material = "|".join(
        f"{name}:{shape}:{total!r}:{absolute!r}"
        for name, shape, total, absolute in digest
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# joint model: one forward, semantic logits + offset units
# ---------------------------------------------------------------------------


class JointEmbedder(torch.nn.Module):
    """Trainable trunk + classifier + offset head in a single forward.

    The semantic trunk is **not** frozen and **not** run under ``no_grad``.  A
    forward hook on ``net.classifier[3]`` (the 256-channel ReLU, stride 8) keeps
    the feature tensor that the classifier's final 1x1 convolution also consumes,
    so one pass through the network produces both heads and gradients reach the
    shared trunk from both losses.  The feature tensor is neither detached nor
    re-computed.
    """

    def __init__(
        self,
        model: Any,
        head: Any,
        *,
        device: str = "cpu",
        tap: Optional[Any] = None,
    ) -> None:
        torch = _require_torch()
        super().__init__()
        self.model = model
        self.head = head
        self.device = device
        module = tap
        if module is None:
            try:
                module = model.net.classifier[3]
            except (AttributeError, IndexError) as error:  # pragma: no cover
                raise JointOffsetsError(
                    "semantic model has no net.classifier[3] tap"
                ) from error
        self.feature: Optional[Any] = None
        self._hook = module.register_forward_hook(self._capture)
        self._tap_channels = int(getattr(module, "out_channels", 0)) or None

    def _capture(self, module: Any, inputs: Any, output: Any) -> None:
        self.feature = output

    def close(self) -> None:
        """Remove the forward hook (idempotent)."""

        handle = getattr(self, "_hook", None)
        if handle is not None:
            handle.remove()
            self._hook = None

    def forward(self, images: Any) -> Dict[str, Any]:
        torch = _require_torch()
        if not torch.is_tensor(images) or images.ndim != 4:
            raise JointOffsetsError("joint forward needs a N x 3 x H x W tensor")
        self.feature = None
        logits = self.model(images)["semantic_logits"]
        features = self.feature
        if features is None:
            raise JointOffsetsError(
                "classifier[3] tap captured no feature; the hook was removed"
            )
        if tuple(logits.shape[-2:]) != tuple(images.shape[-2:]):
            raise JointOffsetsError(
                "semantic logits are not upsampled to the input size: "
                f"{tuple(logits.shape[-2:])} vs {tuple(images.shape[-2:])}"
            )
        # The head runs at the classifier[3] stride (48x48 for a 384 crop) and
        # its two-channel field is bilinear-upsampled to the crop size, exactly
        # like the reused instance-offset lane.  Applying it after upsampling the
        # 256-channel feature map would be a different (and far more expensive)
        # operator, not a reuse of the corrected helper.
        units = self.head(features)
        if tuple(units.shape[-2:]) != tuple(images.shape[-2:]):
            units = F.interpolate(
                units, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        if int(units.shape[1]) != offsets.OFFSET_CHANNELS:
            raise JointOffsetsError(
                f"offset head returned {int(units.shape[1])} channels"
            )
        return {"semantic_logits": logits, "offset_units": units}

    # -- trainable parameter groups ------------------------------------
    def trainable_parameters(self) -> Tuple[List[Any], List[Any]]:
        trunk = [p for p in self.model.parameters() if p.requires_grad]
        head = [p for p in self.head.parameters() if p.requires_grad]
        return trunk, head

    def training_state(self) -> bool:
        """True when either submodule is in training mode (container flag is unused)."""

        return bool(self.model.training or self.head.training)

    def train_mode(self) -> int:
        """Model/head trainable, every BatchNorm left in eval (stats frozen)."""
        self.model.train()
        self.head.train()
        return pilot.freeze_batch_norm(self.model)

    def eval_mode(self) -> None:
        self.model.eval()
        self.head.eval()


def build_joint_embedder(
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    *,
    device: str = "cpu",
    head_seed: int = HEAD_INIT_SEED,
) -> Tuple[JointEmbedder, Dict[str, Any]]:
    """Fresh offset head first, then the pure-FIT semantic checkpoint.

    The head is built before the trunk is constructed so its explicit-generator
    initialisation can never be perturbed by (or depend on) model construction
    order or the global RNG.  Both the trunk and the classifier stay trainable;
    BatchNorm running statistics are frozen at training time by
    :meth:`JointEmbedder.train_mode`.
    """

    torch = _require_torch()
    head = offsets.build_offset_head(
        in_channels=offsets.HEAD_IN_CHANNELS,
        hidden_channels=offsets.HEAD_HIDDEN_CHANNELS,
        out_channels=offsets.OFFSET_CHANNELS,
        seed=int(head_seed),
        device="cpu",
    )
    model, payload = pilot.load_checkpoint_model(semantic_checkpoint, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    head.to(device)
    embedder = JointEmbedder(model, head, device=device)
    report = {
        "semantic_checkpoint": os.path.abspath(semantic_checkpoint),
        "semantic_schema": payload.get("schema"),
        "semantic_role": payload.get("role"),
        "semantic_epoch": payload.get("epoch"),
        "semantic_epochs": payload.get("epochs"),
        "initial_semantic_digest": model_digest(model),
        "head_init_seed": int(head_seed),
        "head_parameters": int(sum(p.numel() for p in head.parameters())),
        "trunk_trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "batch_norm_policy": (
            "running statistics frozen (eval after train()); affine weights and "
            "biases remain trainable"
        ),
        "tap": "net.classifier[3] (stride 8, 256 channels)",
    }
    return embedder, report


# ---------------------------------------------------------------------------
# crop planning: 4 instance-centred + 4 deterministic uniform, or 8 uniform
# ---------------------------------------------------------------------------


def record_centroids(
    entry: Mapping[str, Any], *, num_thing_classes: int = NUM_THING_CLASSES
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Full-native Thing-instance centroids (never a cropped centroid)."""

    return offsets.native_group_centroids(
        entry["instance"],
        entry["instance_valid"],
        entry["semantic"],
        entry["semantic_valid"],
        num_thing_classes=int(num_thing_classes),
    )


def eligible_instance_groups(
    centroids: Mapping[Tuple[int, int], Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Deterministic ordering by ``(class, instance)`` for round-robin cycling."""

    return [
        dict(centroids[key])
        for key in sorted(centroids.keys(), key=lambda key: (int(key[0]), int(key[1])))
    ]


class JointCropPlanner:
    """Deterministic per-record crop plan for one epoch seed.

    If at least one known Thing instance exists, slots 0..3 are
    instance-centred crops (cycling eligible instances from a seeded rotation
    with a seeded +/-64 px jitter) and slots 4..7 are the sampler's own
    deterministic uniform crops.
    If no eligible instance exists, all eight slots are uniform crops.  No crop
    is random at the window level: every centre is a pure function of the epoch
    seed, record index and slot.
    """

    def __init__(
        self,
        *,
        seed: int,
        crop_size: int = CROP_SIZE,
        crops_per_record: int = CROPS_PER_RECORD,
        instance_crops: int = INSTANCE_CROPS_PER_RECORD,
        jitter: int = INSTANCE_JITTER,
        min_valid: int = CROP_MIN_VALID,
    ) -> None:
        self.seed = int(seed)
        self.crop_size = _positive_int("crop_size", int(crop_size))
        self.crops_per_record = _positive_int(
            "crops_per_record", int(crops_per_record)
        )
        self.instance_crops = _positive_int("instance_crops", int(instance_crops))
        if self.instance_crops > self.crops_per_record:
            raise JointOffsetsError("instance_crops must not exceed crops_per_record")
        self.jitter = _positive_int("jitter", int(jitter), minimum=0)
        self.min_valid = _positive_int("min_valid", int(min_valid), minimum=0)
        self.sampler = train.SupportCropSampler(
            crop_size=self.crop_size,
            crops_per_record=self.crops_per_record,
            background_fraction=0.0,
            jitter=0,
            min_valid=self.min_valid,
            seed=self.seed,
        )

    def _rng(self, record_index: int, slot: int) -> Any:
        return deterministic_rng(self.seed, "instance-crop", record_index, slot)

    def plan(
        self,
        entry: Mapping[str, Any],
        *,
        record_index: int,
        centroids: Optional[Mapping[Tuple[int, int], Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return exactly ``crops_per_record`` deterministic slot specs."""

        if centroids is None:
            centroids = record_centroids(entry)
        groups = eligible_instance_groups(centroids)
        anchors = self.sampler.build_anchors(entry, record_index=int(record_index))
        start = 0
        if groups:
            start = int(
                self._rng(record_index, -1).integers(0, len(groups))
            )
        slots: List[Dict[str, Any]] = []
        for slot in range(self.crops_per_record):
            if groups and slot < min(self.instance_crops, self.crops_per_record):
                group = groups[(start + slot) % len(groups)]
                rng = self._rng(record_index, slot)
                delta = rng.integers(-self.jitter, self.jitter + 1, size=2)
                center = (
                    int(round(float(group["row"]))) + int(delta[0]),
                    int(round(float(group["column"]))) + int(delta[1]),
                )
                slots.append(
                    {
                        "slot": int(slot),
                        "kind": "instance",
                        "class": int(group["class"]),
                        "instance": int(group["instance"]),
                        "center": (int(center[0]), int(center[1])),
                        "anchor": None,
                    }
                )
            else:
                anchor = anchors[slot]
                slots.append(
                    {
                        "slot": int(slot),
                        "kind": "uniform",
                        "class": None,
                        "instance": None,
                        "center": None,
                        "anchor": anchor,
                    }
                )
        return slots


def prepare_crop(
    planner: JointCropPlanner,
    entry: Mapping[str, Any],
    centroids: Mapping[Tuple[int, int], Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract one crop with independent masks and full-native offset targets.

    The semantic target keeps the cache's own ``semantic_valid`` and known
    classes; the offset mask is the branch contract
    ``instance_valid & instance >= 0 & semantic_valid & 0 <= semantic < 30``.
    Unknown padding becomes ``-1`` / ``False`` for the respective target and is
    never treated as a negative sample.  ``min_group_pixels=1`` makes the branch
    exactly the stated contract (no extra per-crop group gate).
    """

    numpy = _require_numpy()
    authoritative: Optional[Mapping[str, Any]] = None
    if spec.get("kind") == "instance":
        window = offsets.extract_window(
            entry, spec["center"], size=planner.crop_size
        )
    elif spec.get("kind") == "uniform":
        anchor = spec["anchor"]
        authoritative = planner.sampler.crop(entry, anchor)
        window = offsets.select_window(planner.sampler, entry, anchor)
    else:  # pragma: no cover - defensive
        raise JointOffsetsError(f"unknown crop kind {spec.get('kind')!r}")
    targets = offsets.crop_offset_targets(
        window,
        centroids,
        authoritative=authoritative,
        min_group_pixels=MIN_GROUP_PIXELS,
        norm=OFFSET_NORM,
        num_thing_classes=NUM_THING_CLASSES,
    )
    semantic = numpy.asarray(window["semantic"], dtype=numpy.int64)
    semantic_valid = numpy.asarray(window["semantic_valid"], dtype=bool)
    known_mask = (
        semantic_valid
        & (semantic >= 0)
        & (semantic < NUM_CLASSES)
    )
    semantic_target = numpy.full(semantic.shape, -1, dtype=numpy.int64)
    semantic_target[known_mask] = semantic[known_mask]
    semantic_known = int(known_mask.sum())
    crop = {
        "image": numpy.ascontiguousarray(window["image"], dtype=numpy.float32),
        "semantic_target": semantic_target,
        "semantic_known": semantic_known,
        "offset_targets": targets,
        "offset_mask": targets["mask"],
        "offset_pixels": int(targets["pixels"]),
        "offset_groups": targets["groups"],
        "offset_group_count": int(targets["group_count"]),
        "max_abs_unit": float(targets["max_abs_unit"]),
        "slot": int(spec["slot"]),
        "kind": str(spec["kind"]),
        "class": spec.get("class"),
        "instance": spec.get("instance"),
        "top": int(window["top"]),
        "left": int(window["left"]),
        "padded": bool(window["padded"]),
        "sample_id": str(window.get("sample_id") or ""),
        "crop_size": int(planner.crop_size),
    }
    crop["truncated_groups"] = sum(
        1 for group in targets["groups"] if not group["centroid_inside_crop"]
    )
    crop["skip"] = bool(semantic_known == 0 and crop["offset_pixels"] == 0)
    return crop


class JointEpochSource:
    """One epoch of deterministic crops over the fixed FIT records."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        cache_dir: Optional[str],
        epoch: int = 1,
        base_seed: int = BASE_SEED,
        crop_size: int = CROP_SIZE,
        crops_per_record: int = CROPS_PER_RECORD,
        instance_crops: int = INSTANCE_CROPS_PER_RECORD,
    ) -> None:
        self.records = list(records)
        self.cache_dir = cache_dir
        self.epoch = _positive_int("epoch", int(epoch))
        self.seed = epoch_seed(base_seed, self.epoch)
        self.base = train.SampleSource(self.records, cache_dir=cache_dir)
        self.planner = JointCropPlanner(
            seed=self.seed,
            crop_size=int(crop_size),
            crops_per_record=int(crops_per_record),
            instance_crops=int(instance_crops),
        )
        self.crops_per_record = int(crops_per_record)

    def __len__(self) -> int:
        return len(self.records)

    def get(self, record_index: int) -> Dict[str, Any]:
        entry = self.base.get(record_index)
        centroids = record_centroids(entry)
        specs = self.planner.plan(
            entry, record_index=record_index, centroids=centroids
        )
        crops = [
            prepare_crop(self.planner, entry, centroids, spec) for spec in specs
        ]
        return {
            "record_index": int(record_index),
            "entry": entry,
            "centroids": centroids,
            "specs": specs,
            "crops": crops,
        }


# ---------------------------------------------------------------------------
# losses: branch-local sums, window denominators
# ---------------------------------------------------------------------------


def semantic_sum_loss(logits: Any, target: Any) -> Tuple[Any, int]:
    """Cross-entropy **sum** over known pixels (unknown ignored, never negative)."""

    torch = _require_torch()
    mean_ce, counts = pilot.masked_semantic_ce(logits, target)
    known = int(counts["known_pixels"])
    if known <= 0:
        return mean_ce * 0.0, 0
    loss_sum = mean_ce * float(known)
    if not bool(torch.isfinite(loss_sum)):
        raise JointOffsetsError("non_finite_semantic_loss")
    return loss_sum, known


def offset_sum_loss(
    head_units: Any,
    targets: Mapping[str, Any],
    *,
    beta: float = SMOOTHL1_BETA,
) -> Tuple[Any, int]:
    """Masked SmoothL1 **sum** over ``2 * instance pixels`` (no denominator)."""

    torch = _require_torch()
    numpy = _require_numpy()
    if not torch.is_tensor(head_units) or head_units.ndim != 4:
        raise JointOffsetsError("head units must be a N x 2 x H x W tensor")
    if int(head_units.shape[1]) != offsets.OFFSET_CHANNELS:
        raise JointOffsetsError("head units must have two channels")
    target = torch.from_numpy(
        numpy.array(targets["offsets"], dtype=numpy.float32, copy=True)
    ).to(head_units.device)
    mask = torch.from_numpy(
        numpy.array(targets["mask"], dtype=bool, copy=True)
    ).to(head_units.device)
    if tuple(target.shape[-2:]) != tuple(head_units.shape[-2:]):
        raise JointOffsetsError("offset target/head size mismatch")
    pixels = int(targets.get("pixels", 0))
    if pixels <= 0:
        raise JointOffsetsError("offset_sum_loss called without instance pixels")
    if int(mask.sum().item()) != pixels:
        raise JointOffsetsError("offset mask and pixel count disagree")
    element = F.smooth_l1_loss(
        head_units, target.unsqueeze(0), beta=float(beta), reduction="none"
    )
    weight = mask.unsqueeze(0).unsqueeze(0).to(element.dtype)
    total = (element * weight).sum()
    if not bool(torch.isfinite(total)):
        raise JointOffsetsError("non_finite_offset_loss")
    return total, int(pixels)


def _assert_finite_gradients(embedder: JointEmbedder) -> None:
    torch = _require_torch()
    for parameter in list(embedder.parameters()):
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise JointOffsetsError("non_finite_gradient")


def run_window(
    embedder: JointEmbedder,
    crops: Sequence[Mapping[str, Any]],
    optimizer: Any,
    *,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Stream forward/backward/step over one accumulation window (<= 8 crops).

    Denominators are computed from the **window targets**, before any forward:
    ``semantic_denominator = sum(known pixels)`` and
    ``offset_denominator = sum(instance pixels)``.  Each crop is forwarded,
    scaled by ``count / denominator``, and backwarded **immediately**, so only
    one autograd graph is alive at a time.  The optimizer steps once after the
    window; per-crop losses are never divided again by the accumulation count.
    A crop whose semantic count is zero contributes no semantic term but is
    still forwarded if its offset branch is non-empty (and vice versa); a crop
    with both branches empty must have been filtered out by the caller.
    """

    torch = _require_torch()
    numpy = _require_numpy()
    if not crops:
        raise JointOffsetsError("run_window needs at least one supervised crop")
    semantic_denominator = int(sum(int(c["semantic_known"]) for c in crops))
    offset_denominator = int(sum(int(c["offset_pixels"]) for c in crops))
    if semantic_denominator == 0 and offset_denominator == 0:
        raise JointOffsetsError("run_window called with an entirely empty window")
    optimizer.zero_grad(set_to_none=True)
    semantic_sum = 0.0
    offset_sum = 0.0
    semantic_crops = 0
    offset_crops = 0
    for crop in crops:
        known = int(crop["semantic_known"])
        pixels = int(crop["offset_pixels"])
        if known == 0 and pixels == 0:
            raise JointOffsetsError(
                "zero-supervision crop reached run_window; filter it before forward"
            )
        image = torch.from_numpy(
            numpy.ascontiguousarray(crop["image"], dtype=numpy.float32)
        ).unsqueeze(0).to(device)
        output = embedder(image)
        loss = None
        if semantic_denominator > 0 and known > 0:
            target = torch.from_numpy(
                numpy.ascontiguousarray(crop["semantic_target"], dtype=numpy.int64)
            ).unsqueeze(0).to(device)
            term, counted = semantic_sum_loss(
                output["semantic_logits"], target
            )
            if counted != known:
                raise JointOffsetsError(
                    f"semantic target count drift: {counted} vs {known}"
                )
            loss = term / float(semantic_denominator)
            semantic_sum += float(term.detach().item())
            semantic_crops += 1
        if offset_denominator > 0 and pixels > 0:
            term, counted = offset_sum_loss(
                output["offset_units"], crop["offset_targets"]
            )
            if counted != pixels:
                raise JointOffsetsError("offset target count drift")
            offset_term = term / (2.0 * float(offset_denominator))
            loss = offset_term if loss is None else loss + offset_term
            offset_sum += float(term.detach().item())
            offset_crops += 1
        if loss is None:  # pragma: no cover - guarded above
            raise JointOffsetsError("crop produced no loss term")
        loss.backward()
        del loss, output, image
    _assert_finite_gradients(embedder)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    semantic_loss = (
        semantic_sum / float(semantic_denominator) if semantic_denominator else 0.0
    )
    offset_loss = (
        offset_sum / (2.0 * float(offset_denominator)) if offset_denominator else 0.0
    )
    return {
        "crops": len(crops),
        "semantic_crops": semantic_crops,
        "offset_crops": offset_crops,
        "semantic_pixels": semantic_denominator,
        "offset_pixels": offset_denominator,
        "semantic_sum": round(semantic_sum, 6),
        "offset_sum": round(offset_sum, 6),
        "semantic_loss": round(semantic_loss, 8),
        "offset_loss": round(offset_loss, 8),
        "loss": round(semantic_loss + offset_loss, 8),
        "optimizer_steps": 1,
    }


# ---------------------------------------------------------------------------
# checkpoint (full trunk + fresh head, final epoch only)
# ---------------------------------------------------------------------------


def _semantic_contract(
    semantic_checkpoint: str,
    payload: Mapping[str, Any],
    *,
    initial_digest: str,
) -> Dict[str, Any]:
    return {
        "path": os.path.abspath(semantic_checkpoint),
        "schema": payload.get("schema"),
        "role": payload.get("role"),
        "epoch": payload.get("epoch"),
        "epochs": payload.get("epochs"),
        "num_classes": NUM_CLASSES,
        "initial_digest": str(initial_digest),
        "joint_trained": True,
        "batch_norm_running_stats_frozen": True,
    }


def save_joint_checkpoint(
    path: str,
    embedder: JointEmbedder,
    *,
    epoch: int,
    epochs: int,
    semantic_checkpoint: str,
    semantic_payload: Mapping[str, Any],
    initial_semantic_digest: str,
    metrics: Optional[Mapping[str, Any]] = None,
    training: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Final-epoch-only checkpoint carrying the whole joint model.

    ``training`` must carry the effective arguments of the run that produced the
    weights; module defaults are only a fallback for direct helper calls.
    """

    torch = _require_torch()
    effective_training: Dict[str, Any] = {
        "epochs": int(epochs),
        "trunk_lr": TRUNK_LR,
        "trunk_weight_decay": TRUNK_WEIGHT_DECAY,
        "offset_lr": OFFSET_LR,
        "offset_weight_decay": OFFSET_WEIGHT_DECAY,
        "optimizer": "AdamW",
        "accumulation_steps": ACCUMULATION_STEPS,
        "base_seed": BASE_SEED,
        "epoch_seed_stride": EPOCH_SEED_STRIDE,
        "head_init_seed": HEAD_INIT_SEED,
        "batch_norm": "running statistics frozen; affine parameters trainable",
        "amp": False,
        "early_stopping": False,
        "final_epoch_only": True,
    }
    if training is not None:
        effective_training.update(dict(training))
    seed_base = int(effective_training["base_seed"])
    seed_stride = int(effective_training["epoch_seed_stride"])
    head_seed = int(effective_training["head_init_seed"])
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": VERSION,
        "kind": "joint_semantic_offsets",
        "final_epoch_only": True,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "geometry": {
            "crop_size": int(CROP_SIZE),
            "tile_size": int(TILE_SIZE),
            "tile_stride": int(TILE_STRIDE),
            "offset_norm": float(OFFSET_NORM),
            "feature_stride": offsets.FEATURE_STRIDE,
            "num_classes": NUM_CLASSES,
            "num_thing_classes": NUM_THING_CLASSES,
            "tap": "net.classifier[3]",
            "tap_channels": offsets.HEAD_IN_CHANNELS,
            "head": "Conv2d(256, 64, 1) -> ReLU -> Conv2d(64, 2, 1)",
            "head_in_channels": int(offsets.HEAD_IN_CHANNELS),
            "head_hidden_channels": int(offsets.HEAD_HIDDEN_CHANNELS),
        },
        "model_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in embedder.model.state_dict().items()
        },
        "head_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in embedder.head.state_dict().items()
        },
        "semantic": _semantic_contract(
            semantic_checkpoint,
            semantic_payload,
            initial_digest=initial_semantic_digest,
        ),
        "training": effective_training,
        "seeds": {
            "base_seed": seed_base,
            "epoch_seed_stride": seed_stride,
            "head_init_seed": head_seed,
            "epoch_seeds": {
                str(index): epoch_seed(seed_base, index)
                for index in range(1, int(epochs) + 1)
            },
        },
        "decoder": {
            "instance_lane_reused": True,
            **DECODER_POLICY,
            "labels": "class indices 0..34, -1 unknown (raw 35 accepted as ignore)",
            "instances": "globally unique ids >= 0 on every Thing pixel, -1 otherwise",
            "stuff": "merged per class by the unchanged vectorizer",
            "ground_truth_at_inference": False,
            "saved_semantic_gate": False,
        },
        "inference": {
            "semantic_source": "recomputed from PNG via 384/192 tiled averaged logits",
            "offset_source": "same forward, head units, overlap averaged, *256 once in decode",
            "tile_size": int(TILE_SIZE),
            "tile_stride": int(TILE_STRIDE),
            "crop_size": int(CROP_SIZE),
            "white_pad": True,
            "resize": None,
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


def load_joint_checkpoint(path: str, *, device: str = "cpu") -> Dict[str, Any]:
    """Strict loader for a joint checkpoint (no silent fallback to 256)."""

    torch = _require_torch()
    if not os.path.isfile(path):
        raise JointOffsetsError(f"checkpoint_not_found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise JointOffsetsError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("kind") != "joint_semantic_offsets":
        raise JointOffsetsError(f"checkpoint_kind_mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise JointOffsetsError("checkpoint_must_be_final_epoch_only")
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        raise JointOffsetsError("checkpoint_missing_geometry")
    if int(geometry.get("crop_size", -1)) != CROP_SIZE:
        raise JointOffsetsError(
            f"checkpoint_crop_size_mismatch: expected {CROP_SIZE}, "
            f"got {geometry.get('crop_size')!r}"
        )
    if int(geometry.get("tile_size", -1)) != TILE_SIZE:
        raise JointOffsetsError("checkpoint_tile_size_mismatch")
    if int(geometry.get("tile_stride", -1)) != TILE_STRIDE:
        raise JointOffsetsError("checkpoint_tile_stride_mismatch")
    if float(geometry.get("offset_norm", -1.0)) != OFFSET_NORM:
        raise JointOffsetsError("checkpoint_offset_norm_mismatch")
    if int(geometry.get("num_classes", -1)) != NUM_CLASSES:
        raise JointOffsetsError("checkpoint_num_classes_mismatch")
    if str(geometry.get("tap")) != "net.classifier[3]":
        raise JointOffsetsError("checkpoint_tap_mismatch")
    head_in_channels = _positive_int(
        "head_in_channels", int(geometry.get("head_in_channels", -1))
    )
    head_hidden_channels = _positive_int(
        "head_hidden_channels", int(geometry.get("head_hidden_channels", -1))
    )
    model_state = payload.get("model_state_dict")
    head_state = payload.get("head_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise JointOffsetsError("checkpoint_missing_model_state")
    if not isinstance(head_state, Mapping) or not head_state:
        raise JointOffsetsError("checkpoint_missing_head_state")
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping):
        raise JointOffsetsError("checkpoint_missing_semantic_reference")
    if semantic.get("joint_trained") is not True:
        raise JointOffsetsError("checkpoint_semantic_contract_not_joint")
    result = dict(payload)
    result["model_state_dict"] = {
        key: value.to(device) for key, value in model_state.items()
    }
    result["head_state_dict"] = {
        key: value.to(device) for key, value in head_state.items()
    }
    result["semantic_checkpoint_exists"] = bool(
        os.path.isfile(str(semantic.get("path", "")))
    )
    return result


def load_joint_embedder(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    model_factory: Optional[Any] = None,
) -> Tuple[JointEmbedder, Dict[str, Any]]:
    """Rebuild a joint embedder from a checkpoint.

    ``model_factory`` exists for the synthetic tests (a small stub model cannot
    be produced by :func:`pilot.build_pilot_model`); production callers leave it
    as ``None`` and get DeepLabV3-ResNet50 built without network access.
    """

    torch = _require_torch()
    payload = load_joint_checkpoint(checkpoint_path, device=device)
    if model_factory is None:
        model, _provenance = pilot.build_pilot_model(pretrained=False)
    else:
        model = model_factory()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    head = offsets.build_offset_head(
        in_channels=int(payload["geometry"]["head_in_channels"]),
        hidden_channels=int(payload["geometry"]["head_hidden_channels"]),
        out_channels=offsets.OFFSET_CHANNELS,
        seed=int(payload["seeds"]["head_init_seed"]),
        device="cpu",
    )
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.to(device)
    embedder = JointEmbedder(model, head, device=device)
    embedder.eval_mode()
    return embedder, payload


# ---------------------------------------------------------------------------
# inference: PNG only, 384/192 tiled logits + offset units, reused decoder
# ---------------------------------------------------------------------------


_ACCUM_CHANNELS = NUM_CLASSES + offsets.OFFSET_CHANNELS + 1  # logits + units + weight


def _check_accumulation_bytes(height: int, width: int, *, padded: bool = False) -> None:
    needed = _ACCUM_CHANNELS * int(height) * int(width) * 4
    if needed > MAX_LOGIT_ACCUM_BYTES:
        shape = "padded accumulator" if padded else "image"
        raise JointOffsetsError(
            f"inference_accumulation_too_large: {shape} {width}x{height} needs "
            f"{_ACCUM_CHANNELS} float32 channels (35 logits + 2 units + 1 weight) = "
            f"{needed} bytes; the guard is {MAX_LOGIT_ACCUM_BYTES} bytes"
        )


def _as_image_tensor(image: Any) -> Any:
    torch = _require_torch()
    numpy = _require_numpy()
    if torch.is_tensor(image):
        source = image.detach().to("cpu", dtype=torch.float32)
    else:
        source = torch.from_numpy(
            numpy.ascontiguousarray(numpy.asarray(image, dtype=numpy.float32))
        )
    if source.ndim != 3 or int(source.shape[0]) != 3:
        raise JointOffsetsError(
            f"image must be 3 x H x W in unit range; got {tuple(source.shape)}"
        )
    if not bool(torch.isfinite(source).all()):
        raise JointOffsetsError("non_finite_image")
    height, width = int(source.shape[-2]), int(source.shape[-1])
    if height <= 0 or width <= 0:
        raise JointOffsetsError("empty image")
    if height * width > MAX_INPUT_PIXELS:
        raise JointOffsetsError(
            f"image is {width}x{height} = {height * width} pixels, above "
            f"max_pixels={MAX_INPUT_PIXELS}; no resize is applied"
        )
    _check_accumulation_bytes(height, width)
    return source


def _image_ink_support(image: Any) -> Any:
    """Exact baseline PNG ink support from unit-range CHW pixels.

    Delegates to the shared data helper (min composited RGB < 254) instead of
    re-implementing the threshold, so inference uses the same float comparison
    and boundary behaviour as the semantic baseline.
    """

    numpy = _require_numpy()
    source = _as_image_tensor(image)
    return numpy.asarray(
        raster_data._image_ink_support(source.numpy()), dtype=bool
    )


def predict_joint(
    embedder: JointEmbedder,
    image: Any,
    *,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompute semantics and offset units from one image (no labels read).

    The source is never resized; the bottom/right edge is white-padded to at
    least one tile, the tile positions match the FIT tiling helper, and both
    heads are accumulated in float32 with overlap averaging.  The offset field is
    returned in **head units**; the single ``* 256`` happens inside
    :func:`decode_units`.
    """

    torch = _require_torch()
    numpy = _require_numpy()
    source = _as_image_tensor(image)
    tile_size = _positive_int("tile_size", int(tile_size))
    stride = _positive_int("stride", int(stride))
    if stride > tile_size:
        raise JointOffsetsError("stride must not exceed tile size")
    height, width = int(source.shape[-2]), int(source.shape[-1])
    padded_height = max(tile_size, height)
    padded_width = max(tile_size, width)
    _check_accumulation_bytes(padded_height, padded_width, padded=True)
    if padded_height != height or padded_width != width:
        padded = F.pad(
            source.unsqueeze(0),
            (0, padded_width - width, 0, padded_height - height),
            mode="constant",
            value=1.0,
        ).squeeze(0)
    else:
        padded = source
    rows = fullfit._tile_positions(padded_height, tile_size, stride)
    columns = fullfit._tile_positions(padded_width, tile_size, stride)
    run_device = device if device is not None else getattr(embedder, "device", "cpu")
    state = getattr(embedder, "training_state", None)
    was_training = bool(state()) if callable(state) else bool(
        getattr(embedder, "training", False)
    )
    embedder.eval_mode()
    accum_logits = torch.zeros(
        (NUM_CLASSES, padded_height, padded_width), dtype=torch.float32
    )
    accum_units = torch.zeros(
        (offsets.OFFSET_CHANNELS, padded_height, padded_width), dtype=torch.float32
    )
    weights = torch.zeros((1, padded_height, padded_width), dtype=torch.float32)
    forward_started = time.monotonic()
    try:
        with torch.no_grad():
            for row in rows:
                for column in columns:
                    window = padded[
                        :, row : row + tile_size, column : column + tile_size
                    ].unsqueeze(0).to(run_device)
                    output = embedder(window)
                    logits = output["semantic_logits"]
                    units = output["offset_units"]
                    expected_logits = (1, NUM_CLASSES, tile_size, tile_size)
                    expected_units = (1, offsets.OFFSET_CHANNELS, tile_size, tile_size)
                    if tuple(logits.shape) != expected_logits:
                        raise JointOffsetsError(
                            f"unexpected tile logits shape {tuple(logits.shape)}"
                        )
                    if tuple(units.shape) != expected_units:
                        raise JointOffsetsError(
                            f"unexpected tile unit shape {tuple(units.shape)}"
                        )
                    accum_logits[
                        :, row : row + tile_size, column : column + tile_size
                    ] += logits[0].detach().to("cpu", dtype=torch.float32)
                    accum_units[
                        :, row : row + tile_size, column : column + tile_size
                    ] += units[0].detach().to("cpu", dtype=torch.float32)
                    weights[
                        :, row : row + tile_size, column : column + tile_size
                    ] += 1.0
    finally:
        # Do not use embedder.train(True): that would put BatchNorm back into
        # training mode and resume updating its running statistics.
        if was_training:
            embedder.train_mode()
        else:
            embedder.eval_mode()
    logits = (accum_logits / weights)[:, :height, :width].contiguous()
    units = (accum_units / weights)[:, :height, :width].contiguous()
    if not bool(torch.isfinite(units).all()):
        raise JointOffsetsError("non_finite_offset_units")
    if not bool(torch.isfinite(logits).all()):
        raise JointOffsetsError("non_finite_semantic_logits")
    semantic = logits.argmax(dim=0).to(torch.int16).numpy()
    forward_seconds = round(time.monotonic() - forward_started, 6)
    return {
        "semantic": semantic,
        "semantic_logits": logits,
        "units": units.numpy().astype(numpy.float32, copy=False),
        "tile_rows": len(rows),
        "tile_columns": len(columns),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "padded": bool(padded_height != height or padded_width != width),
        "source_size": (int(height), int(width)),
        "timings": {"tile_forward_seconds": forward_seconds},
    }


def decode_units(
    semantic: Any,
    units: Any,
    *,
    semantic_valid: Any = None,
    bin_px: float = DECODER_POLICY["bin_px"],
    iterations: int = DECODER_POLICY["iterations"],
    radius_px: float = DECODER_POLICY["radius_px"],
    merge_px: float = DECODER_POLICY["merge_px"],
    min_cluster_pixels: int = DECODER_POLICY["min_cluster_pixels"],
) -> Dict[str, Any]:
    """Reuse the settled decoder; ``units * 256`` happens exactly once inside."""

    numpy = _require_numpy()
    field = numpy.asarray(units, dtype=numpy.float32)
    if not bool(numpy.isfinite(field).all()):
        raise JointOffsetsError("non_finite_offset_units")
    return offsets.decode_offsets(
        semantic,
        field,
        semantic_valid=semantic_valid,
        bin_px=float(bin_px),
        iterations=int(iterations),
        radius_px=float(radius_px),
        merge_px=float(merge_px),
        min_cluster_pixels=int(min_cluster_pixels),
        norm=OFFSET_NORM,
        num_thing_classes=NUM_THING_CLASSES,
    )


def predict_and_decode(
    embedder: JointEmbedder,
    image: Any,
    *,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    device: Optional[str] = None,
    semantic_valid: Any = None,
) -> Dict[str, Any]:
    """PNG-only joint inference: recomputed semantics + offset units -> instances."""

    started = time.monotonic()
    field = predict_joint(
        embedder, image, tile_size=tile_size, stride=stride, device=device
    )
    numpy = _require_numpy()
    ink = _image_ink_support(image)
    source = "png_rgb_ink"
    if semantic_valid is not None:
        mask = numpy.asarray(semantic_valid, dtype=bool)
        if mask.shape != ink.shape:
            raise JointOffsetsError("semantic_valid shape does not match the image")
        # Never let a caller-supplied mask add pixels outside PNG ink.
        ink = ink & mask
        source = "png_rgb_ink_intersect_semantic_valid"
    support_pixels = int(ink.sum())
    semantic_input = numpy.array(field["semantic"], copy=True)
    semantic_input[~ink] = -1
    decode_started = time.monotonic()
    decoded = decode_units(
        semantic_input, field["units"], semantic_valid=ink
    )
    semantic_out = numpy.array(decoded["semantic"], copy=True)
    # The vectorizer needs 0..35 and rejects -1; 35 is the ignore class, so
    # white pixels outside PNG ink produce no geometry at all.
    semantic_out[~ink] = offsets.IGNORE_CLASS
    decode_seconds = round(time.monotonic() - decode_started, 6)
    total_seconds = round(time.monotonic() - started, 6)
    timings = {
        "tile_forward_seconds": float(field["timings"]["tile_forward_seconds"]),
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
    }
    stats = dict(decoded["stats"])
    stats["tile_rows"] = int(field["tile_rows"])
    stats["tile_columns"] = int(field["tile_columns"])
    stats["tile_size"] = int(field["tile_size"])
    stats["tile_stride"] = int(field["stride"])
    stats["offset_norm"] = float(OFFSET_NORM)
    stats["crop_size"] = int(CROP_SIZE)
    stats["trained_crop_size"] = int(CROP_SIZE)
    stats["inference_geometry"] = "384 tile / 192 stride / white pad, no resize"
    stats["saved_semantic_gate"] = False
    stats["ground_truth_at_inference"] = False
    stats["semantic_source"] = "recomputed"
    stats["ink_support_pixels"] = support_pixels
    stats["ink_support_policy"] = dict(raster_data.INK_SUPPORT_POLICY)
    stats["semantic_valid_source"] = source
    stats["outside_ink_policy"] = "semantic 35 (ignore) and instance -1"
    stats["timings"] = timings
    return {
        "semantic": semantic_out,
        "instance": decoded["instance"],
        "ink_support": ink,
        "stats": stats,
        "units": field["units"],
        "semantic_logits": field["semantic_logits"],
        "timings": timings,
        "tiling": {
            "tile_size": int(field["tile_size"]),
            "stride": int(field["stride"]),
            "tile_rows": int(field["tile_rows"]),
            "tile_columns": int(field["tile_columns"]),
            "padded": bool(field["padded"]),
            "source_size": field["source_size"],
            "policy": "384/192, white pad bottom/right, no resize, logits+units averaged",
        },
    }


def vectorize_joint(
    semantic: Any,
    instance: Any,
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    viewbox: Sequence[float] = submission.VIEWBOX,
    tol_px: float = 0.5,
) -> Dict[str, Any]:
    """Unchanged vectorizer: Stuff merged per class, Thing ids consumed as given."""

    return vectorizer.vectorize_masks(
        semantic,
        instance,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
        viewbox=viewbox,
        tol_px=tol_px,
    )


def predict_png_joint(
    embedder: JointEmbedder,
    png_path: str,
    *,
    device: Optional[str] = None,
    tol_px: float = 0.5,
) -> Dict[str, Any]:
    """Predict one PNG end to end (semantics recomputed, no GT, no saved map)."""

    started = time.monotonic()
    raster_infer._check_png_input(png_path)
    image = raster_infer._read_png(png_path)
    read_seconds = round(time.monotonic() - started, 6)
    height, width = int(image.shape[1]), int(image.shape[2])
    affine = raster_infer._viewbox_affine(width, height)
    predict_started = time.monotonic()
    result = predict_and_decode(embedder, image, device=device)
    predict_seconds = round(time.monotonic() - predict_started, 6)
    vectorize_started = time.monotonic()
    document = vectorize_joint(
        result["semantic"],
        result["instance"],
        scale_x=affine["scale_x"],
        scale_y=affine["scale_y"],
        offset_x=affine["offset_x"],
        offset_y=affine["offset_y"],
        tol_px=float(tol_px),
    )
    vectorize_seconds = round(time.monotonic() - vectorize_started, 6)
    timings = {
        "read_seconds": read_seconds,
        "predict_and_decode_seconds": predict_seconds,
        "vectorize_seconds": vectorize_seconds,
        "tile_forward_seconds": float(result["timings"]["tile_forward_seconds"]),
        "decode_seconds": float(result["timings"]["decode_seconds"]),
        "total_seconds": round(time.monotonic() - started, 6),
    }
    result["stats"]["timings"] = timings
    return {
        "document": document,
        "semantic": result["semantic"],
        "instance": result["instance"],
        "ink_support": result["ink_support"],
        "units": result["units"],
        "stats": result["stats"],
        "metadata": {
            "png_path": str(Path(png_path).absolute()),
            "source_size": [width, height],
            "affine": {
                "scale_x": affine["scale_x"],
                "scale_y": affine["scale_y"],
                "offset_x": affine["offset_x"],
                "offset_y": affine["offset_y"],
            },
            "viewbox": list(submission.VIEWBOX),
            "viewbox_mapping": (
                "uniform xMidYMid meet: s=min(W,H)/100, tx=(W-100s)/2, "
                "ty=(H-100s)/2, svg=(px-t)/s"
            ),
            "viewbox_mapping_status": (
                "unconfirmed assumption; no SVG metadata at inference"
            ),
            "train_crop_size": int(CROP_SIZE),
            "tile_size": int(TILE_SIZE),
            "tile_stride": int(TILE_STRIDE),
            "offset_norm": float(OFFSET_NORM),
            "geometry_note": "384 train/infer geometry, not the old 256 lane",
            "semantic_source": "recomputed from PNG; no saved semantic identity gate",
            "ground_truth_at_inference": False,
            "ink_support_policy": dict(raster_data.INK_SUPPORT_POLICY),
            "ink_support_pixels": int(result["stats"]["ink_support_pixels"]),
            "ink_support_source": result["stats"]["semantic_valid_source"],
            "outside_ink_policy": result["stats"]["outside_ink_policy"],
            "instance_policy": "vote-clustered offsets; Stuff merged by vectorizer",
            "timings": timings,
            "input_guard": {
                "max_pixels": MAX_INPUT_PIXELS,
                "max_file_bytes": MAX_INPUT_BYTES,
                "source": "png_frontend.FrontendConfig defaults",
                "scope": "input guard only; not a GPU memory guarantee",
            },
        },
    }


# ---------------------------------------------------------------------------
# CPU readiness: measured supervision counts over the authorized FIT cache
# ---------------------------------------------------------------------------


def plan_epochs(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    instance_crops: int = INSTANCE_CROPS_PER_RECORD,
    accumulation_steps: int = ACCUMULATION_STEPS,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Count real supervision for every scheduled crop, without any forward."""

    numpy = _require_numpy()
    epochs = _positive_int("epochs", int(epochs))
    accumulation_steps = _positive_int(
        "accumulation_steps", int(accumulation_steps)
    )
    selected = list(records)
    if max_records is not None:
        selected = selected[: _positive_int("max_records", int(max_records))]
    per_epoch: List[Dict[str, Any]] = []
    class_groups: Dict[str, int] = {}
    unique_groups: set = set()
    cache_classes: set = set()
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        source = JointEpochSource(
            selected,
            cache_dir=cache_dir,
            epoch=epoch,
            base_seed=base_seed,
            crop_size=crop_size,
            crops_per_record=crops_per_record,
            instance_crops=instance_crops,
        )
        scheduled = 0
        skipped = 0
        supervised = 0
        semantic_crops = 0
        offset_crops = 0
        both = 0
        semantic_pixels = 0
        offset_pixels = 0
        instance_slots = 0
        uniform_slots = 0
        truncated_pixels = 0
        records_with_crop = 0
        for record_index in range(len(selected)):
            item = source.get(record_index)
            if epoch == 1:
                semantic_full = numpy.asarray(item["entry"]["semantic"]).reshape(-1)
                valid_full = numpy.asarray(
                    item["entry"]["semantic_valid"], dtype=bool
                ).reshape(-1)
                known_full = valid_full & (semantic_full >= 0) & (
                    semantic_full < NUM_CLASSES
                )
                if bool(known_full.any()):
                    cache_classes.update(
                        int(value)
                        for value in numpy.unique(semantic_full[known_full]).tolist()
                    )
            if any(not crop["skip"] for crop in item["crops"]):
                records_with_crop += 1
            for crop in item["crops"]:
                scheduled += 1
                if crop["kind"] == "instance":
                    instance_slots += 1
                else:
                    uniform_slots += 1
                if crop["skip"]:
                    skipped += 1
                    continue
                supervised += 1
                known = int(crop["semantic_known"])
                pixels = int(crop["offset_pixels"])
                semantic_pixels += known
                offset_pixels += pixels
                if known > 0:
                    semantic_crops += 1
                if pixels > 0:
                    offset_crops += 1
                if known > 0 and pixels > 0:
                    both += 1
                truncated_pixels += int(
                    sum(
                        int(group["crop_pixels"])
                        for group in crop["offset_groups"]
                        if not group["centroid_inside_crop"]
                    )
                )
                sample_id = crop["sample_id"]
                for group in crop["offset_groups"]:
                    key = str(group["class"])
                    class_groups[key] = int(class_groups.get(key, 0)) + 1
                    unique_groups.add(
                        (sample_id, int(group["class"]), int(group["instance"]))
                    )
        windows = int(math.ceil(supervised / float(accumulation_steps)))
        per_epoch.append(
            {
                "epoch": int(epoch),
                "seed": epoch_seed(base_seed, epoch),
                "scheduled_crops": int(scheduled),
                "skipped_both_zero": int(skipped),
                "supervised_crops": int(supervised),
                "semantic_supervised_crops": int(semantic_crops),
                "offset_supervised_crops": int(offset_crops),
                "both_supervised_crops": int(both),
                "semantic_pixels": int(semantic_pixels),
                "offset_pixels": int(offset_pixels),
                "instance_slots": int(instance_slots),
                "uniform_slots": int(uniform_slots),
                "truncated_group_pixels": int(truncated_pixels),
                "windows": windows,
                "optimizer_steps": windows,
                "records_with_supervised_crop": int(records_with_crop),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "records": len(selected),
        "epochs": epochs,
        "crop_size": int(crop_size),
        "tile_size": int(TILE_SIZE),
        "tile_stride": int(TILE_STRIDE),
        "crops_per_record": int(crops_per_record),
        "instance_crops_per_record": int(instance_crops),
        "accumulation_steps": int(accumulation_steps),
        "per_epoch": per_epoch,
        "totals": {
            "scheduled_crops": int(sum(e["scheduled_crops"] for e in per_epoch)),
            "skipped_both_zero": int(sum(e["skipped_both_zero"] for e in per_epoch)),
            "supervised_crops": int(sum(e["supervised_crops"] for e in per_epoch)),
            "semantic_pixels": int(sum(e["semantic_pixels"] for e in per_epoch)),
            "offset_pixels": int(sum(e["offset_pixels"] for e in per_epoch)),
            "windows": int(sum(e["windows"] for e in per_epoch)),
            "optimizer_steps": int(sum(e["optimizer_steps"] for e in per_epoch)),
        },
        "unique_supervised_groups": len(unique_groups),
        "supervised_groups_by_class": {
            key: int(class_groups[key]) for key in sorted(class_groups, key=int)
        },
        "supervised_groups_by_class_note": (
            "keys are semantic classes observed in supervised crops only; absent "
            "classes are not a label remapping"
        ),
        "cache_semantic_classes": len(cache_classes),
        "cache_semantic_class_ids": sorted(int(value) for value in cache_classes),
        "seconds": round(time.monotonic() - started, 3),
    }


def _memory_estimate() -> Dict[str, Any]:
    """Estimate plus the actual streamed guard number once it has run."""

    parameters = 27_200_000  # DeepLabV3-ResNet50/35, approximate
    weight_bytes = parameters * 4 * 4  # param + grad + AdamW m + AdamW v
    weights_mib = round(weight_bytes / 1024**2)
    measured = None
    guard_path = os.path.join(DEFAULT_OUTPUT_DIR, "guard.json")
    if os.path.isfile(guard_path):
        try:
            with open(guard_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, Mapping) and payload.get("schema") == GUARD_SCHEMA:
                measured = {
                    key: payload.get(key)
                    for key in (
                        "status",
                        "device",
                        "cuda",
                        "free_before_mib",
                        "free_after_mib",
                        "peak_allocated_mib",
                        "peak_reserved_mib",
                        "headroom_mib",
                        "required_headroom_mib",
                        "streamed_crops",
                        "max_live_graphs",
                        "trunk_and_head_restored",
                        "optimizer_state_reused",
                        "semantic_pixels",
                        "offset_pixels",
                        "geometry",
                    )
                }
                measured["source"] = "guard.json"
        except (OSError, ValueError, TypeError):  # pragma: no cover - corrupt file
            measured = None
    return {
        "status": (
            "measured_actual_streamed_guard"
            if measured is not None
            else "estimate_only_not_measured"
        ),
        "measured_by": (
            "joint_cuda_accum_guard peak_allocated_mib (executed)"
            if measured is not None
            else "joint_cuda_accum_guard peak_allocated_mib (not executed)"
        ),
        "measured_guard": measured,
        "assumptions": {
            "parameters_approx": parameters,
            "bytes_per_parameter": 16,
            "activation_scope": "one streamed 384 px training graph at a time",
            "batch_size": 1,
            "amp": False,
        },
        "single_graph_estimate_mib": {
            "activations_low": 2500,
            "activations_high": 4500,
            "weights_gradients_optimizer": weights_mib,
            "total_low": weights_mib + 2500,
            "total_high": weights_mib + 4500,
            "note": (
                "rough order-of-magnitude only: one 384 px DeepLabV3-ResNet50 "
                "autograd graph (batch 1, fp32) with a trainable backbone, plus "
                "a 35-class 384 px logits tensor; not measured and not a reservation"
            ),
        },
        "cpu_estimate_mib": {
            "activations_low": 2500,
            "activations_high": 4500,
            "note": "same single-graph activations; RAM estimate only",
        },
        "streaming": (
            "denominators are computed from the eight target masks before any "
            "forward; each non-empty crop is forwarded, scaled, and backwarded "
            "immediately, so only one graph is retained; the optimizer steps once"
        ),
        "guard_acceptance": (
            "free_before_mib - peak_allocated_mib >= max(1024, 512); the peak is "
            "charged once over an actual streamed 8-crop window"
        ),
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
    """The disclosed, non-swept policy and contracts of this lane."""

    return {
        "schema": READINESS_SCHEMA,
        "version": VERSION,
        "kind": "joint_semantic_offsets_prepare",
        "role": "fit",
        "lane": "joint-semantic-offsets",
        "new_files_only": True,
        "shared_source_edits": [],
        "python": sys.executable,
        "manifest": os.path.abspath(manifest_path),
        "cache_dir": os.path.abspath(cache_dir) if cache_dir else None,
        "records": int(records),
        "hypothesis": (
            "the frozen-head pilot could not repair an erroneous semantic map; "
            "jointly fine-tuning semantics and offsets with instance-context "
            "crops is the materially different test"
        ),
        "difference_from_prior_lanes": {
            "semantic_model": "trainable (previous instance-offset lane froze it)",
            "offset_head": "fresh seed 20261108, trained jointly",
            "forward_passes": "one shared forward: semantic logits + classifier[3] tap",
            "crop_geometry": "384 px, instance-context and uniform crops",
            "loss_scope": (
                "semantic CE and offset SmoothL1 are summed separately and each "
                "is divided by its own window denominator"
            ),
        },
        "geometry": {
            "crop_size": int(CROP_SIZE),
            "tile_size": int(TILE_SIZE),
            "tile_stride": int(TILE_STRIDE),
            "offset_norm": float(OFFSET_NORM),
            "explicitly_not": "the old 256 px instance-offset geometry",
            "white_pad": True,
            "resize": None,
        },
        "crops": {
            "crops_per_record": int(CROPS_PER_RECORD),
            "instance_centred": int(INSTANCE_CROPS_PER_RECORD),
            "uniform": int(CROPS_PER_RECORD - INSTANCE_CROPS_PER_RECORD),
            "no_instance_fallback": "all eight crops uniform",
            "instance_selection": (
                "cycle eligible (class, instance) groups from a seeded rotation "
                "with a seeded +/-64 px jitter; width/height independently jittered"
            ),
            "uniform_selection": (
                "the sampler's own deterministic uniform anchors at slots 4..7"
            ),
            "determinism": "pure function of (epoch seed, record index, slot)",
        },
        "head": {
            "module": "Conv2d(256, 64, 1) -> ReLU -> Conv2d(64, 2, 1)",
            "tap": "net.classifier[3] (ReLU, stride 8, 256 channels)",
            "field_resolution": (
                "head runs at the classifier[3] stride (48x48 on a 384 crop), "
                "then the two-channel field is bilinear-upsampled to the crop size"
            ),
            "tap_detached": False,
            "tap_second_forward": False,
            "trainable_parameters": int(
                offsets.HEAD_IN_CHANNELS * offsets.HEAD_HIDDEN_CHANNELS
                + offsets.HEAD_HIDDEN_CHANNELS
                + offsets.HEAD_HIDDEN_CHANNELS * offsets.OFFSET_CHANNELS
                + offsets.OFFSET_CHANNELS
            ),
            "init_seed": int(HEAD_INIT_SEED),
            "init_order": (
                "head is constructed with an explicit torch.Generator before the "
                "trunk is built or any epoch seed is set"
            ),
            "trunk_trainable": True,
            "classifier_trainable": True,
            "batch_norm": (
                "running statistics frozen (eval after train()); affine weights "
                "and biases remain trainable"
            ),
        },
        "loss": {
            "semantic": "CE sum over known semantic_valid pixels / window known pixels",
            "offset": (
                "masked SmoothL1 sum over both field channels on instance pixels "
                "/ (2 * window instance pixels)"
            ),
            "weights": {"semantic": 1.0, "offset": 1.0},
            "beta": SMOOTHL1_BETA,
            "window": int(ACCUMULATION_STEPS),
            "denominators": (
                "independent per streamed eight-crop accumulation window; "
                "target-mask counts are summed before any forward"
            ),
            "branch_independence": (
                "a semantic-zero crop still forwards for its offset branch and "
                "vice versa; only a both-zero crop is skipped before the forward"
            ),
            "semantic_mask": "semantic_valid & 0 <= semantic < 35 (unknown -> -1)",
            "offset_mask": (
                "instance_valid & instance >= 0 & semantic_valid & 0 <= semantic < 30"
            ),
            "min_group_pixels": int(MIN_GROUP_PIXELS),
            "unknown_as_negative": False,
            "missing_labels": "never negatives; excluded from both targets",
            "target": (
                "per-pixel (full-native group centroid - native pixel) / 256, even "
                "for 384 crops; a truncated instance still aims at its true centre"
            ),
            "target_norm": float(OFFSET_NORM),
            "crop_local_denominator": False,
        },
        "training": {
            "epochs": int(epochs),
            "optimizer": "AdamW",
            "trunk_lr": TRUNK_LR,
            "trunk_weight_decay": TRUNK_WEIGHT_DECAY,
            "offset_lr": OFFSET_LR,
            "offset_weight_decay": OFFSET_WEIGHT_DECAY,
            "accumulation_steps": int(ACCUMULATION_STEPS),
            "accumulation": (
                "streamed: window denominators from target masks first, then one "
                "forward + immediate backward per non-empty crop, one optimizer "
                "step per window; no crop loss is divided again by 8"
            ),
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_init_seed": int(HEAD_INIT_SEED),
            "amp": False,
            "early_stopping": False,
            "final_epoch_only": True,
            "class_weights": None,
            "threshold_sweeps": False,
            "tune": False,
            "cuda_authorized": False,
            "cpu_training_run": False,
        },
        "inference": {
            "semantic_source": "recomputed from PNG via 384/192 tiled averaged logits",
            "offset_source": "same forward, head units, overlap averaged",
            "white_pad": True,
            "resize": None,
            "saved_semantic_identity_gate": False,
            "ground_truth_at_inference": False,
            "ink_support": {
                "helper": "recognition_raster_data._image_ink_support",
                "policy": (
                    "alpha-composited PNG RGB, min(R,G,B) < 254, unit-range "
                    "float comparison reused exactly"
                ),
                "default": (
                    "predict_and_decode derives support from the image; no "
                    "semantic_valid needed"
                ),
                "explicit_semantic_valid": (
                    "intersected with PNG ink, never replaces it; cannot add "
                    "background"
                ),
                "decoder_input": "semantic -1 outside ink; decoder known mask intersected with ink",
                "semantic_output": (
                    "35 (ignore) outside ink so the vectorizer accepts it and "
                    "emits no geometry"
                ),
                "instance_output": "-1 outside ink",
                "no_thing_pixels_deleted_within_ink": True,
                "source": "PNG pixels only; no GT, saved map, threshold sweep or new gate",
            },
            "unit_scale": (
                "decode_offsets receives head units and applies the single *256; "
                "callers pass units, never pixels"
            ),
            "decoder": {
                "class": "label-free per-class vote clustering (reused, unchanged)",
                **DECODER_POLICY,
                "fallback": "baseline CC8 for dissolved thin clusters",
                "pixels_deleted": False,
                "stuff": "merged per class by the unchanged vectorizer",
                "all_thing_pixels_assigned": True,
                "pre_registered": True,
                "thresholds_swept": False,
            },
            "vectorizer": "png_raster_vectorize.vectorize_masks, unchanged",
            "metadata": "train/infer 384 px explicit; never reported as 256",
            "cost": {
                "measured_here": False,
                "prior_offset24_build_seconds": 2312.0,
                "prior_dominant_stage": (
                    "CPU mean-shift clustering in the reused offset decoder"
                ),
                "recorded_stages": [
                    "tile_forward_seconds (384/192 shared forward: logits + units)",
                    "decode_seconds (reused vote clustering + mean-shift)",
                    "vectorize_seconds (unchanged vectorizer)",
                    "total_seconds",
                ],
                "note": (
                    "predict_and_decode/predict_png_joint record these stage "
                    "timings per call; no inference run happened during readiness"
                ),
            },
        },
        "data": {
            "records": "role=fit only, from the authorized manifest",
            "cache": "outputs/recognition-pretrained-fullfit-20261111-01/cache",
            "cache_policy": "fixed 204 FIT / 192 groups; no regeneration",
            "network": False,
            "hash": False,
            "protected_roles": [
                "test_final",
                "dev_final",
                "eval_pool",
                "reserve",
                ".runtime",
            ],
            "protected_roles_touched": False,
        },
        "memory": _memory_estimate(),
        "plan": dict(plan),
    }


_PLAN_MARKDOWN = """# Joint semantic + offset pilot (384 px)

## Why this is materially different
The frozen-head offset lane could only fit a residual on top of an already
chosen semantic map.  This lane jointly fine-tunes the semantic trunk and a
fresh offset head from the pure FIT-30 checkpoint in one shared forward pass, so
both losses can reshape the same features.

## Model and tap
| item | value |
| --- | --- |
| trunk | DeepLabV3-ResNet50, 35 classes, initialized from checkpoint_fullfit.pt |
| tap | `net.classifier[3]` (ReLU, stride 8, 256 channels), no detach |
| offset head | Conv1x1 256->64 -> ReLU -> Conv1x1 64->2, seed 20261108; applied at the stride-8 tap, field upsampled to the crop |
| BatchNorm | running statistics frozen; affine parameters trainable |

## Supervision
- semantic target: cache `semantic_valid` & known class; unknown/pad = -1;
  loss = CE sum / window known pixels.
- offset target: `(full-native centroid - native pixel) / 256`, even for 384
  crops; mask = `instance_valid & instance>=0 & semantic_valid & 0<=semantic<30`;
  loss = SmoothL1 sum / (2 * window instance pixels).
- A crop with one empty branch still forwards; both empty is skipped before the
  forward.  Missing labels are never negatives.

## Geometry
Train 384 px crops (4 instance-centred + 4 uniform, or 8 uniform when no known
Thing instance exists); inference 384 px tiles at 192 px stride, averaged logits
and offset units, white pad, no resize.  The offset scale is 256, not the crop size.

## Decoder and vectorizer
The settled instance-offset decoder is reused unchanged (8 px bins, 3 mean-shift
iterations at radius 16, 8 px merge, 16 px minimum cluster); `units * 256` happens
once inside the decoder.  The unchanged vectorizer merges Stuff per class and
consumes the assigned Thing ids.

## Status
CPU readiness and the streamed CUDA guard.  No CPU or CUDA training was run.
Training streams one 384 px graph at a time (immediate backward per crop, one
optimizer step per eight-crop window); the CUDA guard measured that actual
streamed window on cuda:0 with snapshot/restore (weights, grads, flags), one
graph live, a fresh throwaway optimizer, one peak charge plus 1 GiB reserve.
The measured peak is in `guard.json`; it is the only measured memory number and
is not a reservation.  The parent decides authorization after the prior route's
actual scores; no gain is guaranteed.
"""


def prepare(
    manifest_path: str = DEFAULT_MANIFEST,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    max_records: Optional[int] = None,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    write: bool = True,
) -> Dict[str, Any]:
    """CPU-only preparation: role-checked FIT records + measured supervision."""

    train.require_command_role("fit", "fit")
    records = fullfit.load_fit_records(manifest_path)
    plan = plan_epochs(
        records,
        cache_dir=cache_dir,
        epochs=epochs,
        base_seed=base_seed,
        crop_size=crop_size,
        crops_per_record=crops_per_record,
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
    policy["status"] = "cpu_readiness_ready_for_parent_authorization"
    policy["artifacts"] = {
        "prepare": "prepare.json",
        "policy": "policy.json",
        "readiness": "readiness.json",
        "plan": "PLAN.md",
    }
    if write:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "prepare.json"), policy)
        _write_json(
            os.path.join(output_dir, "policy.json"),
            {key: value for key, value in policy.items() if key != "plan"},
        )
        _write_json(os.path.join(output_dir, "readiness.json"), policy)
        with open(
            os.path.join(output_dir, "PLAN.md"), "w", encoding="utf-8"
        ) as handle:
            handle.write(_PLAN_MARKDOWN)
    return policy


# ---------------------------------------------------------------------------
# joint training (parent-scheduled; this module never runs it in tests)
# ---------------------------------------------------------------------------


def train_joint(
    embedder: JointEmbedder,
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    output_dir: str,
    semantic_checkpoint: str,
    semantic_payload: Mapping[str, Any],
    epochs: int = EPOCHS,
    accumulation_steps: int = ACCUMULATION_STEPS,
    base_seed: int = BASE_SEED,
    device: str = "cpu",
    role: str = "fit",
    lr_trunk: float = TRUNK_LR,
    weight_decay_trunk: float = TRUNK_WEIGHT_DECAY,
    lr_offset: float = OFFSET_LR,
    weight_decay_offset: float = OFFSET_WEIGHT_DECAY,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
    guard: bool = False,
    write: bool = True,
) -> Dict[str, Any]:
    """Bounded joint fit: final epoch only, no AMP, no early stopping, no tune."""

    torch = _require_torch()
    train.require_command_role("fit", role)
    if not records:
        raise JointOffsetsError("no fit records")
    epochs = _positive_int("epochs", int(epochs))
    accumulation_steps = _positive_int("accumulation_steps", int(accumulation_steps))
    lr_trunk = _finite("lr_trunk", lr_trunk, minimum=0.0)
    lr_offset = _finite("lr_offset", lr_offset, minimum=0.0)
    weight_decay_trunk = _finite("weight_decay_trunk", weight_decay_trunk, minimum=0.0)
    weight_decay_offset = _finite(
        "weight_decay_offset", weight_decay_offset, minimum=0.0
    )
    if lr_trunk <= 0.0 or lr_offset <= 0.0:
        raise JointOffsetsError("learning rates must be positive")
    initial_digest = model_digest(embedder.model)
    trunk_parameters, head_parameters = embedder.trainable_parameters()
    if not trunk_parameters or not head_parameters:
        raise JointOffsetsError("joint fit needs trainable trunk and head parameters")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": trunk_parameters,
                "lr": lr_trunk,
                "weight_decay": weight_decay_trunk,
                "group": "trunk",
            },
            {
                "params": head_parameters,
                "lr": lr_offset,
                "weight_decay": weight_decay_offset,
                "group": "offset_head",
            },
        ]
    )
    guard_report: Optional[Dict[str, Any]] = None
    if guard and not offsets.is_cuda_device(device):
        raise JointOffsetsError(
            f"guard requires a CUDA device; got {device!r}"
        )
    if guard and offsets.is_cuda_device(device):
        # Snapshot *before* the guard so the fit starts from the exact pre-guard
        # weights (the guard itself also restores them on every exit path).
        joint_state = joint_state_snapshot(embedder)
        guard_report = joint_cuda_accum_guard(
            embedder,
            records,
            cache_dir=cache_dir,
            device=device,
            accumulation_steps=accumulation_steps,
            base_seed=base_seed,
            epoch=1,
        )
        restored = restore_joint_state(embedder, joint_state)
        if not restored:  # pragma: no cover - bit-exact restore contract
            raise JointOffsetsError("guard_trunk_restore_failed")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": trunk_parameters,
                    "lr": lr_trunk,
                    "weight_decay": weight_decay_trunk,
                    "group": "trunk",
                },
                {
                    "params": head_parameters,
                    "lr": lr_offset,
                    "weight_decay": weight_decay_offset,
                    "group": "offset_head",
                },
            ]
        )
    epochs_report: List[Dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        seed = epoch_seed(base_seed, epoch)
        seed_everything(seed)
        frozen_bn = embedder.train_mode()
        source = JointEpochSource(
            records,
            cache_dir=cache_dir,
            epoch=epoch,
            base_seed=base_seed,
            crop_size=crop_size,
            crops_per_record=crops_per_record,
        )
        buffer: List[Dict[str, Any]] = []
        report = {
            "epoch": int(epoch),
            "seed": int(seed),
            "scheduled_crops": 0,
            "skipped_both_zero": 0,
            "supervised_crops": 0,
            "semantic_crops": 0,
            "offset_crops": 0,
            "both_crops": 0,
            "semantic_pixels": 0,
            "offset_pixels": 0,
            "windows": 0,
            "optimizer_steps": 0,
            "bn_modules_frozen": int(frozen_bn),
            "semantic_loss": None,
            "offset_loss": None,
            "loss": None,
            "seconds": 0.0,
        }
        semantic_sums = 0.0
        offset_sums = 0.0
        epoch_started = time.monotonic()

        def flush() -> None:
            nonlocal semantic_sums, offset_sums
            if not buffer:
                return
            result = run_window(embedder, list(buffer), optimizer, device=device)
            report["windows"] += 1
            report["optimizer_steps"] += 1
            report["semantic_pixels"] += int(result["semantic_pixels"])
            report["offset_pixels"] += int(result["offset_pixels"])
            semantic_sums += float(result["semantic_sum"])
            offset_sums += float(result["offset_sum"])
            buffer.clear()

        for record_index in range(len(records)):
            item = source.get(record_index)
            for crop in item["crops"]:
                report["scheduled_crops"] += 1
                if crop["skip"]:
                    report["skipped_both_zero"] += 1
                    continue
                report["supervised_crops"] += 1
                if int(crop["semantic_known"]) > 0:
                    report["semantic_crops"] += 1
                if int(crop["offset_pixels"]) > 0:
                    report["offset_crops"] += 1
                if int(crop["semantic_known"]) > 0 and int(crop["offset_pixels"]) > 0:
                    report["both_crops"] += 1
                buffer.append(crop)
                if len(buffer) >= accumulation_steps:
                    flush()
        flush()
        if report["semantic_pixels"]:
            report["semantic_loss"] = round(
                semantic_sums / float(report["semantic_pixels"]), 8
            )
        if report["offset_pixels"]:
            report["offset_loss"] = round(
                offset_sums / (2.0 * float(report["offset_pixels"])), 8
            )
        report["loss"] = round(
            float(report["semantic_loss"] or 0.0)
            + float(report["offset_loss"] or 0.0),
            8,
        )
        report["seconds"] = round(time.monotonic() - epoch_started, 3)
        epochs_report.append(report)

    checkpoint_path = os.path.join(output_dir, "checkpoint_joint.pt")
    metrics = {
        "schema": FIT_SCHEMA,
        "kind": "joint_semantic_offsets_fit",
        "role": role,
        "epochs": epochs,
        "epoch_reports": epochs_report,
        "accumulation_steps": int(accumulation_steps),
        "crop_size": int(crop_size),
        "tile_size": int(TILE_SIZE),
        "tile_stride": int(TILE_STRIDE),
        "offset_norm": float(OFFSET_NORM),
        "seconds": round(time.monotonic() - started, 3),
        "device": str(device),
        "amp": False,
        "early_stopping": False,
        "final_epoch_only": True,
        "guard": guard_report,
        "initial_semantic_digest": initial_digest,
        "semantic_digest_after": model_digest(embedder.model),
        "batch_norm": "running statistics frozen; affine parameters trainable",
        "losses": (
            "semantic CE sum/known + offset SmoothL1 sum/(2*instance); independent "
            "per-window denominators"
        ),
    }
    saved = None
    if write:
        saved = save_joint_checkpoint(
            checkpoint_path,
            embedder,
            epoch=epochs,
            epochs=epochs,
            semantic_checkpoint=semantic_checkpoint,
            semantic_payload=semantic_payload,
            initial_semantic_digest=initial_digest,
            metrics=metrics,
            training={
                "epochs": int(epochs),
                "trunk_lr": float(lr_trunk),
                "trunk_weight_decay": float(weight_decay_trunk),
                "offset_lr": float(lr_offset),
                "offset_weight_decay": float(weight_decay_offset),
                "accumulation_steps": int(accumulation_steps),
                "base_seed": int(base_seed),
                "crop_size": int(crop_size),
                "crops_per_record": int(crops_per_record),
            },
        )
    metrics["checkpoint"] = saved
    if write:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "fit.json"), metrics)
    return metrics


# ---------------------------------------------------------------------------
# CUDA guard (implemented, deliberately not executed)
# ---------------------------------------------------------------------------


def joint_state_snapshot(embedder: JointEmbedder) -> Dict[str, Any]:
    """CPU copy of trunk and head state, grads and per-module train flags."""

    def _state(module: Any) -> Dict[str, Any]:
        return {
            key: value.detach().to("cpu").clone()
            for key, value in module.state_dict().items()
        }

    def _grads(module: Any) -> Dict[str, Any]:
        return {
            name: (None if parameter.grad is None else parameter.grad.detach().to("cpu").clone())
            for name, parameter in module.named_parameters()
        }

    def _flags(module: Any) -> Dict[str, bool]:
        return {
            name: bool(child.training) for name, child in module.named_modules()
        }

    return {
        "model": _state(embedder.model),
        "head": _state(embedder.head),
        "grads": {"model": _grads(embedder.model), "head": _grads(embedder.head)},
        "training": {"model": _flags(embedder.model), "head": _flags(embedder.head)},
    }


def restore_joint_state(embedder: JointEmbedder, snapshot: Mapping[str, Any]) -> bool:
    """Restore trunk, head, grads and train flags; True only when bit-identical."""

    torch = _require_torch()
    embedder.model.load_state_dict(dict(snapshot["model"]), strict=True)
    embedder.head.load_state_dict(dict(snapshot["head"]), strict=True)
    for name, parameter in embedder.model.named_parameters():
        grad = snapshot["grads"]["model"].get(name)
        parameter.grad = None if grad is None else grad.detach().to(parameter.device).clone()
    for name, parameter in embedder.head.named_parameters():
        grad = snapshot["grads"]["head"].get(name)
        parameter.grad = None if grad is None else grad.detach().to(parameter.device).clone()
    for name, module in embedder.model.named_modules():
        module.training = bool(snapshot["training"]["model"].get(name, module.training))
    for name, module in embedder.head.named_modules():
        module.training = bool(snapshot["training"]["head"].get(name, module.training))
    current_model = embedder.model.state_dict()
    current_head = embedder.head.state_dict()
    grads_ok = all(
        (parameter.grad is None) == (snapshot["grads"]["model"][name] is None)
        and (
            parameter.grad is None
            or torch.equal(
                parameter.grad.detach().to("cpu"), snapshot["grads"]["model"][name]
            )
        )
        for name, parameter in embedder.model.named_parameters()
    ) and all(
        (parameter.grad is None) == (snapshot["grads"]["head"][name] is None)
        and (
            parameter.grad is None
            or torch.equal(
                parameter.grad.detach().to("cpu"), snapshot["grads"]["head"][name]
            )
        )
        for name, parameter in embedder.head.named_parameters()
    )
    flags_ok = all(
        bool(module.training) == bool(snapshot["training"]["model"][name])
        for name, module in embedder.model.named_modules()
    ) and all(
        bool(module.training) == bool(snapshot["training"]["head"][name])
        for name, module in embedder.head.named_modules()
    )
    return (
        all(
            torch.equal(current_model[key].detach().to("cpu"), snapshot["model"][key])
            for key in snapshot["model"]
        )
        and all(
            torch.equal(current_head[key].detach().to("cpu"), snapshot["head"][key])
            for key in snapshot["head"]
        )
        and grads_ok
        and flags_ok
    )


def joint_cuda_accum_guard(
    embedder: JointEmbedder,
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    device: str = "cuda",
    accumulation_steps: int = ACCUMULATION_STEPS,
    base_seed: int = BASE_SEED,
    epoch: int = 1,
    min_free_mib: float = MIN_FREE_MIB,
    reserve_after_peak_mib: float = RESERVE_AFTER_PEAK_MIB,
    crop_size: int = CROP_SIZE,
    crops_per_record: int = CROPS_PER_RECORD,
) -> Dict[str, Any]:
    """Measure one streamed 8-crop window, restoring trunk + head bit-exactly.

    The accumulation window's own denominators are used, exactly as in
    :func:`run_window`.  Only one 384 px autograd graph is alive at a time:
    each crop is forwarded, scaled, and backwarded before the next forward.
    Nothing is written to disk and the optimizer is a throwaway, so no external
    optimizer state is mutated.  Memory acceptance reuses
    :func:`offsets.guard_verdict`: the peak is charged once and 1 GiB must
    remain free after it.
    """

    torch = _require_torch()
    numpy = _require_numpy()
    target = offsets._torch_device(device)
    if target.type != "cuda":
        raise JointOffsetsError(
            f"joint_cuda_accum_guard requires a CUDA device; got {device!r}"
        )
    if not torch.cuda.is_available():
        raise JointOffsetsError("joint_cuda_accum_guard requires CUDA")
    accumulation_steps = _positive_int("accumulation_steps", int(accumulation_steps))
    index = int(target.index) if target.index is not None else int(
        torch.cuda.current_device()
    )
    free_before, total = torch.cuda.mem_get_info(index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)
    source = JointEpochSource(
        records,
        cache_dir=cache_dir,
        epoch=int(epoch),
        base_seed=base_seed,
        crop_size=crop_size,
        crops_per_record=crops_per_record,
    )
    snapshot = joint_state_snapshot(embedder)
    embedder.model.to(target)
    embedder.head.to(target)
    embedder.train_mode()
    trunk_parameters, head_parameters = embedder.trainable_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_parameters, "lr": TRUNK_LR, "weight_decay": TRUNK_WEIGHT_DECAY},
            {"params": head_parameters, "lr": OFFSET_LR, "weight_decay": OFFSET_WEIGHT_DECAY},
        ]
    )
    pending: List[Dict[str, Any]] = []
    try:
        for record_index in range(len(records)):
            item = source.get(record_index)
            for crop in item["crops"]:
                if crop["skip"]:
                    continue
                pending.append(crop)
                if len(pending) >= accumulation_steps:
                    break
            if len(pending) >= accumulation_steps:
                break
        if len(pending) < accumulation_steps:
            raise JointOffsetsError(
                f"guard found only {len(pending)} supervised crops, "
                f"need {accumulation_steps}"
            )
        semantic_denominator = int(
            sum(int(c["semantic_known"]) for c in pending)
        )
        offset_denominator = int(sum(int(c["offset_pixels"]) for c in pending))
        for crop in pending:
            image = torch.from_numpy(
                numpy.ascontiguousarray(crop["image"], dtype=numpy.float32)
            ).unsqueeze(0).to(target)
            output = embedder(image)
            loss = None
            if crop["semantic_known"] > 0 and semantic_denominator > 0:
                target_tensor = torch.from_numpy(
                    numpy.ascontiguousarray(crop["semantic_target"], dtype=numpy.int64)
                ).unsqueeze(0).to(target)
                term, _counted = semantic_sum_loss(
                    output["semantic_logits"], target_tensor
                )
                loss = term / float(semantic_denominator)
            if crop["offset_pixels"] > 0 and offset_denominator > 0:
                term, _counted = offset_sum_loss(
                    output["offset_units"], crop["offset_targets"]
                )
                offset_term = term / (2.0 * float(offset_denominator))
                loss = offset_term if loss is None else loss + offset_term
            loss.backward()
            del loss, output, image
        _assert_finite_gradients(embedder)
        optimizer.step()
        peak_allocated = torch.cuda.max_memory_allocated(index)
        peak_reserved = torch.cuda.max_memory_reserved(index)
        free_after, _ = torch.cuda.mem_get_info(index)
        optimizer.zero_grad(set_to_none=True)
        restored = restore_joint_state(embedder, snapshot)
        torch.cuda.empty_cache()

        verdict = offsets.guard_verdict(
            free_before / (1024.0**2),
            peak_allocated / (1024.0**2),
            reserve_after_peak_mib=reserve_after_peak_mib,
            min_free_mib=min_free_mib,
        )
        return {
            "schema": GUARD_SCHEMA,
            "device": torch.cuda.get_device_name(index),
            "cuda": True,
            "device_index": index,
            "trunk_and_head_restored": bool(restored),
            "gradients_cleared": True,
            "optimizer_state_reused": False,
            "optimizer_state_note": (
                "the guard uses a throwaway AdamW; no external optimizer state "
                "exists to restore, and its gradients are cleared"
            ),
            "total_mib": round(total / (1024.0**2), 1),
            "free_before_mib": round(free_before / (1024.0**2), 1),
            "free_after_mib": round(free_after / (1024.0**2), 1),
            "free_after_note": (
                "measured before the final empty_cache; includes cached/reserved "
                "blocks, so the service headroom is the acceptance number"
            ),
            "streamed_crops": len(pending),
            "max_live_graphs": 1,
            "semantic_pixels": semantic_denominator,
            "offset_pixels": offset_denominator,
            "peak_allocated_mib": round(peak_allocated / (1024.0**2), 1),
            "peak_reserved_mib": round(peak_reserved / (1024.0**2), 1),
            "headroom_mib": verdict["remaining_after_peak_mib"],
            "required_headroom_mib": verdict["required_remaining_mib"],
            "reserve_after_peak_mib": float(reserve_after_peak_mib),
            "min_free_mib": float(min_free_mib),
            "acceptance": (
                "free_before_mib - peak_allocated_mib >= max(1024, 512); "
                "the peak is charged once"
            ),
            "status": verdict["status"],
            "geometry": {"crop_size": int(crop_size), "tile_size": int(TILE_SIZE)},
            "note": (
                "actual streamed 8-crop accumulation window measured: one graph "
                "at a time, never eight retained; trunk and head restored "
                "bit-exactly; no training run follows automatically"
            ),
        }
    except torch.cuda.OutOfMemoryError as error:
        peak_allocated = torch.cuda.max_memory_allocated(index)
        peak_reserved = torch.cuda.max_memory_reserved(index)
        free_after, _ = torch.cuda.mem_get_info(index)
        return {
            "schema": GUARD_SCHEMA,
            "device": torch.cuda.get_device_name(index),
            "cuda": True,
            "device_index": index,
            "status": "oom_guard_stopped",
            "error": str(error).strip()[:1000],
            "total_mib": round(total / (1024.0**2), 1),
            "free_before_mib": round(free_before / (1024.0**2), 1),
            "free_after_mib": round(free_after / (1024.0**2), 1),
            "free_after_note": (
                "measured before the final empty_cache; includes cached/reserved "
                "blocks, so the service headroom is the acceptance number"
            ),
            "peak_allocated_mib": round(peak_allocated / (1024.0**2), 1),
            "peak_reserved_mib": round(peak_reserved / (1024.0**2), 1),
            "streamed_crops": len(pending),
            "max_live_graphs": 1,
            "note": (
                "CUDA OOM inside the streamed 8-crop window; the guard stopped, "
                "changed no crop/accumulation recipe, and restored weights, "
                "gradients and train flags"
            ),
        }
    finally:
        # Purity even on error: weights, grads and train flags go back as they were.
        optimizer.zero_grad(set_to_none=True)
        restore_joint_state(embedder, snapshot)
        del pending
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# CLI (new module only; existing lane CLIs are untouched)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint semantic + offset pilot (384 px, CPU readiness lane)"
    )
    subparsers = parser.add_subparsers(dest="command")
    prepare_parser = subparsers.add_parser("prepare", help="CPU supervision plan")
    prepare_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    prepare_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--epochs", type=int, default=EPOCHS)
    prepare_parser.add_argument("--max-records", type=int, default=None)
    prepare_parser.add_argument("--no-write", action="store_true")

    fit_parser = subparsers.add_parser("fit", help="joint fit (not run here)")
    fit_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    fit_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    fit_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    fit_parser.add_argument("--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT)
    fit_parser.add_argument("--epochs", type=int, default=EPOCHS)
    fit_parser.add_argument("--device", default="cpu")
    fit_parser.add_argument("--guard", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="one PNG, joint decoder")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--png", required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "prepare"
    if command == "prepare":
        policy = prepare(
            args.manifest,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_records=args.max_records,
            write=not args.no_write,
        )
        print(json.dumps(policy["plan"]["totals"], indent=2, sort_keys=True))
        return 0
    if command == "fit":
        train.require_command_role("fit", "fit")
        if args.guard and not offsets.is_cuda_device(args.device):
            parser.error(
                f"--guard requires a CUDA device; got {args.device!r} "
                "(it measures a real CUDA accumulation window)"
            )
        records = fullfit.load_fit_records(args.manifest)
        embedder, report = build_joint_embedder(
            args.semantic_checkpoint, device=args.device
        )
        semantic_payload = {
            "schema": report["semantic_schema"],
            "role": report["semantic_role"],
            "epoch": report["semantic_epoch"],
            "epochs": report["semantic_epochs"],
        }
        metrics = train_joint(
            embedder,
            records,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            semantic_checkpoint=args.semantic_checkpoint,
            semantic_payload=semantic_payload,
            epochs=args.epochs,
            device=args.device,
            guard=bool(args.guard),
        )
        metrics["build_report"] = report
        _write_json(os.path.join(args.output_dir, "fit.json"), metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if command == "predict":
        embedder, _payload = load_joint_embedder(args.checkpoint, device=args.device)
        result = predict_png_joint(embedder, args.png, device=args.device)
        if args.output:
            _write_json(args.output, result["document"])
        print(json.dumps(result["stats"], indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command {command!r}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
