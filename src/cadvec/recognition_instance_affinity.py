"""Frozen-semantic instance embedding + deterministic agglomeration pilot.

This is an *additive* pilot lane.  Nothing in it changes the semantic model,
the semantic loss, the raster vectorizer or any existing decoder:

* the semantic checkpoint is loaded read-only, every parameter is frozen and
  the model is always run in ``eval`` mode (batch norm uses running stats);
* a single ``Conv1x1(256, 8)`` head sits on the ``net.classifier[3]`` ReLU
  output (stride 8) and is the only trainable tensor;
* supervision is the existing :func:`cadvec.recognition_raster_model.
  instance_embedding_loss` applied to *bilinearly upsampled* embeddings at the
  native crop resolution, so thin labels are never down-sampled to stride 8.
  Known different annotated instances (including different classes) repel;
  unknown pixels and Stuff never become negatives;
* decoding is label-free agglomeration of the existing CC8 baseline: fragments
  of the *same class* whose raw (non-normalised) embedding means are closer
  than ``0.75`` in L2 are merged with union-find.  There is no ground truth in
  the decoder, no greedy single-link chain is applied silently (cluster/chain
  sizes are reported), fragments beyond ``4096`` per class stay untouched, and
  Stuff / semantic pixel classes stay bit-identical;
* the only checkpoint produced here carries the head alone plus an explicit
  reference to the frozen semantic checkpoint.  It is never promoted.

The cache contract this lane consumes (the reviewed full-fit cache) provides
``image``, ``semantic`` (class indices ``0..34``, ``-1`` unknown),
``semantic_valid``, ``instance`` (raw local ids, ``-1`` unknown) and
``instance_valid``.  It has **no** independent ``instance_class`` array, so the
loss receives the semantic class indices as ``class_ids`` together with
``semantic_valid`` as ``class_valid``.  The two validity masks stay independent
inside the loss; only the class source is the semantic map, and that is
documented rather than papered over.

CLI::

    python -m cadvec.recognition_instance_affinity prepare --help
    python -m cadvec.recognition_instance_affinity smoke   --help
    python -m cadvec.recognition_instance_affinity fit     --help

``prepare`` and ``smoke`` are CPU-only.  ``fit`` refuses to run unless the
parent schedules it explicitly (``--role fit``), exactly like the semantic
lane.
"""

from __future__ import annotations

import argparse
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

from . import recognition_pretrained_fullfit as fullfit
from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_infer as raster_infer
from . import recognition_raster_model as raster_model
from . import recognition_raster_train as train

AFFINITY_SCHEMA = "cadvec.recognition.instance_affinity/1"
PREPARE_SCHEMA = "cadvec.recognition.instance_affinity_prepare/1"
FIT_SCHEMA = "cadvec.recognition.instance_affinity_fit/1"
SMOKE_SCHEMA = "cadvec.recognition.instance_affinity_smoke/1"
CHECKPOINT_SCHEMA = "cadvec.recognition.instance_affinity_checkpoint/1"
AFFINITY_VERSION = 1

# Feature geometry (fixed by the frozen semantic model).
EMBED_DIM = 8
HEAD_IN_CHANNELS = 256
FEATURE_STRIDE = 8
NUM_CLASSES = pilot.NUM_CLASSES
NUM_THING_CLASSES = raster_model.NUM_THING_CLASSES
IGNORE_CLASS = pilot.NUM_CLASSES  # raw semantic 35 / unknown in class-index space

# Tiling / crops: mirrored from the frozen full-fit lane.
TILE_SIZE = fullfit.TILE_SIZE
TILE_STRIDE = fullfit.TILE_STRIDE
CROP_SIZE = fullfit.CROP_SIZE
CROPS_PER_RECORD = fullfit.CROPS_PER_RECORD
EPOCHS = fullfit.EPOCHS
ACCUMULATION_STEPS = fullfit.ACCUMULATION_STEPS
BASE_SEED = fullfit.BASE_SEED
EPOCH_SEED_STRIDE = fullfit.EPOCH_SEED_STRIDE
HEAD_INIT_SEED = 20261108

# Loss caps (deliberately tighter than the shared default so gradients stay
# bounded; the shared function itself is untouched).
MIN_GROUP_PIXELS = 8
MIN_GROUPS_PER_CROP = 2
MAX_PIXELS_PER_INSTANCE = 512
MAX_INSTANCES = 64
MAX_PAIRS = 1024
DELTA_V = 0.5
DELTA_D = 1.5
VAR_WEIGHT = 1.0
DIST_WEIGHT = 1.0
LR = 1e-3
WEIGHT_DECAY = 0.0

# Decoder policy.
MERGE_DISTANCE = 0.75
MAX_FRAGMENTS_PER_CLASS = 4096
PAIR_CHUNK = 256
MIN_FRAGMENT_PIXELS = 1

# Prepare guard: if any epoch has fewer supervised crops than this, stop cheap.
MIN_ELIGIBLE_CROPS_PER_EPOCH = 200

DEFAULT_MANIFEST = os.path.join(
    "outputs", "recognition-gate3b-fit-tune-20261108-01", "recognition_public_manifest.json"
)
DEFAULT_CACHE_DIR = os.path.join(
    "outputs", "recognition-pretrained-fullfit-20261111-01", "cache"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    "outputs", "recognition-instance-affinity-20261111-01"
)
DEFAULT_SEMANTIC_CHECKPOINT = os.path.join(
    "outputs", "recognition-fullfit-e30-20261111-01", "checkpoint_fullfit.pt"
)


class InstanceAffinityError(pilot.PilotError):
    """Invalid configuration, cache record or decoder input."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:
        raise InstanceAffinityError("numpy is required for this lane")
    return np


def _require_torch() -> Any:
    if torch is None:
        raise InstanceAffinityError("torch is required for this lane")
    return torch


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InstanceAffinityError(f"{name} must be an int >= {minimum}; got {value!r}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise InstanceAffinityError(f"{name} must be finite; got {value!r}")
    return number


# ---------------------------------------------------------------------------
# frozen semantic model, feature tap and embedding head
# ---------------------------------------------------------------------------


def semantic_parameter_digest(model: Any) -> Tuple[Tuple[str, Tuple[int, ...], float, float], ...]:
    """Checksum summary of every state-dict entry (parameters *and* buffers).

    Uses ``float64`` sum / sum-of-abs per tensor.  This is a practical mutation
    alarm, **not** a cryptographic bit-proof: isolation rests on
    ``requires_grad=False``, forced ``eval()`` and the bit-identical semantic
    output tests; the digest only adds an early, cheap tripwire.
    """

    torch = _require_torch()
    digest: List[Tuple[str, Tuple[int, ...], float, float]] = []
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().to("cpu", dtype=torch.float64)
        digest.append(
            (
                str(name),
                tuple(int(v) for v in values.shape),
                float(values.sum().item()),
                float(values.abs().sum().item()),
            )
        )
    return tuple(digest)


def freeze_semantic_model(model: Any) -> Dict[str, Any]:
    """Freeze every semantic parameter, force eval and freeze batch norm."""

    _require_torch()
    total = 0
    frozen = 0
    for parameter in model.parameters():
        total += 1
        parameter.requires_grad_(False)
        frozen += 0 if parameter.requires_grad else 1
    model.eval()
    batch_norm = pilot.freeze_batch_norm(model)
    model.eval()
    return {
        "parameters_total": int(total),
        "parameters_frozen": int(frozen),
        "batch_norm_modules_eval": int(batch_norm),
        "training_mode": bool(getattr(model, "training", False)),
    }


def _classifier_relu(model: Any) -> Any:
    net = getattr(model, "net", None)
    classifier = getattr(net, "classifier", None)
    if classifier is None or len(classifier) < 4:
        raise InstanceAffinityError(
            "semantic wrapper has no net.classifier[3] ReLU to tap"
        )
    return classifier[3]


def build_instance_head(
    *,
    dim: int = EMBED_DIM,
    in_channels: int = HEAD_IN_CHANNELS,
    seed: int = HEAD_INIT_SEED,
    device: str = "cpu",
) -> Any:
    """Deterministic ``Conv1x1(256, 8)`` head (no global RNG dependence)."""

    torch = _require_torch()
    dim = _positive_int("dim", dim)
    in_channels = _positive_int("in_channels", in_channels)
    head = torch.nn.Conv2d(int(in_channels), int(dim), kernel_size=1)
    bound = 1.0 / math.sqrt(float(in_channels))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=generator)
        if head.bias is not None:
            head.bias.uniform_(-bound, bound, generator=generator)
    return head.to(device)


class PilotEmbedder:
    """Read-only semantic wrapper plus a trainable embedding head.

    ``feature_map`` runs the semantic wrapper (ImageNet normalisation included)
    under ``no_grad`` and captures ``net.classifier[3]`` through a forward hook.
    ``crop_embeddings`` applies the head *outside* ``no_grad`` and upsamples the
    result bilinearly to the crop size with ``align_corners=False``, exactly the
    interpolation the semantic wrapper uses.
    """

    def __init__(
        self,
        model: Any,
        *,
        dim: int = EMBED_DIM,
        in_channels: int = HEAD_IN_CHANNELS,
        head: Any = None,
        device: str = "cpu",
    ) -> None:
        torch = _require_torch()
        self.model = model
        self.device = str(device)
        self.dim = int(dim)
        self.in_channels = int(in_channels)
        self.frozen_report = freeze_semantic_model(model)
        if head is None:
            head = build_instance_head(
                dim=self.dim, in_channels=self.in_channels, device=self.device
            )
        self.head = head.to(self.device)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self._tap: Dict[str, Any] = {}
        self._handle = _classifier_relu(model).register_forward_hook(self._hook)

    # -- hook plumbing ---------------------------------------------------
    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
        self._tap["features"] = output

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.remove()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive only
        try:
            self.close()
        except Exception:
            pass

    # -- forwards --------------------------------------------------------
    def feature_map(self, images: Any) -> Dict[str, Any]:
        """Semantic logits + stride-8 ReLU features for NCHW images in [0,1]."""

        torch = _require_torch()
        if not torch.is_tensor(images):
            raise InstanceAffinityError("images must be a torch tensor")
        self.model.eval()
        self._tap.pop("features", None)
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            features = self._tap.pop("features", None)
        if features is None:
            raise InstanceAffinityError("classifier[3] feature tap never fired")
        return {"features": features, "semantic_logits": outputs["semantic_logits"]}

    def crop_embeddings(self, images: Any) -> Any:
        """B x dim x H x W embeddings at the input resolution (grad-carrying)."""

        torch = _require_torch()
        if not torch.is_tensor(images):
            raise InstanceAffinityError("images must be a torch tensor")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        features = self.feature_map(images)["features"]
        logits = self.head(features.detach().to(self.device))
        if tuple(logits.shape[-2:]) != (height, width):
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        return logits

    def native_embeddings(
        self,
        image: Any,
        *,
        tile_size: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
        return_semantic_logits: bool = False,
    ) -> Any:
        """Tiled embeddings at source resolution with overlap averaging.

        Mirrors :func:`cadvec.recognition_pretrained_fullfit.predict_tiled_image`
        (same tile positions, white padding at bottom/right, same averaging),
        but accumulates the upsampled head output instead of class logits.
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
            raise InstanceAffinityError(f"image must be 3 x H x W; got {tuple(source.shape)}")
        if not bool(torch.isfinite(source).all()):
            raise InstanceAffinityError("non_finite_image")
        height, width = int(source.shape[-2]), int(source.shape[-1])
        if height <= 0 or width <= 0:
            raise InstanceAffinityError("empty image")
        tile_size = _positive_int("tile_size", tile_size)
        stride = _positive_int("stride", stride)
        if stride > tile_size:
            raise InstanceAffinityError("stride must not exceed tile size")
        padded_height = max(int(tile_size), height)
        padded_width = max(int(tile_size), width)
        padded = source
        if padded_height != height or padded_width != width:
            padded = torch.nn.functional.pad(
                source.unsqueeze(0),
                (0, padded_width - width, 0, padded_height - height),
                mode="constant",
                value=1.0,
            ).squeeze(0)
        rows = _tile_positions(padded_height, tile_size, stride)
        columns = _tile_positions(padded_width, tile_size, stride)

        self.model.eval()
        # Accumulate on the CPU: the head output may live on CUDA, and an
        # in-place add of a CUDA tile into a CPU buffer raises.  Detaching each
        # upsampled tile to CPU float32 also bounds memory (8 x H x W floats).
        emb_accum: Optional[Any] = None
        logit_accum: Optional[Any] = None
        weights: Optional[Any] = None
        try:
            with torch.no_grad():
                for row in rows:
                    for column in columns:
                        window = padded[
                            :, row : row + tile_size, column : column + tile_size
                        ]
                        features = self.feature_map(window.unsqueeze(0).to(self.device))
                        tile_embed = F.interpolate(
                            self.head(features["features"]),
                            size=(tile_size, tile_size),
                            mode="bilinear",
                            align_corners=False,
                        ).detach().to("cpu", dtype=torch.float32)
                        tile_logits = features["semantic_logits"].detach().to(
                            "cpu", dtype=torch.float32
                        )
                        if emb_accum is None:
                            emb_accum = torch.zeros(
                                (self.dim, padded_height, padded_width),
                                dtype=torch.float32,
                            )
                            weights = torch.zeros(
                                (1, padded_height, padded_width),
                                dtype=torch.float32,
                            )
                            if return_semantic_logits:
                                logit_accum = torch.zeros(
                                    (NUM_CLASSES, padded_height, padded_width),
                                    dtype=torch.float32,
                                )
                        emb_accum[
                            :, row : row + tile_size, column : column + tile_size
                        ] += tile_embed[0]
                        weights[
                            :, row : row + tile_size, column : column + tile_size
                        ] += 1.0
                        if return_semantic_logits and logit_accum is not None:
                            logit_accum[
                                :, row : row + tile_size, column : column + tile_size
                            ] += tile_logits[0]
                assert emb_accum is not None and weights is not None
                emb_accum = emb_accum / weights
        finally:
            self.model.eval()
        embeddings = emb_accum[:, :height, :width].contiguous().numpy().astype(numpy.float32)
        if not return_semantic_logits:
            return embeddings
        assert logit_accum is not None
        logits = (logit_accum / weights)[:, :height, :width].contiguous()
        return {
            "embeddings": embeddings,
            "semantic_logits": logits.numpy().astype(numpy.float32),
            "tile_rows": len(rows),
            "tile_columns": len(columns),
        }


def _tile_positions(length: int, tile: int, stride: int) -> List[int]:
    """Identical to ``fullfit._tile_positions`` (tested against it)."""

    if length <= 0 or tile <= 0 or stride <= 0:
        raise InstanceAffinityError("tile geometry must be positive")
    if stride > tile:
        raise InstanceAffinityError("stride must not exceed tile size")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


# ---------------------------------------------------------------------------
# supervision planning (independent masks preserved)
# ---------------------------------------------------------------------------


def eligible_groups(
    instance_targets: Any,
    instance_valid: Any,
    *,
    class_ids: Any,
    class_valid: Any,
    min_group_pixels: int = MIN_GROUP_PIXELS,
    num_thing_classes: int = NUM_THING_CLASSES,
) -> Dict[str, Any]:
    """Group eligible (class, instance) pairs; independent masks preserved.

    A pixel participates only when ``instance_valid`` says so **and** the
    instance id is non-negative (``0`` is legal; only ``-1`` means no image)
    **and** the class source marks a known Thing class.  ``instance_valid`` and
    ``class_valid`` are never derived from each other here; in this cache the
    class source happens to be ``semantic``/``semantic_valid``, which is why an
    instance-only pixel without a class cannot be supervised.
    """

    numpy = _require_numpy()
    targets = numpy.asarray(instance_targets)
    valid = numpy.asarray(instance_valid, dtype=bool)
    classes = numpy.asarray(class_ids)
    classes_valid = numpy.asarray(class_valid, dtype=bool)
    if targets.shape != valid.shape or targets.shape != classes.shape:
        raise InstanceAffinityError("instance/class arrays must share a shape")
    if classes_valid.shape != targets.shape:
        raise InstanceAffinityError("class_valid shape does not match")
    mask = (
        valid
        & (targets >= 0)
        & classes_valid
        & (classes >= 0)
        & (classes < int(num_thing_classes))
    )
    pixels = int(mask.sum())
    if pixels == 0:
        return {
            "mask_pixels": 0,
            "groups": [],
            "eligible_groups": [],
            "eligible_group_count": 0,
            "eligible_pixels": 0,
            "skipped_group_count": 0,
            "by_class": {},
        }
    selected = numpy.stack(
        [classes[mask].astype(numpy.int64), targets[mask].astype(numpy.int64)], axis=1
    )
    unique, counts = numpy.unique(selected, axis=0, return_counts=True)
    pairs = [
        {
            "class": int(pair[0]),
            "instance": int(pair[1]),
            "pixels": int(count),
        }
        for pair, count in zip(unique, counts)
    ]
    pairs.sort(key=lambda item: (item["class"], item["instance"]))
    eligible = [item for item in pairs if item["pixels"] >= int(min_group_pixels)]
    by_class: Dict[str, int] = {}
    for item in eligible:
        key = str(item["class"])
        by_class[key] = int(by_class.get(key, 0)) + 1
    return {
        "mask_pixels": pixels,
        "groups": pairs,
        "eligible_groups": eligible,
        "eligible_group_count": len(eligible),
        "eligible_pixels": int(sum(item["pixels"] for item in eligible)),
        "skipped_group_count": len(pairs) - len(eligible),
        "by_class": by_class,
    }


def crop_is_supervised(
    crop: Mapping[str, Any],
    *,
    min_group_pixels: int = MIN_GROUP_PIXELS,
    min_groups: int = MIN_GROUPS_PER_CROP,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (usable, group report) without touching the model."""

    report = eligible_groups(
        crop["instance"],
        crop["instance_valid"],
        class_ids=crop["semantic"],
        class_valid=crop["semantic_valid"],
        min_group_pixels=min_group_pixels,
    )
    return report["eligible_group_count"] >= int(min_groups), report


# ---------------------------------------------------------------------------
# CPU preparation: actual counts, no guessing
# ---------------------------------------------------------------------------


def plan_epochs(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_group_pixels: int = MIN_GROUP_PIXELS,
    min_groups: int = MIN_GROUPS_PER_CROP,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
    accumulation_steps: int = ACCUMULATION_STEPS,
) -> Dict[str, Any]:
    """Walk every scheduled crop of every epoch and count real supervision."""

    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    per_epoch: List[Dict[str, Any]] = []
    class_groups: Dict[str, int] = {}
    unique_groups: set = set()
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        source, _sampler = fullfit.make_epoch_source(
            records,
            epoch=epoch,
            cache_dir=cache_dir,
            base_seed=base_seed,
        )
        eligible = 0
        groups = 0
        pixels = 0
        samples: List[str] = []
        for index in range(len(source)):
            crop = source.get(index)
            usable, report = crop_is_supervised(
                crop, min_group_pixels=min_group_pixels, min_groups=min_groups
            )
            if not usable:
                continue
            eligible += 1
            groups += report["eligible_group_count"]
            pixels += report["eligible_pixels"]
            sample_id = str(crop.get("sample_id", ""))
            if sample_id not in samples:
                samples.append(sample_id)
            for item in report["eligible_groups"]:
                key = str(item["class"])
                class_groups[key] = int(class_groups.get(key, 0)) + 1
                unique_groups.add((sample_id, item["class"], item["instance"]))
        per_epoch.append(
            {
                "epoch": epoch,
                "seed": fullfit.epoch_seed(base_seed, epoch),
                "scheduled_crops": int(len(source)),
                "eligible_crops": int(eligible),
                "skipped_crops": int(len(source) - eligible),
                "eligible_groups": int(groups),
                "eligible_pixels": int(pixels),
                "records_with_eligible_crop": len(samples),
                "optimizer_steps": int(
                    math.ceil(eligible / float(accumulation_steps))
                ),
                "below_min_eligible": bool(eligible < int(min_eligible_crops)),
            }
        )
    stop_cheap = any(item["below_min_eligible"] for item in per_epoch)
    return {
        "records": len(records),
        "epochs": epochs,
        "per_epoch": per_epoch,
        "stop_cheap": bool(stop_cheap),
        "min_eligible_crops_per_epoch": int(min_eligible_crops),
        "unique_supervised_groups": len(unique_groups),
        "eligible_groups_by_class": {
            key: int(class_groups[key]) for key in sorted(class_groups, key=int)
        },
        "seconds": round(time.monotonic() - started, 3),
    }


def prepare(
    manifest_path: str,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
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
    )
    policy = {
        "schema": PREPARE_SCHEMA,
        "version": AFFINITY_VERSION,
        "kind": "frozen_semantic_instance_affinity_prepare",
        "role": "fit",
        "manifest": os.path.abspath(manifest_path),
        "cache_dir": os.path.abspath(cache_dir) if cache_dir else None,
        "records": int(len(records)),
        "crops_per_record": int(CROPS_PER_RECORD),
        "crop_size": int(CROP_SIZE),
        "tile_size": int(TILE_SIZE),
        "tile_stride": int(TILE_STRIDE),
        "semantic_frozen": True,
        "semantic_eval_always": True,
        "embedding_head": {
            "module": "Conv2d(256, 8, kernel_size=1)",
            "tap": "net.classifier[3] (ReLU, stride 8)",
            "trainable_parameters": int(
                HEAD_IN_CHANNELS * EMBED_DIM + EMBED_DIM
            ),
            "init_seed": int(HEAD_INIT_SEED),
        },
        "loss": {
            "function": "cadvec.recognition_raster_model.instance_embedding_loss",
            "delta_v": DELTA_V,
            "delta_d": DELTA_D,
            "var_weight": VAR_WEIGHT,
            "dist_weight": DIST_WEIGHT,
            "max_pixels_per_instance": int(MAX_PIXELS_PER_INSTANCE),
            "max_instances_per_sample": int(MAX_INSTANCES),
            "max_pairs": int(MAX_PAIRS),
            "min_group_pixels": int(MIN_GROUP_PIXELS),
            "min_groups_per_crop": int(MIN_GROUPS_PER_CROP),
            "upsample_embeddings_to": "native crop (bilinear, align_corners=False)",
            "class_source": "crop.semantic (class indices) + crop.semantic_valid",
            "instance_class_array_in_cache": False,
            "independent_masks": "instance_valid and class_valid are ANDed, never derived",
            "unknown_negatives": False,
            "stuff_negatives": False,
            "known_instance_repulsion": "all distinct known (class, instance) groups",
        },
        "training": {
            "epochs": int(epochs),
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "accumulation_steps": int(ACCUMULATION_STEPS),
            "epoch_weighting": "group count",
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
        },
        "decoder": {
            "baseline": "semantic CC8 per Thing class (recognition_raster_infer.instance_ids)",
            "merge": "same-class fragment mean raw L2 < merge_distance (union-find)",
            "merge_distance": MERGE_DISTANCE,
            "mean_normalisation": False,
            "max_fragments_per_class": int(MAX_FRAGMENTS_PER_CLASS),
            "excess_fragments": "keep baseline unique id",
            "stuff": "unchanged (-1 instance, pixel classes bit-identical)",
            "ground_truth": "never read",
        },
        "plan": plan,
    }
    if write:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "prepare.json"), policy)
        _write_json(
            os.path.join(output_dir, "policy.json"),
            {key: value for key, value in policy.items() if key != "plan"},
        )
    return policy


# ---------------------------------------------------------------------------
# training (explicitly scheduled by the parent; CPU default)
# ---------------------------------------------------------------------------


def _semantic_contract(
    semantic_checkpoint: str, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "path": os.path.abspath(semantic_checkpoint),
        "schema": payload.get("schema"),
        "role": payload.get("role"),
        "epoch": payload.get("epoch"),
        "epochs": payload.get("epochs"),
        "num_classes": NUM_CLASSES,
        "frozen": True,
        "eval_always": True,
        "batch_norm_running_stats": True,
        "trainable": False,
    }


def save_head_checkpoint(
    path: str,
    embedder: PilotEmbedder,
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
        "version": AFFINITY_VERSION,
        "kind": "instance_affinity_head",
        "final_epoch_only": True,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "embed_dim": int(embedder.dim),
        "in_channels": int(embedder.in_channels),
        "head_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in embedder.head.state_dict().items()
        },
        "head_trainable_parameters": int(
            sum(p.numel() for p in embedder.head.parameters() if p.requires_grad)
        ),
        "semantic": _semantic_contract(semantic_checkpoint, semantic_payload),
        "contract": {
            "loaders_strict": True,
            "model_promotion": False,
            "pixel_metric": "raw_l2",
            "mean_normalisation": False,
            "merge_distance": MERGE_DISTANCE,
            "max_fragments_per_class": int(MAX_FRAGMENTS_PER_CLASS),
            "labels": "class indices 0..34, -1 unknown",
            "instances": "raw local ids >= 0; instance_valid is the pixel authority",
            "baseifier": "recognition_raster_infer.instance_ids (CC8, class offsets)",
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


def load_head_checkpoint(path: str, *, device: str = "cpu") -> Dict[str, Any]:
    torch = _require_torch()
    if not os.path.isfile(path):
        raise InstanceAffinityError(f"checkpoint_not_found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise InstanceAffinityError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("kind") != "instance_affinity_head":
        raise InstanceAffinityError(f"checkpoint_kind_mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise InstanceAffinityError("checkpoint_must_be_final_epoch_only")
    head_state = payload.get("head_state_dict")
    if not isinstance(head_state, Mapping) or not head_state:
        raise InstanceAffinityError("checkpoint_missing_head_state")
    if int(payload.get("embed_dim", -1)) != EMBED_DIM:
        raise InstanceAffinityError("checkpoint_embed_dim_mismatch")
    if int(payload.get("in_channels", -1)) != HEAD_IN_CHANNELS:
        raise InstanceAffinityError("checkpoint_in_channels_mismatch")
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping) or not semantic.get("path"):
        raise InstanceAffinityError("checkpoint_missing_semantic_reference")
    if semantic.get("frozen") is not True or semantic.get("eval_always") is not True:
        raise InstanceAffinityError("checkpoint_semantic_contract_not_frozen")
    result = dict(payload)
    result["head_state_dict"] = {
        key: value.to(device) for key, value in head_state.items()
    }
    result["semantic_reference_exists"] = bool(os.path.isfile(semantic["path"]))
    return result


def _assert_semantic_reference(head_payload: Mapping[str, Any], semantic_checkpoint: str) -> None:
    """Fail fast when a head checkpoint references a different semantic model."""

    reference = head_payload.get("semantic") or {}
    reference_path = reference.get("path")
    if not reference_path:
        raise InstanceAffinityError("head_checkpoint_missing_semantic_reference")
    if os.path.normcase(os.path.abspath(reference_path)) != os.path.normcase(
        os.path.abspath(semantic_checkpoint)
    ):
        raise InstanceAffinityError(
            "semantic_checkpoint_mismatch: head was trained against "
            f"{os.path.abspath(reference_path)}"
        )


def load_embedder(
    semantic_checkpoint: str,
    *,
    head_checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[PilotEmbedder, Dict[str, Any]]:
    """Load the frozen semantic model, then strictly load the head if given."""

    torch = _require_torch()
    model, payload = pilot.load_checkpoint_model(semantic_checkpoint, device=device)
    embedder = PilotEmbedder(model, device=device)
    report: Dict[str, Any] = {
        "semantic_checkpoint": os.path.abspath(semantic_checkpoint),
        "semantic_schema": payload.get("schema"),
        "semantic_epoch": payload.get("epoch"),
        "frozen": embedder.frozen_report,
        "head_loaded": False,
    }
    if head_checkpoint is not None:
        head_payload = load_head_checkpoint(head_checkpoint, device=device)
        _assert_semantic_reference(head_payload, semantic_checkpoint)
        embedder.head.load_state_dict(head_payload["head_state_dict"], strict=True)
        for parameter in embedder.head.parameters():
            parameter.requires_grad_(True)
        embedder.head.to(device)
        report["head_loaded"] = True
        report["head_checkpoint"] = os.path.abspath(head_checkpoint)
        report["head_epoch"] = head_payload.get("epoch")
        report["semantic_reference"] = head_payload.get("semantic")
    return embedder, report


def _crop_tensor(crop: Mapping[str, Any]) -> Any:
    numpy = _require_numpy()
    torch = _require_torch()
    image = numpy.ascontiguousarray(numpy.asarray(crop["image"], dtype=numpy.float32))
    return torch.from_numpy(image).unsqueeze(0)


def _crop_targets(crop: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    numpy = _require_numpy()
    torch = _require_torch()
    targets = torch.from_numpy(
        numpy.array(crop["instance"], dtype=numpy.int64, copy=True)
    ).unsqueeze(0)
    mask = torch.from_numpy(
        numpy.array(crop["instance_valid"], dtype=bool, copy=True)
    ).unsqueeze(0)
    classes = torch.from_numpy(
        numpy.array(crop["semantic"], dtype=numpy.int64, copy=True)
    ).unsqueeze(0)
    class_valid = torch.from_numpy(
        numpy.array(crop["semantic_valid"], dtype=bool, copy=True)
    ).unsqueeze(0)
    return targets, mask, classes, class_valid


def _microbatch_loss(
    embedder: PilotEmbedder, crop: Mapping[str, Any]
) -> Tuple[Any, int, int]:
    """Loss for one supervised crop.

    Eligibility (``crop_is_supervised``) only counts groups with at least
    ``MIN_GROUP_PIXELS`` pixels, but ``instance_embedding_loss`` itself masks
    purely on ``instance_valid & instance >= 0``.  A crop that passes the
    >=2-group gate therefore also contributes any *tiny* valid groups it
    carries, so the loss group count is >= the eligibility count.  Measured on
    the real 204-record cache: 2089/2108/2052/2063/2099 eligible groups vs
    2108/2123/2073/2077/2116 loss groups for epochs 1..5.  This is the existing
    upstream loss semantics and is deliberately not altered after training.
    """

    torch = _require_torch()
    image = _crop_tensor(crop)
    embeddings = embedder.crop_embeddings(image)
    targets, mask, classes, class_valid = _crop_targets(crop)
    components = raster_model.instance_embedding_loss(
        embeddings,
        targets,
        mask,
        class_ids=classes,
        class_valid=class_valid,
        delta_v=DELTA_V,
        delta_d=DELTA_D,
        var_weight=VAR_WEIGHT,
        dist_weight=DIST_WEIGHT,
        max_pixels_per_instance=MAX_PIXELS_PER_INSTANCE,
        max_instances_per_sample=MAX_INSTANCES,
        max_pairs=MAX_PAIRS,
        return_components=True,
    )
    loss = components["loss"]
    if not bool(torch.isfinite(loss)):
        raise InstanceAffinityError("non_finite_instance_loss")
    return loss, int(components["num_groups"]), int(components["num_pixels"])


def _assert_finite_gradient(parameters: Sequence[Any]) -> None:
    """Abort before the optimizer step if any head gradient is non-finite."""

    torch = _require_torch()
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise InstanceAffinityError("non_finite_head_gradient")


def train_instance_head(
    embedder: PilotEmbedder,
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
) -> Dict[str, Any]:
    """Bounded head-only fit: group-weighted loss, frozen semantics, manifest."""

    torch = _require_torch()
    train.require_command_role("fit", role)
    if not records:
        raise InstanceAffinityError("no fit records")
    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    lr = _finite(lr, "lr")
    weight_decay = _finite(weight_decay, "weight_decay")
    if weight_decay < 0.0 or lr <= 0.0:
        raise InstanceAffinityError("lr must be positive and weight_decay >= 0")
    # Freeze before the first optimizer build; the digest then proves it.
    freeze_semantic_model(embedder.model)
    digest_before = semantic_parameter_digest(embedder.model)
    head_parameters = [p for p in embedder.head.parameters() if p.requires_grad]
    if not head_parameters:
        raise InstanceAffinityError("head has no trainable parameters")
    optimizer = torch.optim.AdamW(head_parameters, lr=lr, weight_decay=weight_decay)
    head_init_state = {
        key: value.detach().cpu().clone()
        for key, value in embedder.head.state_dict().items()
    }
    started = time.monotonic()
    epoch_reports: List[Dict[str, Any]] = []
    final_metrics: Dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        torch = _require_torch()
        seed = fullfit.epoch_seed(base_seed, epoch)
        train.seed_everything(seed)
        source, _sampler = fullfit.make_epoch_source(
            records,
            epoch=epoch,
            cache_dir=cache_dir,
            base_seed=base_seed,
        )
        embedder.model.eval()
        embedder.head.train()
        optimizer.zero_grad(set_to_none=True)
        scheduled = 0
        supervised = 0
        skipped = 0
        steps = 0
        group_total = 0
        pixel_total = 0
        weighted_loss = 0.0
        pending: List[Tuple[Any, int]] = []
        pending_groups = 0
        pending_pixels = 0

        def flush() -> None:
            nonlocal steps, group_total, pixel_total, weighted_loss
            nonlocal pending, pending_groups, pending_pixels
            if not pending:
                return
            total = max(1, pending_groups)
            for loss, groups in pending:
                if not bool(torch.isfinite(loss)):
                    raise InstanceAffinityError("non_finite_instance_loss_before_backward")
                (loss * (float(groups) / float(total))).backward()
            _assert_finite_gradient(head_parameters)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            group_total += pending_groups
            pixel_total += pending_pixels
            weighted_loss += sum(float(loss.detach().item()) * g for loss, g in pending)
            pending = []
            pending_groups = 0
            pending_pixels = 0

        for index in range(len(source)):
            scheduled += 1
            crop = source.get(index)
            usable, report = crop_is_supervised(crop)
            if not usable:
                skipped += 1
                continue
            loss, groups, pixels = _microbatch_loss(embedder, crop)
            supervised += 1
            pending.append((loss, groups))
            pending_groups += groups
            pending_pixels += pixels
            if len(pending) >= accumulation_steps:
                flush()
        flush()
        embedder.model.eval()
        if semantic_parameter_digest(embedder.model) != digest_before:
            raise InstanceAffinityError(
                f"semantic parameters changed during epoch {epoch}"
            )
        if any(bool(module.training) for module in embedder.model.modules()):
            embedder.model.eval()
        epoch_report = {
            "epoch": int(epoch),
            "seed": int(seed),
            "scheduled_crops": int(scheduled),
            "supervised_crops": int(supervised),
            "skipped_crops": int(skipped),
            "optimizer_steps": int(steps),
            "groups": int(group_total),
            "pixels": int(pixel_total),
            "loss_group_weighted": (
                (weighted_loss / float(group_total)) if group_total else None
            ),
            "accumulation_steps": int(accumulation_steps),
            "semantic_parameters_unchanged": True,
        }
        epoch_reports.append(epoch_report)
        final_metrics = epoch_report
    head_moved = any(
        not torch.equal(
            head_init_state[key].to(embedder.head.state_dict()[key].device),
            embedder.head.state_dict()[key],
        )
        for key in head_init_state
    )
    result = {
        "schema": FIT_SCHEMA,
        "version": AFFINITY_VERSION,
        "kind": "instance_affinity_fit",
        "final_epoch_only": True,
        "epochs": int(epochs),
        "per_epoch": epoch_reports,
        "final": final_metrics,
        "head_changed_from_init": bool(head_moved),
        "head_trainable_parameters": int(
            sum(p.numel() for p in embedder.head.parameters() if p.requires_grad)
        ),
        "finite_checks": {
            "loss_per_microbatch_aborts_on_nonfinite": True,
            "loss_before_backward_aborts_on_nonfinite": True,
            "grad_before_step_aborts_on_nonfinite": True,
            "nonfinite_detected": False,
            "note": "any detection aborts before optimizer.step; a written fit.json means zero detections",
        },
        "semantic_parameters_frozen": True,
        "semantic_parameters_unchanged": True,
        "semantic_batch_norm_eval": True,
        "loss": {
            "delta_v": DELTA_V,
            "delta_d": DELTA_D,
            "max_pixels_per_instance": int(MAX_PIXELS_PER_INSTANCE),
            "max_instances_per_sample": int(MAX_INSTANCES),
            "max_pairs": int(MAX_PAIRS),
            "group_weighted": True,
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
        "seconds": round(time.monotonic() - started, 3),
    }
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint_instance_affinity.pt")
    checkpoint = save_head_checkpoint(
        checkpoint_path,
        embedder,
        epoch=epochs,
        epochs=epochs,
        semantic_checkpoint=semantic_checkpoint,
        semantic_payload=semantic_payload,
        metrics={"final": final_metrics, "epochs": epoch_reports},
    )
    result["checkpoint"] = checkpoint
    _write_json(os.path.join(output_dir, "fit.json"), result)
    return result


# ---------------------------------------------------------------------------
# deterministic decoder: baseline CC8 -> same-class fragment-mean agglomeration
# ---------------------------------------------------------------------------


def baseline_instance_ids(semantic: Any) -> Any:
    """CC8 baseline in class-index space (Stuff and unknown stay -1).

    ``raster_infer.instance_ids`` consumes class ids *directly*: Thing classes
    0..29 become connected components, 30..34 (Stuff) and 35 (ignore) become
    -1.  Cache labels already live in class-index space, so there is no id
    shift here; shifting class 29 to 30 would silently drop it as Stuff.
    """

    numpy = _require_numpy()
    values = numpy.asarray(semantic)
    if values.ndim != 2 or not numpy.issubdtype(values.dtype, numpy.integer):
        raise InstanceAffinityError("semantic must be a 2D integer array")
    thing = (values >= 0) & (values < NUM_THING_CLASSES)
    raw = numpy.full(values.shape, IGNORE_CLASS, dtype=numpy.int64)
    raw[thing] = values[thing].astype(numpy.int64)
    return numpy.asarray(raster_infer.instance_ids(raw), dtype=numpy.int32)


def _union_find_merge(
    means: Any, *, merge_distance: float, pair_chunk: int
) -> Tuple[List[int], int, int]:
    """Return (roots, merge_edges, pairs_examined) with chunked L2 blocks."""

    numpy = _require_numpy()
    count = int(means.shape[0])
    parent = list(range(count))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root
        return True

    threshold = float(merge_distance) ** 2
    edges = 0
    examined = 0
    for start in range(0, count, int(pair_chunk)):
        stop = min(count, start + int(pair_chunk))
        block = means[start:stop]
        delta = block[:, None, :] - means[None, :, :]
        squared = numpy.einsum("ijk,ijk->ij", delta, delta)
        examined += int((stop - start) * count)
        rows, columns = numpy.nonzero(squared < threshold)
        for row, column in zip(rows.tolist(), columns.tolist()):
            left = start + int(row)
            right = int(column)
            if right <= left:
                continue
            if union(left, right):
                edges += 1
    return [find(item) for item in range(count)], edges, examined


def agglomerate_instances(
    semantic: Any,
    embeddings: Any,
    *,
    semantic_valid: Any = None,
    baseline_instance: Any = None,
    merge_distance: float = MERGE_DISTANCE,
    max_fragments_per_class: int = MAX_FRAGMENTS_PER_CLASS,
    pair_chunk: int = PAIR_CHUNK,
    min_fragment_pixels: int = MIN_FRAGMENT_PIXELS,
) -> Dict[str, Any]:
    """Merge same-class baseline fragments whose raw mean embeddings are close.

    * no ground truth is read: fragments come from the semantic map alone;
    * pixel classes and semantic support are copied through bit-identically;
    * Stuff (30..34) keeps instance ``-1`` and is never clustered;
    * cluster ids are the smallest baseline id in the cluster, so ids stay
      globally unique across classes and can never collide with an untouched
      fragment (baseline ids are already unique across classes);
    * when a class has more than ``max_fragments_per_class`` fragments, the
      lowest ids are merged and the excess keep their baseline id unchanged
      (never dropped);
    * chain sizes are reported; nothing is merged silently.

    Labels are class-index space ``0..34`` with ``-1`` unknown (the cache
    contract).  Evaluation lanes that save the raw ``35`` ignore sentinel must
    normalise it to ``-1`` at their own boundary before calling this function;
    ``semantic_valid`` (when given) gates support without touching pixel
    classes.
    """

    numpy = _require_numpy()
    semantic = numpy.asarray(semantic)
    semantic_snapshot = semantic.copy()
    if semantic.ndim != 2 or not numpy.issubdtype(semantic.dtype, numpy.integer):
        raise InstanceAffinityError("semantic must be a 2D integer array")
    if semantic.size and (int(semantic.min()) < -1 or int(semantic.max()) > 34):
        raise InstanceAffinityError("semantic must hold class indices -1..34")
    embeddings = numpy.asarray(embeddings, dtype=numpy.float32)
    if embeddings.ndim != 3 or tuple(embeddings.shape[1:]) != semantic.shape:
        raise InstanceAffinityError(
            f"embeddings must be D x H x W matching semantic; got {embeddings.shape}"
        )
    if embeddings.shape[0] < 1:
        raise InstanceAffinityError("embedding dimension must be positive")
    if not numpy.isfinite(embeddings).all():
        raise InstanceAffinityError("non_finite_embeddings")
    merge_distance = _finite(merge_distance, "merge_distance")
    if merge_distance < 0.0:
        raise InstanceAffinityError("merge_distance must be >= 0")
    max_fragments_per_class = _positive_int(
        "max_fragments_per_class", max_fragments_per_class
    )
    pair_chunk = _positive_int("pair_chunk", pair_chunk)
    min_fragment_pixels = _positive_int("min_fragment_pixels", min_fragment_pixels)
    if semantic_valid is None:
        known = semantic >= 0
    else:
        known = numpy.asarray(semantic_valid, dtype=bool) & (semantic >= 0)
        if known.shape != semantic.shape:
            raise InstanceAffinityError("semantic_valid shape does not match")
    thing = known & (semantic < NUM_THING_CLASSES)
    if baseline_instance is None:
        # Honor semantic_valid: a class-valued pixel whose support is invalid
        # must not acquire an instance id.
        baseline = baseline_instance_ids(numpy.where(thing, semantic, -1))
    else:
        baseline = numpy.asarray(baseline_instance)
        if baseline.shape != semantic.shape or not numpy.issubdtype(
            baseline.dtype, numpy.integer
        ):
            raise InstanceAffinityError("baseline_instance must be an int array with the same shape")
    baseline = baseline.astype(numpy.int64, copy=True)
    if bool((baseline[thing] < 0).any()):
        raise InstanceAffinityError("baseline_instance has negative ids on Thing pixels")
    if bool((baseline[~thing] != -1).any()):
        raise InstanceAffinityError("baseline_instance must be -1 outside Thing support")
    # Ids must already be unique across classes; otherwise a merge could relabel
    # pixels of another class.  Different classes can never share an id.
    id_class: Dict[int, int] = {}
    for class_value in numpy.unique(semantic[thing]).tolist():
        mask = thing & (semantic == int(class_value))
        for identifier in numpy.unique(baseline[mask]).tolist():
            identifier = int(identifier)
            if identifier in id_class and id_class[identifier] != int(class_value):
                raise InstanceAffinityError(
                    f"baseline_instance id {identifier} spans classes "
                    f"{id_class[identifier]} and {int(class_value)}"
                )
            id_class[identifier] = int(class_value)

    out = numpy.full(semantic.shape, -1, dtype=numpy.int32)
    clusters_total = 0
    merged_clusters = 0
    fragments_total = 0
    candidates_total = 0
    excess_total = 0
    merged_fragments = 0
    merge_edges = 0
    pairs_examined = 0
    chain_sizes: List[int] = []
    chain_histogram: Dict[str, int] = {}
    per_class: Dict[str, Any] = {}
    id_semantic: Dict[int, int] = {}

    for class_id in range(NUM_THING_CLASSES):
        class_mask = (semantic == int(class_id)) & thing
        if not bool(class_mask.any()):
            continue
        fragment_ids = [int(v) for v in numpy.unique(baseline[class_mask & thing])]
        fragment_ids = [value for value in fragment_ids if value >= 0]
        if not fragment_ids:
            raise InstanceAffinityError(f"thing pixels without baseline ids: class {class_id}")
        fragments_total += len(fragment_ids)
        candidates = fragment_ids[:max_fragments_per_class]
        excess = fragment_ids[max_fragments_per_class:]
        excess_total += len(excess)
        # Fragments too small to have a stable mean stay untouched.
        usable: List[int] = []
        means_list: List[Any] = []
        for fragment in candidates:
            mask = class_mask & (baseline == int(fragment))
            pixels = int(mask.sum())
            if pixels < min_fragment_pixels:
                continue
            usable.append(fragment)
            means_list.append(embeddings[:, mask].mean(axis=1, dtype=numpy.float64))
        candidates_total += len(candidates)
        class_report: Dict[str, Any] = {
            "fragments": len(fragment_ids),
            "candidates": len(candidates),
            "excess_kept_baseline": len(excess),
            "clustered_fragments": len(usable),
            "clusters": 0,
            "merged_clusters": 0,
            "max_chain": 1,
        }
        if usable:
            means = numpy.stack(means_list, axis=0)
            roots, edges, examined = _union_find_merge(
                means, merge_distance=merge_distance, pair_chunk=pair_chunk
            )
            merge_edges += int(edges)
            pairs_examined += int(examined)
            groups: Dict[int, List[int]] = {}
            for position, root in enumerate(roots):
                groups.setdefault(int(root), []).append(position)
            for members in groups.values():
                fragment_members = [usable[position] for position in members]
                cluster_id = min(fragment_members)
                for fragment in fragment_members:
                    out[class_mask & (baseline == int(fragment))] = int(cluster_id)
                    id_semantic[int(cluster_id)] = int(class_id)
                clusters_total += 1
                chain_sizes.append(len(fragment_members))
                if len(fragment_members) > 1:
                    merged_clusters += 1
                    merged_fragments += len(fragment_members)
                    key = str(len(fragment_members))
                    chain_histogram[key] = int(chain_histogram.get(key, 0)) + 1
                class_report["clusters"] += 1
                class_report["merged_clusters"] += int(len(fragment_members) > 1)
                class_report["max_chain"] = max(
                    int(class_report["max_chain"]), len(fragment_members)
                )
        # Excess and sub-minimum fragments keep their baseline id untouched.
        for fragment in candidates:
            if fragment in usable:
                continue
            mask = class_mask & (baseline == int(fragment))
            out[mask] = int(fragment)
            id_semantic[int(fragment)] = int(class_id)
            clusters_total += 1
            chain_sizes.append(1)
            class_report["clusters"] += 1
        for fragment in excess:
            mask = class_mask & (baseline == int(fragment))
            out[mask] = int(fragment)
            id_semantic[int(fragment)] = int(class_id)
            clusters_total += 1
            chain_sizes.append(1)
            class_report["clusters"] += 1
        per_class[str(class_id)] = class_report

    # --- invariant checks (cheap, and they are the eval contract) --------
    thing_pixels = int(thing.sum())
    output_thing = int((out >= 0).sum())
    class_per_pixel_ok = True
    for instance_id, class_id in id_semantic.items():
        mask = out == int(instance_id)
        if bool(mask.any()):
            classes = numpy.unique(semantic[mask])
            if classes.tolist() != [int(class_id)]:
                class_per_pixel_ok = False
                break
    same_classes = bool(
        class_per_pixel_ok
        and thing_pixels == output_thing
        and bool((out[~thing] == -1).all())
        and bool((semantic == semantic_snapshot).all())
    )
    stats = {
        "class": "thing-only same-class fragment-mean L2 union-find",
        "merge_distance": merge_distance,
        "mean_normalisation": False,
        "ground_truth_used": False,
        "fragments_total": int(fragments_total),
        "fragments_clustered": int(candidates_total),
        "fragments_excess_kept_baseline": int(excess_total),
        "clusters_total": int(clusters_total),
        "merged_clusters": int(merged_clusters),
        "merged_fragments": int(merged_fragments),
        "merge_edges": int(merge_edges),
        "pairs_examined": int(pairs_examined),
        "max_chain": int(max(chain_sizes) if chain_sizes else 0),
        "mean_chain": (
            float(sum(chain_sizes)) / float(len(chain_sizes)) if chain_sizes else 0.0
        ),
        "merge_chain_size_histogram": {
            key: int(chain_histogram[key]) for key in sorted(chain_histogram, key=int)
        },
        "thing_pixels": thing_pixels,
        "output_thing_pixels": output_thing,
        "pixel_classes_bitidentical": bool(same_classes),
        "stuff_instance_unchanged": bool((out[(semantic >= NUM_THING_CLASSES) & (semantic <= 34)] == -1).all()),
        "unknown_instance_unchanged": bool((out[semantic < 0] == -1).all()),
        "per_class": per_class,
    }
    if not same_classes:
        raise InstanceAffinityError("decoder_invariant_violated: pixel classes or support changed")
    return {
        "instance": out,
        "semantic": semantic,
        "baseline_instance": baseline.astype(numpy.int32),
        "stats": stats,
    }


def decode_from_image(
    embedder: PilotEmbedder,
    image: Any,
    semantic: Any,
    *,
    semantic_valid: Any = None,
    merge_distance: float = MERGE_DISTANCE,
    max_fragments_per_class: int = MAX_FRAGMENTS_PER_CLASS,
    pair_chunk: int = PAIR_CHUNK,
) -> Dict[str, Any]:
    """Saved-semantics decode lane: native embeddings + agglomeration only."""

    embeddings = embedder.native_embeddings(image)
    result = agglomerate_instances(
        semantic,
        embeddings,
        semantic_valid=semantic_valid,
        merge_distance=merge_distance,
        max_fragments_per_class=max_fragments_per_class,
        pair_chunk=pair_chunk,
    )
    result["embeddings"] = embeddings
    return result


# ---------------------------------------------------------------------------
# smoke: CPU load of the real frozen checkpoint + feature shape
# ---------------------------------------------------------------------------


def smoke(
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    *,
    size: int = CROP_SIZE,
    device: str = "cpu",
) -> Dict[str, Any]:
    torch = _require_torch()
    numpy = _require_numpy()
    size = _positive_int("size", size)
    embedder, report = load_embedder(semantic_checkpoint, device=device)
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    outputs = embedder.feature_map(image)
    embeddings = embedder.crop_embeddings(image)
    head_parameters = [p for p in embedder.head.parameters()]
    semantic_requires_grad = sum(
        1 for p in embedder.model.parameters() if p.requires_grad
    )
    result = {
        "schema": SMOKE_SCHEMA,
        "version": AFFINITY_VERSION,
        "device": device,
        "image_shape": [1, 3, size, size],
        "feature_shape": [int(v) for v in outputs["features"].shape],
        "embedding_shape": [int(v) for v in embeddings.shape],
        "head_parameters": int(sum(p.numel() for p in head_parameters)),
        "semantic_parameters_requires_grad": int(semantic_requires_grad),
        "semantic_training": bool(embedder.model.training),
        "finite_embeddings": bool(numpy.isfinite(embeddings.detach().numpy()).all()),
        "report": report,
    }
    if tuple(result["feature_shape"][-2:]) != (
        size // FEATURE_STRIDE,
        size // FEATURE_STRIDE,
    ):
        raise InstanceAffinityError("unexpected feature stride")
    if tuple(result["embedding_shape"]) != (1, EMBED_DIM, size, size):
        raise InstanceAffinityError("unexpected embedding shape")
    if semantic_requires_grad:
        raise InstanceAffinityError("semantic parameters are not frozen")
    embedder.close()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_instance_affinity",
        description=(
            "Frozen-semantic instance affinity pilot: CPU prepare, CPU smoke and "
            "an explicitly scheduled head-only fit."
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

    smoke_parser = subparsers.add_parser("smoke", help="CPU checkpoint smoke")
    smoke_parser.add_argument(
        "--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT
    )
    smoke_parser.add_argument("--size", type=int, default=CROP_SIZE)
    smoke_parser.add_argument("--device", default="cpu")

    fit_parser = subparsers.add_parser("fit", help="head-only fit (scheduled)")
    fit_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    fit_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    fit_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    fit_parser.add_argument(
        "--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT
    )
    fit_parser.add_argument("--role", default="fit")
    fit_parser.add_argument("--device", default="cpu")
    fit_parser.add_argument("--epochs", type=int, default=EPOCHS)
    fit_parser.add_argument("--accumulation-steps", type=int, default=ACCUMULATION_STEPS)
    fit_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    fit_parser.add_argument("--lr", type=float, default=LR)
    fit_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
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
        )
        print(json.dumps(payload["plan"], indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        print(json.dumps(smoke(args.semantic_checkpoint, size=args.size, device=args.device), indent=2, sort_keys=True))
        return 0
    if args.command == "fit":
        train.require_command_role("fit", args.role)
        records = fullfit.load_fit_records(args.manifest)
        model, payload = pilot.load_checkpoint_model(
            args.semantic_checkpoint, device=args.device
        )
        embedder = PilotEmbedder(model, device=args.device)
        result = train_instance_head(
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
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
