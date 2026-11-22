"""Per-pixel semantic **conflict** head: predicts whether the class at a pixel is wrong.

Why this lane exists
--------------------
The joint lane (:mod:`cadvec.recognition_joint_offsets`) recomputes a semantic map
from the PNG only.  Some of those pixels are confidently wrong, and every
downstream primitive built from them inherits the error.  This lane is the
narrowest possible repair: keep the whole map, and decide **per pixel** whether
that pixel's existing class should be abstained (ignore class ``35``) instead of
being written into geometry.

What is supervised
------------------
Only *licensed* pixels: ``semantic_valid`` is true, the ground-truth class is
``0..34`` and the PNG has ink there.  For those pixels the binary target is

    target = 1  iff  judged_class != gt_class
    target = 0  iff  judged_class == gt_class

``judged_class`` is the class the current map claims.  During fit it comes from
the **same frozen forward** that produces the features (local argmax over the
recomputed logits).  At inference it is the class of the **saved native joint
map** (384 tile / 192 stride averaged logits, CC8 saved maps), cropped to the
conflict tile grid without any rescale.  That is a deliberate distribution shift,
disclosed below; the head therefore learns ``P(wrong | features, judged class)``
and does not reuse a single asymmetric input channel.

Everything else is excluded, never turned into a negative: unknown pixels, GT
outside ``0..34``, pixels outside PNG ink, and pixels without a judged class.
Instance labels are irrelevant here and are never read: a ``Thing`` pixel with
``instance == -1`` is still valid semantic evidence, and a component that is
partly unknown only contributes its known pixels - whole components are never
labelled negative.

What the head sees
------------------
A small local convolution head on the frozen ``net.classifier[3]`` tap (256
channels, stride 8), plus the same forward's semantic logits, PNG ink context and
an explicit **one-hot judged-class conditioner** (one channel block per class).
The conditioner is what makes the target well defined when two pixels have
identical features but different judged classes.  Channels are identical at fit
and inference time; only the *source* of the judged class differs.

The head runs at the tap stride and its single-channel conflict logit is
bilinearly upsampled to the crop, exactly like the joint offset head, so its
context is the usual neural receptive field.  No translation-invariance claim is
made and no absolute coordinate is an input.

Hard contract
-------------
* the trunk is frozen: ``requires_grad_(False)``, ``eval()``, BatchNorm running
  statistics untouched, forward under ``torch.no_grad`` - a training step moves
  head parameters only;
* the head is constructed from an explicit ``torch.Generator`` seeded with
  :data:`HEAD_INIT_SEED` (20261108) *before* the trunk is built, so no global RNG
  state is consumed;
* loss is unweighted ``BCEWithLogits`` summed over licensed pixels and divided by
  the **actual licensed pixel mass of the accumulation window**; zero-supervision
  crops are skipped before any forward;
* inference thresholds the mean conflict logit: overlap-averaged logits across
  256/128 tiles, **one** sigmoid after the mean, fixed ``>= 0.5`` -> ignore ``35``;
  no size or min-area filter, no pixel added, no pixel relabelled;
* the output map is therefore a strict subset of the input ink and every kept
  label is bit-identical to the input map.

Honesty notes (read before using numbers from this lane)
---------------------------------------------------------
* Training crops are 256 px; the production semantic reference is the saved
  384/192 average.  The training target class is a **local 256 px proxy** for the
  native global map, so fit-time conflict rates are not the production rate.
* Freezing parameters does not freeze the *inference context*: the averaged
  384/192 map is not reproduced by a single 256 crop.  No claim of "exact
  semantics" is made anywhere in this module.
* At inference the ground-truth-derived ``semantic_valid`` mask does not exist, so
  every PNG ink pixel is scored.  That extrapolation beyond the training mask
  distribution is intentional and reported in :func:`predict_conflict`.

Nothing here trains on real data, reads the GT cache or touches a GPU in its test
suite; the live FIT run is parent-scheduled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from . import recognition_joint_offsets as joint
from . import recognition_pretrained_fullfit as fullfit
from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_data as raster_data

# ---------------------------------------------------------------------------
# constants (settled; these are not knobs)
# ---------------------------------------------------------------------------

VERSION = 1
CHECKPOINT_SCHEMA = "cadvec.recognition.pixel_conflict_checkpoint/1"
READINESS_SCHEMA = "cadvec.recognition.pixel_conflict_readiness/1"
# Artifact schemas for the PNG-only persistence phase.  The writers are published
# so the evaluator can bind to them; writing real samples stays parent-authorised.
CONFLICT_MAP_SCHEMA = "cadvec.recognition.pixel_conflict_map/1"
SEMANTIC_SCHEMA = "cadvec.recognition.pixel_conflict_semantics/1"
CALIBRATION_SCHEMA = "cadvec.recognition.pixel_conflict_calibration/1"
PREDICTIONS_SCHEMA = "cadvec.recognition.pixel_conflict_predictions/1"
DOCUMENT_SCHEMA = "cadvec.recognition.pixel_conflict_document/1"

# Persistence quantities fixed by the evaluation contract (not knobs).
PREDICTION_SAMPLE_LIMIT = 24  # the frozen FIT-24 prefix, one map per sample
FIT_RECORDS = 204
FIT_GROUPS = 192
JUDGED_CLASS_SOURCE = "saved_native_cc8"

# Conflict probability tiling (independent of the 384/192 semantic reference).
CROP_SIZE = 256
TILE_SIZE = 256
TILE_STRIDE = 128
CROPS_PER_RECORD = 8
CROP_MIN_VALID = fullfit.CROP_MIN_VALID  # 32

# Training: 4 epochs, AdamW on the head only.
EPOCHS = 4
ACCUMULATION_STEPS = 8
BASE_SEED = 20261108
EPOCH_SEED_STRIDE = 1000003
HEAD_INIT_SEED = 20261108
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

NUM_CLASSES = pilot.NUM_CLASSES  # 35, the semantic ignore id is 35
IGNORE_CLASS = 35
FEATURE_CHANNELS = 256  # net.classifier[3] tap width
HEAD_HIDDEN_CHANNELS = 32
FEATURE_STRIDE = 8

# One sigmoid after the overlap mean; inclusive at the boundary.
CONFLICT_THRESHOLD = 0.5

MAX_INPUT_PIXELS = joint.MAX_INPUT_PIXELS
MAX_INPUT_BYTES = joint.MAX_INPUT_BYTES
MAX_LOGIT_ACCUM_BYTES = 2 * 1024**3

DEFAULT_JOINT_CHECKPOINT = os.path.join(joint.DEFAULT_OUTPUT_DIR, "checkpoint_joint.pt")
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "recognition-pixel-conflict-20261112-01"
)

_REQUIRED_GEOMETRY = {
    "crop_size": joint.CROP_SIZE,
    "tile_size": joint.TILE_SIZE,
    "tile_stride": joint.TILE_STRIDE,
    "num_classes": NUM_CLASSES,
    "tap": "net.classifier[3]",
    "tap_channels": FEATURE_CHANNELS,
}


class PixelConflictError(pilot.PilotError):
    """Raised for contract violations of the per-pixel conflict lane."""


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover - environment dependent
        raise PixelConflictError("numpy is required for the pixel conflict lane")
    return np


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - environment dependent
        raise PixelConflictError("torch is required for the pixel conflict lane")
    return torch


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PixelConflictError(f"{name} must be an int; got {value!r}")
    if value <= 0:
        raise PixelConflictError(f"{name} must be positive; got {value}")
    return int(value)


def epoch_seed(epoch: int, *, base_seed: int = BASE_SEED) -> int:
    """Deterministic disclosed seed for one epoch: ``base + 1000003 * epoch``."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise PixelConflictError(f"epoch must be a positive int; got {epoch!r}")
    return int(base_seed) + EPOCH_SEED_STRIDE * int(epoch)


def _as_class_array(values: Any, name: str) -> Any:
    numpy = _require_numpy()
    array = numpy.asarray(values)
    if array.ndim != 2:
        raise PixelConflictError(f"{name} must be 2-D H x W; got {array.shape}")
    if array.dtype == bool or not numpy.issubdtype(array.dtype, numpy.integer):
        raise PixelConflictError(
            f"{name} must be an integer class array; got dtype {array.dtype}"
        )
    return array.astype(numpy.int64, copy=False)


def _as_bool_mask(values: Any, name: str, shape: Tuple[int, int]) -> Any:
    numpy = _require_numpy()
    array = numpy.asarray(values)
    if array.shape != shape:
        raise PixelConflictError(
            f"{name} shape {tuple(array.shape)} does not match {shape}"
        )
    if array.dtype != bool:
        raise PixelConflictError(f"{name} must be a bool mask; got {array.dtype}")
    return array


def as_unit_chw(image: Any) -> Any:
    """Validate one unit-range ``3 x H x W`` float image.

    Accepts a numpy array or a torch tensor.  Values must be finite and lie in
    ``[0, 1]``: this lane never rescales and never applies a second ``/255``, so
    an 8-bit-range array is rejected instead of being silently normalised.
    """

    torch_module = _require_torch()
    numpy = _require_numpy()
    if torch_module.is_tensor(image):
        if image.requires_grad:
            raise PixelConflictError("image must not require grad")
        source = image.detach().to("cpu", dtype=torch_module.float32)
    else:
        source = torch_module.from_numpy(
            numpy.ascontiguousarray(numpy.asarray(image, dtype=numpy.float32))
        )
    if source.ndim != 3 or int(source.shape[0]) != 3:
        raise PixelConflictError(
            f"image must be unit-range C x H x W with C=3; got {tuple(source.shape)}"
        )
    if not bool(torch_module.isfinite(source).all()):
        raise PixelConflictError("non_finite_image")
    if float(source.min()) < 0.0 or float(source.max()) > 1.0:
        raise PixelConflictError(
            "image must already be unit range [0, 1]; this lane applies no /255 "
            f"(observed min={float(source.min()):.4f}, "
            f"max={float(source.max()):.4f})"
        )
    height, width = int(source.shape[-2]), int(source.shape[-1])
    if height <= 0 or width <= 0:
        raise PixelConflictError("empty image")
    if height * width > MAX_INPUT_PIXELS:
        raise PixelConflictError(
            f"image is {width}x{height} = {height * width} pixels, above "
            f"max_pixels={MAX_INPUT_PIXELS}; no resize is applied"
        )
    return source


def image_ink(image: Any) -> Any:
    """PNG ink support, delegated to the shared data helper (min RGB < 254/255)."""

    source = as_unit_chw(image)
    return np.asarray(raster_data._image_ink_support(source.numpy()), dtype=bool)


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------


def head_in_channels() -> int:
    """Frozen features + semantic logits + ink context + judged-class one-hot."""

    return FEATURE_CHANNELS + NUM_CLASSES + 1 + NUM_CLASSES


def build_conflict_head(
    *,
    in_channels: int = -1,
    hidden_channels: int = HEAD_HIDDEN_CHANNELS,
    seed: int = HEAD_INIT_SEED,
    device: str = "cpu",
) -> Any:
    """``Conv1x1 -> ReLU -> Conv3x3 -> ReLU -> Conv1x1(->1)`` with explicit init.

    Every weight is drawn from a ``torch.Generator`` seeded with ``seed`` so the
    head never depends on the global RNG and the fit lane can set an epoch seed
    after the head exists without changing a single value.
    """

    torch_module = _require_torch()
    channels = head_in_channels() if in_channels is None or int(in_channels) < 0 else int(in_channels)
    channels = _positive_int("in_channels", channels)
    hidden = _positive_int("hidden_channels", int(hidden_channels))
    generator = torch_module.Generator(device="cpu").manual_seed(int(seed))
    layers: List[Any] = [
        torch_module.nn.Conv2d(channels, hidden, kernel_size=1),
        torch_module.nn.ReLU(inplace=False),
        torch_module.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        torch_module.nn.ReLU(inplace=False),
        torch_module.nn.Conv2d(hidden, 1, kernel_size=1),
    ]
    with torch_module.no_grad():
        for layer in layers:
            if not isinstance(layer, torch_module.nn.Conv2d):
                continue
            fan_in = float(
                layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            )
            bound = 1.0 / math.sqrt(fan_in)
            layer.weight.uniform_(-bound, bound, generator=generator)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound, generator=generator)
    module = torch_module.nn.Sequential(*layers)
    module.to(device)
    return module


def head_parameter_count(*, in_channels: Optional[int] = None) -> int:
    channels = head_in_channels() if in_channels is None else int(in_channels)
    hidden = HEAD_HIDDEN_CHANNELS
    return int(
        channels * hidden
        + hidden
        + hidden * hidden * 9
        + hidden
        + hidden
        + 1
    )


def _judged_one_hot(judged: Any) -> Any:
    """``N x H x W`` long classes -> ``N x 35 x H x W``; class 35 is all zeros."""

    torch_module = _require_torch()
    if not torch_module.is_tensor(judged):
        judged = torch_module.as_tensor(np.asarray(judged), dtype=torch_module.long)
    classes = judged.to(torch_module.long)
    if classes.ndim == 2:
        classes = classes.unsqueeze(0)
    if classes.ndim == 4:
        if int(classes.shape[1]) != 1:
            raise PixelConflictError("judged_class must be N x H x W or N x 1 x H x W")
        classes = classes[:, 0]
    if classes.ndim != 3:
        raise PixelConflictError(
            f"judged_class must be N x H x W; got {tuple(classes.shape)}"
        )
    inside = (classes >= 0) & (classes < NUM_CLASSES)
    clamped = torch_module.where(inside, classes, torch_module.zeros_like(classes))
    one_hot = torch_module.nn.functional.one_hot(
        clamped, num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2)
    return one_hot.to(dtype=torch_module.float32) * inside.unsqueeze(1).to(
        dtype=torch_module.float32
    )


def head_inputs(
    features: Any,
    semantic_logits: Any,
    ink: Any,
    judged_class: Any,
) -> Any:
    """Build the symmetric head input block at the tap resolution.

    ``features`` ``N x 256 x h x w`` (tap), ``semantic_logits`` ``N x 35 x H x W``
    (same forward), ``ink`` ``N x H x W`` bool, ``judged_class`` ``N x H x W``
    integer classes.  Logits and the ink fraction are area-averaged into the tap
    grid; the conditioner is the exact one-hot of each tap cell's majority judged
    class (ties resolved to the lowest id, empty cells to all zeros).  The same
    block is built at fit and inference time - only the judged-class source
    differs.
    """

    torch_module = _require_torch()
    if features.ndim != 4 or int(features.shape[1]) != FEATURE_CHANNELS:
        raise PixelConflictError(
            f"tap features must be N x {FEATURE_CHANNELS} x h x w; "
            f"got {tuple(features.shape)}"
        )
    if semantic_logits.ndim != 4 or int(semantic_logits.shape[1]) != NUM_CLASSES:
        raise PixelConflictError(
            f"semantic logits must be N x {NUM_CLASSES} x H x W; "
            f"got {tuple(semantic_logits.shape)}"
        )
    if tuple(semantic_logits.shape[-2:]) != tuple(ink.shape[-2:]):
        raise PixelConflictError("ink must match the semantic logit resolution")
    if tuple(judged_class.shape[-2:]) != tuple(ink.shape[-2:]):
        raise PixelConflictError("judged_class must match the ink resolution")
    tap_size = tuple(int(value) for value in features.shape[-2:])

    logits_tap = F.adaptive_avg_pool2d(semantic_logits, tap_size)
    ink_tap = F.adaptive_avg_pool2d(
        ink.to(dtype=torch_module.float32).unsqueeze(1), tap_size
    )
    fractions = F.adaptive_avg_pool2d(_judged_one_hot(judged_class), tap_size)
    mass = fractions.sum(dim=1, keepdim=True)
    # Empty cells must not fall back to class 0 through argmax.
    majority = torch_module.where(
        mass > 0.0, fractions.argmax(dim=1, keepdim=True), torch_module.zeros_like(
            mass, dtype=torch_module.long
        )
    )
    conditioner = torch_module.zeros_like(fractions)
    present = mass > 0.0
    if bool(present.any()):
        conditioner.scatter_(1, majority, present.to(dtype=fractions.dtype))
    return torch_module.cat(
        [features.to(dtype=torch_module.float32), logits_tap, ink_tap, conditioner],
        dim=1,
    )


def head_logits(head: Any, inputs: Any, *, output_size: Optional[Tuple[int, int]] = None) -> Any:
    """Run the head at the tap stride and upsample one conflict logit channel."""

    torch_module = _require_torch()
    logits = head(inputs)
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise PixelConflictError(
            f"conflict head must return N x 1 x h x w; got {tuple(logits.shape)}"
        )
    if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
        logits = F.interpolate(
            logits, size=tuple(int(v) for v in output_size),
            mode="bilinear", align_corners=False,
        )
    return logits


# ---------------------------------------------------------------------------
# frozen reference (joint checkpoint, inference context disclosed)
# ---------------------------------------------------------------------------


if torch is not None:  # pragma: no cover - real definition

    class FrozenJointReference(torch.nn.Module):
        """Frozen joint checkpoint exposing the classifier[3] tap + logits.

        Wraps :class:`cadvec.recognition_joint_offsets.JointEmbedder` so the exact
        production loader, normalisation wrapper and tap are reused without editing
        shared source.  Everything is frozen and every forward is ``no_grad``; the
        returned tensors are detached, so a training step can only move the
        conflict head.
        """

        def __init__(self, embedder: Any, payload: Mapping[str, Any], *, device: str = "cpu") -> None:
            super().__init__()
            torch_module = _require_torch()
            self.embedder = embedder
            self.device = str(device)
            self.payload = dict(payload)
            self.calls = 0
            self.embedder.eval_mode()
            # The wrapper itself must also report eval: ``train`` below refuses to
            # flip it, but construction has to start from the frozen state.
            self.train(False)
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.model_digest = joint.model_digest(self.embedder.model)
            self.head_digest = _state_digest(self.embedder.head.state_dict())

        @property
        def training_state(self) -> str:
            return "eval"

        def train(self, mode: bool = True) -> Any:  # noqa: D401 - torch API
            # The trunk never leaves eval: no BN momentum, no dropout change.
            super().train(False)
            self.embedder.eval_mode()
            return self

        def forward(self, images: Any) -> Dict[str, Any]:
            torch_module = _require_torch()
            if not torch_module.is_tensor(images) or images.ndim != 4:
                raise PixelConflictError(
                    "reference forward needs an N x 3 x H x W tensor"
                )
            self.calls += 1
            with torch_module.no_grad():
                output = self.embedder(images)
                features = self.embedder.feature
            if features is None:
                raise PixelConflictError(
                    "classifier[3] tap captured no feature; the hook was removed"
                )
            logits = output["semantic_logits"]
            if tuple(logits.shape[-2:]) != tuple(images.shape[-2:]):
                raise PixelConflictError(
                    "semantic logits are not upsampled to the input size: "
                    f"{tuple(logits.shape[-2:])} vs {tuple(images.shape[-2:])}"
                )
            if int(features.shape[1]) != FEATURE_CHANNELS:
                raise PixelConflictError(
                    f"tap has {int(features.shape[1])} channels, expected "
                    f"{FEATURE_CHANNELS}"
                )
            return {
                "features": features.detach(),
                "semantic_logits": logits.detach(),
                "semantic": logits.detach().argmax(dim=1),
            }

        def close(self) -> None:
            closer = getattr(self.embedder, "close", None)
            if callable(closer):
                closer()


def _state_digest(state: Mapping[str, Any]) -> str:
    torch_module = _require_torch()
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch_module.is_tensor(value):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_reference_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Strict geometry/semantic guards for the frozen reference checkpoint."""

    if not isinstance(payload, Mapping):
        raise PixelConflictError("reference payload must be a mapping")
    if payload.get("schema") != joint.CHECKPOINT_SCHEMA:
        raise PixelConflictError("reference schema is not the final joint checkpoint")
    if payload.get("kind") != "joint_semantic_offsets":
        raise PixelConflictError(f"reference kind mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise PixelConflictError("reference must be the final-epoch-only checkpoint")
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        raise PixelConflictError("reference geometry missing")
    for key, expected in _REQUIRED_GEOMETRY.items():
        actual = geometry.get(key)
        if isinstance(expected, int):
            if int(actual if actual is not None else -1) != expected:
                raise PixelConflictError(
                    f"reference geometry {key}={actual!r}, expected {expected!r}"
                )
        elif str(actual) != str(expected):
            raise PixelConflictError(
                f"reference geometry {key}={actual!r}, expected {expected!r}"
            )
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping):
        raise PixelConflictError("reference semantic block missing")
    if semantic.get("joint_trained") is not True:
        raise PixelConflictError("reference semantic block is not joint-trained")
    inference = payload.get("inference")
    if not isinstance(inference, Mapping):
        raise PixelConflictError("reference inference block missing")
    if int(inference.get("tile_size", -1)) != joint.TILE_SIZE:
        raise PixelConflictError("reference inference tile size mismatch")
    if int(inference.get("tile_stride", -1)) != joint.TILE_STRIDE:
        raise PixelConflictError("reference inference tile stride mismatch")
    return dict(payload)


def reference_metadata(payload: Mapping[str, Any], *, reference: Any = None) -> Dict[str, Any]:
    """Frozen-reference metadata, including what freezing does *not* guarantee."""

    geometry = dict(payload.get("geometry") or {})
    semantic = dict(payload.get("semantic") or {})
    training = dict(payload.get("training") or {})
    inference = dict(payload.get("inference") or {})
    metadata = {
        "schema": "cadvec.recognition.pixel_conflict_reference/1",
        "checkpoint_schema": payload.get("schema"),
        "checkpoint_kind": payload.get("kind"),
        "epoch": payload.get("epoch"),
        "epochs": payload.get("epochs"),
        "final_epoch_only": payload.get("final_epoch_only"),
        "geometry": {
            key: geometry.get(key)
            for key in (
                "crop_size",
                "tile_size",
                "tile_stride",
                "tap",
                "tap_channels",
                "num_classes",
            )
        },
        "semantic": {
            key: semantic.get(key)
            for key in (
                "path",
                "schema",
                "role",
                "epoch",
                "epochs",
                "num_classes",
                "initial_digest",
                "joint_trained",
                "batch_norm_running_stats_frozen",
            )
        },
        "training": {
            key: training.get(key)
            for key in ("base_seed", "epoch_seed_stride", "head_init_seed", "epochs")
        },
        "inference": {
            key: inference.get(key)
            for key in ("tile_size", "tile_stride", "white_pad", "resize")
        },
        "frozen": {
            "parameters_requires_grad": False,
            "module_mode": "eval",
            "batch_norm_running_stats": "frozen (eval)",
            "forward": "torch.no_grad",
        },
        "frozen_does_not_imply": (
            "parameters and BN statistics are frozen, but the production semantic "
            "reference is the 384/192 overlap-averaged map; a single 256 px crop "
            "has a different context and is only a disclosed proxy"
        ),
        "conditioner_consistency": (
            "at fit the judged class is the argmax of the same local 256 px frozen "
            "logits block, so conditioner and logits agree by construction; at "
            "apply the judged class is the saved native 384/192 map while the "
            "logits block is the local 256 forward, so the pair can be inconsistent "
            "(a conditioner class the local logits do not support)"
        ),
    }
    if reference is not None:
        metadata["model_digest"] = str(getattr(reference, "model_digest", ""))
        metadata["frozen_head_digest"] = str(getattr(reference, "head_digest", ""))
    return metadata


def load_conflict_reference(
    checkpoint_path: str = DEFAULT_JOINT_CHECKPOINT,
    *,
    device: str = "cpu",
    model_factory: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the frozen final joint checkpoint as a conflict feature reference.

    ``model_factory`` exists only for synthetic tests (the real lane gets
    DeepLabV3-ResNet50 from :func:`joint.load_joint_embedder` without network
    access).  No shared source is modified and no head is trained here.
    """

    if not os.path.isfile(checkpoint_path):
        raise PixelConflictError(f"reference_checkpoint_not_found: {checkpoint_path}")
    payload = joint.load_joint_checkpoint(checkpoint_path, device=device)
    payload = _validate_reference_payload(payload)
    embedder, _loaded = joint.load_joint_embedder(
        checkpoint_path, device=device, model_factory=model_factory
    )
    reference = FrozenJointReference(embedder, payload, device=device)
    reference.eval()
    report = reference_metadata(payload, reference=reference)
    report["checkpoint_path"] = os.path.abspath(checkpoint_path)
    return reference, report


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def licensed_targets(
    judged_class: Any,
    gt_semantic: Any,
    semantic_valid: Any,
    ink: Any,
) -> Dict[str, Any]:
    """Binary conflict target on licensed pixels only.

    Licensed means: ``semantic_valid`` true, GT class in ``0..34``, PNG ink, and a
    judged class in ``0..34``.  ``target = 1`` iff ``judged_class != gt_class``.
    Instance labels are not an input: they are neither required nor consulted.
    """

    numpy = _require_numpy()
    judged = _as_class_array(judged_class, "judged_class")
    gt = _as_class_array(gt_semantic, "gt_semantic")
    if gt.shape != judged.shape:
        raise PixelConflictError(
            f"gt_semantic shape {gt.shape} does not match judged {judged.shape}"
        )
    shape = (int(gt.shape[0]), int(gt.shape[1]))
    valid = _as_bool_mask(semantic_valid, "semantic_valid", shape)
    ink_mask = _as_bool_mask(ink, "ink", shape)
    if int(gt.min()) < -1 or int(gt.max()) > NUM_CLASSES:
        raise PixelConflictError(
            f"gt_semantic classes must be -1..{NUM_CLASSES}; "
            f"got {int(gt.min())}..{int(gt.max())}"
        )
    if int(judged.min()) < 0 or int(judged.max()) > NUM_CLASSES:
        raise PixelConflictError(
            f"judged_class must be 0..{NUM_CLASSES}; "
            f"got {int(judged.min())}..{int(judged.max())}"
        )

    gt_known = (gt >= 0) & (gt < NUM_CLASSES)
    judged_known = (judged >= 0) & (judged < NUM_CLASSES)
    mask = valid & gt_known & ink_mask & judged_known
    target = np.zeros(shape, dtype=np.float32)
    if bool(mask.any()):
        target[mask] = (judged[mask] != gt[mask]).astype(np.float32)
    licensed = int(mask.sum())
    conflict = int(target.sum())
    return {
        "mask": mask,
        "target": target,
        "licensed_pixels": licensed,
        "conflict_pixels": conflict,
        "agree_pixels": int(licensed - conflict),
        "excluded_unknown_semantic": int((~valid).sum()),
        "excluded_gt_unknown": int((valid & ~gt_known).sum()),
        "excluded_outside_ink": int((valid & gt_known & ~ink_mask).sum()),
        "excluded_no_judged_class": int(
            (valid & gt_known & ink_mask & ~judged_known).sum()
        ),
        "instance_labels_used": False,
    }


# ---------------------------------------------------------------------------
# loss (unweighted BCEWithLogits, window-normalised)
# ---------------------------------------------------------------------------


def conflict_bce_sum(logits: Any, target: Any, mask: Any) -> Tuple[Any, int]:
    """Sum of unweighted BCEWithLogits over ``mask``, plus the pixel count.

    ``logits`` is ``N x 1 x H x W`` (or ``1 x H x W``).  The per-pixel loss is
    multiplied by the mask, so masked-out pixels contribute neither loss nor
    gradient (they are never negatives).  An empty mask yields an exact zero that
    still back-propagates zeros.
    """

    torch_module = _require_torch()
    numpy = _require_numpy()
    if not torch_module.is_tensor(logits):
        raise PixelConflictError("logits must be a torch tensor")
    field = logits
    if field.ndim == 3:
        field = field.unsqueeze(0)
    if field.ndim != 4 or int(field.shape[1]) != 1:
        raise PixelConflictError(
            f"conflict logits must be N x 1 x H x W; got {tuple(field.shape)}"
        )
    if field.shape[0] != 1:
        raise PixelConflictError("conflict loss takes one map at a time")
    field = field[0, 0]
    if not bool(torch_module.isfinite(field).all()):
        raise PixelConflictError("non_finite_conflict_logits")
    mask_array = np.asarray(mask)
    if mask_array.dtype != bool:
        raise PixelConflictError("mask must be a bool array")
    if tuple(mask_array.shape) != tuple(field.shape[-2:]):
        raise PixelConflictError(
            f"mask shape {tuple(mask_array.shape)} does not match logits "
            f"{tuple(field.shape[-2:])}"
        )
    target_array = np.asarray(target)
    if tuple(target_array.shape) != tuple(field.shape[-2:]):
        raise PixelConflictError("target shape does not match the mask")
    if mask_array.any():
        values = target_array[mask_array]
        if np.issubdtype(values.dtype, np.bool_):
            values = values.astype(np.float32)
        if not bool(np.isfinite(values).all()) or not bool(
            np.isin(values, (0.0, 1.0)).all()
        ):
            raise PixelConflictError("target must be binary 0/1 inside the mask")
    mask_tensor = torch_module.from_numpy(mask_array).to(device=field.device)
    target_tensor = torch_module.as_tensor(
        target_array.astype(np.float32, copy=False), device=field.device
    )
    per_pixel = torch_module.nn.functional.binary_cross_entropy_with_logits(
        field, target_tensor, reduction="none"
    )
    total = (per_pixel * mask_tensor.to(dtype=per_pixel.dtype)).sum()
    return total, int(mask_array.sum())


def window_denominator(masks: Iterable[Any]) -> int:
    """Licensed pixel mass of one accumulation window (no forward needed)."""

    numpy = _require_numpy()
    total = 0
    for mask in masks:
        total += int(np.asarray(mask, dtype=bool).sum())
    return total


# ---------------------------------------------------------------------------
# streaming accumulation
# ---------------------------------------------------------------------------


def _as_indicative_crop(crop: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(crop, Mapping):
        raise PixelConflictError("each crop must be a mapping")
    if "image" not in crop or "gt_semantic" not in crop or "semantic_valid" not in crop:
        raise PixelConflictError(
            "crop needs image, gt_semantic and semantic_valid"
        )
    image = as_unit_chw(crop["image"])
    gt = _as_class_array(crop["gt_semantic"], "gt_semantic")
    shape = (int(gt.shape[0]), int(gt.shape[1]))
    if tuple(image.shape[-2:]) != shape:
        raise PixelConflictError("crop image and labels have different shapes")
    valid = _as_bool_mask(crop["semantic_valid"], "semantic_valid", shape)
    ink = crop.get("ink")
    ink_mask = (
        image_ink(image)
        if ink is None
        else _as_bool_mask(ink, "ink", shape)
    )
    judged = crop.get("judged_class")
    if judged is not None:
        judged = _as_class_array(judged, "judged_class")
        if judged.shape != shape:
            raise PixelConflictError("judged_class shape mismatch")
    return {
        "image": image,
        "gt_semantic": gt,
        "semantic_valid": valid,
        "ink": ink_mask,
        "judged_class": judged,
        "sample_id": str(crop.get("sample_id", "")),
    }


def run_conflict_window(
    reference: Any,
    head: Any,
    crops: Sequence[Mapping[str, Any]],
    *,
    optimizer: Any = None,
    accumulation_steps: int = ACCUMULATION_STEPS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """One accumulation window: masks first, then forward/scale/backward per crop.

    The denominator is the licensed pixel mass of the whole window and is computed
    from the masks **before any forward**.  Crops with zero licensed pixels are
    skipped without a forward.  Each supervised crop is scaled by
    ``1 / window_denominator`` and backwarded immediately, so the total gradient
    equals a single normalised window loss and only one graph is live at a time.
    """

    torch_module = _require_torch()
    steps = _positive_int("accumulation_steps", int(accumulation_steps))
    batch = [_as_indicative_crop(crop) for crop in crops]
    if len(batch) != steps:
        raise PixelConflictError(
            f"a window needs exactly {steps} crops; got {len(batch)}"
        )
    masks: List[Any] = []
    for crop in batch:
        judged = crop["judged_class"]
        if judged is None:
            # Licensing requires a judged class in 0..34, so the mask does depend
            # on the judged map.  A placeholder that is *known* everywhere can only
            # over-license, and the equality check after the forward rejects any
            # drift, so the denominator pass is safe without a trunk forward.
            placeholder = np.zeros(crop["gt_semantic"].shape, dtype=np.int64)
            prepared = licensed_targets(
                placeholder, crop["gt_semantic"], crop["semantic_valid"], crop["ink"]
            )
        else:
            prepared = licensed_targets(
                judged, crop["gt_semantic"], crop["semantic_valid"], crop["ink"]
            )
        masks.append(prepared["mask"])
    denominator = window_denominator(masks)
    provided = sum(1 for crop in batch if crop["judged_class"] is not None)
    if provided == 0:
        judged_source = "local_argmax"
    elif provided == len(batch):
        judged_source = "provided"
    else:
        # Mixed windows are named explicitly instead of silently reported as
        # "provided": some crops were conditioned on the local argmax and some on
        # a caller map, which is a different conditioning population.
        judged_source = "mixed"
    stats = {
        "crops_scheduled": len(batch),
        "crops_skipped_zero_supervision": 0,
        "crops_forwarded": 0,
        "licensed_pixels": int(denominator),
        "conflict_pixels": 0,
        "agree_pixels": 0,
        "loss_sum": 0.0,
        "window_denominator": int(denominator),
        "optimizer_steps": 0,
        "judged_source": judged_source,
    }
    if denominator == 0:
        # Nothing in this window is licensed, so every crop is skipped; report the
        # whole batch rather than leaving the counter at zero.
        stats["crops_skipped_zero_supervision"] = len(batch)
        stats["loss_window_mean"] = 0.0
        return stats

    reference.eval()
    head.train()
    touched = False
    scale = 1.0 / float(denominator)
    try:
        for crop, mask in zip(batch, masks):
            licensed = int(mask.sum())
            if licensed == 0:
                stats["crops_skipped_zero_supervision"] += 1
                continue
            images = crop["image"].unsqueeze(0).to(device=device)
            output = reference(images)
            features = output["features"]
            logits = output["semantic_logits"]
            judged = crop["judged_class"]
            if judged is None:
                judged = output["semantic"][0].detach().to("cpu").numpy().astype(np.int64)
            prepared = licensed_targets(
                judged, crop["gt_semantic"], crop["semantic_valid"], crop["ink"]
            )
            if not bool(np.array_equal(prepared["mask"], mask)):
                raise PixelConflictError(
                    "licensed mask changed between the denominator pass and the "
                    "forward; the judged class must not affect licensing"
                )
            inputs = head_inputs(
                features,
                logits,
                torch_module.as_tensor(crop["ink"], device=device).unsqueeze(0),
                torch_module.as_tensor(judged, device=device).unsqueeze(0),
            )
            full_logits = head_logits(
                head, inputs, output_size=tuple(int(v) for v in crop["image"].shape[-2:])
            )
            total, count = conflict_bce_sum(full_logits, prepared["target"], mask)
            (total * scale).backward()
            touched = True
            stats["crops_forwarded"] += 1
            stats["conflict_pixels"] += int(prepared["conflict_pixels"])
            stats["agree_pixels"] += int(prepared["agree_pixels"])
            stats["loss_sum"] += float(total.detach().item())
        if touched and optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            stats["optimizer_steps"] = 1
    finally:
        reference.eval()
    stats["loss_window_mean"] = float(stats["loss_sum"]) * scale
    return stats


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def _check_accumulation_bytes(height: int, width: int) -> None:
    # conflict logits (1) + weight (1) float32
    needed = 2 * int(height) * int(width) * 4
    if needed > MAX_LOGIT_ACCUM_BYTES:
        raise PixelConflictError(
            f"conflict accumulation too large: {width}x{height} needs "
            f"{needed} bytes; the guard is {MAX_LOGIT_ACCUM_BYTES}"
        )


def predict_conflict(
    reference: Any,
    head: Any,
    image: Any,
    *,
    judged_class: Any,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    device: str = "cpu",
    ink: Any = None,
) -> Dict[str, Any]:
    """Overlap-averaged conflict probability for one image (PNG + saved map only).

    ``judged_class`` is the **saved native joint semantic map** (the 384/192
    averaged CC8 result), cropped exactly to each 256 tile grid; it is not the
    local argmax, so a real run must pass it explicitly.  Tile logits are averaged
    with the same white padding rule as the joint lane and exactly **one** sigmoid
    is applied after the mean.  Each tile's head logits are bilinearly upsampled
    from the classifier[3] tap stride to the tile before the mean, because the
    production tap is stride 8 while the contract is per-pixel.  The returned
    probability is exactly 0 outside the PNG ink mask, so the abstain rule can
    never touch a non-ink pixel.  No ground truth is read here.
    """

    torch_module = _require_torch()
    source = as_unit_chw(image)
    tile_size = _positive_int("tile_size", int(tile_size))
    stride = _positive_int("stride", int(stride))
    if stride > tile_size:
        raise PixelConflictError("stride must not exceed tile size")
    height, width = int(source.shape[-2]), int(source.shape[-1])
    judged = _as_class_array(judged_class, "judged_class")
    if tuple(judged.shape) != (height, width):
        raise PixelConflictError(
            f"judged_class shape {tuple(judged.shape)} must equal the image "
            f"({height}, {width}); it is cropped, never rescaled"
        )
    ink_mask = image_ink(source) if ink is None else _as_bool_mask(
        ink, "ink", (height, width)
    )

    padded_height = max(tile_size, height)
    padded_width = max(tile_size, width)
    _check_accumulation_bytes(padded_height, padded_width)
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h or pad_w:
        padded = F.pad(
            source.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=1.0
        ).squeeze(0)
        judged_padded = np.full((padded_height, padded_width), IGNORE_CLASS, dtype=np.int64)
        judged_padded[:height, :width] = judged
        ink_padded = np.zeros((padded_height, padded_width), dtype=bool)
        ink_padded[:height, :width] = ink_mask
    else:
        padded = source
        judged_padded = judged
        ink_padded = ink_mask

    rows = fullfit._tile_positions(padded_height, tile_size, stride)
    columns = fullfit._tile_positions(padded_width, tile_size, stride)
    accum = torch_module.zeros((1, padded_height, padded_width), dtype=torch_module.float32)
    weights = torch_module.zeros((1, padded_height, padded_width), dtype=torch_module.float32)
    started = time.monotonic()
    reference.eval()
    head.eval()
    try:
        with torch_module.no_grad():
            for row in rows:
                for column in columns:
                    window = padded[
                        :, row : row + tile_size, column : column + tile_size
                    ].unsqueeze(0).to(device=device)
                    judged_tile = judged_padded[
                        row : row + tile_size, column : column + tile_size
                    ]
                    ink_tile = torch_module.as_tensor(
                        ink_padded[row : row + tile_size, column : column + tile_size],
                        device=device,
                    ).unsqueeze(0)
                    output = reference(window)
                    inputs = head_inputs(
                        output["features"],
                        output["semantic_logits"],
                        ink_tile,
                        torch_module.as_tensor(judged_tile, device=device).unsqueeze(0),
                    )
                    tile_logits = head_logits(
                        head, inputs, output_size=(tile_size, tile_size)
                    ).detach().to("cpu")
                    if tuple(tile_logits.shape[-2:]) != (tile_size, tile_size):
                        raise PixelConflictError(
                            f"tile logits shape {tuple(tile_logits.shape)} does not "
                            f"match the {tile_size} px tile"
                        )
                    accum[
                        :, row : row + tile_size, column : column + tile_size
                    ] += tile_logits[0]
                    weights[
                        :, row : row + tile_size, column : column + tile_size
                    ] += 1.0
    finally:
        head.eval()
    mean_logits = (accum / weights)[:, :height, :width].contiguous()
    if not bool(torch_module.isfinite(mean_logits).all()):
        raise PixelConflictError("non_finite_conflict_logits")
    raw_probability = torch_module.sigmoid(mean_logits)[0].numpy()
    probability = np.array(raw_probability, dtype=np.float32, copy=True)
    # The head is a per-pixel classifier but the contract is "abstain ink only":
    # probability outside the PNG ink is forced to exactly 0 so the 0.5 rule can
    # never touch a non-ink pixel, whatever the head predicts there.  The raw
    # logits stay unmasked so the arithmetic remains auditable.
    outside_ink = int((~ink_mask).sum())
    if outside_ink:
        probability[~ink_mask] = 0.0
    seconds = round(time.monotonic() - started, 6)
    return {
        "conflict_logits": mean_logits[0].numpy().astype(np.float32, copy=False),
        "conflict_prob": probability.astype(np.float32, copy=False),
        "tile_rows": len(rows),
        "tile_columns": len(columns),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "padded": bool(pad_h or pad_w),
        "source_size": (int(height), int(width)),
        "judged_source": "provided_saved_native_map",
        "judged_note": (
            "judged_class was supplied by the caller (the saved 384/192 averaged "
            "semantic map); it is cropped to each tile, never rescaled"
        ),
        "scored_pixels": "all PNG ink pixels (semantic_valid does not exist at "
        "inference; this extrapolates beyond the training mask)",
        "ink_pixels": int(ink_mask.sum()),
        "prob_masked_outside_ink": outside_ink,
        "prob_masking": (
            "conflict_prob is exactly 0 outside the PNG ink mask; conflict_logits "
            "is the raw overlap mean and is not masked"
        ),
        "timings": {"forward_seconds": seconds},
    }


def apply_abstain(
    semantic: Any,
    conflict_prob: Any,
    *,
    threshold: float = CONFLICT_THRESHOLD,
    ink: Any = None,
) -> Dict[str, Any]:
    """Replace conflict pixels by the ignore class; nothing else changes.

    ``prob >= threshold`` (inclusive at exactly 0.5) marks the pixel abstained.
    No size/min-area filter, no added pixel, no relabelled pixel: the result is a
    strict subset of the input ink and every kept label is identical.
    """

    numpy = _require_numpy()
    labels = _as_class_array(semantic, "semantic")
    probabilities = numpy.asarray(conflict_prob, dtype=numpy.float32)
    if probabilities.shape != labels.shape:
        raise PixelConflictError(
            f"conflict_prob shape {probabilities.shape} does not match semantic "
            f"{labels.shape}"
        )
    if not bool(numpy.isfinite(probabilities).all()):
        raise PixelConflictError("non_finite_conflict_prob")
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise PixelConflictError("conflict_prob must lie in [0, 1]")
    if int(labels.min()) < 0 or int(labels.max()) > NUM_CLASSES:
        raise PixelConflictError(
            f"semantic classes must be 0..{NUM_CLASSES}; "
            f"got {int(labels.min())}..{int(labels.max())}"
        )
    boundary = float(threshold)
    if not math.isfinite(boundary) or boundary <= 0.0 or boundary >= 1.0:
        raise PixelConflictError(
            f"threshold must be inside (0, 1); got {threshold!r}"
        )
    abstain = probabilities >= boundary
    ink_mask = None
    if ink is not None:
        ink_mask = _as_bool_mask(ink, "ink", (int(labels.shape[0]), int(labels.shape[1])))
        outside = int((abstain & ~ink_mask).sum())
        if outside:
            raise PixelConflictError(
                f"{outside} abstained pixels lie outside PNG ink; the result must "
                "stay a subset of the input ink"
            )
    output = numpy.array(labels, copy=True)
    output[abstain] = IGNORE_CLASS
    kept = ~abstain
    if not bool(numpy.array_equal(output[kept], labels[kept])):
        raise PixelConflictError("kept labels changed; abstention must not relabel")
    return {
        "semantic": output,
        "abstained": abstain,
        "stats": {
            "threshold": boundary,
            "boundary": "inclusive (prob >= threshold)",
            "abstained_pixels": int(abstain.sum()),
            "kept_pixels": int(kept.sum()),
            "ink_pixels": int(ink_mask.sum()) if ink_mask is not None else int(labels.size),
            "pixels_added": 0,
            "pixels_relabelled": 0,
            "subset_of_ink": ink_mask is not None,
            "subset_of_ink_note": (
                "verified against the supplied ink mask"
                if ink_mask is not None
                else "no ink mask was supplied, so the ink-subset property is not "
                "verified here; pass ink= to assert it"
            ),
            "same_cc8_and_thresholds": True,
        },
    }


def conflict_mask(
    judged_semantic: Any,
    kept_semantic: Any,
    probability: Any,
    *,
    threshold: float = CONFLICT_THRESHOLD,
) -> Any:
    """Audit view: which pixels were dropped, and whether the drop is consistent.

    Returns the boolean drop mask (``probability >= threshold``) and checks the
    invariant a consumer relies on: every other pixel is bit-identical between the
    judged map and the kept map, no pixel gains a class, and nothing outside the
    drop set becomes the ignore class.
    """

    numpy = _require_numpy()
    judged = _as_class_array(judged_semantic, "judged_semantic")
    kept = _as_class_array(kept_semantic, "kept_semantic")
    if judged.shape != kept.shape:
        raise PixelConflictError("judged and kept semantic shapes differ")
    probabilities = numpy.asarray(probability, dtype=numpy.float32)
    if probabilities.shape != judged.shape:
        raise PixelConflictError("conflict_prob shape does not match the map")
    if not bool(numpy.isfinite(probabilities).all()):
        raise PixelConflictError("non_finite_conflict_prob")
    boundary = float(threshold)
    if not math.isfinite(boundary) or boundary <= 0.0 or boundary >= 1.0:
        raise PixelConflictError(f"threshold must be inside (0, 1); got {threshold!r}")
    dropped = probabilities >= boundary
    if not bool(numpy.array_equal(kept[~dropped], judged[~dropped])):
        raise PixelConflictError("kept pixels changed class")
    if bool((kept[dropped] != IGNORE_CLASS).any()):
        raise PixelConflictError("dropped pixels are not the ignore class")
    if int(kept.min()) < 0 or int(kept.max()) > NUM_CLASSES:
        raise PixelConflictError(f"kept classes must be 0..{NUM_CLASSES}")
    return dropped


# ---------------------------------------------------------------------------
# checkpoint (head only)
# ---------------------------------------------------------------------------


def _require_bound_reference(reference: Any, *, where: str) -> Dict[str, Any]:
    """A head checkpoint must carry a joint-format reference declaration.

    The check proves **format compatibility** only: the declared
    ``checkpoint_schema`` must equal ``joint.CHECKPOINT_SCHEMA``.  It does *not*
    uniquely identify the trained reference, because that schema is shared by every
    joint checkpoint.  The fit driver must therefore load the specified
    ``DEFAULT_JOINT_CHECKPOINT`` path explicitly (and a real run must record that
    path); matching this schema is not evidence that a different checkpoint was
    never used.  Absent, empty, schema-less or different-schema metadata is refused
    on **both** save and load, so a checkpoint can never be written in a state its
    own loader rejects.
    """

    if reference is None:
        raise PixelConflictError(f"{where}: reference metadata is absent")
    if not isinstance(reference, Mapping):
        raise PixelConflictError(f"{where}: reference metadata must be a mapping")
    if not reference:
        raise PixelConflictError(f"{where}: reference metadata is empty")
    declared = reference.get("checkpoint_schema")
    if declared is None:
        raise PixelConflictError(
            f"{where}: reference metadata must declare checkpoint_schema="
            f"{joint.CHECKPOINT_SCHEMA!r}"
        )
    if str(declared) != joint.CHECKPOINT_SCHEMA:
        raise PixelConflictError(
            f"{where}: reference.checkpoint_schema={declared!r}, expected "
            f"{joint.CHECKPOINT_SCHEMA!r}; that is an incompatible joint checkpoint "
            "format"
        )
    return dict(reference)


def save_conflict_checkpoint(
    path: str,
    head: Any,
    *,
    epoch: int,
    epochs: int = EPOCHS,
    reference_meta: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Head-only checkpoint carrying a joint-format reference declaration.

    ``reference_meta`` is mandatory and must declare
    ``checkpoint_schema == joint.CHECKPOINT_SCHEMA``: a head with no reference
    declaration would be unloadable by this module's own strict loader, so it is
    refused at save time rather than written.  The declaration is a format check,
    not a unique identity — it is shared by every joint checkpoint, so the fit
    driver must load the specified ``DEFAULT_JOINT_CHECKPOINT`` path explicitly
    and record that path alongside the head.
    """

    torch_module = _require_torch()
    reference = _require_bound_reference(reference_meta, where="reference_meta")
    try:
        actual_in = int(head[0].in_channels)
        actual_hidden = int(head[-1].in_channels)
    except (AttributeError, IndexError, TypeError):
        actual_in = head_in_channels()
        actual_hidden = HEAD_HIDDEN_CHANNELS
    parameters = int(sum(parameter.numel() for parameter in head.parameters()))
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": VERSION,
        "kind": "pixel_semantic_conflict",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_epoch_only": True,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "geometry": {
            "crop_size": CROP_SIZE,
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
            "feature_stride": FEATURE_STRIDE,
            "num_classes": NUM_CLASSES,
            "ignore_class": IGNORE_CLASS,
            "tap": "net.classifier[3]",
            "tap_channels": FEATURE_CHANNELS,
            "head_in_channels": int(actual_in),
            "head_hidden_channels": int(actual_hidden),
            "head_parameters": int(parameters),
            "conditioner": "one-hot judged class (35) at tap resolution",
        },
        "training": {
            "epochs": int(epochs),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "accumulation_steps": ACCUMULATION_STEPS,
            "base_seed": BASE_SEED,
            "epoch_seed_stride": EPOCH_SEED_STRIDE,
            "head_init_seed": HEAD_INIT_SEED,
            "loss": "unweighted BCEWithLogits / licensed pixel mass",
            "trunk": "frozen (no grad, eval)",
            "amp": False,
            "final_epoch_only": True,
        },
        "threshold": {
            "conflict_probability": CONFLICT_THRESHOLD,
            "boundary": "inclusive",
            "post_filter": "none (no size/min-area)",
        },
        "reference": reference,
        "metrics": dict(metrics or {}),
        "head_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in head.state_dict().items()
        },
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch_module.save(payload, path)
    return payload


def load_conflict_checkpoint(path: str, *, device: str = "cpu") -> Dict[str, Any]:
    """Strict loader: schema, kind, geometry and reference block are checked."""

    torch_module = _require_torch()
    if not os.path.isfile(path):
        raise PixelConflictError(f"checkpoint_not_found: {path}")
    try:
        payload = torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6
        payload = torch_module.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise PixelConflictError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("kind") != "pixel_semantic_conflict":
        raise PixelConflictError(f"checkpoint_kind_mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise PixelConflictError("checkpoint_must_be_final_epoch_only")
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        raise PixelConflictError("checkpoint_missing_geometry")
    for key, expected in (
        ("crop_size", CROP_SIZE),
        ("tile_size", TILE_SIZE),
        ("tile_stride", TILE_STRIDE),
        ("num_classes", NUM_CLASSES),
        ("tap", "net.classifier[3]"),
        ("head_in_channels", head_in_channels()),
    ):
        actual = geometry.get(key)
        if str(actual) != str(expected):
            raise PixelConflictError(
                f"checkpoint geometry {key}={actual!r}, expected {expected!r}"
            )
    # The hidden width is a recorded degree of freedom, not a constant: a
    # synthetic or reduced-width checkpoint must load as long as it is a positive
    # integer.  Only a missing/non-positive value is refused.
    hidden_raw = geometry.get("head_hidden_channels")
    if isinstance(hidden_raw, bool) or not isinstance(hidden_raw, (int, np.integer)):
        raise PixelConflictError(
            "checkpoint geometry head_hidden_channels="
            f"{hidden_raw!r} is not an integer"
        )
    hidden = _positive_int("checkpoint geometry head_hidden_channels", int(hidden_raw))
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise PixelConflictError("checkpoint_missing_training_block")
    seed_raw = training.get("head_init_seed")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, (int, np.integer)):
        raise PixelConflictError(
            f"checkpoint training head_init_seed={seed_raw!r} is not an integer"
        )
    seed = int(seed_raw)
    if seed < 0:
        raise PixelConflictError(
            f"checkpoint training.head_init_seed={seed} must not be negative"
        )
    state = payload.get("head_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise PixelConflictError("checkpoint_missing_head_state")
    reference = payload.get("reference")
    if not isinstance(reference, Mapping):
        raise PixelConflictError("checkpoint_missing_reference_metadata")
    reference = _require_bound_reference(reference, where="checkpoint")
    result = dict(payload)
    result["geometry"] = dict(geometry)
    result["geometry"]["head_hidden_channels"] = int(hidden)
    result["training"] = dict(training)
    result["head_state_dict"] = {key: value.to(device) for key, value in state.items()}
    return result


def load_conflict_head(
    checkpoint_path: str, *, device: str = "cpu"
) -> Tuple[Any, Dict[str, Any]]:
    payload = load_conflict_checkpoint(checkpoint_path, device=device)
    head = build_conflict_head(
        in_channels=int(payload["geometry"]["head_in_channels"]),
        hidden_channels=int(payload["geometry"]["head_hidden_channels"]),
        seed=int(payload["training"]["head_init_seed"]),
        device="cpu",
    )
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.to(device)
    head.eval()
    return head, payload


# ---------------------------------------------------------------------------
# persistence (PNG-only artifacts; no ground truth may be read here)
# ---------------------------------------------------------------------------


def conflict_slug(sample_id: Any) -> str:
    """Persisted-file stem for one sample, matching the evaluator's rule.

    The evaluator derives ``<slug>`` as ``sample_id.replace(":", "_")`` and, more
    generally, replaces every character that is not alphanumeric, ``-`` or ``_``.
    The same rule is applied here so the two sides cannot disagree; the result is
    also a safe single path component (no separator, no ``..``).
    """

    text = str(sample_id).strip()
    if not text:
        raise PixelConflictError("sample_id must be a non-empty string")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    if not slug or set(slug) == {"_"}:
        raise PixelConflictError(f"sample_id {sample_id!r} has no usable slug")
    return slug


def _atomic_write_bytes(path: Path, data: bytes) -> int:
    """Write ``data`` to ``path`` through a same-directory temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only on a write failure
            try:
                temporary.unlink()
            except OSError:
                pass
    return len(data)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> int:
    text = json.dumps(
        _jsonable(payload), ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False
    )
    return _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_npz(path: Path, arrays: Mapping[str, Any]) -> int:
    numpy = _require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            numpy.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only on a write failure
            try:
                temporary.unlink()
            except OSError:
                pass
    return path.stat().st_size


def _jsonable(value: Any, where: str = "payload") -> Any:
    """Strict JSON projection of metadata, refusing what JSON cannot carry safely.

    ``Path`` objects (including nested ones, e.g. a reference checkpoint path)
    become strings, numpy scalars become Python scalars and arrays become lists, so a
    declared reference path cannot be dropped or mangled on the way to disk.  A
    non-finite float, a set, bytes or any other unknown object is refused rather than
    serialised into an artifact the evaluator cannot read.
    """

    numpy = _require_numpy()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PixelConflictError(f"{where}: {value!r} is not a finite JSON number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numpy.generic):
        return _jsonable(value.item(), where)
    if isinstance(value, numpy.ndarray):
        if value.dtype == object:
            raise PixelConflictError(f"{where}: object arrays are not supported metadata")
        return _jsonable(value.tolist(), where)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise PixelConflictError(
        f"{where}: unsupported metadata value of type {type(value).__name__}"
    )


def _resolve_directory(directory: Any) -> Path:
    """Resolve the artifact root before any path is built from it.

    ``None``, an empty or whitespace-only string and a non-path type are refused
    here, so a typo cannot collapse to the current working directory and write
    artifacts beside the checkout.
    """

    if directory is None:
        raise PixelConflictError("output directory is required; got None")
    if isinstance(directory, (str, os.PathLike)):
        text = str(directory).strip()
        if not text:
            raise PixelConflictError("output directory must be a non-empty path")
        return Path(text)
    raise PixelConflictError(
        f"output directory must be a path, not {type(directory).__name__}"
    )


def _prepare_target(directory: Any, subdir: Optional[str], name: str, *, overwrite: bool) -> Path:
    """Resolve one artifact path and refuse to clobber an existing artifact."""

    root = _resolve_directory(directory)
    target = root / subdir / name if subdir else root / name
    if not overwrite and target.exists():
        raise PixelConflictError(
            f"{target} already exists; persistence is write-once, so pass "
            "overwrite=True only for a deliberate re-run"
        )
    return target


def _stamp(created_utc: Optional[str]) -> str:
    if created_utc is None:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = str(created_utc).strip()
    if not text:
        raise PixelConflictError("created_utc must be a non-empty string when given")
    return text


def _as_probability(values: Any) -> Any:
    """One finite ``H x W`` conflict-probability array inside ``[0, 1]``."""

    numpy = _require_numpy()
    array = numpy.asarray(values)
    if array.ndim != 2:
        raise PixelConflictError(f"probability must be 2-D H x W; got {array.shape}")
    if array.dtype == bool or not numpy.issubdtype(array.dtype, numpy.number):
        raise PixelConflictError(f"probability must be numeric; got dtype {array.dtype}")
    if numpy.issubdtype(array.dtype, numpy.complexfloating):
        raise PixelConflictError("probability must be real-valued")
    probability = array.astype(numpy.float32, copy=False)
    if not bool(numpy.isfinite(probability).all()):
        raise PixelConflictError("probability contains non-finite values")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise PixelConflictError(
            f"probability must lie in [0, 1]; got {float(probability.min())}"
            f"..{float(probability.max())}"
        )
    return probability


def _require_same_shape(name: str, array: Any, shape: Tuple[int, int]) -> Any:
    if tuple(array.shape) != tuple(shape):
        raise PixelConflictError(f"{name} shape {tuple(array.shape)} does not match {tuple(shape)}")
    return array


def _sha256_array(array: Any) -> str:
    numpy = _require_numpy()
    digest = hashlib.sha256()
    data = numpy.ascontiguousarray(array)
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def write_conflict_maps(
    directory: Any,
    sample_id: Any,
    *,
    probability: Any,
    judged_class: Any,
    reference_semantic: Any = None,
    threshold: float = CONFLICT_THRESHOLD,
    created_utc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Persist one ``conflict_maps/<slug>.npz`` (probability + judged class).

    The map is PNG-only evidence: no ground truth is read, accepted or recorded.
    ``judged_class`` is the class map the head was conditioned on, so it must be
    the SAVED native CC8 mask; pass ``reference_semantic=`` to have that identity
    asserted against the reference map instead of trusted.
    """

    numpy = _require_numpy()
    if not math.isfinite(float(threshold)) or float(threshold) != float(CONFLICT_THRESHOLD):
        raise PixelConflictError(
            f"threshold must stay at the registered {CONFLICT_THRESHOLD}; got {threshold!r}"
        )
    probabilities = _as_probability(probability)
    judged = _as_class_array(judged_class, "judged_class")
    _require_same_shape("judged_class", judged, probabilities.shape)
    if int(judged.min()) < 0 or int(judged.max()) > NUM_CLASSES:
        raise PixelConflictError(
            f"judged_class ids must be 0..{NUM_CLASSES}; got {int(judged.min())}"
            f"..{int(judged.max())}"
        )
    source = "provided"
    reference_pixels = 0
    if reference_semantic is not None:
        reference = _as_class_array(reference_semantic, "reference_semantic")
        _require_same_shape("reference_semantic", reference, probabilities.shape)
        if int(reference.min()) < 0 or int(reference.max()) > NUM_CLASSES:
            raise PixelConflictError(
                f"reference_semantic ids must be 0..{NUM_CLASSES}; got "
                f"{int(reference.min())}..{int(reference.max())}"
            )
        mismatch = int(numpy.count_nonzero(judged != reference))
        if mismatch:
            raise PixelConflictError(
                f"judged_class is not the saved native reference mask ({mismatch} "
                "pixels differ); the conditioner must be the saved map, not a "
                "recomputed local argmax"
            )
        source = "reference_semantic"
        reference_pixels = int(reference.size)

    slug = conflict_slug(sample_id)
    target = _prepare_target(directory, "conflict_maps", f"{slug}.npz", overwrite=overwrite)
    stamp = _stamp(created_utc)
    judged_int16 = judged.astype(numpy.int16, copy=False)
    size = _atomic_write_npz(
        target,
        {
            "schema": numpy.array(CONFLICT_MAP_SCHEMA),
            "sample_id": numpy.array(str(sample_id)),
            "slug": numpy.array(slug),
            "probability": probabilities,
            "judged_class": judged_int16,
            "threshold": numpy.array(float(threshold)),
            "judged_source": numpy.array(source),
            "created_utc": numpy.array(stamp),
        },
    )
    return {
        "schema": CONFLICT_MAP_SCHEMA,
        "path": str(target),
        "sample_id": str(sample_id),
        "slug": slug,
        "shape": [int(probabilities.shape[0]), int(probabilities.shape[1])],
        "threshold": float(threshold),
        "judged_source": source,
        "judged_pixels": int(judged.size),
        "reference_pixels_checked": reference_pixels,
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_sha256": _sha256_array(probabilities),
        "judged_sha256": _sha256_array(judged_int16),
        "bytes": int(size),
        "created_utc": stamp,
        "ground_truth_read": False,
    }


def write_semantics(
    directory: Any,
    sample_id: Any,
    *,
    semantic: Any,
    instance: Any = None,
    reference_semantic: Any = None,
    created_utc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Persist one ``semantics/<slug>.npz`` (trimmed semantic + CC8 instance).

    ``semantic`` must be a strict subset of the reference ink with unchanged kept
    classes; pass ``reference_semantic=`` (the saved native mask) to assert that
    here.  ``instance`` is recomputed with ``recognition_raster_infer.instance_ids``
    and compared, so a persisted map cannot disagree with the evaluator's CC8.
    """

    numpy = _require_numpy()
    labels = _as_class_array(semantic, "semantic")
    if int(labels.min()) < 0 or int(labels.max()) > NUM_CLASSES:
        raise PixelConflictError(
            f"semantic ids must be 0..{NUM_CLASSES}; got {int(labels.min())}..{int(labels.max())}"
        )
    added = 0
    drifted = 0
    trimmed = 0
    reference_pixels = 0
    if reference_semantic is not None:
        reference = _as_class_array(reference_semantic, "reference_semantic")
        _require_same_shape("reference_semantic", reference, labels.shape)
        if int(reference.min()) < 0 or int(reference.max()) > NUM_CLASSES:
            raise PixelConflictError(
                f"reference_semantic ids must be 0..{NUM_CLASSES}; got "
                f"{int(reference.min())}..{int(reference.max())}"
            )
        kept = labels != IGNORE_CLASS
        reference_ink = reference != IGNORE_CLASS
        added = int(numpy.count_nonzero(kept & ~reference_ink))
        drifted = int(numpy.count_nonzero(kept & (labels != reference)))
        if added:
            raise PixelConflictError(
                f"{added} kept pixels lie outside the reference ink; trimming may "
                "only remove pixels, never add them"
            )
        if drifted:
            raise PixelConflictError(
                f"{drifted} kept pixels changed class; kept labels must be identical "
                "to the reference"
            )
        reference_pixels = int(reference.size)
        trimmed = int(numpy.count_nonzero(reference_ink & ~kept))

    infer = _require_instance_ids()
    computed = infer.instance_ids(labels.astype(numpy.int16))
    instance_source = "recognition_raster_infer.instance_ids"
    if instance is not None:
        supplied = numpy.asarray(instance)
        if supplied.ndim != 2:
            raise PixelConflictError(f"instance must be 2-D H x W; got {supplied.shape}")
        if supplied.dtype == bool or not numpy.issubdtype(supplied.dtype, numpy.integer):
            raise PixelConflictError(f"instance must be an integer array; got {supplied.dtype}")
        _require_same_shape("instance", supplied, labels.shape)
        mismatch = int(numpy.count_nonzero(supplied.astype(numpy.int32) != computed))
        if mismatch:
            raise PixelConflictError(
                f"instance disagrees with recognition_raster_infer.instance_ids on "
                f"{mismatch} pixels; persistence and the evaluator must share one CC8"
            )
        instance_source = "provided (verified against recognition_raster_infer.instance_ids)"

    slug = conflict_slug(sample_id)
    target = _prepare_target(directory, "semantics", f"{slug}.npz", overwrite=overwrite)
    stamp = _stamp(created_utc)
    semantic_int16 = labels.astype(numpy.int16, copy=False)
    instance_int32 = computed.astype(numpy.int32, copy=False)
    size = _atomic_write_npz(
        target,
        {
            "schema": numpy.array(SEMANTIC_SCHEMA),
            "sample_id": numpy.array(str(sample_id)),
            "slug": numpy.array(slug),
            "semantic": semantic_int16,
            "instance": instance_int32,
            "instance_source": numpy.array(instance_source),
            "created_utc": numpy.array(stamp),
        },
    )
    kept_count = int(numpy.count_nonzero(labels != IGNORE_CLASS))
    return {
        "schema": SEMANTIC_SCHEMA,
        "path": str(target),
        "sample_id": str(sample_id),
        "slug": slug,
        "shape": [int(labels.shape[0]), int(labels.shape[1])],
        "kept_pixels": kept_count,
        "ignored_pixels": int(labels.size) - kept_count,
        "reference_pixels_checked": reference_pixels,
        "trimmed_pixels": trimmed,
        "added_pixels": added,
        "class_drift_pixels": drifted,
        "instance_source": instance_source,
        "semantic_sha256": _sha256_array(semantic_int16),
        "instance_sha256": _sha256_array(instance_int32),
        "bytes": int(size),
        "created_utc": stamp,
        "ground_truth_read": False,
    }


def _require_instance_ids() -> Any:
    """The shared CC8 implementation, imported lazily (heavy module)."""

    from . import recognition_raster_infer as infer

    return infer


def write_documents(
    directory: Any,
    sample_id: Any,
    *,
    document: Any,
    created_utc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Persist one ``documents/<slug>.json`` (the vectoriser's 12-field document).

    The document is produced by the same vectoriser/affine/``tol_px = 0.5`` as the
    reference arm; this writer only audits it against the canonical 12-field
    contract and never rebuilds it.  Provenance keys are added next to the twelve
    fields, exactly as the frozen arm's documents are stored.
    """

    if not isinstance(document, Mapping):
        raise PixelConflictError("document must be a mapping of the twelve fields")
    from . import recognition_submission as submission

    fields = tuple(submission.RESULT_FIELDS)
    missing = [field for field in fields if field not in document]
    if missing:
        raise PixelConflictError("document is missing fields: " + ",".join(missing))
    projection = {field: document[field] for field in fields}
    errors = submission.validate_document(projection)
    if errors:
        raise PixelConflictError("invalid_document:" + ",".join(errors))
    if "tol_px" in document:
        declared = document["tol_px"]
        try:
            declared_float = float(declared)
        except (TypeError, ValueError):
            raise PixelConflictError(
                f"document tol_px must be the number 0.5; got {declared!r}"
            ) from None
        if isinstance(declared, bool) or not math.isfinite(declared_float) or (
            declared_float != 0.5
        ):
            raise PixelConflictError(
                f"document tol_px must be the number 0.5; got {declared!r}"
            )

    slug = conflict_slug(sample_id)
    target = _prepare_target(directory, "documents", f"{slug}.json", overwrite=overwrite)
    stamp = _stamp(created_utc)
    payload: Dict[str, Any] = {field: document[field] for field in fields}
    payload.update(
        {
            "schema": DOCUMENT_SCHEMA,
            "sample_id": str(sample_id),
            "slug": slug,
            "tol_px": 0.5,
            "created_utc": stamp,
        }
    )
    for key, value in document.items():
        if key in fields:
            continue
        payload.setdefault(str(key), value)
    size = _atomic_write_json(target, payload)
    return {
        "schema": DOCUMENT_SCHEMA,
        "path": str(target),
        "sample_id": str(sample_id),
        "slug": slug,
        "fields": list(fields),
        "primitives": int(len(projection["commands"])),
        "bytes": int(size),
        "created_utc": stamp,
        "ground_truth_read": False,
    }


def write_calibration(
    directory: Any,
    *,
    epochs: int = EPOCHS,
    threshold: float = CONFLICT_THRESHOLD,
    fit_records: int = FIT_RECORDS,
    fit_groups: int = FIT_GROUPS,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    reference: Any = None,
    gt_read: bool = False,
    oof: bool = False,
    created_utc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write ``conflict_calibration.json``; refuses any ground-truth provenance.

    Every value here is the registered recipe, so drift is an error rather than a
    recorded field: 4 epochs, threshold 0.5, 204 FIT records / 192 groups, no OOF.
    A caller that has already read ground truth cannot record it — it must not be
    calling this writer.

    ``reference`` must carry the absolute reference checkpoint path the head was
    trained against, e.g.::

        reference={"checkpoint": str(Path(joint_checkpoint).resolve()),
                   "checkpoint_schema": joint.CHECKPOINT_SCHEMA}

    A ``Path`` is converted to a string (also when nested).  The schema alone only
    proves joint **format** compatibility; the evaluator needs the path to know
    which trained reference produced the arm, so the path is mandatory there.
    """

    if gt_read:
        raise PixelConflictError(
            "calibration must be written before any ground-truth read; gt_read=True "
            "is refused rather than recorded"
        )
    if oof:
        raise PixelConflictError("no OOF or held-out calibration is authorised for this pilot")
    if int(epochs) != EPOCHS:
        raise PixelConflictError(f"calibration epochs must stay at {EPOCHS}; got {epochs!r}")
    if not math.isfinite(float(threshold)) or float(threshold) != float(CONFLICT_THRESHOLD):
        raise PixelConflictError(
            f"calibration threshold must stay at {CONFLICT_THRESHOLD}; got {threshold!r}"
        )
    if int(fit_records) != FIT_RECORDS or int(fit_groups) != FIT_GROUPS:
        raise PixelConflictError(
            f"calibration must record the authorised {FIT_RECORDS} FIT records / "
            f"{FIT_GROUPS} groups; got {fit_records!r} / {fit_groups!r}"
        )
    if str(checkpoint_schema) != CHECKPOINT_SCHEMA:
        raise PixelConflictError(
            f"checkpoint_schema must be {CHECKPOINT_SCHEMA!r}; got {checkpoint_schema!r}"
        )

    target = _prepare_target(directory, None, "conflict_calibration.json", overwrite=overwrite)
    stamp = _stamp(created_utc)
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "created_utc": stamp,
        "gt_read": False,
        "epochs": int(EPOCHS),
        "threshold": float(CONFLICT_THRESHOLD),
        "boundary": "inclusive (prob >= threshold)",
        "oof": False,
        "fit_records": int(FIT_RECORDS),
        "fit_groups": int(FIT_GROUPS),
        "judged_class": JUDGED_CLASS_SOURCE,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "train_apply_judged_shift": (
            "train conditioner = local 256 argmax; apply conditioner = saved native "
            "384/192 CC8 mask; the head predicts conditional error, not a shared map"
        ),
        "reference": dict(reference) if isinstance(reference, Mapping) else reference,
    }
    size = _atomic_write_json(target, payload)
    return {
        "schema": CALIBRATION_SCHEMA,
        "path": str(target),
        "bytes": int(size),
        "gt_read": False,
        "epochs": int(EPOCHS),
        "threshold": float(CONFLICT_THRESHOLD),
        "fit_records": int(FIT_RECORDS),
        "fit_groups": int(FIT_GROUPS),
        "created_utc": stamp,
    }


def write_predictions_manifest(
    directory: Any,
    samples: Any,
    *,
    reference_arm: Any = None,
    sample_limit: Optional[int] = PREDICTION_SAMPLE_LIMIT,
    gt_read_during_prediction: bool = False,
    created_utc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write ``predictions_manifest.json`` for the frozen PNG-only sample order.

    ``samples`` is the caller's frozen order — the FIT prefix the evaluator will
    re-read — given either as ``sample_id`` strings or as row mappings.  The order
    is preserved verbatim; duplicates, two ids that collapse to one slug, an empty
    list and a short/long list are refused.  Nothing here opens a PNG, a cache or a
    GT file.

    ``reference_arm`` must carry the absolute reference checkpoint path, e.g.
    ``reference_arm={"checkpoint": str(Path(joint_checkpoint).resolve()), ...}``;
    a ``Path`` is converted to a string, also when nested.  The evaluator needs that
    path to identify the arm — the schema is only a format assertion, not an
    identity.
    """

    if gt_read_during_prediction:
        raise PixelConflictError(
            "predictions must be PNG-only; gt_read_during_prediction=True is refused "
            "rather than recorded"
        )
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise PixelConflictError("samples must be a sequence of sample ids or row mappings")
    if not samples:
        raise PixelConflictError("samples must not be empty")
    rows: List[Dict[str, Any]] = []
    seen = set()
    seen_slugs: Dict[str, str] = {}
    for item in samples:
        if isinstance(item, Mapping):
            if "sample_id" not in item:
                raise PixelConflictError("each sample row must carry sample_id")
            row = {str(key): value for key, value in item.items()}
        else:
            row = {"sample_id": item}
        sample_id = str(row["sample_id"]).strip()
        if not sample_id:
            raise PixelConflictError("sample_id must be a non-empty string")
        if sample_id in seen:
            raise PixelConflictError(f"duplicate sample_id {sample_id!r} in the manifest")
        seen.add(sample_id)
        row["sample_id"] = sample_id
        slug = conflict_slug(sample_id)
        if "slug" in row and str(row["slug"]) != slug:
            raise PixelConflictError(
                f"row slug {row['slug']!r} disagrees with conflict_slug({sample_id!r}) "
                f"= {slug!r}"
            )
        if slug in seen_slugs and seen_slugs[slug] != sample_id:
            raise PixelConflictError(
                f"sample ids {seen_slugs[slug]!r} and {sample_id!r} share the slug "
                f"{slug!r}; their artifacts would collide"
            )
        seen_slugs[slug] = sample_id
        row["slug"] = slug
        rows.append(row)
    limit = None if sample_limit is None else int(sample_limit)
    if limit is not None and len(rows) != limit:
        raise PixelConflictError(f"manifest must list the frozen {limit} samples; got {len(rows)}")

    target = _prepare_target(directory, None, "predictions_manifest.json", overwrite=overwrite)
    stamp = _stamp(created_utc)
    payload = {
        "schema": PREDICTIONS_SCHEMA,
        "created_utc": stamp,
        "gt_read_during_prediction": False,
        "count": len(rows),
        "sample_limit": limit,
        "judged_class": JUDGED_CLASS_SOURCE,
        "threshold": float(CONFLICT_THRESHOLD),
        "reference_arm": dict(reference_arm) if isinstance(reference_arm, Mapping) else reference_arm,
        "samples": rows,
    }
    size = _atomic_write_json(target, payload)
    return {
        "schema": PREDICTIONS_SCHEMA,
        "path": str(target),
        "bytes": int(size),
        "count": len(rows),
        "sample_limit": limit,
        "sample_ids": [row["sample_id"] for row in rows],
        "gt_read_during_prediction": False,
        "created_utc": stamp,
    }


# ---------------------------------------------------------------------------
# readiness metadata (metadata only; no cache scan, no gradient step)
# ---------------------------------------------------------------------------


def _memory_plan() -> Dict[str, Any]:
    """Estimate-only memory plan (nothing is measured by this module)."""

    head_bytes = head_parameter_count() * 4
    # head params + grads + AdamW m/v
    head_training_bytes = head_bytes * 4
    tap_bytes = FEATURE_CHANNELS * (CROP_SIZE // FEATURE_STRIDE) ** 2 * 4
    head_input_bytes = head_in_channels() * (CROP_SIZE // FEATURE_STRIDE) ** 2 * 4
    return {
        "status": "estimate_only_not_measured",
        "head_parameters": head_parameter_count(),
        "head_training_bytes": head_training_bytes,
        "head_input_bytes_per_crop": head_input_bytes,
        "tap_bytes_per_crop": tap_bytes,
        "trunk": "frozen, no grad, one 256 px forward per crop under torch.no_grad",
        "single_graph": (
            "only the conflict head graph is retained (input tensors are detached "
            "and the trunk forward runs under no_grad)"
        ),
        "activation_estimate_mib": {
            "low": 700,
            "high": 1200,
            "note": (
                "order-of-magnitude for one 256 px DeepLabV3-ResNet50 no-grad "
                "forward (fp32); the head graph is negligible. Not measured, not "
                "a reservation."
            ),
        },
        "recommended_guard": {
            "reserve_after_peak_mib": 1024.0,
            "minimum_free_mib": 512.0,
            "measured_by": "peak_allocated_mib over an actual streamed 8-crop window",
            "acceptance": "free_before_mib - peak_allocated_mib >= max(1024, 512)",
        },
        "streaming": (
            "denominator from the eight masks before any forward; zero-supervision "
            "crops skipped; each supervised crop scaled by 1/denominator and "
            "backwarded immediately; one optimizer step per window"
        ),
    }


def build_readiness(
    *,
    manifest_path: str,
    joint_checkpoint: str = DEFAULT_JOINT_CHECKPOINT,
    epochs: int = EPOCHS,
) -> Dict[str, Any]:
    """CPU-only metadata report: authorized 204 FIT / 192 groups, nothing scanned."""

    from . import recognition_raster_train as train

    if not os.path.isfile(manifest_path):
        raise PixelConflictError(f"manifest_not_found: {manifest_path}")
    records = train.role_sample_records(manifest_path, "fit", train_scale=256)
    if not records:
        raise PixelConflictError("fit role resolved to zero records")
    roles = sorted({str(record.get("role")) for record in records})
    if roles != ["fit"]:
        raise PixelConflictError(f"fit reader returned roles {roles}")
    groups = len({str(record.get("group_id")) for record in records})
    return {
        "schema": READINESS_SCHEMA,
        "version": VERSION,
        "kind": "pixel_semantic_conflict_prepare",
        "lane": "pixel-semantic-conflict",
        "new_files_only": True,
        "shared_source_edits": [],
        "python": sys.executable,
        "manifest": os.path.abspath(manifest_path),
        "records": len(records),
        "groups": groups,
        "role": "fit",
        "train_scale": 256,
        "cache_scanned": False,
        "ground_truth_scanned": False,
        "hypothesis": (
            "a per-pixel conflict probability can abstain the specific pixels whose "
            "existing class is wrong, shrinking geometry without a whole-component "
            "keep/drop decision"
        ),
        "reference": {
            "checkpoint": os.path.abspath(joint_checkpoint),
            "exists": os.path.isfile(joint_checkpoint),
            "native_semantics": "saved joint CC8 maps (384 tile / 192 stride averaged)",
            "training_target_class": (
                "local 256 px frozen argmax (disclosed proxy for the native map)"
            ),
            "inference_judged_class": (
                "saved native 384/192 semantic map cropped to each 256 tile; never "
                "the local argmax"
            ),
        },
        "geometry": {
            "crop_size": CROP_SIZE,
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
            "crops_per_record": CROPS_PER_RECORD,
            "crop_min_valid": CROP_MIN_VALID,
            "feature_stride": FEATURE_STRIDE,
            "head_runs_at": "classifier[3] tap stride, bilinear upsample to the crop",
        },
        "head": {
            "module": (
                "Conv1x1(327->32) -> ReLU -> Conv3x3(32->32) -> ReLU -> Conv1x1(32->1)"
            ),
            "in_channels": head_in_channels(),
            "inputs": [
                "frozen classifier[3] features (256)",
                "same-forward semantic logits, area-averaged to tap stride (35)",
                "PNG ink fraction per tap cell (1)",
                "one-hot judged class per tap cell (35)",
            ],
            "parameters": head_parameter_count(),
            "init_seed": HEAD_INIT_SEED,
            "init_order": "explicit torch.Generator before the trunk is built",
            "trunk_trainable": False,
            "trunk_mode": "eval, requires_grad False, forward under no_grad",
        },
        "targets": {
            "licensed": "semantic_valid & gt 0..34 & PNG ink & judged 0..34",
            "target": "1 iff judged_class != gt_class",
            "positives": "all labelled Stuff and Thing(instance=-1) pixels",
            "instance_labels": "never read",
            "negatives": "licensed agreeing pixels only; unknowns are excluded",
            "whole_component_negatives": False,
            "gt_used_outside_fit": False,
        },
        "training": {
            "epochs": int(epochs),
            "optimizer": "AdamW (head only)",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "unweighted BCEWithLogits / licensed pixel mass",
            "accumulation_steps": ACCUMULATION_STEPS,
            "base_seed": BASE_SEED,
            "epoch_seed_stride": EPOCH_SEED_STRIDE,
            "epoch_seeds": {str(i): epoch_seed(i) for i in range(1, int(epochs) + 1)},
            "head_init_seed": HEAD_INIT_SEED,
            "class_weight_sweep": False,
            "cross_validation": False,
            "all_204_records_trained": True,
            "system_oof": False,
        },
        "inference": {
            "threshold": CONFLICT_THRESHOLD,
            "boundary": "inclusive",
            "post_filter": "none (no size or min-area filter)",
            "pixel_added": False,
            "pixel_relabelled": False,
            "subset_of_input_ink": True,
            "subset_of_input_ink_verified_by": (
                "predict_conflict zeroes conflict_prob outside the PNG ink mask and "
                "apply_abstain rejects any abstained pixel outside the ink it is "
                "given"
            ),
            "prob_masked_outside_ink": True,
            "cc8_vectorizer_thresholds": "unchanged",
            "future_driver_window": (
                "a future fit driver may not run a partial tail window: every "
                "window must carry exactly ACCUMULATION_STEPS=8 crops, which "
                "run_conflict_window enforces by raising on a short batch"
            ),
        },
        "persistence": {
            "published_roles": [
                "write_conflict_maps",
                "write_semantics",
                "write_documents",
                "write_calibration",
                "write_predictions_manifest",
            ],
            "artifacts": {
                "write_conflict_maps": "conflict_maps/<slug>.npz",
                "write_semantics": "semantics/<slug>.npz",
                "write_documents": "documents/<slug>.json",
                "write_calibration": "conflict_calibration.json",
                "write_predictions_manifest": "predictions_manifest.json",
            },
            "slug": "sample_id with every char outside [A-Za-z0-9_-] replaced by _",
            "distinct_slugs_required": (
                "two sample ids that collapse to one slug (e.g. a:00 and a_00) are "
                "refused, because their artifacts would collide"
            ),
            "reference_declaration": {
                "calibration": {"reference": {"checkpoint": "ABSOLUTE_PATH"}},
                "manifest": {"reference_arm": {"checkpoint": "ABSOLUTE_PATH"}},
                "path_required_by_evaluator": (
                    "the checkpoint path identifies the trained arm; the schema proves "
                    "only joint format compatibility, not identity"
                ),
                "Path_accepted": (
                    "Path objects are converted to strings, including nested ones; "
                    "numpy scalars/arrays become JSON scalars/lists; any other "
                    "non-JSON object is refused with PixelConflictError"
                ),
            },
            "output_directory": (
                "None, empty and whitespace-only directories are refused before any "
                "path is built, so a typo cannot write beside the checkout"
            ),
            "tol_px": "a document that declares tol_px must declare exactly 0.5",
            "write_once": "an existing artifact is refused unless overwrite=True",
            "atomic": "tmp file + os.replace + fsync; a failed write leaves no artifact",
            "ground_truth": (
                "every writer records gt_read/gt_read_during_prediction as false and "
                "refuses to record a true flag; no writer opens a GT or cache file"
            ),
            "schemas": {
                "conflict_map": CONFLICT_MAP_SCHEMA,
                "semantics": SEMANTIC_SCHEMA,
                "document": DOCUMENT_SCHEMA,
                "calibration": CALIBRATION_SCHEMA,
                "predictions": PREDICTIONS_SCHEMA,
            },
            "authorised_scope": (
                "the APIs are published for the evaluator to bind; generating the 24 "
                "prediction artifacts stays a separate parent-authorised phase"
            ),
            "sample_limit": PREDICTION_SAMPLE_LIMIT,
        },
        "disclosures": [
            "training crops are 256 px while the production semantic reference is "
            "the 384/192 overlap average; the fit-time target class is a local proxy",
            "frozen parameters do not freeze the inference context; no claim of "
            "exact native semantics is made",
            "all PNG ink pixels are scored at inference because semantic_valid does "
            "not exist there; this extrapolates beyond the training mask",
            "no translation-invariance claim: the head sees a normal neural receptive "
            "field and no absolute coordinate",
            "conditioner and logits are consistent at fit (judged class = argmax of "
            "the same local 256 px logits) but can be inconsistent at apply (judged "
            "class = saved native 384/192 map, logits = local 256 forward); the head "
            "is a conditional-error predictor, not the same predictor in both roles",
            "persistence writers are published and unit-tested on synthetic arrays "
            "only; no real sample, cache or ground-truth file has been read or written",
        ],
        "memory": _memory_plan(),
        "cli": {
            "prepare": "python -m cadvec.recognition_pixel_conflict prepare",
            "note": "no fit or inference CLI is exposed in this CPU preparation commit",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-pixel semantic conflict lane (CPU preparation only)"
    )
    subparsers = parser.add_subparsers(dest="command")
    prepare = subparsers.add_parser("prepare", help="write readiness metadata")
    prepare.add_argument("--manifest", default=joint.DEFAULT_MANIFEST)
    prepare.add_argument("--joint-checkpoint", default=DEFAULT_JOINT_CHECKPOINT)
    prepare.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command not in (None, "prepare"):
        parser.error(f"unknown command {args.command!r}")
    report = build_readiness(
        manifest_path=args.manifest,
        joint_checkpoint=args.joint_checkpoint,
        epochs=int(args.epochs),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    target = os.path.join(args.output_dir, "prepare.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"records={report['records']} groups={report['groups']} -> {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
