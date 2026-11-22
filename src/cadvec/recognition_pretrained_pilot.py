"""Independent pretrained semantic-segmentation pilot lane.

This module is a *pilot*: a mature torchvision DeepLabV3-ResNet50 encoder
initialised from the official COCO/VOC checkpoint, with a fresh 35-class
semantic head.  It is deliberately separate from the GNN raster runner:

* fit only -- the role guard runs before the manifest is opened;
* crops are 256 px, produced by the reviewed fit-only support sampler
  (:class:`cadvec.recognition_raster_train.SupportCropSampler`);
* the model input is the image tensor only (explicit ImageNet normalisation);
  no label, coordinate or instance feature enters the forward pass;
* the loss is masked semantic cross-entropy on known classes 0..34; unknown
  (-1) pixels are excluded, never treated as a negative class;
* instance labels are untouched -- instance supervision is not part of this
  lane; the diagnostic decoder reuses the current ink-support + CC8 path for
  comparability;
* no tune evaluation, no class weighting, no sweep, no default-behaviour
  change to any other module.

Run ``python -m cadvec.recognition_pretrained_pilot --help`` for the three
commands (``prepare``, ``smoke``, ``fit``).  ``fit`` never starts implicitly:
the parent schedules the GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import struct
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

try:  # pragma: no cover - environment dependent
    import torchvision  # type: ignore
    from torchvision.models.segmentation import (  # type: ignore
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    TORCHVISION_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    torchvision = None  # type: ignore
    DeepLabV3_ResNet50_Weights = None  # type: ignore
    deeplabv3_resnet50 = None  # type: ignore
    TORCHVISION_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from . import recognition_raster_train as train
from .recognition_contract import ROLE_FIT, ROLE_TUNE, SEED, RoleAccessError

PILOT_SCHEMA = "cadvec.recognition.pretrained_pilot/1"
PILOT_PREPARE_SCHEMA = "cadvec.recognition.pretrained_pilot_prepare/1"
PILOT_SMOKE_SCHEMA = "cadvec.recognition.pretrained_pilot_smoke/1"
PILOT_FIT_SCHEMA = "cadvec.recognition.pretrained_pilot_fit/1"
PILOT_CHECKPOINT_SCHEMA = "cadvec.recognition.pretrained_pilot_checkpoint/1"
PILOT_VERSION = 1

NUM_CLASSES = 35
IGNORE_INDEX = -1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_SCALE = 32
DEFAULT_CROP_SIZE = 256
DEFAULT_CROPS_PER_RECORD = 4
DEFAULT_CROP_MIN_VALID = 32
DEFAULT_CROP_JITTER = 0
DEFAULT_CROP_BACKGROUND_FRACTION = 0.0
DEFAULT_MAX_FIT_SAMPLES = 24
DEFAULT_SEED = SEED
DEFAULT_SMOKE_SIZE = 32

ISOLATED_ENV_FALLBACK = {
    "pair": "torch==2.5.1 + torchvision==0.20.1 (cu121)",
    "status": "documented fallback only; not installed",
    "trigger": "only if the current torchvision cannot run this lane's model factory",
    "constraint": "never replace or downgrade the existing torch/CUDA installation",
}

PRETRAINED_MODEL_ID = "torchvision/deeplabv3_resnet50"
PRETRAINED_WEIGHTS_NAME = "COCO_WITH_VOC_LABELS_V1"
PRETRAINED_URL = "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth"
PRETRAINED_SHA256 = (
    "cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06"
)
PRETRAINED_FILE_BYTES = 168312152
PRETRAINED_LICENSE = (
    "torchvision BSD-3-Clause; official checkpoint hosted by download.pytorch.org; "
    "trained on a COCO subset with Pascal VOC labels (21 classes), not on "
    "competition data. Checkpoint licence metadata is not published separately."
)
PRETRAINED_NOTE = (
    "21-class classifier (classifier.4) and aux_classifier are skipped; the "
    "35-class head is randomly initialised, so only encoder/ASPP/head-trunk "
    "features are pretrained."
)
HEAD_REPLACED_KEYS = ("classifier.4.weight", "classifier.4.bias")
BATCH_NORM_POLICY = (
    "batch_norm frozen to pretrained running statistics during training "
    "(batch size 1 makes the ASPP global-pool BN see one value per channel); "
    "BN affine weights still receive gradients"
)
SKIP_POLICY = {
    "zero_supervision_crops": "semantic_valid == 0 and instance_valid == 0 -> skipped",
    "semantic_only_extra_skip": "semantic_valid == 0 -> skipped (no semantic supervision)",
    "synthetic_labels": False,
    "oversampling": False,
    "exposure_schedule": "fixed crop count/index from SupportCropSampler; skipped crops consume no optimizer step",
}


class PilotError(train.RecognitionError, ValueError):
    """Raised for invalid pilot configuration or unsupported environment."""


# ---------------------------------------------------------------------------
# dependency / provenance helpers
# ---------------------------------------------------------------------------


def _torchvision_capability() -> Dict[str, Any]:
    """Describe which torchvision pieces this lane actually needs.

    The installed wheel may be a model-zoo-only build without the compiled
    ``torchvision._C`` extension.  DeepLabV3 is pure ``torch.nn`` code, so that
    is sufficient; ``torchvision.ops``/``io`` would not be, and are not used.
    """

    requires: List[str] = []
    try:
        for item in importlib_metadata.requires("torchvision") or []:
            if str(item).lower().startswith("torch"):
                requires.append(str(item))
    except Exception:  # pragma: no cover - metadata absent
        pass
    c_extension = bool(getattr(torchvision, "_C", None) is not None)
    return {
        "model_factory": deeplabv3_resnet50 is not None,
        "weights_enum": DeepLabV3_ResNet50_Weights is not None,
        "compiled_extension": c_extension,
        "compiled_ops_available": c_extension,
        "cuda_build": getattr(getattr(torchvision, "version", None), "cuda", None),
        "torch_requirement": requires,
        "used_features": ["models.segmentation.deeplabv3_resnet50", "weights enum"],
        "unused_features": ["ops", "io", "transforms.v2 (compiled paths)"],
        "compatibility": (
            "model-zoo definitions are pure PyTorch and import cleanly; "
            "no compiled torchvision kernel is on the forward/backward path"
            if not c_extension
            else "compiled torchvision extension present"
        ),
    }


def dependency_status() -> Dict[str, Any]:
    """Report import state without downloading or reading any data."""

    try:  # local import keeps module import cheap and survives a broken PIL
        import PIL  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        PIL = None  # type: ignore

    status: Dict[str, Any] = {
        "torch": torch is not None,
        "numpy": np is not None,
        "pillow": PIL is not None,
        "torchvision": torchvision is not None,
        "torch_version": getattr(torch, "__version__", None),
        "torchvision_version": getattr(torchvision, "__version__", None),
        "pillow_version": getattr(PIL, "__version__", None),
        "numpy_version": getattr(np, "__version__", None),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "torchvision_import_error": TORCHVISION_IMPORT_ERROR,
    }
    if torch is not None:
        status["cudnn_version"] = torch.backends.cudnn.version()
        status["device_count"] = int(torch.cuda.device_count())
        status["cuda_device_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    else:
        status["cudnn_version"] = None
        status["device_count"] = 0
        status["cuda_device_names"] = []
    blockers: List[str] = []
    for name in ("torch", "numpy", "pillow", "torchvision"):
        if not status[name]:
            blocker = f"missing_{name}"
            if name == "torchvision" and TORCHVISION_IMPORT_ERROR:
                blocker = f"{blocker}: {TORCHVISION_IMPORT_ERROR}"
            blockers.append(blocker)
    status["blockers"] = blockers
    status["ready"] = not blockers
    status["torchvision_capability"] = _torchvision_capability()
    status["isolation_fallback"] = dict(ISOLATED_ENV_FALLBACK)
    return status


def _require_torch() -> Any:
    if torch is None:
        raise PilotError("PyTorch is required for the pretrained pilot")
    return torch


def _require_torchvision() -> Any:
    if torchvision is None or deeplabv3_resnet50 is None:
        raise PilotError(
            "torchvision DeepLabV3 is unavailable"
            + (f": {TORCHVISION_IMPORT_ERROR}" if TORCHVISION_IMPORT_ERROR else "")
        )
    return torchvision


def pretrained_metadata() -> Dict[str, Any]:
    """Describe the official checkpoint; no download is triggered."""

    meta: Dict[str, Any] = {
        "model_id": PRETRAINED_MODEL_ID,
        "weights_enum": (
            "torchvision.models.segmentation.DeepLabV3_ResNet50_Weights."
            + PRETRAINED_WEIGHTS_NAME
        ),
        "url": PRETRAINED_URL,
        "sha256": PRETRAINED_SHA256,
        "file_bytes": PRETRAINED_FILE_BYTES,
        "license": PRETRAINED_LICENSE,
        "source": "download.pytorch.org (official PyTorch model zoo)",
        "pretrained_classes": 21,
        "loaded_into": "encoder + ASPP + DeepLab head trunk",
        "replaced_head_keys": list(HEAD_REPLACED_KEYS),
        "head_initialization": "random (torch default init)",
        "claim": PRETRAINED_NOTE,
        "torchvision_version": getattr(torchvision, "__version__", None),
    }
    if torchvision is not None and DeepLabV3_ResNet50_Weights is not None:
        try:
            enum = getattr(DeepLabV3_ResNet50_Weights, PRETRAINED_WEIGHTS_NAME)
            meta["enum_url"] = enum.url
            meta["enum_name"] = enum.name
            meta["enum_meta"] = {
                "num_params": enum.meta.get("num_params"),
                "metrics": enum.meta.get("_metrics"),
                "file_size_mb": enum.meta.get("_file_size"),
                "recipe": enum.meta.get("recipe"),
            }
        except Exception as exc:  # pragma: no cover - enum API drift
            meta["enum_error"] = f"{type(exc).__name__}: {exc}"
    return meta


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class PilotSegmenter(torch.nn.Module if torch is not None else object):  # type: ignore
    """Image-only DeepLabV3 wrapper: normalise, run, upsample logits.

    ``forward`` accepts exactly one positional argument (the image tensor).
    Labels are never concatenated or encoded as extra channels.
    """

    def __init__(self, net: Any, *, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        torch = _require_torch()
        self.net = net
        self.num_classes = int(num_classes)
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, images: Any) -> Dict[str, Any]:
        torch = _require_torch()
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise PilotError(
                f"model input must be NCHW with 3 channels; got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            images = images.float()
        images = images.to(dtype=self.imagenet_mean.dtype)
        height, width = int(images.shape[-2]), int(images.shape[-1])
        normalised = (images - self.imagenet_mean) / self.imagenet_std
        outputs = self.net(normalised)
        logits = outputs["out"] if isinstance(outputs, Mapping) else outputs
        if tuple(logits.shape[-2:]) != (height, width):
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        return {"semantic_logits": logits}


def _select_pretrained_keys(
    model: Any, state: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Keep shapes that match; report skipped and missing keys."""

    own = model.state_dict()
    loadable: Dict[str, Any] = {}
    skipped: List[str] = []
    for key, value in state.items():
        if not hasattr(value, "shape") or key not in own:
            skipped.append(str(key))
            continue
        if tuple(own[key].shape) != tuple(value.shape):
            skipped.append(str(key))
            continue
        loadable[key] = value
    missing = [str(key) for key in own if key not in loadable]
    return loadable, sorted(skipped), sorted(missing)


def _load_state_dict_from_url(
    url: str, *, cache_dir: Optional[str] = None, progress: bool = False
) -> Mapping[str, Any]:
    torch = _require_torch()
    kwargs: Dict[str, Any] = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kwargs["model_dir"] = cache_dir
    try:
        state = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=progress, check_hash=True, **kwargs
        )
    except TypeError:  # pragma: no cover - older torch
        state = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=progress, **kwargs
        )
    if not isinstance(state, Mapping):
        raise PilotError("pretrained checkpoint is not a state dict")
    return state


def build_pilot_model(
    *,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    weights_cache: Optional[str] = None,
    progress: bool = False,
    num_classes: int = NUM_CLASSES,
) -> Tuple[Any, Dict[str, Any]]:
    """Build a 35-class DeepLabV3-ResNet50 and optionally load official weights.

    ``weights_backbone=None`` is passed explicitly so constructing the model
    with ``pretrained=False`` never touches the network.
    """

    _require_torchvision()
    torch = _require_torch()
    provenance = pretrained_metadata()
    provenance["pretrained"] = bool(pretrained)
    provenance["loaded_key_count"] = 0
    provenance["skipped_keys"] = []
    provenance["missing_keys"] = []
    provenance["weights_sha256"] = None
    provenance["weights_sha256_verified"] = None
    provenance["weights_file"] = None

    if pretrained and weights_path is not None and not os.path.isfile(weights_path):
        raise PilotError(f"pretrained_weights_not_found: {weights_path}")

    net = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=int(num_classes),
        aux_loss=False,
    )
    if getattr(net, "aux_classifier", None) is not None:  # pragma: no cover
        raise PilotError("auxiliary classifier must be disabled for semantic-only fit")
    if int(net.classifier[-1].out_channels) != int(num_classes):  # pragma: no cover
        raise PilotError("replacement head does not have the expected class count")

    model = PilotSegmenter(net, num_classes=int(num_classes))

    if pretrained:
        if weights_path is not None:
            if not os.path.isfile(weights_path):
                raise PilotError(f"pretrained_weights_not_found: {weights_path}")
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - torch < 2.6 signature
                state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            provenance["weights_file"] = os.path.abspath(weights_path)
            digest = _sha256_file(weights_path)
            provenance["weights_sha256"] = digest
            provenance["weights_sha256_verified"] = digest == PRETRAINED_SHA256
        else:
            state = _load_state_dict_from_url(
                PRETRAINED_URL, cache_dir=weights_cache, progress=progress
            )
            cached = os.path.join(
                weights_cache
                or os.path.join(torch.hub.get_dir(), "checkpoints"),
                os.path.basename(PRETRAINED_URL),
            )
            if os.path.isfile(cached):
                try:
                    digest = _sha256_file(cached)
                    provenance["weights_file"] = os.path.abspath(cached)
                    provenance["weights_sha256"] = digest
                    provenance["weights_sha256_verified"] = digest == PRETRAINED_SHA256
                except OSError:  # pragma: no cover - cache cleanup race
                    pass
        if not isinstance(state, Mapping):
            raise PilotError("pretrained checkpoint is not a state dict")
        loadable, skipped, missing = _select_pretrained_keys(model.net, state)
        model.net.load_state_dict(loadable, strict=False)
        provenance["loaded_key_count"] = len(loadable)
        provenance["skipped_keys"] = skipped
        provenance["missing_keys"] = missing
        provenance["skipped_key_count"] = len(skipped)
        provenance["missing_key_count"] = len(missing)

    provenance["parameters"] = int(sum(param.numel() for param in model.parameters()))
    provenance["trainable_parameters"] = int(
        sum(param.numel() for param in model.parameters() if param.requires_grad)
    )
    provenance["num_classes"] = int(num_classes)
    provenance["batch_norm_policy"] = BATCH_NORM_POLICY
    return model, provenance


def freeze_batch_norm(model: Any) -> int:
    """Switch BN modules to eval (running stats) after ``model.train()``."""

    torch = _require_torch()
    frozen = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen += 1
    return frozen


def load_checkpoint_model(
    checkpoint_path: str, *, device: str = "cpu"
) -> Tuple[Any, Dict[str, Any]]:
    torch = _require_torch()
    if not os.path.isfile(checkpoint_path):
        raise PilotError(f"checkpoint_not_found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6 signature
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != PILOT_CHECKPOINT_SCHEMA:
        raise PilotError(f"checkpoint_schema_mismatch: {checkpoint_path}")
    if payload.get("role") != ROLE_FIT:
        raise RoleAccessError(
            f"checkpoint role must be {ROLE_FIT!r}; got {payload.get('role')!r}"
        )
    model, _ = build_pilot_model(pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, dict(payload)


# ---------------------------------------------------------------------------
# loss / batches / epoch
# ---------------------------------------------------------------------------


def masked_semantic_ce(logits: Any, target: Any) -> Tuple[Any, Dict[str, Any]]:
    """Cross-entropy over known 0..34 pixels only; unknown is ignored."""

    torch = _require_torch()
    if logits.ndim != 4 or int(logits.shape[1]) != NUM_CLASSES:
        raise PilotError(
            f"semantic logits must be N x {NUM_CLASSES} x H x W; got {tuple(logits.shape)}"
        )
    if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
        raise PilotError(
            f"target/logit size mismatch: {tuple(target.shape[-2:])} vs {tuple(logits.shape[-2:])}"
        )
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if int(target.shape[0]) != int(logits.shape[0]):
        raise PilotError("target/logit batch mismatch")
    if not bool(torch.isfinite(logits).all()):
        raise PilotError("non_finite_logits")
    target = target.to(device=logits.device, dtype=torch.long)
    target = torch.where(
        (target < 0) | (target >= NUM_CLASSES),
        torch.full_like(target, IGNORE_INDEX),
        target,
    )
    total = int(target.numel())
    known = int((target >= 0).sum().item())
    if known == 0:
        return logits.sum() * 0.0, {
            "known_pixels": 0,
            "ignored_pixels": total,
            "total_pixels": total,
        }
    loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    if not bool(torch.isfinite(loss)):
        raise PilotError("non_finite_loss")
    return loss, {"known_pixels": known, "ignored_pixels": total - known, "total_pixels": total}


def _entry_skip_reason(entry: Mapping[str, Any]) -> Optional[str]:
    semantic_valid = np.asarray(entry["semantic_valid"], dtype=bool)
    if bool(entry.get("skip_training")):
        return "zero_supervision_crop"
    if int(semantic_valid.sum()) == 0:
        return "no_known_semantic_pixels"
    remaining = np.asarray(entry["semantic"], dtype=np.int64)[semantic_valid]
    if remaining.size == 0 or not bool(((remaining >= 0) & (remaining < NUM_CLASSES)).any()):
        return "no_known_semantic_pixels"
    return None


def make_batch(
    source: Any, indices: Sequence[int], *, device: str = "cpu"
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch crops, drop zero-supervision entries, stack image + semantic only."""

    torch = _require_torch()
    images: List[Any] = []
    targets: List[Any] = []
    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for index in indices:
        entry = source.get(int(index))
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
        image = np.asarray(entry["image"], dtype=np.float32)
        semantic = np.asarray(entry["semantic"], dtype=np.int64)
        if image.ndim != 3 or image.shape[0] != 3:
            raise PilotError(f"image must be 3 x H x W; got {image.shape}")
        if semantic.shape != image.shape[-2:]:
            raise PilotError(
                f"semantic/image size mismatch: {semantic.shape} vs {image.shape[-2:]}"
            )
        images.append(image)
        targets.append(semantic)
        known = int(((semantic >= 0) & (semantic < NUM_CLASSES)).sum())
        kept.append(
            {
                "index": int(index),
                "sample_id": str(entry.get("sample_id") or ""),
                "crop_index": int(entry.get("crop_index", 0)),
                "known_pixels": known,
            }
        )

    if not kept:
        return None, skipped
    shapes = {tuple(image.shape) for image in images}
    target_shapes = {tuple(target.shape) for target in targets}
    if len(shapes) != 1 or len(target_shapes) != 1:
        raise PilotError(f"inconsistent crop shapes in batch: {sorted(shapes)}")
    image_tensor = torch.from_numpy(np.stack(images, axis=0)).to(device)
    target_tensor = torch.from_numpy(np.stack(targets, axis=0)).to(device)
    return (
        {
            "images": image_tensor,
            "semantic": target_tensor,
            "entries": kept,
            "known_pixels": int(sum(entry["known_pixels"] for entry in kept)),
        },
        skipped,
    )


def fit_epoch(
    model: Any,
    source: Any,
    optimizer: Any,
    *,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    epoch: int = 1,
) -> Dict[str, Any]:
    """One deterministic pass over the fixed crop source."""

    torch = _require_torch()
    model.train()
    frozen_bn = freeze_batch_norm(model)
    scheduled_entries = len(source)
    scheduled_batches = int(math.ceil(scheduled_entries / float(batch_size)))
    steps = 0
    skipped_crops = 0
    skipped_batches = 0
    zero_known_crops = 0
    samples = 0
    known_total = 0
    loss_weighted = 0.0
    started = time.monotonic()

    for start in range(0, scheduled_entries, int(batch_size)):
        indices = list(range(start, min(start + int(batch_size), scheduled_entries)))
        batch, skipped = make_batch(source, indices, device=device)
        skipped_crops += len(skipped)
        for record in skipped:
            if record["reason"] == "no_known_semantic_pixels":
                zero_known_crops += 1
        if batch is None:
            skipped_batches += 1
            continue
        logits = model(batch["images"])["semantic_logits"]
        expected = (
            int(batch["images"].shape[0]),
            NUM_CLASSES,
            int(batch["images"].shape[-2]),
            int(batch["images"].shape[-1]),
        )
        if tuple(logits.shape) != expected:
            raise PilotError(f"unexpected logits shape: {tuple(logits.shape)} vs {expected}")
        loss, stats = masked_semantic_ce(logits, batch["semantic"])
        if stats["known_pixels"] == 0:
            zero_known_crops += len(batch["entries"])
            skipped_batches += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        samples += len(batch["entries"])
        known_total += stats["known_pixels"]
        loss_weighted += float(loss.detach().item()) * stats["known_pixels"]

    elapsed = time.monotonic() - started
    return {
        "epoch": int(epoch),
        "scheduled_crops": int(scheduled_entries),
        "scheduled_batches": int(scheduled_batches),
        "optimizer_steps": int(steps),
        "samples_trained": int(samples),
        "skipped_crops": int(skipped_crops),
        "skipped_batches": int(skipped_batches),
        "zero_known_semantic_crops": int(zero_known_crops),
        "known_pixels": int(known_total),
        "loss": (loss_weighted / known_total) if known_total else None,
        "batch_norm_frozen_modules": int(frozen_bn),
        "seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: Any,
    *,
    epoch: int,
    epochs: int,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
) -> str:
    torch = _require_torch()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "schema": PILOT_CHECKPOINT_SCHEMA,
        "version": PILOT_VERSION,
        "role": ROLE_FIT,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "num_classes": NUM_CLASSES,
        "model_id": PRETRAINED_MODEL_ID,
        "config": dict(config),
        "pretrained": dict(provenance),
        "train_metrics": dict(train_metrics),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


# ---------------------------------------------------------------------------
# prepare / smoke / fit
# ---------------------------------------------------------------------------


def _estimate(
    fit_records: Optional[int],
    *,
    crops_per_record: int,
    batch_size: int,
    epochs: int,
) -> Dict[str, Any]:
    if not fit_records:
        return {"available": False, "reason": "fit record count unknown (no --manifest)"}
    scheduled = int(fit_records) * int(crops_per_record)
    batches = int(math.ceil(scheduled / float(batch_size)))
    observed_skip_fraction = 0.125  # same 24-record crop pilot: 12/96 zero-valid
    return {
        "available": True,
        "fit_records": int(fit_records),
        "scheduled_crops_per_epoch": scheduled,
        "upper_bound_optimizer_steps_per_epoch": batches,
        "upper_bound_total_steps": batches * int(epochs),
        "observed_prior_zero_supervision_fraction": observed_skip_fraction,
        "expected_optimizer_steps_per_epoch": int(round(batches * (1.0 - observed_skip_fraction))),
        "expected_total_steps": int(round(batches * (1.0 - observed_skip_fraction) * int(epochs))),
        "gpu_memory_estimate_mb": [3000, 5000],
        "estimate_note": (
            "batch 1, crop 256, fp32 ResNet50-DeepLabV3; validate with "
            "torch.cuda.max_memory_allocated on the parent GPU"
        ),
    }


def _display_path(path: str) -> str:
    """Repo-relative POSIX path when possible, else absolute."""

    absolute = os.path.abspath(path)
    try:
        return os.path.relpath(absolute, os.getcwd()).replace(os.sep, "/")
    except ValueError:  # different Windows drive
        return absolute.replace(os.sep, "/")


def prepare(
    *,
    out: str,
    manifest_path: Optional[str] = None,
    fit_limit: int = DEFAULT_MAX_FIT_SAMPLES,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    weights_cache: Optional[str] = None,
    train_scale: int = DEFAULT_TRAIN_SCALE,
    progress: bool = False,
) -> Dict[str, Any]:
    """Environment/weights preparation. Reads role metadata only, never pixels."""

    os.makedirs(out, exist_ok=True)
    status = dependency_status()
    payload: Dict[str, Any] = {
        "schema": PILOT_PREPARE_SCHEMA,
        "version": PILOT_VERSION,
        "status": "ready",
        "role_scope": {"training": ROLE_FIT, "tune": "explicit evaluation only"},
        "dependencies": status,
        "config": {
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "lr": DEFAULT_LR,
            "crop_size": DEFAULT_CROP_SIZE,
            "crops_per_record": DEFAULT_CROPS_PER_RECORD,
            "crop_min_valid": DEFAULT_CROP_MIN_VALID,
            "crop_jitter": DEFAULT_CROP_JITTER,
            "crop_background_fraction": DEFAULT_CROP_BACKGROUND_FRACTION,
            "train_scale": int(train_scale),
            "max_fit_samples": int(fit_limit),
            "seed": DEFAULT_SEED,
            "model_id": PRETRAINED_MODEL_ID,
            "num_classes": NUM_CLASSES,
            "class_weighting": "none",
            "sweep": False,
            "amp": False,
        },
        "policy": {
            "semantic_only": True,
            "instance_supervision": False,
            "instance_labels_untouched": True,
            "model_input": "image tensor only (ImageNet normalised); no labels/coordinates",
            "loss": "masked semantic CE, ignore_index=-1 (unknown excluded)",
            "detection_loss": "not used; a standard detection loss would score unknown as background",
            "unknown_handling": "excluded from the loss; never a negative/background target",
            "crop_skip": dict(SKIP_POLICY),
            "batch_norm": BATCH_NORM_POLICY,
            "tune_evaluation": "not part of this pilot CLI",
        },
        "fallback_environment": dict(ISOLATED_ENV_FALLBACK),
        "build_scope": sorted(
            [
                "src/cadvec/recognition_pretrained_pilot.py",
                "tests/test_recognition_pretrained_pilot.py",
                _display_path(out),
            ]
        ),
    }
    if manifest_path is not None:
        manifest_abs = os.path.abspath(manifest_path)
        records = train.role_sample_records(
            manifest_path, ROLE_FIT, train_scale=int(train_scale)
        )
        limited = records[: int(fit_limit)]
        payload["data"] = {
            "manifest": manifest_abs,
            "manifest_sha256": _sha256_file(manifest_path),
            "fit_records_total": len(records),
            "fit_records_used": len(limited),
            "train_scale": int(train_scale),
            "sample_ids": [str(record.get("sample_id")) for record in limited],
            "pixels_read": False,
        }
    else:
        records = []
        limited = []
        payload["data"] = {
            "manifest": None,
            "fit_records_total": None,
            "fit_records_used": None,
            "pixels_read": False,
        }

    if not status["ready"]:
        payload["status"] = "blocked"
        payload["blockers"] = status["blockers"]
        payload["weights"] = None
    else:
        try:
            model, provenance = build_pilot_model(
                pretrained=bool(pretrained),
                weights_path=weights_path,
                weights_cache=weights_cache,
                progress=progress,
            )
            payload["weights"] = provenance
        except Exception as exc:
            payload["status"] = "blocked"
            payload["blockers"] = [f"{type(exc).__name__}: {exc}"]
            payload["weights"] = None

    payload["estimates"] = _estimate(
        len(limited) if manifest_path is not None else None,
        crops_per_record=DEFAULT_CROPS_PER_RECORD,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
    )
    payload["commands"] = {
        "prepare": (
            f'python -m cadvec.recognition_pretrained_pilot prepare --manifest '
            f'"<dataset_manifest.json>" --out "{out}"'
        ),
        "smoke": (
            f'python -m cadvec.recognition_pretrained_pilot smoke --out "{out}" '
            f"--pretrained --device cpu"
        ),
        "fit": (
            f'python -m cadvec.recognition_pretrained_pilot fit --manifest '
            f'"<dataset_manifest.json>" --role fit --out "{out}" --device cuda '
            f"--epochs {DEFAULT_EPOCHS} --batch-size {DEFAULT_BATCH_SIZE} "
            f"--lr {DEFAULT_LR} --crop-size {DEFAULT_CROP_SIZE} "
            f"--crops-per-record {DEFAULT_CROPS_PER_RECORD} "
            f"--max-fit-samples {DEFAULT_MAX_FIT_SAMPLES}"
        ),
    }
    _write_json(os.path.join(out, "prepare_manifest.json"), payload)
    return payload


def smoke(
    *,
    out: Optional[str] = None,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    weights_cache: Optional[str] = None,
    device: str = "cpu",
    size: int = DEFAULT_SMOKE_SIZE,
    progress: bool = False,
) -> Dict[str, Any]:
    """Synthetic CPU/GPU forward+backward; never reads a sample."""

    torch = _require_torch()
    if isinstance(size, bool) or not isinstance(size, int) or size < 16:
        raise PilotError("smoke size must be an int >= 16")
    resolved = train.resolve_device(device)
    train.seed_everything(DEFAULT_SEED)
    model, provenance = build_pilot_model(
        pretrained=bool(pretrained),
        weights_path=weights_path,
        weights_cache=weights_cache,
        progress=progress,
    )
    model.to(resolved)
    model.train()
    frozen_bn = freeze_batch_norm(model)
    generator = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED)
    images = torch.rand(1, 3, size, size, generator=generator)
    target = torch.randint(0, NUM_CLASSES, (1, size, size), generator=generator)
    target[:, :, : max(1, size // 4)] = IGNORE_INDEX
    images = images.to(resolved)
    target = target.to(resolved)
    if resolved.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    logits = model(images)["semantic_logits"]
    if tuple(logits.shape) != (1, NUM_CLASSES, size, size):
        raise PilotError(f"unexpected smoke logits shape: {tuple(logits.shape)}")
    loss, stats = masked_semantic_ce(logits, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [
        float(param.grad.detach().abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    ]
    optimizer.step()
    elapsed = time.monotonic() - started
    if not grads or not any(value > 0.0 for value in grads):
        raise PilotError("smoke backward produced no non-zero gradients")
    payload: Dict[str, Any] = {
        "schema": PILOT_SMOKE_SCHEMA,
        "version": PILOT_VERSION,
        "status": "ok",
        "mode": "synthetic",
        "samples_read": 0,
        "device": resolved,
        "size": int(size),
        "pretrained": bool(pretrained),
        "weights": provenance,
        "loss": float(loss.detach().item()),
        "loss_stats": stats,
        "logits_shape": list(logits.shape),
        "gradient_tensors": len(grads),
        "gradient_nonzero_tensors": sum(1 for value in grads if value > 0.0),
        "batch_norm_frozen_modules": int(frozen_bn),
        "optimizer": "Adam",
        "lr": DEFAULT_LR,
        "seconds": round(elapsed, 3),
    }
    if resolved.startswith("cuda"):
        payload["peak_cuda_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1
        )
        payload["device_name"] = torch.cuda.get_device_name(0)
    if out:
        os.makedirs(out, exist_ok=True)
        payload["written"] = os.path.join(out, "smoke_manifest.json")
        _write_json(payload["written"], payload)
    return payload


def run_fit(
    manifest_path: str,
    *,
    role: str,
    out: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    train_scale: int = DEFAULT_TRAIN_SCALE,
    max_fit_samples: int = DEFAULT_MAX_FIT_SAMPLES,
    crop_size: int = DEFAULT_CROP_SIZE,
    crops_per_record: int = DEFAULT_CROPS_PER_RECORD,
    crop_min_valid: int = DEFAULT_CROP_MIN_VALID,
    crop_jitter: int = DEFAULT_CROP_JITTER,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    weights_cache: Optional[str] = None,
    cache: bool = True,
    progress: bool = False,
) -> Dict[str, Any]:
    """Fit the 35-class pilot on the frozen fit prefix; no tune evaluation."""

    train.require_command_role("fit", role)
    os.makedirs(out, exist_ok=True)
    for name, value, minimum in (
        ("epochs", epochs, 1),
        ("batch_size", batch_size, 1),
        ("crop_size", crop_size, 1),
        ("crops_per_record", crops_per_record, 1),
        ("crop_min_valid", crop_min_valid, 0),
        ("crop_jitter", crop_jitter, 0),
        ("max_fit_samples", max_fit_samples, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise PilotError(f"{name} must be an int >= {minimum}; got {value!r}")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
        raise PilotError(f"lr must be a positive number; got {lr!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PilotError(f"seed must be an int; got {seed!r}")

    status = dependency_status()
    if not status["ready"]:
        payload = {
            "schema": PILOT_FIT_SCHEMA,
            "version": PILOT_VERSION,
            "status": "blocked",
            "blockers": status["blockers"],
            "dependencies": status,
            "out": os.path.abspath(out),
        }
        _write_json(os.path.join(out, "fit_manifest.json"), payload)
        return payload

    torch = _require_torch()
    resolved = train.resolve_device(device)
    all_records = train.role_sample_records(
        manifest_path, ROLE_FIT, train_scale=int(train_scale)
    )
    records = all_records[: int(max_fit_samples)]
    if not records:
        payload = {
            "schema": PILOT_FIT_SCHEMA,
            "version": PILOT_VERSION,
            "status": "blocked",
            "blockers": ["no_fit_samples"],
            "out": os.path.abspath(out),
        }
        _write_json(os.path.join(out, "fit_manifest.json"), payload)
        return payload

    os.makedirs(out, exist_ok=True)
    train.seed_everything(int(seed))
    sampler = train.SupportCropSampler(
        crop_size=int(crop_size),
        crops_per_record=int(crops_per_record),
        background_fraction=DEFAULT_CROP_BACKGROUND_FRACTION,
        jitter=int(crop_jitter),
        min_valid=int(crop_min_valid),
        seed=int(seed),
    )
    source = train.SampleSource(
        records,
        cache_dir=os.path.join(out, "cache") if cache else None,
        crop_sampler=sampler,
    )
    model, provenance = build_pilot_model(
        pretrained=bool(pretrained),
        weights_path=weights_path,
        weights_cache=weights_cache,
        progress=progress,
    )
    model.to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    config = {
        "role": ROLE_FIT,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "device": resolved,
        "train_scale": int(train_scale),
        "max_fit_samples": int(max_fit_samples),
        "crop_size": int(crop_size),
        "crops_per_record": int(crops_per_record),
        "crop_min_valid": int(crop_min_valid),
        "crop_jitter": int(crop_jitter),
        "crop_background_fraction": DEFAULT_CROP_BACKGROUND_FRACTION,
        "crop_policy": sampler.policy(),
        "model_id": PRETRAINED_MODEL_ID,
        "num_classes": NUM_CLASSES,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "loss": "masked_semantic_ce(ignore_index=-1)",
        "detection_loss": "not used",
        "class_weighting": "none",
        "instance_supervision": False,
        "amp": False,
        "batch_norm": BATCH_NORM_POLICY,
        "checkpoint_policy": "checkpoint_final.pt overwritten after each epoch (no per-epoch copies)",
        "tune_evaluation": "not performed",
    }
    metrics_path = os.path.join(out, "metrics.json")
    history: List[Dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, int(epochs) + 1):
        epoch_metrics = fit_epoch(
            model,
            source,
            optimizer,
            device=resolved,
            batch_size=int(batch_size),
            epoch=epoch,
        )
        checkpoint_path = os.path.join(out, "checkpoint_final.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=epoch,
            epochs=int(epochs),
            config=config,
            provenance=provenance,
            train_metrics=epoch_metrics,
        )
        history.append(
            {
                **epoch_metrics,
                "checkpoint": os.path.abspath(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
            }
        )
        _write_json(
            metrics_path,
            {
                "schema": PILOT_FIT_SCHEMA,
                "version": PILOT_VERSION,
                "role": ROLE_FIT,
                "epochs_requested": int(epochs),
                "epochs_completed": epoch,
                "config": config,
                "provenance": provenance,
                "history": history,
            },
        )

    elapsed = time.monotonic() - started
    if resolved.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
    else:
        peak = None
    payload = {
        "schema": PILOT_FIT_SCHEMA,
        "version": PILOT_VERSION,
        "status": "ok",
        "role": ROLE_FIT,
        "out": os.path.abspath(out),
        "manifest": os.path.abspath(manifest_path),
        "data": {
            "fit_records_total_loaded": len(all_records),
            "fit_records_used": len(records),
            "sample_ids": [str(record.get("sample_id")) for record in records],
            "train_scale": int(train_scale),
        },
        "config": config,
        "provenance": provenance,
        "crop_stats": sampler.stats(),
        "history": history,
        "final": {
            "checkpoint": os.path.abspath(os.path.join(out, "checkpoint_final.pt")),
            "checkpoint_sha256": history[-1]["checkpoint_sha256"] if history else None,
            "epoch": history[-1]["epoch"] if history else 0,
        },
        "estimates": _estimate(
            len(records),
            crops_per_record=int(crops_per_record),
            batch_size=int(batch_size),
            epochs=int(epochs),
        ),
        "actual": {
            "optimizer_steps": sum(entry["optimizer_steps"] for entry in history),
            "skipped_crops": sum(entry["skipped_crops"] for entry in history),
            "seconds": round(elapsed, 3),
            "peak_cuda_memory_mb": peak,
        },
        "selection": "final_epoch_only",
        "tune_evaluation": "not performed (explicit evaluation is out of this pilot lane)",
    }
    _write_json(os.path.join(out, "fit_manifest.json"), payload)
    return payload


def predict_png(
    png_path: str,
    checkpoint_path: str,
    *,
    device: str = "cpu",
    tol_px: float = 0.5,
) -> Dict[str, Any]:
    """Diagnostic prediction reusing the current ink-support + CC8 decoder."""

    from . import recognition_raster_infer as raster_infer

    model, payload = load_checkpoint_model(checkpoint_path, device=device)
    result = raster_infer._predict(png_path, model, device, tol_px)
    metadata = result.setdefault("metadata", {})
    metadata["pilot_checkpoint"] = os.path.abspath(checkpoint_path)
    metadata["pilot_checkpoint_epoch"] = payload.get("epoch")
    metadata["pilot_decoder"] = (
        "reused recognition_raster_infer._predict: PNG ink support + semantic_cc8_v1"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_pretrained_pilot",
        description="Independent DeepLabV3-ResNet50 semantic pilot (fit only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="check environment/weights; role metadata only")
    prep.add_argument("--manifest", default=None)
    prep.add_argument("--out", required=True)
    prep.add_argument("--fit-limit", type=int, default=DEFAULT_MAX_FIT_SAMPLES)
    prep.add_argument("--train-scale", type=int, default=DEFAULT_TRAIN_SCALE)
    prep.add_argument("--no-pretrained", action="store_true")
    prep.add_argument("--weights-path", default=None)
    prep.add_argument("--weights-cache", default=None)
    prep.add_argument("--progress", action="store_true")

    smoke_cmd = sub.add_parser("smoke", help="synthetic forward/backward, no data")
    smoke_cmd.add_argument("--out", default=None)
    smoke_cmd.add_argument("--pretrained", action="store_true")
    smoke_cmd.add_argument("--weights-path", default=None)
    smoke_cmd.add_argument("--weights-cache", default=None)
    smoke_cmd.add_argument("--device", default="cpu")
    smoke_cmd.add_argument("--size", type=int, default=DEFAULT_SMOKE_SIZE)
    smoke_cmd.add_argument("--progress", action="store_true")

    fit_cmd = sub.add_parser("fit", help="fit the pilot on the frozen fit prefix")
    fit_cmd.add_argument("--manifest", required=True)
    fit_cmd.add_argument("--role", required=True)
    fit_cmd.add_argument("--out", required=True)
    fit_cmd.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    fit_cmd.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    fit_cmd.add_argument("--lr", type=float, default=DEFAULT_LR)
    fit_cmd.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit_cmd.add_argument("--device", default="auto")
    fit_cmd.add_argument("--train-scale", type=int, default=DEFAULT_TRAIN_SCALE)
    fit_cmd.add_argument("--max-fit-samples", type=int, default=DEFAULT_MAX_FIT_SAMPLES)
    fit_cmd.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    fit_cmd.add_argument("--crops-per-record", type=int, default=DEFAULT_CROPS_PER_RECORD)
    fit_cmd.add_argument("--crop-min-valid", type=int, default=DEFAULT_CROP_MIN_VALID)
    fit_cmd.add_argument("--crop-jitter", type=int, default=DEFAULT_CROP_JITTER)
    fit_cmd.add_argument("--no-pretrained", action="store_true")
    fit_cmd.add_argument("--weights-path", default=None)
    fit_cmd.add_argument("--weights-cache", default=None)
    fit_cmd.add_argument("--no-cache", action="store_true")
    fit_cmd.add_argument("--progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            payload = prepare(
                out=args.out,
                manifest_path=args.manifest,
                fit_limit=args.fit_limit,
                pretrained=not args.no_pretrained,
                weights_path=args.weights_path,
                weights_cache=args.weights_cache,
                train_scale=args.train_scale,
                progress=args.progress,
            )
        elif args.command == "smoke":
            payload = smoke(
                out=args.out,
                pretrained=args.pretrained,
                weights_path=args.weights_path,
                weights_cache=args.weights_cache,
                device=args.device,
                size=args.size,
                progress=args.progress,
            )
        else:
            payload = run_fit(
                args.manifest,
                role=args.role,
                out=args.out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                train_scale=args.train_scale,
                max_fit_samples=args.max_fit_samples,
                crop_size=args.crop_size,
                crops_per_record=args.crops_per_record,
                crop_min_valid=args.crop_min_valid,
                crop_jitter=args.crop_jitter,
                pretrained=not args.no_pretrained,
                weights_path=args.weights_path,
                weights_cache=args.weights_cache,
                cache=not args.no_cache,
                progress=args.progress,
            )
    except (PilotError, RoleAccessError, train.RasterTrainError) as exc:
        print(
            json.dumps(
                {"status": "error", "type": type(exc).__name__, "reason": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("status") in ("ok", "ready") else 1


__all__ = [
    "BATCH_NORM_POLICY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CROP_MIN_VALID",
    "DEFAULT_CROP_SIZE",
    "DEFAULT_CROPS_PER_RECORD",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_MAX_FIT_SAMPLES",
    "DEFAULT_SEED",
    "IGNORE_INDEX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_CLASSES",
    "PILOT_CHECKPOINT_SCHEMA",
    "PILOT_FIT_SCHEMA",
    "PILOT_PREPARE_SCHEMA",
    "PILOT_SCHEMA",
    "PILOT_SMOKE_SCHEMA",
    "PilotError",
    "PilotSegmenter",
    "PRETRAINED_LICENSE",
    "PRETRAINED_MODEL_ID",
    "PRETRAINED_URL",
    "PRETRAINED_WEIGHTS_NAME",
    "build_pilot_model",
    "build_parser",
    "dependency_status",
    "fit_epoch",
    "freeze_batch_norm",
    "load_checkpoint_model",
    "main",
    "make_batch",
    "masked_semantic_ce",
    "predict_png",
    "prepare",
    "pretrained_metadata",
    "run_fit",
    "save_checkpoint",
    "smoke",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
