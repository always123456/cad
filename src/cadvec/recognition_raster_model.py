"""High-resolution raster recognition model and masked losses (CAD route).

This module is deliberately self-contained: it owns only the pixel branch of
the recognition route and has no knowledge of datasets, splits, scorers or
files on disk.  It imports PyTorch lazily so that the standard-library core of
``cadvec`` keeps working without it; callers get an explicit
``MissingDependencyError`` instead of an ``ImportError`` at random points.

Label space (frozen by :mod:`cadvec.recognition_contract`):

* class index ``c`` in ``0..34`` corresponds to raw semantic ``c + 1``;
* ``-1`` (or any negative value) means *unknown* and is never trained on;
* class indices ``0..29`` are Things (instance-capable), ``30..34`` are Stuff.

The model is a compact U-Net variant that keeps the input resolution: every
decoder step and the final heads are brought back to the exact ``H x W`` of the
input with bilinear interpolation, so semantic logits and embeddings are
pixel-aligned with the target masks at full resolution.  Normalisation uses
:class:`torch.nn.GroupNorm` only, which works with batch size 1; there is no
pretrained weight download and no external asset.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

from .recognition_contract import (
    RAW_SEMANTIC_MAX,
    RAW_SEMANTIC_MIN,
    THING_MAX,
    THING_MIN,
    MissingDependencyError,
    RecognitionError,
)

try:  # pragma: no cover - torch presence is tested indirectly
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - broken/absent install
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Optional[BaseException] = exc
else:  # pragma: no cover - the import itself is the tested path
    _TORCH_IMPORT_ERROR = None

NUM_CLASSES = int(RAW_SEMANTIC_MAX)  # 35 class indices, 0..34
NUM_THING_CLASSES = int(THING_MAX - THING_MIN + 1)  # 30, class indices 0..29
NUM_STUFF_CLASSES = NUM_CLASSES - NUM_THING_CLASSES  # 5, class indices 30..34
EMBED_DIM = 8
IGNORE_INDEX = -100
NO_INSTANCE = -1
INPUT_MEAN = 0.5
INPUT_STD = 0.5
SCHEMA = "cadvec.recognition.raster_model/1"

# ``nn.Module`` when torch is importable, ``object`` otherwise.  Keeping a base
# class lets the module import cleanly without torch while every constructor
# raises an explicit dependency error.
_ModuleBase = nn.Module if nn is not None else object


class RasterModelError(RecognitionError, ValueError):
    code = "raster_model_error"


def dependency_status() -> Dict[str, Any]:
    """Report torch availability without installing or downloading anything."""

    result: Dict[str, Any] = {
        "torch": importlib.util.find_spec("torch") is not None,
        "available": False,
        "version": None,
        "cuda": False,
    }
    if not result["torch"]:
        return result
    if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover - broken install
        result["import_error"] = f"{type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}"
        return result
    result["version"] = getattr(torch, "__version__", None)
    try:
        result["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver-level failure
        result["cuda"] = False
    result["available"] = True
    return result


def require_torch() -> None:
    """Raise :class:`MissingDependencyError` when torch cannot be imported."""

    status = dependency_status()
    if status["available"]:
        return
    detail = status.get("import_error")
    message = "recognition_raster_model requires PyTorch"
    if detail:
        message = f"{message} ({detail})"
    raise MissingDependencyError(message)


# ---------------------------------------------------------------------------
# label-space helpers
# ---------------------------------------------------------------------------


def class_index_to_raw_semantic(class_index: int) -> int:
    """Map a 0-based class index (0..34) to the raw semantic id (1..35)."""

    value = int(class_index)
    if not 0 <= value < NUM_CLASSES:
        raise RasterModelError(f"class index {value} is outside [0, {NUM_CLASSES})")
    return value + int(RAW_SEMANTIC_MIN)


def raw_semantic_to_class_index(raw_semantic: int) -> int:
    """Map a raw semantic id (1..35) to the 0-based class index (0..34)."""

    value = int(raw_semantic)
    if not int(RAW_SEMANTIC_MIN) <= value <= int(RAW_SEMANTIC_MAX):
        raise RasterModelError(
            f"raw semantic {value} is outside [{RAW_SEMANTIC_MIN}, {RAW_SEMANTIC_MAX}]"
        )
    return value - int(RAW_SEMANTIC_MIN)


def is_thing_class(class_index: int) -> bool:
    """True when a 0-based class index denotes a Thing (0..29)."""

    return 0 <= int(class_index) < NUM_THING_CLASSES


# ---------------------------------------------------------------------------
# masked semantic loss
# ---------------------------------------------------------------------------


def _check_3d(name: str, tensor: Any, reference: Any) -> Any:
    if not torch.is_tensor(tensor):
        raise RasterModelError(f"{name} must be a torch.Tensor")
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.dim() != 3:
        raise RasterModelError(f"{name} must be BxHxW (or Bx1xHxW)")
    expected = (reference.shape[0], reference.shape[2], reference.shape[3])
    if tuple(tensor.shape) != expected:
        raise RasterModelError(
            f"{name} shape {tuple(tensor.shape)} does not match logits spatial layout {expected}"
        )
    return tensor


def _as_bool_mask(name: str, mask: Any, reference: Any) -> Any:
    mask = _check_3d(name, mask, reference)
    return mask.to(device=reference.device, dtype=torch.bool)


def _require_integer(name: str, tensor: Any) -> Any:
    if tensor.is_floating_point() or tensor.is_complex():
        raise RasterModelError(f"{name} must hold integer ids")
    return tensor


def masked_semantic_loss(
    logits: Any,
    targets: Any,
    valid: Any,
    *,
    weight: Optional[Any] = None,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> Any:
    """Cross entropy over *valid* pixels only.

    ``logits`` is ``B x C x H x W``; ``targets`` is ``B x H x W`` (or
    ``B x 1 x H x W``) of 0-based class indices; ``valid`` is a boolean mask of
    the same spatial layout.  A pixel is trained on iff ``valid`` is true *and*
    its target is non-negative and not ``ignore_index``; every other pixel is
    excluded rather than pushed towards a background class.  When no pixel
    survives, the result is a differentiable zero connected to ``logits``.
    """

    require_torch()
    if not torch.is_tensor(logits):
        raise RasterModelError("logits must be a torch.Tensor")
    if logits.dim() != 4:
        raise RasterModelError("logits must be BxCxHxW")
    num_classes = int(logits.shape[1])
    if num_classes < 2:
        raise RasterModelError("logits must have at least two classes")
    target = _check_3d("targets", targets, logits)
    if target.is_floating_point():
        raise RasterModelError("targets must hold integer class indices")
    keep = _as_bool_mask("valid", valid, logits)
    flat_target = target.reshape(-1).to(device=logits.device, dtype=torch.long)
    flat_keep = keep.reshape(-1) & (flat_target != int(ignore_index)) & (flat_target >= 0)
    if not bool(flat_keep.any()):
        return logits.sum() * 0.0
    selected = flat_target[flat_keep]
    if bool((selected >= num_classes).any()):
        raise RasterModelError(
            f"semantic target index is outside [0, {num_classes})"
        )
    pixel_logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    if weight is not None:
        weight = torch.as_tensor(weight, device=logits.device, dtype=logits.dtype)
        if weight.numel() != num_classes:
            raise RasterModelError("class weight length does not match logits channels")
    return F.cross_entropy(
        pixel_logits[flat_keep],
        selected,
        weight=weight,
        label_smoothing=float(label_smoothing),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# instance embedding discriminative loss
# ---------------------------------------------------------------------------


def _even_positions(count: int, limit: int, device: Any) -> Any:
    """Deterministic, evenly spread and strictly increasing indices.

    Used both to bound the pixels sampled per instance and to bound the number
    of instances/pairs considered, so the loss cost stays bounded on large
    images without randomness (identical inputs give identical losses).
    """

    if limit <= 0:
        raise RasterModelError("sampling limits must be positive")
    if count <= limit:
        return torch.arange(int(count), device=device, dtype=torch.long)
    steps = torch.arange(int(limit), device=device, dtype=torch.long)
    return (steps * int(count)) // int(limit)


def instance_embedding_loss(
    embeddings: Any,
    instance_targets: Any,
    instance_valid: Any,
    *,
    class_ids: Optional[Any] = None,
    class_valid: Optional[Any] = None,
    instance_class: Optional[Any] = None,
    instance_class_valid: Optional[Any] = None,
    semantic_targets: Optional[Any] = None,
    semantic_valid: Optional[Any] = None,
    num_thing_classes: int = NUM_THING_CLASSES,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    var_weight: float = 1.0,
    dist_weight: float = 1.0,
    max_pixels_per_instance: int = 1024,
    max_instances_per_sample: int = 256,
    max_pairs: int = 4096,
    return_components: bool = False,
) -> Any:
    """Discriminative instance-embedding loss on valid Thing pixels only.

    Inputs (all ``B x H x W`` unless noted):

    * ``embeddings`` - ``B x D x H x W`` float feature map;
    * ``instance_targets`` - non-negative instance ids, ``-1`` for no instance;
    * ``instance_valid`` - explicit mask; whether a pixel may take part in the
      loss is decided here, *independently* of any semantic mask;
    * ``class_ids`` / ``class_valid`` - explicit 0-based class indices and their
      validity (aliases: ``instance_class`` / ``instance_class_valid``).  Only
      classes ``0..num_thing_classes-1`` are used; Stuff and unknown classes are
      excluded.  ``class_valid`` is the authority for class membership, so a
      pixel whose semantic mask is missing can still contribute as long as its
      class is known here.

    If ``class_ids`` is omitted, the loss falls back to ``semantic_targets`` /
    ``semantic_valid`` (class indices aligned with the semantic head).  If both
    are omitted, grouping is class-agnostic and every valid instance pixel is
    used - the caller is then responsible for passing Thing-only instances.

    Groups are keyed by (sample, class, instance).  Attraction pulls pixels
    towards their own group mean; repulsion pushes the means of distinct groups
    apart, including same-class-but-different-instance pairs.  There are no
    background or unlabeled negatives.  Per-instance pixel sampling and the
    instance/pair caps are deterministic (evenly spaced), so huge images stay
    bounded and repeated calls are exactly reproducible.
    """

    require_torch()
    if not torch.is_tensor(embeddings):
        raise RasterModelError("embeddings must be a torch.Tensor")
    if embeddings.dim() != 4:
        raise RasterModelError("embeddings must be BxDxHxW")
    if not embeddings.is_floating_point():
        raise RasterModelError("embeddings must be floating point")
    batch, dim, height, width = (int(v) for v in embeddings.shape)
    if dim < 1:
        raise RasterModelError("embedding dimension must be positive")

    instance_target = _check_3d("instance_targets", instance_targets, embeddings)
    _require_integer("instance_targets", instance_target)
    mask = _as_bool_mask("instance_valid", instance_valid, embeddings)
    mask = mask & (instance_target.to(device=embeddings.device) >= int(NO_INSTANCE) + 1)

    if instance_class is not None:
        if class_ids is not None:
            raise RasterModelError("pass either class_ids or instance_class, not both")
        class_ids = instance_class
    if instance_class_valid is not None:
        if class_valid is not None:
            raise RasterModelError("pass either class_valid or instance_class_valid, not both")
        class_valid = instance_class_valid

    class_source: Optional[Any] = None
    if class_ids is not None:
        class_source = _check_3d("class_ids", class_ids, embeddings)
        _require_integer("class_ids", class_source)
        if class_valid is not None:
            class_keep = _as_bool_mask("class_valid", class_valid, embeddings)
        else:
            class_keep = torch.ones_like(mask)
        class_source = class_source.to(device=embeddings.device)
        mask = mask & class_keep & (class_source >= 0) & (class_source < int(num_thing_classes))
        class_ids_used = class_source
    elif semantic_targets is not None:
        class_source = _check_3d("semantic_targets", semantic_targets, embeddings)
        _require_integer("semantic_targets", class_source)
        if semantic_valid is None:
            raise RasterModelError(
                "semantic_valid is required when class_ids are derived from semantic_targets"
            )
        class_keep = _as_bool_mask("semantic_valid", semantic_valid, embeddings)
        class_source = class_source.to(device=embeddings.device)
        mask = mask & class_keep & (class_source >= 0) & (class_source < int(num_thing_classes))
        class_ids_used = class_source
    else:
        class_ids_used = torch.zeros_like(instance_target, device=embeddings.device)

    if not bool(mask.any()):
        zero = embeddings.sum() * 0.0
        if return_components:
            return {
                "loss": zero,
                "var": zero,
                "dist": zero,
                "num_groups": 0,
                "num_pixels": 0,
            }
        return zero

    device = embeddings.device
    sampled_features = []
    sampled_group_ids = []
    group_means = []
    group_counter = 0
    instance_flat = instance_target.to(device=device, dtype=torch.long)

    for sample in range(batch):
        sample_mask = mask[sample]
        if not bool(sample_mask.any()):
            continue
        features = embeddings[sample].permute(1, 2, 0)  # H x W x D
        selected_features = features[sample_mask]
        selected_class = class_ids_used[sample][sample_mask].to(dtype=torch.long)
        selected_instance = instance_flat[sample][sample_mask]
        keys = torch.stack((selected_class, selected_instance), dim=1)
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True, sorted=True)
        counts = torch.bincount(inverse, minlength=int(unique.shape[0]))
        order = torch.argsort(inverse, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        chosen = _even_positions(int(unique.shape[0]), int(max_instances_per_sample), device)
        for position in chosen.tolist():
            start = int(starts[position].item())
            count = int(counts[position].item())
            offset = _even_positions(count, int(max_pixels_per_instance), device)
            index = order[start : start + count][offset]
            group = selected_features[index]
            sampled_features.append(group)
            sampled_group_ids.append(
                torch.full((int(group.shape[0]),), group_counter, dtype=torch.long, device=device)
            )
            group_means.append(group.mean(dim=0, keepdim=True))
            group_counter += 1

    if group_counter == 0:  # pragma: no cover - mask.any() already checked
        zero = embeddings.sum() * 0.0
        if return_components:
            return {"loss": zero, "var": zero, "dist": zero, "num_groups": 0, "num_pixels": 0}
        return zero

    means = torch.cat(group_means, dim=0)  # G x D
    features_all = torch.cat(sampled_features, dim=0)  # S x D
    group_all = torch.cat(sampled_group_ids, dim=0)  # S

    distance_to_mean = torch.linalg.vector_norm(features_all - means[group_all], dim=1)
    var_term = (torch.clamp(distance_to_mean - float(delta_v), min=0.0) ** 2).mean()

    if group_counter >= 2:
        pairwise = torch.cdist(means, means)
        left, right = torch.triu_indices(group_counter, group_counter, offset=1, device=device)
        if int(left.numel()) > int(max_pairs):
            keep = _even_positions(int(left.numel()), int(max_pairs), device)
            left, right = left[keep], right[keep]
        pair_distance = pairwise[left, right]
        dist_term = (
            torch.clamp(2.0 * float(delta_d) - pair_distance, min=0.0) ** 2
        ).mean()
    else:
        dist_term = means.sum() * 0.0

    loss = float(var_weight) * var_term + float(dist_weight) * dist_term
    if return_components:
        return {
            "loss": loss,
            "var": var_term,
            "dist": dist_term,
            "num_groups": int(group_counter),
            "num_pixels": int(features_all.shape[0]),
        }
    return loss


# A descriptive alias: the loss is the classic instance discriminative loss.
instance_discriminative_loss = instance_embedding_loss


def combined_raster_loss(
    outputs: Mapping[str, Any],
    *,
    semantic_targets: Any,
    semantic_valid: Any,
    instance_targets: Any,
    instance_valid: Any,
    class_ids: Optional[Any] = None,
    class_valid: Optional[Any] = None,
    instance_class: Optional[Any] = None,
    instance_class_valid: Optional[Any] = None,
    semantic_weight: float = 1.0,
    instance_weight: float = 1.0,
    semantic_class_weight: Optional[Any] = None,
    label_smoothing: float = 0.0,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    max_pixels_per_instance: int = 1024,
    max_instances_per_sample: int = 256,
    max_pairs: int = 4096,
) -> Dict[str, Any]:
    """Combine masked semantic CE with the Thing-only instance loss.

    ``outputs`` is the dict returned by :meth:`RasterUNet.forward`.  The
    semantic term trains on every valid semantic pixel (Thing and Stuff alike,
    unknown excluded); the instance term trains only on valid Thing pixels,
    derived from ``class_ids``/``class_valid`` when given and otherwise from the
    semantic fields.  Stuff and unknown pixels never act as instance negatives.

    Returns ``{"loss", "semantic", "instance"}`` so callers can log the parts.
    """

    require_torch()
    if not isinstance(outputs, Mapping):
        raise RasterModelError("outputs must be the dict returned by RasterUNet.forward")
    for key in ("semantic_logits", "embeddings"):
        if key not in outputs:
            raise RasterModelError(f"outputs is missing {key!r}")

    semantic_term = masked_semantic_loss(
        outputs["semantic_logits"],
        semantic_targets,
        semantic_valid,
        weight=semantic_class_weight,
        label_smoothing=float(label_smoothing),
    )

    class_is_explicit = class_ids is not None or instance_class is not None
    derived: Dict[str, Any] = {}
    if not class_is_explicit:
        derived = {"semantic_targets": semantic_targets, "semantic_valid": semantic_valid}
    instance_term = instance_embedding_loss(
        outputs["embeddings"],
        instance_targets,
        instance_valid,
        class_ids=class_ids,
        class_valid=class_valid,
        instance_class=instance_class,
        instance_class_valid=instance_class_valid,
        delta_v=float(delta_v),
        delta_d=float(delta_d),
        max_pixels_per_instance=int(max_pixels_per_instance),
        max_instances_per_sample=int(max_instances_per_sample),
        max_pairs=int(max_pairs),
        **derived,
    )
    total = float(semantic_weight) * semantic_term + float(instance_weight) * instance_term
    return {"loss": total, "semantic": semantic_term, "instance": instance_term}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def _norm_groups(channels: int, max_groups: int) -> int:
    groups = max(1, min(int(max_groups), int(channels)))
    while int(channels) % groups != 0:
        groups -= 1
    return groups


class _DoubleConv(_ModuleBase):
    """Two 3x3 convolutions with GroupNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, max_groups: int) -> None:
        require_torch()
        super().__init__()
        groups = _norm_groups(out_channels, max_groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Any) -> Any:
        return self.block(x)


@dataclass(frozen=True, slots=True)
class RasterModelConfig:
    """Construction parameters for :class:`RasterUNet` (JSON-friendly)."""

    num_classes: int = NUM_CLASSES
    embedding_dim: int = EMBED_DIM
    in_channels: int = 3
    base_channels: int = 24
    depth: int = 3
    norm_groups: int = 8
    center_head: bool = False
    normalize_input: bool = True
    input_mean: float = INPUT_MEAN
    input_std: float = INPUT_STD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": int(self.num_classes),
            "embedding_dim": int(self.embedding_dim),
            "in_channels": int(self.in_channels),
            "base_channels": int(self.base_channels),
            "depth": int(self.depth),
            "norm_groups": int(self.norm_groups),
            "center_head": bool(self.center_head),
            "normalize_input": bool(self.normalize_input),
            "input_mean": float(self.input_mean),
            "input_std": float(self.input_std),
        }


class RasterUNet(_ModuleBase):
    """Compact U-Net producing full-resolution semantic and embedding maps.

    ``forward(image)`` takes ``B x in_channels x H x W`` and returns a dict:

    * ``semantic_logits`` - ``B x num_classes x H x W`` (class index space);
    * ``embeddings`` - ``B x embedding_dim x H x W``;
    * ``center_logits`` - ``B x 1 x H x W`` or ``None`` when disabled;
    * ``output_size`` - ``(H, W)``, equal to the input resolution.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBED_DIM,
        in_channels: int = 3,
        base_channels: int = 24,
        depth: int = 3,
        norm_groups: int = 8,
        center_head: bool = False,
        normalize_input: bool = True,
        input_mean: float = INPUT_MEAN,
        input_std: float = INPUT_STD,
        embed_dim: Optional[int] = None,
    ) -> None:
        require_torch()
        super().__init__()
        if embed_dim is not None:
            if int(embedding_dim) != int(EMBED_DIM) and int(embedding_dim) != int(embed_dim):
                raise RasterModelError(
                    "pass either embedding_dim or embed_dim, not two different values"
                )
            embedding_dim = int(embed_dim)
        if int(num_classes) < 2:
            raise RasterModelError("num_classes must be at least 2")
        if not 1 <= int(embedding_dim) <= 64:
            raise RasterModelError("embedding_dim must be in [1, 64]")
        if int(in_channels) < 1:
            raise RasterModelError("in_channels must be positive")
        if int(base_channels) < 4:
            raise RasterModelError("base_channels must be at least 4")
        if not 1 <= int(depth) <= 5:
            raise RasterModelError("depth must be in [1, 5]")
        if float(input_std) <= 0.0:
            raise RasterModelError("input_std must be positive")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.embed_dim = self.embedding_dim  # backwards-compatible attribute
        self.depth = int(depth)
        widths = tuple(int(base_channels) * (2 ** level) for level in range(self.depth + 1))
        self.widths = widths

        self.stem = _DoubleConv(self.in_channels, widths[0], int(norm_groups))
        self.down = nn.ModuleList()
        self.encoder = nn.ModuleList()
        for level in range(1, self.depth + 1):
            self.down.append(
                nn.Conv2d(
                    widths[level - 1], widths[level], 3, stride=2, padding=1, bias=False
                )
            )
            self.encoder.append(_DoubleConv(widths[level], widths[level], int(norm_groups)))
        self.bottleneck = _DoubleConv(widths[-1], widths[-1], int(norm_groups))

        self.decoder = nn.ModuleList()
        for step in range(self.depth + 1):
            level = self.depth - step
            above = widths[level + 1] if level + 1 <= self.depth else widths[level]
            self.decoder.append(
                _DoubleConv(above + widths[level], widths[level], int(norm_groups))
            )

        self.semantic_head = nn.Conv2d(widths[0], self.num_classes, 1)
        self.embedding_head = nn.Conv2d(widths[0], self.embed_dim, 1)
        self.center_head = nn.Conv2d(widths[0], 1, 1) if bool(center_head) else None

        if bool(normalize_input):
            mean = torch.full((1, self.in_channels, 1, 1), float(input_mean))
            std = torch.full((1, self.in_channels, 1, 1), float(input_std))
            self.register_buffer("input_mean", mean)
            self.register_buffer("input_std", std)
        else:
            self.register_buffer("input_mean", None)
            self.register_buffer("input_std", None)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: Any) -> Dict[str, Any]:
        require_torch()
        if not torch.is_tensor(image):
            raise RasterModelError("image must be a torch.Tensor")
        if image.dim() != 4:
            raise RasterModelError("image must be BxCxHxW")
        if int(image.shape[1]) != self.in_channels:
            raise RasterModelError(
                f"image has {int(image.shape[1])} channels, expected {self.in_channels}"
            )
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if height < 1 or width < 1:
            raise RasterModelError("image spatial size must be positive")

        features = image
        if self.input_mean is not None and self.input_std is not None:
            features = (features - self.input_mean.to(dtype=features.dtype)) / self.input_std.to(
                dtype=features.dtype
            )

        skips = [self.stem(features)]
        for downsample, block in zip(self.down, self.encoder):
            skips.append(block(downsample(skips[-1])))
        features = self.bottleneck(skips[-1])

        for level, block in enumerate(self.decoder):
            skip = skips[-1 - level]
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat([features, skip], dim=1))

        features = F.interpolate(
            features, size=(height, width), mode="bilinear", align_corners=False
        )
        return {
            "semantic_logits": self.semantic_head(features),
            "embeddings": self.embedding_head(features),
            "center_logits": None if self.center_head is None else self.center_head(features),
            "output_size": (height, width),
        }


def build_raster_model(
    config: Optional[Any] = None, **overrides: Any
) -> "RasterUNet":
    """Build a :class:`RasterUNet` from a config, mapping or overrides.

    ``embed_dim`` is accepted as a backwards-compatible alias of
    ``embedding_dim`` in mappings and overrides.
    """

    require_torch()
    allowed = set(RasterModelConfig().to_dict())

    def _normalize(values: Mapping[str, Any]) -> Dict[str, Any]:
        data = dict(values)
        alias = data.pop("embed_dim", None)
        if alias is not None:
            if "embedding_dim" in data and int(data["embedding_dim"]) != int(alias):
                raise RasterModelError(
                    "pass either embedding_dim or embed_dim, not two different values"
                )
            data["embedding_dim"] = alias
        unknown = set(data) - allowed
        if unknown:
            raise RasterModelError(f"unknown config keys: {sorted(unknown)}")
        return data

    if config is None:
        resolved = RasterModelConfig()
    elif isinstance(config, RasterModelConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = RasterModelConfig(**_normalize(config))
    else:
        raise RasterModelError("config must be RasterModelConfig, a mapping or None")
    if overrides:
        resolved = replace(resolved, **_normalize(overrides))
    return RasterUNet(**resolved.to_dict())


__all__ = [
    "EMBED_DIM",
    "IGNORE_INDEX",
    "INPUT_MEAN",
    "INPUT_STD",
    "NO_INSTANCE",
    "NUM_CLASSES",
    "NUM_STUFF_CLASSES",
    "NUM_THING_CLASSES",
    "RasterModelConfig",
    "RasterModelError",
    "RasterUNet",
    "SCHEMA",
    "build_raster_model",
    "class_index_to_raw_semantic",
    "combined_raster_loss",
    "dependency_status",
    "instance_discriminative_loss",
    "instance_embedding_loss",
    "is_thing_class",
    "masked_semantic_loss",
    "raw_semantic_to_class_index",
    "require_torch",
]
