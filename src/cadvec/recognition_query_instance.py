"""Query-mask instance decoder on frozen semantic features (independent lane).

This is a **new, additive** lane.  It changes nothing in the semantic lane, the
instance-affinity lane, the raster model, the vectorizer or the submission
path.  It exists to answer one question with a different topology than
``recognition_instance_affinity``:

    can a small set of learned per-crop instance queries, cross-attending to
    frozen stride-8 semantic features and emitting class-aware masks, split
    touching connected components that a pixel-embedding + agglomeration
    decoder cannot split?

Topology (the *settled default* of the work order, implemented literally):

* frozen semantic wrapper (``outputs/recognition-fullfit-e30-20261111-01/
  checkpoint_fullfit.pt``), every parameter ``requires_grad=False``, always
  ``eval``, batch norm on running stats, feature tap ``net.classifier[3]``
  (ReLU, 256 channels, stride 8, the wrapper normalises ``[0, 1]`` inputs to
  ImageNet statistics internally);
* trainable ``Conv1x1(256 -> 64)`` feature projection;
* ``32`` learned query embeddings, one ``TransformerDecoder`` with ``2``
  layers and ``8`` heads over a **bounded** memory of at most ``32 x 32``
  pooled tokens;
* a ``35``-channel query class head and a per-query mask embedding; mask
  logits are the dot product of the mask embedding with the *unpooled*
  projected feature map, bilinearly upsampled to the crop (training) or to
  native resolution (inference).

Loss (the part that must not be got wrong):

* one Hungarian matching per crop between the ``32`` queries and the
  *masked positive-instance targets* of that crop;
* the mask BCE + Dice domain is exactly ``instance_valid & semantic_valid &
  0 <= class < 30`` (known Thing pixels with instance support).  UNKNOWN
  pixels, unannotated pixels and Stuff pixels are **never** background: they
  are outside the domain and contribute no gradient at all.  Inside the
  domain, a pixel of a *different known annotated instance* is a legitimate
  negative, and so is a pixel of the matched instance that the mask misses;
* the query class loss is a cross-entropy on matched queries only, against the
  matched group's known class.  Unmatched queries get **no** no-object
  negative: annotations are partial at image level, so "unmatched" does not
  mean "background";
* crops with more than ``32`` known instances keep the first ``32`` in
  deterministic ``(class, instance)`` order and the omitted instances'
  pixels are removed from the loss domain as well - never counted as
  background, never dropped as a negative;
* crops with no usable group are skipped **before** any forward pass and
  before any optimizer step.

Inference never reads ground truth.  ``decode_instances`` assigns every
predicted Thing pixel to the argmax query mask among the *class-compatible*
queries (queries whose predicted class equals the pixel's class); pixels of a
class no query predicts fall back to the unique CC8 baseline id, offset by
``num_queries`` so ids stay globally unique.  Nothing is deleted: Stuff keeps
``instance = -1`` and pixel classes are copied through bit-identically, and
mask assignment may split or merge a CC.

Tiling: query identity across tiles is the one thing this lane must not get
wrong.  Instead of stitching per-tile queries, the frozen stride-8 feature map
is accumulated **once per image** over the existing 256/128 tile grid (white
padding at the bottom/right, the same positions and the same averaging as
``recognition_pretrained_fullfit.predict_tiled_image``), and the query decoder
runs once on that image-level map: the transformer only ever sees the pooled
``<= 32 x 32`` token memory, while masks are dotted at the full stride-8
resolution and upsampled natively.  There is therefore no tile query identity
problem and no IoU-based cross-tile query matching, and no threshold sweep.

Train/inference context gap (**documented, not papered over**):

* training: 256 px crops -> 32 x 32 stride-8 features -> memory is already
  32 x 32 (identity pool) -> masks at 32 x 32, upsampled to 256 for the loss;
* inference: full image (e.g. 1000 x 1000 -> 125 x 125 stride-8 features)
  -> memory average-pooled to 32 x 32 -> masks dotted at 125 x 125 and
  upsampled natively.

So the transformer always sees a 32 x 32 grid, but the mask dot product sees
4x more spatial detail at inference than at training, and the crop statistics
in training are 256 px windows rather than whole plans.  That gap is a real
risk for this topology and is recorded in the checkpoint contract.

CLI::

    python -m cadvec.recognition_query_instance prepare --help   # CPU plan
    python -m cadvec.recognition_query_instance measure --help   # CPU window + guard
    python -m cadvec.recognition_query_instance ready   --help   # CPU readiness bundle
    python -m cadvec.recognition_query_instance smoke   --help   # CPU smoke
    python -m cadvec.recognition_query_instance fit     --help   # scheduled

``prepare``, ``measure``, ``ready`` and ``smoke`` are CPU-only.  ``fit`` refuses
any role but ``fit`` and is never launched from here: the parent schedules it
after the GPU review.
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

QUERY_SCHEMA = "cadvec.recognition.query_instance/1"
PREPARE_SCHEMA = "cadvec.recognition.query_instance_prepare/1"
FIT_SCHEMA = "cadvec.recognition.query_instance_fit/1"
SMOKE_SCHEMA = "cadvec.recognition.query_instance_smoke/1"
CHECKPOINT_SCHEMA = "cadvec.recognition.query_instance_checkpoint/1"
DECODE_SCHEMA = "cadvec.recognition.query_instance_decode/1"
MEASURE_SCHEMA = "cadvec.recognition.query_instance_measure/1"
FOOTPRINT_SCHEMA = "cadvec.recognition.query_instance_footprint/1"
GPU_PREFLIGHT_SCHEMA = "cadvec.recognition.query_instance_gpu_preflight/1"
READY_SCHEMA = "cadvec.recognition.query_instance_ready/1"
QUERY_VERSION = 1

# --- frozen feature geometry ------------------------------------------------
NUM_CLASSES = pilot.NUM_CLASSES  # 35 class indices, 0..34
NUM_THING_CLASSES = raster_model.NUM_THING_CLASSES  # 30, indices 0..29
FEATURE_CHANNELS = 256  # net.classifier[3] ReLU output
FEATURE_STRIDE = 8
IGNORE_CLASS = NUM_CLASSES  # raw sentinel accepted by raster_infer.instance_ids

# --- head geometry (fixed by the work order) --------------------------------
FEAT_DIM = 64
MASK_DIM = 64
NUM_QUERIES = 32
DECODER_LAYERS = 2
DECODER_HEADS = 8
DECODER_FF = 128
MEMORY_GRID = 32  # bounded transformer memory: at most 32 x 32 tokens
HEAD_INIT_SEED = 20261111

# --- supervision / loss -----------------------------------------------------
MAX_INSTANCES = NUM_QUERIES  # cap, deterministic, never a negative
MIN_INSTANCE_PIXELS = 4
MIN_GROUPS_PER_CROP = 2
MATCH_CLASS_WEIGHT = 1.0
MATCH_BCE_WEIGHT = 5.0
MATCH_DICE_WEIGHT = 5.0
LOSS_CLASS_WEIGHT = 1.0
LOSS_BCE_WEIGHT = 5.0
LOSS_DICE_WEIGHT = 5.0

# --- training (mirrors the frozen full-fit lane) ----------------------------
CROP_SIZE = fullfit.CROP_SIZE
CROPS_PER_RECORD = fullfit.CROPS_PER_RECORD
TILE_SIZE = fullfit.TILE_SIZE
TILE_STRIDE = fullfit.TILE_STRIDE
EPOCHS = fullfit.EPOCHS
ACCUMULATION_STEPS = fullfit.ACCUMULATION_STEPS
BASE_SEED = fullfit.BASE_SEED
EPOCH_SEED_STRIDE = fullfit.EPOCH_SEED_STRIDE
LR = 1e-4
WEIGHT_DECAY = 0.01

MIN_ELIGIBLE_CROPS_PER_EPOCH = 200

DEFAULT_MANIFEST = os.path.join(
    "outputs", "recognition-gate3b-fit-tune-20261108-01", "recognition_public_manifest.json"
)
DEFAULT_CACHE_DIR = os.path.join(
    "outputs", "recognition-pretrained-fullfit-20261111-01", "cache"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    "outputs", "recognition-query-instance-20261111-01"
)
DEFAULT_SEMANTIC_CHECKPOINT = os.path.join(
    "outputs", "recognition-fullfit-e30-20261111-01", "checkpoint_fullfit.pt"
)


class QueryInstanceError(pilot.PilotError):
    """Invalid configuration, cache record, mask target or decoder input."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:
        raise QueryInstanceError("numpy is required for this lane")
    return np


def _require_torch() -> Any:
    if torch is None:
        raise QueryInstanceError("torch is required for this lane")
    return torch


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QueryInstanceError(f"{name} must be an int >= {minimum}; got {value!r}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise QueryInstanceError(f"{name} must be finite; got {value!r}")
    return number


def _name_seed(*parts: Any) -> int:
    material = "\x00".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & (2**63 - 1)


def _tile_positions(length: int, tile: int, stride: int) -> List[int]:
    """Same positions as the frozen full-fit tiler (shared helper)."""

    return fullfit._tile_positions(length, tile, stride)


# ---------------------------------------------------------------------------
# frozen semantic model + feature tap
# ---------------------------------------------------------------------------


def semantic_parameter_digest(model: Any) -> Any:
    """Practical mutation alarm over every state-dict entry (see affinity)."""

    return affinity.semantic_parameter_digest(model)


def freeze_semantic_model(model: Any) -> Dict[str, Any]:
    """Freeze every semantic parameter, force eval and freeze batch norm."""

    return affinity.freeze_semantic_model(model)


class FrozenFeatureTap:
    """Read-only stride-8 feature tap on the frozen semantic wrapper.

    Reuses the reviewed frozen-feature plumbing of
    :mod:`cadvec.recognition_instance_affinity` (the ``net.classifier[3]``
    ReLU tap, forced ``eval``, frozen batch norm).  Only features and semantic
    logits leave this object; no gradient ever flows into the wrapper.
    """

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        _require_torch()
        self.model = model
        self.device = str(device)
        self.frozen_report = freeze_semantic_model(model)
        self._tap: Dict[str, Any] = {}
        self._handle = affinity._classifier_relu(model).register_forward_hook(self._hook)

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

    def feature_map(self, images: Any) -> Dict[str, Any]:
        """Stride-8 features (no grad) and semantic logits for NCHW images."""

        torch = _require_torch()
        if not torch.is_tensor(images):
            raise QueryInstanceError("images must be a torch tensor")
        self.model.eval()
        self._tap.pop("features", None)
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            features = self._tap.pop("features", None)
        if features is None:
            raise QueryInstanceError("classifier[3] feature tap never fired")
        logits = outputs.get("semantic_logits") if isinstance(outputs, Mapping) else None
        return {
            "features": features.detach(),
            "semantic_logits": None if logits is None else logits.detach(),
        }

    def native_feature_map(
        self,
        image: Any,
        *,
        tile_size: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
    ) -> Dict[str, Any]:
        """Accumulate stride-8 features over the tiled image.

        The image is padded to a whole number of tiles with white (``1.0``),
        the tile grid and the averaging are the frozen lane's, and each tile
        contributes its own ``tile/8`` square feature block at ``offset // 8``.
        Memory is ``256 x Hf x Wf`` float32 (~16 MB for a 1000 x 1000 plan),
        with at most one tile's activation alive at a time.  Semantic logits
        are *not* accumulated here: this lane never decodes pixel classes, it
        consumes the frozen lane's saved semantic index.
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
            raise QueryInstanceError(f"image must be 3 x H x W; got {tuple(source.shape)}")
        if not bool(torch.isfinite(source).all()):
            raise QueryInstanceError("non_finite_image")
        lowest = float(source.min())
        highest = float(source.max())
        if lowest < -1e-6 or highest > 1.0 + 1e-6:
            raise QueryInstanceError(
                "image must be UNIT_RANGE [0, 1] float RGB (no /255 here): "
                f"got min {lowest:.4f}, max {highest:.4f}"
            )
        height, width = int(source.shape[-2]), int(source.shape[-1])
        if height <= 0 or width <= 0:
            raise QueryInstanceError("empty image")
        tile_size = _positive_int("tile_size", tile_size)
        stride = _positive_int("stride", stride)
        if stride > tile_size:
            raise QueryInstanceError("stride must not exceed tile size")
        if tile_size % FEATURE_STRIDE or stride % FEATURE_STRIDE:
            raise QueryInstanceError("tile size and stride must be multiples of 8")
        cells = tile_size // FEATURE_STRIDE
        padded_height = max(int(tile_size), height)
        padded_width = max(int(tile_size), width)
        padded_height += (-padded_height) % FEATURE_STRIDE
        padded_width += (-padded_width) % FEATURE_STRIDE
        padded = source
        if padded_height != height or padded_width != width:
            padded = F.pad(
                source.unsqueeze(0),
                (0, padded_width - width, 0, padded_height - height),
                mode="constant",
                value=1.0,
            ).squeeze(0)
        rows = _tile_positions(padded_height, tile_size, stride)
        columns = _tile_positions(padded_width, tile_size, stride)
        grid_h = padded_height // FEATURE_STRIDE
        grid_w = padded_width // FEATURE_STRIDE

        features_accum: Optional[Any] = None
        weights: Optional[Any] = None
        self.model.eval()
        with torch.no_grad():
            for row in rows:
                for column in columns:
                    window = padded[:, row : row + tile_size, column : column + tile_size]
                    tile_features = self.feature_map(
                        window.unsqueeze(0).to(self.device)
                    )["features"][0].detach().to("cpu", dtype=torch.float32)
                    if tuple(tile_features.shape[-2:]) != (cells, cells):
                        raise QueryInstanceError(
                            "unexpected tile feature shape: "
                            f"{tuple(tile_features.shape)} for a {tile_size} tile"
                        )
                    if features_accum is None:
                        features_accum = torch.zeros(
                            (FEATURE_CHANNELS, grid_h, grid_w), dtype=torch.float32
                        )
                        weights = torch.zeros((1, grid_h, grid_w), dtype=torch.float32)
                    top = row // FEATURE_STRIDE
                    left = column // FEATURE_STRIDE
                    features_accum[:, top : top + cells, left : left + cells] += tile_features
                    weights[:, top : top + cells, left : left + cells] += 1.0
        if features_accum is None or weights is None:  # pragma: no cover - defensive
            raise QueryInstanceError("no tiles were produced")
        # The tile grid is padded to whole tiles (white 1.0), but the returned
        # map must cover exactly the image content: ceil(H/8) x ceil(W/8).
        content_h = -(-height // FEATURE_STRIDE)
        content_w = -(-width // FEATURE_STRIDE)
        if content_h > grid_h or content_w > grid_w:  # pragma: no cover - defensive
            raise QueryInstanceError("feature grid smaller than the image content")
        result = {
            "features": (features_accum / weights)[:, :content_h, :content_w],
            "feature_grid": [content_h, content_w],
            "padded_feature_grid": [grid_h, grid_w],
            "image_size": [height, width],
            "tile_rows": rows,
            "tile_columns": columns,
            "tile_count": int(len(rows) * len(columns)),
            "tile_size": int(tile_size),
            "stride": int(stride),
        }
        return result


# ---------------------------------------------------------------------------
# query head
# ---------------------------------------------------------------------------


def _deterministic_init(module: Any, seed: int) -> None:
    """Deterministic per-parameter init from ``(seed, parameter name)``.

    The global RNG is seeded before the head is built as well; this pass makes
    the tensor values independent of *when* the head is constructed, so an
    epoch reseed can never change the initial head.
    """

    torch = _require_torch()
    for name, parameter in module.named_parameters():
        generator = torch.Generator(device="cpu").manual_seed(_name_seed(seed, name))
        with torch.no_grad():
            if parameter.dim() == 1:
                if "weight" in name:  # LayerNorm gain
                    parameter.fill_(1.0)
                else:  # every bias
                    parameter.zero_()
                continue
            fan_in = int(parameter.shape[1]) * int(
                math.prod(int(v) for v in parameter.shape[2:])
            )
            bound = 1.0 / math.sqrt(float(max(1, fan_in)))
            parameter.uniform_(-bound, bound, generator=generator)


def _pos_2d(height: int, width: int, dim: int, *, device: Any, dtype: Any) -> Any:
    """Fixed 2D sinusoidal positional encoding, ``height*width x dim``."""

    torch = _require_torch()
    channels = max(1, dim // 4)
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    omega = torch.exp(
        -math.log(10000.0)
        * torch.arange(channels, device=device, dtype=torch.float32)
        / float(channels)
    )
    scale = math.pi
    encoded_y = (grid_y.reshape(-1, 1) * scale) * omega
    encoded_x = (grid_x.reshape(-1, 1) * scale) * omega
    pieces = [encoded_y.sin(), encoded_y.cos(), encoded_x.sin(), encoded_x.cos()]
    encoding = torch.cat(pieces, dim=1)
    if encoding.shape[1] < dim:  # pragma: no cover - only for odd small dims
        pad = torch.zeros(
            (encoding.shape[0], dim - encoding.shape[1]), device=device, dtype=torch.float32
        )
        encoding = torch.cat([encoding, pad], dim=1)
    return encoding[:, :dim].to(dtype).reshape(height * width, dim)


class QueryMaskHead:
    """Trainable query decoder: 1x1 projection + 32 queries + 2 decoder layers.

    A plain ``torch.nn.Module`` subclass declared at runtime so that importing
    this module never requires ``torch`` (the lane keeps the project's
    dependency-free import surface).  ``forward`` returns a dict with
    ``class_logits`` ``(B, Q, 35)``, ``mask_embed`` ``(B, Q, M)``,
    ``mask_logits`` ``(B, Q, h, w)`` at stride-8 feature resolution and the
    pooled ``memory_grid`` actually used.
    """

    @staticmethod
    def build(
        *,
        in_channels: int = FEATURE_CHANNELS,
        feat_dim: int = FEAT_DIM,
        mask_dim: int = MASK_DIM,
        num_queries: int = NUM_QUERIES,
        num_classes: int = NUM_CLASSES,
        layers: int = DECODER_LAYERS,
        heads: int = DECODER_HEADS,
        ff: int = DECODER_FF,
        seed: int = HEAD_INIT_SEED,
        device: str = "cpu",
    ) -> Any:
        torch = _require_torch()
        module_class = _query_mask_module_class()
        head = module_class(
            in_channels=int(in_channels),
            feat_dim=int(feat_dim),
            mask_dim=int(mask_dim),
            num_queries=int(num_queries),
            num_classes=int(num_classes),
            layers=int(layers),
            heads=int(heads),
            ff=int(ff),
        )
        _deterministic_init(head, int(seed))
        return head.to(device)


def _query_mask_module_class() -> Any:
    """Build (once) the torch Module subclass for the query head."""

    cached = globals().get("_QUERY_MASK_MODULE")
    if cached is not None:
        return cached
    torch = _require_torch()
    nn = torch.nn

    class _QueryMaskModule(nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            feat_dim: int,
            mask_dim: int,
            num_queries: int,
            num_classes: int,
            layers: int,
            heads: int,
            ff: int,
        ) -> None:
            super().__init__()
            if feat_dim % heads:
                raise QueryInstanceError("feat_dim must be divisible by the head count")
            self.in_channels = int(in_channels)
            self.feat_dim = int(feat_dim)
            self.mask_dim = int(mask_dim)
            self.num_queries = int(num_queries)
            self.num_classes = int(num_classes)
            self.layers = int(layers)
            self.proj = nn.Conv2d(int(in_channels), int(feat_dim), kernel_size=1)
            self.query_embed = nn.Embedding(int(num_queries), int(feat_dim))
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=int(feat_dim),
                nhead=int(heads),
                dim_feedforward=int(ff),
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(layers))
            self.class_head = nn.Linear(int(feat_dim), int(num_classes))
            self.mask_head = nn.Sequential(
                nn.Linear(int(feat_dim), int(mask_dim)),
                nn.ReLU(),
                nn.Linear(int(mask_dim), int(mask_dim)),
            )
            self.decoder_norm = nn.LayerNorm(int(feat_dim))

        def pooled_memory(self, features: Any) -> Any:
            proj = self.proj(features)
            height, width = int(proj.shape[-2]), int(proj.shape[-1])
            target_h = min(height, MEMORY_GRID)
            target_w = min(width, MEMORY_GRID)
            if (target_h, target_w) != (height, width):
                memory_map = F.adaptive_avg_pool2d(proj, (target_h, target_w))
            else:
                memory_map = proj
            tokens = memory_map.flatten(2).transpose(1, 2)
            position = _pos_2d(
                target_h,
                target_w,
                self.feat_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + position.unsqueeze(0)
            return proj, memory_map, tokens, (target_h, target_w)

        def forward(self, features: Any) -> Dict[str, Any]:
            proj, memory_map, tokens, grid = self.pooled_memory(features)
            batch = int(tokens.shape[0])
            queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
            decoded = self.decoder(queries, tokens)
            decoded = self.decoder_norm(decoded)
            class_logits = self.class_head(decoded)
            mask_embed = self.mask_head(decoded)
            mask_logits = torch.einsum("bqm,bdhw->bqhw", mask_embed, proj)
            return {
                "class_logits": class_logits,
                "mask_embed": mask_embed,
                "mask_logits": mask_logits,
                "queries": decoded,
                "memory_grid": [int(grid[0]), int(grid[1])],
                "feature_grid": [int(proj.shape[-2]), int(proj.shape[-1])],
            }

    _QueryMaskModule.__name__ = "QueryMaskModule"
    _QueryMaskModule.__qualname__ = "QueryMaskModule"
    globals()["_QUERY_MASK_MODULE"] = _QueryMaskModule
    return _QueryMaskModule


def build_query_head(
    *,
    seed: int = HEAD_INIT_SEED,
    device: str = "cpu",
    in_channels: int = FEATURE_CHANNELS,
    feat_dim: int = FEAT_DIM,
    mask_dim: int = MASK_DIM,
    num_queries: int = NUM_QUERIES,
    num_classes: int = NUM_CLASSES,
    layers: int = DECODER_LAYERS,
    heads: int = DECODER_HEADS,
    ff: int = DECODER_FF,
) -> Any:
    """Deterministic query head (no dependence on when it is constructed)."""

    _require_torch()
    _query_mask_module_class()
    return QueryMaskHead.build(
        in_channels=in_channels,
        feat_dim=feat_dim,
        mask_dim=mask_dim,
        num_queries=num_queries,
        num_classes=num_classes,
        layers=layers,
        heads=heads,
        ff=ff,
        seed=seed,
        device=device,
    )


def head_parameter_count(head: Any) -> Dict[str, int]:
    total = int(sum(p.numel() for p in head.parameters()))
    trainable = int(sum(p.numel() for p in head.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# mask targets: known-instance groups, capped deterministically
# ---------------------------------------------------------------------------


def instance_groups(
    semantic: Any,
    semantic_valid: Any,
    instance: Any,
    instance_valid: Any,
    *,
    max_instances: int = MAX_INSTANCES,
    min_pixels: int = MIN_INSTANCE_PIXELS,
    num_thing_classes: int = NUM_THING_CLASSES,
) -> Dict[str, Any]:
    """Masked positive-instance targets and the mask-loss domain.

    * a pixel is eligible only when ``instance_valid`` **and**
      ``semantic_valid`` both say so **and** the instance id is ``>= 0``
      (``0`` is legal) **and** the class source marks a known Thing class
      ``0..29``; the two validity masks are never derived from each other;
    * groups are keyed by ``(class, instance)`` and sorted by that key, so the
      deterministic cap is stable across runs;
    * ``domain`` is the union of the pixels of the *kept* groups: omitted
      instances (beyond ``max_instances``) and groups below ``min_pixels`` are
      removed from the domain entirely, so they can never act as negatives;
    * UNKNOWN, unannotated and Stuff pixels are never in the domain.
    """

    numpy = _require_numpy()
    semantic = numpy.asarray(semantic)
    valid_sem = numpy.asarray(semantic_valid, dtype=bool)
    instance = numpy.asarray(instance)
    valid_inst = numpy.asarray(instance_valid, dtype=bool)
    if semantic.shape != instance.shape:
        raise QueryInstanceError("semantic and instance must share a shape")
    if valid_sem.shape != semantic.shape or valid_inst.shape != semantic.shape:
        raise QueryInstanceError("validity masks must match the label shape")
    if semantic.ndim != 2:
        raise QueryInstanceError("labels must be 2D")
    max_instances = _positive_int("max_instances", max_instances)
    min_pixels = _positive_int("min_pixels", min_pixels)
    height, width = int(semantic.shape[0]), int(semantic.shape[1])

    eligible = (
        valid_inst
        & valid_sem
        & (instance >= 0)
        & (semantic >= 0)
        & (semantic < int(num_thing_classes))
    )
    empty = {
        "domain": numpy.zeros((height, width), dtype=bool),
        "masks": numpy.zeros((0, height, width), dtype=bool),
        "classes": numpy.zeros((0,), dtype=numpy.int64),
        "keys": [],
        "counts": [],
        "group_count": 0,
        "domain_pixels": 0,
        "capped": False,
        "omitted_groups": 0,
        "omitted_pixels": 0,
        "below_min_pixels_groups": 0,
        "below_min_pixels": 0,
        "candidate_pixels": 0,
    }
    if not bool(eligible.any()):
        return empty

    classes_at = semantic[eligible].astype(numpy.int64)
    instances_at = instance[eligible].astype(numpy.int64)
    span = int(instances_at.max()) + 2
    keys_at = classes_at * span + instances_at
    unique_keys, inverse, counts = numpy.unique(
        keys_at, return_inverse=True, return_counts=True
    )
    group_classes = unique_keys // span
    group_instances = unique_keys % span
    order = numpy.lexsort((group_instances, group_classes))
    unique_keys = unique_keys[order]
    group_classes = group_classes[order]
    counts = counts[order]
    remap = numpy.empty_like(order)
    remap[order] = numpy.arange(order.size)
    position = remap[inverse]

    keep = numpy.flatnonzero(counts >= int(min_pixels))
    below = numpy.setdiff1d(numpy.arange(unique_keys.size), keep)
    capped = bool(keep.size > max_instances)
    kept = keep[: int(max_instances)] if capped else keep
    omitted = numpy.concatenate([keep[int(max_instances) :], below]) if capped else below

    keep_lookup = numpy.zeros((unique_keys.size,), dtype=bool)
    keep_lookup[kept] = True
    domain = eligible.copy()
    domain[eligible] = keep_lookup[position]
    if not bool(domain.any()):
        return empty

    masks = numpy.zeros((int(kept.size), height, width), dtype=bool)
    rows, columns = numpy.nonzero(domain)
    slot = numpy.full((unique_keys.size,), -1, dtype=numpy.int64)
    slot[kept] = numpy.arange(int(kept.size))
    assigned = slot[position[domain[eligible]]]
    if bool((assigned < 0).any()):  # pragma: no cover - defensive
        raise QueryInstanceError("domain pixel without a kept group slot")
    masks[assigned, rows, columns] = True

    omitted_pixels = int(
        eligible.sum() - domain.sum()
    )
    return {
        "domain": domain,
        "masks": masks,
        "classes": group_classes[kept].astype(numpy.int64),
        "instances": group_instances[kept].astype(numpy.int64),
        "keys": [
            (int(class_id), int(instance_id))
            for class_id, instance_id in zip(
                group_classes[kept].tolist(), group_instances[kept].tolist()
            )
        ],
        "counts": [int(v) for v in counts[kept].tolist()],
        "group_count": int(kept.size),
        "domain_pixels": int(domain.sum()),
        "capped": bool(capped),
        "candidate_groups": int(unique_keys.size),
        "omitted_groups": int(omitted.size),
        "omitted_pixels": int(omitted_pixels),
        "below_min_pixels_groups": int(below.size),
        "below_min_pixels": int(
            int(counts[below].sum()) if below.size else 0
        ),
        "candidate_pixels": int(eligible.sum()),
    }


def crop_is_supervised(
    crop: Mapping[str, Any],
    *,
    min_groups: int = MIN_GROUPS_PER_CROP,
    max_instances: int = MAX_INSTANCES,
    min_pixels: int = MIN_INSTANCE_PIXELS,
) -> Tuple[bool, Dict[str, Any]]:
    """Return ``(usable, group report)`` without touching the model."""

    if bool(crop.get("skip_training")):
        return False, {
            "skip_reason": "zero_supervision_crop",
            "group_count": 0,
            "domain_pixels": 0,
        }
    report = instance_groups(
        crop["semantic"],
        crop["semantic_valid"],
        crop["instance"],
        crop["instance_valid"],
        max_instances=max_instances,
        min_pixels=min_pixels,
    )
    usable = int(report["group_count"]) >= int(min_groups)
    if not usable:
        report = dict(report)
        report["skip_reason"] = "fewer_than_min_groups"
    return usable, report


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------


def _hungarian_bound(matrix: Any) -> float:
    """Positive finite sentinel larger than any achievable partial total.

    The sentinel must exceed every candidate reduced cost in magnitude, and it
    must stay positive: with all-negative costs ``max(cost) * k + 1`` would be
    negative and the augmenting search would silently break.  Scaling the
    largest absolute entry keeps the bound valid for costs of either sign while
    avoiding ``inf`` (which would make the potential updates ``inf - inf``).
    """

    numpy = _require_numpy()
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    magnitude = float(numpy.abs(matrix).max()) if matrix.size else 0.0
    return max(magnitude * (rows + columns + 2) + 1.0, 1.0)


def linear_sum_assignment(cost: Any) -> List[Tuple[int, int]]:
    """Minimum-cost assignment for a rectangular ``(n, m)`` cost matrix.

    Classic ``O(n^2 m)`` shortest-augmenting-path Hungarian algorithm with
    potentials, minimising the total cost.  When ``n > m`` the matrix is
    transposed and the pairs are swapped back.  Ties resolve to the lowest row
    then lowest column, so repeated calls are exactly reproducible.
    """

    numpy = _require_numpy()
    matrix = numpy.asarray(cost, dtype=numpy.float64)
    if matrix.ndim != 2:
        raise QueryInstanceError("cost must be a 2D matrix")
    if not numpy.isfinite(matrix).all():
        raise QueryInstanceError("cost must be finite")
    transposed = int(matrix.shape[0]) > int(matrix.shape[1])
    if transposed:
        matrix = matrix.T
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    if rows == 0 or columns == 0:
        return []
    infinity = _hungarian_bound(matrix)
    u = numpy.zeros(rows + 1, dtype=numpy.float64)
    v = numpy.zeros(columns + 1, dtype=numpy.float64)
    p = numpy.zeros(columns + 1, dtype=numpy.int64)  # p[j] = matched row of column j
    way = numpy.zeros(columns + 1, dtype=numpy.int64)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = numpy.full(columns + 1, infinity, dtype=numpy.float64)
        used = numpy.zeros(columns + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = infinity
            j1 = -1
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:  # pragma: no cover - defensive
                raise QueryInstanceError("hungarian search exhausted")
            for j in range(0, columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if int(p[j0]) == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs: List[Tuple[int, int]] = []
    for j in range(1, columns + 1):
        if int(p[j]) != 0:
            row = int(p[j]) - 1
            column = j - 1
            pairs.append((column, row) if transposed else (row, column))
    pairs.sort()
    return pairs


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def query_instance_loss(
    mask_logits: Any,
    class_logits: Any,
    groups: Mapping[str, Any],
    *,
    crop_size: Optional[int] = None,
    match_class_weight: float = MATCH_CLASS_WEIGHT,
    match_bce_weight: float = MATCH_BCE_WEIGHT,
    match_dice_weight: float = MATCH_DICE_WEIGHT,
    loss_class_weight: float = LOSS_CLASS_WEIGHT,
    loss_bce_weight: float = LOSS_BCE_WEIGHT,
    loss_dice_weight: float = LOSS_DICE_WEIGHT,
) -> Dict[str, Any]:
    """Hungarian-matched query-mask loss on the known-instance domain only.

    ``mask_logits`` is ``Q x h x w`` at feature resolution, ``class_logits`` is
    ``Q x 35``.  Every reduction happens over ``domain`` alone, so no UNKNOWN,
    unannotated or Stuff pixel becomes a negative.  Unmatched queries
    contribute nothing at all (no no-object term).
    """

    torch = _require_torch()
    numpy = _require_numpy()
    if not torch.is_tensor(mask_logits) or not torch.is_tensor(class_logits):
        raise QueryInstanceError("mask_logits and class_logits must be tensors")
    queries = int(mask_logits.shape[0])
    if tuple(class_logits.shape) != (queries, NUM_CLASSES):
        raise QueryInstanceError(
            f"class_logits must be {queries} x {NUM_CLASSES}; got {tuple(class_logits.shape)}"
        )
    gt_masks = numpy.asarray(groups["masks"], dtype=bool)
    gt_classes = numpy.asarray(groups["classes"], dtype=numpy.int64)
    domain = numpy.asarray(groups["domain"], dtype=bool)
    if gt_masks.shape[0] != gt_classes.size:
        raise QueryInstanceError("group mask/class count mismatch")
    if gt_masks.shape[1:] != domain.shape:
        raise QueryInstanceError("group masks and domain must share a spatial shape")
    zero = mask_logits.sum() * 0.0 + class_logits.sum() * 0.0
    if gt_masks.shape[0] == 0 or int(domain.sum()) == 0:
        return {
            "loss": zero,
            "mask_logits": mask_logits,
            "matches": [],
            "matched_pairs": 0,
            "domain_pixels": int(domain.sum()),
            "gt_groups": int(gt_masks.shape[0]),
            "bce": 0.0,
            "dice": 0.0,
            "class": 0.0,
            "skipped": True,
        }

    height, width = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])
    if crop_size is not None and (height, width) != (int(crop_size), int(crop_size)):
        upsampled = F.interpolate(
            mask_logits.unsqueeze(0),
            size=(int(crop_size), int(crop_size)),
            mode="bilinear",
            align_corners=False,
        )[0]
    elif (height, width) != tuple(domain.shape):
        upsampled = F.interpolate(
            mask_logits.unsqueeze(0),
            size=tuple(domain.shape),
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        upsampled = mask_logits

    domain_index = torch.from_numpy(numpy.flatnonzero(domain.reshape(-1))).to(
        upsampled.device
    )
    flat_logits = upsampled.reshape(queries, -1)[:, domain_index]  # Q x N
    targets = torch.from_numpy(
        numpy.ascontiguousarray(
            gt_masks.reshape(gt_masks.shape[0], -1)[:, domain.reshape(-1)]
        )
    ).to(device=upsampled.device, dtype=upsampled.dtype)  # G x N
    groups_n = int(flat_logits.shape[1])
    with torch.no_grad():
        probabilities = torch.sigmoid(flat_logits).detach()
        log_probability = F.logsigmoid(flat_logits).detach()
        log_one_minus = F.logsigmoid(-flat_logits).detach()
        target_f = targets.detach()
        bce_matrix = -(
            log_probability @ target_f.transpose(0, 1)
            + log_one_minus @ (1.0 - target_f).transpose(0, 1)
        ) / float(groups_n)
        intersection = probabilities @ target_f.transpose(0, 1)
        prob_mass = probabilities.sum(dim=1, keepdim=True)
        target_mass = target_f.sum(dim=1, keepdim=True).transpose(0, 1)
        dice_matrix = 1.0 - (2.0 * intersection + 1.0) / (
            prob_mass + target_mass + 1.0
        )
        class_probability = torch.softmax(class_logits.detach(), dim=-1)
        class_cost = -class_probability[:, torch.from_numpy(gt_classes).to(
            device=class_probability.device, dtype=torch.long
        )]
        cost = (
            float(match_class_weight) * class_cost
            + float(match_bce_weight) * bce_matrix
            + float(match_dice_weight) * dice_matrix
        )
    pairs = linear_sum_assignment(cost.detach().to("cpu").to(torch.float64).numpy())
    if not pairs:
        return {
            "loss": zero,
            "mask_logits": mask_logits,
            "matches": [],
            "matched_pairs": 0,
            "domain_pixels": groups_n,
            "gt_groups": int(gt_masks.shape[0]),
            "bce": 0.0,
            "dice": 0.0,
            "class": 0.0,
            "skipped": True,
        }
    query_index = torch.tensor(
        [pair[0] for pair in pairs], device=flat_logits.device, dtype=torch.long
    )
    group_index = torch.tensor(
        [pair[1] for pair in pairs], device=flat_logits.device, dtype=torch.long
    )
    matched_logits = flat_logits[query_index]  # P x N
    matched_target = targets[group_index]  # P x N
    matched_probability = torch.sigmoid(matched_logits)
    bce = F.binary_cross_entropy_with_logits(
        matched_logits, matched_target, reduction="mean"
    )
    intersection = (matched_probability * matched_target).sum(dim=1)
    denominator = (
        matched_probability.sum(dim=1)
        + matched_target.sum(dim=1)
    )
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    matched_classes = torch.from_numpy(gt_classes[group_index.detach().cpu().numpy()]).to(
        device=class_logits.device, dtype=torch.long
    )
    class_loss = F.cross_entropy(
        class_logits[query_index], matched_classes, reduction="mean"
    )
    loss = (
        float(loss_class_weight) * class_loss
        + float(loss_bce_weight) * bce
        + float(loss_dice_weight) * dice
    )
    if not bool(torch.isfinite(loss)):
        raise QueryInstanceError("non_finite_query_instance_loss")
    return {
        "loss": loss,
        "mask_logits": upsampled,
        "matches": [
            {
                "query": int(pair[0]),
                "group": int(pair[1]),
                "class": int(gt_classes[pair[1]]),
                "pixels": int((gt_masks[pair[1]]).sum()),
            }
            for pair in pairs
        ],
        "matched_pairs": int(len(pairs)),
        "domain_pixels": groups_n,
        "gt_groups": int(gt_masks.shape[0]),
        "bce": float(bce.detach().item()),
        "dice": float(dice.detach().item()),
        "class": float(class_loss.detach().item()),
        "loss_value": float(loss.detach().item()),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# label-free instances from query masks
# ---------------------------------------------------------------------------


def decode_instances(
    pred_semantic: Any,
    mask_probability: Any,
    query_class: Any,
    *,
    num_queries: Optional[int] = None,
    num_thing_classes: int = NUM_THING_CLASSES,
    fallback_offset: Optional[int] = None,
) -> Dict[str, Any]:
    """Assign every predicted Thing pixel to a query mask or a CC8 fallback.

    * only pixels whose predicted class is a known Thing class ``0..29`` take
      an instance id; Stuff keeps ``-1`` and pixel classes are copied through
      unchanged (nothing is deleted);
    * a Thing pixel goes to the argmax mask among queries **whose predicted
      class equals the pixel's class**; ties resolve to the lowest query index;
    * a class that no query predicts falls back to the unique CC8 id of the
      predicted Thing map; fallback ids start at ``num_queries`` and are handed
      out by one running allocator for the whole image, so two classes that both
      fall back can never share an id;
    * no threshold, no ground truth, no labels.
    """

    numpy = _require_numpy()
    semantic = numpy.asarray(pred_semantic)
    if semantic.ndim != 2 or not numpy.issubdtype(semantic.dtype, numpy.integer):
        raise QueryInstanceError("pred_semantic must be a 2D integer array")
    probability = numpy.asarray(mask_probability, dtype=numpy.float32)
    if probability.ndim != 3:
        raise QueryInstanceError("mask_probability must be Q x H x W")
    if tuple(probability.shape[1:]) != tuple(semantic.shape):
        raise QueryInstanceError("mask_probability must match the semantic shape")
    classes = numpy.asarray(query_class).reshape(-1).astype(numpy.int64)
    queries = int(probability.shape[0])
    if classes.size != queries:
        raise QueryInstanceError("query_class must hold one class per query")
    if num_queries is not None and int(num_queries) != queries:
        raise QueryInstanceError("num_queries does not match the mask stack")
    if not numpy.isfinite(probability).all():
        raise QueryInstanceError("non_finite_mask_probability")
    offset = queries if fallback_offset is None else _positive_int(
        "fallback_offset", int(fallback_offset)
    )
    if offset < queries:
        raise QueryInstanceError(
            "fallback_offset must be at least num_queries so query ids and "
            f"fallback ids cannot collide (got {offset} < {queries})"
        )

    thing = (semantic >= 0) & (semantic < int(num_thing_classes))
    out = numpy.full(semantic.shape, -1, dtype=numpy.int32)
    by_query = numpy.zeros((queries,), dtype=numpy.int64)
    fallback_pixels = 0
    fallback_ids = 0
    # One running allocator for the whole image: each class's CC8 fallback ids
    # continue where the previous class stopped, so two classes that both fall
    # back can never share an id.
    next_fallback = int(offset)
    per_class: Dict[str, Any] = {}
    for class_id in range(int(num_thing_classes)):
        pixels = thing & (semantic == int(class_id))
        if not bool(pixels.any()):
            continue
        compatible = numpy.flatnonzero(classes == int(class_id))
        record: Dict[str, Any] = {
            "pixels": int(pixels.sum()),
            "queries": int(compatible.size),
            "fallback_pixels": 0,
        }
        if compatible.size:
            block = probability[compatible][:, pixels]  # (Qc, P)
            chosen = numpy.argmax(block, axis=0)
            ids = compatible[chosen]
            out[pixels] = ids.astype(numpy.int32)
            counts = numpy.bincount(chosen, minlength=int(compatible.size))
            for position, count in enumerate(counts.tolist()):
                by_query[int(compatible[position])] += int(count)
            record["query_pixels"] = int(pixels.sum())
        else:
            raw = numpy.full(semantic.shape, IGNORE_CLASS, dtype=numpy.int64)
            raw[pixels] = int(class_id)
            baseline = numpy.asarray(raster_infer.instance_ids(raw), dtype=numpy.int64)
            unique = numpy.unique(baseline[pixels])
            unique = unique[unique >= 0]
            remapped = numpy.full(semantic.shape, -1, dtype=numpy.int64)
            for identifier in unique.tolist():
                remapped[baseline == int(identifier)] = next_fallback
                next_fallback += 1
                fallback_ids += 1
            out[pixels] = remapped[pixels].astype(numpy.int32)
            fallback_pixels += int(pixels.sum())
            record["fallback_pixels"] = int(pixels.sum())
            record["query_pixels"] = 0
        per_class[str(class_id)] = record

    thing_pixels = int(thing.sum())
    assigned = int((out >= 0).sum())
    stuff_pixels = int(((semantic >= num_thing_classes) & (semantic <= NUM_CLASSES - 1)).sum())
    unknown_pixels = int((semantic < 0).sum())
    stats = {
        "schema": DECODE_SCHEMA,
        "version": QUERY_VERSION,
        "num_queries": queries,
        "ground_truth_used": False,
        "thresholds": None,
        "thing_pixels": thing_pixels,
        "assigned_thing_pixels": assigned,
        "unassigned_thing_pixels": int(thing_pixels - assigned),
        "fallback_pixels": int(fallback_pixels),
        "fallback_instances": int(fallback_ids),
        "stuff_pixels": stuff_pixels,
        "unknown_pixels": unknown_pixels,
        "stuff_unchanged": bool((out[~thing] == -1).all()),
        "queries_used": int((by_query > 0).sum()),
        "query_pixels": {
            str(index): int(count) for index, count in enumerate(by_query.tolist())
        },
        "instances": int(numpy.unique(out[thing]).size),
        "per_class": per_class,
        "pixel_classes_bitidentical": True,
    }
    if assigned != thing_pixels:
        raise QueryInstanceError("decoder_invariant_violated: Thing pixel without id")
    return {"instance": out, "stats": stats}


# ---------------------------------------------------------------------------
# prepare: real CPU supervision counts (no model, no guesses)
# ---------------------------------------------------------------------------


def eligible_crops_sufficient(
    count: int, threshold: int = MIN_ELIGIBLE_CROPS_PER_EPOCH
) -> bool:
    """The single eligibility boundary used by the plan and the readiness gate.

    ``count >= threshold`` passes.  Exactly ``threshold`` eligible crops is
    sufficient, which is why the plan marks an epoch cheap only strictly below
    the threshold; before this helper the plan used ``<`` while the readiness
    gate used ``>``, so the boundary crop count meant two different things.
    """

    return int(count) >= int(threshold)


def plan_epochs(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Optional[str],
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_groups: int = MIN_GROUPS_PER_CROP,
    max_instances: int = MAX_INSTANCES,
    min_pixels: int = MIN_INSTANCE_PIXELS,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
    accumulation_steps: int = ACCUMULATION_STEPS,
) -> Dict[str, Any]:
    """Walk every scheduled crop of every epoch and count real supervision."""

    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    per_epoch: List[Dict[str, Any]] = []
    class_groups: Dict[str, int] = {}
    unique_groups: set = set()
    capped_crops = 0
    omitted_pixels = 0
    single_group_crops = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        source, _sampler = fullfit.make_epoch_source(
            records, epoch=epoch, cache_dir=cache_dir, base_seed=base_seed
        )
        eligible = 0
        at_least_one = 0
        groups_total = 0
        pixels_total = 0
        samples: List[str] = []
        for index in range(len(source)):
            crop = source.get(index)
            if bool(crop.get("skip_training")):
                continue
            report = instance_groups(
                crop["semantic"],
                crop["semantic_valid"],
                crop["instance"],
                crop["instance_valid"],
                max_instances=max_instances,
                min_pixels=min_pixels,
            )
            count = int(report["group_count"])
            if count >= 1:
                at_least_one += 1
            if count < int(min_groups):
                continue
            eligible += 1
            groups_total += count
            pixels_total += int(report["domain_pixels"])
            if report["capped"]:
                capped_crops += 1
                omitted_pixels += int(report["omitted_pixels"])
            if count == 1:
                single_group_crops += 1
            sample_id = str(crop.get("sample_id", ""))
            if sample_id not in samples:
                samples.append(sample_id)
            for class_id, instance_id in report["keys"]:
                key = str(class_id)
                class_groups[key] = int(class_groups.get(key, 0)) + 1
                unique_groups.add((sample_id, class_id, instance_id))
        per_epoch.append(
            {
                "epoch": epoch,
                "seed": fullfit.epoch_seed(base_seed, epoch),
                "scheduled_crops": int(len(source)),
                "crops_with_one_or_more_groups": int(at_least_one),
                "eligible_crops": int(eligible),
                "skipped_crops": int(len(source) - eligible),
                "eligible_groups": int(groups_total),
                "eligible_pixels": int(pixels_total),
                "records_with_eligible_crop": len(samples),
                "optimizer_steps": int(math.ceil(eligible / float(accumulation_steps))),
                "below_min_eligible": bool(
                    not eligible_crops_sufficient(eligible, min_eligible_crops)
                ),
            }
        )
    stop_cheap = any(item["below_min_eligible"] for item in per_epoch)
    return {
        "records": len(records),
        "epochs": epochs,
        "per_epoch": per_epoch,
        "stop_cheap": bool(stop_cheap),
        "min_eligible_crops_per_epoch": int(min_eligible_crops),
        "min_groups_per_crop": int(min_groups),
        "unique_supervised_groups": len(unique_groups),
        "eligible_groups_by_class": {
            key: int(class_groups[key]) for key in sorted(class_groups, key=int)
        },
        "capped_crops": int(capped_crops),
        "omitted_instance_pixels": int(omitted_pixels),
        "single_group_crops": int(single_group_crops),
        "seconds": round(time.monotonic() - started, 3),
    }


def _seed_roles(epochs: int) -> Dict[str, Any]:
    """Disambiguate the two seed constants used by this lane.

    ``BASE_SEED`` and ``HEAD_INIT_SEED`` are different numbers on purpose, and
    each one only drives one thing.  Reporting both next to their role keeps a
    reader from mistaking ``head.init_seed`` for a stale copy of
    ``readiness.base_seed``.
    """

    global_seed = fullfit.epoch_seed(BASE_SEED, 1)
    return {
        "base_seed": int(BASE_SEED),
        "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
        "head_init_seed": int(HEAD_INIT_SEED),
        "distinct_head_and_base_seed": bool(int(HEAD_INIT_SEED) != int(BASE_SEED)),
        "global_seed_before_head_build": int(global_seed),
        "epoch_seeds": {
            str(index): int(fullfit.epoch_seed(BASE_SEED, index))
            for index in range(1, int(epochs) + 1)
        },
        "role": {
            "base_seed": (
                "crop schedule derivation and the per-epoch global RNG only; it "
                "never touches head weights"
            ),
            "head_init_seed": (
                "per-parameter head initialisation only, via a per-name "
                "torch.Generator; it never touches the crop schedule"
            ),
            "global_seed_before_head_build": (
                "epoch_seed(base_seed, 1), set with seed_everything before the "
                "head is built as the work order requires; it does not determine "
                "head weights because the head inits from its own generators"
            ),
        },
        "note": (
            "head.init_seed is not stale and not a mis-read of base_seed: the two "
            "constants have separate roles. Measured proof lives in "
            "seed-reconciliation.json (the initial head digest is identical under "
            "two different global RNG states) and in fit/fit.json seeds."
        ),
    }


def prepare(
    manifest_path: str,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    min_groups: int = MIN_GROUPS_PER_CROP,
    min_eligible_crops: int = MIN_ELIGIBLE_CROPS_PER_EPOCH,
    role: str = train.ROLE_FIT,
    write: bool = True,
) -> Dict[str, Any]:
    """CPU-only preparation: role-checked fit records + measured supervision."""

    # The role is checked before the manifest is even opened, so a caller that
    # asks for a forbidden role is refused without touching any file.
    train.require_command_role(train.ROLE_FIT, role)
    records = fullfit.load_fit_records(manifest_path)
    plan = plan_epochs(
        records,
        cache_dir=cache_dir,
        epochs=epochs,
        base_seed=base_seed,
        min_groups=min_groups,
        min_eligible_crops=min_eligible_crops,
    )
    policy = {
        "schema": PREPARE_SCHEMA,
        "version": QUERY_VERSION,
        "kind": "frozen_semantic_query_mask_prepare",
        "role": str(role),
        "manifest": os.path.abspath(manifest_path),
        "cache_dir": os.path.abspath(cache_dir) if cache_dir else None,
        "records": int(len(records)),
        "crops_per_record": int(CROPS_PER_RECORD),
        "crop_size": int(CROP_SIZE),
        "tile_size": int(TILE_SIZE),
        "tile_stride": int(TILE_STRIDE),
        "semantic_frozen": True,
        "semantic_eval_always": True,
        "head": {
            "topology": "1x1 projection + learned queries + 2-layer transformer decoder + mask dot",
            "feature_tap": "net.classifier[3] (ReLU, 256 channels, stride 8)",
            "feature_projection": f"Conv2d({FEATURE_CHANNELS}, {FEAT_DIM}, kernel_size=1)",
            "num_queries": int(NUM_QUERIES),
            "decoder_layers": int(DECODER_LAYERS),
            "decoder_heads": int(DECODER_HEADS),
            "decoder_ff": int(DECODER_FF),
            "mask_embedding": int(MASK_DIM),
            "memory_grid_max": int(MEMORY_GRID),
            "class_head_classes": int(NUM_CLASSES),
            "init_seed": int(HEAD_INIT_SEED),
            "init": "deterministic per-parameter (seed, name); global seed set before build",
        },
        "loss": {
            "matching": "per-crop Hungarian on class + BCE + Dice cost (detached)",
            "match_weights": {
                "class": MATCH_CLASS_WEIGHT,
                "bce": MATCH_BCE_WEIGHT,
                "dice": MATCH_DICE_WEIGHT,
            },
            "loss_weights": {
                "class": LOSS_CLASS_WEIGHT,
                "bce": LOSS_BCE_WEIGHT,
                "dice": LOSS_DICE_WEIGHT,
            },
            "mask_domain": (
                "instance_valid & semantic_valid & 0 <= class < 30 "
                "(known Thing pixels with instance support only)"
            ),
            "unknown_negatives": False,
            "unannotated_negatives": False,
            "stuff_negatives": False,
            "different_known_instance_negatives": True,
            "unmatched_query_no_object_term": False,
            "class_supervision": "matched queries only, matched group class",
            "max_instances_per_crop": int(MAX_INSTANCES),
            "instance_cap": "first (class, instance) after sort; omitted pixels leave the domain",
            "min_instance_pixels": int(MIN_INSTANCE_PIXELS),
            "min_instance_pixels_note": (
                "query-specific threshold: a group must cover at least 4 known "
                "Thing pixels to enter the mask domain. It is deliberately lower "
                "than the affinity lane's 8 because a query can model a small "
                "object that a per-instance embedding merge would drop; it was "
                "fixed before the fit and is not tuned."
            ),
            "min_groups_per_crop": int(MIN_GROUPS_PER_CROP),
            "empty_supervision": "skipped before forward; no optimizer step",
        },
        "training": {
            "epochs": int(epochs),
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "accumulation_steps": int(ACCUMULATION_STEPS),
            "epoch_weighting": "matched group mass per crop (explicit)",
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "seed_before_head_build": True,
            "batch_norm": "frozen (semantic wrapper is eval-only)",
        },
        "decoder": {
            "query_class": "argmax of the 35-class query head (predictions only)",
            "assignment": "argmax mask prob among class-compatible queries; lowest index on ties",
            "fallback": (
                "unique CC8 id of predicted Thing map from one running allocator "
                "that starts at num_queries and never restarts per class, so ids "
                "stay global and collision free; fallback_offset must be >= num_queries"
            ),
            "stuff": "unchanged (-1 instance, pixel classes bit-identical)",
            "thresholds": None,
            "ground_truth": "never read",
            "input_validation": (
                "native_feature_map enforces finite float UNIT_RANGE [0, 1] "
                "3 x H x W and refuses other ranges; no /255 anywhere"
            ),
            "tiling": (
                "single image-level stride-8 feature map; tiles pad only the "
                "window grid internally and the returned map is cropped to "
                "ceil(H/8) x ceil(W/8) of the real image; decoder runs once per image"
            ),
            "tile_query_stitching": "not needed (no per-tile query identity)",
            "train_infer_context_gap": (
                "train: 256 crop -> 32x32 features, memory 32x32, masks at 32x32; "
                "infer: whole image at stride 8 (e.g. 1000 px -> 125x125), memory "
                "average-pooled to 32x32, masks dotted at 125x125 and upsampled natively"
            ),
        },
        "seed_roles": _seed_roles(epochs),
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
# checkpoint
# ---------------------------------------------------------------------------


def head_config() -> Dict[str, int]:
    return {
        "in_channels": int(FEATURE_CHANNELS),
        "feat_dim": int(FEAT_DIM),
        "mask_dim": int(MASK_DIM),
        "num_queries": int(NUM_QUERIES),
        "num_classes": int(NUM_CLASSES),
        "layers": int(DECODER_LAYERS),
        "heads": int(DECODER_HEADS),
        "ff": int(DECODER_FF),
        "memory_grid": int(MEMORY_GRID),
    }


def save_head_checkpoint(
    path: str,
    head: Any,
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
        "version": QUERY_VERSION,
        "kind": "query_instance_head",
        "final_epoch_only": True,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "head_config": head_config(),
        "head_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in head.state_dict().items()
        },
        "head_parameters": head_parameter_count(head),
        "semantic": {
            "path": os.path.abspath(semantic_checkpoint),
            "schema": semantic_payload.get("schema"),
            "role": semantic_payload.get("role"),
            "epoch": semantic_payload.get("epoch"),
            "epochs": semantic_payload.get("epochs"),
            "num_classes": NUM_CLASSES,
            "frozen": True,
            "eval_always": True,
            "batch_norm_running_stats": True,
            "trainable": False,
        },
        "contract": {
            "loaders_strict": True,
            "model_promotion": False,
            "labels": "class indices 0..34, -1 unknown",
            "instances": "raw local ids >= 0; instance_valid is the pixel authority",
            "mask_domain": "known Thing pixels with instance support only",
            "unknown_negatives": False,
            "unmatched_query_no_object_term": False,
            "decoder": "class-compatible argmax query mask, CC8 fallback offset by num_queries",
            "thresholds": None,
            "train_infer_context_gap": True,
        },
        "seeds": {
            "base_seed": int(BASE_SEED),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_init_seed": int(HEAD_INIT_SEED),
            "distinct": bool(int(HEAD_INIT_SEED) != int(BASE_SEED)),
            "note": (
                "head_init_seed initialises the trainable head; base_seed seeds the "
                "crop schedule and the global RNG per epoch. They are deliberately "
                "different constants."
            ),
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
    """Strict checkpoint load: schema, kind, config, final-epoch, frozen ref."""

    torch = _require_torch()
    if not os.path.isfile(path):
        raise QueryInstanceError(f"checkpoint_not_found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise QueryInstanceError(f"checkpoint_schema_mismatch: {path}")
    if payload.get("kind") != "query_instance_head":
        raise QueryInstanceError(f"checkpoint_kind_mismatch: {payload.get('kind')!r}")
    if payload.get("final_epoch_only") is not True:
        raise QueryInstanceError("checkpoint_must_be_final_epoch_only")
    state = payload.get("head_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise QueryInstanceError("checkpoint_missing_head_state")
    config = payload.get("head_config")
    if not isinstance(config, Mapping):
        raise QueryInstanceError("checkpoint_missing_head_config")
    for key, value in head_config().items():
        if int(config.get(key, -1)) != int(value):
            raise QueryInstanceError(
                f"checkpoint_head_config_mismatch: {key}={config.get(key)!r} != {value}"
            )
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping) or not semantic.get("path"):
        raise QueryInstanceError("checkpoint_missing_semantic_reference")
    if semantic.get("frozen") is not True or semantic.get("eval_always") is not True:
        raise QueryInstanceError("checkpoint_semantic_contract_not_frozen")
    result = dict(payload)
    result["head_state_dict"] = {key: value.to(device) for key, value in state.items()}
    result["semantic_reference_exists"] = bool(os.path.isfile(semantic["path"]))
    return result


def assert_semantic_reference(payload: Mapping[str, Any], semantic_checkpoint: str) -> None:
    reference = payload.get("semantic") or {}
    reference_path = reference.get("path")
    if not reference_path:
        raise QueryInstanceError("head_checkpoint_missing_semantic_reference")
    if os.path.normcase(os.path.abspath(reference_path)) != os.path.normcase(
        os.path.abspath(semantic_checkpoint)
    ):
        raise QueryInstanceError(
            "semantic_checkpoint_mismatch: head was trained against "
            f"{os.path.abspath(reference_path)}"
        )


def assert_semantic_identity(
    reference: Mapping[str, Any], loaded_payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare the stored semantic fingerprint with the file actually loaded.

    ``assert_semantic_reference`` only proves the paths agree; two different
    checkpoints can sit at the same path over time.  This compares the recorded
    ``schema``, ``role``, ``epoch`` and ``epochs`` of the frozen semantic model
    against what the loaded payload reports, so a swapped or re-trained
    checkpoint is refused instead of silently mislabelled.  No hashing, no file
    reads beyond the payload already in memory.
    """

    checked: Dict[str, Any] = {}
    for field in ("schema", "role", "epoch", "epochs"):
        stored = reference.get(field)
        actual = loaded_payload.get(field)
        checked[field] = {"stored": stored, "loaded": actual}
        if stored is None:
            continue
        if stored != actual:
            raise QueryInstanceError(
                f"semantic_identity_mismatch: {field} stored {stored!r} but the "
                f"checkpoint at the referenced path reports {actual!r}"
            )
    return checked


def load_query_head(
    semantic_checkpoint: str,
    *,
    head_checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the frozen semantic wrapper, the tap and (optionally) the head."""

    model, payload = pilot.load_checkpoint_model(semantic_checkpoint, device=device)
    tap = FrozenFeatureTap(model, device=device)
    head = build_query_head(device=device)
    report: Dict[str, Any] = {
        "semantic_checkpoint": os.path.abspath(semantic_checkpoint),
        "semantic_schema": payload.get("schema"),
        "semantic_epoch": payload.get("epoch"),
        "frozen": tap.frozen_report,
        "head_loaded": False,
        "head_parameters": head_parameter_count(head),
    }
    if head_checkpoint is not None:
        stored = load_head_checkpoint(head_checkpoint, device=device)
        assert_semantic_reference(stored, semantic_checkpoint)
        identity = assert_semantic_identity(stored.get("semantic") or {}, payload)
        head.load_state_dict(stored["head_state_dict"], strict=True)
        head.to(device)
        report["head_loaded"] = True
        report["head_checkpoint"] = os.path.abspath(head_checkpoint)
        report["head_epoch"] = stored.get("epoch")
        report["semantic_reference"] = stored.get("semantic")
        report["semantic_identity"] = identity
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    return tap, head, report


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _crop_tensor(crop: Mapping[str, Any]) -> Any:
    numpy = _require_numpy()
    torch = _require_torch()
    image = numpy.ascontiguousarray(numpy.asarray(crop["image"], dtype=numpy.float32))
    return torch.from_numpy(image).unsqueeze(0)


def _crop_loss(head: Any, tap: FrozenFeatureTap, crop: Mapping[str, Any]) -> Dict[str, Any]:
    """One crop: frozen features -> head -> matched loss (no ground truth out)."""

    torch = _require_torch()
    image = _crop_tensor(crop)
    features = tap.feature_map(image)["features"]
    outputs = head(features)
    groups = instance_groups(
        crop["semantic"],
        crop["semantic_valid"],
        crop["instance"],
        crop["instance_valid"],
    )
    result = query_instance_loss(
        outputs["mask_logits"][0],
        outputs["class_logits"][0],
        groups,
        crop_size=int(crop["image"].shape[-1]),
    )
    if not bool(torch.isfinite(result["loss"])):
        raise QueryInstanceError("non_finite_query_instance_loss")
    result["groups"] = groups
    return result


def _assert_finite_gradient(parameters: Sequence[Any]) -> None:
    torch = _require_torch()
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise QueryInstanceError("non_finite_query_head_gradient")


def train_query_head(
    tap: FrozenFeatureTap,
    head: Any,
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
    """Head-only fit: frozen features, group-mass weighting, frozen semantics."""

    torch = _require_torch()
    train.require_command_role("fit", role)
    if not records:
        raise QueryInstanceError("no fit records")
    epochs = _positive_int("epochs", epochs)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    lr = _finite(lr, "lr")
    weight_decay = _finite(weight_decay, "weight_decay")
    if lr <= 0.0 or weight_decay < 0.0:
        raise QueryInstanceError("lr must be positive and weight_decay >= 0")
    freeze_semantic_model(tap.model)
    digest_before = semantic_parameter_digest(tap.model)
    head_parameters = [p for p in head.parameters() if p.requires_grad]
    if not head_parameters:
        raise QueryInstanceError("head has no trainable parameters")
    optimizer = torch.optim.AdamW(head_parameters, lr=lr, weight_decay=weight_decay)
    guard = _new_guard(device)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
    }
    started = time.monotonic()
    epoch_reports: List[Dict[str, Any]] = []
    final_metrics: Dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        torch = _require_torch()
        seed = fullfit.epoch_seed(base_seed, epoch)
        train.seed_everything(seed)
        source, _sampler = fullfit.make_epoch_source(
            records, epoch=epoch, cache_dir=cache_dir, base_seed=base_seed
        )
        tap.model.eval()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        scheduled = 0
        supervised = 0
        skipped = 0
        steps = 0
        pairs_total = 0
        domain_total = 0
        capped = 0
        weighted_loss = 0.0
        pending: List[Tuple[Any, int]] = []
        pending_pairs = 0
        pending_domain = 0

        def flush() -> None:
            nonlocal steps, pairs_total, domain_total, weighted_loss
            nonlocal pending, pending_pairs, pending_domain
            if not pending:
                return
            total = max(1, pending_pairs)
            for loss, pairs in pending:
                if not bool(torch.isfinite(loss)):
                    raise QueryInstanceError("non_finite_loss_before_backward")
                (loss * (float(pairs) / float(total))).backward()
            _assert_finite_gradient(head_parameters)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sample_resource_peak(guard)
            steps += 1
            pairs_total += pending_pairs
            domain_total += pending_domain
            weighted_loss += sum(
                float(loss.detach().item()) * float(pairs) for loss, pairs in pending
            )
            pending = []
            pending_pairs = 0
            pending_domain = 0

        for index in range(len(source)):
            scheduled += 1
            crop = source.get(index)
            usable, report = crop_is_supervised(crop)
            if not usable:
                skipped += 1
                continue
            result = _crop_loss(head, tap, crop)
            if result["skipped"] or result["matched_pairs"] <= 0:
                skipped += 1
                continue
            supervised += 1
            if report["capped"]:
                capped += 1
            pending.append((result["loss"], int(result["matched_pairs"])))
            pending_pairs += int(result["matched_pairs"])
            pending_domain += int(result["domain_pixels"])
            if len(pending) >= accumulation_steps:
                flush()
        flush()
        tap.model.eval()
        if semantic_parameter_digest(tap.model) != digest_before:
            raise QueryInstanceError(f"semantic parameters changed during epoch {epoch}")
        if any(bool(module.training) for module in tap.model.modules()):
            tap.model.eval()
        epoch_report = {
            "epoch": int(epoch),
            "seed": int(seed),
            "scheduled_crops": int(scheduled),
            "supervised_crops": int(supervised),
            "skipped_crops": int(skipped),
            "capped_crops": int(capped),
            "optimizer_steps": int(steps),
            "matched_pairs": int(pairs_total),
            "domain_pixels": int(domain_total),
            "loss_group_mass_weighted": (
                (weighted_loss / float(pairs_total)) if pairs_total else None
            ),
            "accumulation_steps": int(accumulation_steps),
            "semantic_parameters_unchanged": True,
        }
        epoch_reports.append(epoch_report)
        final_metrics = epoch_report
    moved = any(
        not torch.equal(
            initial_state[key].to(head.state_dict()[key].device), head.state_dict()[key]
        )
        for key in initial_state
    )
    result = {
        "schema": FIT_SCHEMA,
        "version": QUERY_VERSION,
        "kind": "query_instance_fit",
        "final_epoch_only": True,
        "epochs": int(epochs),
        "per_epoch": epoch_reports,
        "final": final_metrics,
        "head_changed_from_init": bool(moved),
        "head_parameters": head_parameter_count(head),
        "finite_checks": {
            "loss_per_crop_aborts_on_nonfinite": True,
            "loss_before_backward_aborts_on_nonfinite": True,
            "grad_before_step_aborts_on_nonfinite": True,
            "nonfinite_detected": False,
            "note": "any detection aborts before optimizer.step; a written fit.json means zero detections",
        },
        "determinism": {
            "head_initial_state_reproducible": True,
            "basis": "per-parameter generator seeded from (head_init_seed, name)",
            "cuda_attention_backward_deterministic": None,
            "note": (
                "the initial head and the crop schedule are exactly reproducible; "
                "PyTorch's CUDA memory-efficient attention backward is not "
                "bit-deterministic by default, so repeated GPU runs can differ in "
                "low-order bits. torch.use_deterministic_algorithms was NOT enabled: "
                "it would change this authorised recipe."
            ),
        },
        "semantic_parameters_frozen": True,
        "semantic_parameters_unchanged": True,
        "semantic_batch_norm_eval": True,
        "loss": {
            "matching": "per-crop Hungarian on class + BCE + Dice cost",
            "mask_domain": "known Thing pixels with instance support only",
            "unknown_negatives": False,
            "unmatched_query_no_object_term": False,
            "group_mass_weighted": True,
        },
        "seeds": {
            "base_seed": int(base_seed),
            "epoch_seed_stride": int(EPOCH_SEED_STRIDE),
            "head_init_seed": int(HEAD_INIT_SEED),
            "global_seed_before_head_build": int(fullfit.epoch_seed(base_seed, 1)),
            "head_init_policy": (
                "every head parameter is initialised from its own "
                "torch.Generator seeded with (head_init_seed, parameter name), so "
                "the initial head is independent of construction order and of the "
                "global RNG; the global seed is still set before the build as the "
                "work order requires"
            ),
            "head_init_seed_is_not_base_seed": bool(HEAD_INIT_SEED != int(base_seed)),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "accumulation_steps": int(accumulation_steps),
        },
        "resource_guard": guard_report(guard),
        "seconds": round(time.monotonic() - started, 3),
    }
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint_query_instance.pt")
    result["checkpoint"] = save_head_checkpoint(
        checkpoint_path,
        head,
        epoch=epochs,
        epochs=epochs,
        semantic_checkpoint=semantic_checkpoint,
        semantic_payload=semantic_payload,
        metrics={"final": final_metrics, "epochs": epoch_reports},
    )
    _write_json(os.path.join(output_dir, "fit.json"), result)
    return result


# ---------------------------------------------------------------------------
# inference: one image-level feature map, one decoder call
# ---------------------------------------------------------------------------


def predict_mask_logits(
    tap: FrozenFeatureTap,
    head: Any,
    image: Any,
    *,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Native query masks for one image (no ground truth anywhere).

    The image is the **unit-range** ``3 x H x W`` array the frozen wrapper and
    the cache store (``[0, 1]``, no ``/255``); the wrapper normalises it
    internally.  Masks are returned at stride-8 feature resolution and must be
    bilinearly resized to the image size by the caller.
    """

    torch = _require_torch()
    tapped = tap.native_feature_map(image, tile_size=tile_size, stride=stride)
    features = tapped["features"].unsqueeze(0).to(device)
    was_training = bool(head.training)
    head.eval()
    with torch.no_grad():
        outputs = head(features)
    if was_training:  # pragma: no cover - training never calls this
        head.train()
    height, width = int(tapped["features"].shape[-2]), int(tapped["features"].shape[-1])
    mask_logits = outputs["mask_logits"][0]
    mask_probability = torch.sigmoid(mask_logits)
    class_probability = torch.softmax(outputs["class_logits"][0], dim=-1)
    return {
        "mask_logits": mask_logits.detach().to("cpu"),
        "mask_probability": mask_probability.detach().to("cpu"),
        "class_probability": class_probability.detach().to("cpu"),
        "query_class": class_probability.argmax(dim=-1).to("cpu"),
        "feature_shape": [height, width],
        "feature_grid": tapped["feature_grid"],
        "tile_count": tapped["tile_count"],
        "tile_size": tapped["tile_size"],
        "stride": tapped["stride"],
        "memory_grid": outputs["memory_grid"],
    }


class QueryInstanceDecoder:
    """Loaded, ready-to-call decoder (frozen wrapper + fitted query head).

    Built by :func:`load_decoder`, consumed by :func:`decode_from_image`.  The
    semantic weights, the feature tap and the head are all in eval mode, no
    parameter requires a gradient, and nothing here reads labels.
    """

    def __init__(
        self,
        *,
        tap: FrozenFeatureTap,
        head: Any,
        report: Mapping[str, Any],
        device: str = "cpu",
        tile_size: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
    ) -> None:
        self.tap = tap
        self.head = head
        self.report = dict(report)
        self.device = str(device)
        self.tile_size = _positive_int("tile_size", int(tile_size))
        self.stride = _positive_int("stride", int(stride))
        self.num_queries = int(head.num_queries)
        self.num_thing_classes = int(NUM_THING_CLASSES)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": DECODE_SCHEMA,
            "version": QUERY_VERSION,
            "kind": "query_instance_decoder",
            "device": self.device,
            "tile_size": int(self.tile_size),
            "stride": int(self.stride),
            "num_queries": int(self.num_queries),
            "num_thing_classes": int(self.num_thing_classes),
            "input_range": "[0, 1] unit RGB float (no /255 here)",
            "image_layout": "3 x H x W",
            "semantic_index": "saved int16/int32 -1..34 channel, shape H x W",
            "ground_truth_used": False,
            "thresholds": None,
            "load": dict(self.report),
        }

    def predict_masks(
        self, image: Any, *, semantic: Any = None
    ) -> Dict[str, Any]:
        output = predict_mask_logits(
            self.tap,
            self.head,
            image,
            tile_size=self.tile_size,
            stride=self.stride,
            device=self.device,
        )
        height, width = _image_height_width(image)
        probability = F.interpolate(
            output["mask_probability"].unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].numpy()
        return {
            "mask_probability": probability,
            "query_class": output["query_class"].numpy(),
            "class_probability": output["class_probability"].numpy(),
            "feature_grid": output["feature_grid"],
            "memory_grid": output["memory_grid"],
            "tile_count": output["tile_count"],
            "image_size": [int(height), int(width)],
        }

    def decode(
        self,
        image: Any,
        semantic: Any,
        *,
        semantic_valid: Any = None,
    ) -> Dict[str, Any]:
        masks = self.predict_masks(image, semantic=semantic)
        decoded = decode_instances(
            semantic,
            masks["mask_probability"],
            masks["query_class"],
            num_thing_classes=self.num_thing_classes,
        )
        decoded["stats"]["semantic_source"] = "saved_semantic_index"
        decoded["stats"]["semantic_valid_supplied"] = semantic_valid is not None
        decoded["stats"]["image_size"] = masks["image_size"]
        decoded["stats"]["feature_grid"] = masks["feature_grid"]
        decoded["stats"]["memory_grid"] = masks["memory_grid"]
        decoded["stats"]["tile_count"] = masks["tile_count"]
        decoded["stats"]["query_class"] = masks["query_class"].tolist()
        return decoded

    def close(self) -> None:
        self.tap.close()


def _image_height_width(image: Any) -> Tuple[int, int]:
    torch = _require_torch()
    if torch.is_tensor(image):
        if image.ndim != 3:
            raise QueryInstanceError("image must be 3 x H x W")
        return int(image.shape[-2]), int(image.shape[-1])
    array = _require_numpy().asarray(image)
    if array.ndim != 3:
        raise QueryInstanceError("image must be 3 x H x W")
    return int(array.shape[-2]), int(array.shape[-1])


def load_decoder(
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    head_checkpoint: Optional[str] = None,
    device: str = "cpu",
    *,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> QueryInstanceDecoder:
    """Load frozen semantics + fitted query head into a ready decoder.

    ``head_checkpoint`` is required for a fitted decoder; without it the head
    is the deterministic initial one (useful for shape smoke tests, never for
    evaluation).
    """

    tap, head, report = load_query_head(
        semantic_checkpoint, head_checkpoint=head_checkpoint, device=device
    )
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return QueryInstanceDecoder(
        tap=tap,
        head=head,
        report=report,
        device=device,
        tile_size=tile_size,
        stride=stride,
    )


def decode_from_image(
    decoder: QueryInstanceDecoder,
    image: Any,
    saved_semantic_index: Any,
    *,
    semantic_valid: Any = None,
) -> Dict[str, Any]:
    """Instance map for one unit-range image and its saved semantic index.

    ``image`` is ``3 x H x W`` float in ``[0, 1]`` (as decoded from the PNG and
    stored in the cache; never divided by 255 here).  ``saved_semantic_index``
    is the frozen lane's ``H x W`` integer index in ``-1..34``.  Returns
    ``{"instance": int32 H x W, "stats": {...}}`` where every Thing pixel
    (class ``0..29``) carries a non-negative id that is globally unique across
    classes, and Stuff / UNKNOWN pixels keep ``-1``.

    ``semantic_valid`` is accepted for API symmetry and recorded; the saved
    index already encodes the unknown sentinel, so it never changes the ids.
    """

    if not isinstance(decoder, QueryInstanceDecoder):
        raise QueryInstanceError("decoder must come from load_decoder()")
    height, width = _image_height_width(image)
    semantic = _require_numpy().asarray(saved_semantic_index)
    if semantic.shape != (height, width):
        raise QueryInstanceError(
            f"semantic index shape {tuple(semantic.shape)} != image {(height, width)}"
        )
    if not _require_numpy().issubdtype(semantic.dtype, _require_numpy().integer):
        raise QueryInstanceError("saved_semantic_index must be an integer array")
    if int(semantic.min()) < -1 or int(semantic.max()) > NUM_CLASSES - 1:
        raise QueryInstanceError("saved_semantic_index must lie in -1..34")
    if semantic_valid is not None:
        valid = _require_numpy().asarray(semantic_valid, dtype=bool)
        if valid.shape != (height, width):
            raise QueryInstanceError("semantic_valid must match the image shape")
    return decoder.decode(image, semantic, semantic_valid=semantic_valid)


def predict_image_instances(
    tap: FrozenFeatureTap,
    head: Any,
    image: Any,
    *,
    semantic: Any,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Label-free instance map; ``semantic`` is the saved frozen index."""

    numpy = _require_numpy()
    torch = _require_torch()
    output = predict_mask_logits(
        tap, head, image, tile_size=tile_size, stride=stride, device=device
    )
    height, width = _image_height_width(image)
    semantic_map = numpy.asarray(semantic)
    if semantic_map.shape != (height, width):
        raise QueryInstanceError("semantic map must match the image shape")
    probability = F.interpolate(
        output["mask_probability"].unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0].numpy()
    decoded = decode_instances(semantic_map, probability, output["query_class"].numpy())
    decoded["stats"]["semantic_source"] = "saved_semantic_index"
    decoded["stats"]["feature_grid"] = output["feature_grid"]
    decoded["stats"]["memory_grid"] = output["memory_grid"]
    decoded["stats"]["tile_count"] = output["tile_count"]
    decoded["semantic"] = semantic_map
    return decoded


# ---------------------------------------------------------------------------
# resource guard: measured, not assumed
# ---------------------------------------------------------------------------

MEMORY_LIMIT_BYTES = 6 * 1024 ** 3  # hard guard, 6 GiB
CPU_RSS_SOFT_LIMIT_BYTES = 8 * 1024 ** 3
GPU_HEADROOM_BYTES = 1 * 1024 ** 3  # the parent's "free >= peak + 1 GiB" rule


def _process_rss_bytes() -> Optional[int]:
    try:  # pragma: no cover - environment dependent
        import psutil  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return int(psutil.Process().memory_info().rss)


def _cuda_memory_info() -> Optional[Dict[str, Any]]:
    """Free/total device memory, or ``None`` when CUDA is not in play."""

    torch = _require_torch()
    try:  # pragma: no cover - environment dependent
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - environment dependent
        return None
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _new_guard(device: str) -> Dict[str, Any]:
    torch = _require_torch()
    state: Dict[str, Any] = {
        "device": str(device),
        "limit_bytes": int(MEMORY_LIMIT_BYTES),
        "cuda": False,
        "peak_cuda_bytes": None,
        "peak_rss_bytes": _process_rss_bytes(),
        "checks": 0,
        "memory_at_start": None,
        "min_free_bytes": None,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        state["cuda"] = True
        state["memory_at_start"] = _cuda_memory_info()
    return state


def _sample_resource_peak(state: Dict[str, Any]) -> None:
    """Record peak memory after a flush and abort if the 6 GiB guard trips."""

    torch = _require_torch()
    state["checks"] = int(state.get("checks", 0)) + 1
    if state.get("cuda"):
        peak = int(torch.cuda.max_memory_allocated())
        state["peak_cuda_bytes"] = max(int(state.get("peak_cuda_bytes") or 0), peak)
        info = _cuda_memory_info()
        if info is not None:
            previous = state.get("min_free_bytes")
            state["min_free_bytes"] = (
                int(info["free_bytes"])
                if previous is None
                else min(int(previous), int(info["free_bytes"]))
            )
        if peak > int(state["limit_bytes"]):
            raise QueryInstanceError(
                f"resource_guard_memory_exceeded: {peak} > {state['limit_bytes']} bytes"
            )
    rss = _process_rss_bytes()
    if rss is not None:
        state["peak_rss_bytes"] = max(int(state.get("peak_rss_bytes") or 0), rss)


def preflight_gpu(
    device: str = "cuda",
    *,
    observed_peak_bytes: Optional[int] = None,
    required_bytes: Optional[int] = None,
    headroom_bytes: int = GPU_HEADROOM_BYTES,
) -> Dict[str, Any]:
    """Measure real free device memory before a run, not the nominal budget.

    The 6 GiB guard bounds what *this* process allocates; it says nothing about
    a desktop compositor or a game already holding several GiB, so the decision
    to start is made from the measured free memory at that moment.

    With ``observed_peak_bytes`` (a peak already measured by this lane on this
    device) the rule is the parent's: proceed only when
    ``free >= observed_peak + headroom``.  Without it the conservative
    ``MEMORY_LIMIT_BYTES`` guard limit is used instead, which requires the whole
    nominal budget to be free.
    """

    torch = _require_torch()
    if observed_peak_bytes is not None and required_bytes is not None:
        raise QueryInstanceError("pass observed_peak_bytes or required_bytes, not both")
    if observed_peak_bytes is not None:
        required = _positive_int("observed_peak_bytes", int(observed_peak_bytes))
        basis = "measured_peak_plus_headroom"
    else:
        required = int(MEMORY_LIMIT_BYTES if required_bytes is None else required_bytes)
        basis = "guard_limit_plus_headroom"
    report: Dict[str, Any] = {
        "schema": GPU_PREFLIGHT_SCHEMA,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "basis": basis,
        "observed_peak_bytes": (
            int(observed_peak_bytes) if observed_peak_bytes is not None else None
        ),
        "observed_peak_gib": (
            round(float(observed_peak_bytes) / float(1024 ** 3), 4)
            if observed_peak_bytes is not None
            else None
        ),
        "guard_limit_bytes": int(MEMORY_LIMIT_BYTES),
        "guard_limit_gib": round(float(MEMORY_LIMIT_BYTES) / float(1024 ** 3), 3),
        "required_bytes": required,
        "required_gib": round(float(required) / float(1024 ** 3), 3),
        "headroom_bytes": int(headroom_bytes),
        "headroom_gib": round(float(headroom_bytes) / float(1024 ** 3), 3),
        "free_bytes": None,
        "total_bytes": None,
        "free_gib": None,
        "needed_bytes": int(required + headroom_bytes),
        "needed_gib": round(float(required + headroom_bytes) / float(1024 ** 3), 3),
        "device_name": None,
        "proceed": False,
        "reason": None,
    }
    if not report["cuda_available"]:
        report["reason"] = "cuda_unavailable"
        return report
    report["device_name"] = str(torch.cuda.get_device_name(0))
    info = _cuda_memory_info()
    if info is None:  # pragma: no cover - defensive
        report["reason"] = "free_memory_unreadable"
        return report
    report["free_bytes"] = int(info["free_bytes"])
    report["total_bytes"] = int(info["total_bytes"])
    report["free_gib"] = round(float(info["free_bytes"]) / float(1024 ** 3), 4)
    report["proceed"] = bool(int(info["free_bytes"]) >= int(report["needed_bytes"]))
    report["reason"] = "free_memory_sufficient" if report["proceed"] else "insufficient_free_memory"
    return report


def guard_report(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Serialisable view of a guard state (bytes and GiB)."""

    peak_cuda = state.get("peak_cuda_bytes")
    peak_rss = state.get("peak_rss_bytes")
    return {
        "limit_gib": round(float(state["limit_bytes"]) / float(1024 ** 3), 3),
        "limit_bytes": int(state["limit_bytes"]),
        "device": state.get("device"),
        "cuda_measured": bool(state.get("cuda")),
        "peak_cuda_bytes": int(peak_cuda) if peak_cuda is not None else None,
        "peak_cuda_gib": (
            round(float(peak_cuda) / float(1024 ** 3), 4) if peak_cuda is not None else None
        ),
        "peak_process_rss_bytes": int(peak_rss) if peak_rss is not None else None,
        "peak_process_rss_gib": (
            round(float(peak_rss) / float(1024 ** 3), 4) if peak_rss is not None else None
        ),
        "checks": int(state.get("checks", 0)),
        "aborts_over_limit": True,
        "memory_at_start": state.get("memory_at_start"),
        "min_free_bytes": (
            int(state["min_free_bytes"]) if state.get("min_free_bytes") is not None else None
        ),
        "min_free_gib": (
            round(float(state["min_free_bytes"]) / float(1024 ** 3), 4)
            if state.get("min_free_bytes") is not None
            else None
        ),
        "note": (
            "CUDA peak is measured with torch.cuda.max_memory_allocated() after "
            "every optimizer step and aborts before the next step when it exceeds "
            "the limit; free device memory is sampled at the same points and the "
            "minimum is reported, because other processes share the device."
        ),
    }


def footprint_report(image_side: int = 1000, *, bytes_per_value: int = 4) -> Dict[str, Any]:
    """Analytic activation footprint of this topology (no allocation)."""

    image_side = _positive_int("image_side", image_side)
    grid = int(math.ceil(image_side / float(FEATURE_STRIDE)))
    pooled = min(grid, MEMORY_GRID)
    rows = {
        "frozen_stride8_features": FEATURE_CHANNELS * grid * grid,
        "projected_features": FEAT_DIM * grid * grid,
        "transformer_memory": FEAT_DIM * pooled * pooled,
        "attention_scores_peak": DECODER_HEADS * pooled * pooled * pooled,
        "mask_logits": NUM_QUERIES * grid * grid,
        "class_logits": NUM_QUERIES * NUM_CLASSES,
    }
    bytes_rows = {key: int(value) * int(bytes_per_value) for key, value in rows.items()}
    return {
        "schema": FOOTPRINT_SCHEMA,
        "image_side": int(image_side),
        "feature_grid": int(grid),
        "memory_grid": int(pooled),
        "bytes_per_value": int(bytes_per_value),
        "activation_bytes": bytes_rows,
        "activation_total_bytes": int(sum(bytes_rows.values())),
        "activation_total_mib": round(
            float(sum(bytes_rows.values())) / float(1024 ** 2), 3
        ),
        "scope": (
            "activations only, one image at inference or one crop at training; "
            "excludes frozen wrapper weights and the CUDA context"
        ),
    }


# ---------------------------------------------------------------------------
# CPU measurement of the real 8-crop accumulation window
# ---------------------------------------------------------------------------


def measure(
    manifest_path: str,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    crops: int = ACCUMULATION_STEPS,
    device: str = "cpu",
    base_seed: int = BASE_SEED,
    accumulation_steps: int = ACCUMULATION_STEPS,
    image_side: int = 1000,
    role: str = train.ROLE_FIT,
    write: bool = True,
) -> Dict[str, Any]:
    """Run one real accumulation window on CPU and prove it is finite.

    This is a *measurement*, not a fit: gradients are accumulated exactly as
    the training loop does, but no optimizer step is taken and nothing is
    written to the head checkpoint.
    """

    torch = _require_torch()
    numpy = _require_numpy()
    train.require_command_role(train.ROLE_FIT, role)
    crops = _positive_int("crops", crops)
    accumulation_steps = _positive_int("accumulation_steps", accumulation_steps)
    records = fullfit.load_fit_records(manifest_path)
    if not records:
        raise QueryInstanceError("no fit records")
    started = time.monotonic()

    model, payload = pilot.load_checkpoint_model(semantic_checkpoint, device=device)
    tap = FrozenFeatureTap(model, device=device)
    train.seed_everything(fullfit.epoch_seed(base_seed, 1))
    head = build_query_head(seed=HEAD_INIT_SEED, device=device)
    freeze_semantic_model(tap.model)
    digest_before = semantic_parameter_digest(tap.model)
    tap.model.eval()
    head.train()
    guard = _new_guard(device)
    parameters = [p for p in head.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.grad = None

    source, _sampler = fullfit.make_epoch_source(
        records, epoch=1, cache_dir=cache_dir, base_seed=base_seed
    )
    per_crop: List[Dict[str, Any]] = []
    losses: List[Tuple[Any, int]] = []
    pairs_total = 0
    domain_total = 0
    seen = 0
    skipped = 0
    for index in range(len(source)):
        if len(per_crop) >= crops:
            break
        seen += 1
        crop = source.get(index)
        usable, report = crop_is_supervised(crop)
        if not usable:
            skipped += 1
            continue
        result = _crop_loss(head, tap, crop)
        if result["skipped"] or int(result["matched_pairs"]) <= 0:
            skipped += 1
            continue
        loss = result["loss"]
        if not bool(torch.isfinite(loss)):
            raise QueryInstanceError("non_finite_query_instance_loss")
        losses.append((loss, int(result["matched_pairs"])))
        pairs_total += int(result["matched_pairs"])
        domain_total += int(result["domain_pixels"])
        per_crop.append(
            {
                "sample_id": str(crop.get("sample_id")),
                "crop_index": int(crop.get("crop_index", index)),
                "group_count": int(result["groups"]["group_count"]),
                "matched_pairs": int(result["matched_pairs"]),
                "domain_pixels": int(result["domain_pixels"]),
                "loss": round(float(loss.detach().item()), 6),
                "capped": bool(result["groups"]["capped"]),
            }
        )

    if not losses:
        tap.close()
        raise QueryInstanceError("no supervised crops were found for the measurement")

    total = max(1, pairs_total)
    for loss, pairs in losses:
        (loss * (float(pairs) / float(total))).backward()
    _assert_finite_gradient(parameters)
    _sample_resource_peak(guard)

    grad_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            grad_norm += float(parameter.grad.detach().pow(2).sum().item())
    grad_norm = math.sqrt(grad_norm)
    if not math.isfinite(grad_norm):
        raise QueryInstanceError("non_finite_query_head_gradient")
    moved_semantic = semantic_parameter_digest(tap.model) != digest_before
    if moved_semantic:
        raise QueryInstanceError("semantic parameters changed during measurement")
    tap.close()

    footprint = footprint_report(image_side)
    plan = plan_epochs(
        records,
        cache_dir=cache_dir,
        epochs=1,
        base_seed=base_seed,
        min_groups=MIN_GROUPS_PER_CROP,
        min_eligible_crops=MIN_ELIGIBLE_CROPS_PER_EPOCH,
        accumulation_steps=accumulation_steps,
    )
    result = {
        "schema": MEASURE_SCHEMA,
        "version": QUERY_VERSION,
        "kind": "query_instance_cpu_measure",
        "role": "fit",
        "manifest": os.path.abspath(manifest_path),
        "cache_dir": os.path.abspath(cache_dir) if cache_dir else None,
        "semantic_checkpoint": os.path.abspath(semantic_checkpoint),
        "semantic_epoch": payload.get("epoch"),
        "device": str(device),
        "crops_requested": int(crops),
        "crops_measured": int(len(per_crop)),
        "crops_scanned": int(seen),
        "crops_skipped": int(skipped),
        "accumulation_steps": int(accumulation_steps),
        "one_full_window": bool(len(per_crop) >= accumulation_steps),
        "matched_pairs": int(pairs_total),
        "domain_pixels": int(domain_total),
        "mean_loss": round(
            sum(float(loss.detach().item()) * float(pairs) for loss, pairs in losses)
            / float(total),
            6,
        ),
        "gradient_norm": round(grad_norm, 8),
        "gradients_finite": True,
        "losses_finite": True,
        "optimizer_step_taken": False,
        "semantic_parameters_unchanged": True,
        "per_crop": per_crop,
        "resource_guard": guard_report(guard),
        "footprint": footprint,
        "eligibility_epoch1": {
            "eligible_crops": per_epoch_value(plan, "eligible_crops"),
            "scheduled_crops": per_epoch_value(plan, "scheduled_crops"),
            "optimizer_steps": per_epoch_value(plan, "optimizer_steps"),
            "min_eligible_crops_per_epoch": int(MIN_ELIGIBLE_CROPS_PER_EPOCH),
            "stop_cheap": bool(plan["stop_cheap"]),
        },
        "seconds": round(time.monotonic() - started, 3),
    }
    result["resource_verdict"] = _resource_verdict(result)
    if write:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "measure.json"), result)
    return result


def per_epoch_value(plan: Mapping[str, Any], key: str) -> Any:
    per_epoch = plan.get("per_epoch") or []
    if not per_epoch:
        return None
    return per_epoch[0].get(key)


def _resource_verdict(result: Mapping[str, Any]) -> Dict[str, Any]:
    guard = result["resource_guard"]
    footprint = result["footprint"]
    eligible = result["eligibility_epoch1"]
    checks = {
        "one_full_accumulation_window_measured": bool(result["one_full_window"]),
        "gradients_finite": bool(result["gradients_finite"]),
        "losses_finite": bool(result["losses_finite"]),
        "semantic_frozen": bool(result["semantic_parameters_unchanged"]),
        "activation_footprint_under_limit": bool(
            int(footprint["activation_total_bytes"]) < int(MEMORY_LIMIT_BYTES)
        ),
        "eligible_crops_over_trigger": bool(
            eligible_crops_sufficient(eligible["eligible_crops"] or 0)
        ),
        "cuda_peak_not_measured_on_cpu": not bool(guard["cuda_measured"]),
    }
    return {
        "checks": checks,
        "cpu_ready": bool(
            checks["one_full_accumulation_window_measured"]
            and checks["gradients_finite"]
            and checks["losses_finite"]
            and checks["semantic_frozen"]
            and checks["activation_footprint_under_limit"]
            and checks["eligible_crops_over_trigger"]
        ),
        "gpu_peak_pending": not bool(guard["cuda_measured"]),
        "note": (
            "CUDA peak memory is measured by the same guard whenever the fit runs "
            "with --device cuda; on CPU only the analytic footprint and process RSS "
            "are available, which is why cpu_ready does not imply gpu memory is proven."
        ),
    }


# ---------------------------------------------------------------------------
# READY bundle: prepare + measure + this lane's tests, all CPU
# ---------------------------------------------------------------------------


def _run_lane_tests() -> Dict[str, Any]:
    import importlib
    import unittest

    try:
        module = importlib.import_module("tests.test_recognition_query_instance")
    except Exception as error:  # pragma: no cover - defensive
        return {"ran": 0, "failures": [], "errors": [f"import_failed: {error}"], "ok": False}
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
    outcome = runner.run(suite)
    return {
        "module": "tests.test_recognition_query_instance",
        "ran": int(outcome.testsRun),
        "failures": [str(item[0]) for item in outcome.failures],
        "errors": [str(item[0]) for item in outcome.errors],
        "ok": bool(outcome.wasSuccessful()),
    }


def ready(
    manifest_path: str = DEFAULT_MANIFEST,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    semantic_checkpoint: str = DEFAULT_SEMANTIC_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    base_seed: int = BASE_SEED,
    crops: int = ACCUMULATION_STEPS,
    run_tests: bool = True,
) -> Dict[str, Any]:
    """Write the full CPU readiness bundle for the parent's review."""

    policy = prepare(
        manifest_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=epochs,
        base_seed=base_seed,
    )
    measurement = measure(
        manifest_path,
        cache_dir=cache_dir,
        semantic_checkpoint=semantic_checkpoint,
        output_dir=output_dir,
        crops=crops,
        base_seed=base_seed,
    )
    tests = _run_lane_tests() if run_tests else {"skipped": True}
    try:
        smoke_result: Dict[str, Any] = smoke(
            semantic_checkpoint, size=CROP_SIZE, device="cpu"
        )
    except Exception as error:  # pragma: no cover - environment dependent
        smoke_result = {"ok": False, "error": str(error)[:400]}
    else:
        smoke_result["ok"] = True
    ready_command = (
        "python -m cadvec.recognition_query_instance fit "
        f'--manifest "{os.path.abspath(manifest_path)}" '
        f'--cache-dir "{os.path.abspath(cache_dir)}" '
        f'--output-dir "{os.path.join(os.path.abspath(output_dir), "fit")}" '
        f'--semantic-checkpoint "{os.path.abspath(semantic_checkpoint)}" '
        "--role fit --device cuda --epochs 5 --accumulation-steps 8"
    )
    plan = policy["plan"]
    bundles = {
        "prepare.json": "supervision plan for every scheduled epoch",
        "policy.json": "topology, loss contract, decoder and tiling policy",
        "measure.json": "one measured accumulation window, gradient finiteness, guard",
        "verification.json": "this lane's unit tests (CPU, no data)",
        "READY.md": "human-readable readiness and the exact GPU command",
    }
    bundle = {
        "schema": READY_SCHEMA,
        "version": QUERY_VERSION,
        "kind": "query_instance_ready",
        "cpu_ready": bool(measurement["resource_verdict"]["cpu_ready"]),
        "gpu_training_authorised": False,
        "gpu_note": (
            "this bundle does not start training; the parent reviews it and "
            "authorises the GPU run"
        ),
        "command": ready_command,
        "command_device": "cuda",
        "epochs": int(epochs),
        "base_seed": int(base_seed),
        "seed_roles": _seed_roles(epochs),
        "records": int(policy["records"]),
        "scheduled_crops_per_epoch": int(plan["per_epoch"][0]["scheduled_crops"]),
        "eligible_crops_per_epoch": int(plan["per_epoch"][0]["eligible_crops"]),
        "optimizer_steps_per_epoch": int(plan["per_epoch"][0]["optimizer_steps"]),
        "per_epoch": [
            {
                "epoch": int(item["epoch"]),
                "seed": int(item["seed"]),
                "scheduled_crops": int(item["scheduled_crops"]),
                "eligible_crops": int(item["eligible_crops"]),
                "eligible_groups": int(item["eligible_groups"]),
                "optimizer_steps": int(item["optimizer_steps"]),
                "records_with_eligible_crop": int(item["records_with_eligible_crop"]),
                "below_min_eligible": bool(item["below_min_eligible"]),
            }
            for item in plan["per_epoch"]
        ],
        "min_eligible_crops_per_epoch": int(MIN_ELIGIBLE_CROPS_PER_EPOCH),
        "stop_cheap": bool(plan["stop_cheap"]),
        "unique_supervised_groups": int(plan["unique_supervised_groups"]),
        "classes_with_groups": sorted(
            int(key) for key in plan["eligible_groups_by_class"]
        ),
        "cap": {
            "max_instances_per_crop": int(MAX_INSTANCES),
            "capped_crops": int(plan["capped_crops"]),
            "omitted_instance_pixels": int(plan["omitted_instance_pixels"]),
            "single_group_crops": int(plan["single_group_crops"]),
            "policy": (
                "first (class, instance) after deterministic sort; omitted pixels "
                "leave the loss domain, never counted as background"
            ),
        },
        "resource": {
            "guard": measurement["resource_guard"],
            "footprint": measurement["footprint"],
            "verdict": measurement["resource_verdict"],
        },
        "measurement": {
            "crops_measured": int(measurement["crops_measured"]),
            "mean_loss": measurement["mean_loss"],
            "gradient_norm": measurement["gradient_norm"],
            "seconds": measurement["seconds"],
        },
        "tests": tests,
        "smoke": smoke_result,
        "decoder_api": {
            "load": "load_decoder(semantic_checkpoint, head_checkpoint, device) -> QueryInstanceDecoder",
            "decode": (
                "decode_from_image(decoder, image UNIT_CHW, saved_semantic_index, "
                "semantic_valid=None) -> {'instance': int32 HxW, 'stats': {...}}"
            ),
            "input_range": "[0, 1] unit RGB float, 3 x H x W, no /255 applied here",
            "output_contract": (
                "every Thing pixel (0..29) gets a non-negative id, ids are globally "
                "unique across classes, Stuff (30..34) and UNKNOWN (-1) keep -1, "
                "pixel classes and Stuff support are copied through unchanged, "
                "splitting a connected component is allowed"
            ),
            "label_free": True,
            "thresholds": None,
            "tiling": (
                "per-image stride-8 feature accumulation over the 256/128 tile grid, "
                "one decoder call per image, no per-tile query stitching"
            ),
            "cost": "one frozen forward pass per image plus one small head pass",
        },
        "artifacts": bundles,
        "not_claimed": [
            "no GPU step has run for this lane",
            "no accuracy, PQ, or instance-split fidelity number is claimed",
            "no full reproduction of the semantic lane or the affinity lane",
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "verification.json"), {"schema": READY_SCHEMA, "tests": tests})
    _write_json(os.path.join(output_dir, "readiness.json"), bundle)
    _write_ready_markdown(os.path.join(output_dir, "READY.md"), bundle)
    return bundle


def _write_ready_markdown(path: str, bundle: Mapping[str, Any]) -> None:
    resource = bundle["resource"]
    footprint = resource["footprint"]
    guard = resource["guard"]
    tests = bundle["tests"]
    lines = [
        "# Query-mask instance lane - CPU readiness",
        "",
        "Independent, additive lane. Nothing in the semantic lane, the",
        "instance-affinity lane, the raster model, the vectorizer or the",
        "submission path was edited.",
        "",
        "## Status",
        "",
        f"- cpu_ready: **{bool(bundle['cpu_ready'])}**",
        "- gpu_training_authorised: **false** (parent reviews this bundle first)",
        f"- unit tests: {tests.get('ran', 0)} run, ok = {bool(tests.get('ok'))}",
        "",
        "## Supervision plan (fit role, 204 records)",
        "",
        "| epoch | seed | scheduled | eligible | groups | steps | records |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in bundle["per_epoch"]:
        lines.append(
            f"| {item['epoch']} | {item['seed']} | {item['scheduled_crops']} |"
            f" {item['eligible_crops']} | {item['eligible_groups']} |"
            f" {item['optimizer_steps']} | {item['records_with_eligible_crop']} |"
        )
    lines += [
        "",
        f"- trigger: every epoch must have at least {bundle['min_eligible_crops_per_epoch']}"
        " eligible crops (count >= trigger passes; <= 2 groups per crop counts as cheap)",
        f"- unique supervised (class, instance) groups over {bundle['epochs']} epochs:"
        f" {bundle['unique_supervised_groups']}",
        f"- classes with groups: {bundle['classes_with_groups']}",
        f"- capped crops (more than 32 instances): {bundle['cap']['capped_crops']},"
        f" pixels excluded from the loss domain: {bundle['cap']['omitted_instance_pixels']}",
        f"- single-group crops: {bundle['cap']['single_group_crops']} (never a negative)",
        f"- stop_cheap: {bool(bundle['stop_cheap'])}",
        "",
        "## Seed roles (two different constants, on purpose)",
        "",
        "| constant | value | drives |",
        "| --- | --- | --- |",
        f"| base_seed | {bundle['seed_roles']['base_seed']} | crop schedule and the"
        " per-epoch global RNG |",
        f"| head_init_seed | {bundle['seed_roles']['head_init_seed']} | per-parameter"
        " head initialisation |",
        f"| global seed before head build |"
        f" {bundle['seed_roles']['global_seed_before_head_build']} |"
        " seed_everything(epoch_seed(base_seed, 1)) as the work order requires |",
        f"- epoch seeds: {list(bundle['seed_roles']['epoch_seeds'].values())}",
        "- head.init_seed is therefore not a stale copy of base_seed; the measured",
        "  initial-head digest is identical under two different global RNG states",
        "  (see seed-reconciliation.json).",
        "",
        "## Train / inference context gap (documented, not papered over)",
        "",
        "* training: 256 px crop -> 32 x 32 stride-8 features -> transformer memory",
        " 32 x 32 (identity pool) -> masks dotted at 32 x 32 and upsampled to 256 for",
        " the loss;",
        "* inference: whole image -> stride-8 grid (1000 px -> 125 x 125) -> memory",
        " average-pooled to at most 32 x 32 -> masks dotted at 125 x 125 and upsampled",
        " natively.",
        "",
        "The transformer always sees a 32 x 32 grid, but the mask dot product sees",
        "4x finer detail at inference, and crop statistics differ from whole-plan",
        "statistics. This is a real risk of this topology, recorded in the checkpoint",
        "contract rather than hidden.",
        "",
        "## Measured one accumulation window (CPU)",
        "",
        f"- crops measured: {bundle['measurement']['crops_measured']}",
        f"- mean group-mass-weighted loss: {bundle['measurement']['mean_loss']}",
        f"- gradient norm: {bundle['measurement']['gradient_norm']}",
        f"- seconds: {bundle['measurement']['seconds']}",
        f"- peak process RSS: {guard['peak_process_rss_gib']} GiB",
        f"- CUDA peak measured: {bool(guard['cuda_measured'])} (guard limit"
        f" {guard['limit_gib']} GiB, aborts before a step that would exceed it)",
        "",
        "## Decoder API (for the paired eval)",
        "",
        f"- `{bundle['decoder_api']['load']}`",
        f"- `{bundle['decoder_api']['decode']}`",
        f"- input: {bundle['decoder_api']['input_range']}",
        f"- output: {bundle['decoder_api']['output_contract']}",
        f"- label-free: {bundle['decoder_api']['label_free']}, thresholds:"
        f" {bundle['decoder_api']['thresholds']}",
        f"- tiling: {bundle['decoder_api']['tiling']}",
        "",
        f"- smoke (CPU, 256 px): ok = {bool(bundle['smoke'].get('ok'))},"
        f" mask logits {bundle['smoke'].get('mask_logits_shape')},"
        f" head parameters {bundle['smoke'].get('head_parameters')}",
        "",
        "## Analytic activation footprint",
        "",
        f"- image side {footprint['image_side']} -> stride-8 grid"
        f" {footprint['feature_grid']}, transformer memory {footprint['memory_grid']}",
        f"- activations total: {footprint['activation_total_mib']} MiB ({footprint['scope']})",
        "",
    ]
    lines.append("| activation | bytes |")
    lines.append("| --- | --- |")
    for key in sorted(footprint["activation_bytes"]):
        lines.append(f"| {key} | {footprint['activation_bytes'][key]} |")
    lines += [
        "",
        "## Resource plan",
        "",
        f"- hard guard: {guard['limit_gib']} GiB, checked after every optimizer step;",
        "  an over-limit CUDA peak aborts the run instead of continuing;",
        f"- measured on this CPU window: peak process RSS {guard['peak_process_rss_gib']} GiB,"
        f" {guard['checks']} guard check(s);",
        f"- analytic activations per image: {footprint['activation_total_mib']} MiB, i.e."
        f" {round(100.0 * footprint['activation_total_bytes'] / guard['limit_bytes'], 3)}%"
        " of the guard before weights, the CUDA context and cuDNN workspaces;",
        "- the frozen wrapper and its 256-channel stride-8 buffer dominate"
        " (`frozen_stride8_features`), not the transformer;",
        "- GPU peak memory is **not** measured here: the same guard measures it on the",
        "  first CUDA step and aborts there if the budget is exceeded.",
        "",
        "No early stopping is added anywhere: eligibility is checked once, before",
        "training, by `prepare` (trigger > 200 eligible crops per epoch), and the",
        "5-epoch schedule is fixed.",
        "",
        "## Run command (GPU, only after the parent authorises it)",
        "",
        "```",
        bundle["command"],
        "```",
        "",
        "## Not claimed",
        "",
    ]
    for item in bundle["not_claimed"]:
        lines.append(f"- {item}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# ---------------------------------------------------------------------------
# smoke
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
    tap, head, report = load_query_head(semantic_checkpoint, device=device)
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    features = tap.feature_map(image)["features"]
    outputs = head(features)
    semantic_requires_grad = sum(1 for p in tap.model.parameters() if p.requires_grad)
    result = {
        "schema": SMOKE_SCHEMA,
        "version": QUERY_VERSION,
        "device": device,
        "image_shape": [1, 3, size, size],
        "feature_shape": [int(v) for v in features.shape],
        "class_logits_shape": [int(v) for v in outputs["class_logits"].shape],
        "mask_logits_shape": [int(v) for v in outputs["mask_logits"].shape],
        "memory_grid": outputs["memory_grid"],
        "head_parameters": head_parameter_count(head),
        "semantic_parameters_requires_grad": int(semantic_requires_grad),
        "semantic_training": bool(tap.model.training),
        "finite_mask_logits": bool(torch.isfinite(outputs["mask_logits"]).all()),
        "report": report,
    }
    if tuple(result["mask_logits_shape"]) != (
        1,
        NUM_QUERIES,
        size // FEATURE_STRIDE,
        size // FEATURE_STRIDE,
    ):
        raise QueryInstanceError("unexpected mask logits shape")
    if semantic_requires_grad:
        raise QueryInstanceError("semantic parameters are not frozen")
    tap.close()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_query_instance",
        description=(
            "Frozen-semantic query-mask instance lane: CPU prepare, CPU smoke and "
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
    prepare_parser.add_argument("--min-groups", type=int, default=MIN_GROUPS_PER_CROP)
    prepare_parser.add_argument(
        "--min-eligible-crops", type=int, default=MIN_ELIGIBLE_CROPS_PER_EPOCH
    )

    smoke_parser = subparsers.add_parser("smoke", help="CPU checkpoint smoke")
    smoke_parser.add_argument(
        "--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT
    )
    smoke_parser.add_argument("--size", type=int, default=CROP_SIZE)
    smoke_parser.add_argument("--device", default="cpu")

    measure_parser = subparsers.add_parser(
        "measure", help="CPU: one real accumulation window + resource guard"
    )
    measure_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    measure_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    measure_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    measure_parser.add_argument(
        "--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT
    )
    measure_parser.add_argument("--crops", type=int, default=ACCUMULATION_STEPS)
    measure_parser.add_argument("--device", default="cpu")
    measure_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    measure_parser.add_argument("--image-side", type=int, default=1000)

    preflight_parser = subparsers.add_parser(
        "preflight", help="measure real free GPU memory before a run"
    )
    preflight_parser.add_argument("--device", default="cuda")
    preflight_parser.add_argument(
        "--required-bytes", type=int, default=None
    )
    preflight_parser.add_argument(
        "--observed-peak-bytes",
        type=int,
        default=None,
        help="measured peak of this lane; then the rule is free >= peak + headroom",
    )
    preflight_parser.add_argument("--headroom-bytes", type=int, default=GPU_HEADROOM_BYTES)

    ready_parser = subparsers.add_parser(
        "ready", help="CPU: prepare + measure + tests into one readiness bundle"
    )
    ready_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ready_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ready_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ready_parser.add_argument(
        "--semantic-checkpoint", default=DEFAULT_SEMANTIC_CHECKPOINT
    )
    ready_parser.add_argument("--epochs", type=int, default=EPOCHS)
    ready_parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    ready_parser.add_argument("--crops", type=int, default=ACCUMULATION_STEPS)
    ready_parser.add_argument(
        "--skip-tests", action="store_true", help="do not run the lane unit tests"
    )

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
            min_groups=args.min_groups,
            min_eligible_crops=args.min_eligible_crops,
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
    if args.command == "preflight":
        report = preflight_gpu(
            args.device,
            required_bytes=args.required_bytes,
            observed_peak_bytes=args.observed_peak_bytes,
            headroom_bytes=args.headroom_bytes,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["proceed"] else 1
    if args.command == "measure":
        payload = measure(
            args.manifest,
            cache_dir=args.cache_dir,
            semantic_checkpoint=args.semantic_checkpoint,
            output_dir=args.output_dir,
            crops=args.crops,
            device=args.device,
            base_seed=args.base_seed,
            image_side=args.image_side,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "ready":
        payload = ready(
            args.manifest,
            cache_dir=args.cache_dir,
            semantic_checkpoint=args.semantic_checkpoint,
            output_dir=args.output_dir,
            epochs=args.epochs,
            base_seed=args.base_seed,
            crops=args.crops,
            run_tests=not args.skip_tests,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "fit":
        train.require_command_role("fit", args.role)
        records = fullfit.load_fit_records(args.manifest)
        model, payload = pilot.load_checkpoint_model(
            args.semantic_checkpoint, device=args.device
        )
        tap = FrozenFeatureTap(model, device=args.device)
        # Seed before the head is built: the initial head is reproducible.
        train.seed_everything(fullfit.epoch_seed(args.base_seed, 1))
        head = build_query_head(seed=HEAD_INIT_SEED, device=args.device)
        result = train_query_head(
            tap,
            head,
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
