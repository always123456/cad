"""112x112 mask head for the direct-instance delivery CLI (experiment cad-maskres01).

Nothing here modifies the project tree.  The model is built by the project's own
``model.model.build_model`` and only ``roi_heads.mask_predictor`` is replaced:

    14x14 pooled  --conv5_mask(2x deconv 256->256)--> 28
                  --conv6_mask(2x deconv 256->256)--> 56
                  --conv7_mask(2x deconv 256->256)--> 112
                  --mask_fcn_logits(1x1 256->36)----> 36 logits

``conv5_mask`` and ``mask_fcn_logits`` are shape-identical to the frozen
checkpoint's predictor and are reused; ``conv6_mask`` / ``conv7_mask`` are fresh
(torchvision's own kaiming init, deterministic under the same seed), so the fine
checkpoint's ``model_state`` loads strictly into this module and every key that
is not part of ``roi_heads.mask_predictor.`` is bit-identical to the base
checkpoint ``fbc7f50c...92ec8``.

Inference paste stays consistent: ``paste_masks_in_image`` pads by 1 mask cell and
scales boxes by (M+2)/M, which is the same convention the ``paste_consistent``
training target uses at output_size=M, now M=112.

Chunking is mandatory at 112x112 on a 6 GB card.  ``torchvision`` ``roi_heads``
calls the three mask modules once with every detection of the image stacked, so a
112x112 predictor materialises ``N * 256 * 112 * 112`` floats (~3.85 GB at
N=300).  All three modules are per-ROI, so running them over groups of
``<= chunk`` ROIs and concatenating along dim 0 returns the same logits while
peak VRAM stays linear in ``chunk``.  Measured on the fine-head lane: boxes,
scores and labels are bitwise identical to the unchunked path, mask probability
max |delta| 3.5e-4 and zero pixels above 1e-2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

# Provenance of the delivered 112x112 checkpoint (cad-finehead-train02, update 728)
# and of the earlier update-250 snapshot that measured the same architecture. Both
# artefacts stay outside the repository (they are training by-products), so the
# record is the sha256 alone: no machine path is baked into the delivery.
FINE_CKPT_UPDATE250 = {
    "lane": "cad-maskres01/fine_mask112_upt250_snapshot.pt",
    "sha256_prefix": "850481c5",
}
FINE_CKPT_UPDATE728 = {
    "lane": "cad-finehead-train02/ckpt/fine_mask112_u728.pt",
    "sha256": "7a2cd109d59d63116528193145a97b31bbbd96ec19972ed4c3cda44e783461f9",
}

FINE_OUT = 112
POOL = 14
NUM_CLASSES = 36
DIM = 256
PARAMETERS = 3 * (DIM * DIM * 2 * 2 + DIM) + (NUM_CLASSES * DIM + NUM_CLASSES)  # 796452
MASK_PREFIX = "roi_heads.mask_predictor."


class FineMaskPredictor(nn.Module):
    """14x14 -> 112x112 class-specific logits, same convention as MaskRCNNPredictor."""

    def __init__(self, in_channels: int = DIM, dim_reduced: int = DIM,
                 num_classes: int = NUM_CLASSES, pool: int = POOL, out: int = FINE_OUT):
        super().__init__()
        if out != pool * 8:
            raise ValueError("fine head maps pool %d to 8x (%d), got %d" % (pool, pool * 8, out))
        if in_channels != DIM or dim_reduced != DIM:
            raise ValueError("channel dims are fixed to 256/256")
        self.in_channels = int(in_channels)
        self.dim_reduced = int(dim_reduced)
        self.num_classes = int(num_classes)
        self.pool_size = int(pool)
        self.output_size = int(out)
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)   # 14 -> 28
        self.conv6_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)   # 28 -> 56
        self.conv7_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)   # 56 -> 112
        self.relu = nn.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for module in (self.conv5_mask, self.conv6_mask, self.conv7_mask, self.mask_fcn_logits):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv5_mask(x))
        x = self.relu(self.conv6_mask(x))
        x = self.relu(self.conv7_mask(x))
        return self.mask_fcn_logits(x)


def layer_keys(prefix: str = MASK_PREFIX) -> Dict[str, Tuple[int, ...]]:
    """The eight tensors of the head and their shapes (name -> shape)."""

    return {
        "conv5_mask.weight": (DIM, DIM, 2, 2),
        "conv5_mask.bias": (DIM,),
        "conv6_mask.weight": (DIM, DIM, 2, 2),
        "conv6_mask.bias": (DIM,),
        "conv7_mask.weight": (DIM, DIM, 2, 2),
        "conv7_mask.bias": (DIM,),
        "mask_fcn_logits.weight": (NUM_CLASSES, DIM, 1, 1),
        "mask_fcn_logits.bias": (NUM_CLASSES,),
    }


def install_fine_mask_predictor(
    model: nn.Module,
    *,
    old_state: Optional[Dict[str, torch.Tensor]] = None,
    out_size: int = FINE_OUT,
) -> Dict[str, Any]:
    """Replace ``roi_heads.mask_predictor``; reuse every shape-identical tensor.

    ``old_state`` is a checkpoint's ``model_state`` (keys prefixed with
    ``roi_heads.mask_predictor.``), normally the base 28x28 checkpoint.  Returns
    the reuse report: which tensors were copied from it, which are fresh, and any
    shape mismatch.
    """

    head = FineMaskPredictor(out=out_size)
    new_state = head.state_dict()
    reused, fresh, mismatched = [], [], []
    if old_state is not None:
        subset = {
            key[len(MASK_PREFIX):]: value
            for key, value in old_state.items()
            if key.startswith(MASK_PREFIX)
        }
        with torch.no_grad():
            for name, target in new_state.items():
                if name in subset:
                    source = subset[name]
                    if tuple(source.shape) == tuple(target.shape):
                        target.copy_(source)
                        reused.append(name)
                        continue
                    mismatched.append(
                        {"key": name, "old": list(source.shape), "new": list(target.shape)}
                    )
                fresh.append(name)
    else:
        fresh = sorted(new_state.keys())

    old = getattr(model.roi_heads, "mask_predictor", None)
    if old is not None:
        old_shapes = {k: list(v.shape) for k, v in old.state_dict().items()}
    else:
        old_shapes = {}
    model.roi_heads.mask_predictor = head
    return {
        "output_size": int(out_size),
        "pool_size": POOL,
        "reused_keys": sorted(reused),
        "fresh_keys": sorted(fresh),
        "shape_mismatched_keys": mismatched,
        "old_predictor_shapes": old_shapes,
        "parameters": int(sum(p.numel() for p in head.parameters())),
    }


class _MaskStash(nn.Module):
    """First mask module in the chain: remember the inputs, return a token."""

    def __init__(self) -> None:
        super().__init__()
        self.args: Optional[Tuple[Any, Any, Any]] = None

    def forward(self, features, proposals, image_shapes):
        self.args = (features, proposals, image_shapes)
        first = next(iter(features.values())) if hasattr(features, "values") else features[0]
        return torch.zeros(0, device=first.device)


class _MaskIdentity(nn.Module):
    def forward(self, x):
        return x


class _ChunkedMaskRunner(nn.Module):
    """Last mask module in the chain: run pool -> head -> predictor per ROI chunk."""

    def __init__(self, pool: nn.Module, head: nn.Module, predictor: nn.Module,
                 stash: _MaskStash, chunk: int) -> None:
        super().__init__()
        self.pool = pool
        self.head = head
        self.predictor = predictor
        self.stash = stash
        self.chunk = int(chunk)
        self.calls = 0
        self.max_rois = 0

    def forward(self, _token):
        if self.stash.args is None:
            raise RuntimeError("chunked mask: predictor called without pool")
        features, proposals, image_shapes = self.stash.args
        self.stash.args = None
        self.calls += 1
        if proposals:
            self.max_rois = max(self.max_rois, max(int(p.shape[0]) for p in proposals))
        groups: List[list] = []
        current: list = []
        filled = 0
        for image_index, proposal in enumerate(proposals):
            total = int(proposal.shape[0])
            start = 0
            while start < total:
                take = min(self.chunk - filled, total - start)
                current.append((image_index, start, start + take))
                start += take
                filled += take
                if filled >= self.chunk:
                    groups.append(current)
                    current, filled = [], 0
        if current:
            groups.append(current)
        outputs = []
        for group in groups:
            sub_proposals = [proposals[i][a:b] for i, a, b in group]
            sub_shapes = [image_shapes[i] for i, a, b in group]
            pooled = self.pool(features, sub_proposals, sub_shapes)
            outputs.append(self.predictor(self.head(pooled)))
        if not outputs:
            raise RuntimeError("chunked mask: empty ROI batch")
        return torch.cat(outputs, dim=0)


def install_chunked_mask(model: nn.Module, *, chunk: int = 24) -> Dict[str, Any]:
    """Inference-only memory fix: run pool -> head -> predictor over ROI chunks.

    ``chunk <= 0`` restores the original modules (the unchunked path, which OOMs
    at 112x112 on a 6 GB card).
    """

    roi_heads = model.roi_heads
    if hasattr(model, "_mask_orig"):
        pool, head, predictor = model._mask_orig
    else:
        pool = roi_heads.mask_roi_pool
        head = roi_heads.mask_head
        predictor = roi_heads.mask_predictor
        model._mask_orig = (pool, head, predictor)
    if pool is None or head is None or predictor is None:
        raise RuntimeError("model has no mask branch to chunk")
    chunk = int(chunk)
    if chunk <= 0:
        roi_heads.mask_roi_pool = pool
        roi_heads.mask_head = head
        roi_heads.mask_predictor = predictor
        return {"chunk": 0, "installed": False}
    stash = _MaskStash()
    runner = _ChunkedMaskRunner(pool, head, predictor, stash, chunk)
    roi_heads.mask_roi_pool = stash
    roi_heads.mask_head = _MaskIdentity()
    roi_heads.mask_predictor = runner
    return {"chunk": chunk, "installed": True, "runner": runner}


def build_with_fine_head(
    config,
    *,
    out_size: int = FINE_OUT,
    old_state: Optional[Dict[str, torch.Tensor]] = None,
    weights_file: Optional[str] = None,
    model_root: str = "",
):
    """Build through the project builder, then swap in the fine mask predictor."""

    if model_root:
        import sys

        if model_root not in sys.path:
            sys.path.insert(0, model_root)
    from model import model as detector  # the project package, unmodified

    model, info = detector.build_model(config, weights_file=weights_file)
    report = install_fine_mask_predictor(model, old_state=old_state, out_size=out_size)
    info = dict(info)
    info["mask_head_spatial_output"] = int(out_size)
    info["mask_predictor"] = report
    info["fine_mask_head"] = (
        "conv5_mask 14->28 (reused) + conv6_mask 28->56 + conv7_mask 56->112 + 1x1 logits 36"
    )
    return model, info, report


def load_fine_checkpoint(
    checkpoint: str,
    *,
    device: torch.device,
    out_size: int = FINE_OUT,
    model_root: str = "",
):
    """Loader compatible with ``instance_eval.batch.run_batch(loader=...)``.

    Returns ``(model, config, info, payload)`` exactly like
    ``model.infer.load_checkpoint``, with the 112x112 predictor in place and the
    checkpoint loaded strictly.
    """

    import dataclasses

    if model_root:
        import sys

        if model_root not in sys.path:
            sys.path.insert(0, model_root)
    from model import model as detector

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError("checkpoint has no model_state: %r" % checkpoint)
    declared = payload.get("fine_mask") or {}
    head_out = int(declared.get("output_size", out_size))
    if head_out != int(out_size):
        raise RuntimeError(
            "checkpoint declares mask head %r, --mask-head 112 needs %d" % (head_out, out_size)
        )
    config = dataclasses.replace(
        detector.DetectorConfig.from_dict(payload["config"]), weights="none"
    )
    model, info, _report = build_with_fine_head(
        config,
        out_size=head_out,
        old_state=None,
        weights_file=None,
        model_root=model_root,
    )
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=True)
    if missing or unexpected:  # pragma: no cover - strict=True already refused
        raise RuntimeError(
            "strict load mismatch: missing=%r unexpected=%r" % (missing, unexpected)
        )
    model.to(device)
    model.eval()
    detector.set_inference_transform(model, config)
    info = dict(info)
    info["mask_roi_pool_output_size"] = int(model.roi_heads.mask_roi_pool.output_size[0])
    info["fine_mask_declared"] = {
        key: value for key, value in declared.items() if key != "model_state"
    }
    return model, config, info, payload


def head_report(payload: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance for the receipt: what the checkpoint says about its own head."""

    declared = payload.get("fine_mask") or {}
    state = payload.get("model_state") or {}
    mask_keys = sorted(key for key in state if key.startswith(MASK_PREFIX))
    return {
        "output_size": int(declared.get("output_size", report.get("output_size", 0))),
        "pool_size": POOL,
        "deconv_stages": 3,
        "parameters": int(report.get("parameters") or PARAMETERS),
        "reused_keys": list(report.get("reused_keys") or []),
        "fresh_keys": list(report.get("fresh_keys") or []),
        "shape_mismatched_keys": list(report.get("shape_mismatched_keys") or []),
        "checkpoint_mask_tensors": mask_keys,
        "checkpoint_other_tensors": int(len(state) - len(mask_keys)),
        "declared": {k: v for k, v in declared.items() if k != "model_state"},
        "load": "torch.load_state_dict(strict=True) on the whole state dict",
    }
