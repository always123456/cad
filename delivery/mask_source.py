"""Instance stroke-mask head at 256^2, as a delivery mask source.

The shipped 112x112 mask branch cannot resolve a 1 px stroke inside a box wider
than 112 px (median mask IoU 0.06 on the >1000 px^2 bucket), and the geometry-only
fallback (`ink` inside the box widened by 2 px) over-covers, because the crop
takes every stroke of the neighbouring symbols too.  This module runs the trained
``MaskHeadUNet`` on the *ink* of one box at a time, at 256^2, and returns the
native-resolution boolean mask

    ink >= gate  AND  prob(pasted to the window) >= tau

which is the column measured in ``evidence/instance-mask-head-256-20261120/``
(median IoU 0.6098 on the large detected-box bucket, tau 0.30, gate 96).

The window rule, the area resample, the four input channels and the paste are
copied from ``tools/instance_mask_head.py`` (``window_of`` / ``crop_native`` /
``resample`` / ``box_channel`` / ``uv_channels`` / ``make_sample`` /
``paste_native`` / ``MaskHeadUNet``) and must stay identical to it: the checkpoint
was trained on exactly those crops.  Nothing in ``tools/`` is imported at
runtime, so a delivery container needs neither that file nor its data caches.

Everything here is a pure function of the ink raster, the boxes and the
checkpoint; the caller decides which instances to replace.
"""

from __future__ import annotations

import os

import numpy as np

try:  # the delivery lane always has torch, but importing this module must not fail
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only without torch  # noqa: BLE001
    torch = None
    nn = None
    F = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAD_SIZE = 256          # crop resolution the 256^2 head works at
HEAD_WIDTH = 32          # UNet width of that arm (1,948,577 params)
HEAD_PAD = 8             # padding around the given box, in native pixels
HEAD_MIN_SIDE = 64       # smaller boxes are zoomed to this window side
HEAD_TAU = 0.30          # probability threshold, fitted on fit stems (gate 96)
HEAD_GATE = 96           # ink grey level the selection is clamped to
HEAD_MIN_INK_SIDE = 64   # box side above which the rule replaces the mask
HEAD_BATCH = 16
HEAD_PARAMETERS = 1948577

HEAD_STAT_KEYS = (
    "instances",
    "replaced",
    "below_threshold",
    "empty_head_kept_original",
    "empty_masks",
)

# The arm ships only once it clears the +0.008 PQ gate, so the checkpoint lives in
# ``runs/`` until then and in ``weights/`` after (see weights/MANIFEST.json).
WEIGHTS_CKPT = os.path.join(REPO, "weights", "mask_head256_u1.pt")
RUNS_CKPT = os.path.join(REPO, "runs", "instance_mask_head256", "best.pt")


def default_checkpoint():
    """``weights/mask_head256_u1.pt`` when it exists, else the training run's ``best.pt``."""

    if os.path.isfile(WEIGHTS_CKPT):
        return WEIGHTS_CKPT
    return RUNS_CKPT


# --------------------------------------------------------------------------- #
# ink: the continuous map the head was trained on
# --------------------------------------------------------------------------- #
def continuous_ink(rgba):
    """``255 - min(RGB composited over white)``; 0 = white, 255 = black.

    The same arithmetic as ``delivery/refine.ink_from_rgba`` (composite128 is
    exactly ``> 127`` here, because ``min(RGB) < 128`` on the composite) and the
    same as ``tools/instance_mask_head.ink_continuous``.
    """

    array = np.asarray(rgba)
    rgb = array[..., :3].astype(np.int32, copy=False)
    if array.shape[2] >= 4:
        alpha = array[..., 3:4].astype(np.int32, copy=False)
    else:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.int32)
    composited = (rgb * alpha + 255 * (255 - alpha)) // 255
    return (255 - composited.min(axis=2)).astype(np.uint8)


def continuous_ink_from_png(path):
    """Continuous ink map of a PNG at its native raster."""

    from PIL import Image  # noqa: PLC0415 - mirrors the tool, which needs Pillow too

    with Image.open(path) as handle:
        rgba = np.asarray(handle.convert("RGBA"))
    return continuous_ink(rgba)


# --------------------------------------------------------------------------- #
# crops: exact integer window -> area resample
# --------------------------------------------------------------------------- #
def window_of(box, pad=HEAD_PAD, min_side=HEAD_MIN_SIDE):
    """Square integer ``(x0, x1, y0, y1)`` window around a box.

    ``side = max(box_side, min_side) + pad``, so a small symbol is zoomed to the
    minimum window instead of being resampled to nothing.  The window may leave
    the canvas: the crop is zero-padded there.
    """

    x0, y0, x1, y1 = (float(v) for v in box)
    side = max(x1 - x0, y1 - y0, float(min_side)) + float(pad)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    xa = int(np.floor(cx - side / 2.0))
    ya = int(np.floor(cy - side / 2.0))
    xb = int(np.ceil(cx + side / 2.0))
    yb = int(np.ceil(cy + side / 2.0))
    return xa, max(xb, xa + 2), ya, max(yb, ya + 2)


def crop_native(image, window):
    """Zero-padded native-resolution crop of ``window``."""

    xa, xb, ya, yb = window
    out = np.zeros((yb - ya, xb - xa), dtype=image.dtype)
    sx0, sx1 = max(0, xa), min(image.shape[1], xb)
    sy0, sy1 = max(0, ya), min(image.shape[0], yb)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa] = image[sy0:sy1, sx0:sx1]
    return out


def resample(crop, size=HEAD_SIZE):
    """Area-resample a native crop: each cell becomes its mean ink density."""

    tensor = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))
    return F.interpolate(tensor.view(1, 1, *tensor.shape), size=(size, size),
                         mode="area")[0, 0].numpy()


def box_channel(window, box, size=HEAD_SIZE):
    """1 inside the given box, in crop-normalised coordinates."""

    xa, xb, ya, yb = window
    span_x, span_y = float(xb - xa), float(yb - ya)
    ramp = (np.arange(size, dtype=np.float32) + 0.5) / size
    u = ramp * span_x + xa
    v = ramp * span_y + ya
    bx0, by0, bx1, by1 = (float(t) for t in box)
    inside = (u[None, :] >= bx0) & (u[None, :] < bx1) & (v[:, None] >= by0) & (v[:, None] < by1)
    return inside.astype(np.float32)


def uv_channels(size=HEAD_SIZE):
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    u = np.repeat(coords[None, :], size, axis=0)
    v = np.repeat(coords[:, None], size, axis=1)
    return u, v


def make_crop(ink, box, window=None, size=HEAD_SIZE, pad=HEAD_PAD, min_side=HEAD_MIN_SIDE):
    """``(4 x size x size crops, window)`` for one box on the ink raster."""

    if window is None:
        window = window_of(box, pad=pad, min_side=min_side)
    ink_crop = resample(crop_native(ink, window), size) / 255.0
    u, v = uv_channels(size)
    crops = np.stack([ink_crop, box_channel(window, box, size), u, v], axis=0).astype(np.float32)
    return crops, window


def paste_native(prob, window, shape):
    """Bilinearly upsample a ``size`` probability map into ``window`` of ``shape``."""

    xa, xb, ya, yb = window
    tensor = torch.from_numpy(np.ascontiguousarray(prob, dtype=np.float32))[None, None]
    big = F.interpolate(tensor, size=(yb - ya, xb - xa), mode="bilinear",
                        align_corners=False)[0, 0].numpy()
    out = np.zeros(shape, dtype=np.float32)
    sx0, sx1 = max(0, xa), min(shape[1], xb)
    sy0, sy1 = max(0, ya), min(shape[0], yb)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0:sy1, sx0:sx1] = big[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
    return out


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class MaskHeadUNet(nn.Module if nn is not None else object):
    """3-level UNet, one logit out; byte-identical to ``tools/instance_mask_head.py``."""

    def __init__(self, width: int = HEAD_WIDTH, in_ch: int = 4):
        super().__init__()
        c = int(width)
        self.stem = _block(in_ch, c)
        self.down1 = _block(c, 2 * c)
        self.down2 = _block(2 * c, 4 * c)
        self.down3 = _block(4 * c, 8 * c)
        self.up3 = _block(8 * c + 4 * c, 4 * c)
        self.up2 = _block(4 * c + 2 * c, 2 * c)
        self.up1 = _block(2 * c + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, crops):
        s0 = self.stem(crops)
        s1 = self.down1(F.max_pool2d(s0, 2))
        s2 = self.down2(F.max_pool2d(s1, 2))
        s3 = self.down3(F.max_pool2d(s2, 2))
        y = self.up3(torch.cat([F.interpolate(s3, size=s2.shape[-2:], mode="bilinear",
                                              align_corners=False), s2], 1))
        y = self.up2(torch.cat([F.interpolate(y, size=s1.shape[-2:], mode="bilinear",
                                              align_corners=False), s1], 1))
        y = self.up1(torch.cat([F.interpolate(y, size=s0.shape[-2:], mode="bilinear",
                                              align_corners=False), s0], 1))
        return self.head(y)


def load_head(path=None, device="cpu"):
    """``(model in eval mode, meta)`` from a checkpoint of ``tools/instance_mask_head.py``.

    That trainer saves ``vars(args)`` next to the weights, and ``args`` carries the
    ``set_defaults(func=command_train)`` pointer, so unpickling needs a stand-in
    for that name in ``__main__``.  It is never called.
    """

    import __main__  # noqa: PLC0415 - only to satisfy the pickled namespace

    if not hasattr(__main__, "command_train"):
        setattr(__main__, "command_train", _unused_command)
    path = default_checkpoint() if path is None else str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("mask-head checkpoint not found: %s" % (path,))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload.get("args") or {}
    size = int(args.get("size") or HEAD_SIZE)
    width = int(args.get("width") or HEAD_WIDTH)
    model = MaskHeadUNet(width=width)
    model.load_state_dict(payload["model"])
    model.eval().to(device)
    return model, {
        "checkpoint": os.path.abspath(path),
        "size": size,
        "width": width,
        "epoch": payload.get("epoch"),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
    }


def _unused_command(*_args, **_kwargs):
    return None


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def head_masks(ink, boxes, model, size=HEAD_SIZE, tau=HEAD_TAU, gate=HEAD_GATE,
               device="cpu", batch=HEAD_BATCH, pad=HEAD_PAD, min_side=HEAD_MIN_SIDE):
    """One native-resolution boolean mask per box: ``ink >= gate`` & ``prob >= tau``.

    ``ink`` is the continuous 0..255 map at the emitter's raster (the same raster
    the masks live on), ``boxes`` are ``[x0, y0, x1 + 1, y1 + 1]`` in that raster.
    A box with no ink above ``gate`` and no probability above ``tau`` comes back
    empty; the caller keeps the refined mask in that case.
    """

    if torch is None:  # pragma: no cover - the delivery lane always has torch
        raise RuntimeError("torch is required for --mask-source head")
    ink = np.asarray(ink)
    shape = (int(ink.shape[0]), int(ink.shape[1]))
    gate_map = ink >= int(gate)
    out = []
    if not len(boxes):
        return out
    device = str(device)
    with torch.no_grad():
        for start in range(0, len(boxes), int(batch)):
            chunk = list(boxes)[start:start + int(batch)]
            crops, windows = [], []
            for box in chunk:
                crop, window = make_crop(ink, box, size=size, pad=pad, min_side=min_side)
                crops.append(crop)
                windows.append(window)
            tensor = torch.from_numpy(np.stack(crops)).to(device)
            prob = torch.sigmoid(model(tensor).float())[:, 0].cpu().numpy()
            for index, window in enumerate(windows):
                big = paste_native(prob[index], window, shape)
                out.append(np.logical_and(big >= float(tau), gate_map))
    return out
