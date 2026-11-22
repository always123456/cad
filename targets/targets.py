"""Canonical direct-instance target contract for the renewed PNG route.

The PNG is the only model input; the 12-field official GT document supplies the
canonical class/instance keys.  This module turns the pair into per-key masks
plus a label map, exclusive pixel boxes, a dense ink mask and a loss domain.

Hard rules (see ``README.md`` for the full contract):

* one mask per canonical key: Thing ``0..29`` with ``instance >= 0`` keyed
  ``(class, instance)``; Stuff ``30..34`` keyed ``(class, -1)``;
  ``semantic == 35`` and Thing ``instance < 0`` are skipped entirely;
* masks come from the project rasterizer
  ``src/cadvec/recognition_raster_data.py`` (``_draw_mask`` / ``_segment_key``)
  with canvas ``(100, 100)`` and one pixel per native pixel, so they are
  bit-identical to the canonical scorer's raster;
* ``labels = canonical class + 1`` (0 means unknown / no target);
* ``ink`` follows the project PNG policy: ``min(R, G, B) < 254 / 255`` after
  compositing over white;
* ``loss_domain = union(all target masks) | blank_negative`` where
  ``blank_negative = ~ink & ~unknown_geometry & ~union``.  Ink that no
  canonical key explains is excluded; skipped GT geometry (semantic 35, Thing
  without instance, e.g. an arc's four-point chord) is never a negative even
  when it falls outside the PNG ink; and a known target pixel whose raster
  includes no PNG ink is still a positive, never a negative.

The image array is used only for shape and ink.  Geometry comes exclusively
from the document, never from pixels, and masks are *not* clipped to ink: ink
only defines the loss domain.  ``crop_targets`` is a plain window utility, not
a sampling policy.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# contract constants (canonical label space, mirrors recognition_contract)
# --------------------------------------------------------------------------- #

THING_MIN = 0
THING_MAX = 29
STUFF_MIN = 30
STUFF_MAX = 34
IGNORE_CLASS = 35
LABEL_OFFSET = 1

CANVAS_SIZE: Tuple[float, float] = (100.0, 100.0)
INK_SUPPORT_THRESHOLD = 254
INK_MIN_VALUE = INK_SUPPORT_THRESHOLD / 255.0

CANONICAL_DOCUMENT_FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
)

# The checkout locates its own ``src``; ``CADVEC_SRC_DIR`` overrides it when the
# package is vendored somewhere else.  Nothing machine-specific is baked in: the
# search walks up from this file, so the tree works at any path (repo root,
# ``runs/<lane>/targets``, a copied delivery bundle, a container mount).
# A private package alias keeps the project modules from colliding with any
# installed ``cadvec`` and keeps their relative imports intact.
PROJECT_PACKAGE = "_cadvec_direct_instance_targets"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _locate_src_dir(start: str) -> str:
    """Nearest ancestor whose ``src`` holds the project rasterizer, or ``""``."""

    current = start
    for _ in range(8):
        candidate = os.path.join(current, "src")
        if os.path.isfile(os.path.join(candidate, "cadvec", "recognition_raster_data.py")):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return ""


_DEFAULT_SRC_DIR = _locate_src_dir(_HERE)

_RASTER_MODULE: Any = None

_INK_SUPPORT_POLICY = {
    "source": "alpha-composited PNG RGB only",
    "channel_threshold_8bit": INK_SUPPORT_THRESHOLD,
    "rule": "min(R, G, B) < threshold",
    "background": "UNKNOWN",
}

MASK_POLICY = (
    "raw canonical rasterizer output; never clipped to ink "
    "(ink only defines loss_domain)"
)


class TargetError(ValueError):
    """Malformed image, document or target input; the contract fails closed."""


__all__ = [
    "TargetError",
    "CANVAS_SIZE",
    "CANONICAL_DOCUMENT_FIELDS",
    "IGNORE_CLASS",
    "INK_MIN_VALUE",
    "INK_SUPPORT_THRESHOLD",
    "LABEL_OFFSET",
    "MASK_POLICY",
    "PROJECT_PACKAGE",
    "STUFF_MAX",
    "STUFF_MIN",
    "THING_MAX",
    "THING_MIN",
    "build_targets",
    "crop_targets",
    "project_paths",
    "project_raster",
]


# --------------------------------------------------------------------------- #
# project rasterizer import (explicit source path, private package alias)
# --------------------------------------------------------------------------- #


def project_paths() -> Dict[str, str]:
    """Return the resolved project ``src``/``cadvec`` paths used for rastering."""

    src = os.environ.get("CADVEC_SRC_DIR") or _DEFAULT_SRC_DIR
    if not src:
        raise TargetError(
            "no project src/ found from %r; set CADVEC_SRC_DIR to the checkout's src/"
            % _HERE
        )
    src = os.path.abspath(src)
    cadvec = os.path.join(src, "cadvec")
    return {
        "src": src,
        "cadvec": cadvec,
        "raster_data": os.path.join(cadvec, "recognition_raster_data.py"),
    }


def project_raster() -> Any:
    """Import the canonical project rasterizer under a private package alias."""

    global _RASTER_MODULE
    if _RASTER_MODULE is not None:
        return _RASTER_MODULE
    paths = project_paths()
    if not os.path.isfile(paths["raster_data"]):
        raise TargetError(
            "project rasterizer not found at %r; set CADVEC_SRC_DIR"
            % paths["raster_data"]
        )
    package = sys.modules.get(PROJECT_PACKAGE)
    if package is None:
        spec = importlib.util.spec_from_file_location(
            PROJECT_PACKAGE,
            os.path.join(paths["cadvec"], "__init__.py"),
            submodule_search_locations=[paths["cadvec"]],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise TargetError("cannot build an import spec for %r" % paths["cadvec"])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PROJECT_PACKAGE] = package
        try:
            spec.loader.exec_module(package)
        except Exception as exc:  # pragma: no cover - broken checkout
            sys.modules.pop(PROJECT_PACKAGE, None)
            raise TargetError("cannot import project package: %s" % exc) from exc
    try:
        module = importlib.import_module(PROJECT_PACKAGE + ".recognition_raster_data")
    except Exception as exc:  # pragma: no cover - broken checkout
        raise TargetError("cannot import project rasterizer: %s" % exc) from exc
    for name in ("_draw_mask", "_segment_key"):
        if not hasattr(module, name):  # pragma: no cover - API drift
            raise TargetError("project rasterizer lacks %r" % name)
    _RASTER_MODULE = module
    return module


# --------------------------------------------------------------------------- #
# scalar / array validation helpers
# --------------------------------------------------------------------------- #


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TargetError("%s must be an integer; got %r" % (name, value))
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TargetError("%s must be a real number; got %r" % (name, value))
    number = float(value)
    if not math.isfinite(number):
        raise TargetError("%s must be finite; got %r" % (name, value))
    return number


def _as_sequence(value: Any, name: str) -> List[Any]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TargetError("%s must be a sequence; got %r" % (name, type(value).__name__))
    try:
        return list(value)
    except TypeError as exc:
        raise TargetError("%s must be iterable: %s" % (name, exc)) from exc


def _canvas_size(canvas_size: Any) -> Tuple[float, float]:
    values = _as_sequence(canvas_size, "canvas_size")
    if len(values) != 2:
        raise TargetError("canvas_size must be a (width, height) pair")
    width = _as_float(values[0], "canvas_size[0]")
    height = _as_float(values[1], "canvas_size[1]")
    if width <= 0 or height <= 0:
        raise TargetError("canvas_size must be positive")
    return width, height


def _resolve_raster_size(
    raster_size: Any, width: int, height: int
) -> Tuple[int, int]:
    if raster_size is None:
        return width, height
    if isinstance(raster_size, (tuple, list)):
        values = _as_sequence(raster_size, "raster_size")
        if len(values) != 2:
            raise TargetError("raster_size sequence must be (width, height)")
        raster_w = _as_int(values[0], "raster_size[0]")
        raster_h = _as_int(values[1], "raster_size[1]")
    else:
        raster_w = raster_h = _as_int(raster_size, "raster_size")
    if raster_w <= 0 or raster_h <= 0:
        raise TargetError("raster_size must be positive")
    if (raster_w, raster_h) != (width, height):
        raise TargetError(
            "raster_size %r must equal the image spatial shape (width=%d, height=%d); "
            "the contract is native-resolution and never rescales"
            % ((raster_w, raster_h), width, height)
        )
    return raster_w, raster_h


def _image_array(image: Any) -> Tuple[np.ndarray, str, int, int]:
    """Validate/normalise an image to float32 CHW in ``[0, 1]``.

    Accepts numpy arrays, array-likes and CPU/GPU tensors exposing
    ``detach().cpu().numpy()``.  Only RGB (3 channel) CHW or HWC input is
    accepted; anything else fails closed.
    """

    if image is None:
        raise TargetError("image is required")
    value = image
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            value = value.detach().cpu().numpy()
        except Exception as exc:
            raise TargetError("cannot convert tensor image to numpy: %s" % exc) from exc
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TargetError("image is not array-like: %s" % exc) from exc
    if array.ndim != 3:
        raise TargetError(
            "image must be 3D CHW or HWC; got shape %r" % (tuple(array.shape),)
        )
    if array.dtype == np.bool_ or not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise TargetError("image dtype must be numeric RGB; got %r" % (array.dtype,))
    if array.shape[0] == 3:
        layout = "chw"
        height, width = int(array.shape[1]), int(array.shape[2])
        channels = np.ascontiguousarray(array, dtype=np.float32)
    elif array.shape[2] == 3:
        layout = "hwc"
        height, width = int(array.shape[0]), int(array.shape[1])
        channels = np.ascontiguousarray(np.transpose(array, (2, 0, 1)), dtype=np.float32)
    else:
        raise TargetError(
            "image needs exactly 3 RGB channels on axis 0 (CHW) or axis 2 (HWC); "
            "got shape %r" % (tuple(array.shape),)
        )
    if height <= 0 or width <= 0:
        raise TargetError("image spatial size must be positive")
    if not np.isfinite(channels).all():
        raise TargetError("image contains non-finite values")
    if np.issubdtype(array.dtype, np.integer):
        low = float(channels.min())
        high = float(channels.max())
        if low < 0.0 or high > 255.0:
            raise TargetError(
                "integer image values must be in 0..255; got [%g, %g]" % (low, high)
            )
        channels = channels / 255.0
    else:
        low = float(channels.min())
        high = float(channels.max())
        if low < -1e-6 or high > 1.0 + 1e-6:
            raise TargetError(
                "float image values must be in [0, 1]; got [%g, %g]" % (low, high)
            )
        channels = np.clip(channels, 0.0, 1.0)
    return channels, layout, height, width


# --------------------------------------------------------------------------- #
# document validation
# --------------------------------------------------------------------------- #


def _document_primitives(document: Any) -> List[Dict[str, Any]]:
    """Validate a GT document and return canonical primitive records."""

    if not isinstance(document, Mapping):
        raise TargetError("gt_document must be a mapping")
    missing = [field for field in CANONICAL_DOCUMENT_FIELDS if field not in document]
    if missing:
        raise TargetError("gt_document is missing fields %r" % (missing,))

    commands = _as_sequence(document["commands"], "commands")
    args = _as_sequence(document["args"], "args")
    lengths = _as_sequence(document["lengths"], "lengths")
    semantic_ids = _as_sequence(document["semanticIds"], "semanticIds")
    instance_ids = _as_sequence(document["instanceIds"], "instanceIds")
    count = len(commands)
    for name, values in (
        ("args", args),
        ("lengths", lengths),
        ("semanticIds", semantic_ids),
        ("instanceIds", instance_ids),
    ):
        if len(values) != count:
            raise TargetError(
                "gt_document.%s has length %d, expected %d" % (name, len(values), count)
            )

    width = _as_float(document["width"], "width")
    height = _as_float(document["height"], "height")
    if width <= 0 or height <= 0:
        raise TargetError("gt_document width/height must be positive")

    records: List[Dict[str, Any]] = []
    for index in range(count):
        command = _as_int(commands[index], "commands[%d]" % index)
        if command not in (0, 1, 2, 3):
            raise TargetError(
                "commands[%d]=%r is unsupported; only line (0,1) and "
                "circle/ellipse (2,3) exist" % (index, command)
            )
        row = _as_sequence(args[index], "args[%d]" % index)
        if len(row) != 8:
            raise TargetError("args[%d] must hold four x,y points" % index)
        points = [
            (
                _as_float(row[offset], "args[%d][%d]" % (index, offset)),
                _as_float(row[offset + 1], "args[%d][%d]" % (index, offset + 1)),
            )
            for offset in (0, 2, 4, 6)
        ]
        semantic = _as_int(semantic_ids[index], "semanticIds[%d]" % index)
        if not (THING_MIN <= semantic <= IGNORE_CLASS):
            raise TargetError(
                "semanticIds[%d]=%r is outside 0..35" % (index, semantic)
            )
        instance = _as_int(instance_ids[index], "instanceIds[%d]" % index)
        if instance < -1:
            raise TargetError("instanceIds[%d]=%r is below -1" % (index, instance))
        records.append(
            {
                "index": index,
                "command": command,
                "points": points,
                "semantic": semantic,
                "instance": instance,
            }
        )
    return records


def _skipped_reason(semantic: int, instance: int) -> str:
    if semantic == IGNORE_CLASS:
        return "semantic35"
    if THING_MIN <= semantic <= THING_MAX and instance < 0:
        return "thing_missing_instance"
    return "other"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def build_targets(
    image: Any,
    gt_document: Mapping[str, Any],
    *,
    raster_size: Any = None,
    canvas_size: Tuple[float, float] = CANVAS_SIZE,
) -> Dict[str, Any]:
    """Build canonical per-key target masks from one PNG and its GT document.

    ``image`` supplies only the native shape and the ink support; geometry comes
    solely from ``gt_document``.  Masks are rasterised at native resolution by
    the project rasterizer, so ``raster_size`` is an assertion, not a rescale.
    """

    channels, layout, height, width = _image_array(image)
    canvas = _canvas_size(canvas_size)
    _resolve_raster_size(raster_size, width, height)
    records = _document_primitives(gt_document)
    raster = project_raster()

    ink = np.min(channels, axis=0) < INK_MIN_VALUE

    grouped: Dict[Tuple[int, int], np.ndarray] = {}
    unknown_geometry = np.zeros((height, width), dtype=bool)
    skipped = {"semantic35": 0, "thing_missing_instance": 0, "other": 0}
    for record in records:
        key = raster._segment_key(record["semantic"], record["instance"])
        mask = raster._draw_mask(
            record["command"], record["points"], width, height, canvas
        )
        if key is None:
            unknown_geometry |= mask
            skipped[_skipped_reason(record["semantic"], record["instance"])] += 1
            continue
        if key in grouped:
            grouped[key] |= mask
        else:
            grouped[key] = np.array(mask, dtype=bool, copy=True)

    keys = tuple(sorted(grouped))
    masks = np.zeros((len(keys), height, width), dtype=bool)
    for index, key in enumerate(keys):
        masks[index] = grouped[key]

    labels = np.zeros((height, width), dtype=np.int64)
    union = np.zeros((height, width), dtype=bool)
    overlap = np.zeros((height, width), dtype=bool)
    conflict = np.zeros((height, width), dtype=bool)
    for index, key in enumerate(keys):
        mask = masks[index]
        overlap |= union & mask
        value = key[0] + LABEL_OFFSET
        labelled = labels > 0
        conflict |= labelled & mask & (labels != value)
        labels[mask & ~labelled] = value
        union |= mask

    boxes = np.zeros((len(keys), 4), dtype=np.float32)
    empty = 0
    for index in range(len(keys)):
        rows = np.flatnonzero(masks[index].any(axis=1))
        cols = np.flatnonzero(masks[index].any(axis=0))
        if rows.size == 0 or cols.size == 0:
            empty += 1
            continue
        boxes[index] = (
            float(cols[0]),
            float(rows[0]),
            float(cols[-1] + 1),
            float(rows[-1] + 1),
        )

    # A blank pixel licenses a negative only when no target mask and no skipped
    # GT geometry (e.g. an arc's four-point chord) touches it.  Known targets
    # keep precedence, so their masks may overlap each other but never the
    # negative mask; the domain is the union of both.
    blank = ~ink & ~unknown_geometry & ~union
    loss_domain = union | blank
    known_ink = ink & union
    unknown_ink = ink & ~union
    unknown_from_geometry = unknown_ink & unknown_geometry
    unknown_geometry_blank = unknown_geometry & ~ink

    diagnostics = {
        "primitive_count": len(records),
        "target_count": len(keys),
        "empty_target_count": int(empty),
        "ignored_semantic35_primitives": int(skipped["semantic35"]),
        "skipped_thing_missing_instance_primitives": int(
            skipped["thing_missing_instance"]
        ),
        "skipped_primitive_count": int(sum(skipped.values())),
        "ink_pixels": int(np.count_nonzero(ink)),
        "ink_free_pixels": int(np.count_nonzero(~ink)),
        "target_pixels": int(np.count_nonzero(union)),
        "known_ink_pixels": int(np.count_nonzero(known_ink)),
        "unknown_ink_pixels": int(np.count_nonzero(unknown_ink)),
        "unknown_ink_from_skipped_geometry_pixels": int(
            np.count_nonzero(unknown_from_geometry)
        ),
        "unknown_ink_residual_pixels": int(
            np.count_nonzero(unknown_ink & ~unknown_geometry)
        ),
        "loss_domain_pixels": int(np.count_nonzero(loss_domain)),
        "blank_pixels": int(np.count_nonzero(blank)),
        "unknown_geometry_pixels": int(np.count_nonzero(unknown_geometry)),
        "unknown_geometry_ink_pixels": int(
            np.count_nonzero(unknown_geometry & ink)
        ),
        "unknown_geometry_blank_pixels": int(np.count_nonzero(unknown_geometry_blank)),
        "overlap_pixels": int(np.count_nonzero(overlap)),
        "conflict_pixels": int(np.count_nonzero(conflict)),
        "ink_support_threshold_8bit": INK_SUPPORT_THRESHOLD,
    }

    return {
        "masks": masks,
        "labels": labels,
        "boxes": boxes,
        "keys": keys,
        "ink": ink,
        "loss_domain": loss_domain,
        "unknown_geometry": unknown_geometry,
        "blank_negative": blank,
        "native_size": {"height": int(height), "width": int(width)},
        "raster_size": {"width": int(width), "height": int(height)},
        "canvas_size": {"width": float(canvas[0]), "height": float(canvas[1])},
        "image_layout": layout,
        "label_offset": LABEL_OFFSET,
        "mask_policy": MASK_POLICY,
        "ink_policy": dict(_INK_SUPPORT_POLICY),
        "domain_policy": (
            "union(target masks) | ((~ink) & (~unknown_geometry) & (~union)); "
            "negatives are disjoint from every target mask and from skipped "
            "GT geometry, which is UNKNOWN"
        ),
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------- #
# crop utility (window only, no sampling policy, no rescaling)
# --------------------------------------------------------------------------- #

_TARGET_FIELDS = (
    "masks",
    "labels",
    "boxes",
    "keys",
    "ink",
    "loss_domain",
    "unknown_geometry",
    "blank_negative",
    "native_size",
)


def _native_hw(value: Any) -> Tuple[int, int]:
    if isinstance(value, Mapping):
        if "height" in value and "width" in value:
            height = _as_int(value["height"], "native_size.height")
            width = _as_int(value["width"], "native_size.width")
        else:
            raise TargetError("native_size mapping needs height and width")
    else:
        values = _as_sequence(value, "native_size")
        if len(values) != 2:
            raise TargetError("native_size must be (height, width)")
        height = _as_int(values[0], "native_size[0]")
        width = _as_int(values[1], "native_size[1]")
    if height <= 0 or width <= 0:
        raise TargetError("native_size must be positive")
    return height, width


def _validate_target(target: Any) -> Dict[str, Any]:
    if not isinstance(target, Mapping):
        raise TargetError("target must be a mapping")
    missing = [field for field in _TARGET_FIELDS if field not in target]
    if missing:
        raise TargetError("target is missing fields %r" % (missing,))
    height, width = _native_hw(target["native_size"])
    masks = np.asarray(target["masks"])
    if masks.ndim != 3:
        raise TargetError("target masks must be NHW; got shape %r" % (masks.shape,))
    if masks.shape[1:] != (height, width):
        raise TargetError(
            "target masks %r disagree with native_size %r"
            % (masks.shape[1:], (height, width))
        )
    count = int(masks.shape[0])
    if masks.dtype != np.bool_:
        raise TargetError("target masks must be bool; got %r" % (masks.dtype,))
    labels = np.asarray(target["labels"])
    if labels.ndim != 2 or labels.shape != (height, width):
        raise TargetError("target labels must match the native shape")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TargetError("target labels must be an integer array")
    ink = np.asarray(target["ink"])
    if ink.ndim != 2 or ink.shape != (height, width) or ink.dtype != np.bool_:
        raise TargetError("target ink must be a bool HW array")
    domain = np.asarray(target["loss_domain"])
    if domain.ndim != 2 or domain.shape != (height, width) or domain.dtype != np.bool_:
        raise TargetError("target loss_domain must be a bool HW array")
    unknown_geometry = np.asarray(target["unknown_geometry"])
    if (
        unknown_geometry.ndim != 2
        or unknown_geometry.shape != (height, width)
        or unknown_geometry.dtype != np.bool_
    ):
        raise TargetError("target unknown_geometry must be a bool HW array")
    blank_negative = np.asarray(target["blank_negative"])
    if (
        blank_negative.ndim != 2
        or blank_negative.shape != (height, width)
        or blank_negative.dtype != np.bool_
    ):
        raise TargetError("target blank_negative must be a bool HW array")
    boxes = np.asarray(target["boxes"])
    if boxes.ndim != 2 or boxes.shape != (count, 4):
        raise TargetError(
            "target boxes must be Nx4; got %r for %d targets" % (boxes.shape, count)
        )
    keys = target["keys"]
    if isinstance(keys, (str, bytes)) or not hasattr(keys, "__len__"):
        raise TargetError("target keys must be a sequence")
    if len(keys) != count:
        raise TargetError("target keys length %d != mask count %d" % (len(keys), count))
    return {
        "height": height,
        "width": width,
        "masks": masks,
        "labels": labels,
        "boxes": boxes.astype(np.float32, copy=True),
        "ink": ink,
        "loss_domain": domain,
        "unknown_geometry": unknown_geometry,
        "blank_negative": blank_negative,
        "keys": tuple(tuple(int(part) for part in key) for key in keys),
    }


def crop_targets(
    image: Any,
    target: Mapping[str, Any],
    origin: Tuple[int, int] = (0, 0),
    size: int = 384,
) -> Dict[str, Any]:
    """Crop a native window out of an image/target pair without rescaling.

    The window is ``[x, x + size) x [y, y + size)`` in native pixels.  Parts
    outside the native image are padded: no ink, no mask, no label, no loss
    domain, no unknown geometry, no blank negative, and white image pixels
    (1.0).  Boxes are translated and clipped to the window; targets with no
    visible pixel get a zero box and ``visibility=False``.  Padded pixels stay
    out of the loss domain because they do not exist in the source image.
    Zero targets (empty document, only ``semantic 35`` or Thing ``instance < 0``
    primitives) crop to an empty target axis, not an error.
    """

    values = _as_sequence(origin, "origin")
    if len(values) != 2:
        raise TargetError("origin must be an (x, y) pair")
    origin_x = _as_int(values[0], "origin[0]")
    origin_y = _as_int(values[1], "origin[1]")
    window_size = _as_int(size, "size")
    if window_size <= 0:
        raise TargetError("size must be positive")

    data = _validate_target(target)
    height = data["height"]
    width = data["width"]

    cropped_image = None
    layout = "chw"
    if image is not None:
        channels, layout, image_height, image_width = _image_array(image)
        if (image_height, image_width) != (height, width):
            raise TargetError(
                "image spatial shape %r disagrees with target native_size %r"
                % ((image_height, image_width), (height, width))
            )
        cropped_image = np.ones((3, window_size, window_size), dtype=np.float32)

    count = int(data["masks"].shape[0])
    masks = np.zeros((count, window_size, window_size), dtype=bool)
    boxes = np.zeros((count, 4), dtype=np.float32)
    labels = np.zeros((window_size, window_size), dtype=data["labels"].dtype)
    ink = np.zeros((window_size, window_size), dtype=bool)
    domain = np.zeros((window_size, window_size), dtype=bool)
    unknown_geometry = np.zeros((window_size, window_size), dtype=bool)
    blank_negative = np.zeros((window_size, window_size), dtype=bool)

    source_x0 = max(0, origin_x)
    source_y0 = max(0, origin_y)
    source_x1 = min(width, origin_x + window_size)
    source_y1 = min(height, origin_y + window_size)
    if source_x1 > source_x0 and source_y1 > source_y0:
        dest_x = source_x0 - origin_x
        dest_y = source_y0 - origin_y
        span_x = source_x1 - source_x0
        span_y = source_y1 - source_y0
        masks[:, dest_y : dest_y + span_y, dest_x : dest_x + span_x] = data["masks"][
            :, source_y0:source_y1, source_x0:source_x1
        ]
        labels[dest_y : dest_y + span_y, dest_x : dest_x + span_x] = data["labels"][
            source_y0:source_y1, source_x0:source_x1
        ]
        ink[dest_y : dest_y + span_y, dest_x : dest_x + span_x] = data["ink"][
            source_y0:source_y1, source_x0:source_x1
        ]
        domain[dest_y : dest_y + span_y, dest_x : dest_x + span_x] = data[
            "loss_domain"
        ][source_y0:source_y1, source_x0:source_x1]
        unknown_geometry[
            dest_y : dest_y + span_y, dest_x : dest_x + span_x
        ] = data["unknown_geometry"][source_y0:source_y1, source_x0:source_x1]
        blank_negative[
            dest_y : dest_y + span_y, dest_x : dest_x + span_x
        ] = data["blank_negative"][source_y0:source_y1, source_x0:source_x1]
        if cropped_image is not None:
            cropped_image[
                :, dest_y : dest_y + span_y, dest_x : dest_x + span_x
            ] = channels[:, source_y0:source_y1, source_x0:source_x1]

    # ``reshape(count, -1)`` rejects the zero-target case, so reduce the two
    # spatial axes explicitly; this is empty (shape (0,)) for count == 0.
    visibility = masks.any(axis=(1, 2))
    if count:
        shifted = np.stack(
            (
                data["boxes"][:, 0] - float(origin_x),
                data["boxes"][:, 1] - float(origin_y),
                data["boxes"][:, 2] - float(origin_x),
                data["boxes"][:, 3] - float(origin_y),
            ),
            axis=1,
        )
        shifted = np.clip(shifted, 0.0, float(window_size))
        boxes[visibility] = shifted[visibility].astype(np.float32)

    if cropped_image is not None and layout == "hwc":
        cropped_image = np.transpose(cropped_image, (1, 2, 0))

    return {
        "image": cropped_image,
        "masks": masks,
        "labels": labels,
        "boxes": boxes,
        "keys": data["keys"],
        "ink": ink,
        "loss_domain": domain,
        "unknown_geometry": unknown_geometry,
        "blank_negative": blank_negative,
        "visibility": visibility.astype(bool),
        "visible_count": int(np.count_nonzero(visibility)),
        "target_count": count,
        "origin": (int(origin_x), int(origin_y)),
        "size": int(window_size),
        "window": (int(source_x0), int(source_y0), int(source_x1), int(source_y1)),
        "native_size": {"height": int(height), "width": int(width)},
        "raster_size": {"width": int(window_size), "height": int(window_size)},
        "image_layout": layout if cropped_image is not None else None,
    }
