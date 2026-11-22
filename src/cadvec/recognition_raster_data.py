"""Raster data and PQ scorer foundation for the recognition image route.

This module deliberately mirrors the temporary exact scorer used for local
diagnostics (``json_pq_eval_exact.py`` in the OpenCode temp directory) rather
than the dense 128-point curve raster used by the candidate pipeline:

* logical canvas ``(100, 100)``, raster ``1000 x 1000`` (configurable for tests);
* four equal-arclength geometry samples per primitive;
* line/arc masks drawn with ``ImageDraw.line(..., width=1)``;
* circle/ellipse masks drawn from the sample bounding box with
  ``ImageDraw.ellipse(..., width=1)``;
* strict same-class IoU ``> 0.5`` plus Hungarian assignment, dataset micro
  totals.

The compatibility is *internal*: the official scorer implementation has not
been published as a Python module, so parity with the organiser's scorer is
unconfirmed.  Do not treat the output as an official metric.

Label policy (fixed here, independent of any painter order):

* raw ``semantic-id`` 1..35 maps to class ``raw - 1``; class ``35`` is ignored.
* missing/invalid semantic labels are ``-1`` UNKNOWN and never negative.
* unlabelled or background ink stays UNKNOWN.
* Thing classes ``0..29`` with a negative/unknown instance are not emitted as
  instance segments, but their known semantic raster pixels are retained.
* Stuff classes ``30..34`` are merged per class with instance ``-1``.
* missing semantic does not erase an independently known instance label.
* conflicting semantic or instance labels at a pixel become UNKNOWN for that
  task only; no primitive wins by paint order.
* segments remain independent masks and may overlap.

Dense training targets (``build_sample``) use the parser's curve-flattened SVG
geometry rather than the four-point document arguments, so curved paths are not
reduced to chords.  Each primitive target is also intersected with a shared
image-ink support mask derived only from the alpha-composited PNG RGB: a pixel
with no visible image ink is UNKNOWN for both semantic and instance tasks.  The
official 12-field document converter and scorer above remain four-sample exact
reproductions and are not changed by this dense-target path.

Only the PNG becomes model input.  SVG stroke colour, width and layer styling
are never exposed as image features.

Numpy and Pillow are optional imports: importing the module without them
succeeds, and only raster functions raise :class:`RasterDataDependencyError`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # optional at import time so contract-only callers still work
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore

try:  # optional at import time
    from PIL import Image, ImageDraw  # type: ignore
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore

from . import geometry, svg_parse
from .recognition_baseline import RoleReader, _rows_for_groups, _train_dir
from .recognition_contract import (
    ROLE_FIT,
    ROLE_TUNE,
    STUFF_MAX,
    STUFF_MIN,
    THING_MAX,
    validate_manifest,
)

CANVAS_SIZE: Tuple[float, float] = (100.0, 100.0)
RASTER_SIZE = 1000
IGNORE_CLASS = 35

# A pixel shows image ink when at least one 8-bit RGB channel is below this
# value after compositing the PNG over white.  The fit PNGs use an exactly
# white transparent background, so blank pixels stay at 255; 254 keeps faint
# antialiased edges of thin coloured strokes while rejecting blank support.
INK_SUPPORT_THRESHOLD = 254
INK_SUPPORT_POLICY = {
    "source": "alpha-composited PNG RGB only",
    "channel_threshold_8bit": INK_SUPPORT_THRESHOLD,
    "rule": "min(R, G, B) < threshold",
    "supports_both_tasks": True,
    "background": "UNKNOWN",
}

DOCUMENT_FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
    "obj_cts",
    "boxes",
    "rgb",
    "layerIds",
    "widths",
)

SCORER_POLICY = {
    "name": "internal-exact-reproduction",
    "source": "json_pq_eval_exact.py (OpenCode temp diagnostics)",
    "canvas": list(CANVAS_SIZE),
    "raster": RASTER_SIZE,
    "iou_threshold": 0.5,
    "assignment": "hungarian",
    "aggregation": "dataset-micro",
    "official_compatibility": "unconfirmed",
}

__all__ = [
    "RasterDataError",
    "RasterDataDependencyError",
    "CANVAS_SIZE",
    "RASTER_SIZE",
    "INK_SUPPORT_THRESHOLD",
    "INK_SUPPORT_POLICY",
    "SCORER_POLICY",
    "load_role_samples",
    "build_sample",
    "document_from_parsed",
    "document_from_svg",
    "rasterize_document",
    "score_documents",
]


class RasterDataError(ValueError):
    """Invalid input to the raster data or scoring API."""


class RasterDataDependencyError(RuntimeError):
    """Raised when numpy or Pillow is required but unavailable."""


# ---------------------------------------------------------------------------
# dependency guards


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover - environment dependent
        raise RasterDataDependencyError(
            "numpy is required for raster data; install the 'image' extra"
        )
    return np


def _require_pillow() -> Tuple[Any, Any]:
    if Image is None or ImageDraw is None:  # pragma: no cover
        raise RasterDataDependencyError(
            "Pillow is required for raster data; install Pillow"
        )
    return Image, ImageDraw


# ---------------------------------------------------------------------------
# role-filtered sample listing


def _safe_basename(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RasterDataError(f"{field} missing or not a string")
    if value in (".", "..") or os.path.basename(value) != value:
        raise RasterDataError(f"{field} must be a bare file name: {value!r}")
    return value


def _contained_path(root: str, name: str) -> str:
    resolved_root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(resolved_root, name))
    if os.path.normcase(os.path.dirname(candidate)) != os.path.normcase(resolved_root):
        raise RasterDataError(f"asset escapes train_final: {name!r}")
    return candidate


def load_role_samples(
    manifest_path: str,
    role: str,
    scale: int,
) -> List[Dict[str, Any]]:
    """Return metadata records for a frozen fit/tune prefix.

    ``role`` is validated before the manifest is opened, and group filtering
    happens before any asset path is built or read.  Records contain
    ``sample_id``, ``png_path``, ``svg_path`` and ``group_id`` plus a small
    amount of provenance.  No PNG or SVG content is opened here.
    """

    if role not in (ROLE_FIT, ROLE_TUNE):
        raise RasterDataError(
            f"role must be {ROLE_FIT!r} or {ROLE_TUNE!r}; got {role!r}"
        )
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise RasterDataError(f"scale must be an int; got {scale!r}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RasterDataError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise RasterDataError("manifest must be a JSON object")

    validate_manifest(manifest)
    reader = RoleReader(manifest, role)
    groups = reader.group_ids(scale)
    rows = _rows_for_groups(manifest, groups, role=role)

    root = _train_dir(manifest)
    samples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        png_name = _safe_basename(row.get("png_name"), "png_name")
        svg_name = _safe_basename(row.get("svg_name"), "svg_name")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RasterDataError("sample row has no sample_id")
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RasterDataError(f"{sample_id}: group_id missing")
        samples.append(
            {
                "sample_id": sample_id,
                "png_path": _contained_path(root, png_name),
                "svg_path": _contained_path(root, svg_name),
                "group_id": group_id,
                "role": role,
                "scale": scale,
                "index": index,
                "split": row.get("split"),
                "png_name": png_name,
                "svg_name": svg_name,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# official-style JSON document conversion


def _first_command(commands: Any) -> int:
    for entry in commands or ():
        if not isinstance(entry, Mapping):
            continue
        command = str(entry.get("command") or "")
        if command in ("M", "Z", "m", "z"):
            continue
        if command in ("L", "H", "V", "l", "h", "v"):
            return 0
        return 1
    return 0


def _semantic_class(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return IGNORE_CLASS
    if 1 <= raw <= IGNORE_CLASS:
        return raw - 1
    return IGNORE_CLASS


def _instance_id(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return -1
    return raw


def _primitive_labels(primitive: Mapping[str, Any]) -> Tuple[int, int]:
    """Return the official ``(semantic, instance)`` mapping for a primitive.

    This is shared by the four-sample document converter and the dense target
    builder so both paths keep identical key/label conversion.
    """

    label = primitive.get("label") or {}
    semantic_raw = (label.get("semantic") or {}).get("value")
    instance_raw = (label.get("instance") or {}).get("value")
    semantic = _semantic_class(semantic_raw)
    instance = _instance_id(instance_raw)
    if STUFF_MIN - 1 <= semantic <= STUFF_MAX - 1:
        instance = -1
    return semantic, instance


def _rgb(attributes: Mapping[str, Any]) -> List[int]:
    text = attributes.get("stroke")
    if not isinstance(text, str):
        return [0, 0, 0]
    text = text.strip()
    if not (text.startswith("rgb(") and text.endswith(")")):
        return [0, 0, 0]
    parts = [part.strip() for part in text[4:-1].split(",")]
    if len(parts) != 3:
        return [0, 0, 0]
    values: List[int] = []
    try:
        for part in parts:
            values.append(max(0, min(255, int(round(float(part))))))
    except (TypeError, ValueError):
        return [0, 0, 0]
    return values


def _stroke_width(attributes: Mapping[str, Any]) -> float:
    try:
        return float(attributes.get("stroke-width") or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _points(samples: Any) -> Optional[List[List[float]]]:
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return None
    if len(samples) != 4:
        return None
    points: List[List[float]] = []
    for sample in samples:
        if not isinstance(sample, Sequence) or len(sample) != 2:
            return None
        try:
            x = float(sample[0])
            y = float(sample[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        points.append([x, y])
    return points


def document_from_parsed(parsed: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a ``svg_parse.parse_svg`` result into the official 12-field JSON.

    Geometry samples come from the existing geometry utilities (arc-length
    sampling for paths, four cardinal points for circles/ellipses).  Primitives
    without exactly four finite samples are skipped exactly like the temporary
    scorer; ``document_from_svg`` reports how many were skipped.
    """

    if not isinstance(parsed, Mapping):
        raise RasterDataError("parsed SVG must be a mapping")
    view_box = parsed.get("view_box") or {}
    try:
        width = int(view_box["width"])
        height = int(view_box["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RasterDataError("parsed SVG needs a numeric view_box width/height") from exc
    if width <= 0 or height <= 0:
        raise RasterDataError("view_box width/height must be positive")

    commands: List[int] = []
    args: List[List[float]] = []
    lengths: List[float] = []
    semantic_ids: List[int] = []
    instance_ids: List[int] = []
    layer_ids: List[int] = []
    widths: List[float] = []
    rgb_values: List[List[int]] = []
    objects: "Dict[Tuple[int, int], List[List[float]]]" = {}

    for primitive in parsed.get("primitives") or ():
        geometry = primitive.get("geometry") or {}
        points = _points(geometry.get("samples"))
        if points is None:
            continue
        element = primitive.get("element")
        if element == "circle":
            command = 2
        elif element == "ellipse":
            command = 3
        else:
            command = _first_command(geometry.get("commands"))

        semantic, instance = _primitive_labels(primitive)

        layer = primitive.get("layer_index")
        try:
            layer_id = int(layer or 1)
        except (TypeError, ValueError):
            layer_id = 1
        attributes = primitive.get("attributes") or {}

        commands.append(command)
        flat: List[float] = []
        for x, y in points:
            flat.extend((x, y))
        args.append(flat)
        lengths.append(float(geometry.get("length") or 0.0))
        semantic_ids.append(semantic)
        instance_ids.append(instance)
        layer_ids.append(layer_id)
        widths.append(_stroke_width(attributes))
        rgb_values.append(_rgb(attributes))

        if (
            semantic is not IGNORE_CLASS
            and semantic <= THING_MAX - 1
            and instance >= 0
        ):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            objects.setdefault((semantic, instance), []).append(
                [x0, y0, x1, y1, float(semantic), float(instance)]
            )

    boxes: List[List[float]] = []
    obj_cts: List[List[float]] = []
    for (_, _), box_rows in objects.items():
        # A Thing instance may span several primitives; aggregate its extent.
        x0 = min(row[0] for row in box_rows)
        y0 = min(row[1] for row in box_rows)
        x1 = max(row[2] for row in box_rows)
        y1 = max(row[3] for row in box_rows)
        semantic = int(box_rows[0][4])
        instance = int(box_rows[0][5])
        boxes.append([x0, y0, x1, y1, semantic])
        obj_cts.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0, instance])

    document = {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "width": width,
        "height": height,
        "obj_cts": obj_cts,
        "boxes": boxes,
        "rgb": rgb_values,
        "layerIds": layer_ids,
        "widths": widths,
    }
    # Keep exactly the twelve official fields, in a stable order.
    return {field: document[field] for field in DOCUMENT_FIELDS}


def document_from_svg(
    svg_path: str,
    *,
    curve_steps: int = 128,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse *svg_path* with the repository parser and convert it.

    ``curve_steps=128`` matches the temporary exact scorer.  The parser's own
    split policy still applies, so protected splits stay blocked unless the
    caller has explicit authorisation.
    """

    parsed = svg_parse.parse_svg(
        svg_path,
        split=split,
        curve_steps=curve_steps,
        record_commands=True,
    )
    return document_from_parsed(parsed)


def _skipped_geometry(parsed: Mapping[str, Any]) -> int:
    count = 0
    for primitive in parsed.get("primitives") or ():
        geometry = primitive.get("geometry") or {}
        if _points(geometry.get("samples")) is None:
            count += 1
    return count


def _finite_points(samples: Any) -> Optional[List[List[float]]]:
    """Return all finite points in a sample sequence (at least two required)."""

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return None
    points: List[List[float]] = []
    for sample in samples:
        if not isinstance(sample, Sequence) or len(sample) != 2:
            return None
        try:
            x = float(sample[0])
            y = float(sample[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        points.append([x, y])
    return points if len(points) >= 2 else None


def _dense_primitive_records(parsed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build dense drawing records for training targets.

    ``parse_svg`` exposes four arc-length samples only.  For paths we therefore
    re-run the repository's own path parser at ``curve_steps=128`` and draw its
    curve-flattened subpaths.  If a caller supplies an already dense
    ``geometry.samples`` sequence, it is used directly.  Circle/ellipse records
    keep the official circle/ellipse command and bounding-box drawing, which is
    exact for those primitives and independent of sample density.
    """

    records: List[Dict[str, Any]] = []
    for primitive in parsed.get("primitives") or ():
        if not isinstance(primitive, Mapping):
            continue
        geometry_info = primitive.get("geometry") or {}
        samples = _finite_points(geometry_info.get("samples"))
        element = primitive.get("element")
        point_sets: Optional[List[List[List[float]]]] = None
        command = 0
        if element == "circle":
            command = 2
            if samples:
                point_sets = [samples]
        elif element == "ellipse":
            command = 3
            if samples:
                point_sets = [samples]
        else:
            # Prefer explicitly dense samples when a caller provides them.
            if samples is not None and len(samples) > 4:
                point_sets = [samples]
            else:
                path_data = geometry_info.get("d")
                if isinstance(path_data, str) and path_data.strip():
                    try:
                        dense = geometry.parse_path_data(path_data, curve_steps=128)
                    except (geometry.PathDataError, ValueError):
                        dense = None
                    if dense:
                        subpaths = [
                            subpath
                            for subpath in dense.get("subpaths") or ()
                            if len(subpath) >= 2
                        ]
                        if subpaths:
                            point_sets = [
                                [[float(x), float(y)] for x, y in subpath]
                                for subpath in subpaths
                            ]
                if point_sets is None and samples is not None:
                    point_sets = [samples]
        if not point_sets:
            continue
        semantic, instance = _primitive_labels(primitive)
        records.append(
            {
                "command": command,
                "point_sets": point_sets,
                "semantic": semantic,
                "instance": instance,
            }
        )
    return records


# ---------------------------------------------------------------------------
# segment rasterisation


def _raster_axes(
    raster_size: Any,
    canvas_size: Tuple[float, float],
) -> Tuple[int, int]:
    if isinstance(raster_size, (tuple, list)):
        if len(raster_size) != 2:
            raise RasterDataError("raster_size sequence must be (width, height)")
        width, height = raster_size
    else:
        width = height = raster_size
    try:
        raster_w = int(width)
        raster_h = int(height)
    except (TypeError, ValueError) as exc:
        raise RasterDataError(f"invalid raster_size {raster_size!r}") from exc
    if raster_w <= 0 or raster_h <= 0:
        raise RasterDataError("raster_size must be positive")
    canvas_w, canvas_h = float(canvas_size[0]), float(canvas_size[1])
    if canvas_w <= 0 or canvas_h <= 0:
        raise RasterDataError("canvas_size must be positive")
    return raster_w, raster_h


def _document_primitives(document: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    commands = document.get("commands") or []
    args = document.get("args") or []
    semantic_ids = document.get("semanticIds") or []
    instance_ids = document.get("instanceIds") or []
    count = len(commands)
    for key, values in (
        ("args", args),
        ("semanticIds", semantic_ids),
        ("instanceIds", instance_ids),
    ):
        if len(values) != count:
            raise RasterDataError(f"document field {key} has inconsistent length")
    for index, command in enumerate(commands):
        row = args[index]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 8:
            raise RasterDataError(f"document args[{index}] must hold four points")
        points = [
            [float(row[offset]), float(row[offset + 1])]
            for offset in (0, 2, 4, 6)
        ]
        yield {
            "command": int(command),
            "points": points,
            "semantic": int(semantic_ids[index]),
            "instance": int(instance_ids[index]),
        }


def _draw_mask_sets(
    command: int,
    point_sets: Sequence[Sequence[Sequence[float]]],
    raster_w: int,
    raster_h: int,
    canvas_size: Tuple[float, float],
) -> Any:
    """Rasterise one primitive's point sets at the requested raster shape.

    Each subpath is drawn separately so a pen-up move is never turned into a
    connection.  Coordinates are scaled by the actual raster shape, and Pillow
    clips naturally at its bounds; no transform or label geometry is warped for
    999- versus 1000-pixel source images.
    """

    numpy = _require_numpy()
    Image_, ImageDraw_ = _require_pillow()
    sx = raster_w / float(canvas_size[0])
    sy = raster_h / float(canvas_size[1])
    image = Image_.new("L", (raster_w, raster_h), 0)
    draw = ImageDraw_.Draw(image)
    if command in (0, 1):
        for points in point_sets:
            scaled = [(float(x) * sx, float(y) * sy) for x, y in points]
            if len(scaled) >= 2:
                draw.line(scaled, fill=1, width=1)
    elif command in (2, 3):
        scaled = [
            (float(x) * sx, float(y) * sy)
            for points in point_sets
            for x, y in points
        ]
        if not scaled:
            return numpy.asarray(image) > 0
        xs = [point[0] for point in scaled]
        ys = [point[1] for point in scaled]
        draw.ellipse([min(xs), min(ys), max(xs), max(ys)], outline=1, width=1)
    else:
        raise RasterDataError(f"unsupported command {command}")
    return numpy.asarray(image) > 0


def _draw_mask(
    command: int,
    points: Sequence[Sequence[float]],
    raster_w: int,
    raster_h: int,
    canvas_size: Tuple[float, float],
) -> Any:
    """Official document-compatible single-polyline mask wrapper."""

    return _draw_mask_sets(command, [points], raster_w, raster_h, canvas_size)


def _segment_key(semantic: int, instance: int) -> Optional[Tuple[int, int]]:
    if semantic == IGNORE_CLASS:
        return None
    if 0 <= semantic <= THING_MAX - 1:
        if instance < 0:
            return None
        return (semantic, instance)
    if STUFF_MIN - 1 <= semantic <= STUFF_MAX - 1:
        return (semantic, -1)
    return None


def _segment_masks(
    document: Mapping[str, Any],
    raster_w: int,
    raster_h: int,
    canvas_size: Tuple[float, float],
) -> List[Dict[str, Any]]:
    numpy = _require_numpy()
    grouped: "Dict[Tuple[int, int], Any]" = {}
    order: List[Tuple[int, int]] = []
    for primitive in _document_primitives(document):
        key = _segment_key(primitive["semantic"], primitive["instance"])
        if key is None:
            continue
        mask = _draw_mask(
            primitive["command"],
            primitive["points"],
            raster_w,
            raster_h,
            canvas_size,
        )
        if key not in grouped:
            grouped[key] = mask
            order.append(key)
        else:
            grouped[key] = numpy.logical_or(grouped[key], mask)
    return [
        {"class_id": key[0], "instance_id": key[1], "mask": grouped[key]}
        for key in order
    ]


def rasterize_document(
    document: Mapping[str, Any],
    raster_size: Any = RASTER_SIZE,
    canvas_size: Tuple[float, float] = CANVAS_SIZE,
) -> List[Dict[str, Any]]:
    """Rasterise a 12-field document into independent class/instance segments.

    Thing classes require a known non-negative instance; Stuff classes merge per
    class; class 35 is ignored.  Masks may overlap and are never composited into
    a winner-takes-all label image.
    """

    if not isinstance(document, Mapping):
        raise RasterDataError("document must be a mapping")
    raster_w, raster_h = _raster_axes(raster_size, canvas_size)
    return _segment_masks(document, raster_w, raster_h, canvas_size)


# ---------------------------------------------------------------------------
# dense sample construction


def _load_image(png_path: str) -> Any:
    numpy = _require_numpy()
    Image_, _ = _require_pillow()
    try:
        with Image_.open(png_path) as image:
            image.load()
            rgba = image.convert("RGBA")
            canvas = Image_.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image_.alpha_composite(canvas, rgba).convert("RGB")
            array = numpy.asarray(rgb, dtype=numpy.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise RasterDataError(f"cannot read PNG {png_path!r}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise RasterDataError(f"unexpected PNG shape {array.shape!r}")
    return numpy.transpose(array, (2, 0, 1)).copy()


def _image_ink_support(image: Any) -> Any:
    """Shared visible-ink support from the composited RGB image only.

    No SVG colour, stroke width or layer information participates.  A pixel is
    supported only when its RGB is not blank (see ``INK_SUPPORT_POLICY``).
    """

    numpy = _require_numpy()
    threshold = INK_SUPPORT_THRESHOLD / 255.0
    return numpy.min(image, axis=0) < threshold


def _dense_label_entries(
    records: Sequence[Mapping[str, Any]],
    support: Any,
    raster_w: int,
    raster_h: int,
) -> Tuple[List[Tuple[int, Any]], List[Tuple[Tuple[int, int], int, Any]]]:
    """Dense semantic/instance entries, already restricted to image ink."""

    numpy = _require_numpy()
    semantic_entries: List[Tuple[int, Any]] = []
    instance_entries: List[Tuple[Tuple[int, int], int, Any]] = []
    for record in records:
        mask = _draw_mask_sets(
            record["command"],
            record["point_sets"],
            raster_w,
            raster_h,
            CANVAS_SIZE,
        )
        mask = numpy.logical_and(mask, support)
        semantic = record["semantic"]
        instance = record["instance"]
        if 0 <= semantic <= THING_MAX - 1:
            semantic_entries.append((semantic, mask))
            if instance >= 0:
                instance_entries.append(((semantic, instance), instance, mask))
        elif STUFF_MIN - 1 <= semantic <= STUFF_MAX - 1:
            semantic_entries.append((semantic, mask))
        elif instance >= 0:
            # Missing/invalid semantic does not erase an independently known
            # instance label; correctness still requires visible image ink.
            instance_entries.append(((semantic, instance), instance, mask))
    return semantic_entries, instance_entries


def _dense_segment_masks(
    records: Sequence[Mapping[str, Any]],
    support: Any,
    raster_w: int,
    raster_h: int,
) -> List[Dict[str, Any]]:
    """Dense class/instance segment masks, restricted to visible image ink."""

    numpy = _require_numpy()
    grouped: "Dict[Tuple[int, int], Any]" = {}
    order: List[Tuple[int, int]] = []
    for record in records:
        key = _segment_key(record["semantic"], record["instance"])
        if key is None:
            continue
        mask = _draw_mask_sets(
            record["command"],
            record["point_sets"],
            raster_w,
            raster_h,
            CANVAS_SIZE,
        )
        mask = numpy.logical_and(mask, support)
        if key not in grouped:
            grouped[key] = mask
            order.append(key)
        else:
            grouped[key] = numpy.logical_or(grouped[key], mask)
    return [
        {"class_id": key[0], "instance_id": key[1], "mask": grouped[key]}
        for key in order
    ]


def _combine_known(entries: Sequence[Tuple[int, Any]], shape: Tuple[int, int]) -> Any:
    """Combine independent masks, resolving conflicts to UNKNOWN (-1)."""

    numpy = _require_numpy()
    values = numpy.full(shape, -1, dtype=numpy.int64)
    conflict = numpy.zeros(shape, dtype=bool)
    for value, mask in entries:
        has_value = values >= 0
        both = has_value & mask
        conflict |= both & (values != value)
        assign = mask & ~conflict
        values[assign] = value
    values[conflict] = -1
    return values


def _primitive_label_entries(
    document: Mapping[str, Any],
    raster_w: int,
    raster_h: int,
) -> Tuple[List[Tuple[int, Any]], List[Tuple[Tuple[int, int], int, Any]]]:
    """Yield semantic entries and candidate instance entries."""

    semantic_entries: List[Tuple[int, Any]] = []
    instance_entries: List[Tuple[Tuple[int, int], int, Any]] = []
    for primitive in _document_primitives(document):
        mask = _draw_mask(
            primitive["command"],
            primitive["points"],
            raster_w,
            raster_h,
            CANVAS_SIZE,
        )
        semantic = primitive["semantic"]
        instance = primitive["instance"]
        if 0 <= semantic <= THING_MAX - 1:
            semantic_entries.append((semantic, mask))
            if instance >= 0:
                instance_entries.append(((semantic, instance), instance, mask))
        elif STUFF_MIN - 1 <= semantic <= STUFF_MAX - 1:
            semantic_entries.append((semantic, mask))
        elif instance >= 0:
            # Missing/invalid semantic does not erase an independently known
            # instance label; the two validity masks stay independent.
            instance_entries.append(((semantic, instance), instance, mask))
        # class 35 (IGNORE_CLASS) never paints a semantic label.
    return semantic_entries, instance_entries


def build_sample(png_path: str, svg_path: str, sample_id: str) -> Dict[str, Any]:
    """Build one training/validation sample from a PNG and its SVG labels.

    The image is returned as float32 CHW RGB in ``[0, 1]`` with transparent
    pixels composited over white.  Semantic targets are ``0..34`` or ``-1``
    UNKNOWN; instance targets are locally numbered from ``1`` (0 is reserved)
    or ``-1`` UNKNOWN.  Validity masks are independent per task.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise RasterDataError("sample_id must be a non-empty string")

    image = _load_image(png_path)
    parsed = svg_parse.parse_svg(
        svg_path,
        split=None,
        curve_steps=128,
        record_commands=True,
    )
    document = document_from_parsed(parsed)
    raster_h, raster_w = int(image.shape[1]), int(image.shape[2])

    # Dense training targets use the parser's curve-flattened geometry.  The
    # official ``document`` above deliberately remains four-sample exact.
    dense_records = _dense_primitive_records(parsed)
    support = _image_ink_support(image)
    semantic_entries, instance_entries = _dense_label_entries(
        dense_records, support, raster_w, raster_h
    )
    semantic = _combine_known(semantic_entries, (raster_h, raster_w))

    local_map: Dict[Tuple[int, int], int] = {}
    local_entries: List[Tuple[int, Any]] = []
    for key, _instance, mask in instance_entries:
        if key not in local_map:
            local_map[key] = len(local_map) + 1
        local_entries.append((local_map[key], mask))
    instance = _combine_known(local_entries, (raster_h, raster_w))

    segments = _dense_segment_masks(dense_records, support, raster_w, raster_h)

    numpy = _require_numpy()
    return {
        "image": image,
        "semantic": semantic,
        "semantic_valid": semantic >= 0,
        "instance": instance,
        "instance_valid": instance >= 0,
        "segments": segments,
        "gt_document": document,
        "metadata": {
            "sample_id": sample_id,
            "png_path": os.path.abspath(png_path),
            "svg_path": os.path.abspath(svg_path),
            "source_width": raster_w,
            "source_height": raster_h,
            "curve_steps": 128,
            "view_box_width": document["width"],
            "view_box_height": document["height"],
            "primitive_count": len(document["commands"]),
            "dense_primitive_count": len(dense_records),
            "skipped_geometry": _skipped_geometry(parsed),
            "segment_count": len(segments),
            "ink_support_threshold": INK_SUPPORT_THRESHOLD,
            "ink_support_policy": dict(INK_SUPPORT_POLICY),
            "ink_support_pixels": int(numpy.count_nonzero(support)),
            "target_geometry": "dense parser path geometry clipped to PNG shape",
            "official_document_geometry": "four arc-length samples (unchanged)",
            "semantic_unknown_pixels": int(numpy.count_nonzero(semantic < 0)),
            "instance_unknown_pixels": int(numpy.count_nonzero(instance < 0)),
            "scorer_compatibility": SCORER_POLICY["official_compatibility"],
            "label_policy": {
                "ignored_class": IGNORE_CLASS,
                "unknown": -1,
                "thing_classes": [0, THING_MAX - 1],
                "stuff_classes": [STUFF_MIN - 1, STUFF_MAX - 1],
                "background_is_negative": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# scoring


def _hungarian(cost: Any) -> List[Tuple[int, int]]:
    """Minimum-cost assignment for an ``n x m`` matrix (n <= m).

    Classic shortest augmenting path implementation, kept dependency-free so the
    scorer does not require scipy.
    """

    numpy = _require_numpy()
    matrix = numpy.asarray(cost, dtype=numpy.float64)
    n, m = matrix.shape
    if n == 0 or m == 0:
        return []
    transposed = False
    if n > m:
        matrix = matrix.T
        n, m = m, n
        transposed = True

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(matrix[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: List[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] == 0:
            continue
        row, col = p[j] - 1, j - 1
        if transposed:
            row, col = col, row
        pairs.append((row, col))
    return pairs


# Exact intersection counts never materialise the ``G x P x H x W`` Cartesian
# product: each boolean block below is reduced to ``int64`` straight away, so
# the working set stays bounded independently of the mask count and image size.
_IOU_TEMP_BUDGET_BYTES = 16 * 1024 * 1024
# ``.sum(dtype=int64)`` promotes its bool input once per block, so the element
# budget reserves 8 bytes per block element (1 byte bool + int64 reduction).
_IOU_REDUCE_ITEM_BYTES = 8
_IOU_BLOCK_ELEMENTS = _IOU_TEMP_BUDGET_BYTES // _IOU_REDUCE_ITEM_BYTES


def _iou_block_plan(
    gt_count: int,
    pred_count: int,
    pixel_count: int,
    element_budget: int = _IOU_BLOCK_ELEMENTS,
) -> List[Tuple[int, int, int, int]]:
    """Tile the ``gt x pred x pixel`` product into bounded temporary blocks.

    Each tuple is ``(gt_start, gt_stop, pixel_start, pixel_stop)``; the union of
    the blocks covers the whole product exactly once, pixel blocks first for a
    fixed gt block.  A block's boolean temp holds
    ``(gt_stop - gt_start) * pred_count * (pixel_stop - pixel_start)`` elements,
    never more than ``element_budget``.  A single element that alone exceeds the
    budget still yields one block (never an empty plan for non-empty input).
    """

    gt_count = int(gt_count)
    pred_count = int(pred_count)
    pixel_count = int(pixel_count)
    if gt_count <= 0 or pred_count <= 0 or pixel_count <= 0:
        return []
    budget = max(1, int(element_budget))
    if pred_count * pixel_count <= budget:
        pixel_block = pixel_count
    else:
        # A single pixel column costs ``pred_count`` bools for one gt row.
        pixel_block = max(1, budget // pred_count)
    rows_per_block = max(1, budget // (pred_count * pixel_block))

    plan: List[Tuple[int, int, int, int]] = []
    for gt_start in range(0, gt_count, rows_per_block):
        gt_stop = min(gt_count, gt_start + rows_per_block)
        for pixel_start in range(0, pixel_count, pixel_block):
            pixel_stop = min(pixel_count, pixel_start + pixel_block)
            plan.append((gt_start, gt_stop, pixel_start, pixel_stop))
    return plan


def _iou_intersection(gt_flat: Any, pred_flat: Any) -> Any:
    """Exact ``int64`` boolean intersection counts for flattened mask stacks.

    ``gt_flat`` is ``(G, pixels)`` and ``pred_flat`` is ``(P, pixels)``.  The
    result is ``(G, P)`` counts; only one bounded ``(rows, P, pixels)`` boolean
    block is alive at a time instead of the full Cartesian product.
    """

    numpy = _require_numpy()
    gt_count, pixel_count = int(gt_flat.shape[0]), int(gt_flat.shape[1])
    pred_count = int(pred_flat.shape[0])
    intersection = numpy.zeros((gt_count, pred_count), dtype=numpy.int64)
    for gt_start, gt_stop, pixel_start, pixel_stop in _iou_block_plan(
        gt_count, pred_count, pixel_count
    ):
        block = numpy.bitwise_and(
            gt_flat[gt_start:gt_stop, None, pixel_start:pixel_stop],
            pred_flat[None, :, pixel_start:pixel_stop],
        )
        intersection[gt_start:gt_stop] += block.sum(axis=2, dtype=numpy.int64)
    return intersection


def _iou_matrix(gt_masks: Sequence[Any], pred_masks: Sequence[Any]) -> Any:
    numpy = _require_numpy()
    gt_stack = numpy.stack([numpy.asarray(mask).astype(bool) for mask in gt_masks])
    pred_stack = numpy.stack(
        [numpy.asarray(mask).astype(bool) for mask in pred_masks]
    )
    if gt_stack.shape[1:] != pred_stack.shape[1:]:
        raise ValueError(
            "gt and pred masks must share one image shape: "
            f"gt={gt_stack.shape[1:]} pred={pred_stack.shape[1:]}"
        )
    gt_flat = gt_stack.reshape(gt_stack.shape[0], -1)
    pred_flat = pred_stack.reshape(pred_stack.shape[0], -1)
    intersection = _iou_intersection(gt_flat, pred_flat)
    gt_area = gt_flat.sum(axis=1, dtype=numpy.int64)[:, None]
    pred_area = pred_flat.sum(axis=1, dtype=numpy.int64)[None, :]
    union = gt_area + pred_area - intersection
    iou = numpy.zeros(intersection.shape, dtype=numpy.float64)
    valid = union > 0
    iou[valid] = intersection[valid].astype(numpy.float64) / union[valid]
    return iou


def _score_image(
    gt_segments: Sequence[Mapping[str, Any]],
    pred_segments: Sequence[Mapping[str, Any]],
) -> Dict[int, Tuple[int, int, int, float]]:
    gt_by_class: Dict[int, List[Any]] = {}
    pred_by_class: Dict[int, List[Any]] = {}
    for segment in gt_segments:
        gt_by_class.setdefault(int(segment["class_id"]), []).append(segment["mask"])
    for segment in pred_segments:
        pred_by_class.setdefault(int(segment["class_id"]), []).append(segment["mask"])

    totals: Dict[int, Tuple[int, int, int, float]] = {}
    for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
        gt_masks = gt_by_class.get(class_id, [])
        pred_masks = pred_by_class.get(class_id, [])
        tp = 0
        sum_iou = 0.0
        if gt_masks and pred_masks:
            iou = _iou_matrix(gt_masks, pred_masks)
            for row, col in _hungarian(-iou):
                value = float(iou[row, col])
                if value > 0.5:
                    tp += 1
                    sum_iou += value
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        totals[class_id] = (tp, fp, fn, sum_iou)
    return totals


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _ratio_block(tp: int, fp: int, fn: int, sum_iou: float) -> Dict[str, Any]:
    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = _safe_ratio(sum_iou, denominator)
    sq = _safe_ratio(sum_iou, tp)
    rq = _safe_ratio(tp, denominator)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "SumIoU": float(sum_iou),
        "PQ": pq,
        "SQ": sq,
        "RQ": rq,
    }


def score_documents(
    gt_documents: Sequence[Mapping[str, Any]],
    pred_documents: Sequence[Mapping[str, Any]],
    *,
    raster_size: Any = RASTER_SIZE,
    canvas_size: Tuple[float, float] = CANVAS_SIZE,
) -> Dict[str, Any]:
    """Score prediction documents against GT using internal-exact PQ rules.

    Returns JSON-safe dataset-micro totals under ``TP``/``FP``/``FN``/
    ``SumIoU``/``PQ``/``SQ``/``RQ`` plus ``per_class`` entries.  Matching is
    strict same-class IoU ``> 0.5`` with Hungarian assignment; segments are
    independent masks and may overlap.
    """

    if len(gt_documents) != len(pred_documents):
        raise RasterDataError(
            f"document count mismatch: gt={len(gt_documents)} "
            f"pred={len(pred_documents)}"
        )
    raster_w, raster_h = _raster_axes(raster_size, canvas_size)

    totals: Dict[int, List[float]] = {}
    for index, (gt_document, pred_document) in enumerate(
        zip(gt_documents, pred_documents)
    ):
        try:
            gt_segments = _segment_masks(gt_document, raster_w, raster_h, canvas_size)
            pred_segments = _segment_masks(
                pred_document, raster_w, raster_h, canvas_size
            )
        except RasterDataError as exc:
            raise RasterDataError(f"document {index}: {exc}") from exc
        for class_id, (tp, fp, fn, sum_iou) in _score_image(
            gt_segments, pred_segments
        ).items():
            row = totals.setdefault(class_id, [0, 0, 0, 0.0])
            row[0] += tp
            row[1] += fp
            row[2] += fn
            row[3] += sum_iou

    per_class: Dict[str, Dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    total_iou = 0.0
    for class_id in sorted(totals):
        tp, fp, fn, sum_iou = totals[class_id]
        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)
        total_iou += float(sum_iou)
        per_class[str(class_id)] = _ratio_block(
            int(tp), int(fp), int(fn), float(sum_iou)
        )

    aggregate = _ratio_block(total_tp, total_fp, total_fn, total_iou)
    policy = dict(SCORER_POLICY)
    policy["raster"] = [raster_w, raster_h]
    policy["canvas"] = [float(canvas_size[0]), float(canvas_size[1])]
    result = {
        **aggregate,
        "per_class": per_class,
        "n_images": len(gt_documents),
        "raster_size": [raster_w, raster_h],
        "canvas_size": [float(canvas_size[0]), float(canvas_size[1])],
        "policy": policy,
    }
    # Round the aggregate and per-class metrics for stable JSON output.
    for block in [result, *per_class.values()]:
        block["SumIoU"] = round(float(block["SumIoU"]), 12)
        for key in ("PQ", "SQ", "RQ"):
            if block[key] is not None:
                block[key] = round(float(block[key]), 12)
    return result
