"""Structured SVG reading for the CAD floor-plan training split.

What is preserved as a raw fact
-------------------------------
* ``viewBox`` string and parsed values, root ``width`` / ``height`` strings
* layer (``<g>``) document order, nesting path, ``id`` / ``inkscape:label``,
  and every raw attribute
* primitive document order (stable ``local_index``) and every raw attribute,
  including the original ``semantic-id`` / ``instance-id`` strings
* raw path data, raw circle / ellipse attributes, optional ``transform`` string

What is derived (and labelled as such)
--------------------------------------
* four arc-length sample points, polyline length, bounding box
* label diagnostics
* optional mapping / placeholder values, only when an explicit config supplies
  them (see :mod:`cadvec.attributes`)

What is never done here
-----------------------
* converting raw ``semantic-id`` into the platform 0-34 index
* treating a missing label as 35 / -1
* reading annotations from ``test_final`` (guarded by :mod:`cadvec.policy`)
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import geometry
from .attributes import DEFAULT_CONFIG, LabelConfig, read_label, validate_labels
from .diagnostics import Diagnostic, summarise
from .policy import ensure_annotation_access, is_ignored_name, split_of_path

GEOMETRY_TAGS = ("path", "circle", "ellipse")
_PATH_TOLERANCE = 1e-6


class SVGParseError(ValueError):
    """The file could not be parsed as XML at all."""


@dataclass
class LayerRecord:
    index: int
    parent: Optional[int]
    path: Tuple[int, ...]
    document_index: int
    attributes: Dict[str, str]
    id: Optional[str] = None
    label: Optional[str] = None
    groupmode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "parent": self.parent,
            "path": list(self.path),
            "document_index": self.document_index,
            "id": self.id,
            "label": self.label,
            "groupmode": self.groupmode,
            "attributes": dict(self.attributes),
        }


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _namespace_map(data: bytes) -> Dict[str, str]:
    """Collect XML namespace prefixes (prefix -> uri) from the document."""

    prefixes: Dict[str, str] = {}
    try:
        for event, payload in ET.iterparse(io.BytesIO(data), events=("start-ns",)):
            prefix, uri = payload
            if prefix:
                prefixes[prefix] = uri
    except ET.ParseError:
        pass
    return prefixes


def _raw_attribute_name(name: str, uri_to_prefix: Dict[str, str]) -> str:
    if name.startswith("{") and "}" in name:
        uri, local = name[1:].split("}", 1)
        prefix = uri_to_prefix.get(uri)
        return f"{prefix}:{local}" if prefix else local
    return name


def _raw_attributes(
    element: ET.Element, uri_to_prefix: Dict[str, str]
) -> Dict[str, str]:
    return {
        _raw_attribute_name(key, uri_to_prefix): value
        for key, value in element.attrib.items()
    }


def _parse_view_box(
    raw: Optional[str],
    width_attr: Optional[str],
    height_attr: Optional[str],
    file: str,
) -> Tuple[Optional[Dict[str, Any]], List[Diagnostic]]:
    diagnostics: List[Diagnostic] = []
    info: Dict[str, Any] = {
        "raw": raw,
        "width_attr": width_attr,
        "height_attr": height_attr,
        "valid": False,
    }
    if raw is None:
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_MISSING",
                message="root <svg> has no viewBox; coordinate units are undefined",
                severity="error",
                file=file,
            )
        )
        return info, diagnostics

    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_INVALID",
                message=f"viewBox {raw!r} does not contain four numbers",
                severity="error",
                file=file,
            )
        )
        return info, diagnostics
    try:
        values = [float(part) for part in parts]
    except ValueError:
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_INVALID",
                message=f"viewBox {raw!r} contains non-numeric parts",
                severity="error",
                file=file,
            )
        )
        return info, diagnostics

    if not all(math.isfinite(value) for value in values):
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_INVALID",
                message=f"viewBox {raw!r} contains a non-finite value (nan/inf); "
                "coordinates cannot be trusted",
                severity="error",
                file=file,
                detail={"values": [str(value) for value in values]},
            )
        )
        return info, diagnostics

    x, y, width, height = values
    info.update({"x": x, "y": y, "width": width, "height": height})
    if width <= 0 or height <= 0:
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_INVALID",
                message=f"viewBox {raw!r} has non-positive width/height",
                severity="error",
                file=file,
                detail={"values": values},
            )
        )
        return info, diagnostics
    if x != 0 or y != 0:
        diagnostics.append(
            Diagnostic(
                code="VIEWBOX_NONZERO_ORIGIN",
                message=f"viewBox origin is ({x:g}, {y:g}); coordinates are not "
                "normalised to (0, 0)",
                severity="warning",
                file=file,
            )
        )
    info["valid"] = True
    return info, diagnostics


def _bbox_inside(bbox: List[float], view_box: Optional[Dict[str, Any]]) -> bool:
    if not view_box or not view_box.get("valid"):
        return True
    x, y = float(view_box["x"]), float(view_box["y"])
    width, height = float(view_box["width"]), float(view_box["height"])
    return (
        bbox[0] >= x - _PATH_TOLERANCE
        and bbox[1] >= y - _PATH_TOLERANCE
        and bbox[2] <= x + width + _PATH_TOLERANCE
        and bbox[3] <= y + height + _PATH_TOLERANCE
    )


def _parse_path_geometry(
    d: str,
    curve_steps: int,
    record_commands: bool,
    diagnostic_sink: List[Diagnostic],
    file: str,
    local_index: int,
    element: str,
) -> Dict[str, Any]:
    geometry_info: Dict[str, Any] = {"type": "path", "d": d}
    try:
        parsed = geometry.parse_path_data(d, curve_steps=curve_steps)
    except geometry.PathDataError as exc:
        diagnostic_sink.append(
            Diagnostic(
                code="PATH_DATA_ERROR",
                message=f"path data could not be parsed: {exc}",
                severity="error",
                file=file,
                element=element,
                local_index=local_index,
            )
        )
        return geometry_info

    subpaths = parsed["subpaths"]
    lengths = geometry.subpath_lengths(subpaths)
    total_length = sum(lengths)  # saturates to inf; never raises OverflowError
    geometry_info["subpath_count"] = len(subpaths)
    geometry_info["command_count"] = len(parsed["commands"])

    if not all(math.isfinite(length) for length in lengths) or not math.isfinite(
        total_length
    ):
        # Finite endpoints can still overflow when their difference or sum is
        # taken.  Keep the raw data, drop every derived value we cannot trust.
        diagnostic_sink.append(
            Diagnostic(
                code="PATH_GEOMETRY_NONFINITE",
                message="path length overflows to a non-finite value; raw d is kept "
                "but length, bbox and samples are withheld",
                severity="error",
                file=file,
                element=element,
                local_index=local_index,
                detail={
                    "subpath_lengths": [str(value) for value in lengths],
                    "total": str(total_length),
                },
            )
        )
        geometry_info.update(
            {
                "subpath_lengths": None,
                "drawable_subpath_count": None,
                "length": None,
                "bbox": None,
                "samples": [],
                "sample_basis": "unavailable: non-finite derived geometry",
            }
        )
        if record_commands:
            geometry_info["commands"] = parsed["commands"]
        return geometry_info

    drawn_points = [
        point
        for subpath, length in zip(subpaths, lengths)
        if length > 0.0
        for point in subpath
    ]
    geometry_info["subpath_lengths"] = lengths
    geometry_info["drawable_subpath_count"] = sum(1 for length in lengths if length > 0.0)
    geometry_info["length"] = total_length
    geometry_info["bbox"] = geometry.bbox_of(drawn_points)
    # Arc-length samples are taken over drawn segments only; the jump between
    # two subpaths is a pen-up move, never a connection.
    geometry_info["samples"] = geometry.sample_subpaths_by_length(subpaths, 4)
    geometry_info["sample_basis"] = "drawn segments only; moveto moves are not connections"
    if not _finite_geometry(geometry_info):
        diagnostic_sink.append(
            Diagnostic(
                code="PATH_GEOMETRY_NONFINITE",
                message="path sampling or bbox is not finite; raw d is kept but "
                "derived values are withheld",
                severity="error",
                file=file,
                element=element,
                local_index=local_index,
            )
        )
        geometry_info.update(
            {
                "subpath_lengths": None,
                "drawable_subpath_count": None,
                "length": None,
                "bbox": None,
                "samples": [],
                "sample_basis": "unavailable: non-finite derived geometry",
            }
        )
    if record_commands:
        geometry_info["commands"] = parsed["commands"]
    return geometry_info


def parse_svg(
    path: str,
    *,
    split: Optional[str] = None,
    allow_test_annotations: bool = False,
    config: LabelConfig = DEFAULT_CONFIG,
    curve_steps: int = 8,
    record_commands: bool = True,
    include_digest: bool = False,
) -> Dict[str, Any]:
    """Parse one SVG file into a traceable dictionary.

    Raises :class:`cadvec.policy.AnnotationAccessError` before any read when
    the split policy forbids annotation access.
    """

    geometry.validate_curve_steps(curve_steps)
    resolved_split = ensure_annotation_access(
        path, split=split, allow_test_annotations=allow_test_annotations
    )
    file_label = os.path.basename(path)

    with open(path, "rb") as handle:
        data = handle.read()

    digest = hashlib.sha256(data).hexdigest() if include_digest else None

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SVGParseError(f"{file_label}: XML parse error: {exc}") from exc

    uri_to_prefix = {uri: prefix for prefix, uri in _namespace_map(data).items()}
    diagnostics: List[Diagnostic] = []

    def diag(code: str, message: str, severity: str = "warning", **kwargs) -> None:
        diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                severity=severity,
                file=file_label,
                split=resolved_split,
                **kwargs,
            )
        )

    root_tag = _local_name(root.tag)
    if root_tag != "svg":
        diag(
            "ROOT_NOT_SVG",
            f"document root is <{root_tag or '?'}>, expected <svg>",
            severity="error",
        )

    root_attributes = _raw_attributes(root, uri_to_prefix)
    view_box, view_diagnostics = _parse_view_box(
        root_attributes.get("viewBox"),
        root_attributes.get("width"),
        root_attributes.get("height"),
        file_label,
    )
    diagnostics.extend(view_diagnostics)

    layers: List[LayerRecord] = []
    primitives: List[Dict[str, Any]] = []
    element_counts: Dict[str, int] = {}
    layer_ids_seen: Dict[str, int] = {}

    state = {
        "document_index": 0,
        "layer_index": 0,
        "local_index": 0,
        "defs_depth": 0,
    }

    def visit(
        element: ET.Element,
        layer_stack: List[LayerRecord],
        inherited_transform: Optional[List[str]],
    ) -> None:
        for child in element:
            tag = _local_name(child.tag)
            if not tag:
                continue  # comments / processing instructions
            state["document_index"] += 1
            document_index = state["document_index"]
            element_counts[tag] = element_counts.get(tag, 0) + 1
            attributes = _raw_attributes(child, uri_to_prefix)

            if tag == "defs":
                state["defs_depth"] += 1
                visit(
                    child,
                    layer_stack,
                    _append_transform(inherited_transform, attributes, "container"),
                )
                state["defs_depth"] -= 1
                continue

            if tag == "g":
                state["layer_index"] += 1
                layer_index = state["layer_index"]
                layer_id = attributes.get("id")
                layer_label = (
                    attributes.get("inkscape:label")
                    or attributes.get("label")
                    or None
                )
                record = LayerRecord(
                    index=layer_index,
                    parent=layer_stack[-1].index if layer_stack else None,
                    path=tuple(item.index for item in layer_stack) + (layer_index,),
                    document_index=document_index,
                    attributes=attributes,
                    id=layer_id,
                    label=layer_label,
                    groupmode=attributes.get("inkscape:groupmode"),
                )
                layers.append(record)
                if layer_id:
                    layer_ids_seen[layer_id] = layer_ids_seen.get(layer_id, 0) + 1
                if layer_label is None:
                    diag(
                        "LAYER_WITHOUT_LABEL",
                        "layer <g> has neither inkscape:label nor label",
                        severity="info",
                        element="g",
                    )
                layer_stack.append(record)
                visit(
                    child,
                    layer_stack,
                    _append_transform(
                        inherited_transform, attributes, f"group:{layer_index}"
                    ),
                )
                layer_stack.pop()
                continue

            if tag in GEOMETRY_TAGS:
                if state["defs_depth"] > 0:
                    diag(
                        "GEOMETRY_INSIDE_DEFS",
                        f"<{tag}> inside <defs> is not emitted as a primitive",
                        severity="info",
                        element=tag,
                    )
                    continue
                _visit_geometry(
                    child,
                    tag,
                    attributes,
                    document_index,
                    layer_stack,
                    inherited_transform,
                    primitives,
                    diagnostics,
                    config,
                    file_label,
                    resolved_split,
                    view_box,
                    curve_steps,
                    record_commands,
                    state,
                )
                continue

            # Container elements (e.g. <a>, <switch>) are walked; leaves counted only.
            visit(
                child,
                layer_stack,
                _append_transform(inherited_transform, attributes, f"container:{tag}"),
            )

    visit(root, [], _append_transform(None, root_attributes, "root"))

    for layer_id, count in layer_ids_seen.items():
        if count > 1:
            diag(
                "DUPLICATE_LAYER_ID",
                f"layer id {layer_id!r} appears {count} times",
                severity="warning",
                element="g",
            )

    transforms_affecting_geometry = [
        item for item in primitives if item["transform_chain"]
    ]
    if transforms_affecting_geometry:
        sources = sorted(
            {
                entry["source"]
                for item in transforms_affecting_geometry
                for entry in item["transform_chain"]
            }
        )
        diag(
            "TRANSFORM_NOT_APPLIED",
            f"{len(transforms_affecting_geometry)} geometry primitive(s) sit under "
            "root/group/primitive transforms that are recorded but not applied; "
            "coordinates and samples stay in the local pre-transform frame, are "
            "not rendered absolute coordinates, and must not be treated as final "
            "training geometry",
            severity="warning",
            detail={
                "primitives": len(transforms_affecting_geometry),
                "sources": sources,
                "coordinate_frame": "local-pre-transform",
                "transform_applied": False,
            },
        )

    if not primitives:
        diag("NO_GEOMETRY_PRIMITIVES", "no path/circle/ellipse found", severity="warning")

    stats = {
        "elements": state["document_index"],
        "element_counts": dict(sorted(element_counts.items())),
        "layers": len(layers),
        "primitives": len(primitives),
        "paths": sum(1 for item in primitives if item["element"] == "path"),
        "circles": sum(1 for item in primitives if item["element"] == "circle"),
        "ellipses": sum(1 for item in primitives if item["element"] == "ellipse"),
        "labeled_semantic": sum(
            1 for item in primitives if item["label"]["semantic"]["value"] is not None
        ),
        "missing_semantic": sum(
            1 for item in primitives if not item["label"]["semantic"]["present"]
        ),
        "labeled_instance": sum(
            1 for item in primitives if item["label"]["instance"]["value"] is not None
        ),
        "missing_instance": sum(
            1 for item in primitives if not item["label"]["instance"]["present"]
        ),
    }

    return {
        "file": file_label,
        "path": path,
        "split": resolved_split,
        "sha256": digest,
        "root": {
            "tag": root_tag,
            "attributes": root_attributes,
        },
        "view_box": view_box,
        "layers": [layer.to_dict() for layer in layers],
        "primitives": primitives,
        "diagnostics": [item.to_dict() for item in diagnostics],
        "diagnostic_summary": summarise(diagnostics),
        "stats": stats,
        "policy": {
            "annotation_access": "train-svg"
            if resolved_split == "train_final"
            else ("test-svg (explicit opt-in)" if resolved_split == "test_final" else "unspecified"),
            "configuration": config.to_dict(),
        },
    }


def _append_transform(
    inherited: Optional[List[Dict[str, str]]],
    attributes: Dict[str, str],
    source: str,
) -> Optional[List[Dict[str, str]]]:
    """Append this element's ``transform`` to the outer-to-inner chain."""

    transform = attributes.get("transform")
    if not transform:
        return inherited
    return (inherited or []) + [{"source": source, "raw": transform}]


def _visit_geometry(
    element: ET.Element,
    tag: str,
    attributes: Dict[str, str],
    document_index: int,
    layer_stack: List[LayerRecord],
    inherited_transform: Optional[List[str]],
    primitives: List[Dict[str, Any]],
    diagnostics: List[Diagnostic],
    config: LabelConfig,
    file_label: str,
    split: Optional[str],
    view_box: Optional[Dict[str, Any]],
    curve_steps: int,
    record_commands: bool,
    state: Dict[str, int],
) -> None:
    state["local_index"] += 1
    local_index = state["local_index"]
    innermost = layer_stack[-1] if layer_stack else None

    def diag(code: str, message: str, severity: str = "warning") -> None:
        diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                severity=severity,
                file=file_label,
                split=split,
                element=tag,
                local_index=local_index,
            )
        )

    geometry_info: Optional[Dict[str, Any]] = None
    if tag == "path":
        raw_d = attributes.get("d")
        if raw_d is None:
            diag("PATH_WITHOUT_D", "<path> has no d attribute", severity="error")
            geometry_info = {"type": "path", "d": None}
        else:
            geometry_info = _parse_path_geometry(
                raw_d,
                curve_steps,
                record_commands,
                diagnostics,
                file_label,
                local_index,
                tag,
            )
    elif tag == "circle":
        geometry_info = _circle_geometry(attributes, diag)
    elif tag == "ellipse":
        geometry_info = _ellipse_geometry(attributes, diag)

    semantic = read_label(attributes, config.semantic_aliases, "semantic")
    instance = read_label(attributes, config.instance_aliases, "instance")
    label_diagnostics, derived = validate_labels(
        semantic,
        instance,
        config,
        file=file_label,
        split=split,
        element=tag,
        local_index=local_index,
    )
    diagnostics.extend(label_diagnostics)

    label = {
        "semantic": semantic.to_dict(),
        "instance": instance.to_dict(),
    }
    if derived:
        label["derived"] = derived

    transform = attributes.get("transform")
    transform_chain = list(inherited_transform or [])
    if transform:
        transform_chain.append({"source": "primitive", "raw": transform})
    if geometry_info is not None:
        geometry_info["coordinate_frame"] = "local-pre-transform"

    record: Dict[str, Any] = {
        "local_index": local_index,
        "document_index": document_index,
        "element": tag,
        "layer_index": innermost.index if innermost else None,
        "layer_path": list(innermost.path) if innermost else [],
        "layer_id": innermost.id if innermost else None,
        "layer_label": innermost.label if innermost else None,
        "attributes": attributes,
        "label": label,
        "geometry": geometry_info,
        "transform": transform,
        "transform_inherited": [
            entry["raw"] for entry in (inherited_transform or [])
        ],
        "transform_chain": transform_chain,
        "transform_applied": False,
        "transform_resolved": not transform_chain,
        "coordinate_frame": "local-pre-transform",
    }

    bbox = (geometry_info or {}).get("bbox")
    if bbox is not None and not _bbox_inside(bbox, view_box):
        diag(
            "GEOMETRY_OUTSIDE_VIEWBOX",
            f"primitive bbox {bbox} lies outside the viewBox",
            severity="info",
        )

    primitives.append(record)


def _float_attr(
    attributes: Dict[str, str], name: str, default: Optional[float], diag
) -> Optional[float]:
    raw = attributes.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        diag(
            "GEOMETRY_ATTRIBUTE_INVALID",
            f"attribute {name}={raw!r} is not a number",
            severity="error",
        )
        return None
    if not math.isfinite(value):
        diag(
            "GEOMETRY_ATTRIBUTE_INVALID",
            f"attribute {name}={raw!r} is not finite (nan/inf)",
            severity="error",
        )
        return None
    return value


def _finite_geometry(info: Dict[str, Any]) -> bool:
    length = info.get("length")
    if length is not None and not math.isfinite(length):
        return False
    for value in info.get("bbox") or []:
        if not math.isfinite(value):
            return False
    for sample in info.get("samples") or []:
        if any(not math.isfinite(value) for value in sample):
            return False
    return True


def _circle_geometry(attributes: Dict[str, str], diag) -> Dict[str, Any]:
    cx = _float_attr(attributes, "cx", 0.0, diag)
    cy = _float_attr(attributes, "cy", 0.0, diag)
    radius = _float_attr(attributes, "r", None, diag)
    info: Dict[str, Any] = {"type": "circle", "cx": cx, "cy": cy, "r": radius}
    if cx is None or cy is None or radius is None:
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
        return info
    if radius < 0:
        diag("CIRCLE_NEGATIVE_RADIUS", f"r={radius!r} is negative", severity="error")
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
        return info
    info["length"] = 2.0 * math.pi * radius
    info["bbox"] = [cx - radius, cy - radius, cx + radius, cy + radius]
    info["samples"] = geometry.circle_samples(cx, cy, radius, 4)
    if not _finite_geometry(info):
        diag(
            "GEOMETRY_NONFINITE",
            "circle geometry overflows to a non-finite length, bbox or sample",
            severity="error",
        )
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
    return info


def _ellipse_geometry(attributes: Dict[str, str], diag) -> Dict[str, Any]:
    cx = _float_attr(attributes, "cx", 0.0, diag)
    cy = _float_attr(attributes, "cy", 0.0, diag)
    rx = _float_attr(attributes, "rx", None, diag)
    ry = _float_attr(attributes, "ry", None, diag)
    info: Dict[str, Any] = {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}
    if cx is None or cy is None or rx is None or ry is None:
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
        return info
    if rx < 0 or ry < 0:
        diag("ELLIPSE_NEGATIVE_RADIUS", f"rx={rx!r} ry={ry!r} must be non-negative", severity="error")
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
        return info
    info["length"] = geometry.ellipse_perimeter(rx, ry)
    info["length_note"] = "Ramanujan II approximation"
    info["bbox"] = [cx - rx, cy - ry, cx + rx, cy + ry]
    info["samples"] = geometry.ellipse_samples(cx, cy, rx, ry, 4)
    if not _finite_geometry(info):
        diag(
            "GEOMETRY_NONFINITE",
            "ellipse geometry overflows to a non-finite length, bbox or sample",
            severity="error",
        )
        info["length"] = None
        info["bbox"] = None
        info["samples"] = []
    return info


def iter_svg_names(directory: str) -> List[str]:
    """Sorted, filtered SVG file names of a split directory (no reads)."""

    names = [
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".svg") and not is_ignored_name(name)
    ]
    names.sort()
    return names


def parse_split(
    directory: str,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    allow_test_annotations: bool = False,
    config: LabelConfig = DEFAULT_CONFIG,
    curve_steps: int = 8,
    record_commands: bool = False,
    collect_samples: int = 0,
) -> Dict[str, Any]:
    """Parse the selected SVGs of a split and return an aggregate report.

    ``limit`` (applied first) defines the one selected set used for both the
    summary and ``collect_samples``: samples are taken from that set only and
    are never re-read from the directory.  Sample documents keep commands and
    a digest for traceability.
    """

    geometry.validate_curve_steps(curve_steps)
    resolved_split = split or split_of_path(directory)
    ensure_annotation_access(
        directory, split=resolved_split, allow_test_annotations=allow_test_annotations
    )

    names = iter_svg_names(directory)
    if limit is not None:
        names = names[: max(int(limit), 0)]
    sample_limit = min(max(int(collect_samples), 0), len(names))

    diagnostic_counts: Dict[str, int] = {}
    diagnostic_severity: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    diagnostic_examples: List[Dict[str, Any]] = []
    diagnostic_total = 0
    semantic_values: Dict[int, int] = {}
    instance_negatives = 0
    instance_values: Dict[int, int] = {}
    view_boxes: Dict[str, int] = {}
    problem_files: List[Dict[str, Any]] = []
    files_with_error_diagnostics = 0
    sample_records: List[Dict[str, Any]] = []
    files_parsed = 0
    files_failed = 0
    primitives_total = 0
    layers_total = 0
    paths = circles = ellipses = 0
    labeled_semantic = labeled_instance = 0
    missing_semantic = missing_instance = 0
    zero_geometry_files = 0
    empty_d = 0
    transformed_primitives = 0
    length_total = 0.0
    length_count = 0
    length_sum_finite = True

    for name in names:
        path = os.path.join(directory, name)
        want_sample = len(sample_records) < sample_limit
        try:
            document = parse_svg(
                path,
                split=resolved_split,
                allow_test_annotations=allow_test_annotations,
                config=config,
                curve_steps=curve_steps,
                record_commands=record_commands or want_sample,
                include_digest=want_sample,
            )
        except (SVGParseError, OSError, UnicodeDecodeError) as exc:
            files_failed += 1
            diagnostic_total += 1
            diagnostic_counts["SVG_FILE_ERROR"] = diagnostic_counts.get("SVG_FILE_ERROR", 0) + 1
            diagnostic_severity["error"] += 1
            if len(diagnostic_examples) < 100:
                diagnostic_examples.append(
                    {
                        "code": "SVG_FILE_ERROR",
                        "severity": "error",
                        "message": str(exc),
                        "file": name,
                    }
                )
            problem_files.append(
                {"file": name, "kind": "parse_failure", "error": str(exc)}
            )
            continue

        files_parsed += 1
        stats = document["stats"]
        primitives_total += stats["primitives"]
        layers_total += stats["layers"]
        paths += stats["paths"]
        circles += stats["circles"]
        ellipses += stats["ellipses"]
        labeled_semantic += stats["labeled_semantic"]
        labeled_instance += stats["labeled_instance"]
        missing_semantic += stats["missing_semantic"]
        missing_instance += stats["missing_instance"]
        if stats["primitives"] == 0:
            zero_geometry_files += 1

        for record in document["diagnostics"]:
            diagnostic_total += 1
            code = record["code"]
            diagnostic_counts[code] = diagnostic_counts.get(code, 0) + 1
            severity = record.get("severity", "warning")
            diagnostic_severity[severity] = diagnostic_severity.get(severity, 0) + 1
            if len(diagnostic_examples) < 100:
                diagnostic_examples.append(record)

        for primitive in document["primitives"]:
            semantic = primitive["label"]["semantic"]["value"]
            instance = primitive["label"]["instance"]["value"]
            if semantic is not None:
                semantic_values[semantic] = semantic_values.get(semantic, 0) + 1
            if instance is not None:
                instance_values[instance] = instance_values.get(instance, 0) + 1
                if instance < 0:
                    instance_negatives += 1
            geometry_info = primitive["geometry"] or {}
            length = geometry_info.get("length")
            if isinstance(length, (int, float)) and not isinstance(length, bool):
                length_count += 1
                if length_sum_finite and math.isfinite(float(length)):
                    candidate = length_total + float(length)
                    if math.isfinite(candidate):
                        length_total = candidate
                    else:
                        length_sum_finite = False
                else:
                    length_sum_finite = False
            if primitive["element"] == "path" and not geometry_info.get("d"):
                empty_d += 1
            if primitive.get("transform_chain"):
                transformed_primitives += 1

        view_box = document["view_box"]
        key = str(view_box.get("raw")) if view_box else "missing"
        view_boxes[key] = view_boxes.get(key, 0) + 1

        if want_sample:
            sample_records.append(document)

        file_errors = [
            item for item in document["diagnostics"] if item["severity"] == "error"
        ]
        if file_errors:
            files_with_error_diagnostics += 1
            problem_files.append(
                {
                    "file": name,
                    "kind": "error_diagnostics",
                    "error_count": len(file_errors),
                    "codes": sorted({item["code"] for item in file_errors}),
                }
            )

    summary = {
        "total": diagnostic_total,
        "by_severity": diagnostic_severity,
        "by_code": dict(sorted(diagnostic_counts.items())),
        "examples": diagnostic_examples,
    }
    return {
        "split": resolved_split,
        "directory": directory,
        "generated_at": None,
        "file_selection": {
            "svg_files_total": len(iter_svg_names(directory)),
            "svg_files_considered": len(names),
            "limit": limit,
            "selected_files": list(names[:100]),
            "selected_files_truncated": len(names) > 100,
            "samples_requested": max(int(collect_samples), 0),
            "samples_collected": len(sample_records),
            "excluded_names": "._* and .DS_Store",
        },
        "files": {
            "parsed": files_parsed,
            "failed": files_failed,
            "with_error_diagnostics": files_with_error_diagnostics,
            "with_zero_primitives": zero_geometry_files,
        },
        "primitives": {
            "total": primitives_total,
            "paths": paths,
            "circles": circles,
            "ellipses": ellipses,
            "labeled_semantic": labeled_semantic,
            "missing_semantic": missing_semantic,
            "labeled_instance": labeled_instance,
            "missing_instance": missing_instance,
            "instance_negative": instance_negatives,
            "path_without_usable_d": empty_d,
            "with_unapplied_transform": transformed_primitives,
            "mean_length": (length_total / length_count)
            if length_count and length_sum_finite
            else None,
        },
        "layers": {"total": layers_total},
        "raw_semantic_values": dict(sorted(semantic_values.items())),
        "raw_instance_values": dict(sorted(instance_values.items())),
        "view_box_variants": dict(
            sorted(view_boxes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "diagnostics": summary,
        "problem_files": problem_files[:100],
        "samples": sample_records,
        "policy": {
            "annotation_access": "train-svg"
            if resolved_split == "train_final"
            else ("test-svg (explicit opt-in)" if resolved_split == "test_final" else "unspecified"),
            "configuration": config.to_dict(),
        },
        "notes": [
            "raw_semantic_values counts attribute values as written in the SVGs; "
            "no platform index mapping is applied",
            "missing labels are counted, not replaced",
            "arc-length samples cover drawn segments only; a moveto move between "
            "two subpaths is not a connection",
            "transforms are recorded as an outer-to-inner chain but never applied; "
            "coordinates stay local and transforms_applied is always false",
        ],
    }
