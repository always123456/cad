"""Pure data/geometry adapter for the PNG line-diagnostic runner.

This module is deliberately file-free and image-library-free: it only accepts
already-parsed structures (a ``cadvec.svg_parse`` document, a viewBox mapping,
polylines, or a ``cadvec.png_frontend`` result) and returns versioned
strict-JSON dictionaries.  It does not read SVG/PNG/label files, does not call
the predictor, does not write outputs, does not group samples and does not
compute evaluation metrics.

Pipeline position::

    parse_svg document -> gt_from_document -> map_viewbox_to_pixels
                       -> apply_pixel_mapping -> geometric_eval.evaluate_pair
    png_frontend result -> pred_from_frontend -> geometric_eval.evaluate_pair

Contracts and limits
--------------------
* ``gt_from_document`` accepts the *structural* ``parse_svg`` document.  That
  document has no schema/version field, which is recorded explicitly as
  ``parser_schema_present: false``; the adapter does not invent parser schema.
  Every path is rebuilt from its raw ``d`` with
  ``geometry.parse_path_data(d, curve_steps=128)``.  The parser's four samples
  and its 8-step lengths are never used.
* Only paths whose original commands are M/L/H/V/Z with positive drawn length
  enter ``polylines_straight``; any C/S/Q/T/A command keeps the whole path in
  ``curved_or_mixed`` and contributes only to ``polylines_all``.  Circles and
  ellipses are closed 128-segment polylines in ``polylines_all`` only and are
  never renamed as straight strokes.
* Any root/group/primitive transform makes the whole drawing ``excluded``
  before geometry is trusted.  The adapter does not apply matrices.
* ``map_viewbox_to_pixels`` implements ``xMidYMid meet`` (SVG default) and
  ``none`` plus the nine standard Min/Mid/Max meet alignments.  ``slice``,
  unknown values, ambiguous root units, missing/invalid viewBox or unsupported
  viewport modes return ``alignment_pending``.  ``ok`` only means the maths is
  defined; ``verification`` is always ``"unverified"``.
* Pixel mapping never clips or clamps.  The mapping metadata tells the runner
  how to detect out-of-viewport points and mark alignment pending instead of
  silently dropping GT.
* Frontend statuses map explicitly: ``ok`` -> lines; ``empty_input`` /
  ``no_lines_detected`` -> legal empty prediction (reason kept);
  ``processing_limited`` -> ``evaluation_limited``; input/decode/unsupported
  errors -> ``failed``; malformed/unknown schemas or statuses -> ``failed``,
  never an empty prediction.

All returned values use finite floats only and serialise with
``json.dumps(..., allow_nan=False)``.  Invalid arguments raise
:class:`AdapterInputError`; geometry/content problems are reported through
``status``/``reasons`` so the runner can count them explicitly.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import geometric_eval as _eval
from . import geometry as _geometry

SCHEMA = "cadvec.png_diagnostic_adapter/1"
SCHEMA_VERSION = 1
PARSER_CONTRACT = "cadvec.parse_svg.structural/v1"
FRONTEND_SCHEMA = "cadvec.png_frontend"
FRONTEND_SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_EXCLUDED = "excluded"
STATUS_FAILED = "failed"
STATUS_ALIGNMENT_PENDING = "alignment_pending"

PRED_OK = "ok"
PRED_EMPTY = "empty"
PRED_LIMITED = "evaluation_limited"
PRED_FAILED = "failed"

CURVE_STEPS_DEFAULT = 128
CIRCLE_STEPS_DEFAULT = 128

_ALIGNMENTS = (
    "xMinYMin",
    "xMidYMin",
    "xMaxYMin",
    "xMinYMid",
    "xMidYMid",
    "xMaxYMid",
    "xMinYMax",
    "xMidYMax",
    "xMaxYMax",
)
_LENGTH_RE = re.compile(
    r"^([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)([A-Za-z%]*)$"
)


class AdapterInputError(ValueError):
    """Invalid adapter argument or malformed input structure."""


def _number(value: Any, what: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterInputError(f"{what} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise AdapterInputError(f"{what} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise AdapterInputError(f"{what} must be positive, got {value!r}")
    return number


def _step_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterInputError(f"{what} must be a positive integer, got {value!r}")
    return value


def _result(**payload: Any) -> Dict[str, Any]:
    document = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION}
    document.update(payload)
    return document


def _all_finite_points(points: Sequence[Sequence[float]]) -> bool:
    return all(
        len(point) == 2 and math.isfinite(point[0]) and math.isfinite(point[1])
        for point in points
    )


def _close_ring(points: List[List[float]]) -> List[List[float]]:
    if not points:
        return []
    return points + [[points[0][0], points[0][1]]]


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for index in range(1, len(points)):
        total += math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
    return total


def _label_stats(primitive: Mapping[str, Any]) -> Dict[str, Any]:
    label = primitive.get("label") or {}
    semantic = label.get("semantic") or {}
    instance = label.get("instance") or {}
    semantic_present = bool(semantic.get("present"))
    instance_present = bool(instance.get("present"))
    semantic_value = semantic.get("value")
    instance_value = instance.get("value")
    return {
        "raw_semantic_present": semantic_present,
        "raw_semantic_nonempty": semantic_present and semantic_value is not None,
        "raw_instance_present": instance_present,
        "raw_instance_nonempty": instance_present and instance_value is not None,
        "raw_instance_value": instance_value if instance_present else None,
    }


def _transform_sources(
    doc: Mapping[str, Any], primitive: Mapping[str, Any]
) -> List[str]:
    sources: List[str] = []
    root_attributes = (doc.get("root") or {}).get("attributes") or {}
    if root_attributes.get("transform"):
        sources.append("root")
    chain = primitive.get("transform_chain") or []
    for entry in chain:
        if isinstance(entry, Mapping) and entry.get("source"):
            source = str(entry["source"])
            if source not in sources:
                sources.append(source)
    for key in ("transform", "transform_inherited"):
        value = primitive.get(key)
        if value and not chain:
            marker = "primitive" if key == "transform" else "inherited"
            if marker not in sources:
                sources.append(marker)
    return sources


def gt_from_document(
    doc: Mapping[str, Any],
    *,
    curve_steps: int = CURVE_STEPS_DEFAULT,
    circle_steps: int = CIRCLE_STEPS_DEFAULT,
) -> Dict[str, Any]:
    """Build versioned GT polylines from a ``parse_svg`` document.

    Returns ``status`` one of ``ok`` / ``excluded`` / ``failed``.
    ``excluded`` means an unresolved transform made the whole drawing unusable
    as absolute GT; ``failed`` means at least one primitive's geometry could
    not be rebuilt.  Neither is an empty-success pair.  The input document is
    only read, never mutated.
    """

    curve_steps = _step_count(curve_steps, "curve_steps")
    circle_steps = _step_count(circle_steps, "circle_steps")
    if not isinstance(doc, Mapping):
        raise AdapterInputError("gt_from_document expects a mapping document")
    primitives = doc.get("primitives")
    if not isinstance(primitives, list):
        raise AdapterInputError("document must contain a 'primitives' list")
    root = doc.get("root") or {}
    if not isinstance(root, Mapping):
        raise AdapterInputError("document 'root' must be a mapping when present")

    root_transform = bool((root.get("attributes") or {}).get("transform"))
    records: List[Dict[str, Any]] = []
    polylines_all: List[List[List[float]]] = []
    polylines_straight: List[List[List[float]]] = []
    reasons: List[str] = []

    label_counts = {
        "raw_semantic_present": 0,
        "raw_semantic_nonempty": 0,
        "raw_instance_present": 0,
        "raw_instance_nonempty": 0,
    }
    instance_values: set = set()

    # Transform preflight over *all* primitives, before any geometry is
    # parsed.  This keeps exclusion statistics independent of primitive order.
    preflight: List[Dict[str, Any]] = []
    element_totals = {"path": 0, "circle": 0, "ellipse": 0}
    affected = 0
    transform_sources: List[str] = ["root"] if root_transform else []
    for index, primitive in enumerate(primitives):
        if not isinstance(primitive, Mapping):
            raise AdapterInputError(f"primitive {index} must be a mapping")
        element = str(primitive.get("element") or "").lower()
        stats = _label_stats(primitive)
        for key in label_counts:
            if stats[key]:
                label_counts[key] += 1
        if stats["raw_instance_nonempty"]:
            instance_values.add(stats["raw_instance_value"])
        sources = _transform_sources(doc, primitive)
        if element in element_totals:
            element_totals[element] += 1
        if sources:
            affected += 1
            for source in sources:
                if source not in transform_sources:
                    transform_sources.append(source)
        preflight.append(
            {
                "index": index,
                "primitive": primitive,
                "element": element,
                "local_index": primitive.get("local_index"),
                "stats": stats,
                "sources": sources,
            }
        )

    counts = {
        "primitives_total": len(primitives),
        "path_total": element_totals["path"],
        "circle_total": element_totals["circle"],
        "ellipse_total": element_totals["ellipse"],
        "path_positive": 0,
        "straight_only_paths": 0,
        "curved_or_mixed_paths": 0,
        "circle_valid": 0,
        "ellipse_valid": 0,
        "zero_length": 0,
        "invalid_geometry": 0,
        "unsupported_primitives": 0,
        "transformed_primitives": affected,
        "subpaths_total": 0,
        "zero_length_subpaths": 0,
        "polylines_all": 0,
        "polylines_straight": 0,
        "drawings_excluded": 0,
        "drawings_failed": 0,
    }

    if affected or root_transform:
        for item in preflight:
            sources = item["sources"]
            records.append(
                {
                    "primitive_index": item["index"],
                    "local_index": item["local_index"],
                    "element": item["element"] or None,
                    "status": "excluded_unresolved_transform",
                    "category": None,
                    "length_svg_units": None,
                    "subpath_count": 0,
                    "included_in_all": False,
                    "included_in_straight": False,
                    "raw_semantic_present": item["stats"]["raw_semantic_present"],
                    "raw_instance_present": item["stats"]["raw_instance_present"],
                    "transformed": bool(sources),
                    "reasons": (
                        ["unapplied_transform"]
                        if sources
                        else ["drawing_excluded_due_to_unresolved_transform"]
                    ),
                }
            )
        counts["drawings_excluded"] = 1
        return _result(
            parser_contract=PARSER_CONTRACT,
            parser_contract_status="assumed_structural",
            parser_schema_present="schema" in doc and doc.get("schema") is not None,
            parser_schema=doc.get("schema"),
            parser_schema_version=doc.get("schema_version"),
            status=STATUS_EXCLUDED,
            reasons=["unapplied_transform"],
            complete=False,
            counts=counts,
            raw_label_counts={
                **label_counts,
                "raw_instance_distinct_nonempty": len(instance_values),
                "definition": "primitive-level presence of configured semantic/"
                "instance aliases with a parsed integer value; not platform IDs "
                "and not used to filter GT",
            },
            excluded={
                "drawings": 1,
                "primitives": len(primitives),
                "affected_primitives": affected,
                "transform_sources": transform_sources,
                "raw_instance_nonempty": label_counts["raw_instance_nonempty"],
                "raw_instance_distinct_nonempty": len(instance_values),
                "reason": "unapplied_transform",
            },
            primitive_records=records,
            polylines_all=[],
            polylines_straight=[],
            approximation={
                "curve_steps": curve_steps,
                "circle_steps": circle_steps,
                "path_curves": "fixed-step polyline approximation; 128 steps is "
                "not a 1 px geometric error guarantee",
                "circle_ellipse": "closed polyline reference only; circle/ellipse "
                "type recovery is not claimed",
                "circle_length": "2*pi*r",
                "ellipse_length": "Ramanujan-II approximation",
                "units": "source SVG user units before viewBox/pixel mapping; "
                "geometry is not evaluated for excluded drawings",
            },
        )

    invalid = False
    unsupported = 0

    for item in preflight:
        index = item["index"]
        primitive = item["primitive"]
        element = item["element"]
        local_index = item["local_index"]
        stats = item["stats"]

        record: Dict[str, Any] = {
            "primitive_index": index,
            "local_index": local_index,
            "element": element or None,
            "status": "included",
            "category": None,
            "length_svg_units": None,
            "subpath_count": 0,
            "included_in_all": False,
            "included_in_straight": False,
            "raw_semantic_present": stats["raw_semantic_present"],
            "raw_instance_present": stats["raw_instance_present"],
            "transformed": False,
            "reasons": [],
        }

        geometry_info = primitive.get("geometry") or {}
        if not isinstance(geometry_info, Mapping):
            raise AdapterInputError(
                f"primitive {index} geometry must be a mapping when present"
            )

        if element == "path":
            raw_d = geometry_info.get("d")
            if not isinstance(raw_d, str) or not raw_d.strip():
                invalid = True
                counts["invalid_geometry"] += 1
                record.update({"status": "invalid", "reasons": ["missing_path_data"]})
                reasons.append("missing_path_data")
                records.append(record)
                continue
            try:
                parsed = _geometry.parse_path_data(raw_d, curve_steps=curve_steps)
            except _geometry.PathDataError as exc:
                invalid = True
                counts["invalid_geometry"] += 1
                record.update(
                    {
                        "status": "invalid",
                        "reasons": [f"path_data_error:{exc}"],
                    }
                )
                reasons.append("path_data_error")
                records.append(record)
                continue

            subpath_polylines: List[List[List[float]]] = []
            subpath_lengths: List[float] = []
            subpath_invalid = False
            for subpath in parsed["subpaths"]:
                points = [[point[0], point[1]] for point in subpath]
                if not _all_finite_points(points):
                    subpath_invalid = True
                    break
                length = _polyline_length(points)
                if not math.isfinite(length):
                    subpath_invalid = True
                    break
                if length > 0.0 and len(points) >= 2:
                    subpath_polylines.append(points)
                    subpath_lengths.append(length)
                else:
                    counts["zero_length_subpaths"] += 1
            overflow_reason = None
            if subpath_invalid:
                overflow_reason = "non_finite_path_geometry"
            else:
                try:
                    total_length = (
                        math.fsum(subpath_lengths) if subpath_lengths else 0.0
                    )
                except OverflowError:
                    total_length = math.inf
                    overflow_reason = "numeric_overflow:path_length_sum"
                if not math.isfinite(total_length) and overflow_reason is None:
                    overflow_reason = "numeric_overflow:path_length_sum"
            if overflow_reason is not None:
                invalid = True
                counts["invalid_geometry"] += 1
                record.update({"status": "invalid", "reasons": [overflow_reason]})
                reasons.append(overflow_reason)
                records.append(record)
                continue

            record["length_svg_units"] = total_length
            record["subpath_count"] = len(subpath_polylines)
            counts["subpaths_total"] += len(subpath_polylines)
            if total_length <= 0.0:
                counts["zero_length"] += 1
                record.update(
                    {"status": "zero_length", "reasons": ["zero_drawn_length"]}
                )
                records.append(record)
                continue

            counts["path_positive"] += 1
            classification = _eval.classify_path_geometry(
                raw_d, drawn_length=total_length
            )
            category = classification["category"]
            record["category"] = category
            if category == _eval.CATEGORY_STRAIGHT_ONLY:
                counts["straight_only_paths"] += 1
            elif category == _eval.CATEGORY_CURVED_OR_MIXED:
                counts["curved_or_mixed_paths"] += 1
            else:
                unsupported += 1
                counts["unsupported_primitives"] += 1
                record["reasons"] = [classification.get("reason") or "unsupported"]

            polylines_all.extend(subpath_polylines)
            record["included_in_all"] = True
            if category == _eval.CATEGORY_STRAIGHT_ONLY:
                polylines_straight.extend(subpath_polylines)
                record["included_in_straight"] = True
            records.append(record)
            continue

        if element in ("circle", "ellipse"):
            cx = geometry_info.get("cx", 0.0)
            cy = geometry_info.get("cy", 0.0)
            try:
                cx = _number(cx, "circle/ellipse cx")
                cy = _number(cy, "circle/ellipse cy")
            except AdapterInputError:
                invalid = True
                counts["invalid_geometry"] += 1
                record.update(
                    {"status": "invalid", "reasons": ["non_finite_shape_center"]}
                )
                reasons.append("non_finite_shape_center")
                records.append(record)
                continue

            if element == "circle":
                radius = geometry_info.get("r")
                try:
                    radius = _number(radius, "circle r")
                except AdapterInputError:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {"status": "invalid", "reasons": ["invalid_circle_radius"]}
                    )
                    reasons.append("invalid_circle_radius")
                    records.append(record)
                    continue
                if radius < 0.0:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {"status": "invalid", "reasons": ["negative_circle_radius"]}
                    )
                    reasons.append("negative_circle_radius")
                    records.append(record)
                    continue
                if radius == 0.0:
                    counts["zero_length"] += 1
                    record.update(
                        {"status": "zero_length", "reasons": ["zero_circle_radius"]}
                    )
                    records.append(record)
                    continue
                try:
                    points = _close_ring(
                        _geometry.circle_samples(cx, cy, radius, circle_steps)
                    )
                    length = 2.0 * math.pi * radius
                except OverflowError:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {
                            "status": "invalid",
                            "reasons": ["numeric_overflow:shape_length"],
                        }
                    )
                    reasons.append("numeric_overflow:shape_length")
                    records.append(record)
                    continue
                counts["circle_valid"] += 1
                category = _eval.CATEGORY_CIRCLE
                approximation = "circle_polyline_reference"
            else:
                rx = geometry_info.get("rx")
                ry = geometry_info.get("ry")
                try:
                    rx = _number(rx, "ellipse rx")
                    ry = _number(ry, "ellipse ry")
                except AdapterInputError:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {"status": "invalid", "reasons": ["invalid_ellipse_radius"]}
                    )
                    reasons.append("invalid_ellipse_radius")
                    records.append(record)
                    continue
                if rx < 0.0 or ry < 0.0:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {"status": "invalid", "reasons": ["negative_ellipse_radius"]}
                    )
                    reasons.append("negative_ellipse_radius")
                    records.append(record)
                    continue
                if rx == 0.0 or ry == 0.0:
                    counts["zero_length"] += 1
                    record.update(
                        {"status": "zero_length", "reasons": ["zero_ellipse_radius"]}
                    )
                    records.append(record)
                    continue
                try:
                    points = _close_ring(
                        _geometry.ellipse_samples(cx, cy, rx, ry, circle_steps)
                    )
                    length = _geometry.ellipse_perimeter(rx, ry)
                except OverflowError:
                    invalid = True
                    counts["invalid_geometry"] += 1
                    record.update(
                        {
                            "status": "invalid",
                            "reasons": ["numeric_overflow:shape_length"],
                        }
                    )
                    reasons.append("numeric_overflow:shape_length")
                    records.append(record)
                    continue
                counts["ellipse_valid"] += 1
                category = _eval.CATEGORY_ELLIPSE
                approximation = "Ramanujan-II"

            shape_reason = None
            if not _all_finite_points(points):
                shape_reason = "numeric_overflow:shape_points"
            elif not math.isfinite(length):
                shape_reason = "numeric_overflow:shape_length"
            if shape_reason is not None:
                invalid = True
                counts["invalid_geometry"] += 1
                record.update({"status": "invalid", "reasons": [shape_reason]})
                reasons.append(shape_reason)
                records.append(record)
                continue

            record.update(
                {
                    "category": category,
                    "length_svg_units": length,
                    "subpath_count": 1,
                    "included_in_all": True,
                    "reasons": [f"approximation:{approximation}"],
                }
            )
            polylines_all.append(points)
            records.append(record)
            continue

        unsupported += 1
        counts["unsupported_primitives"] += 1
        record.update(
            {"status": "unsupported", "reasons": ["unsupported_element"]}
        )
        records.append(record)

    if invalid:
        status = STATUS_FAILED
        polylines_all = []
        polylines_straight = []
        counts["polylines_all"] = 0
        counts["polylines_straight"] = 0
        counts["drawings_failed"] = 1
    else:
        status = STATUS_OK
        counts["polylines_all"] = len(polylines_all)
        counts["polylines_straight"] = len(polylines_straight)

    complete = status == STATUS_OK and unsupported == 0
    if status == STATUS_OK and unsupported:
        reasons.append("unsupported_primitives_present")

    parser_schema_present = "schema" in doc and doc.get("schema") is not None
    return _result(
        parser_contract=PARSER_CONTRACT,
        parser_contract_status="assumed_structural",
        parser_schema_present=parser_schema_present,
        parser_schema=doc.get("schema"),
        parser_schema_version=doc.get("schema_version"),
        status=status,
        reasons=list(dict.fromkeys(reasons)),
        complete=complete,
        counts={**counts, **{
            "transformed_primitives": affected,
        }},
        raw_label_counts={
            **label_counts,
            "raw_instance_distinct_nonempty": len(instance_values),
            "definition": "primitive-level presence of configured semantic/instance "
            "aliases with a parsed integer value; not platform IDs and not used "
            "to filter GT",
        },
        excluded=(
            {
                "drawings": 1,
                "primitives": len(primitives),
                "affected_primitives": affected,
                "transform_sources": transform_sources,
                "raw_instance_nonempty": label_counts["raw_instance_nonempty"],
                "raw_instance_distinct_nonempty": len(instance_values),
                "reason": "unapplied_transform",
            }
            if status == STATUS_EXCLUDED
            else None
        ),
        primitive_records=records,
        polylines_all=polylines_all,
        polylines_straight=polylines_straight,
        approximation={
            "curve_steps": curve_steps,
            "circle_steps": circle_steps,
            "path_curves": "fixed-step polyline approximation; 128 steps is not a "
            "1 px geometric error guarantee",
            "circle_ellipse": "closed polyline reference only; circle/ellipse "
            "type recovery is not claimed",
            "circle_length": "2*pi*r",
            "ellipse_length": "Ramanujan-II approximation",
            "units": "source SVG user units before viewBox/pixel mapping; not pixels",
        },
    )


def _parse_par(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None:
        return {
            "raw": None,
            "align": "xMidYMid",
            "meet_or_slice": "meet",
            "defaulted": True,
            "reason": None,
        }
    if not isinstance(raw, str):
        raise AdapterInputError("preserve_aspect_ratio_raw must be text or None")
    tokens = raw.strip().split()
    if not tokens:
        return {
            "raw": raw,
            "align": None,
            "meet_or_slice": None,
            "defaulted": False,
            "reason": "empty_preserve_aspect_ratio",
        }
    align = tokens[0]
    if align == "defer":
        return {
            "raw": raw,
            "align": None,
            "meet_or_slice": None,
            "defaulted": False,
            "reason": "defer_not_supported",
        }
    if align == "none":
        if len(tokens) != 1:
            return {
                "raw": raw,
                "align": None,
                "meet_or_slice": None,
                "defaulted": False,
                "reason": "invalid_none_syntax",
            }
        return {
            "raw": raw,
            "align": "none",
            "meet_or_slice": None,
            "defaulted": False,
            "reason": None,
        }
    if align not in _ALIGNMENTS:
        return {
            "raw": raw,
            "align": None,
            "meet_or_slice": None,
            "defaulted": False,
            "reason": "unknown_alignment",
        }
    if len(tokens) > 2:
        return {
            "raw": raw,
            "align": None,
            "meet_or_slice": None,
            "defaulted": False,
            "reason": "invalid_preserve_aspect_ratio",
        }
    meet_or_slice = tokens[1] if len(tokens) == 2 else "meet"
    if meet_or_slice not in ("meet", "slice"):
        return {
            "raw": raw,
            "align": None,
            "meet_or_slice": None,
            "defaulted": False,
            "reason": "unknown_meet_or_slice",
        }
    if meet_or_slice == "slice":
        return {
            "raw": raw,
            "align": align,
            "meet_or_slice": "slice",
            "defaulted": False,
            "reason": "slice_not_implemented",
        }
    return {
        "raw": raw,
        "align": align,
        "meet_or_slice": "meet",
        "defaulted": False,
        "reason": None,
    }


def _parse_root_length(raw: Any, name: str) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {"attr": name, "raw": raw, "value": None, "unit": None, "status": "missing"}
    if not isinstance(raw, str):
        raise AdapterInputError(f"root {name} attribute must be text when present")
    match = _LENGTH_RE.match(raw.strip())
    if match is None:
        return {
            "attr": name,
            "raw": raw,
            "value": None,
            "unit": None,
            "status": "unparsable",
        }
    value = float(match.group(1))
    unit = match.group(2).lower()
    if not math.isfinite(value) or value <= 0.0:
        return {
            "attr": name,
            "raw": raw,
            "value": None,
            "unit": unit,
            "status": "non_positive_or_non_finite",
        }
    if unit in ("", "px"):
        return {"attr": name, "raw": raw, "value": value, "unit": unit or "unitless", "status": "ok"}
    return {
        "attr": name,
        "raw": raw,
        "value": None,
        "unit": unit,
        "status": "unsupported_unit",
    }


def map_viewbox_to_pixels(
    view_box: Any,
    png_width: Any,
    png_height: Any,
    preserve_aspect_ratio_raw: Optional[str] = None,
    *,
    viewport_mode: str = "full_svg_viewport",
) -> Dict[str, Any]:
    """Compute the explicit viewBox -> PNG pixel mapping.

    ``status="ok"`` only means the transform is mathematically defined;
    ``verification`` stays ``"unverified"``.  Ambiguous inputs return
    ``alignment_pending`` with reasons.  No screenshot margin or crop is
    inferred.
    """

    width = _number(png_width, "png_width", positive=True)
    height = _number(png_height, "png_height", positive=True)
    if viewport_mode != "full_svg_viewport":
        return _result(
            status=STATUS_ALIGNMENT_PENDING,
            reasons=[f"unsupported_viewport_mode:{viewport_mode}"],
            png={"width": width, "height": height},
            viewport_mode=viewport_mode,
            view_box_raw=None,
            scale_x=None,
            scale_y=None,
            offset_x=None,
            offset_y=None,
            non_uniform=None,
            viewport=None,
            content_rect=None,
            preserve_aspect_ratio=None,
            root_size=None,
            verification="unverified",
            assumptions=[],
            evidence=[],
            clip={"applied": False},
        )

    if not isinstance(view_box, Mapping):
        return _result(
            status=STATUS_ALIGNMENT_PENDING,
            reasons=["viewbox_missing_or_not_a_mapping"],
            png={"width": width, "height": height},
            viewport_mode=viewport_mode,
            view_box_raw=None,
            scale_x=None,
            scale_y=None,
            offset_x=None,
            offset_y=None,
            non_uniform=None,
            viewport=None,
            content_rect=None,
            preserve_aspect_ratio=None,
            root_size=None,
            verification="unverified",
            assumptions=[],
            evidence=[],
            clip={"applied": False},
        )

    root_size = {
        "width": _parse_root_length(view_box.get("width_attr"), "width"),
        "height": _parse_root_length(view_box.get("height_attr"), "height"),
    }
    par = _parse_par(preserve_aspect_ratio_raw)
    pending_reasons: List[str] = []
    if par["reason"]:
        pending_reasons.append(par["reason"])
    for side in ("width", "height"):
        status = root_size[side]["status"]
        if status in ("unsupported_unit", "unparsable", "non_positive_or_non_finite"):
            pending_reasons.append(f"root_{side}_unit_unresolved:{status}")

    view_box_raw = view_box.get("raw")
    valid = bool(view_box.get("valid"))
    try:
        vbx = _number(view_box.get("x", 0.0), "viewBox x")
        vby = _number(view_box.get("y", 0.0), "viewBox y")
        vbw = _number(view_box.get("width"), "viewBox width", positive=True)
        vbh = _number(view_box.get("height"), "viewBox height", positive=True)
    except AdapterInputError:
        valid = False
        vbx = vby = vbw = vbh = 0.0

    if not valid or vbw <= 0.0 or vbh <= 0.0:
        if "viewbox_missing_or_invalid" not in pending_reasons:
            pending_reasons.append("viewbox_missing_or_invalid")

    if pending_reasons:
        return _result(
            status=STATUS_ALIGNMENT_PENDING,
            reasons=list(dict.fromkeys(pending_reasons)),
            png={"width": width, "height": height},
            viewport_mode=viewport_mode,
            view_box_raw=view_box_raw,
            scale_x=None,
            scale_y=None,
            offset_x=None,
            offset_y=None,
            non_uniform=None,
            viewport=None,
            content_rect=None,
            preserve_aspect_ratio=par,
            root_size=root_size,
            verification="unverified",
            assumptions=[],
            evidence=[
                "mapping requires a valid viewBox, resolvable root units and a "
                "supported preserveAspectRatio; no margin/crop guess is made"
            ],
            clip={"applied": False},
        )

    sx = width / vbw
    sy = height / vbh
    if par["align"] == "none":
        scale_x, scale_y = sx, sy
        content_width, content_height = width, height
        content_x = content_y = 0.0
        non_uniform = True
    else:
        scale = min(sx, sy)
        scale_x = scale_y = scale
        content_width = vbw * scale
        content_height = vbh * scale
        align = par["align"]
        x_align = align[:4]
        y_align = align[4:8]
        if x_align == "xMin":
            content_x = 0.0
        elif x_align == "xMid":
            content_x = (width - content_width) / 2.0
        elif x_align == "xMax":
            content_x = width - content_width
        else:  # pragma: no cover - guarded by _ALIGNMENTS
            raise AdapterInputError(f"unsupported x alignment: {align!r}")
        if y_align == "YMin":
            content_y = 0.0
        elif y_align == "YMid":
            content_y = (height - content_height) / 2.0
        elif y_align == "YMax":
            content_y = height - content_height
        else:  # pragma: no cover - guarded by _ALIGNMENTS
            raise AdapterInputError(f"unsupported y alignment: {align!r}")
        non_uniform = False

    offset_x = content_x - vbx * scale_x
    offset_y = content_y - vby * scale_y
    for name, value in (
        ("scale_x", scale_x),
        ("scale_y", scale_y),
        ("offset_x", offset_x),
        ("offset_y", offset_y),
        ("content_x", content_x),
        ("content_y", content_y),
        ("content_width", content_width),
        ("content_height", content_height),
    ):
        if not math.isfinite(value):
            raise AdapterInputError(f"mapping {name} overflowed float64")

    assumptions = [
        "pixel coordinates use integer pixel centres; no half-pixel shift",
        "root width/height (when present) are CSS px or unitless and the root "
        "viewport is scaled to the PNG dimensions",
        "no screenshot margin, crop or arbitrary offset is inferred",
    ]
    if par["defaulted"]:
        assumptions.append("preserveAspectRatio absent: SVG default xMidYMid meet")

    return _result(
        status=STATUS_OK,
        reasons=[],
        png={"width": width, "height": height},
        viewport_mode=viewport_mode,
        view_box_raw=view_box_raw,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
        non_uniform=non_uniform,
        viewport={"x": 0.0, "y": 0.0, "width": width, "height": height},
        content_rect={
            "x": content_x,
            "y": content_y,
            "width": content_width,
            "height": content_height,
        },
        preserve_aspect_ratio=par,
        root_size=root_size,
        verification="unverified",
        assumptions=assumptions,
        evidence=[
            f"x' = x * {scale_x} + {offset_x}",
            f"y' = y * {scale_y} + {offset_y}",
            f"content rect {content_width} x {content_height} at "
            f"({content_x}, {content_y})",
        ],
        clip={
            "applied": False,
            "note": "runner must compare mapped points with viewport/content_rect "
            "and mark alignment_pending instead of silently dropping GT",
        },
    )


def apply_pixel_mapping(
    polylines: Iterable[Iterable[Sequence[float]]], mapping: Mapping[str, Any]
) -> List[List[List[float]]]:
    """Apply an ``ok`` viewBox mapping without clipping or simplification."""

    if not isinstance(mapping, Mapping):
        raise AdapterInputError("apply_pixel_mapping expects a mapping document")
    if mapping.get("status") != STATUS_OK:
        raise AdapterInputError(
            f"pixel mapping is not usable: status={mapping.get('status')!r}"
        )
    scale_x = _number(mapping.get("scale_x"), "mapping scale_x", positive=True)
    scale_y = _number(mapping.get("scale_y"), "mapping scale_y", positive=True)
    offset_x = _number(mapping.get("offset_x"), "mapping offset_x")
    offset_y = _number(mapping.get("offset_y"), "mapping offset_y")

    mapped: List[List[List[float]]] = []
    for polyline in polylines:
        output_points: List[List[float]] = []
        for point in polyline:
            if len(point) != 2:
                raise AdapterInputError(f"point must be (x, y), got {point!r}")
            x = _number(point[0], "polyline x")
            y = _number(point[1], "polyline y")
            px = x * scale_x + offset_x
            py = y * scale_y + offset_y
            if not (math.isfinite(px) and math.isfinite(py)):
                raise AdapterInputError("mapped coordinate overflowed float64")
            output_points.append([px, py])
        mapped.append(output_points)
    return mapped


def _frontend_failure(reason: str, frontend_status: Any, details: Any = None) -> Dict[str, Any]:
    return _result(
        status=PRED_FAILED,
        reasons=[reason],
        frontend_status=frontend_status,
        frontend_reason=details,
        polylines=[],
        line_count=0,
        input=None,
    )


def pred_from_frontend(result: Any) -> Dict[str, Any]:
    """Convert a ``cadvec.png_frontend`` result into prediction polylines.

    ``ok`` yields every raw line as a two-point polyline (no merging).
    ``empty_input`` / ``no_lines_detected`` are legal empty predictions with
    their reason kept.  ``processing_limited`` becomes
    ``evaluation_limited``; input/decode/unsupported errors become ``failed``.
    Unknown or malformed schemas never become empty predictions.
    """

    if not isinstance(result, Mapping):
        raise AdapterInputError("pred_from_frontend expects a mapping result")
    if result.get("schema") != FRONTEND_SCHEMA:
        return _frontend_failure(
            "invalid_frontend_schema", result.get("status"), result.get("schema")
        )
    version = result.get("schema_version")
    if isinstance(version, bool) or version != FRONTEND_SCHEMA_VERSION:
        return _frontend_failure(
            "invalid_frontend_schema_version", result.get("status"), version
        )
    status = result.get("status")
    if not isinstance(status, str):
        return _frontend_failure("invalid_frontend_status", status)

    def pixel_size() -> Optional[Dict[str, int]]:
        block = result.get("input")
        if not isinstance(block, Mapping):
            return None
        width = block.get("width")
        height = block.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            return None
        return {"width": width, "height": height}

    input_out = pixel_size()
    lines = result.get("lines")
    line_count = result.get("line_count")

    if status in ("empty_input", "no_lines_detected"):
        if not isinstance(lines, list) or lines:
            return _frontend_failure(
                "empty_status_requires_explicit_empty_lines",
                status,
                result.get("status_reason"),
            )
        if (
            isinstance(line_count, bool)
            or not isinstance(line_count, int)
            or line_count != 0
        ):
            return _frontend_failure(
                "empty_status_line_count_must_be_zero",
                status,
                result.get("status_reason"),
            )
        if input_out is None:
            return _frontend_failure(
                "missing_or_invalid_frontend_dimensions",
                status,
                result.get("status_reason"),
            )
        return _result(
            status=PRED_EMPTY,
            reasons=[result.get("status_reason") or status],
            frontend_status=status,
            frontend_reason=result.get("status_reason"),
            polylines=[],
            line_count=0,
            input=input_out,
        )

    if status == "processing_limited":
        if lines not in (None, []):
            return _frontend_failure(
                "limited_status_with_lines", status, result.get("status_reason")
            )
        return _result(
            status=PRED_LIMITED,
            reasons=[result.get("status_reason") or status],
            frontend_status=status,
            frontend_reason=result.get("status_reason"),
            polylines=[],
            line_count=0,
            input=input_out,
        )

    if status in ("input_error", "decode_error", "unsupported_image"):
        return _result(
            status=PRED_FAILED,
            reasons=[result.get("status_reason") or status],
            frontend_status=status,
            frontend_reason=result.get("status_reason"),
            polylines=[],
            line_count=0,
            input=input_out,
        )

    if status != "ok":
        return _frontend_failure("unknown_frontend_status", status)

    if input_out is None:
        return _frontend_failure(
            "missing_or_invalid_frontend_dimensions", status
        )
    width = input_out["width"]
    height = input_out["height"]
    if not isinstance(lines, list):
        return _frontend_failure("missing_frontend_lines", status)
    if isinstance(line_count, bool) or not isinstance(line_count, int):
        return _frontend_failure("invalid_frontend_line_count", status)
    if line_count != len(lines):
        return _frontend_failure("frontend_line_count_mismatch", status)
    if not lines:
        return _frontend_failure("ok_without_lines", status)

    polylines: List[List[List[float]]] = []
    for index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            return _frontend_failure(
                "invalid_frontend_line", status, f"line {index} is not a mapping"
            )
        try:
            x1 = _number(line.get("x1"), f"line {index} x1")
            y1 = _number(line.get("y1"), f"line {index} y1")
            x2 = _number(line.get("x2"), f"line {index} x2")
            y2 = _number(line.get("y2"), f"line {index} y2")
        except AdapterInputError as exc:
            return _frontend_failure("invalid_frontend_line", status, str(exc))
        if not (
            0.0 <= x1 <= width - 1
            and 0.0 <= x2 <= width - 1
            and 0.0 <= y1 <= height - 1
            and 0.0 <= y2 <= height - 1
        ):
            return _frontend_failure(
                "frontend_line_out_of_bounds", status, f"line {index}"
            )
        polylines.append([[x1, y1], [x2, y2]])

    return _result(
        status=PRED_OK,
        reasons=[],
        frontend_status=status,
        frontend_reason=result.get("status_reason"),
        polylines=polylines,
        line_count=len(polylines),
        input=input_out,
        notes=[
            "all raw frontend lines are kept; no collinear merge or dedup",
            "line-only frontend: circles/ellipses are not type-recovered",
        ],
    )
