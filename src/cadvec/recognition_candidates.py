"""Candidate construction and conservative SVG/PNG association.

The wrapper never modifies the accepted parser/frontend/adapter modules.  It
rebuilds primitive geometry with explicit primitive/subpath/segment identity,
keeps supervision separate, and applies the frozen cover/touch rule, relaxed
only for several covering primitives of the same known Thing instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import geometry, png_diagnostic_adapter as adapter
from .recognition_contract import ASSOCIATION_TOLERANCE_PX, CANDIDATE_SCHEMA, semantic_kind

TOLERANCE_PX = ASSOCIATION_TOLERANCE_PX


class CandidateBuildError(ValueError):
    code = "candidate_build_error"


@dataclass(frozen=True)
class GTPrimitive:
    primitive_uid: str
    local_index: int
    element: str
    polylines: Tuple[Tuple[Tuple[float, float], ...], ...]
    complete: bool
    reason: Optional[str]
    semantic_present: bool
    semantic_status: str
    semantic_raw: Optional[int]
    instance_present: bool
    instance_status: str
    instance_raw: Optional[int]


def _label_fields(primitive: Mapping[str, Any]) -> Dict[str, Any]:
    label = primitive.get("label") or {}
    semantic = label.get("semantic") or {}
    instance = label.get("instance") or {}
    sp = bool(semantic.get("present"))
    sv = semantic.get("value")
    ip = bool(instance.get("present"))
    iv = instance.get("value")
    if not sp:
        ss = "missing"
    elif isinstance(sv, int) and not isinstance(sv, bool) and 1 <= sv <= 35:
        ss = "ok"
    else:
        ss = "invalid"
    if not ip:
        i_s = "missing"
    elif not isinstance(iv, int) or isinstance(iv, bool):
        i_s = "invalid"
    elif iv < -1:
        i_s = "invalid"
    elif iv == -1:
        i_s = "no_instance"
    else:
        i_s = "ok"
    return {
        "semantic_present": sp,
        "semantic_status": "ok" if ss == "ok" else ss,
        "semantic_raw": sv if isinstance(sv, int) and not isinstance(sv, bool) else None,
        "instance_present": ip,
        "instance_status": i_s,
        "instance_raw": iv if isinstance(iv, int) and not isinstance(iv, bool) else None,
    }


def _polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_to_polyline(px: float, py: float, points: Sequence[Tuple[float, float]]) -> float:
    if not points:
        return math.inf
    if len(points) == 1:
        return math.hypot(px - points[0][0], py - points[0][1])
    return min(
        _point_segment_distance(
            px, py, points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]
        )
        for i in range(1, len(points))
    )


def build_gt_primitives(
    document: Mapping[str, Any],
    sample_id: str,
    *,
    curve_steps: int = 128,
) -> List[GTPrimitive]:
    primitives = document.get("primitives")
    if not isinstance(primitives, list):
        raise CandidateBuildError("document must contain a primitives list")
    result: List[GTPrimitive] = []
    for primitive in primitives:
        if not isinstance(primitive, Mapping):
            raise CandidateBuildError("primitive must be a mapping")
        local_index = int(primitive.get("local_index") or 0)
        uid = f"{sample_id}#p{local_index}"
        element = str(primitive.get("element") or "").lower()
        labels = _label_fields(primitive)
        complete = True
        reason = None
        polylines: List[Tuple[Tuple[float, float], ...]] = []
        if primitive.get("transform_chain"):
            complete = False
            reason = "unapplied_transform"
        elif element == "path":
            geometry_info = primitive.get("geometry") or {}
            raw_d = geometry_info.get("d")
            if not isinstance(raw_d, str) or not raw_d.strip():
                complete = False
                reason = "missing_path_data"
            else:
                try:
                    parsed = geometry.parse_path_data(raw_d, curve_steps=curve_steps)
                except geometry.PathDataError:
                    complete = False
                    reason = "path_data_error"
                else:
                    for subpath in parsed["subpaths"]:
                        points = tuple((float(p[0]), float(p[1])) for p in subpath)
                        if points and _polyline_length(points) > 0.0 and math.isfinite(_polyline_length(points)):
                            polylines.append(points)
                    if not polylines:
                        complete = False
                        reason = "zero_length_geometry"
        elif element in ("circle", "ellipse"):
            info = primitive.get("geometry") or {}
            try:
                cx, cy = float(info.get("cx")), float(info.get("cy"))
                if element == "circle":
                    r = float(info.get("r"))
                    if not all(math.isfinite(v) for v in (cx, cy, r)) or r <= 0.0:
                        raise ValueError
                    points = geometry.circle_samples(cx, cy, r, 128)
                else:
                    rx, ry = float(info.get("rx")), float(info.get("ry"))
                    if not all(math.isfinite(v) for v in (cx, cy, rx, ry)) or rx <= 0.0 or ry <= 0.0:
                        raise ValueError
                    points = geometry.ellipse_samples(cx, cy, rx, ry, 128)
                points = points + [points[0]]
                polylines.append(tuple((float(p[0]), float(p[1])) for p in points))
            except (TypeError, ValueError, OverflowError):
                complete = False
                reason = "invalid_shape_geometry"
        else:
            complete = False
            reason = "unsupported_element"
        if not complete:
            polylines = []
        result.append(
            GTPrimitive(
                primitive_uid=uid,
                local_index=local_index,
                element=element,
                polylines=tuple(polylines),
                complete=complete,
                reason=reason,
                **labels,
            )
        )
    return result


def _mapping_reason(mapping: Any) -> str:
    if not isinstance(mapping, Mapping):
        return "pixel_mapping_unavailable:not_a_mapping"
    status = str(mapping.get("status") or "unknown")
    reasons = mapping.get("reasons") or []
    if isinstance(reasons, (list, tuple)) and reasons:
        return f"pixel_mapping_unavailable:{status}:{reasons[0]}"
    return f"pixel_mapping_unavailable:{status}"


def map_gt_primitives_to_pixels(
    primitives: Sequence[GTPrimitive],
    mapping: Mapping[str, Any],
) -> List[GTPrimitive]:
    """Map every supported GT polyline through the accepted adapter mapping.

    The mapping is accepted only when ``status == "ok"``.  A missing, invalid,
    unsupported or failed mapping excludes every complete GT primitive; it does
    not drop a competitor or silently compare SVG user units with PNG pixels.
    The returned mapping retains the adapter's ``verification`` field
    (normally ``"unverified"``) and does not upgrade it.
    """

    if not isinstance(mapping, Mapping) or mapping.get("status") != adapter.STATUS_OK:
        reason = _mapping_reason(mapping)
        return [
            replace(
                primitive,
                polylines=(),
                complete=False,
                reason=primitive.reason or reason,
            )
            for primitive in primitives
        ]
    mapped: List[GTPrimitive] = []
    for primitive in primitives:
        if not primitive.complete:
            mapped.append(primitive)
            continue
        try:
            polylines = adapter.apply_pixel_mapping(primitive.polylines, mapping)
        except (adapter.AdapterInputError, TypeError, ValueError, OverflowError) as exc:
            reason = f"pixel_mapping_error:{exc}"
            return [
                replace(
                    item,
                    polylines=(),
                    complete=False,
                    reason=item.reason or reason,
                )
                for item in primitives
            ]
        if len(polylines) != len(primitive.polylines):
            reason = "pixel_mapping_error:polyline_count_changed"
            return [
                replace(
                    item,
                    polylines=(),
                    complete=False,
                    reason=item.reason or reason,
                )
                for item in primitives
            ]
        mapped.append(
            replace(
                primitive,
                polylines=tuple(
                    tuple((float(point[0]), float(point[1])) for point in polyline)
                    for polyline in polylines
                ),
            )
        )
    return mapped


def build_pixel_gt_primitives(
    document: Mapping[str, Any],
    sample_id: str,
    *,
    png_width: Any,
    png_height: Any,
    preserve_aspect_ratio_raw: Optional[str] = None,
    viewport_mode: str = "full_svg_viewport",
    curve_steps: int = 128,
) -> Tuple[List[GTPrimitive], Dict[str, Any]]:
    """Build GT and map all supported geometry to PNG pixel coordinates.

    This is the single association boundary: callers must use the returned
    pixel-space GT with :func:`associate_png_candidates`.  Mapping is delegated
    to ``png_diagnostic_adapter.map_viewbox_to_pixels``; formulas are not
    duplicated here.  If mapping is unavailable, the returned GT is marked
    incomplete and association will exclude the whole image.
    """

    if not isinstance(document, Mapping):
        raise CandidateBuildError("document must be a mapping")
    primitives = build_gt_primitives(document, sample_id, curve_steps=curve_steps)
    try:
        mapping = adapter.map_viewbox_to_pixels(
            document.get("view_box"),
            png_width,
            png_height,
            preserve_aspect_ratio_raw,
            viewport_mode=viewport_mode,
        )
    except (adapter.AdapterInputError, TypeError, ValueError, OverflowError) as exc:
        mapping = {
            "status": adapter.STATUS_ALIGNMENT_PENDING,
            "reasons": [f"mapping_error:{exc}"],
            "verification": "unverified",
        }
    return map_gt_primitives_to_pixels(primitives, mapping), dict(mapping)


def _excluded_supervision(uid: str, status: str = "excluded") -> Dict[str, Any]:
    return {
        "candidate_uid": uid,
        "semantic_raw": None,
        "semantic_status": status,
        "instance_raw": None,
        "instance_status": status,
    }


def _direct_supervision(uid: str, primitive: GTPrimitive) -> Dict[str, Any]:
    return {
        "candidate_uid": uid,
        "semantic_raw": primitive.semantic_raw,
        "semantic_status": primitive.semantic_status,
        "instance_raw": primitive.instance_raw,
        "instance_status": primitive.instance_status,
    }


def _association_identity(primitive: GTPrimitive) -> Optional[Tuple[str, int, int]]:
    """Return a contract-backed annotation identity, or ``None`` if unknown.

    Only ``known semantic Thing`` with a ``known non-negative instance`` is an
    identity.  Missing/invalid labels and Thing ``instance=-1`` are unknown and
    must never be equated; ``instance_raw`` equality across different semantics
    is not identity either.  Stuff (raw 31-35) has no contract-defined object
    identity, so it never yields one here and keeps the old association rule.
    """

    if primitive.semantic_status != "ok" or primitive.instance_status != "ok":
        return None
    semantic = primitive.semantic_raw
    instance = primitive.instance_raw
    if semantic_kind(semantic) != "Thing":
        return None
    if not isinstance(instance, int) or isinstance(instance, bool) or instance < 0:
        return None
    return ("Thing", int(semantic), int(instance))


def _candidate(
    *,
    candidate_uid: str,
    sample_id: str,
    group_id: Optional[str],
    role: Optional[str],
    source_kind: str,
    primitive_uid: Optional[str],
    local_index: Optional[int],
    subpath_index: int,
    segment_index: int,
    element: str,
    points: Sequence[Tuple[float, float]],
    supervision: Mapping[str, Any],
    association: Mapping[str, Any],
    image: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "version": 1,
        "candidate_uid": candidate_uid,
        "sample_id": sample_id,
        "group_id": group_id,
        "role": role,
        "source_kind": source_kind,
        "primitive_uid": primitive_uid,
        "local_index": local_index,
        "subpath_index": subpath_index,
        "segment_index": segment_index,
        "element": element,
        "geometry": {
            "space": "png_px" if source_kind == "png_line" else "svg_user",
            "points": [[float(p[0]), float(p[1])] for p in points],
            "points_px": [[float(p[0]), float(p[1])] for p in points]
            if source_kind == "png_line"
            else None,
            "curve_steps": 128 if source_kind != "png_line" else None,
            "approximation": "polyline_128" if source_kind != "png_line" else None,
        },
        "association": dict(association),
        "supervision": dict(supervision),
        "image": dict(image or {}),
    }


def build_svg_candidates(
    document: Mapping[str, Any],
    sample_id: str,
    *,
    group_id: Optional[str] = None,
    role: Optional[str] = None,
    curve_steps: int = 128,
) -> Dict[str, Any]:
    gt = build_gt_primitives(document, sample_id, curve_steps=curve_steps)
    complete = all(item.complete for item in gt)
    candidates: List[Dict[str, Any]] = []
    if complete:
        for primitive in gt:
            for subpath_index, polyline in enumerate(primitive.polylines):
                for segment_index in range(1, len(polyline)):
                    uid = f"{primitive.primitive_uid}#s{subpath_index}#g{segment_index - 1}"
                    supervision = _direct_supervision(uid, primitive)
                    association = {
                        "status": "not_applicable",
                        "partial": False,
                        "reason": None,
                        "gt_primitive_uids": [primitive.primitive_uid],
                        "d_cover_px": None,
                        "d_touch_px": None,
                        "tolerance_px": TOLERANCE_PX,
                    }
                    candidates.append(
                        _candidate(
                            candidate_uid=uid,
                            sample_id=sample_id,
                            group_id=group_id,
                            role=role,
                            source_kind="svg_primitive",
                            primitive_uid=primitive.primitive_uid,
                            local_index=primitive.local_index,
                            subpath_index=subpath_index,
                            segment_index=segment_index - 1,
                            element=primitive.element,
                            points=(polyline[segment_index - 1], polyline[segment_index]),
                            supervision=supervision,
                            association=association,
                        )
                    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "version": 1,
        "sample_id": sample_id,
        "group_id": group_id,
        "role": role,
        "geometry_complete": complete,
        "primitives": gt,
        "candidates": candidates,
        "excluded": [item for item in gt if not item.complete],
    }


def build_png_candidates(
    prediction: Mapping[str, Any],
    sample_id: str,
    *,
    group_id: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    status = prediction.get("status")
    if status in ("empty_input", "no_lines_detected"):
        return []
    if status != "ok":
        raise CandidateBuildError(f"frontend status is not usable: {status!r}")
    lines = prediction.get("lines")
    if not isinstance(lines, list):
        raise CandidateBuildError("frontend result has no lines list")
    image = prediction.get("input") or {}
    candidates: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            raise CandidateBuildError("line record must be a mapping")
        x1, y1, x2, y2 = (float(line[key]) for key in ("x1", "y1", "x2", "y2"))
        uid = f"{sample_id}#l{index}"
        candidates.append(
            _candidate(
                candidate_uid=uid,
                sample_id=sample_id,
                group_id=group_id,
                role=role,
                source_kind="png_line",
                primitive_uid=None,
                local_index=None,
                subpath_index=0,
                segment_index=index,
                element="line",
                points=((x1, y1), (x2, y2)),
                supervision=_excluded_supervision(uid, "excluded"),
                association={
                    "status": "unmatched",
                    "partial": False,
                    "reason": None,
                    "gt_primitive_uids": [],
                    "d_cover_px": None,
                    "d_touch_px": None,
                    "tolerance_px": TOLERANCE_PX,
                },
                image={"width": image.get("width"), "height": image.get("height")},
            )
        )
    return candidates


def _candidate_points(candidate: Mapping[str, Any]) -> List[Tuple[float, float]]:
    geometry_block = candidate.get("geometry") or {}
    points = geometry_block.get("points_px") or geometry_block.get("points") or []
    return [(float(p[0]), float(p[1])) for p in points]


def associate_png_candidates(
    candidates: Sequence[Dict[str, Any]],
    primitives: Sequence[GTPrimitive],
    *,
    tolerance_px: float = TOLERANCE_PX,
) -> Dict[str, int]:
    """Associate each candidate with at most one GT primitive.

    A primitive covers the candidate when all three query points lie within
    ``tolerance_px``.  Covering primitives that share one *known* annotation
    identity (``Thing`` with ``semantic_status == "ok"`` and a non-negative
    ``instance_status == "ok"`` instance) are treated as one annotation: any of
    them may cover, and touching siblings of that identity do not veto.  Ties
    resolve to the nearest cover, first in GT order.  Unknown labels, Thing
    ``instance = -1`` and Stuff (raw 31-35) yield no identity, so they keep the
    original single-cover veto.
    """
    if any(not primitive.complete for primitive in primitives):
        for candidate in candidates:
            candidate["association"] = {
                "status": "excluded",
                "partial": False,
                "reason": "geometry_incomplete",
                "gt_primitive_uids": [],
                "d_cover_px": None,
                "d_touch_px": None,
                "tolerance_px": tolerance_px,
            }
            candidate["supervision"] = _excluded_supervision(candidate["candidate_uid"])
        return {"unique": 0, "ambiguous": 0, "unmatched": 0, "excluded": len(candidates), "partial": 0}
    counts = {"unique": 0, "ambiguous": 0, "unmatched": 0, "excluded": 0, "partial": 0}
    for candidate in candidates:
        points = _candidate_points(candidate)
        if len(points) < 2:
            candidate["association"].update({"status": "excluded", "reason": "candidate_geometry_invalid"})
            candidate["supervision"] = _excluded_supervision(candidate["candidate_uid"])
            counts["excluded"] += 1
            continue
        mid = ((points[0][0] + points[-1][0]) / 2.0, (points[0][1] + points[-1][1]) / 2.0)
        query = [points[0], mid, points[-1]]
        scored: List[Tuple[GTPrimitive, float, float]] = []
        for primitive in primitives:
            if not primitive.complete:
                continue
            distances = [
                min(_point_to_polyline(x, y, polyline) for polyline in primitive.polylines)
                for x, y in query
            ]
            scored.append((primitive, max(distances), min(distances)))
        covers = [item for item in scored if item[1] <= tolerance_px]
        touches = [item for item in scored if item[2] <= tolerance_px]
        cover_identity = _association_identity(covers[0][0]) if covers else None
        identity_agrees = cover_identity is not None and all(
            _association_identity(item[0]) == cover_identity for item in covers
        )
        cover_ids = {id(item[0]) for item in covers}
        blocking_touches = [
            item
            for item in touches
            if id(item[0]) not in cover_ids
            and not (identity_agrees and _association_identity(item[0]) == cover_identity)
        ]
        if covers and not blocking_touches and (len(covers) == 1 or identity_agrees):
            # Same known Thing identity may be covered by several primitives
            # (junction/corner supervision); the closest cover, first in GT
            # order on ties, is the deterministic representative.
            primitive = min(covers, key=lambda item: item[1])[0]
            status = "unique"
            partial = False
            reason = None
            uid = primitive.primitive_uid
            supervision = _direct_supervision(candidate["candidate_uid"], primitive)
            counts["unique"] += 1
        elif touches:
            status = "ambiguous"
            partial = True
            reason = "touch_without_cover" if not covers else ("multiple_cover" if len(covers) > 1 else "other_touch")
            uid = None
            supervision = _excluded_supervision(candidate["candidate_uid"], "excluded")
            counts["ambiguous"] += 1
            counts["partial"] += 1
        else:
            status = "unmatched"
            partial = False
            reason = None
            uid = None
            supervision = _excluded_supervision(candidate["candidate_uid"], "excluded")
            counts["unmatched"] += 1
        candidate["association"] = {
            "status": status,
            "partial": partial,
            "reason": reason,
            "gt_primitive_uids": [uid] if uid else [],
            "d_cover_px": None if not covers else min(item[1] for item in covers),
            "d_touch_px": None if not touches else min(item[2] for item in touches),
            "tolerance_px": tolerance_px,
        }
        candidate["supervision"] = supervision
    return counts
