"""Internal pre-submission adapter/validator for the dataset README §4 JSON.

The contract is the public ``datasets/README.md`` section 4 ("预测 JSON 格式"):
one UTF-8 JSON per input image, twelve required fields, and strict array/count
correspondence.  This module is deliberately **not** an official platform
validator; it catches local shape/count errors before submission.  Official
evaluator details, primitive-ID mapping and the exact platform schema remain
unconfirmed.

The adapter consumes the existing PNG-only GNN JSONL (one row per candidate)
and emits the README-shaped document.  It only reads the PNG header for
``width``/``height``; it never parses SVG, labels, manifests or role data.

An optional shadow postprocessor (``shadow_postprocess``) is layered on top of
the same rows.  It can merge contiguous, same-sample, same-Thing-semantic,
same-instance-group line candidates into one circular arc (``command=1``) when
explicit foreground evidence and tangent/residual/coverage gates pass.  It
never closes a partial arc and never emits a circle: the current PNG frontend
is line-only, so circle recovery is intentionally unimplemented.  Shadow mode
always returns the untouched baseline document plus the candidate document and
diagnostics; ``adapt_predictions`` and the official output path are unchanged.

Coordinates are calibrated from the PNG raster back to the source SVG viewBox
using the paired-render assumption: ``viewBox="0 0 100 100"`` rendered as a
nominal ``xMidYMid meet`` fit.  For a PNG ``W x H`` the uniform scale is
``s = min(W, H) / 100`` and the letterbox offsets are
``tx = (W - 100*s)/2``, ``ty = (H - 100*s)/2``; pixel points map as
``x_svg = (x_px - tx)/s``, ``y_svg = (y_px - ty)/s``.  Output ``width`` and
``height`` are therefore the fixed viewBox size (100), never the PNG IHDR
dimensions.  Any non-100 viewBox supplied by a caller is rejected; the default
assumption stays internal and the official compatibility remains unconfirmed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "cadvec.recognition.submission_check/1"
RESULT_FIELDS = (
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
THING_MIN = 0
THING_MAX = 29
STUFF_MIN = 30
STUFF_MAX = 34
IGNORE_ID = 35
COMMAND_LINE = 0
COMMAND_ARC = 1
COMMAND_CIRCLE = 2
CURVE_SHADOW_SCHEMA = "cadvec.recognition.curve_shadow/1"
CIRCLE_STATUS = "unimplemented"
CIRCLE_BLOCK_REASON = "png_frontend_line_only"
DEFAULT_RGB = (0, 0, 0)
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_LAYER_ID = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VIEWBOX = (0.0, 0.0, 100.0, 100.0)


class SubmissionError(ValueError):
    """A local contract violation; not an official platform result."""

    code = "submission_error"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _reject_constant(token: str) -> None:
    raise SubmissionError(f"non_finite_json:{token}")


def _no_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    document: Dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise SubmissionError(f"duplicate_json_key:{key}")
        document[key] = value
    return document


def loads_strict(text: str) -> Any:
    """Parse JSON while rejecting NaN/Infinity and duplicate object keys."""

    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"json_decode:{exc.msg}") from exc


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = loads_strict(line)
                except SubmissionError as exc:
                    raise SubmissionError(f"jsonl_line_{line_number}:{exc}") from exc
                if not isinstance(row, dict):
                    raise SubmissionError(f"jsonl_line_{line_number}:not_an_object")
                rows.append(row)
    except OSError as exc:
        raise SubmissionError(f"jsonl_read:{type(exc).__name__}:{exc}") from exc
    return rows


def png_dimensions(path: str) -> Tuple[int, int]:
    """Read width/height from the PNG IHDR without decoding pixels."""

    try:
        with open(path, "rb") as handle:
            data = handle.read(24)
    except OSError as exc:
        raise SubmissionError(f"png_read:{type(exc).__name__}:{exc}") from exc
    if len(data) < 24 or data[:8] != PNG_SIGNATURE:
        raise SubmissionError("png_header_invalid")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise SubmissionError("png_dimensions_invalid")
    return int(width), int(height)


def _normalize_viewbox(
    viewbox: Optional[Sequence[float]],
) -> Tuple[float, float, float, float]:
    """Return the supported viewBox or reject anything but 0 0 100 100."""

    if viewbox is None:
        return VIEWBOX
    if not isinstance(viewbox, (list, tuple)) or len(viewbox) != 4:
        raise SubmissionError("viewbox_invalid")
    try:
        values = tuple(float(value) for value in viewbox)
    except (TypeError, ValueError) as exc:
        raise SubmissionError("viewbox_invalid") from exc
    if any(not math.isfinite(value) for value in values):
        raise SubmissionError("viewbox_invalid")
    if values != VIEWBOX:
        rendered = ",".join(f"{value:g}" for value in values)
        raise SubmissionError(f"viewbox_unsupported:{rendered}")
    return values  # type: ignore[return-value]


def viewbox_transform(
    width: int,
    height: int,
    viewbox: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Return the PNG->viewBox uniform scale and letterbox offsets.

    The calibration is the nominal ``xMidYMid meet`` fit of the supported
    ``0 0 100 100`` viewBox, which all observed fit samples use.
    """

    if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
        raise SubmissionError("image_dimensions_invalid")
    _, _, box_width, box_height = _normalize_viewbox(viewbox)
    scale = min(width / box_width, height / box_height)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SubmissionError("image_scale_invalid")
    offset_x = (width - box_width * scale) / 2.0
    offset_y = (height - box_height * scale) / 2.0
    return scale, offset_x, offset_y


def _points(row: Mapping[str, Any]) -> List[Tuple[float, float]]:
    geometry = row.get("geometry_px")
    if not isinstance(geometry, list) or len(geometry) < 2:
        raise SubmissionError(f"geometry_missing:{row.get('candidate_uid')}")
    points: List[Tuple[float, float]] = []
    for point in geometry:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise SubmissionError(f"geometry_point_invalid:{row.get('candidate_uid')}")
        x, y = point
        if not _is_number(x) or not _is_number(y):
            raise SubmissionError(f"geometry_point_non_finite:{row.get('candidate_uid')}")
        points.append((float(x), float(y)))
    return points


def _four_samples(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Resample a polyline to four arc-length points (endpoints included)."""

    if len(points) < 2:
        raise SubmissionError("geometry_too_short")
    cumulative = [0.0]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.hypot(x2 - x1, y2 - y1))
    total = cumulative[-1]
    if not math.isfinite(total) or total <= 0.0:
        raise SubmissionError("geometry_zero_length")

    def at(distance: float) -> Tuple[float, float]:
        if distance <= 0.0:
            return points[0]
        if distance >= total:
            return points[-1]
        for index in range(len(points) - 1):
            start, end = cumulative[index], cumulative[index + 1]
            if distance <= end:
                span = end - start
                if span <= 0.0:
                    return points[index + 1]
                fraction = (distance - start) / span
                x1, y1 = points[index]
                x2, y2 = points[index + 1]
                return (x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction)
        return points[-1]

    return [at(total * fraction) for fraction in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)]


def _json_semantic(raw: Any) -> int:
    if not _is_int(raw) or not 1 <= raw <= 35:
        raise SubmissionError(f"semantic_raw_invalid:{raw}")
    return int(raw) - 1


def adapt_predictions(
    rows: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    rgb: Tuple[int, int, int] = DEFAULT_RGB,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    layer_id: int = DEFAULT_LAYER_ID,
    viewbox: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Convert PNG-only GNN prediction rows to the README §4 document.

    PNG pixel geometry is inverted to the source SVG viewBox before sampling;
    ``width``/``height`` describe that viewBox (100x100), not the PNG pixels.
    """

    box = _normalize_viewbox(viewbox)
    scale, offset_x, offset_y = viewbox_transform(width, height, box)
    commands: List[int] = []
    args: List[List[float]] = []
    lengths: List[float] = []
    semantic_ids: List[int] = []
    instance_ids: List[int] = []
    colours: List[List[int]] = []
    layer_ids: List[int] = []
    widths: List[float] = []

    first_sample_id: Optional[str] = None
    seen_uids: set = set()
    instance_order: List[int] = []
    group_to_instance: Dict[str, int] = {}
    instance_semantic: Dict[int, int] = {}
    instance_points: Dict[int, List[Tuple[float, float]]] = {}

    for row in rows:
        uid = row.get("candidate_uid")
        if not isinstance(uid, str) or not uid or uid in seen_uids:
            raise SubmissionError(f"candidate_uid_invalid:{uid}")
        seen_uids.add(uid)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SubmissionError(f"sample_id_invalid:{uid}")
        if row.get("source_kind") != "png_line":
            raise SubmissionError(f"source_kind_unsupported:{uid}:{row.get('source_kind')}")
        if first_sample_id is None:
            first_sample_id = sample_id
        elif sample_id != first_sample_id:
            raise SubmissionError("sample_id_mixed")

        semantic = _json_semantic(row.get("semantic_raw_pred"))
        points = [
            ((x - offset_x) / scale, (y - offset_y) / scale)
            for x, y in _points(row)
        ]
        samples = _four_samples(points)
        flat = [value for point in samples for value in point]
        total_length = 0.0
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            total_length += math.hypot(x2 - x1, y2 - y1)

        group = row.get("instance_group")
        if THING_MIN <= semantic <= THING_MAX:
            if not isinstance(group, str) or not group:
                raise SubmissionError(f"thing_instance_group_missing:{uid}")
            if group not in group_to_instance:
                instance_id = len(instance_order)
                group_to_instance[group] = instance_id
                instance_order.append(instance_id)
                instance_semantic[instance_id] = semantic
                instance_points[instance_id] = []
            else:
                instance_id = group_to_instance[group]
                if instance_semantic[instance_id] != semantic:
                    raise SubmissionError(f"instance_semantic_mismatch:{uid}")
            instance_points[instance_id].extend(samples)
            instance_ids.append(instance_id)
        else:
            if group is not None:
                raise SubmissionError(f"stuff_instance_group_present:{uid}")
            instance_ids.append(-1)

        commands.append(COMMAND_LINE)
        args.append(flat)
        lengths.append(total_length)
        semantic_ids.append(semantic)
        colours.append([int(rgb[0]), int(rgb[1]), int(rgb[2])])
        layer_ids.append(int(layer_id))
        widths.append(float(stroke_width))

    obj_cts: List[List[float]] = []
    boxes: List[List[float]] = []
    for instance_id in instance_order:
        points = instance_points.get(instance_id) or []
        if not points:
            raise SubmissionError(f"instance_without_points:{instance_id}")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        obj_cts.append([center_x, center_y, 0, instance_id])
        boxes.append([x_min, y_min, x_max, y_max, instance_semantic[instance_id]])

    return {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "width": int(box[2]),
        "height": int(box[3]),
        "obj_cts": obj_cts,
        "boxes": boxes,
        "rgb": colours,
        "layerIds": layer_ids,
        "widths": widths,
    }


@dataclass(frozen=True)
class CurveShadowConfig:
    """Strict gates for the opt-in shadow arc prototype.

    Arc recovery is disabled unless ``enable_arc`` is set *and* an explicit
    ``foreground_check`` callable is supplied.  Circle recovery has no switch:
    the PNG frontend is line-only, so ``command=2`` is never emitted here.
    """

    enable_arc: bool = False
    min_segments: int = 3
    join_tolerance: float = 1.0
    max_tangent_continuity_deg: float = 25.0
    max_tangent_deviation_deg: float = 15.0
    max_step_deg: float = 45.0
    min_arc_sweep_deg: float = 30.0
    max_arc_sweep_deg: float = 300.0
    min_radius: float = 1.0
    max_radius: float = 250.0
    max_fit_residual: float = 0.75
    max_residual_ratio: float = 0.05
    min_foreground_coverage: float = 0.85
    coverage_samples: int = 33


class _CurveGate(Exception):
    """Internal rejection carrying a stable gate name for diagnostics."""

    def __init__(self, gate: str, detail: Optional[Any] = None) -> None:
        super().__init__(gate)
        self.gate = gate
        self.detail = detail


ForegroundCheck = Callable[[Sequence[Tuple[float, float]]], float]


def _curve_config_error(config: CurveShadowConfig) -> Optional[str]:
    if config.min_segments < 3:
        return "min_segments"
    if not (math.isfinite(config.join_tolerance) and config.join_tolerance > 0.0):
        return "join_tolerance"
    for name, value in (
        ("max_tangent_continuity_deg", config.max_tangent_continuity_deg),
        ("max_tangent_deviation_deg", config.max_tangent_deviation_deg),
        ("max_step_deg", config.max_step_deg),
    ):
        if not (math.isfinite(value) and 0.0 < value <= 90.0):
            return name
    if not (
        math.isfinite(config.min_arc_sweep_deg)
        and math.isfinite(config.max_arc_sweep_deg)
        and 0.0 < config.min_arc_sweep_deg < config.max_arc_sweep_deg < 360.0
    ):
        return "arc_sweep_range"
    if not (
        math.isfinite(config.min_radius)
        and math.isfinite(config.max_radius)
        and 0.0 < config.min_radius < config.max_radius
    ):
        return "radius_range"
    if not (math.isfinite(config.max_fit_residual) and config.max_fit_residual >= 0.0):
        return "max_fit_residual"
    if not (math.isfinite(config.max_residual_ratio) and config.max_residual_ratio > 0.0):
        return "max_residual_ratio"
    if not (
        math.isfinite(config.min_foreground_coverage)
        and 0.0 <= config.min_foreground_coverage <= 1.0
    ):
        return "min_foreground_coverage"
    if config.coverage_samples < 2:
        return "coverage_samples"
    return None


def _distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _endpoint_gap(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> float:
    return min(
        _distance(left[0], right[0]),
        _distance(left[0], right[-1]),
        _distance(left[-1], right[0]),
        _distance(left[-1], right[-1]),
    )


def _direction(
    points: Sequence[Tuple[float, float]], *, from_end: bool
) -> Optional[Tuple[float, float]]:
    pairs = (
        list(zip(points, points[1:]))[::-1]
        if from_end
        else list(zip(points, points[1:]))
    )
    for start, end in pairs:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 1e-12:
            return (dx / length, dy / length)
    return None


def _fit_circle(
    points: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], float, float]:
    """Kasa algebraic circle fit; returns centre, radius and max radial error."""

    count = len(points)
    if count < 3:
        raise _CurveGate("fit_degenerate", {"points": count})
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    u = [point[0] - mean_x for point in points]
    v = [point[1] - mean_y for point in points]
    suu = sum(value * value for value in u)
    svv = sum(value * value for value in v)
    suv = sum(a * b for a, b in zip(u, v))
    suuu = sum(value ** 3 for value in u)
    svvv = sum(value ** 3 for value in v)
    suvv = sum(a * b * b for a, b in zip(u, v))
    svuu = sum(b * a * a for a, b in zip(u, v))
    determinant = suu * svv - suv * suv
    if abs(determinant) <= 1e-12 * max(1.0, abs(suu * svv)):
        raise _CurveGate("fit_degenerate", {"points": count})
    rhs_u = 0.5 * (suuu + suvv)
    rhs_v = 0.5 * (svvv + svuu)
    centre_u = (rhs_u * svv - rhs_v * suv) / determinant
    centre_v = (rhs_v * suu - rhs_u * suv) / determinant
    radius_sq = centre_u * centre_u + centre_v * centre_v + (suu + svv) / count
    if not math.isfinite(radius_sq) or radius_sq <= 0.0:
        raise _CurveGate("fit_degenerate", {"points": count})
    radius = math.sqrt(radius_sq)
    centre = (mean_x + centre_u, mean_y + centre_v)
    residual = max(
        abs(math.hypot(point[0] - centre[0], point[1] - centre[1]) - radius)
        for point in points
    )
    return centre, radius, residual


def _contiguous_components(
    indexes: Sequence[int],
    points: Sequence[Sequence[Tuple[float, float]]],
    tolerance: float,
) -> List[List[int]]:
    ordered = sorted(indexes)
    adjacency: Dict[int, List[int]] = {index: [] for index in ordered}
    for position, left in enumerate(ordered):
        for right in ordered[position + 1:]:
            if _endpoint_gap(points[left], points[right]) <= tolerance:
                adjacency[left].append(right)
                adjacency[right].append(left)
    unseen = set(ordered)
    components: List[List[int]] = []
    while unseen:
        start = min(unseen)
        unseen.discard(start)
        stack = [start]
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour in unseen:
                    unseen.discard(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _path_chain(
    component: Sequence[int],
    points: Sequence[Sequence[Tuple[float, float]]],
    adjacency: Mapping[int, Sequence[int]],
    config: CurveShadowConfig,
) -> List[Tuple[float, float]]:
    """Order one contiguous component into a single directed chain or reject."""

    if len(component) < config.min_segments:
        raise _CurveGate("too_few_segments", {"segments": len(component)})
    degrees = {index: len(adjacency[index]) for index in component}
    if any(degree > 2 for degree in degrees.values()):
        raise _CurveGate("ambiguous_chain", {"degree": max(degrees.values())})
    if all(degree == 2 for degree in degrees.values()):
        raise _CurveGate("closed_contour_circle_unimplemented")
    ends = sorted(index for index in component if degrees[index] == 1)
    if len(ends) != 2:
        raise _CurveGate("chain_not_path", {"endpoints": len(ends)})

    order: List[int] = []
    previous = -1
    current = ends[0]
    while True:
        order.append(current)
        following = [index for index in adjacency[current] if index != previous]
        if not following:
            break
        if len(following) > 1:
            raise _CurveGate("ambiguous_chain", {"branch_at": current})
        previous, current = current, following[0]
    if sorted(order) != sorted(component):
        raise _CurveGate("chain_not_path", {"ordered": len(order)})

    first_points = list(points[order[0]])
    if len(order) == 1:
        chain = first_points
    else:
        next_points = list(points[order[1]])

        def exit_gap(candidate: Sequence[Tuple[float, float]]) -> float:
            return min(
                _distance(candidate[-1], next_points[0]),
                _distance(candidate[-1], next_points[-1]),
            )

        forward = first_points
        reverse = list(reversed(first_points))
        chain = reverse if exit_gap(reverse) < exit_gap(forward) - 1e-12 else forward

    for index in order[1:]:
        segment = list(points[index])
        if _distance(chain[-1], segment[0]) > _distance(chain[-1], segment[-1]):
            segment.reverse()
        gap = _distance(chain[-1], segment[0])
        if gap > config.join_tolerance:
            raise _CurveGate("join_gap", {"gap": gap})
        previous_direction = _direction(chain, from_end=True)
        next_direction = _direction(segment, from_end=False)
        if previous_direction and next_direction:
            dot = max(
                -1.0,
                min(
                    1.0,
                    previous_direction[0] * next_direction[0]
                    + previous_direction[1] * next_direction[1],
                ),
            )
            angle = math.acos(dot)
            if angle > math.radians(config.max_tangent_continuity_deg):
                raise _CurveGate("tangent_continuity", {"angle_deg": math.degrees(angle)})
        start = 1 if gap <= 1e-9 else 0
        chain.extend(segment[start:])

    deduped: List[Tuple[float, float]] = []
    for point in chain:
        if not deduped or _distance(point, deduped[-1]) > 1e-9:
            deduped.append(point)
    if len(deduped) < config.min_segments + 1:
        raise _CurveGate("too_few_points", {"points": len(deduped)})
    return deduped


def _unwrap_angles(
    points: Sequence[Tuple[float, float]],
    centre: Tuple[float, float],
    config: CurveShadowConfig,
) -> Tuple[float, float]:
    angles = [
        math.atan2(point[1] - centre[1], point[0] - centre[0]) for point in points
    ]
    unwrapped = [angles[0]]
    max_step = math.radians(config.max_step_deg)
    for angle in angles[1:]:
        delta = (angle - unwrapped[-1] + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) > max_step:
            raise _CurveGate("angular_step", {"step_deg": math.degrees(abs(delta))})
        unwrapped.append(unwrapped[-1] + delta)
    deltas = [
        later - earlier for earlier, later in zip(unwrapped, unwrapped[1:])
    ]
    signs = {0 if abs(delta) <= 1e-9 else (1 if delta > 0.0 else -1) for delta in deltas}
    signs.discard(0)
    if len(signs) > 1:
        raise _CurveGate("sweep_direction")
    sweep = unwrapped[-1] - unwrapped[0]
    return unwrapped[0], sweep


def _arc_samples(
    centre: Tuple[float, float], radius: float, start: float, sweep: float, count: int = 4
) -> List[Tuple[float, float]]:
    return [
        (
            centre[0] + radius * math.cos(start + sweep * step / (count - 1)),
            centre[1] + radius * math.sin(start + sweep * step / (count - 1)),
        )
        for step in range(count)
    ]


def _baseline_primitives(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    primitives: List[Dict[str, Any]] = []
    for index, command in enumerate(document["commands"]):
        flat = document["args"][index]
        samples = [
            (float(flat[pair * 2]), float(flat[pair * 2 + 1])) for pair in range(4)
        ]
        primitives.append(
            {
                "command": int(command),
                "samples": samples,
                "length": float(document["lengths"][index]),
                "semantic": int(document["semanticIds"][index]),
                "instance": int(document["instanceIds"][index]),
                "rgb": list(document["rgb"][index]),
                "layer": int(document["layerIds"][index]),
                "width": float(document["widths"][index]),
            }
        )
    return primitives


def _document_from_primitives(
    primitives: Sequence[Mapping[str, Any]], width: int, height: int
) -> Dict[str, Any]:
    commands: List[int] = []
    args: List[List[float]] = []
    lengths: List[float] = []
    semantic_ids: List[int] = []
    instance_ids: List[int] = []
    colours: List[List[int]] = []
    layer_ids: List[int] = []
    widths: List[float] = []
    instance_order: List[int] = []
    instance_semantic: Dict[int, int] = {}
    instance_points: Dict[int, List[Tuple[float, float]]] = {}

    for primitive in primitives:
        samples = list(primitive["samples"])
        semantic = int(primitive["semantic"])
        instance = int(primitive["instance"])
        commands.append(int(primitive["command"]))
        args.append([value for point in samples for value in point])
        lengths.append(float(primitive["length"]))
        semantic_ids.append(semantic)
        instance_ids.append(instance)
        colours.append([int(channel) for channel in primitive["rgb"]])
        layer_ids.append(int(primitive["layer"]))
        widths.append(float(primitive["width"]))
        if THING_MIN <= semantic <= THING_MAX:
            if instance < 0:
                raise SubmissionError("candidate_thing_instance_negative")
            if instance not in instance_points:
                instance_order.append(instance)
                instance_semantic[instance] = semantic
                instance_points[instance] = []
            elif instance_semantic[instance] != semantic:
                raise SubmissionError(f"candidate_instance_semantic_mismatch:{instance}")
            instance_points[instance].extend(samples)
        elif instance != -1:
            raise SubmissionError("candidate_stuff_instance_present")

    obj_cts: List[List[float]] = []
    boxes: List[List[float]] = []
    for instance in instance_order:
        points = instance_points[instance]
        if not points:
            raise SubmissionError(f"candidate_instance_without_points:{instance}")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        obj_cts.append([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0, instance])
        boxes.append([x_min, y_min, x_max, y_max, instance_semantic[instance]])

    return {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "width": int(width),
        "height": int(height),
        "obj_cts": obj_cts,
        "boxes": boxes,
        "rgb": colours,
        "layerIds": layer_ids,
        "widths": widths,
    }


def _shadow_bundle(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    status: str,
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": CURVE_SHADOW_SCHEMA,
        "version": 1,
        "status": status,
        "mode": "shadow",
        "official_output": "baseline_unchanged",
        "changed": candidate != baseline,
        "baseline": baseline,
        "candidate": candidate,
        "diagnostics": diagnostics,
    }


def shadow_postprocess(
    rows: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    viewbox: Optional[Sequence[float]] = None,
    config: Optional[CurveShadowConfig] = None,
    foreground_check: Optional[ForegroundCheck] = None,
    rgb: Tuple[int, int, int] = DEFAULT_RGB,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    layer_id: int = DEFAULT_LAYER_ID,
) -> Dict[str, Any]:
    """Return ``{baseline, candidate, diagnostics}`` for opt-in curve recovery.

    The baseline is produced by :func:`adapt_predictions` and is never mutated.
    ``foreground_check`` receives sampled points in the original PNG pixel
    coordinate system and must return foreground coverage in ``[0, 1]``.
    Without it the arc path is blocked, so the candidate equals the baseline.
    """

    active = config or CurveShadowConfig()
    config_error = _curve_config_error(active)
    if config_error is not None:
        raise SubmissionError(f"curve_config_invalid:{config_error}")

    baseline = adapt_predictions(
        rows,
        width,
        height,
        rgb=rgb,
        stroke_width=stroke_width,
        layer_id=layer_id,
        viewbox=viewbox,
    )
    diagnostics: Dict[str, Any] = {
        "gates": asdict(active),
        "arc": {
            "status": "disabled" if not active.enable_arc else "blocked",
            "emitted": 0,
            "groups": [],
            "rejections": [],
        },
        "circle": {
            "status": CIRCLE_STATUS,
            "reason": CIRCLE_BLOCK_REASON,
            "emitted": 0,
        },
    }

    if not active.enable_arc:
        diagnostics["arc"]["reason"] = "arc_disabled"
        return _shadow_bundle(baseline, baseline, "disabled", diagnostics)
    if foreground_check is None:
        diagnostics["arc"]["reason"] = "foreground_coverage_unavailable"
        return _shadow_bundle(baseline, baseline, "blocked", diagnostics)

    box = _normalize_viewbox(viewbox)
    scale, offset_x, offset_y = viewbox_transform(width, height, box)
    points_by_row: List[List[Tuple[float, float]]] = [
        [((x - offset_x) / scale, (y - offset_y) / scale) for x, y in _points(row)]
        for row in rows
    ]
    baseline_primitives = _baseline_primitives(baseline)

    grouped: Dict[Tuple[str, int, str], List[int]] = {}
    for index, row in enumerate(rows):
        semantic = int(baseline["semanticIds"][index])
        group = row.get("instance_group")
        if (
            THING_MIN <= semantic <= THING_MAX
            and isinstance(group, str)
            and group
        ):
            key = (str(row["sample_id"]), int(row["semantic_raw_pred"]), group)
            grouped.setdefault(key, []).append(index)

    replacements: Dict[int, Dict[str, Any]] = {}
    replaced_members: Dict[int, List[int]] = {}
    for key, indexes in grouped.items():
        if len(indexes) < active.min_segments:
            continue
        adjacency: Dict[int, List[int]] = {index: [] for index in indexes}
        for position, left in enumerate(sorted(indexes)):
            for right in sorted(indexes)[position + 1:]:
                if _endpoint_gap(points_by_row[left], points_by_row[right]) <= active.join_tolerance:
                    adjacency[left].append(right)
                    adjacency[right].append(left)
        group_report: Dict[str, Any] = {
            "sample_id": key[0],
            "semantic_raw": key[1],
            "instance_group": key[2],
            "candidate_uids": [rows[index]["candidate_uid"] for index in indexes],
            "arcs": [],
        }
        for component in _contiguous_components(indexes, points_by_row, active.join_tolerance):
            if len(component) < 2:
                continue
            try:
                chain = _path_chain(component, points_by_row, adjacency, active)
                centre, radius, residual = _fit_circle(chain)
                if not (
                    math.isfinite(radius)
                    and active.min_radius <= radius <= active.max_radius
                ):
                    raise _CurveGate("radius_limit", {"radius": radius})
                if residual > active.max_fit_residual or residual > (
                    active.max_residual_ratio * radius
                ):
                    raise _CurveGate(
                        "fit_residual_limit",
                        {
                            "residual": residual,
                            "radius": radius,
                            "abs_limit": active.max_fit_residual,
                            "ratio_limit": active.max_residual_ratio * radius,
                        },
                    )
                start, sweep = _unwrap_angles(chain, centre, active)
                sweep_deg = math.degrees(abs(sweep))
                if sweep_deg >= 360.0 - 1e-9:
                    raise _CurveGate("closed_arc_circle_blocked", {"sweep_deg": sweep_deg})
                if not (
                    active.min_arc_sweep_deg
                    <= sweep_deg
                    <= active.max_arc_sweep_deg
                ):
                    raise _CurveGate("arc_sweep_limit", {"sweep_deg": sweep_deg})
                for first, second in zip(chain, chain[1:]):
                    dx, dy = second[0] - first[0], second[1] - first[1]
                    chord = math.hypot(dx, dy)
                    if chord <= 1e-12:
                        continue
                    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
                    radial = math.atan2(middle[1] - centre[1], middle[0] - centre[0])
                    chord_direction = (dx / chord, dy / chord)
                    tangent = (-math.sin(radial), math.cos(radial))
                    dot = abs(
                        chord_direction[0] * tangent[0]
                        + chord_direction[1] * tangent[1]
                    )
                    deviation = math.degrees(math.acos(max(0.0, min(1.0, dot))))
                    if deviation > active.max_tangent_deviation_deg:
                        raise _CurveGate(
                            "tangent_deviation",
                            {"deviation_deg": deviation, "radius": radius},
                        )
                coverage_points = _arc_samples(
                    centre,
                    radius,
                    start,
                    sweep,
                    active.coverage_samples,
                )
                pixel_points = [
                    (point[0] * scale + offset_x, point[1] * scale + offset_y)
                    for point in coverage_points
                ]
                try:
                    coverage_value = foreground_check(pixel_points)
                except Exception as exc:  # pragma: no cover - defensive path
                    raise _CurveGate(
                        "foreground_coverage_error", {"error": type(exc).__name__}
                    ) from exc
                if not _is_number(coverage_value) or not (
                    0.0 <= float(coverage_value) <= 1.0
                ):
                    raise _CurveGate(
                        "foreground_coverage_invalid", {"coverage": coverage_value}
                    )
                coverage = float(coverage_value)
                if coverage < active.min_foreground_coverage:
                    raise _CurveGate(
                        "foreground_coverage_low",
                        {
                            "coverage": coverage,
                            "minimum": active.min_foreground_coverage,
                        },
                    )
                samples = _arc_samples(centre, radius, start, sweep, 4)
                first_index = min(component)
                replacements[first_index] = {
                    "command": COMMAND_ARC,
                    "samples": samples,
                    "length": radius * abs(sweep),
                    "semantic": int(baseline["semanticIds"][first_index]),
                    "instance": int(baseline["instanceIds"][first_index]),
                    "rgb": list(baseline["rgb"][first_index]),
                    "layer": int(baseline["layerIds"][first_index]),
                    "width": float(baseline["widths"][first_index]),
                }
                replaced_members[first_index] = list(component)
                group_report["arcs"].append(
                    {
                        "candidate_uids": [
                            rows[index]["candidate_uid"] for index in component
                        ],
                        "centre": [centre[0], centre[1]],
                        "radius": radius,
                        "residual": residual,
                        "sweep_deg": sweep_deg,
                        "arc_length": radius * abs(sweep),
                        "foreground_coverage": coverage,
                    }
                )
            except _CurveGate as exc:
                diagnostics["arc"]["rejections"].append(
                    {
                        "sample_id": key[0],
                        "semantic_raw": key[1],
                        "instance_group": key[2],
                        "candidate_uids": [
                            rows[index]["candidate_uid"] for index in component
                        ],
                        "gate": exc.gate,
                        "detail": exc.detail,
                    }
                )
        diagnostics["arc"]["groups"].append(group_report)

    if not replacements:
        diagnostics["arc"]["status"] = "noop"
        diagnostics["arc"]["reason"] = "no_arc_candidate_accepted"
        return _shadow_bundle(baseline, baseline, "noop", diagnostics)

    consumed = {
        member
        for members in replaced_members.values()
        for member in members
        if member not in replacements
    }
    primitives: List[Mapping[str, Any]] = []
    for index in range(len(rows)):
        if index in replacements:
            primitives.append(replacements[index])
        elif index in consumed:
            continue
        else:
            primitives.append(baseline_primitives[index])
    try:
        candidate = _document_from_primitives(
            primitives, baseline["width"], baseline["height"]
        )
        candidate_errors = validate_document(candidate)
    except SubmissionError as exc:
        candidate_errors = [str(exc)]
        candidate = baseline
    if candidate_errors:
        diagnostics["arc"]["status"] = "candidate_invalid"
        diagnostics["arc"]["errors"] = candidate_errors
        return _shadow_bundle(baseline, baseline, "candidate_invalid", diagnostics)

    arc_count = sum(1 for command in candidate["commands"] if command == COMMAND_ARC)
    diagnostics["arc"]["emitted"] = arc_count
    diagnostics["arc"]["status"] = "accepted" if arc_count else "noop"
    return _shadow_bundle(
        baseline,
        candidate,
        "candidate" if candidate != baseline else "noop",
        diagnostics,
    )


def shadow_postprocess_file(
    predictions_path: str,
    png_path: str,
    out_path: str,
    *,
    enable_arc: bool = False,
    config: Optional[CurveShadowConfig] = None,
    foreground_check: Optional[ForegroundCheck] = None,
) -> Dict[str, Any]:
    """Read one PNG-only JSONL and write the shadow bundle (baseline+candidate)."""

    rows = _read_jsonl(predictions_path)
    width, height = png_dimensions(png_path)
    active = config or CurveShadowConfig(enable_arc=enable_arc)
    bundle = shadow_postprocess(
        rows, width, height, config=active, foreground_check=foreground_check
    )
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return bundle


def validate_document(document: Any) -> List[str]:
    """Return local contract errors for one README §4 result document."""

    errors: List[str] = []
    if not isinstance(document, Mapping):
        return ["document_not_object"]

    missing = [field for field in RESULT_FIELDS if field not in document]
    extra = [field for field in document if field not in RESULT_FIELDS]
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    if extra:
        errors.append("unknown_fields:" + ",".join(sorted(str(field) for field in extra)))
    if missing:
        return errors

    arrays = {}
    for field in (
        "commands",
        "args",
        "lengths",
        "semanticIds",
        "instanceIds",
        "rgb",
        "layerIds",
        "widths",
    ):
        value = document.get(field)
        if not isinstance(value, list):
            errors.append(f"{field}_not_array")
        else:
            arrays[field] = value
    if len(arrays) != 8:
        return errors

    n = len(arrays["commands"])
    for field, value in arrays.items():
        if len(value) != n:
            errors.append(f"{field}_length_mismatch:{len(value)}!={n}")

    width, height = document.get("width"), document.get("height")
    if not _is_int(width) or width <= 0:
        errors.append("width_invalid")
    if not _is_int(height) or height <= 0:
        errors.append("height_invalid")

    for index, value in enumerate(arrays["commands"]):
        if not _is_int(value) or not 0 <= value <= 3:
            errors.append(f"commands_invalid:{index}")
    for index, value in enumerate(arrays["lengths"]):
        if not _is_number(value) or float(value) < 0.0:
            errors.append(f"lengths_invalid:{index}")
    for index, value in enumerate(arrays["semanticIds"]):
        if not _is_int(value) or not 0 <= value <= 35:
            errors.append(f"semanticIds_invalid:{index}")
    for index, value in enumerate(arrays["instanceIds"]):
        if not _is_int(value) or value < -1:
            errors.append(f"instanceIds_invalid:{index}")
    for index, value in enumerate(arrays["rgb"]):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(not _is_int(channel) or not 0 <= channel <= 255 for channel in value)
        ):
            errors.append(f"rgb_invalid:{index}")
    for index, value in enumerate(arrays["layerIds"]):
        if not _is_int(value) or value < 1:
            errors.append(f"layerIds_invalid:{index}")
    for index, value in enumerate(arrays["widths"]):
        if not _is_number(value) or float(value) < 0.0:
            errors.append(f"widths_invalid:{index}")
    for index, value in enumerate(arrays["args"]):
        if not isinstance(value, (list, tuple)) or len(value) != 8:
            errors.append(f"args_shape_invalid:{index}")
            continue
        if any(not _is_number(item) for item in value):
            errors.append(f"args_non_finite:{index}")

    for index, (semantic, instance) in enumerate(
        zip(arrays["semanticIds"], arrays["instanceIds"])
    ):
        if not _is_int(semantic) or not _is_int(instance):
            continue
        if THING_MIN <= semantic <= THING_MAX:
            if instance < 0:
                errors.append(f"thing_instance_negative:{index}")
        elif STUFF_MIN <= semantic <= STUFF_MAX or semantic == IGNORE_ID:
            if instance != -1:
                errors.append(f"stuff_instance_not_minus_one:{index}")

    obj_cts, boxes = document.get("obj_cts"), document.get("boxes")
    if not isinstance(obj_cts, list):
        errors.append("obj_cts_not_array")
    if not isinstance(boxes, list):
        errors.append("boxes_not_array")
    if not isinstance(obj_cts, list) or not isinstance(boxes, list):
        return errors
    if len(obj_cts) != len(boxes):
        errors.append(f"obj_cts_boxes_length_mismatch:{len(obj_cts)}!={len(boxes)}")

    instance_order: List[int] = []
    instance_semantic: Dict[int, int] = {}
    instance_members: Dict[int, List[int]] = {}
    for index, (semantic, instance) in enumerate(
        zip(arrays["semanticIds"], arrays["instanceIds"])
    ):
        if not _is_int(semantic) or not _is_int(instance):
            continue
        if THING_MIN <= semantic <= THING_MAX and instance >= 0:
            if instance not in instance_semantic:
                instance_order.append(instance)
                instance_semantic[instance] = semantic
                instance_members[instance] = []
            elif instance_semantic[instance] != semantic:
                errors.append(f"instance_semantic_mismatch:{instance}")
            instance_members[instance].append(index)

    if len(obj_cts) != len(instance_order):
        errors.append(f"obj_cts_instance_count_mismatch:{len(obj_cts)}!={len(instance_order)}")
    if len(boxes) != len(instance_order):
        errors.append(f"boxes_instance_count_mismatch:{len(boxes)}!={len(instance_order)}")

    for position, instance in enumerate(instance_order):
        if position >= len(obj_cts) or position >= len(boxes):
            break
        center = obj_cts[position]
        box = boxes[position]
        if not isinstance(center, (list, tuple)) or len(center) != 4:
            errors.append(f"obj_cts_shape_invalid:{position}")
            continue
        if not isinstance(box, (list, tuple)) or len(box) != 5:
            errors.append(f"boxes_shape_invalid:{position}")
            continue
        if any(not _is_number(value) for value in center):
            errors.append(f"obj_cts_non_finite:{position}")
            continue
        if any(not _is_number(value) for value in box):
            errors.append(f"boxes_non_finite:{position}")
            continue
        if not _is_int(center[3]) or int(center[3]) != instance:
            errors.append(f"obj_cts_instance_order_mismatch:{position}")
        if float(center[2]) != 0.0:
            errors.append(f"obj_cts_third_value_not_zero:{position}")
        if not _is_int(box[4]) or int(box[4]) != instance_semantic[instance]:
            errors.append(f"boxes_semantic_mismatch:{position}")

        member_indexes = instance_members.get(instance) or []
        xs: List[float] = []
        ys: List[float] = []
        for member in member_indexes:
            if member >= len(arrays["args"]):
                continue
            flat = arrays["args"][member]
            if not isinstance(flat, (list, tuple)) or len(flat) != 8:
                continue
            for pair in range(4):
                x, y = flat[pair * 2], flat[pair * 2 + 1]
                if _is_number(x) and _is_number(y):
                    xs.append(float(x))
                    ys.append(float(y))
        if not xs or not ys:
            errors.append(f"instance_without_finite_points:{instance}")
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        expected_box = (x_min, y_min, x_max, y_max)
        expected_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        actual_box = tuple(float(value) for value in box[:4])
        actual_center = (float(center[0]), float(center[1]))
        if any(not _close(left, right) for left, right in zip(actual_box, expected_box)):
            errors.append(f"boxes_geometry_mismatch:{position}")
        if any(not _close(left, right) for left, right in zip(actual_center, expected_center)):
            errors.append(f"obj_cts_geometry_mismatch:{position}")

    return errors


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-6 * max(1.0, abs(float(right)))


def adapt_predictions_file(predictions_path: str, png_path: str, out_path: str) -> Dict[str, Any]:
    """Read one PNG-only JSONL, emit and validate one README-shaped JSON."""

    rows = _read_jsonl(predictions_path)
    width, height = png_dimensions(png_path)
    document = adapt_predictions(rows, width, height)
    errors = validate_document(document)
    if errors:
        raise SubmissionError(";".join(errors))
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return document


def check_results_dir(results_dir: str, png_dir: str) -> Dict[str, Any]:
    """Check one JSON per PNG, strict parsing and all document constraints."""

    if not os.path.isdir(results_dir):
        raise SubmissionError(f"results_dir_missing:{results_dir}")
    if not os.path.isdir(png_dir):
        raise SubmissionError(f"png_dir_missing:{png_dir}")

    pngs: Dict[str, str] = {}
    for name in sorted(os.listdir(png_dir)):
        if name.startswith("._") or name == ".DS_Store":
            continue
        path = os.path.join(png_dir, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() != ".png":
            continue
        stem = os.path.splitext(name)[0]
        if stem in pngs:
            raise SubmissionError(f"duplicate_png_stem:{stem}")
        pngs[stem] = path

    jsons: Dict[str, str] = {}
    non_json: List[str] = []
    for name in sorted(os.listdir(results_dir)):
        if name.startswith("._") or name == ".DS_Store":
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() != ".json":
            non_json.append(name)
            continue
        stem = os.path.splitext(name)[0]
        if stem in jsons:
            raise SubmissionError(f"duplicate_json_stem:{stem}")
        jsons[stem] = path

    missing = sorted(set(pngs) - set(jsons))
    extra = sorted(set(jsons) - set(pngs))
    errors: List[str] = []
    if non_json:
        errors.append("results_non_json:" + ",".join(non_json))
    if missing:
        errors.append("missing_json:" + ",".join(missing))
    if extra:
        errors.append("extra_json:" + ",".join(extra))

    documents = 0
    for stem in sorted(set(jsons) & set(pngs)):
        try:
            with open(jsons[stem], "r", encoding="utf-8") as handle:
                document = json.load(
                    handle,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_no_duplicate_keys,
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, SubmissionError) as exc:
            errors.append(f"json_read:{stem}:{type(exc).__name__}:{exc}")
            continue
        documents += 1
        for error in validate_document(document):
            errors.append(f"{stem}:{error}")

    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "ok" if not errors else "failed",
        "png_count": len(pngs),
        "json_count": len(jsons),
        "documents_checked": documents,
        "missing": missing,
        "extra": extra,
        "errors": errors,
    }


def predict_submission(checkpoint: str, png: str, out_path: str) -> Dict[str, Any]:
    """Run the existing PNG-only GNN path and emit one README-shaped JSON."""

    from . import recognition_gnn_train as trainer

    with tempfile.TemporaryDirectory(prefix="cadvec-submission-") as tmp:
        predictions_path = os.path.join(tmp, "predictions.jsonl")
        summary = trainer.predict_png(checkpoint, png, predictions_path)
        if summary.get("status") != "ok":
            raise SubmissionError(f"predict_png_status:{summary.get('status')}")
        document = adapt_predictions_file(predictions_path, png, out_path)
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "ok",
        "result": out_path,
        "prediction_count": len(document["commands"]),
        "instance_count": len(document["obj_cts"]),
        "model_id": summary.get("model_id"),
        "checkpoint": checkpoint,
        "checkpoint_scope": "fit_tune_not_frozen",
        "official_compatibility": "unconfirmed",
        "contract": "datasets/README.md section 4",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cadvec.recognition_submission")
    sub = parser.add_subparsers(dest="command", required=True)

    adapt = sub.add_parser("adapt", help="convert one PNG-only JSONL to one README JSON")
    adapt.add_argument("--predictions", required=True)
    adapt.add_argument("--png", required=True)
    adapt.add_argument("--out", required=True)

    check = sub.add_parser("check", help="check a results/ directory against PNG names")
    check.add_argument("--results-dir", required=True)
    check.add_argument("--png-dir", required=True)

    shadow = sub.add_parser(
        "shadow",
        help="emit baseline+candidate curve-shadow bundle; baseline stays official",
    )
    shadow.add_argument("--predictions", required=True)
    shadow.add_argument("--png", required=True)
    shadow.add_argument("--out", required=True)
    shadow.add_argument(
        "--enable-arc",
        action="store_true",
        help="opt in to arc recovery; without foreground evidence it stays blocked",
    )

    predict = sub.add_parser("predict", help="run predict-png and emit one README JSON")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--png", required=True)
    predict.add_argument("--out", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "adapt":
            document = adapt_predictions_file(args.predictions, args.png, args.out)
            output = {
                "schema": SCHEMA,
                "version": 1,
                "status": "ok",
                "result": args.out,
                "prediction_count": len(document["commands"]),
                "instance_count": len(document["obj_cts"]),
                "official_compatibility": "unconfirmed",
            }
            code = 0
        elif args.command == "check":
            output = check_results_dir(args.results_dir, args.png_dir)
            code = 0 if output["status"] == "ok" else 1
        elif args.command == "shadow":
            bundle = shadow_postprocess_file(
                args.predictions,
                args.png,
                args.out,
                enable_arc=bool(args.enable_arc),
            )
            output = {
                "schema": SCHEMA,
                "version": 1,
                "status": bundle["status"],
                "result": args.out,
                "changed": bundle["changed"],
                "baseline_prediction_count": len(bundle["baseline"]["commands"]),
                "candidate_prediction_count": len(bundle["candidate"]["commands"]),
                "arc": bundle["diagnostics"]["arc"],
                "circle": bundle["diagnostics"]["circle"],
                "official_compatibility": "unconfirmed",
            }
            code = 0
        else:
            output = predict_submission(args.checkpoint, args.png, args.out)
            code = 0
    except SubmissionError as exc:
        output = {
            "schema": SCHEMA,
            "version": 1,
            "status": "failed",
            "errors": [str(exc)],
        }
        code = 2
    except Exception as exc:
        output = {
            "schema": SCHEMA,
            "version": 1,
            "status": "failed",
            "errors": [f"{type(exc).__name__}:{exc}"],
        }
        code = 2

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
