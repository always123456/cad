"""Pure geometric evaluator for PNG line-recovery diagnostics.

Scope and contract
------------------
This module is deliberately geometry-only and side-effect free.  It accepts
explicit, finite pixel polylines (or segments) and never opens SVG, PNG,
label files, manifests or any path.  It knows nothing about the platform,
semantic/instance IDs, models, or PQ, and it produces no score thresholds.

Two geometries are compared: a reference (GT) side and a prediction side.

* Source positive-length segments are sampled at the midpoint of equal
  arc-length intervals.  The interval length is the sample weight, so all
  weights on one side sum to that side's drawn length.  Default maximum
  interval is 1 px; a side is capped at ``max_samples`` query points.
* When the requested step would exceed the cap, one uniform effective step is
  chosen (per side) so every positive segment keeps at least one interval and
  the whole drawn length is still represented.  The chosen step and whether
  it was raised are recorded as ``requested_step`` / ``effective_step`` /
  ``sampling_limited``.
* If there are more positive-length segments than ``max_samples``, at least
  one interval per segment is impossible: the pair is returned as
  ``evaluation_limited`` with lengths preserved and no coverage or distance.
* Distance is the minimum truncated-projection Euclidean distance from a
  query point to the target's continuous segments, never to a four-point
  summary.  Disconnected polylines are never bridged.
* Coverage for tolerance ``t`` is
  ``C(A->B,t) = sum(w_i * I[d_i <= t]) / sum(w_i)``; tolerances default to
  1, 2 and 4 px.  Each direction also reports arc-length weighted mean,
  median and P95 distance.  A weighted quantile is the smallest distance at
  which cumulative weight reaches the target fraction.
* Duplicate segments are an independent, tolerance-limited proxy: prediction
  endpoints are snapped to a 0.5 px grid, endpoint order is normalised, and
  extra segments per normalised key are counted.  The original prediction is
  never merged; this is not a full partial-overlap or collinear-fragmentation
  solver.
* Empty supports follow the fixed contract: GT non-empty/prediction empty
  gives GT->prediction coverage 0 and the reverse side null; the mirror case
  is symmetric; both empty is ``empty_pair`` with all coverage/distance null.
  Zero drawn length counts as empty.
* Non-finite coordinates or configuration raise :class:`EvaluationInputError`
  before any metric is produced.  Arithmetic overflow discovered while
  measuring returns ``status="failed"`` with a reason, never NaN/Inf and
  never a fake success.  Every returned document is strict-JSON serialisable
  with ``allow_nan=False``.

Aggregation (``summarise``) reports selected/eligible/excluded/failed/
alignment_pending/evaluation_limited/empty_pair counts, per-direction valid
image counts, support/hit lengths, no-target support length, macro and micro
coverage, and duplicate totals.  Empty predictions keep their zero recall in
both macro and micro coverage.  Per-image weighted quantiles are reported,
but no global quantile is fabricated from them: the summary only combines
weighted means (support-weighted) and explicitly says so.

A summary carries the shared ``support_scope``, per-direction metric names and
the requested configuration (step, tolerance set, duplicate quantum).  Mixing
different scopes (overall coverage vs straight-only affinity), different
tolerance sets or different requested steps is rejected instead of silently
combining denominators; callers summarise each group separately.  Sampling
budget or a raised ``effective_step`` may differ per image.  Plain status
records such as ``{"status": "failed"}`` need no fabricated metric fields.
Numeric aggregation overflow raises :class:`EvaluationInputError` rather than
emitting NaN/Inf.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "cadvec.geometric_eval/1"
SCHEMA_VERSION = 1
INPUT_SCHEMA = "cadvec.geometric_eval.input/1"

DEFAULT_STEP_PX = 1.0
DEFAULT_MAX_SAMPLES = 20000
DEFAULT_TOLERANCES: Tuple[float, float, float] = (1.0, 2.0, 4.0)
DEFAULT_DUPLICATE_QUANTUM = 0.5

SUPPORT_ALL_STROKES = "all_strokes"
SUPPORT_STRAIGHT_ONLY = "straight_only"
SUPPORT_SCOPES = (SUPPORT_ALL_STROKES, SUPPORT_STRAIGHT_ONLY)

STATUS_OK = "ok"
STATUS_EMPTY_PAIR = "empty_pair"
STATUS_EVALUATION_LIMITED = "evaluation_limited"
STATUS_FAILED = "failed"
STATUS_EXCLUDED = "excluded"
STATUS_ALIGNMENT_PENDING = "alignment_pending"

SUMMARY_STATUSES = (
    STATUS_OK,
    STATUS_EMPTY_PAIR,
    STATUS_EVALUATION_LIMITED,
    STATUS_FAILED,
    STATUS_EXCLUDED,
    STATUS_ALIGNMENT_PENDING,
)

CATEGORY_STRAIGHT_ONLY = "straight_only"
CATEGORY_CURVED_OR_MIXED = "curved_or_mixed"
CATEGORY_CIRCLE = "circle"
CATEGORY_ELLIPSE = "ellipse"
CATEGORY_UNSUPPORTED = "unsupported"

_CURVE_COMMANDS = frozenset("CSQTA")
_STRAIGHT_COMMANDS = frozenset("MLHVZ")
_KNOWN_COMMANDS = _CURVE_COMMANDS | _STRAIGHT_COMMANDS
_TOKEN_RE = re.compile(
    r"[A-Za-z]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)

Point = Tuple[float, float]


class EvaluationInputError(ValueError):
    """Invalid or non-finite input geometry or configuration."""


class _NumericFailure(Exception):
    """Arithmetic on accepted finite inputs produced a non-finite value."""


class _SamplingImpossible(Exception):
    """Not enough sample budget to give every positive segment one interval."""


def _finite(value: float, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvaluationInputError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationInputError(f"{what} must be finite, got {value!r}")
    return number


def _tolerance_key(tolerance: float) -> str:
    return format(tolerance, "g")


@dataclass(frozen=True)
class PolylineSet:
    """Immutable polyline collection; each polyline is an independent stroke."""

    polylines: Tuple[Tuple[Point, ...], ...]

    @classmethod
    def from_polylines(cls, polylines: Iterable[Iterable[Sequence[float]]]) -> "PolylineSet":
        cleaned: List[Tuple[Point, ...]] = []
        for polyline in polylines:
            points: List[Point] = []
            for index, point in enumerate(polyline):
                if len(point) != 2:
                    raise EvaluationInputError(
                        f"polyline point #{index} must be (x, y), got {point!r}"
                    )
                points.append(
                    (_finite(point[0], "x"), _finite(point[1], "y"))
                )
            cleaned.append(tuple(points))
        return cls(tuple(cleaned))

    @classmethod
    def from_segments(cls, segments: Iterable[Sequence[float]]) -> "PolylineSet":
        polylines = []
        for index, segment in enumerate(segments):
            if len(segment) != 4:
                raise EvaluationInputError(
                    f"segment #{index} must be (x1, y1, x2, y2), got {segment!r}"
                )
            polylines.append(
                [
                    (segment[0], segment[1]),
                    (segment[2], segment[3]),
                ]
            )
        return cls.from_polylines(polylines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polyline_count": len(self.polylines),
            "point_count": sum(len(item) for item in self.polylines),
        }


def as_polyline_set(value: Any) -> PolylineSet:
    """Accept a :class:`PolylineSet` or an explicit list of polylines."""

    if isinstance(value, PolylineSet):
        return value
    try:
        return PolylineSet.from_polylines(value)
    except TypeError as exc:
        raise EvaluationInputError(f"geometry must be polylines: {exc}") from exc


@dataclass(frozen=True)
class _Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float


def _segments_of(geometry: PolylineSet) -> List[_Segment]:
    segments: List[_Segment] = []
    for polyline in geometry.polylines:
        for index in range(1, len(polyline)):
            x1, y1 = polyline[index - 1]
            x2, y2 = polyline[index]
            length = math.hypot(x2 - x1, y2 - y1)
            if not math.isfinite(length):
                raise _NumericFailure("segment length overflowed float64")
            if length > 0.0:
                segments.append(_Segment(x1, y1, x2, y2, length))
    return segments


def _total_length(segments: Sequence[_Segment]) -> float:
    total = 0.0
    for segment in segments:
        total += segment.length
        if not math.isfinite(total):
            raise _NumericFailure("drawn length sum overflowed float64")
    return total


def _interval_counts(segments: Sequence[_Segment], step: float) -> List[int]:
    counts: List[int] = []
    for segment in segments:
        ratio = segment.length / step
        if not math.isfinite(ratio):
            raise _NumericFailure("sampling interval ratio overflowed float64")
        counts.append(max(1, int(math.ceil(ratio))))
    return counts


def _plan_intervals(
    segments: Sequence[_Segment], step: float, max_samples: int
) -> Tuple[List[int], float, bool]:
    """Return per-segment interval counts, effective step and limited flag."""

    if len(segments) > max_samples:
        raise _SamplingImpossible(
            f"{len(segments)} positive segments exceed max_samples={max_samples}"
        )

    counts = _interval_counts(segments, step)
    if sum(counts) <= max_samples:
        return counts, step, False

    # Uniform target step: keep one interval per positive segment and keep the
    # total interval count within budget.  ``sum(counts) == len(segments)`` at
    # the upper bound, so the search is always feasible.
    low = step
    high = max(segment.length for segment in segments)
    for _ in range(64):
        # Overflow-safe midpoint: ``(low + high) / 2`` can reach inf when both
        # lengths are near 1e308, which previously produced effective_step=inf.
        mid = low + (high - low) / 2.0
        if not math.isfinite(mid):
            raise _NumericFailure("sampling step search overflowed float64")
        candidate = _interval_counts(segments, mid)
        if sum(candidate) <= max_samples:
            high = mid
        else:
            low = mid
    effective = high
    if not math.isfinite(effective) or effective <= 0.0:
        raise _NumericFailure("effective sampling step is not finite")
    counts = _interval_counts(segments, effective)
    if sum(counts) > max_samples:  # float rounding guard; max length is feasible
        effective = max(segment.length for segment in segments)
        if not math.isfinite(effective) or effective <= 0.0:
            raise _NumericFailure("effective sampling step is not finite")
        counts = _interval_counts(segments, effective)
    return counts, effective, True


def _samples(
    segments: Sequence[_Segment], counts: Sequence[int]
) -> List[Tuple[float, float, float]]:
    """Return ``(x, y, weight)`` intervals covering each segment fully."""

    samples: List[Tuple[float, float, float]] = []
    for segment, count in zip(segments, counts):
        interval = segment.length / count
        ux = (segment.x2 - segment.x1) / segment.length
        uy = (segment.y2 - segment.y1) / segment.length
        for index in range(count):
            distance = (index + 0.5) * interval
            x = segment.x1 + ux * distance
            y = segment.y1 + uy * distance
            if not (math.isfinite(x) and math.isfinite(y)):
                raise _NumericFailure("sample coordinate overflowed float64")
            samples.append((x, y, interval))
    return samples


def _point_segment_distance(
    px: float, py: float, segment: _Segment
) -> float:
    dx = segment.x2 - segment.x1
    dy = segment.y2 - segment.y1
    denominator = dx * dx + dy * dy
    if not math.isfinite(denominator):
        raise _NumericFailure("segment projection denominator overflowed float64")
    if denominator == 0.0:  # pragma: no cover - zero segments are filtered
        return math.hypot(px - segment.x1, py - segment.y1)
    numerator = (px - segment.x1) * dx + (py - segment.y1) * dy
    if not math.isfinite(numerator):
        raise _NumericFailure("segment projection numerator overflowed float64")
    ratio = max(0.0, min(1.0, numerator / denominator))
    distance = math.hypot(
        px - (segment.x1 + ratio * dx), py - (segment.y1 + ratio * dy)
    )
    if not math.isfinite(distance):
        raise _NumericFailure("point-to-segment distance overflowed float64")
    return distance


def _nearest_distance(
    px: float, py: float, target: Sequence[_Segment]
) -> float:
    best = math.inf
    for segment in target:
        distance = _point_segment_distance(px, py, segment)
        if distance < best:
            best = distance
    if not math.isfinite(best):
        raise _NumericFailure("nearest distance is not finite")
    return best


def _weighted_quantile(
    distances: Sequence[Tuple[float, float]], total_weight: float, fraction: float
) -> float:
    """Smallest distance whose cumulative weight reaches ``fraction``."""

    target = total_weight * fraction
    cumulative = 0.0
    for distance, weight in sorted(distances, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= target:
            return distance
    return sorted(distances, key=lambda item: item[0])[-1][0]


def _distance_summary(
    distances: Sequence[Tuple[float, float]], total_weight: float
) -> Dict[str, float]:
    weighted = math.fsum(distance * weight for distance, weight in distances)
    mean = weighted / total_weight
    if not math.isfinite(mean):
        raise _NumericFailure("weighted mean distance overflowed float64")
    return {
        "mean": mean,
        "median": _weighted_quantile(distances, total_weight, 0.5),
        "p95": _weighted_quantile(distances, total_weight, 0.95),
    }


def _empty_direction(reason: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "empty_source",
        "reason": reason,
        "support_length": 0.0,
        "sample_count": 0,
        "requested_step": None,
        "effective_step": None,
        "sampling_limited": False,
        "represented_length": 0.0,
        "positive_segments": 0,
        "target_length": extra.pop("target_length", 0.0),
        "coverage": None,
        "hit_length": None,
        "distance": None,
        "no_target": False,
        "no_target_support_length": 0.0,
    }
    payload.update(extra)
    return payload


def _evaluate_direction(
    source: PolylineSet,
    target: PolylineSet,
    *,
    step: float,
    max_samples: int,
    tolerances: Sequence[float],
) -> Dict[str, Any]:
    source_segments = _segments_of(source)
    target_segments = _segments_of(target)
    source_length = _total_length(source_segments)
    target_length = _total_length(target_segments)

    if not source_segments:
        payload = _empty_direction("zero_source_length", target_length=target_length)
        if not target_segments:
            payload["status"] = "empty_source"
        return payload

    counts, effective, limited = _plan_intervals(
        source_segments, step, max_samples
    )
    sample_points = _samples(source_segments, counts)
    represented = math.fsum(sample[2] for sample in sample_points)
    metadata = {
        "support_length": source_length,
        "sample_count": len(sample_points),
        "requested_step": step,
        "effective_step": effective,
        "sampling_limited": limited,
        "represented_length": represented,
        "positive_segments": len(source_segments),
        "target_length": target_length,
    }
    if not math.isfinite(represented) or abs(represented - source_length) > 1e-9 * max(
        1.0, source_length
    ):
        raise _NumericFailure("sample weights do not close over drawn length")
    for name in ("support_length", "target_length", "effective_step", "represented_length"):
        value = metadata[name]
        if value is not None and not math.isfinite(value):
            raise _NumericFailure(f"{name} is not finite")

    if not target_segments:
        payload = _empty_direction("empty_target", **metadata)
        payload.update(
            {
                "status": "no_target",
                "coverage": {_tolerance_key(t): 0.0 for t in tolerances},
                "hit_length": {_tolerance_key(t): 0.0 for t in tolerances},
                "no_target": True,
                "no_target_support_length": source_length,
            }
        )
        return payload

    distances: List[Tuple[float, float]] = []
    for px, py, weight in sample_points:
        distances.append((_nearest_distance(px, py, target_segments), weight))

    hit = {_tolerance_key(t): 0.0 for t in tolerances}
    for distance, weight in distances:
        for tolerance in tolerances:
            if distance <= tolerance:
                hit[_tolerance_key(tolerance)] += weight

    coverage = {
        key: (hit[key] / represented if represented else None)
        for key in hit
    }
    payload = _empty_direction("evaluated", **metadata)
    payload.update(
        {
            "status": "ok",
            "coverage": coverage,
            "hit_length": hit,
            "distance": _distance_summary(distances, represented),
            "no_target": False,
            "no_target_support_length": 0.0,
        }
    )
    return payload


def _quantise(value: float, quantum: float) -> float:
    return math.floor(value / quantum + 0.5) * quantum


def duplicate_summary(
    pred: PolylineSet, *, quantum: float = DEFAULT_DUPLICATE_QUANTUM
) -> Dict[str, Any]:
    """Endpoint-quantised duplicate proxies; predictions are never merged."""

    quantum = _finite(quantum, "duplicate_quantum")
    if quantum <= 0.0:
        raise EvaluationInputError("duplicate_quantum must be positive")
    segments = _segments_of(pred)
    original_length = _total_length(segments)
    groups: Dict[Tuple[float, float, float, float], List[float]] = {}
    for segment in segments:
        a = (_quantise(segment.x1, quantum), _quantise(segment.y1, quantum))
        b = (_quantise(segment.x2, quantum), _quantise(segment.y2, quantum))
        if b < a:
            a, b = b, a
        groups.setdefault((a[0], a[1], b[0], b[1]), []).append(segment.length)

    duplicate_segments = sum(len(lengths) - 1 for lengths in groups.values())
    duplicate_length = math.fsum(
        math.fsum(lengths[1:]) for lengths in groups.values() if len(lengths) > 1
    )
    ratio = (
        duplicate_length / original_length if original_length > 0.0 else None
    )
    return {
        "quantum_px": quantum,
        "original_segments": len(segments),
        "unique_segments": len(groups),
        "duplicate_segments": duplicate_segments,
        "original_length": original_length,
        "duplicate_length": duplicate_length,
        "duplicate_length_ratio": ratio,
        "method": "endpoint-quantised duplicate proxy; not full overlap matching",
        "merged": False,
    }


def evaluate_pair(
    gt: Any,
    pred: Any,
    *,
    image_id: Optional[str] = None,
    support_scope: str = SUPPORT_ALL_STROKES,
    step_px: float = DEFAULT_STEP_PX,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    tolerances: Sequence[float] = DEFAULT_TOLERANCES,
    duplicate_quantum: float = DEFAULT_DUPLICATE_QUANTUM,
) -> Dict[str, Any]:
    """Evaluate one GT/prediction pair in explicit pixel coordinates.

    ``gt``/``pred`` accept a :class:`PolylineSet` or an iterable of polylines
    (a segment is a two-point polyline; see :meth:`PolylineSet.from_segments`).
    ``support_scope`` only labels the metric: pass geometry that has already
    been filtered by :func:`classify_primitive`.  With
    ``support_scope="straight_only"`` the prediction-to-GT direction is named
    ``affinity`` rather than overall coverage.
    """

    if support_scope not in SUPPORT_SCOPES:
        raise EvaluationInputError(
            f"support_scope must be one of {SUPPORT_SCOPES}, got {support_scope!r}"
        )
    step = _finite(step_px, "step_px")
    if step <= 0.0:
        raise EvaluationInputError("step_px must be positive")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise EvaluationInputError("max_samples must be an integer")
    if max_samples <= 0:
        raise EvaluationInputError("max_samples must be positive")
    tolerance_values = tuple(_finite(item, "tolerance") for item in tolerances)
    if not tolerance_values or any(item <= 0.0 for item in tolerance_values):
        raise EvaluationInputError("tolerances must be positive")
    if len(set(tolerance_values)) != len(tolerance_values):
        raise EvaluationInputError("tolerances must be unique")

    gt_set = as_polyline_set(gt)
    pred_set = as_polyline_set(pred)
    quantum = _finite(duplicate_quantum, "duplicate_quantum")

    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "input_schema": INPUT_SCHEMA,
        "image_id": image_id,
        "support_scope": support_scope,
        "coordinate_space": "pixel; integer coordinates are pixel centres",
        "config": {
            "step_px": step,
            "max_samples": max_samples,
            "tolerances": list(tolerance_values),
            "duplicate_quantum_px": quantum,
        },
        "gt": {**gt_set.to_dict(), "positive_segments": None, "length": None},
        "pred": {**pred_set.to_dict(), "positive_segments": None, "length": None},
        "duplicate": {},
        "status": STATUS_OK,
        "status_reason": None,
        "directions": {},
    }

    limited = False
    try:
        gt_segments = _segments_of(gt_set)
        pred_segments = _segments_of(pred_set)
        gt_length = _total_length(gt_segments)
        pred_length = _total_length(pred_segments)
        result["gt"].update(
            {"positive_segments": len(gt_segments), "length": gt_length}
        )
        result["pred"].update(
            {"positive_segments": len(pred_segments), "length": pred_length}
        )
        result["duplicate"] = duplicate_summary(pred_set, quantum=quantum)

        try:
            gt_to_pred = _evaluate_direction(
                gt_set,
                pred_set,
                step=step,
                max_samples=max_samples,
                tolerances=tolerance_values,
            )
            pred_to_gt = _evaluate_direction(
                pred_set,
                gt_set,
                step=step,
                max_samples=max_samples,
                tolerances=tolerance_values,
            )
        except _SamplingImpossible as exc:
            limited = True
            result["status_reason"] = str(exc)
            gt_to_pred = _empty_direction(
                "positive_segment_count_exceeds_sample_limit",
                target_length=pred_length,
                support_length=gt_length,
                positive_segments=len(gt_segments),
                sample_count=None,
                requested_step=step,
                effective_step=None,
                sampling_limited=True,
                represented_length=None,
            )
            gt_to_pred.update(
                {
                    "status": STATUS_EVALUATION_LIMITED,
                    "coverage": None,
                    "hit_length": None,
                    "distance": None,
                }
            )
            pred_to_gt = _empty_direction(
                "positive_segment_count_exceeds_sample_limit",
                target_length=gt_length,
                support_length=pred_length,
                positive_segments=len(pred_segments),
                sample_count=None,
                requested_step=step,
                effective_step=None,
                sampling_limited=True,
                represented_length=None,
            )
            pred_to_gt.update(
                {
                    "status": STATUS_EVALUATION_LIMITED,
                    "coverage": None,
                    "hit_length": None,
                    "distance": None,
                }
            )
    except _NumericFailure as exc:
        return _failed_result(result, f"numeric_overflow: {exc}")
    except OverflowError as exc:  # pragma: no cover - defensive belt
        return _failed_result(result, f"numeric_overflow: {exc}")

    gt_to_pred["metric_name"] = (
        "straight_target_coverage"
        if support_scope == SUPPORT_STRAIGHT_ONLY
        else "gt_coverage"
    )
    pred_to_gt["metric_name"] = (
        "affinity" if support_scope == SUPPORT_STRAIGHT_ONLY else "overall_coverage"
    )
    gt_to_pred["direction"] = "gt_to_pred"
    pred_to_gt["direction"] = "pred_to_gt"
    result["directions"] = {"gt_to_pred": gt_to_pred, "pred_to_gt": pred_to_gt}

    if limited:
        result["status"] = STATUS_EVALUATION_LIMITED
    elif (
        gt_to_pred["status"] == "empty_source"
        and pred_to_gt["status"] == "empty_source"
    ):
        result["status"] = STATUS_EMPTY_PAIR
    elif any(
        direction.get("sampling_limited") for direction in (gt_to_pred, pred_to_gt)
    ):
        # Metrics are valid at the coarser uniform step, but the precision loss
        # is part of the result and must stay visible.
        result["status_reason"] = "sampling_step_raised_to_fit_max_samples"
    return result


def _failed_result(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    result["status"] = STATUS_FAILED
    result["status_reason"] = reason
    result["directions"] = {}
    result["duplicate"] = result.get("duplicate") or {}
    return result


# ---------------------------------------------------------------------------
# R2: line-only support classification from raw commands (no file reads)
# ---------------------------------------------------------------------------


def path_commands(value: Any) -> List[str]:
    """Extract command letters from path data or a command sequence."""

    if isinstance(value, str):
        letters = []
        for match in _TOKEN_RE.finditer(value):
            token = match.group(0)
            if token.isalpha():
                letters.append(token)
        return letters

    letters = []
    try:
        items = list(value)
    except TypeError as exc:
        raise EvaluationInputError(f"path commands must be text or a sequence: {exc}")
    for item in items:
        if isinstance(item, dict):
            token = item.get("command")
        else:
            token = item
        if not isinstance(token, str) or not token.strip():
            raise EvaluationInputError(f"invalid path command item: {item!r}")
        token = token.strip()
        if len(token) != 1 or not token.isalpha():
            raise EvaluationInputError(f"invalid path command token: {token!r}")
        letters.append(token)
    return letters


def classify_path_geometry(
    value: Any, *, drawn_length: Optional[float] = None
) -> Dict[str, Any]:
    """Classify a path as straight-only, curved/mixed, or unsupported.

    ``value`` is raw path data or a sequence of command letters/dicts.  For a
    command sequence the drawn length is not implied, so pass
    ``drawn_length``; a text ``d`` computes it with the existing parser.  Only
    M/L/H/V/Z with positive drawn length is ``straight_only``.
    """

    letters = path_commands(value)
    upper = [letter.upper() for letter in letters]
    unknown = sorted({letter for letter in upper if letter not in _KNOWN_COMMANDS})
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "commands": upper,
        "category": CATEGORY_UNSUPPORTED,
        "drawn_length": None,
        "reason": None,
    }
    if unknown:
        payload["reason"] = "unsupported_command:" + ",".join(unknown)
        return payload
    if not upper:
        payload["reason"] = "empty_path"
        return payload

    if any(letter in _CURVE_COMMANDS for letter in upper):
        payload["category"] = CATEGORY_CURVED_OR_MIXED
        payload["reason"] = "curve_command_present"
        return payload

    if drawn_length is None:
        if not isinstance(value, str):
            raise EvaluationInputError(
                "drawn_length is required when classifying a command sequence"
            )
        from . import geometry as _geometry

        try:
            parsed = _geometry.parse_path_data(value)
        except _geometry.PathDataError as exc:
            payload["reason"] = f"unparsable_path:{exc}"
            return payload
        try:
            length = math.fsum(
                _geometry.polyline_length(subpath) for subpath in parsed["subpaths"]
            )
        except OverflowError:
            payload["reason"] = "drawn_length_overflow"
            return payload
        if not math.isfinite(length):
            payload["reason"] = "drawn_length_overflow"
            return payload
    else:
        length = _finite(drawn_length, "drawn_length")
    payload["drawn_length"] = length
    if length <= 0.0:
        payload["reason"] = "zero_drawn_length"
        return payload
    payload["category"] = CATEGORY_STRAIGHT_ONLY
    payload["reason"] = "only_MLHVZ_with_positive_length"
    return payload


def classify_primitive(
    element: str, *, path_commands_value: Any = None, drawn_length: Optional[float] = None
) -> Dict[str, Any]:
    """Classify one primitive by element name and, for paths, its commands."""

    name = str(element).strip().lower()
    if name == "path":
        if path_commands_value is None:
            raise EvaluationInputError("path classification requires path commands")
        payload = classify_path_geometry(
            path_commands_value, drawn_length=drawn_length
        )
        payload["element"] = "path"
        return payload
    if name in (CATEGORY_CIRCLE, CATEGORY_ELLIPSE):
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "element": name,
            "commands": [],
            "category": name,
            "drawn_length": None,
            "reason": "shape_not_converted_by_line_frontend",
        }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "element": name or None,
        "commands": [],
        "category": CATEGORY_UNSUPPORTED,
        "drawn_length": None,
        "reason": "unsupported_element",
    }


# ---------------------------------------------------------------------------
# Aggregation over per-image results and explicit statuses
# ---------------------------------------------------------------------------


def _summary_number(value: Any, what: str) -> float:
    number = 0.0 if value is None else float(value)
    if not math.isfinite(number):
        raise EvaluationInputError(
            f"summary {what} must be finite, got {value!r}; refusing to emit NaN/inf"
        )
    return number


def _summary_add(total: float, value: float, what: str) -> float:
    result = total + value
    if not math.isfinite(result):
        raise EvaluationInputError(
            f"summary {what} overflowed float64; split or reduce the group"
        )
    return result


def _direction_summary(
    results: Sequence[Dict[str, Any]], key: str
) -> Dict[str, Any]:
    images_with_metric = 0
    no_target_images = 0
    support_length = 0.0
    hit_length: Dict[str, float] = {}
    coverage_values: Dict[str, List[float]] = {}
    distance_means: List[Tuple[float, float]] = []
    no_target_support = 0.0

    for result in results:
        direction = result.get("directions", {}).get(key)
        if not direction:
            continue
        if direction.get("no_target"):
            no_target_images += 1
            no_target_support = _summary_add(
                no_target_support,
                _summary_number(direction.get("support_length"), "no_target_support_length"),
                "no_target_support_length",
            )
        coverage = direction.get("coverage")
        if not coverage:
            continue
        images_with_metric += 1
        support_length = _summary_add(
            support_length,
            _summary_number(direction.get("support_length"), "support_length"),
            "support_length",
        )
        for tolerance, value in (direction.get("hit_length") or {}).items():
            hit_length[tolerance] = _summary_add(
                hit_length.get(tolerance, 0.0),
                _summary_number(value, f"hit_length[{tolerance}]"),
                "hit_length",
            )
        for tolerance, value in coverage.items():
            coverage_values.setdefault(tolerance, []).append(
                _summary_number(value, f"coverage[{tolerance}]")
            )
        distance = direction.get("distance")
        if distance:
            distance_means.append(
                (
                    _summary_number(distance.get("mean"), "distance mean"),
                    _summary_number(
                        direction.get("support_length"), "distance support weight"
                    ),
                )
            )

    macro = None
    if coverage_values:
        macro = {}
        for tolerance, values in sorted(coverage_values.items()):
            total = 0.0
            for value in values:
                total = _summary_add(total, value, "coverage macro total")
            macro[tolerance] = total / len(values)

    micro = None
    if support_length > 0.0:
        micro = {
            tolerance: (hit_length.get(tolerance, 0.0) / support_length)
            for tolerance in sorted(hit_length)
        }

    distance_mean_of_means = None
    if distance_means:
        total = 0.0
        for mean, _ in distance_means:
            total = _summary_add(total, mean, "distance mean total")
        distance_mean_of_means = total / len(distance_means)

    distance_weighted_mean = None
    if distance_means:
        weight_total = 0.0
        for _, weight in distance_means:
            weight_total = _summary_add(weight_total, weight, "distance weight total")
        if weight_total > 0.0:
            try:
                weighted_total = math.fsum(
                    mean * weight for mean, weight in distance_means
                )
            except OverflowError as exc:
                raise EvaluationInputError(
                    "summary weighted distance overflowed float64"
                ) from exc
            if not math.isfinite(weighted_total):
                raise EvaluationInputError(
                    "summary weighted distance is not finite"
                )
            distance_weighted_mean = weighted_total / weight_total
    return {
        "images_with_metric": images_with_metric,
        "images_no_target": no_target_images,
        "support_length": support_length,
        "hit_length": dict(sorted(hit_length.items())),
        "coverage_macro": macro,
        "coverage_micro": micro,
        "distance_mean_of_image_means": distance_mean_of_means,
        "distance_support_weighted_mean": distance_weighted_mean,
        "no_target_support_length": no_target_support,
    }


def _normalised_config(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config = result.get("config")
    if not isinstance(config, dict):
        return None
    tolerances = config.get("tolerances")
    if not isinstance(tolerances, (list, tuple)) or not tolerances:
        raise EvaluationInputError(
            "result config must contain a non-empty tolerances list"
        )
    max_samples = config.get("max_samples")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int)
    ):
        raise EvaluationInputError("result config max_samples must be an integer")
    # max_samples is deliberately not part of the shared identity: a sampling
    # budget that only triggers a raised effective_step is allowed per image.
    return {
        "step_px": _summary_number(config.get("step_px"), "config step_px"),
        "tolerances": tuple(
            sorted(_summary_number(item, "config tolerance") for item in tolerances)
        ),
        "duplicate_quantum_px": _summary_number(
            config.get("duplicate_quantum_px"), "config duplicate_quantum_px"
        ),
    }


def _metric_names(result: Dict[str, Any]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for key, direction in (result.get("directions") or {}).items():
        if isinstance(direction, dict) and "metric_name" in direction:
            names[key] = str(direction["metric_name"])
    return names


def summarise(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-image result documents with their explicit statuses.

    Allowed statuses are ``ok``, ``empty_pair``, ``evaluation_limited``,
    ``failed``, ``excluded`` and ``alignment_pending``; unknown statuses raise.
    ``eligible`` counts ``ok`` images only.  Macro coverage averages each
    image's defined coverage equally; micro coverage sums hit length over
    support length.  Empty prediction images keep their zero recall in both.
    No global median/P95 is produced; only support-weighted means are combined.

    Records that carry a schema, ``support_scope``, metric names or an
    evaluation config must agree with the rest of the group.  Mixing overall
    coverage with straight-only affinity, different tolerance sets or
    different requested steps raises :class:`EvaluationInputError`; summarise
    each group separately.  Plain status records need no extra fields.
    Aggregation overflow raises instead of returning NaN/Inf.
    """

    results = list(results)
    counts = {
        "selected": len(results),
        "eligible": 0,
        "excluded": 0,
        "failed": 0,
        "alignment_pending": 0,
        "evaluation_limited": 0,
        "empty_pair": 0,
    }
    eligible_results: List[Dict[str, Any]] = []
    duplicate_segments = 0
    duplicate_length = 0.0
    original_length = 0.0
    images_with_duplicates = 0

    schema_seen = False
    stored_schema: Any = None
    stored_version: Any = None
    scope_seen = False
    stored_scope: Any = None
    config_seen = False
    stored_config: Optional[Dict[str, Any]] = None
    stored_metric_names: Dict[str, str] = {}

    for result in results:
        if not isinstance(result, dict):
            raise EvaluationInputError(
                f"result must be a mapping, got {type(result)!r}"
            )
        status = result.get("status")
        if status not in SUMMARY_STATUSES:
            raise EvaluationInputError(f"unknown result status: {status!r}")

        if "schema" in result:
            schema = result.get("schema")
            version = result.get("schema_version")
            if not schema_seen:
                stored_schema, stored_version = schema, version
                schema_seen = True
            elif schema != stored_schema or version != stored_version:
                raise EvaluationInputError(
                    "cannot summarise mixed schemas/versions: "
                    f"{stored_schema!r}/v{stored_version!r} vs {schema!r}/v{version!r}"
                )

        if "support_scope" in result:
            scope = result.get("support_scope")
            if not scope_seen:
                stored_scope = scope
                scope_seen = True
            elif scope != stored_scope:
                raise EvaluationInputError(
                    "cannot summarise mixed support scopes: "
                    f"{stored_scope!r} vs {scope!r}; summarise each scope separately"
                )

        config = _normalised_config(result)
        if config is not None:
            if not config_seen:
                stored_config = config
                config_seen = True
            elif config != stored_config:
                raise EvaluationInputError(
                    "cannot summarise mixed evaluation configuration "
                    f"(tolerances/requested step/duplicate quantum): "
                    f"{stored_config!r} vs {config!r}"
                )

        names = _metric_names(result)
        for key, name in names.items():
            if key in stored_metric_names and stored_metric_names[key] != name:
                raise EvaluationInputError(
                    "cannot summarise mixed metric identity for "
                    f"{key!r}: {stored_metric_names[key]!r} vs {name!r}"
                )
            stored_metric_names.setdefault(key, name)

        if status == STATUS_OK:
            counts["eligible"] += 1
            eligible_results.append(result)
        else:
            counts[status] += 1

        duplicate = result.get("duplicate") or {}
        if duplicate.get("duplicate_segments"):
            images_with_duplicates += 1
        duplicate_count = duplicate.get("duplicate_segments") or 0
        if isinstance(duplicate_count, bool) or not isinstance(duplicate_count, int):
            raise EvaluationInputError(
                "duplicate_segments must be an integer when present"
            )
        duplicate_segments += duplicate_count
        duplicate_length = _summary_add(
            duplicate_length,
            _summary_number(duplicate.get("duplicate_length"), "duplicate_length"),
            "duplicate_length",
        )
        original_length = _summary_add(
            original_length,
            _summary_number(duplicate.get("original_length"), "original_length"),
            "original_length",
        )

    raised_step_images = 0
    for result in eligible_results:
        for direction in (result.get("directions") or {}).values():
            if isinstance(direction, dict) and direction.get("sampling_limited"):
                raised_step_images += 1
                break

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "support_scope": stored_scope if scope_seen else None,
        "metric_names": dict(stored_metric_names) if stored_metric_names else None,
        "config": (
            {
                **stored_config,
                "tolerances": list(stored_config["tolerances"]),
            }
            if stored_config is not None
            else None
        ),
        "sampling": {
            "images_with_raised_step": raised_step_images,
            "note": "effective_step and max_samples may differ per image by "
            "design; requested step, tolerance set and duplicate quantum must "
            "match within one summary",
        },
        "counts": counts,
        "directions": {
            "gt_to_pred": _direction_summary(eligible_results, "gt_to_pred"),
            "pred_to_gt": _direction_summary(eligible_results, "pred_to_gt"),
        },
        "duplicate": {
            "images_with_duplicates": images_with_duplicates,
            "duplicate_segments": duplicate_segments,
            "duplicate_length": duplicate_length,
            "original_length": original_length,
            "duplicate_length_ratio": (
                duplicate_length / original_length if original_length > 0.0 else None
            ),
        },
        "notes": [
            "coverage_macro averages per-image coverage equally over eligible "
            "images with defined coverage; coverage_micro is hit/support length",
            "empty predictions keep coverage 0 in both macro and micro",
            "no global median/P95 is derived from per-image quantiles",
            "distance_support_weighted_mean combines per-image weighted means",
            "duplicate totals cover every selected result that carries duplicate "
            "metadata; coverage aggregates use eligible images only",
            "mixed support scopes, metric names or evaluation configs are rejected; "
            "summarise each group separately",
            "effective_step and max_samples may vary per image; requested step, "
            "tolerances and duplicate quantum are shared",
        ],
    }
