"""Geometry-only node features and local graph for recognition.

The public API accepts only :class:`NodeGeometry`.  Labels, associations and
targets live in separate ``Supervision`` records and cannot be passed here by
accident.  Node and edge feature order is frozen by the contract module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .recognition_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RecognitionError,
    Supervision,
)


class FeatureLimitError(RecognitionError):
    code = "feature_limit"


@dataclass(frozen=True, slots=True)
class NodeGeometry:
    """Immutable geometry identity only; no labels, target or association."""

    candidate_uid: str
    sample_id: str
    source_kind: str
    points_px: Tuple[Tuple[float, float], ...]
    image_width: float = 1.0
    image_height: float = 1.0


@dataclass(frozen=True, slots=True)
class Edge:
    source_uid: str
    target_uid: str


@dataclass(frozen=True)
class FeatureConfig:
    edge_radius_px: float = 24.0
    knn: int = 8
    max_nodes: int = 20953
    max_edges: int = 50_000


def _finite_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("point must be (x, y)")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("coordinate must be finite")
        out.append((x, y))
    return out


def _length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def _bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _midpoint(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    x0, y0 = points[0]
    x1, y1 = points[-1]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _orientation(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = points[0], points[-1]
    angle = math.atan2(y1 - y0, x1 - x0)
    return math.sin(angle), math.cos(angle)


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


def _bbox_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _bbox_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _endpoint_gap(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    if not a or not b:
        return math.inf
    return min(
        math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        for pa in (a[0], a[-1])
        for pb in (b[0], b[-1])
    )


def _angle_diff_deg(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    sa, ca = _orientation(a)
    sb, cb = _orientation(b)
    dot = max(-1.0, min(1.0, ca * cb + sa * sb))
    angle = math.degrees(math.acos(dot))
    return min(angle, 180.0 - angle)


def _parallel_offset(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    if not a or not b:
        return math.inf
    d_ab = sum(_point_to_polyline(x, y, b) for x, y in (a[0], a[-1])) / 2.0
    d_ba = sum(_point_to_polyline(x, y, a) for x, y in (b[0], b[-1])) / 2.0
    return min(d_ab, d_ba)


def extract_node_features(
    nodes: Sequence[NodeGeometry], config: Optional[FeatureConfig] = None
) -> List[Dict[str, float]]:
    config = config or FeatureConfig()
    if len(nodes) > config.max_nodes:
        raise FeatureLimitError(f"node count {len(nodes)} exceeds {config.max_nodes}")
    points = [_finite_points(node.points_px) for node in nodes]
    lengths = [_length(item) for item in points]
    bboxes = [_bbox(item) if item else (0.0, 0.0, 0.0, 0.0) for item in points]
    mids = [_midpoint(item) for item in points]
    orientations = [_orientation(item) for item in points]
    result: List[Dict[str, float]] = []
    for index, node in enumerate(nodes):
        len_i = lengths[index]
        bx0, by0, bx1, by1 = bboxes[index]
        others = [j for j in range(len(nodes)) if j != index]
        distances = sorted(
            ((math.hypot(mids[index][0] - mids[j][0], mids[index][1] - mids[j][1]), j)
             for j in others),
            key=lambda item: item[0],
        )
        nearest_d, nearest_j = distances[0] if distances else (0.0, None)
        knn = distances[: min(4, len(distances))]
        knn_mean = sum(d for d, _ in knn) / len(knn) if knn else 0.0
        local = [j for d, j in distances if d <= 24.0]
        if local:
            coherence = sum(
                abs(
                    orientations[index][1] * orientations[j][1]
                    + orientations[index][0] * orientations[j][0]
                )
                for j in local
            ) / len(local)
        else:
            coherence = 0.0
        if nearest_j is None:
            ratio = 1.0
        else:
            max_len = max(len_i, lengths[nearest_j])
            ratio = 0.0 if max_len <= 0.0 else min(len_i, lengths[nearest_j]) / max_len
        width = float(node.image_width) if node.image_width else 1.0
        height = float(node.image_height) if node.image_height else 1.0
        values = [
            len_i,
            max(0.0, bx1 - bx0),
            max(0.0, by1 - by0),
            max(0.0, bx1 - bx0) * max(0.0, by1 - by0),
            math.log((max(0.0, bx1 - bx0) + 1.0) / (max(0.0, by1 - by0) + 1.0)),
            orientations[index][0],
            orientations[index][1],
            (math.hypot(
                points[index][0][0] - points[index][-1][0],
                points[index][0][1] - points[index][-1][1],
            ) / len_i) if len_i > 0.0 else 0.0,
            mids[index][0] / width,
            mids[index][1] / height,
            nearest_d,
            knn_mean,
            float(len(local)),
            coherence,
            ratio,
        ]
        result.append(dict(zip(NODE_FEATURE_NAMES, values)))
    return result


def build_local_graph(
    nodes: Sequence[NodeGeometry], config: Optional[FeatureConfig] = None
) -> Tuple[List[Edge], Dict[str, Tuple[str, ...]]]:
    config = config or FeatureConfig()
    if len(nodes) > config.max_nodes:
        raise FeatureLimitError(f"node count {len(nodes)} exceeds {config.max_nodes}")
    points = [_finite_points(node.points_px) for node in nodes]
    mids = [_midpoint(item) for item in points]
    n = len(nodes)
    pairs: set[Tuple[int, int]] = set()
    for i in range(n):
        if n <= 1:
            break
        distances = sorted(
            ((math.hypot(mids[i][0] - mids[j][0], mids[i][1] - mids[j][1]), j)
             for j in range(n) if j != i),
            key=lambda item: item[0],
        )
        for _, j in distances[: config.knn]:
            pairs.add((min(i, j), max(i, j)))
        for j in range(i + 1, n):
            if _endpoint_gap(points[i], points[j]) <= config.edge_radius_px:
                pairs.add((i, j))
    if len(pairs) > config.max_edges:
        raise FeatureLimitError(f"edge count {len(pairs)} exceeds {config.max_edges}")
    edges = [Edge(nodes[i].candidate_uid, nodes[j].candidate_uid) for i, j in sorted(pairs)]
    adjacency: Dict[str, List[str]] = {node.candidate_uid: [] for node in nodes}
    for edge in edges:
        adjacency[edge.source_uid].append(edge.target_uid)
        adjacency[edge.target_uid].append(edge.source_uid)
    return edges, {key: tuple(sorted(value)) for key, value in adjacency.items()}


def extract_edge_features(
    nodes: Sequence[NodeGeometry],
    edges: Sequence[Edge],
    config: Optional[FeatureConfig] = None,
) -> List[Dict[str, float]]:
    del config
    by_uid = {node.candidate_uid: node for node in nodes}
    points = {node.candidate_uid: _finite_points(node.points_px) for node in nodes}
    result: List[Dict[str, float]] = []
    for edge in edges:
        a = by_uid[edge.source_uid]
        b = by_uid[edge.target_uid]
        pa, pb = points[a.candidate_uid], points[b.candidate_uid]
        box_a, box_b = _bbox(pa), _bbox(pb)
        len_a, len_b = _length(pa), _length(pb)
        max_len = max(len_a, len_b)
        values = [
            _endpoint_gap(pa, pb),
            math.hypot(
                _midpoint(pa)[0] - _midpoint(pb)[0],
                _midpoint(pa)[1] - _midpoint(pb)[1],
            ),
            _angle_diff_deg(pa, pb),
            0.0 if max_len <= 0.0 else math.log(max_len / max(min(len_a, len_b), 1e-12)),
            _bbox_gap(box_a, box_b),
            _bbox_overlap(box_a, box_b),
            _parallel_offset(pa, pb),
            _endpoint_gap(pa, pb),
        ]
        result.append(dict(zip(EDGE_FEATURE_NAMES, values)))
    return result


def node_geometry_from_candidate(candidate: Mapping[str, Any]) -> NodeGeometry:
    """Strip supervision/association by construction."""
    geometry = candidate.get("geometry") or {}
    points = geometry.get("points_px") or geometry.get("points") or []
    image = candidate.get("image") or {}
    return NodeGeometry(
        candidate_uid=str(candidate["candidate_uid"]),
        sample_id=str(candidate.get("sample_id") or ""),
        source_kind=str(candidate.get("source_kind") or ""),
        points_px=tuple((float(p[0]), float(p[1])) for p in points),
        image_width=float(image.get("width") or 1.0),
        image_height=float(image.get("height") or 1.0),
    )
