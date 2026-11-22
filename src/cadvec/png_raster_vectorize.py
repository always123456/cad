"""Raster masks -> README section 4 vector JSON (line-only baseline).

This module converts one semantic-ID raster and one instance-ID raster into
the twelve-field document described in ``datasets/README.md`` section 4.  It is
an explicitly synthetic-tested baseline: it has no accuracy or PQ claim, it
never reads the dataset, and it never infers a PNG -> SVG transform from image
dimensions.  The caller must inject the affine mapping explicitly::

    x_view = x_px * scale_x + offset_x
    y_view = y_px * scale_y + offset_y

where ``scale_*`` is viewBox units per pixel.

Label policy (mirrors the project raster policy):

* ``semantic`` holds JSON class IDs ``0..35``.  ``35`` is background/ignore and
  is dropped completely.
* Thing classes ``0..29`` require a non-negative local ``instance`` id.  A
  negative Thing local id is a contract error.
* Stuff classes ``30..34`` are merged per class and emitted with instance
  ``-1``.  Their local instance ids are ignored.
* Output Thing instance ids are namespaced by assigning a fresh sequential id
  to every distinct ``(semantic_id, local_instance_id)`` pair.  The same local
  id in two semantic classes therefore never collides.  Disconnected raster
  components with the same key stay in the same output instance group.

Geometry policy:

* A class-instance mask is skeletonized with a pure NumPy Zhang-Suen thinning.
  Masks whose raw skeleton is short compared with their boundary (filled or
  blobby regions) are replaced by their one-pixel outline first, then that
  outline is skeletonized.  If thinning collapses an elongated thin band to a
  few pixels, the longest 8-connected path through the mask is used instead;
  that fallback is skipped for multi-component masks so no component is lost.
* The skeleton is split into 8-connected chains.  Junctions and endpoints are
  shared between the chains that meet there.
* Every chain is simplified with Ramer-Douglas-Peucker at ``tol_px`` pixels and
  then cut into overlapping windows of at most four samples.  Consecutive
  windows share an endpoint, so the emitted line primitives remain connected.
  Shorter windows are padded by repeating their final sample.  Only command
  ``0`` (Line) is emitted.
* Lengths are computed from the emitted, affine-transformed four samples in
  viewBox units.  ``obj_cts`` and ``boxes`` are recomputed from exactly those
  samples, as the local validator does.
* The opt-in ``emitter="runs"`` path replaces that geometry completely: each
  per-class/per-instance mask is decomposed into its maximal *inclusive*
  horizontal pixel runs, ordered by row and then by start column.  Every run
  becomes one Line with four collinear samples derived from its two inclusive
  endpoints (``x0 .. x1`` at the run's row), so a run contributes exactly
  ``|x1 - x0| * |scale_x|`` viewBox units and a singleton run collapses to four
  identical samples with length ``0``.  There is no skeletonization, no RDP and
  no fallback on this path; ``tol_px`` is still validated but unused.

Exactness bounds: re-rasterizing a ``runs`` document with the local canonical
rasterizer (``recognition_raster_data.rasterize_document``, canvas 100x100,
raster 1000x1000) reproduces the synthetic per-instance masks exactly for the
canonical affine ``scale = 0.1`` and zero offsets.  That is a property of the
local rasterizer and of the tested Pillow build (coordinate truncation keeps
every composed inverse ``(x * 0.1) * 10`` at or above the intended pixel
column); it is not mathematical exactness of binary floats, and a nominal 999
raster is measurably different.  It is not a claim of official evaluator or
raster compatibility.

Limits: the blob/stroke heuristic is a raster heuristic, not a semantic
segmentation; thick or very elongated filled regions can be treated as strokes
(and vice versa).  A line-only representation cannot recover arcs or circles.
Output compatibility with the unpublished official evaluator is unconfirmed.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # The project keeps its stdlib-only core importable without NumPy.
    import numpy as np
except Exception:  # pragma: no cover - exercised only when NumPy is absent
    np = None  # type: ignore

from .recognition_submission import validate_document

SCHEMA = "cadvec.recognition.png_raster_vectorize/1"
COMMAND_LINE = 0
THING_MIN = 0
THING_MAX = 29
STUFF_MIN = 30
STUFF_MAX = 34
IGNORE_ID = 35
DEFAULT_RGB = (0, 0, 0)
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_LAYER_ID = 1
MAX_SAMPLES = 4
EMITTERS: Tuple[str, ...] = ("skeleton", "runs")
# Raw-skeleton length divided by half the boundary length.  Strokes approach
# or exceed 1; filled/blobby regions fall clearly below it.
BLOB_SKELETON_RATIO = 0.65
EIGHT_NEIGHBORS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


class VectorizeError(ValueError):
    """Invalid raster input or an internally inconsistent result document."""


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ImportError("png_raster_vectorize requires numpy")


def _label_array(value: Any, name: str) -> "np.ndarray":
    _require_numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise VectorizeError(f"{name}_not_2d:{array.ndim}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise VectorizeError(f"{name}_empty:{array.shape}")
    if array.dtype.kind == "b":
        raise VectorizeError(f"{name}_boolean_dtype")
    if array.dtype.kind in "iu":
        return array.astype(np.int64, copy=False)
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)):
            raise VectorizeError(f"{name}_non_finite")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise VectorizeError(f"{name}_non_integral")
        return rounded.astype(np.int64)
    raise VectorizeError(f"{name}_unsupported_dtype:{array.dtype}")


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VectorizeError(f"{name}_not_a_number") from exc
    if not math.isfinite(number):
        raise VectorizeError(f"{name}_non_finite")
    return number


def _viewbox_size(value: Any, name: str) -> int:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise VectorizeError(f"{name}_not_positive")
    rounded = int(round(number))
    if rounded <= 0:
        raise VectorizeError(f"{name}_rounds_to_zero:{number}")
    return rounded


def _boundary(mask: "np.ndarray") -> "np.ndarray":
    """Pixels with at least one 4-neighbour outside the mask."""

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[0:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, 0:-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _neighbour_fields(image: "np.ndarray") -> Tuple["np.ndarray", ...]:
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    p2 = padded[0:-2, 1:-1]  # north
    p3 = padded[0:-2, 2:]  # north-east
    p4 = padded[1:-1, 2:]  # east
    p5 = padded[2:, 2:]  # south-east
    p6 = padded[2:, 1:-1]  # south
    p7 = padded[2:, 0:-2]  # south-west
    p8 = padded[1:-1, 0:-2]  # west
    p9 = padded[0:-2, 0:-2]  # north-west
    return p2, p3, p4, p5, p6, p7, p8, p9


def _zhang_suen(mask: "np.ndarray") -> "np.ndarray":
    """Zhang-Suen thinning in pure NumPy (8-connected skeleton)."""

    image = mask.astype(bool, copy=True)
    while True:
        changed = False
        for step in (0, 1):
            if not image.any():
                return image
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbour_fields(image)
            count = (
                p2.astype(np.int16)
                + p3.astype(np.int16)
                + p4.astype(np.int16)
                + p5.astype(np.int16)
                + p6.astype(np.int16)
                + p7.astype(np.int16)
                + p8.astype(np.int16)
                + p9.astype(np.int16)
            )
            transitions = (
                (~p2 & p3).astype(np.int16)
                + (~p3 & p4).astype(np.int16)
                + (~p4 & p5).astype(np.int16)
                + (~p5 & p6).astype(np.int16)
                + (~p6 & p7).astype(np.int16)
                + (~p7 & p8).astype(np.int16)
                + (~p8 & p9).astype(np.int16)
                + (~p9 & p2).astype(np.int16)
            )
            if step == 0:
                first = (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
            else:
                first = (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))
            remove = (
                image
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & first
            )
            if remove.any():
                image[remove] = False
                changed = True
        if not changed:
            return image


def _stroke_skeleton(mask: "np.ndarray") -> "np.ndarray":
    """Skeletonize a stroke mask; outline a filled/blobby mask first."""

    raw = _zhang_suen(mask)
    if not raw.any():
        return mask.astype(bool, copy=True)
    boundary = _boundary(mask)
    if not boundary.any():
        return raw
    ratio = float(raw.sum()) / max(1.0, float(boundary.sum()) / 2.0)
    if ratio >= BLOB_SKELETON_RATIO:
        return raw
    outline = _zhang_suen(boundary)
    return outline if outline.any() else boundary


def _longest_path_in_mask(mask: "np.ndarray") -> Optional["np.ndarray"]:
    """Longest shortest path through an 8-connected mask, as (x, y) points.

    This is a safety net for thin diagonal bands where Zhang-Suen can collapse
    almost the whole band to a couple of pixels.  It returns ``None`` when the
    mask has more than one connected component, so separate objects are never
    dropped.
    """

    coordinates = np.argwhere(mask)
    count = int(coordinates.shape[0])
    if count == 0:
        return None
    pixels = [(int(column), int(row)) for row, column in coordinates]
    index = {pixel: position for position, pixel in enumerate(pixels)}

    def breadth_first(start: int) -> Tuple[int, List[int], int]:
        distances = [-1] * count
        parents = [-1] * count
        distances[start] = 0
        queue = deque([start])
        farthest = start
        seen = 0
        while queue:
            current = queue.popleft()
            seen += 1
            x, y = pixels[current]
            for dx, dy in EIGHT_NEIGHBORS:
                neighbour = index.get((x + dx, y + dy))
                if neighbour is None or distances[neighbour] >= 0:
                    continue
                distances[neighbour] = distances[current] + 1
                parents[neighbour] = current
                queue.append(neighbour)
                if distances[neighbour] > distances[farthest]:
                    farthest = neighbour
        return farthest, parents, seen

    end, _, seen = breadth_first(0)
    if seen != count:
        return None
    end, parents, _ = breadth_first(end)
    path: List[int] = []
    current = end
    while current >= 0:
        path.append(current)
        current = parents[current]
    path.reverse()
    return np.asarray([pixels[position] for position in path], dtype=float)


def _skeleton_is_collapsed(mask: "np.ndarray", skeleton: "np.ndarray") -> bool:
    """Whether thinning lost an implausibly large part of a thin mask."""

    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or columns.size == 0:
        return False
    diameter = float(
        max(int(rows[-1] - rows[0]) + 1, int(columns[-1] - columns[0]) + 1)
    )
    if diameter <= 2.0:
        return False
    if float(skeleton.sum()) >= 0.8 * diameter:
        return False
    area = float(mask.sum())
    boundary = float(_boundary(mask).sum())
    thickness = 2.0 * area / max(1.0, boundary)
    return thickness <= 3.0



def _extract_chains(skeleton: "np.ndarray") -> List["np.ndarray"]:
    """Split a skeleton into 8-connected chains of (x, y) float points."""

    coordinates = np.argwhere(skeleton)
    if coordinates.size == 0:
        return []
    pixels = [(int(column), int(row)) for row, column in coordinates]
    index = {pixel: position for position, pixel in enumerate(pixels)}

    neighbours: List[List[int]] = []
    for x, y in pixels:
        found = []
        for dx, dy in EIGHT_NEIGHBORS:
            position = index.get((x + dx, y + dy))
            if position is None:
                continue
            if dx and dy:
                # Do not add the diagonal across a triangle.  A one-pixel
                # rectangular corner (or a staircase's 2 x 2 block) is
                # 8-adjacent diagonally as well; keeping that edge would
                # shortcut the corner and break the chain topology.
                if (x + dx, y) in index or (x, y + dy) in index:
                    continue
            found.append(position)
        neighbours.append(sorted(found))

    degree = [len(item) for item in neighbours]
    visited = set()
    chains: List[List[int]] = []

    def edge_key(left: int, right: int) -> Tuple[int, int]:
        return (left, right) if left < right else (right, left)

    # Open chains start at endpoints or junctions, so junctions are shared
    # endpoints of every chain that meets there.
    for start in range(len(pixels)):
        if degree[start] == 2:
            continue
        if degree[start] == 0:
            chains.append([start])
            continue
        for neighbour in neighbours[start]:
            key = edge_key(start, neighbour)
            if key in visited:
                continue
            visited.add(key)
            path = [start, neighbour]
            previous, current = start, neighbour
            while degree[current] == 2:
                candidates = [item for item in neighbours[current] if item != previous]
                if not candidates:
                    break
                following = candidates[0]
                next_key = edge_key(current, following)
                if next_key in visited:
                    break
                visited.add(next_key)
                path.append(following)
                previous, current = current, following
            chains.append(path)

    # Remaining edges belong to closed loops (every node has degree two).
    for start in range(len(pixels)):
        for neighbour in neighbours[start]:
            key = edge_key(start, neighbour)
            if key in visited:
                continue
            visited.add(key)
            path = [start, neighbour]
            previous, current = start, neighbour
            while current != start:
                if degree[current] != 2:
                    break
                candidates = [item for item in neighbours[current] if item != previous]
                if not candidates:
                    break
                following = candidates[0]
                next_key = edge_key(current, following)
                if next_key in visited:
                    break
                visited.add(next_key)
                path.append(following)
                previous, current = current, following
            if current == start:
                chains.append(path)

    return [
        np.asarray([pixels[position] for position in path], dtype=float)
        for path in chains
        if path
    ]


def _distance_to_segment(
    points: "np.ndarray", start: "np.ndarray", end: "np.ndarray"
) -> "np.ndarray":
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared <= 0.0:
        return np.linalg.norm(points - start, axis=1)
    raw = ((points - start) @ segment) / length_squared
    clamped = np.clip(raw, 0.0, 1.0)
    projection = start + clamped[:, None] * segment
    return np.linalg.norm(points - projection, axis=1)


def _rdp_open(points: "np.ndarray", tolerance: float) -> "np.ndarray":
    count = int(points.shape[0])
    if count <= 2 or tolerance <= 0.0:
        return points
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        interior = points[first + 1 : last]
        distances = _distance_to_segment(interior, points[first], points[last])
        offset = int(np.argmax(distances))
        if float(distances[offset]) > tolerance:
            pivot = first + 1 + offset
            keep[pivot] = True
            stack.append((first, pivot))
            stack.append((pivot, last))
    return points[keep]


def _rdp(points: "np.ndarray", tolerance: float) -> "np.ndarray":
    count = int(points.shape[0])
    if count <= 2 or tolerance <= 0.0:
        return points
    closed = bool(np.array_equal(points[0], points[-1]))
    if not closed:
        return _rdp_open(points, tolerance)
    interior = points[1:-1]
    if interior.size == 0:
        return points
    distances = np.linalg.norm(interior - points[0], axis=1)
    pivot = 1 + int(np.argmax(distances))
    left = _rdp_open(points[: pivot + 1], tolerance)
    right = _rdp_open(points[pivot:], tolerance)
    return np.vstack([left, right[1:]])


def _chunks(points: "np.ndarray", max_samples: int = MAX_SAMPLES) -> List["np.ndarray"]:
    """Overlapping windows of at most ``max_samples`` points.

    Every window after the first starts at the previous window's last point, so
    the chains stay connected.  The final window always reaches the chain end.
    """

    count = int(points.shape[0])
    if count == 0:
        return []
    if count == 1:
        return [points]
    stride = max(1, int(max_samples) - 1)
    result: List["np.ndarray"] = []
    cursor = 0
    while cursor < count - 1:
        stop = min(cursor + stride, count - 1)
        result.append(points[cursor : stop + 1])
        cursor = stop
    return result


def _padding(args: Sequence[float]) -> List[float]:
    """Repeat the final sample until the four-sample argument is complete."""

    values = [float(item) for item in args]
    while len(values) < 8:
        values.extend(values[-2:])
    return values[:8]


def _polyline_length(args: Sequence[float]) -> float:
    total = 0.0
    for offset in range(0, 6, 2):
        total += math.hypot(
            float(args[offset + 2]) - float(args[offset]),
            float(args[offset + 3]) - float(args[offset + 1]),
        )
    return total


def _transform(
    points: "np.ndarray", scale_x: float, scale_y: float, offset_x: float, offset_y: float
) -> "np.ndarray":
    transformed = np.empty_like(points, dtype=float)
    transformed[:, 0] = points[:, 0] * scale_x + offset_x
    transformed[:, 1] = points[:, 1] * scale_y + offset_y
    return transformed


def _fallback_chain(mask: "np.ndarray") -> "np.ndarray":
    coordinates = np.argwhere(mask)
    row, column = coordinates[0]
    return np.asarray([[float(column), float(row)]], dtype=float)


def _emit_mask(
    mask: "np.ndarray",
    semantic_id: int,
    instance_id: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    tolerance: float,
) -> List[Dict[str, Any]]:
    skeleton = _stroke_skeleton(mask)
    if _skeleton_is_collapsed(mask, skeleton):
        path = _longest_path_in_mask(mask)
        chains = [path] if path is not None else _extract_chains(skeleton)
    else:
        chains = _extract_chains(skeleton)
    if not chains:
        chains = [_fallback_chain(mask)]

    primitives: List[Dict[str, Any]] = []
    for chain in chains:
        simplified = _rdp(chain, tolerance)
        if simplified.size == 0:
            continue
        if simplified.shape[0] == 1:
            simplified = np.vstack([simplified, simplified])
        for window in _chunks(simplified):
            if window.shape[0] == 0:
                continue
            view_points = _transform(window, scale_x, scale_y, offset_x, offset_y)
            flat: List[float] = []
            for x, y in view_points:
                flat.extend([float(x), float(y)])
            args = _padding(flat)
            primitives.append(
                {
                    "command": COMMAND_LINE,
                    "args": args,
                    "length": _polyline_length(args),
                    "semantic": int(semantic_id),
                    "instance": int(instance_id),
                }
            )
    return primitives


def _row_runs(mask: "np.ndarray") -> List[Tuple[int, int, int]]:
    """Maximal inclusive horizontal runs of ``mask``, ordered row then x."""

    runs: List[Tuple[int, int, int]] = []
    for row in np.flatnonzero(np.any(mask, axis=1)).tolist():
        padded = np.concatenate(([False], mask[row], [False]))
        # A transition at position ``j`` compares ``padded[j]`` with
        # ``padded[j + 1]``: turning on starts a run, turning off ends one.
        edges = np.flatnonzero(padded[1:] != padded[:-1]).tolist()
        for start, after in zip(edges[0::2], edges[1::2]):
            runs.append((int(row), int(start), int(after) - 1))
    return runs


def _emit_mask_runs(
    mask: "np.ndarray",
    semantic_id: int,
    instance_id: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> List[Dict[str, Any]]:
    """Emit one Line per maximal inclusive horizontal run of ``mask``.

    The four samples are ``_padding`` over the run's two inclusive endpoints,
    transformed by the same explicit affine as the skeleton emitter.  A run
    keeps its exact span, and a singleton run repeats one point four times.
    No skeleton, RDP or fallback participates on this path.
    """

    primitives: List[Dict[str, Any]] = []
    for row, start, stop in _row_runs(mask):
        source = np.asarray(
            [[float(start), float(row)], [float(stop), float(row)]], dtype=float
        )
        view_points = _transform(source, scale_x, scale_y, offset_x, offset_y)
        flat = [float(value) for value in view_points.reshape(-1)]
        args = _padding(flat)
        primitives.append(
            {
                "command": COMMAND_LINE,
                "args": args,
                "length": _polyline_length(args),
                "semantic": int(semantic_id),
                "instance": int(instance_id),
            }
        )
    return primitives


def vectorize_masks(
    semantic: Any,
    instance: Any,
    scale_x: Any,
    scale_y: Any,
    offset_x: Any = 0.0,
    offset_y: Any = 0.0,
    viewbox: Sequence[Any] = (0.0, 0.0, 100.0, 100.0),
    tol_px: Any = 0.5,
    *,
    emitter: str = "skeleton",
) -> Dict[str, Any]:
    """Vectorize semantic/instance rasters into the official 12-field dict.

    ``semantic``: ``H x W`` integer array of JSON class ids ``0..35``.
    ``instance``: matching array of local instance ids; Thing classes need
    ``>= 0`` while Stuff ids are ignored.
    ``scale_x``/``scale_y``/``offset_x``/``offset_y``: explicit pixel ->
    viewBox affine, ``x_view = x_px * scale_x + offset_x`` (same for y).
    ``viewbox``: ``(x, y, width, height)``; only width/height are used, and
    they become the document's positive integer ``width``/``height``.
    ``tol_px``: RDP tolerance in source pixel units (default 0.5).
    ``emitter``: ``"skeleton"`` (default, the unchanged line-skeleton baseline)
    or ``"runs"`` (opt-in maximal horizontal row runs; ``tol_px`` is validated
    but unused because runs are never simplified).
    """

    _require_numpy()
    if not isinstance(emitter, str):
        raise VectorizeError(f"emitter_not_a_string:{type(emitter).__name__}")
    if emitter not in EMITTERS:
        raise VectorizeError(f"emitter_unsupported:{emitter}")
    semantics = _label_array(semantic, "semantic")
    instances = _label_array(instance, "instance")
    if semantics.shape != instances.shape:
        raise VectorizeError(
            "shape_mismatch:"
            f"semantic={semantics.shape},instance={instances.shape}"
        )
    if int(semantics.min()) < 0 or int(semantics.max()) > IGNORE_ID:
        raise VectorizeError(
            f"semantic_out_of_range:{int(semantics.min())}..{int(semantics.max())}"
        )

    scale_x_value = _finite_float(scale_x, "scale_x")
    scale_y_value = _finite_float(scale_y, "scale_y")
    if scale_x_value == 0.0 or scale_y_value == 0.0:
        raise VectorizeError("scale_must_be_non_zero")
    offset_x_value = _finite_float(offset_x, "offset_x")
    offset_y_value = _finite_float(offset_y, "offset_y")
    tolerance = _finite_float(tol_px, "tol_px")
    if tolerance < 0.0:
        raise VectorizeError(f"tol_px_negative:{tolerance}")

    box = tuple(viewbox)
    if len(box) != 4:
        raise VectorizeError(f"viewbox_not_four_values:{len(box)}")
    for position, value in enumerate(box):
        _finite_float(value, f"viewbox[{position}]")
    width = _viewbox_size(box[2], "viewbox_width")
    height = _viewbox_size(box[3], "viewbox_height")

    def emit_mask(
        mask: "np.ndarray", semantic_id: int, instance_id: int
    ) -> List[Dict[str, Any]]:
        if emitter == "runs":
            return _emit_mask_runs(
                mask,
                semantic_id,
                instance_id,
                scale_x_value,
                scale_y_value,
                offset_x_value,
                offset_y_value,
            )
        return _emit_mask(
            mask,
            semantic_id,
            instance_id,
            scale_x_value,
            scale_y_value,
            offset_x_value,
            offset_y_value,
            tolerance,
        )

    primitives: List[Dict[str, Any]] = []
    next_thing_instance = 0
    for semantic_value in np.unique(semantics).tolist():
        semantic_id = int(semantic_value)
        if semantic_id == IGNORE_ID:
            continue
        if semantic_id < THING_MIN or semantic_id > STUFF_MAX:
            raise VectorizeError(f"semantic_out_of_range:{semantic_id}")
        semantic_mask = semantics == semantic_id
        if semantic_id >= STUFF_MIN:
            # Stuff is merged per class; every local instance id becomes -1.
            primitives.extend(
                emit_mask(
                    semantic_mask,
                    semantic_id,
                    -1,
                )
            )
            continue

        local_ids = np.unique(instances[semantic_mask])
        if local_ids.size and int(local_ids.min()) < 0:
            raise VectorizeError(
                f"thing_instance_negative:semantic={semantic_id},"
                f"min={int(local_ids.min())}"
            )
        for local_value in local_ids.tolist():
            local_id = int(local_value)
            mask = semantic_mask & (instances == local_id)
            output_instance = next_thing_instance
            next_thing_instance += 1
            primitives.extend(
                emit_mask(
                    mask,
                    semantic_id,
                    output_instance,
                )
            )

    commands: List[int] = []
    args: List[List[float]] = []
    lengths: List[float] = []
    semantic_ids: List[int] = []
    instance_ids: List[int] = []
    rgb: List[List[int]] = []
    layer_ids: List[int] = []
    widths: List[float] = []
    for primitive in primitives:
        commands.append(int(primitive["command"]))
        args.append([float(item) for item in primitive["args"]])
        lengths.append(float(primitive["length"]))
        semantic_ids.append(int(primitive["semantic"]))
        instance_ids.append(int(primitive["instance"]))
        rgb.append(list(DEFAULT_RGB))
        layer_ids.append(int(DEFAULT_LAYER_ID))
        widths.append(float(DEFAULT_STROKE_WIDTH))

    # obj_cts/boxes are recomputed from exactly the emitted samples, in the
    # same first-occurrence instance order the validator uses.
    instance_order: List[int] = []
    instance_semantic: Dict[int, int] = {}
    instance_points: Dict[int, List[Tuple[float, float]]] = {}
    for index, instance_id in enumerate(instance_ids):
        if instance_id < 0:
            continue
        if instance_id not in instance_semantic:
            instance_order.append(instance_id)
            instance_semantic[instance_id] = semantic_ids[index]
            instance_points[instance_id] = []
        if instance_semantic[instance_id] != semantic_ids[index]:
            raise VectorizeError(
                f"instance_semantic_mismatch:{instance_id}"
            )
        for offset in range(0, 8, 2):
            instance_points[instance_id].append(
                (args[index][offset], args[index][offset + 1])
            )

    obj_cts: List[List[float]] = []
    boxes: List[List[float]] = []
    for instance_id in instance_order:
        points = instance_points[instance_id]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        obj_cts.append(
            [
                float((x_min + x_max) / 2.0),
                float((y_min + y_max) / 2.0),
                0.0,
                int(instance_id),
            ]
        )
        boxes.append(
            [
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
                int(instance_semantic[instance_id]),
            ]
        )

    document: Dict[str, Any] = {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "width": int(width),
        "height": int(height),
        "obj_cts": obj_cts,
        "boxes": boxes,
        "rgb": rgb,
        "layerIds": layer_ids,
        "widths": widths,
    }
    errors = validate_document(document)
    if errors:
        raise VectorizeError("invalid_document:" + ",".join(errors))
    return document
