"""Experimental PNG-only short skeleton-token extractor.

This is a bounded, opt-in diagnostic: it reads one PNG (or a synthetic mask),
composites alpha over white, thresholds visible ink with the same 32-level
deviation convention as :mod:`cadvec.png_frontend`, thins the mask with the
deterministic 8-connected Zhang-Suen implementation from
:mod:`cadvec.png_raster_vectorize`, traces the 8-neighbour skeleton graph and
emits ordered polylines whose arc length is at most
:data:`MAX_TOKEN_ARC_PX` pixels.  Remainders shorter than the limit are kept.

Deliberate boundaries:

* **PNG-only, single input, no resize**: one path is opened; dimensions and
  file bytes are bounded by :class:`png_frontend.FrontendConfig` defaults
  (40M pixels / 64 MiB) before decoding.
* **Tokens are geometry only**: no semantics, no instances, no association,
  no arc/length sweep.  ``source_kind`` is always ``"png_skeleton"``.
* **No disconnected-ink jumps**: tokens follow edges of one 8-neighbour graph;
  a diagonal edge is suppressed when it would cut across a filled triangle,
  matching ``png_raster_vectorize._extract_chains``.
* **Deterministic**: sorted vertices, sorted incident edges, canonical token
  order and zero-padded ids.
* **Experimental**: the output carries ``experimental: True`` and a capability
  block stating that nothing is inferred beyond ink geometry.

Public entry points: :func:`extract_png`, :func:`extract_mask`.
Schema: ``cadvec.png_skeleton_frontend`` version ``1``.
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import png_frontend as frontend
from . import png_raster_vectorize as vectorizer

SCHEMA = "cadvec.png_skeleton_frontend"
SCHEMA_VERSION = 1

#: Same strict dark-ink deviation as ``png_frontend``'s default foreground.
FOREGROUND_THRESHOLD = 32
BACKGROUND_VALUE = 255

#: Longest token arc in PNG pixels; remainders are preserved as shorter tokens.
MAX_TOKEN_ARC_PX = 8.0
#: A token keeps at least one skeleton pixel; a lone vertex has arc length 0.
MIN_TOKEN_ARC_PX = 1.0
DIAGONAL_STEP_PX = math.sqrt(2.0)

SOURCE_KIND = "png_skeleton"
_ARC_EPS = 1e-12
_PNG_HEADER_BYTES = 24  # signature(8) + chunk length(4) + "IHDR"(4) + dims(8)

#: Input guard only: frontend resource defaults, not a GPU/memory guarantee.
_LIMITS = frontend.FrontendConfig()
MAX_PIXELS = _LIMITS.max_pixels
MAX_FILE_BYTES = _LIMITS.max_file_bytes
_LIMITS_SOURCE = "png_frontend.FrontendConfig defaults"


class PngSkeletonError(Exception):
    """Explicit rejection of an input that cannot be safely tokenized."""


# ---------------------------------------------------------------------------
# input guards


def _check_png_input(png_path: str) -> Tuple[int, int, Optional[int]]:
    """Bound file bytes and IHDR dimensions before any decode.

    Returns ``(width, height, file_bytes)``.  Mirrors the inference guard: the
    stat check runs first, then the 24-byte PNG prefix is parsed.
    """

    try:
        file_bytes: Optional[int] = os.path.getsize(png_path)
    except OSError:
        file_bytes = None
    if file_bytes is not None and file_bytes > MAX_FILE_BYTES:
        raise PngSkeletonError(
            f"PNG file is {file_bytes} bytes, above max_file_bytes="
            f"{MAX_FILE_BYTES}; nothing was decoded"
        )
    try:
        with open(png_path, "rb") as handle:
            header = handle.read(_PNG_HEADER_BYTES)
    except OSError as error:
        raise PngSkeletonError(f"PNG cannot be read: {error}") from error
    if len(header) < _PNG_HEADER_BYTES:
        raise PngSkeletonError(
            f"PNG header is {len(header)} bytes, shorter than {_PNG_HEADER_BYTES}"
        )
    if header[:8] != frontend.PNG_SIGNATURE:
        raise PngSkeletonError("input must be PNG")
    if header[12:16] != b"IHDR":
        raise PngSkeletonError("PNG has no IHDR header; nothing was decoded")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise PngSkeletonError(
            f"PNG dimensions must be positive; got {width}x{height}"
        )
    if width * height > MAX_PIXELS:
        raise PngSkeletonError(
            f"image is {width}x{height} = {width * height} pixels, above "
            f"max_pixels={MAX_PIXELS}; no resize is applied"
        )
    return int(width), int(height), file_bytes


def _check_mask_input(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise PngSkeletonError(f"mask must be 2-D, got shape {array.shape}")
    if array.size == 0:
        raise PngSkeletonError("mask must not be empty")
    if array.size > MAX_PIXELS:
        raise PngSkeletonError(
            f"mask is {array.shape[0]}x{array.shape[1]} = {array.size} pixels, "
            f"above max_pixels={MAX_PIXELS}"
        )
    if array.dtype == bool:
        return array
    return array != 0


# ---------------------------------------------------------------------------
# raster to ink mask


def _foreground_mask_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite over white and threshold dark ink like the frontend."""

    colours = rgba[..., :3].astype(np.float32)
    alpha = (rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    composed = colours * alpha + float(BACKGROUND_VALUE) * (1.0 - alpha)
    grey = (
        composed[..., 0] * 0.299
        + composed[..., 1] * 0.587
        + composed[..., 2] * 0.114
    )
    return (float(BACKGROUND_VALUE) - grey) > float(FOREGROUND_THRESHOLD)


# ---------------------------------------------------------------------------
# skeleton graph


def _skeletonize(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Thin the ink bounding box and return ``(skeleton, row0, column0)``.

    Restricting thinning to the foreground bounding box keeps the working set
    proportional to ink extent, not canvas area.  Empty padding outside the box
    is identical for the thinning neighbourhood, so coordinates offset back to
    the PNG frame unchanged.
    """

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    row0 = int(rows[0])
    row1 = int(rows[-1]) + 1
    column0 = int(columns[0])
    column1 = int(columns[-1]) + 1
    sub = np.ascontiguousarray(mask[row0:row1, column0:column1])
    skeleton = vectorizer._zhang_suen(sub)
    if not skeleton.any():  # defensive: never drop ink silently
        skeleton = sub
    return skeleton, row0, column0


def _adjacency(skeleton: np.ndarray) -> Tuple[
    List[Tuple[int, int]], List[List[int]], List[Tuple[int, int]]
]:
    """Build the deterministic 8-neighbour graph used by the tracer.

    A diagonal edge is omitted when either bridging orthogonal pixel exists, so
    a one-pixel rectangular corner is not short-circuited.  Vertices are the
    row-major skeleton pixels ``(x, y)``; neighbours are sorted by vertex index.
    """

    pixels = [
        (int(column), int(row)) for row, column in np.argwhere(skeleton)
    ]
    index = {pixel: position for position, pixel in enumerate(pixels)}
    neighbours: List[List[int]] = []
    for x, y in pixels:
        found: List[int] = []
        for dx, dy in vectorizer.EIGHT_NEIGHBORS:
            neighbour = index.get((x + dx, y + dy))
            if neighbour is None:
                continue
            if dx and dy and (
                (x + dx, y) in index or (x, y + dy) in index
            ):
                continue
            found.append(neighbour)
        neighbours.append(sorted(found))
    edges: List[Tuple[int, int]] = []
    for position, found in enumerate(neighbours):
        for neighbour in found:
            if neighbour > position:
                edges.append((position, neighbour))
    return pixels, neighbours, edges


def _edge_key(left: int, right: int) -> Tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _trace_chains(
    neighbours: List[List[int]],
) -> Tuple[List[List[int]], Dict[str, int]]:
    """Split the graph into chains, consuming every edge exactly once.

    Open chains start at endpoints and junctions (so a junction vertex is the
    shared endpoint of every branch).  Remaining edges belong to closed loops
    and are walked back to their start vertex.  A final greedy pass guarantees
    completeness even for degenerate adjacency; it is expected to contribute
    nothing and is counted in ``fallback_chains``.
    """

    degrees = [len(found) for found in neighbours]
    used: set = set()
    chains: List[List[int]] = []

    def walk(start: int, first: int) -> List[int]:
        used.add(_edge_key(start, first))
        path = [start, first]
        previous, current = start, first
        while degrees[current] == 2:
            options = [item for item in neighbours[current] if item != previous]
            if not options:
                break
            following = options[0]
            key = _edge_key(current, following)
            if key in used:
                break
            used.add(key)
            path.append(following)
            previous, current = current, following
        return path

    # Open chains: endpoints (degree 1) and junctions (degree >= 3).
    for start in range(len(neighbours)):
        if degrees[start] == 0:
            chains.append([start])
            continue
        if degrees[start] == 2:
            continue
        for neighbour in neighbours[start]:
            if _edge_key(start, neighbour) in used:
                continue
            chains.append(walk(start, neighbour))

    # Closed loops: every vertex on the loop has degree two.
    for start in range(len(neighbours)):
        for neighbour in neighbours[start]:
            key = _edge_key(start, neighbour)
            if key in used:
                continue
            used.add(key)
            path = [start, neighbour]
            previous, current = start, neighbour
            while current != start and degrees[current] == 2:
                options = [
                    item for item in neighbours[current] if item != previous
                ]
                if not options:
                    break
                following = options[0]
                next_key = _edge_key(current, following)
                if next_key in used:
                    break
                used.add(next_key)
                path.append(following)
                previous, current = current, following
            chains.append(path)

    # Completeness fallback; normally unreachable.
    fallback = 0
    for start in range(len(neighbours)):
        for neighbour in neighbours[start]:
            key = _edge_key(start, neighbour)
            if key in used:
                continue
            fallback += 1
            used.add(key)
            path = [start, neighbour]
            previous, current = start, neighbour
            while True:
                options = [
                    item
                    for item in neighbours[current]
                    if item != previous
                    and _edge_key(current, item) not in used
                ]
                if not options:
                    break
                following = options[0]
                used.add(_edge_key(current, following))
                path.append(following)
                previous, current = current, following
            chains.append(path)

    diagnostics = {
        "endpoint_vertices": sum(1 for degree in degrees if degree == 1),
        "junction_vertices": sum(1 for degree in degrees if degree >= 3),
        "isolated_vertices": sum(1 for degree in degrees if degree == 0),
        "fallback_chains": fallback,
    }
    return chains, diagnostics


# ---------------------------------------------------------------------------
# token splitting


def _split_chain(
    points: Sequence[Tuple[int, int]], max_arc: float
) -> List[Tuple[List[Tuple[int, int]], float]]:
    """Split one chain into consecutive vertices with arc <= ``max_arc``.

    The split vertex is shared by both tokens but no edge is duplicated; a
    remainder shorter than ``max_arc`` (including a single edge) is kept.  A
    lone vertex becomes a zero-length token so short ink is never dropped.
    """

    if len(points) == 1:
        return [([points[0]], 0.0)]
    tokens: List[Tuple[List[Tuple[int, int]], float]] = []
    current: List[Tuple[int, int]] = [points[0]]
    accumulated = 0.0
    for point in points[1:]:
        step = math.hypot(
            float(point[0] - current[-1][0]), float(point[1] - current[-1][1])
        )
        if current and accumulated + step > max_arc + _ARC_EPS:
            tokens.append((current, accumulated))
            current = [current[-1], point]
            accumulated = step
        else:
            current.append(point)
            accumulated += step
    tokens.append((current, accumulated))
    return tokens


def _token_sort_key(token: Dict[str, Any]) -> Tuple[Any, ...]:
    points = token["points"]
    return (
        points[0][1],
        points[0][0],
        points[-1][1],
        points[-1][0],
        len(points),
        token["length"],
    )


# ---------------------------------------------------------------------------
# shared extraction path


def _extract(
    mask: np.ndarray,
    *,
    input_block: Dict[str, Any],
) -> Dict[str, Any]:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    diagnostics: Dict[str, Any] = {
        "foreground_pixels": int(mask.sum()),
        "input_pixels": int(width * height),
        "max_pixels": MAX_PIXELS,
        "max_file_bytes": MAX_FILE_BYTES,
        "graph_vertices": 0,
        "graph_edges": 0,
        "consumed_edges": 0,
        "dropped_edges": 0,
        "endpoint_vertices": 0,
        "junction_vertices": 0,
        "isolated_vertices": 0,
        "fallback_chains": 0,
        "chains": 0,
        "closed_loops": 0,
        "single_vertex_chains": 0,
        "skeleton_pixels": 0,
        "tokens": 0,
        "single_pixel_tokens": 0,
        "split_tokens": 0,
        "max_token_arc_px": 0.0,
        "total_token_arc_px": 0.0,
    }
    tokens: List[Dict[str, Any]] = []
    if diagnostics["foreground_pixels"]:
        skeleton, row0, column0 = _skeletonize(mask)
        diagnostics["skeleton_pixels"] = int(skeleton.sum())
        pixels, neighbours, edges = _adjacency(skeleton)
        diagnostics["graph_vertices"] = len(pixels)
        diagnostics["graph_edges"] = len(edges)
        chains, chain_diagnostics = _trace_chains(neighbours)
        diagnostics.update(chain_diagnostics)
        diagnostics["chains"] = len(chains)
        diagnostics["consumed_edges"] = sum(
            max(0, len(chain) - 1) for chain in chains
        )
        diagnostics["dropped_edges"] = int(
            diagnostics["graph_edges"] - diagnostics["consumed_edges"]
        )

        raw_tokens: List[Tuple[List[Tuple[int, int]], float]] = []
        for chain in chains:
            if not chain:
                continue
            points = [
                (pixels[position][0] + column0, pixels[position][1] + row0)
                for position in chain
            ]
            if len(points) == 1:
                diagnostics["single_vertex_chains"] += 1
            if len(chain) > 2 and chain[0] == chain[-1]:
                diagnostics["closed_loops"] += 1
            raw_tokens.extend(_split_chain(points, MAX_TOKEN_ARC_PX))

        for points, length in raw_tokens:
            token = {
                "id": "",
                "points": [[int(x), int(y)] for x, y in points],
                "length": round(float(length), 6),
                "source_kind": SOURCE_KIND,
            }
            tokens.append(token)
        tokens.sort(key=_token_sort_key)
        for index, token in enumerate(tokens):
            token["id"] = f"t{index:04d}"

        diagnostics["tokens"] = len(tokens)
        diagnostics["single_pixel_tokens"] = sum(
            1 for token in tokens if len(token["points"]) == 1
        )
        diagnostics["split_tokens"] = sum(
            1 for token in tokens if len(token["points"]) > 2
        )
        if tokens:
            lengths = [float(token["length"]) for token in tokens]
            diagnostics["max_token_arc_px"] = round(max(lengths), 6)
            diagnostics["total_token_arc_px"] = round(sum(lengths), 6)

    status = "ok" if tokens else "empty_input"
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experimental": True,
        "source_kind": SOURCE_KIND,
        "input": input_block,
        "config": {
            "background": "white",
            "background_value": BACKGROUND_VALUE,
            "polarity": "dark_on_light",
            "foreground_threshold": FOREGROUND_THRESHOLD,
            "skeleton": "zhang_suen_8",
            "thinning_scope": "foreground_bounding_box",
            "diagonal_policy": "suppress diagonal edge across a filled triangle",
            "junction_policy": "each incident edge starts its own chain at degree != 2",
            "closed_loop_policy": "cycle chain starts and ends at the same vertex",
            "token_split_policy": "consecutive vertices; cumulative arc <= max",
            "max_token_arc_px": MAX_TOKEN_ARC_PX,
            "min_token_arc_px": MIN_TOKEN_ARC_PX,
            "diagonal_step_px": round(DIAGONAL_STEP_PX, 6),
            "no_length_sweep": True,
            "no_resize": True,
            "max_pixels": MAX_PIXELS,
            "max_file_bytes": MAX_FILE_BYTES,
            "limits_source": _LIMITS_SOURCE,
        },
        "capability": {
            "experimental": True,
            "inferred_semantics": False,
            "inferred_instances": False,
            "association": None,
            "note": (
                "experimental ink-geometry tokens only; no semantic or instance "
                "label is inferred and no association is performed"
            ),
        },
        "tokens": tokens,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# public API


def extract_mask(mask: Any) -> Dict[str, Any]:
    """Tokenize a 2-D synthetic mask (any nonzero value is ink).

    Coordinates are PNG-frame ``[x, y]`` pixel indices, so callers can compare
    them with the mask directly.  No resize, padding or semantic inference.
    """

    ink = _check_mask_input(mask)
    height, width = int(ink.shape[0]), int(ink.shape[1])
    return _extract(
        ink,
        input_block={
            "source": "mask",
            "file_name": None,
            "file_bytes": None,
            "width": width,
            "height": height,
        },
    )


def extract_png(png_path: str) -> Dict[str, Any]:
    """Read one PNG, composite alpha over white and emit short skeleton tokens."""

    width, height, file_bytes = _check_png_input(str(png_path))
    try:
        with Image.open(png_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception as error:  # Pillow raises several decode-specific types
        raise PngSkeletonError(f"PNG decode failed: {error}") from error
    if rgba.shape[0] != height or rgba.shape[1] != width:
        raise PngSkeletonError(
            "decoded PNG size "
            f"{rgba.shape[1]}x{rgba.shape[0]} disagrees with header {width}x{height}"
        )
    mask = _foreground_mask_from_rgba(rgba)
    return _extract(
        mask,
        input_block={
            "source": "path",
            "file_name": os.path.basename(str(png_path)),
            "file_bytes": file_bytes,
            "width": width,
            "height": height,
            "decoded_mode": "RGBA",
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Minimal CLI: print the JSON result for one PNG path."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Experimental PNG skeleton token extractor (no semantics)."
    )
    parser.add_argument("png", help="path to one PNG file")
    args = parser.parse_args(argv)
    try:
        result = extract_png(args.png)
    except PngSkeletonError as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
