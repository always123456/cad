#!/usr/bin/env python
"""Ground-truth-free ink refinement of predicted instance masks (arms A6 / A1).

The shipped Mask R-CNN mask is a fat band; the drawing's true strokes are 1 px in
the canonical raster.  The input PNG carries those strokes, so the predicted mask
only has to say *which* ink belongs to the instance; the geometry then comes from
the image.  Nothing here scores anything and nothing here reads ground truth.

Rules (``RULE_*``), applied to the native-raster mask ``M`` and the PNG ink map
``I`` of the same shape:

``a6`` -- ink components (8-connected, taken from the ORIGINAL PNG at its native
    resolution) whose share of pixels inside ``M`` is at least ``coverage``
    (``COMPONENT_COVERAGE`` = 0.5 by default, ``--refine-ratio`` on the CLI) are
    kept whole, and their union is the new mask.  Empty -> ``M & I``.  Still
    empty -> ``M``.  (The 0.5 arm: 0.036694672 -> 0.060834794 micro PQ on the 387
    held-out records, both halves improving.  A ratio sweep over the same 387
    records found 0.0 -- keep every ink component the mask touches -- reaches
    0.116281992 at score 0.99 / cap 20 with the same composite128 ink, so the
    ratio and not the score threshold was the defect.)

The comparison is always ``fraction >= coverage``, inclusive, so ``coverage`` 0.0
keeps every component that is present in the mask's crop (i.e. every ink
component the predicted mask touches) and 1.0 keeps only fully covered ones.

``a1`` -- ``M & I``; empty -> ``M``.  The threshold-robust variant (0.055837724),
    one switch away because A6's bigger gain depends on the ink definition.

Invariants preserved from the measuring lane (``cad-ink-refine01``):

* the ink map is built from the PNG itself, never from a resized or re-quantised
  copy, so components are those of the native raster;
* a mask is never emptied: every branch falls back, because the frozen emitter
  skips empty instances and the emitted instance count must not move;
* the mask is refined on the native raster, before the delivery CLI pads a
  crop-residual (1000x999 / 999x1000) record back to its square canvas;
* this module never rasterises vector geometry.  The bit-exact truncation rule
  (``int(v)``, not ``int(v + 0.5)``) lives in the frozen emitter and is only
  asserted by the tests here, never re-implemented.
"""

from __future__ import annotations

import numbers

import numpy as np

__all__ = [
    "RULES",
    "INK_RULES",
    "DEFAULT_RULE",
    "DEFAULT_INK_RULE",
    "COMPONENT_COVERAGE",
    "COMPONENT_COVERAGE_RANGE",
    "validate_coverage",
    "new_stats",
    "ink_from_rgba",
    "ink_from_png",
    "label8",
    "refine_mask",
    "refine_masks",
]

RULE_NONE = "none"
RULE_A6 = "a6"
RULE_A1 = "a1"
RULES = (RULE_NONE, RULE_A6, RULE_A1)
DEFAULT_RULE = RULE_A6

# ``nonzero`` is the delivered definition: the PNG is RGBA with a black
# background and coloured strokes, so ink is every pixel with any nonzero colour
# channel.  ``composite128`` is the measuring lane's operating point
# (``min(R,G,B) < 128`` after white alpha compositing); it is kept selectable so
# the A6 numbers can be reproduced and compared, and because it is ~1.3-2.8x
# thinner on the same PNGs.
INK_NONZERO = "nonzero"
INK_COMPOSITE128 = "composite128"
INK_RULES = (INK_NONZERO, INK_COMPOSITE128)
DEFAULT_INK_RULE = INK_NONZERO

INK_DEFINITIONS = {
    INK_NONZERO: "any of R,G,B nonzero on the raw RGBA PNG (alpha ignored), native raster",
    INK_COMPOSITE128: "min(R,G,B) < 128 after compositing alpha over white, native raster",
}

# Component-inclusion ratio: an 8-connected ink component of the mask's crop is
# kept when ``inside / component_size >= ratio``.  Inclusive, so 0.0 keeps every
# component present in the crop (the measured best arm) and 1.0 only fully
# covered ones.  The default is the 0.5 arm this package shipped; it is a knob,
# never a magic number inside the rule.
COMPONENT_COVERAGE = 0.5
COMPONENT_COVERAGE_RANGE = (0.0, 1.0)


def validate_coverage(ratio=None):
    """The component-inclusion ratio as a float in [0.0, 1.0]; NaN/inf refused.

    ``None`` means the default (0.5).  Note that 0.0 is a *valid* value and must
    not be treated as "not supplied": the measuring lane's best arm is 0.0.
    """

    if ratio is None:
        return COMPONENT_COVERAGE
    if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
        raise ValueError(
            "component coverage refused: a real number is required, got %s %r"
            % (type(ratio).__name__, ratio)
        )
    number = float(ratio)
    if number != number or number in (float("inf"), float("-inf")):  # NaN / +-inf
        raise ValueError("component coverage refused: not finite (%r)" % (number,))
    low, high = COMPONENT_COVERAGE_RANGE
    if number < low or number > high:
        raise ValueError(
            "component coverage refused: must be in %r..%r inclusive, got %r" % (low, high, number)
        )
    return number


# --------------------------------------------------------------------------- #
# ink
# --------------------------------------------------------------------------- #
def ink_from_rgba(rgba, rule=DEFAULT_INK_RULE):
    """Boolean ink map of an ``(H, W, 4)`` (or ``(H, W, 3)``) uint8 array."""

    if rule not in INK_RULES:
        raise ValueError("unknown ink rule %r (expected one of %r)" % (rule, INK_RULES))
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("ink needs an (H, W, 3+) array, got shape %r" % (array.shape,))
    rgb = array[..., :3].astype(np.int32, copy=False)
    if rule == INK_NONZERO:
        return np.asarray((rgb != 0).any(axis=2), dtype=bool)
    if array.shape[2] >= 4:
        alpha = array[..., 3:4].astype(np.int32, copy=False)
    else:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.int32)
    composited = (rgb * alpha + 255 * (255 - alpha)) // 255
    return np.asarray(composited.min(axis=2) < 128, dtype=bool)


def ink_from_png(path, rule=DEFAULT_INK_RULE):
    """Ink map of a PNG at its native raster, plus the raster size.

    Returns ``(ink, (width, height))``.  The PNG is read once, converted to RGBA
    and never resized: components therefore come from the original image.
    """

    from PIL import Image  # noqa: PLC0415 - keeps the module importable without Pillow

    with Image.open(path) as handle:
        rgba = np.asarray(handle.convert("RGBA"))
    height, width = int(rgba.shape[0]), int(rgba.shape[1])
    return ink_from_rgba(rgba, rule), (width, height)


# --------------------------------------------------------------------------- #
# 8-connected components
# --------------------------------------------------------------------------- #
def _label8_scipy(mask):
    from scipy import ndimage  # noqa: PLC0415

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32, copy=False), int(count)


def _label8_python(mask):
    """Two-pass 8-connected labelling over horizontal runs (no scipy needed).

    Run intervals on adjacent rows are 8-adjacent when they overlap after each is
    widened by one pixel, i.e. ``b0 <= a1 and a0 <= b1`` for half-open runs.
    """

    height = int(mask.shape[0])
    row_runs = []
    total = 0
    for y in range(height):
        row = mask[y]
        if not row.any():
            row_runs.append(())
            continue
        columns = np.flatnonzero(row)
        breaks = np.flatnonzero(np.diff(columns) > 1)
        starts = [0] + (breaks + 1).tolist()
        stops = breaks.tolist() + [columns.size - 1]
        runs = [(int(columns[s]), int(columns[e]) + 1) for s, e in zip(starts, stops)]
        row_runs.append(tuple((total + i, run) for i, run in enumerate(runs)))
        total += len(runs)

    parent = list(range(total))

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for y in range(1, height):
        previous = row_runs[y - 1]
        current = row_runs[y]
        if not previous or not current:
            continue
        i = 0
        for run_id, (a0, a1) in current:
            while i < len(previous) and previous[i][1][1] < a0:
                i += 1
            j = i
            while j < len(previous) and previous[j][1][0] <= a1:
                union(previous[j][0], run_id)
                j += 1

    labels = np.zeros(mask.shape, dtype=np.int32)
    remap = {}
    for y, runs in enumerate(row_runs):
        for run_id, (a0, a1) in runs:
            root = find(run_id)
            if root not in remap:
                remap[root] = len(remap) + 1
            labels[y, a0:a1] = remap[root]
    return labels, len(remap)


def label8(mask, force_numpy=False):
    """8-connected labels of ``mask`` plus the component count.

    scipy is used when importable (it is present in both the delivery interpreter
    and the CUDA runtime); ``force_numpy=True`` exercises the pure-numpy path.
    """

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("label8 needs a 2-D mask, got shape %r" % (array.shape,))
    if not array.any():
        return np.zeros(array.shape, dtype=np.int32), 0
    if force_numpy:
        return _label8_python(array)
    try:
        return _label8_scipy(array)
    except ImportError:  # pragma: no cover - scipy is present in both runtimes
        return _label8_python(array)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
_STAT_KEYS = (
    "instances",
    "changed",
    "components",
    "kept_components",
    "instances_without_component",  # A6 found no ink component to keep
    "a6_to_a1",                     # A6 empty -> M & I
    "a1_to_original",               # M & I empty -> M (inside the A6 chain)
    "a1_rule_empty",                # the A1 rule fell back to M
    "dropped_instances",            # must stay 0: refinement never empties a mask
)


def new_stats():
    """A zeroed counter block; pass one to ``refine_masks`` to accumulate."""

    return {key: 0 for key in _STAT_KEYS}


def _bump(stats, key):
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + 1


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def refine_mask(mask, ink, rule=DEFAULT_RULE, stats=None, coverage=None):
    """Refine one native-raster boolean mask; returns a boolean mask of the same shape.

    ``coverage`` is the component-inclusion ratio for ``a6`` (0.0..1.0 inclusive,
    default ``COMPONENT_COVERAGE``); it is validated before anything is read, so a
    bad value cannot half-run a batch.
    """

    if rule not in RULES:
        raise ValueError("unknown refine rule %r (expected one of %r)" % (rule, RULES))
    ratio = validate_coverage(coverage)
    original = np.asarray(mask, dtype=bool)
    if original.ndim != 2:
        raise ValueError("refine needs a 2-D mask, got shape %r" % (original.shape,))
    if rule == RULE_NONE:
        return original

    ink_array = np.asarray(ink, dtype=bool)
    if ink_array.shape != original.shape:
        raise ValueError(
            "mask and ink must share the native raster: %r vs %r"
            % (original.shape, ink_array.shape)
        )
    _bump(stats, "instances")
    if not original.any():
        # Nothing to refine and nothing the emitter would have written either way.
        return original

    if rule == RULE_A1:
        refined = original & ink_array
        if not refined.any():
            _bump(stats, "a1_rule_empty")
            refined = original
        _count_change(original, refined, stats)
        return refined

    # A6: whole ink components whose covered share reaches ``ratio``.  The
    # component labeller runs on a crop of the mask bbox widened by 2 px, exactly
    # as the measuring lane did; the ink itself still comes from the original PNG.
    # ``fraction >= ratio`` is inclusive, so 0.0 keeps every component present in
    # the crop -- the sweep lane's measured best arm.
    rows = np.flatnonzero(original.any(axis=1))
    columns = np.flatnonzero(original.any(axis=0))
    y0 = max(0, int(rows[0]) - 2)
    y1 = min(original.shape[0], int(rows[-1]) + 3)
    x0 = max(0, int(columns[0]) - 2)
    x1 = min(original.shape[1], int(columns[-1]) + 3)
    crop_ink = ink_array[y0:y1, x0:x1]
    crop_mask = original[y0:y1, x0:x1]
    labels, count = label8(crop_ink)
    grown = np.zeros(original.shape, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)
        sizes[0] = 0
        inside = np.bincount(labels[crop_mask].ravel(), minlength=count + 1).astype(np.int64)
        present = sizes > 0
        fraction = np.zeros(count + 1, dtype=np.float64)
        fraction[present] = inside[present] / sizes[present]
        kept = np.flatnonzero(present & (fraction >= ratio))
        if stats is not None:
            stats["components"] = int(stats.get("components", 0)) + int(present.sum())
            stats["kept_components"] = int(stats.get("kept_components", 0)) + int(kept.size)
            if kept.size == 0:
                stats["instances_without_component"] = int(stats.get("instances_without_component", 0)) + 1
        if kept.size:
            grown[y0:y1, x0:x1] = np.isin(labels, kept)

    if not grown.any():
        _bump(stats, "a6_to_a1")
        grown = original & ink_array
        if not grown.any():
            _bump(stats, "a1_to_original")
            grown = original

    if not grown.any():  # pragma: no cover - defensive, unreachable while M is non-empty
        _bump(stats, "dropped_instances")
        return original
    _count_change(original, grown, stats)
    return grown


def _count_change(original, refined, stats):
    if stats is None:
        return
    if np.array_equal(original, refined):
        return
    stats["changed"] = int(stats.get("changed", 0)) + 1


def refine_masks(masks, ink, rule=DEFAULT_RULE, stats=None, coverage=None):
    """Refine a list of native-raster masks; returns the list in the same order."""

    if rule not in RULES:
        raise ValueError("unknown refine rule %r (expected one of %r)" % (rule, RULES))
    ratio = validate_coverage(coverage)
    if rule == RULE_NONE:
        return [np.asarray(mask, dtype=bool) for mask in masks]
    return [refine_mask(mask, ink, rule=rule, stats=stats, coverage=ratio) for mask in masks]
