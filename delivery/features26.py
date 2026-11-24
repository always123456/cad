"""The frozen pool's 26 feature columns, computed from our own pipeline output.

Pure functions only: no repository-relative paths, no I/O, no imports from the
delivery CLI.  A caller that already holds, for ONE image at its native raster,

* ``boxes``  -- int xyxy per candidate, ``[x_min, y_min, x_max + 1, y_max + 1]``
               of the *refined* mask (the frozen pool's own convention, see
               ``evidence/v3-features/units.md`` section 1 and 4),
* ``scores`` -- the detector score per candidate,
* ``labels`` -- the raw predicted class ids per candidate (2..29 in this pool),
* ``masks``  -- the refined boolean mask per candidate, native raster (used only
               when ``mask_source="final"``; see ``MASK_SOURCES``),
* ``ink``    -- the boolean ink raster of the PNG (``min(R,G,B) < 128`` after
               alpha compositing over white, i.e. ``refine.py``'s
               ``composite128`` rule), native raster,
* ``width``, ``height`` -- the PNG's native size,

gets the same ``(n, 26)`` float32 matrix the frozen pool's ``feats`` array holds,
in the same column order, and can feed it to ``delivery/model_u250_blend.pkl``.

Mask source (``MASK_SOURCES``)
------------------------------
The pool's mask columns are **not** computed on the refined ``a6`` mask.  They
are computed on a mask rebuilt from the ink raster alone::

    mask = ink AND crop(box widened by 2 px)        # POOL_MASK_PAD = 2
    crop = [y0-2 : y1+2, x0-2 : x1+2]  clipped to the raster

``mask_source="ink_box_pad2"`` (default) rebuilds it and reproduces the pool's
per-column hit rates below (``evidence/v3-features/MASK-PARITY.md``);
``mask_source="final"`` keeps the caller's refined mask (the pre-2026-11-18
behaviour, kept for audits).  ``formulas="legacy"`` restores the old
expressions as well, so ``mask_source="final", formulas="legacy"`` is
bit-identical to the previous implementation.

Column status (pool recipe, hit rate at ``rel < 1e-3`` over the 246-candidate
probe in ``MASK-PARITY.md``):

===================  ==========================================================
score                detector score, reproduced bit-exactly by the GPU rerun
label_id             raw predicted class id
box_w .. box_area_frac  exact formulas over ``boxes`` and ``width``/``height``
mask_area            ``ln(mask_px + 1)`` -- the +1 is the pool's, not a log guard
fill                 ``mask_px / (px_w * px_h)`` -- no +1 in the denominator
ink_in_mask          ``ln(|ink & mask| + 1)``; equals ``mask_area`` because the
                     rebuilt mask is by construction a subset of the ink
ink_frac_mask        ``|ink & mask| / mask_px``, i.e. the pool's {0, 1} indicator
comp_largest_frac    largest 8-connected component of the mask / ``mask_px``
ncomp_ge25           8-connected components of ``ink & box`` (inclusive slice,
                     no pad) -- a *second, tighter* raster, not a px filter
ncomp_total          8-connected components of the mask
thick_mask           ``max(2, 2 * mask_px / mask_perim)``
thick_ink            the same formula on ``|ink & mask|`` with ``mask_perim``
ink_frac_box         ``|ink[y0:y1+1, x0:x1+1]| / (px_w * px_h)`` -- inclusive
                     slice, exclusive denominator (the pool's off-by-one)
colour_logp          **undefined**: written as 0.0 (the top of the pool's range)
colour_margin        **undefined**: written as 0.0
nn_sameclass_dist    degenerate constant 1.0 in the whole pool
nn_sameclass_flag    degenerate constant 0.0 in the whole pool
img_ncand            candidate count of the image
rank_frac            ``rank / (n - 1)`` by descending score, 0.0 when n <= 1
score_gap            ``s_i - s_{i+1}`` in that order, 0.0 for the last row
score_below01        ``1.0`` when ``score < 0.99`` (the column name means 0.99)
mask_perim           4-neighbour boundary *pixel* count of the mask
===================  ==========================================================
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "COLUMN_STATUS",
    "MASK_SOURCES",
    "POOL_MASK_PAD",
    "bbox_from_mask",
    "mask_components",
    "mask_perimeter",
    "compute_features",
    "ink_box_mask",
]

#: Column order of ``evidence/v3-features/shard_rows.npz['feats']``.
FEATURE_NAMES = (
    "score",
    "label_id",
    "box_w",
    "box_h",
    "box_aspect",
    "box_area",
    "box_area_frac",
    "mask_area",
    "fill",
    "ink_in_mask",
    "ink_frac_mask",
    "comp_largest_frac",
    "ncomp_ge25",
    "ncomp_total",
    "thick_mask",
    "thick_ink",
    "ink_frac_box",
    "colour_logp",
    "colour_margin",
    "nn_sameclass_dist",
    "nn_sameclass_flag",
    "img_ncand",
    "rank_frac",
    "score_gap",
    "score_below01",
    "mask_perim",
    "dbox_ratio",
    "dbox_iou",
    "dbox_w",
    "dbox_h",
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

#: How each column stands against the frozen pool (used by the drift report).
COLUMN_STATUS = {
    "score": "exact (candidate record)",
    "label_id": "exact (candidate record)",
    "box_w": "exact formula (boxes + w/h)",
    "box_h": "exact formula (boxes + w/h)",
    "box_aspect": "exact formula (boxes + w/h)",
    "box_area": "exact formula (boxes + w/h)",
    "box_area_frac": "exact formula (boxes + w/h)",
    "mask_area": "pool recipe (ink & crop(box,2)), ln(px+1)",
    "fill": "pool recipe, px/(w*h) without the +1",
    "ink_in_mask": "pool recipe, ln(ib+1); mask is a subset of the ink",
    "ink_frac_mask": "pool recipe, ib/px == the pool's {0,1} indicator",
    "comp_largest_frac": "largest 8-connected component of the mask / px",
    "ncomp_ge25": "8-connected components of ink & box (pad 0), not a px filter",
    "ncomp_total": "8-connected components of the mask",
    "thick_mask": "pool recipe, max(2, 2*px/perim)",
    "thick_ink": "pool recipe, max(2, 2*ib/perim) with the mask's perim",
    "ink_frac_box": "ink over the inclusive box slice / (w*h)",
    "colour_logp": "undefined -- written as 0.0",
    "colour_margin": "undefined -- written as 0.0",
    "nn_sameclass_dist": "degenerate constant 1.0 in the pool",
    "nn_sameclass_flag": "degenerate constant 0.0 in the pool",
    "img_ncand": "exact (candidate count)",
    "rank_frac": "exact formula (rank / (n - 1))",
    "score_gap": "exact formula (s_i - s_i+1)",
    "score_below01": "exact formula (score < 0.99)",
    "mask_perim": "pool recipe, 4-neighbour boundary pixel count",
    "dbox_ratio": "log(decoded_box_area+1) - log(mask_box_area+1); >0 = decoded box is larger",
    "dbox_iou": "IoU(decoded_box, mask_box); decoded box from --head-box regression",
    "dbox_w": "log(decoded_box_width + 1)",
    "dbox_h": "log(decoded_box_height + 1)",
}

#: Accepted ``compute_features(mask_source=...)`` values.
#:
#: ``ink_box_pad2``  the pool's raster: ``ink & crop(box widened by 2 px)``
#: ``final``         the caller's refined mask (previous behaviour)
MASK_SOURCES = ("ink_box_pad2", "final")

#: Crop margin the pool's mask raster uses on each side of the candidate box.
POOL_MASK_PAD = 2

COMPONENT_MIN_AREA = 25  # only used by ``formulas="legacy"``


# --------------------------------------------------------------------------- #
# mask geometry helpers
# --------------------------------------------------------------------------- #
def bbox_from_mask(mask):
    """``[x_min, y_min, x_max + 1, y_max + 1]`` of a boolean mask; ``None`` if empty.

    This is the pool's box convention: ``boxes`` are the refined masks' bounding
    boxes with the exclusive ``+1`` upper bound (units.md section 1).
    """

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("bbox_from_mask needs a 2-D mask, got shape %r" % (array.shape,))
    rows = np.flatnonzero(array.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(array.any(axis=0))
    return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]


def mask_perimeter(mask):
    """Number of mask pixels with at least one 4-neighbour outside the mask.

    The frozen pool's ``mask_perim`` is a boundary *pixel* count: its
    ``perim / mask_px`` median is 0.975 on 1-2 px strokes, which a
    boundary-pixel count reproduces and a geometric perimeter does not.
    """

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask_perimeter needs a 2-D mask, got shape %r" % (array.shape,))
    if not array.any():
        return 0
    padded = np.pad(array, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return int(np.count_nonzero(core & ~interior))


def _label8_runbased(mask):
    """Two-pass 8-connected labelling over horizontal runs (no scipy needed)."""

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


def mask_components(mask):
    """``(labels, sizes)`` of the 8-connected components of a mask (sizes[0] = 0).

    scipy is used when present; the pure-numpy run labeller is the fallback, so
    the module works in a bare interpreter too.
    """

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask_components needs a 2-D mask, got shape %r" % (array.shape,))
    if not array.any():
        return np.zeros(array.shape, dtype=np.int32), np.zeros(1, dtype=np.int64)
    try:
        from scipy import ndimage  # noqa: PLC0415 - optional fast path

        labels, count = ndimage.label(array, structure=np.ones((3, 3), dtype=bool))
        labels = labels.astype(np.int32, copy=False)
    except ImportError:  # pragma: no cover - scipy is present in both runtimes
        labels, count = _label8_runbased(array)
    sizes = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)
    sizes[0] = 0
    return labels, sizes


# --------------------------------------------------------------------------- #
# the pool's mask raster
# --------------------------------------------------------------------------- #
def ink_box_mask(ink, x0, y0, x1, y1, width, height, pad=POOL_MASK_PAD):
    """``(crop, origin)`` -- the pool's mask raster for one candidate.

    ``crop`` is ``ink`` restricted to the box widened by ``pad`` px on every
    side (clipped to the raster), as a tight 2-D boolean array; outside that
    crop the pool's mask is empty, so the crop is a complete description of it.
    ``origin`` is the crop's ``(y, x)`` offset in the full raster.

    The box is the pool's ``[x_min, y_min, x_max + 1, y_max + 1]`` convention,
    so ``y1 + pad`` is ``y_max + 3``: the crop reaches two pixels past the last
    mask row, exactly as the frozen pool's own crop does.
    """

    ink_array = np.asarray(ink, dtype=bool)
    if ink_array.ndim != 2:
        raise ValueError("ink must be a 2-D raster, got shape %r" % (ink_array.shape,))
    top = max(0, int(y0) - int(pad))
    left = max(0, int(x0) - int(pad))
    bottom = min(int(height), int(y1) + int(pad))
    right = min(int(width), int(x1) + int(pad))
    crop = np.zeros((max(0, bottom - top), max(0, right - left)), dtype=bool)
    if crop.size:
        crop[:, :] = ink_array[top:bottom, left:right]
    return crop, (top, left)


def _refined_mask_crop(mask, x0, y0, x1, y1, width, height, pad=POOL_MASK_PAD):
    """``(crop, origin)`` of a caller-supplied mask, widened to cover it fully."""

    array = np.asarray(mask, dtype=bool)
    if array.shape != (int(height), int(width)):
        raise ValueError(
            "refined mask %r does not match the declared %dx%d"
            % (array.shape, int(width), int(height))
        )
    top = max(0, int(y0) - int(pad))
    left = max(0, int(x0) - int(pad))
    bottom = min(int(height), int(y1) + int(pad))
    right = min(int(width), int(x1) + int(pad))
    bounds = bbox_from_mask(array)
    if bounds is not None:
        top = min(top, bounds[1])
        left = min(left, bounds[0])
        bottom = max(bottom, bounds[3])
        right = max(right, bounds[2])
    crop = np.zeros((max(0, bottom - top), max(0, right - left)), dtype=bool)
    if crop.size:
        crop[:, :] = array[top:bottom, left:right]
    return crop, (top, left)


# --------------------------------------------------------------------------- #
# the feature matrix
# --------------------------------------------------------------------------- #
def compute_features(boxes, scores, labels, masks, ink, width, height,
                     mask_source="ink_box_pad2", formulas="pool",
                     decoded_boxes=None):
    """``(n, 26)`` float32 features for one image's candidates.

    ``boxes``/``scores``/``labels``/``masks`` are per candidate and must share
    their order; ``ink`` is the image's native boolean ink raster and
    ``width``/``height`` its native size.  Nothing here reads ground truth.

    ``mask_source`` picks the raster the mask columns are measured on:
    ``"ink_box_pad2"`` (the default) rebuilds the pool's own raster,
    ``ink & crop(box widened by 2 px)``; ``"final"`` keeps the caller's refined
    mask.  ``formulas`` is ``"pool"`` (the pool's exact expressions) or
    ``"legacy"`` (the pre-2026-11-18 expressions, for audits).
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    n = int(scores.size)
    if mask_source not in MASK_SOURCES:
        raise ValueError(
            "mask_source must be one of %r, got %r" % (MASK_SOURCES, mask_source)
        )
    if formulas not in ("pool", "legacy"):
        raise ValueError("formulas must be 'pool' or 'legacy', got %r" % (formulas,))
    if not (len(boxes) == len(labels) == len(masks) == n):
        raise ValueError(
            "feature inputs are ragged: %d boxes, %d labels, %d masks, %d scores"
            % (len(boxes), len(labels), len(masks), n)
        )

    ink_array = np.asarray(ink, dtype=bool)
    if ink_array.shape != (int(height), int(width)):
        raise ValueError(
            "ink raster %r does not match the declared %dx%d"
            % (ink_array.shape, int(width), int(height))
        )

    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    image_area = float(int(width) * int(height))

    # -- candidate-record columns ------------------------------------------ #
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    score_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        score_gap[order[:-1]] = scores[order[:-1]] - scores[order[1:]]
        rank_frac = rank.astype(np.float64) / float(n - 1)
    else:
        rank_frac = np.zeros(n, dtype=np.float64)

    out[:, FEATURE_INDEX["score"]] = scores
    out[:, FEATURE_INDEX["label_id"]] = labels
    out[:, FEATURE_INDEX["img_ncand"]] = float(n)
    out[:, FEATURE_INDEX["rank_frac"]] = rank_frac
    out[:, FEATURE_INDEX["score_gap"]] = score_gap
    out[:, FEATURE_INDEX["score_below01"]] = (scores < 0.99).astype(np.float64)
    out[:, FEATURE_INDEX["nn_sameclass_dist"]] = 1.0
    out[:, FEATURE_INDEX["nn_sameclass_flag"]] = 0.0
    out[:, FEATURE_INDEX["colour_logp"]] = 0.0
    out[:, FEATURE_INDEX["colour_margin"]] = 0.0

    # -- box columns -------------------------------------------------------- #
    px_w = boxes[:, 2] - boxes[:, 0]
    px_h = boxes[:, 3] - boxes[:, 1]
    box_w = np.log(px_w + 1.0)
    box_h = np.log(px_h + 1.0)
    box_area = np.log(px_w * px_h + 1.0)
    out[:, FEATURE_INDEX["box_w"]] = box_w
    out[:, FEATURE_INDEX["box_h"]] = box_h
    out[:, FEATURE_INDEX["box_aspect"]] = box_w - box_h
    out[:, FEATURE_INDEX["box_area"]] = box_area
    out[:, FEATURE_INDEX["box_area_frac"]] = (px_w * px_h) / image_area

    # -- decoded-box columns (0.0 when decoded_boxes is None) --------------- #
    # The box-head regression box (--head-box decoded) is what the stroke head
    # uses as its window; a candidate whose decoded box covers more than its
    # mask box is more likely to be rescued by the head, so these columns give
    # the selector that signal directly.
    if decoded_boxes is not None:
        dboxes = np.asarray(decoded_boxes, dtype=np.float64).reshape(-1, 4)
        if len(dboxes) != n:
            raise ValueError(
                "decoded_boxes ragged: %d boxes for %d candidates" % (len(dboxes), n)
            )
        dpx_w = dboxes[:, 2] - dboxes[:, 0]
        dpx_h = dboxes[:, 3] - dboxes[:, 1]
        dbox_area = dpx_w * dpx_h
        out[:, FEATURE_INDEX["dbox_w"]] = np.log(dpx_w + 1.0)
        out[:, FEATURE_INDEX["dbox_h"]] = np.log(dpx_h + 1.0)
        out[:, FEATURE_INDEX["dbox_ratio"]] = np.log(dbox_area + 1.0) - box_area
        ix0 = np.maximum(boxes[:, 0], dboxes[:, 0])
        iy0 = np.maximum(boxes[:, 1], dboxes[:, 1])
        ix1 = np.minimum(boxes[:, 2], dboxes[:, 2])
        iy1 = np.minimum(boxes[:, 3], dboxes[:, 3])
        inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
        union = (px_w * px_h) + dbox_area - inter
        out[:, FEATURE_INDEX["dbox_iou"]] = np.where(union > 0, inter / union, 0.0)

    # -- mask columns ------------------------------------------------------- #
    for i in range(n):
        box = boxes[i].astype(np.int64)
        y0 = max(0, int(box[1]))
        y1 = min(int(height), max(y0, int(box[3])))
        x0 = max(0, int(box[0]))
        x1 = min(int(width), max(x0, int(box[2])))
        px_area = float(px_w[i] * px_h[i])

        if formulas == "legacy":
            mask = np.asarray(masks[i], dtype=bool)
            if mask.shape != (int(height), int(width)):
                raise ValueError(
                    "candidate %d mask %r does not match the declared %dx%d"
                    % (i, mask.shape, int(width), int(height))
                )
            mask_px = int(np.count_nonzero(mask))
            ink_px = int(np.count_nonzero(ink_array & mask))
            perim = mask_perimeter(mask)
            ink_perim = mask_perimeter(ink_array & mask)

            area = float(mask_px)
            out[i, FEATURE_INDEX["mask_area"]] = np.log(area) if area > 0 else 0.0
            out[i, FEATURE_INDEX["fill"]] = mask_px / (px_w[i] * px_h[i] + 1.0)
            out[i, FEATURE_INDEX["ink_in_mask"]] = np.log(ink_px) if ink_px > 0 else 0.0
            out[i, FEATURE_INDEX["ink_frac_mask"]] = (ink_px / area) if area > 0 else 0.0
            out[i, FEATURE_INDEX["thick_mask"]] = max(2.0, 2.0 * mask_px / perim) if perim else 2.0
            out[i, FEATURE_INDEX["thick_ink"]] = max(2.0, 2.0 * ink_px / ink_perim) if ink_perim else 2.0
            out[i, FEATURE_INDEX["mask_perim"]] = float(perim)

            crop = mask[y0:y1, x0:x1]
            counted_px = ink_px
            if crop.size:
                _labels, sizes = mask_components(ink_array[y0:y1, x0:x1] & crop)
                sizes = sizes[1:]
                total = int(sizes.size)
                largest = int(sizes.max()) if total else 0
                out[i, FEATURE_INDEX["ncomp_total"]] = float(total)
                out[i, FEATURE_INDEX["ncomp_ge25"]] = float(
                    int(np.count_nonzero(sizes >= COMPONENT_MIN_AREA))
                )
                out[i, FEATURE_INDEX["comp_largest_frac"]] = (
                    (largest / counted_px) if counted_px > 0 else 0.0
                )
            ink_box = int(np.count_nonzero(ink_array[y0:y1, x0:x1]))
            out[i, FEATURE_INDEX["ink_frac_box"]] = (
                ink_box / px_area if px_area > 0 else 0.0
            )
            continue

        # -- the pool's expressions, on the pool's mask raster -------------- #
        if mask_source == "ink_box_pad2":
            crop, (top, left) = ink_box_mask(
                ink_array, x0, y0, x1, y1, width, height
            )
        else:
            crop, (top, left) = _refined_mask_crop(
                masks[i], x0, y0, x1, y1, width, height
            )
        bottom = min(int(height), top + int(crop.shape[0]))
        right = min(int(width), left + int(crop.shape[1]))
        ink_crop = ink_array[top:bottom, left:right]

        mask_px = int(np.count_nonzero(crop))
        ink_px = int(np.count_nonzero(ink_crop & crop)) if crop.size else 0
        perim = mask_perimeter(crop) if crop.size else 0

        out[i, FEATURE_INDEX["mask_area"]] = float(np.log(mask_px + 1.0))
        out[i, FEATURE_INDEX["fill"]] = (mask_px / px_area) if px_area > 0 else 0.0
        out[i, FEATURE_INDEX["ink_in_mask"]] = float(np.log(ink_px + 1.0))
        out[i, FEATURE_INDEX["ink_frac_mask"]] = (ink_px / mask_px) if mask_px else 0.0
        out[i, FEATURE_INDEX["thick_mask"]] = (
            max(2.0, 2.0 * mask_px / perim) if perim else 2.0
        )
        out[i, FEATURE_INDEX["thick_ink"]] = (
            max(2.0, 2.0 * ink_px / perim) if perim else 2.0
        )
        out[i, FEATURE_INDEX["mask_perim"]] = float(perim)

        _labels, sizes = mask_components(crop)
        sizes = sizes[1:]
        largest = int(sizes.max()) if sizes.size else 0
        out[i, FEATURE_INDEX["ncomp_total"]] = float(sizes.size)
        out[i, FEATURE_INDEX["comp_largest_frac"]] = (
            (largest / mask_px) if mask_px else 0.0
        )

        # the pool's second raster: ink inside the box, inclusive slice, no pad
        patch = ink_array[y0:min(int(height), y1 + 1), x0:min(int(width), x1 + 1)]
        ink_box = int(np.count_nonzero(patch))
        out[i, FEATURE_INDEX["ink_frac_box"]] = (
            ink_box / px_area if px_area > 0 else 0.0
        )
        _labels2, sizes2 = mask_components(patch)
        out[i, FEATURE_INDEX["ncomp_ge25"]] = float(max(0, int(sizes2.size) - 1))

    return out.astype(np.float32)
