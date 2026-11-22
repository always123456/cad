"""The v3 wide-pool selection rule, as one pure function.

This is the stage-2 rule the frozen v3 readout was emitted with.  It is not a
reconstruction: ``tools/rule_parity.py check`` replays it over the frozen
candidate pool (``evidence/v3-features/shard_rows.npz``, 16192 candidates over
387 held-out images) with the archived out-of-fold probabilities
(``probs_blend.pkl``) and gets **all 387 documents back, instance for instance**
-- the same count, the same class multiset, the same class sequence, and a
one-to-one box pairing whose worst IoU is 0.6526 (the gap to 1.0 is the mask-bbox
vs candidate-box difference, not a different pick).  ``RULE-PARITY.md`` in
``evidence/v3-features/`` records the sweep.

The rule
--------
::

    candidates (already the delivery decode: score >= 0.50, <= 300 per image)
      -> Things only                     (class < 30)
      -> p >= thr                        (inclusive; thr 0.60)
      -> greedy in p-descending order    (ties keep candidate/decode order)
           a kept box suppresses every *same-class* box with IoU >= tau
                                                              (tau 0.20, inclusive)
      -> global max-detections cap (300, survivor order kept)
      -> per-image cap (20, top-N by p, ties keep decode order)

Four things are load-bearing and were each measured:

* **same-class, not all-class** NMS.  All-class NMS drags 8 of the 387 images
  onto a different instance set (379/387); no NMS at all, 296/387.
* **the greedy order is probability, not decode order.**  Greedy in bare
  candidate order gets 361/387 -- 26 images differ, because decode order is not
  sorted by the blend probability.
* **cap 20 applies after NMS**, on the keep order.  Taking the top 20 by p first
  and then applying NMS gets 384/387; dropping the cap entirely 382/387 (5
  images reach 21-22 survivors).
* **the selection unit is the candidate**, i.e. every primitive of a kept
  instance moves with it.  There is no per-primitive rule.

``thr`` and ``tau`` are read as documented (0.60 / 0.20).  They are only
identified to an interval on this pool -- no blend probability lands within
2.1e-4 of 0.60 (so ``>=`` and ``>`` are indistinguishable), and no same-class
pair falls in [0.20, 0.22) in a way that changes a pick (tau 0.20 and 0.22 both
give 387/387, 0.18 and 0.25 give 384).

Everything else that was floated and rejected:

* a ``score >= 0.99`` bypass ("shipped keep"): with the *blend probability* as the
  score only 4 candidates qualify and all 4 are already kept, so the axis is
  invisible here (387/387 either way); with the *detector* score 1729 qualify,
  860 of them below thr, and honouring that gets 132/387.  The frozen documents
  are the no-bypass reading.
* Stuff (class >= 30) exclusion timing: the pool contains no Stuff candidate at
  all (0 of 16192), so before/after/never are the same 387/387.  Classes >= 30
  are dropped by the decoder long before this function sees them; the guard
  below is belt and braces.
* the 300-per-image decode gate: the pool is capped at 226 candidates per image,
  so this gate never binds and cannot be what separates 0.220984 from 0.220305.

Pure: numpy only, no I/O, no imports from the delivery CLI's heavier modules.
``delivery/nmsbox.py`` holds the same greedy pass for the older wide readout; this
module delegates to it so the repository keeps one implementation.
"""

from __future__ import annotations

import numpy as np

from nmsbox import STUFF_CLASS_START, select_by_box_nms

#: The frozen cell (``v2_emit.py`` ``CELLS["blendnms060"]``).
V3_THRESHOLD = 0.60
V3_BOX_IOU = 0.20
V3_PER_IMAGE_CAP = 20
V3_MAX_DETECTIONS = 300

__all__ = [
    "STUFF_CLASS_START",
    "V3_THRESHOLD",
    "V3_BOX_IOU",
    "V3_PER_IMAGE_CAP",
    "V3_MAX_DETECTIONS",
    "select_v3",
    "select_v3_counts",
]


def _unpack(candidates, probs):
    """Accept a mapping or ``(boxes, classes, scores)`` positional arrays."""

    if isinstance(candidates, dict):
        return (
            np.asarray(candidates["boxes"], dtype=np.float64).reshape(-1, 4),
            np.asarray(candidates["classes"], dtype=np.int64).reshape(-1),
            np.asarray(candidates.get("scores", probs), dtype=np.float64).reshape(-1),
        )
    if len(candidates) != 3:
        raise ValueError("pass a mapping or (boxes, classes, scores)")
    boxes, classes, scores = candidates
    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64).reshape(-1),
        np.asarray(scores, dtype=np.float64).reshape(-1),
    )


def select_v3(candidates, probs, thr=V3_THRESHOLD, tau=V3_BOX_IOU, cap=V3_PER_IMAGE_CAP):
    """Select the instances of ONE image; returns kept candidate indices.

    ``candidates`` is either a mapping with ``boxes`` (xyxy native pixels, decode
    order), ``classes`` (canonical ids) and ``scores`` (detector confidence), or a
    ``(boxes, classes, scores)`` triple.  ``probs`` is the blend probability per
    candidate, in the same order.  The returned indices are in emission order:
    the order the document's ``instanceIds`` must follow.
    """

    boxes, classes, scores = _unpack(candidates, probs)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if not (len(boxes) == len(classes) == len(scores) == len(probs)):
        raise ValueError(
            "selection input mismatch: %d boxes, %d classes, %d scores, %d probs"
            % (len(boxes), len(classes), len(scores), len(probs))
        )
    if len(probs) == 0:
        return np.zeros(0, dtype=np.int64)

    # The rule ranks by the blend probability, so it is handed over as the score;
    # the detector confidence is not consulted (see the 0.99-bypass note above).
    result = select_by_box_nms(
        boxes,
        classes,
        probs,
        score_threshold=thr,
        box_iou=tau,
        max_detections=V3_MAX_DETECTIONS,
        per_image_cap=int(cap),
    )
    return np.asarray(result["indices"], dtype=np.int64)


def select_v3_counts(candidates, probs, thr=V3_THRESHOLD, tau=V3_BOX_IOU,
                     cap=V3_PER_IMAGE_CAP):
    """``(indices, counts)``; ``counts`` has the per-stage drop counts."""

    boxes, classes, scores = _unpack(candidates, probs)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(probs) == 0:
        return np.zeros(0, dtype=np.int64), {
            "candidates": 0, "after_threshold": 0, "after_nms": 0,
            "after_cap": 0, "after_per_image_cap": 0, "selected": 0,
            "nms_suppressed": 0, "threshold_dropped": 0, "cap_dropped": 0,
            "per_image_cap_dropped": 0,
        }
    result = select_by_box_nms(
        boxes,
        classes,
        probs,
        score_threshold=thr,
        box_iou=tau,
        max_detections=V3_MAX_DETECTIONS,
        per_image_cap=int(cap),
    )
    return np.asarray(result["indices"], dtype=np.int64), result["counts"]
