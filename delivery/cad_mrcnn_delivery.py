#!/usr/bin/env python
"""Competitor-facing PNG -> JSON delivery CLI for the Mask R-CNN direct-instance model.

One input PNG produces ``<out>/results/<stem>.json`` holding exactly the twelve
delivery-contract fields (commands, args, lengths, semanticIds, instanceIds,
width, height, obj_cts, boxes, rgb, layerIds, widths) and nothing else.  No score
field is ever written: scores exist only inside the selection stage and in the
selection report / run receipt.

The pipeline reuses the frozen lane code instead of re-implementing it:

    competition PNG  ->  model.infer.load_png / predict_arrays   (lane, frozen)
                     ->  CLI selection stage                     (this file)
                     ->  model.emit.emit_instances               (lane, frozen)
                     ->  instance_eval.contract.validate_prediction_document

``instance_eval.batch.run_batch`` is the driver.  It is called with
``inputs=<image-only inputs file>``, ``score_threshold=None`` and
``max_detections=None`` so the decode keeps the checkpoint's own settings
(box_score_thresh 0.05, box_nms_thresh 0.5, box_detections_per_img 300,
mask threshold 0.5), and with an injected ``emit`` that performs the selection
stage.  Ground truth is never an input: the only per-image data the driver reads
is the PNG itself, and the inputs file it is handed is built by this CLI from
``sample_id`` / ``group_id`` / ``png_path`` only.

Selection order (identical to the shipped YOLO delivery CLI)::

    decoded candidates
      -> same-class mask IoU NMS   (suppress when iou > nms_iou, strict)
      -> score threshold           (keep when score > score_threshold, strict)
      -> global max-detections cap (decode order preserved)
      -> optional per-image cap    (top-N by score; ties keep decode order)

Every knob is validated before any model, checkpoint, PNG or inputs access.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import sys
import time

import mask_paste  # noqa: E402 - ships next to this CLI, holds no model state

# --------------------------------------------------------------------------- #
# locations (read-only references into the lane; nothing here is modified)
# --------------------------------------------------------------------------- #
DELIVERY_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(DELIVERY_ROOT)
REPO_ROOT = MODEL_ROOT
SRC_DIR = os.path.join(REPO_ROOT, "src")
EVAL_DIR = os.path.join(MODEL_ROOT, "evaluation")
DEFAULT_CHECKPOINT = os.path.join(
    MODEL_ROOT, "runs", "grouped_expanded_paste_consistent01", "checkpoint.pt"
)

SCHEMA_SELECTION = "cad-png-delivery-mrcnn01.selection/1"
SCHEMA_RECEIPT = "cad-png-delivery-mrcnn01.receipt/1"
SCHEMA_INDEX = "cad-png-delivery-mrcnn01.results_index/1"
SCHEMA_INPUTS = "cadvec.direct_instance.inputs.v1"

# --resume: what a half-finished --out may still hold.  Documents land in
# ``results/<stem>.json`` as soon as their emit is done, so a run that died after
# twenty minutes leaves the finished ones behind; these names are the run's own
# bookkeeping.  Anything else (a PNG, a stray JSON, an unknown directory) is
# refused rather than silently reused: that directory is then not this run's.
RESUME_BOOKKEEPING = (
    "infer_report.json",
    "results_index.json",
    "run_receipt.json",
    "selection_report.json",
)
RESUME_RUN_DIRS = ("inputs", "results", "predictions", "failures")
PROGRESS_EVERY = 10
SLOW_IMAGE_S = 5.0

DOCUMENT_FIELDS = (
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

# Measured settings of the shipped direct-instance decoder.  The flags are
# additive: leaving them alone reproduces the lane's own decode.
DEFAULT_SCORE_THRESHOLD = 0.05
DEFAULT_MASK_THRESHOLD = 0.5
# The measured mask-binarisation choices this CLI accepts.  0.5 is the frozen
# decoder's own value and the default; the rest are the coarse control for the
# same thin-stroke loss that --mask-paste logits attacks exactly.
MASK_THRESHOLD_CHOICES = (0.5, 0.45, 0.40, 0.35)
# Where the mask sigmoid happens relative to the native-canvas bilinear resize.
DEFAULT_MASK_PASTE = mask_paste.DEFAULT_MASK_PASTE
DEFAULT_MASK_LOGIT_PAD = mask_paste.DEFAULT_LOGIT_PAD
DEFAULT_NMS_IOU = 0.5
DEFAULT_MAX_DETECTIONS = 300
DECODER_MAX_DETECTIONS = 300
DEFAULT_DEVICE = "cuda:0"

# Mask refinement (refine.py).  ``a6`` is the measured arm and the default;
# ``a1`` is the threshold-robust variant one switch away; ``none`` reproduces the
# sealed baseline exactly.
DEFAULT_REFINE_RULE = "a6"
DEFAULT_REFINE_INK = "nonzero"
DEFAULT_REFINE_RATIO = 0.5
REFINE_RATIO_RANGE = (0.0, 1.0)
REFINE_RULES = ("none", "a6", "a1")
REFINE_INK_RULES = ("nonzero", "composite128")
REFINE_STAGE = (
    "after selection (score threshold, same-class mask-IoU NMS, global max-detections cap, "
    "per-image cap) and before vectorisation, on the native raster; the square-canvas pad of a "
    "non-square record (right/bottom, to max(width, height)) is applied afterwards, so the PNG "
    "ink and the mask share the image's own resolution"
)
REFINE_MEASURED = {
    "none": {"val387_micro_pq": 0.036694672, "tp": 120, "fp": 1008, "fn": 2924},
    "a1": {"val387_micro_pq": 0.055837724, "tp": 168, "fp": None, "fn": None},
    "a6": {"val387_micro_pq": 0.060834794, "tp": 176, "fp": 952, "fn": 2868},
}
REFINE_RATIO_SWEEP = {
    "measured_on": "all 387 held-out records, score 0.99, composite128 ink, canonical scorer",
    "ratio_0.0": {
        "score_threshold": 0.99,
        "per_image_cap": 20,
        "micro_pq": 0.116281992,
        "tp": 332,
        "fp": 814,
        "fn": 2712,
        "sum_iou": 243.610774,
        "sq": 0.733767,
        "rq": 0.158473,
        "emitted": 1146,
        "half_a_pq": 0.122104861,
        "half_b_pq": 0.109863105,
        "both_halves_improve": True,
    },
    "ratio_0.5": {"score_threshold": 0.99, "per_image_cap": 12, "micro_pq": 0.060834794},
    "finding": (
        "the component-inclusion ratio, not the score threshold, was the defect: keeping every "
        "8-connected ink component the predicted mask touches (ratio 0.0) nearly doubles canonical "
        "PQ at the same score threshold, and both deterministic halves improve"
    ),
    "reproduce": (
        "--refine-mask a6 --refine-ink composite128 --refine-ratio 0.0 --score-threshold 0.99 "
        "--per-image-cap 20"
    ),
}
REFINE_MEASURED_NOTE = (
    "measured on all 387 held-out records with the canonical scorer at score>=0.99, "
    "max-detections 300, per-image cap 12; a6/a1 were measured with the composite128 ink "
    "(min(R,G,B) < 128 after compositing over white), which is 1.3-2.8x thinner than the "
    "delivered nonzero ink on the same PNGs, and the measuring lane's own threshold sweep "
    "puts a6 below the baseline once the ink is thick (T=250 -> 0.034865). The delivered "
    "default therefore differs in the ink definition from the measuring lane; --refine-ink "
    "composite128 reproduces the measured arm exactly"
)

# Ink-in-box mask for instances whose box is larger than the 112^2 head resolves.
#
# The mask branch is 112x112 and its logits are pasted by a bilinear resize, so a
# 1 px stroke inside a box wider than 112 px is sub-pixel: the head emits a
# fragment (median mask IoU 0.06 on the >1000 px^2 bucket) while its same-class
# box is right 83.7% of the time.  For every selected instance whose box side
# exceeds --mask-ink-min-side the mask that reaches the emitter is therefore the
# PNG ink inside the box widened by INK_BOX_PAD px (clamped to the raster), and
# the head's own mask is ignored; a box at or below the threshold keeps its
# refined mask untouched, and an empty crop keeps the original mask too.
#
# The rule is OFF by default (DEFAULT_INK_MIN_SIDE = 0): measured end to end on
# the 387 held-out records it is only +0.001306 PQ (0.247198 -> 0.248504) in the
# post placement, short of the +0.008 it was asked to clear, because the a6
# refinement already intersects the mask with the same composite128 ink, so the
# replacement mostly widens the crop onto foreign ink (115 images lose SumIoU,
# 50 gain).  The measured arm and its numbers stay in the CLI help and in
# evidence/ink-in-box-postselect-20261120.json.
#
# ``pre`` applies the rule before the blend selector (so the selector's boxes and
# 26 features are read off ink boxes); ``post`` -- the default -- applies it
# after selection, so the selector sees exactly the refined masks it always saw
# and only the emitted document changes.  The pre placement was measured end to
# end at +0.000746 PQ (evidence/ink-in-box-delivery-20261120.json); shrinking
# the boxes before the selector moves box-IoU suppression and cancels the mask
# gain that the mask-level experiment saw.
DEFAULT_INK_MIN_SIDE = 0
INK_MIN_SIDE_RANGE = (0, 4096)
INK_BOX_PAD = 2
DEFAULT_INK_STAGE = "post"
INK_STAGES = ("pre", "post")
INK_BOX_MEASURED = {
    "measured_on": "dev_final clean-dev, mask level: the shipped selection reproduced "
    "(blend 0.70 / 0.15 / 28) and the selected large instances' masks replaced by ink & crop(box +-2)",
    "no_rule": {"micro_pq": 0.243974, "tp": 772, "fp": 479, "fn": 2328},
    "by_min_side": {
        16: {"micro_pq": 0.259829, "tp": 819, "fp": 432, "fn": None},
        32: {"micro_pq": 0.260336, "tp": 821, "fp": 430, "fn": None},
        48: {"micro_pq": 0.258563, "tp": None, "fp": None, "fn": None},
        64: {"micro_pq": 0.254436, "tp": None, "fp": None, "fn": None},
        96: {"micro_pq": 0.246937, "tp": None, "fp": None, "fn": None},
        112: {"micro_pq": 0.247275, "tp": None, "fp": None, "fn": None},
    },
    "evidence": "evidence/ink-in-box-pq-20261120.json, "
    "evidence/large-instance-failure-20261120.json",
    "note": "a mask-level measurement, not a document-pipeline one; the delivery run's own "
    "held-out numbers are 0.248504 with the rule on (post placement, 32) in "
    "evidence/ink-in-box-postselect-20261120.json and 0.247198 with it off",
}
INK_BOX_DELIVERY_MEASURED = {
    "measured_on": "387 held-out records (datasets/train_final stems of "
    "evidence/results-holdout-387-20261119), strict IoU > 0.5, micro PQ, tools/score_local.py",
    "shipped": {"micro_pq": 0.247197922831, "tp": 763, "fp": 518, "fn": 2281, "instances": 1281},
    "post_32": {"micro_pq": 0.248504038873, "tp": 777, "fp": 504, "fn": 2267, "instances": 1281},
    "pre_32": {"micro_pq": 0.247943878854, "tp": 783, "fp": 531, "fn": 2261, "instances": 1314},
    "gate": 0.255198,
    "default_should_be": 0,
    "evidence": "evidence/ink-in-box-postselect-20261120.json",
}

# Mask source for the instances the 112^2 branch cannot resolve (--mask-source).
#
# The mask branch is 112x112 with a bilinear paste, so one stroke of a symbol in
# a box wider than 112 px is sub-pixel.  ``head`` runs the trained 256^2 instance
# stroke head (delivery/mask_source.py, the arm of tools/instance_mask_head.py and
# evidence/instance-mask-head-256-20261120/REPORT.md) on the ink of each selected
# instance whose box side exceeds --mask-head-min-side, and the mask that reaches
# the emitter is ``ink >= gate`` AND ``prob >= tau`` inside the head's own window;
# an empty prediction keeps the refined mask.  ``ink`` selects the measured
# ink-in-box arm instead (--mask-ink-min-side / --mask-ink-stage), ``mask`` -- the
# default -- leaves the mask the refinement produced, i.e. the shipped documents.
MASK_SOURCES = ("mask", "head", "ink")
DEFAULT_MASK_SOURCE = "mask"
DEFAULT_MASK_HEAD_MIN_SIDE = 64
DEFAULT_MASK_HEAD_TAU = 0.30
DEFAULT_MASK_HEAD_GATE = 96
MASK_HEAD_MEASURED = {
    "measured_on": "clean-dev detected boxes, 256^2 window, gate 96, tau 0.30 "
    "(evidence/instance-mask-head-256-20261120/REPORT.md)",
    "median_iou_large_bucket": 0.6098,
    "share_above_0.5_large_bucket": 0.714,
    "median_iou_all_boxes": 0.5980,
    "share_above_0.5_all_boxes": 0.675,
    "shipped_masks": {"median_iou_large_bucket": 0.3168, "share_above_0.5_large_bucket": 0.3229},
    "ink_in_box": {"median_iou_large_bucket": 0.4935},
    "parameters": 1948577,
    "evidence": "evidence/instance-mask-head-256-20261120/REPORT.md",
}

# Mask head (how many pixels the mask branch predicts before pasting).
#
# ``28`` is the shipped head of the frozen checkpoint and stays the default: its
# decode path is untouched by this flag.  ``112`` swaps in the fine head
# (conv5_mask 14->28 reused, conv6/conv7 fresh, 1x1 logits 36) from the fine-head
# lane and needs a chunked mask runner: the unchunked path materialises
# ``N * 256 * 112 * 112`` floats (~3.85 GB at N=300) and OOMs on a 6 GB card.
# Chunking is proven not to change boxes/scores/labels (bitwise) or the mask
# decision (probability max |delta| 3.5e-4, 0 pixels above 1e-2).
MASK_HEADS = ("28", "112")
DEFAULT_MASK_HEAD = "28"
MASK_HEAD_OUTPUT = {"28": 28, "112": 112}
DEFAULT_MASK_CHUNK = 24
MASK_CHUNK_RANGE = (1, 1024)
MASK_HEAD_MEASURED = {
    "scorer": "cad-parent-ship-score01/score_ship.py, canonical, 387 held-out / 3044 GT",
    "flags": (
        "--drop-stuff --refine-mask a6 --refine-ink composite128 --refine-ratio 0.0 "
        "--score-threshold 0.99 --max-detections 300 --per-image-cap 20"
    ),
    "28": {
        "pq": 0.120714704164,
        "tp": 321,
        "fp": 557,
        "fn": 2723,
        "sum_iou": 236.721534866,
        "results_dir": "cad-png-delivery-mrcnn03-runs/ship-387-b00-01/results",
    },
    "28_stuff_kept": {"pq": 0.125862769467},
    "112_u250": {
        "pq": 0.137443379498,
        "tp": 370,
        "fp": 540,
        "fn": 2674,
        "sum_iou": 271.725561267,
    },
    "finding": (
        "the boxes/scores/labels of the decode are identical, so the head cannot add detections; "
        "it changes which refined masks clear IoU 0.5 (TP 321 -> 370, FP 557 -> 540) and lifts "
        "SumIoU 236.72 -> 271.73, i.e. it is a mask-fidelity win that converts near-misses"
    ),
}

SELECTION_ORDER = (
    "candidates -> same-class mask IoU NMS (iou > nms_iou, strict) -> score threshold "
    "(score > score_threshold, strict) -> global max-detections cap (decode order kept) "
    "-> per-image cap (top-N by score, ties keep decode order)"
)

# Mirrors instance_eval.contract.SEALED_TOKENS; duplicated so that knob validation
# stays free of any lane import.
SEALED_TOKENS = ("test_final", "dev_final", "eval_pool", "reserve", ".runtime")

# --allow-official-inputs (off by default) admits one case only: an input PNG that
# sits inside a directory named ``datasets`` (the official corpora, e.g.
# ``datasets/dev_final``), for an offline self-check.  --out, --checkpoint,
# --inputs and every non-PNG path keep the full sealed-token refusal.
OFFICIAL_INPUT_DIRNAME = "datasets"

DEVICE_RE = re.compile(r"^(?:cpu|cuda(?::\d+)?)$")

# A read of one of these suffixes from outside the declared roles is treated as a
# delivery violation (an SVG or a ground-truth style artefact was opened).
DATA_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".json",
    ".csv",
    ".txt",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".xml",
    ".yaml",
    ".yml",
    ".geojson",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
class DeliveryRefused(Exception):
    """A fail-closed refusal. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


def _refuse(code: str, detail: str) -> DeliveryRefused:
    return DeliveryRefused(code, detail)


# --------------------------------------------------------------------------- #
# knob validation (pure, no imports of the lane, no file or model access)
# --------------------------------------------------------------------------- #
def _selection_float(name, value, low, high, *, high_open):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(
            "selection_policy",
            "%s refused: a real number is required, got %s %r" % (name, type(value).__name__, value),
        )
    number = float(value)
    if not math.isfinite(number):
        raise _refuse("selection_policy", "%s refused: not finite (%r)" % (name, number))
    if number <= low:
        raise _refuse("selection_policy", "%s refused: must be > %r, got %r" % (name, low, number))
    if (number >= high) if high_open else (number > high):
        bound = "< %r" % (high,) if high_open else "<= %r" % (high,)
        raise _refuse("selection_policy", "%s refused: must be %s, got %r" % (name, bound, number))
    return number


def _selection_int(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse("selection_policy", "%s refused: integer required, got %r" % (name, value))
    if value < low or value > high:
        raise _refuse(
            "selection_policy", "%s refused: must be in %d..%d, got %d" % (name, low, high, value)
        )
    return value


def _refine_ratio(value):
    """Validate ``--refine-ratio``; 0.0 and 1.0 are both legal values."""

    if value is None:
        return DEFAULT_REFINE_RATIO
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(
            "refine_policy",
            "--refine-ratio refused: a real number is required, got %s %r"
            % (type(value).__name__, value),
        )
    number = float(value)
    if not math.isfinite(number):
        raise _refuse("refine_policy", "--refine-ratio refused: not finite (%r)" % (number,))
    low, high = REFINE_RATIO_RANGE
    if number < low or number > high:
        raise _refuse(
            "refine_policy",
            "--refine-ratio refused: must be in %r..%r inclusive, got %r" % (low, high, number),
        )
    return number


def validate_refine(refine_mask=None, refine_ink=None, refine_ratio=None):
    """Validate the three refinement knobs; return the frozen ``refine`` block."""

    rule = DEFAULT_REFINE_RULE if refine_mask in (None, "") else str(refine_mask)
    ink = DEFAULT_REFINE_INK if refine_ink in (None, "") else str(refine_ink)
    coverage = _refine_ratio(refine_ratio)
    if rule not in REFINE_RULES:
        raise _refuse(
            "refine_policy",
            "--refine-mask refused: %r is not one of %r" % (rule, list(REFINE_RULES)),
        )
    if ink not in REFINE_INK_RULES:
        raise _refuse(
            "refine_policy",
            "--refine-ink refused: %r is not one of %r" % (ink, list(REFINE_INK_RULES)),
        )
    if coverage != _refine().validate_coverage(coverage):  # pragma: no cover - same check twice
        raise _refuse("refine_policy", "--refine-ratio refused: %r" % (coverage,))
    return {
        "rule": rule,
        "ink": ink,
        "ink_definition": _refine().INK_DEFINITIONS[ink],
        "component_coverage": coverage,
        "component_coverage_default": DEFAULT_REFINE_RATIO,
        "component_coverage_range": list(REFINE_RATIO_RANGE),
        "component_coverage_rule": (
            "an 8-connected ink component of the mask-bbox crop (expanded by 2 px) is kept when "
            "inside / component_size >= component_coverage (inclusive); 0.0 keeps every component "
            "present in the crop, 1.0 only fully covered ones"
        ),
        "component_connectivity": "8-connected, from the original PNG at its native resolution (no resize)",
        "applied_in": REFINE_STAGE,
        "stage_order": "selection -> refine -> square-canvas pad -> vectorise",
        "fallbacks": {
            "a6": "no kept component -> mask & ink -> mask",
            "a1": "mask & ink -> mask",
            "none": "mask unchanged",
        },
        "never_drops_an_instance": True,
        "measured_on_val387": REFINE_MEASURED[rule],
        "measured_note": REFINE_MEASURED_NOTE,
        "measured_at_component_coverage": DEFAULT_REFINE_RATIO,
        "ratio_sweep": REFINE_RATIO_SWEEP,
    }


def _refine():
    """Import ``refine.py`` next to this file (kept lazy so ``--refine-mask none``
    and the knob validation stay numpy-free)."""

    import importlib.util  # noqa: PLC0415

    module = sys.modules.get("cad_refine")
    if module is not None:
        return module
    path = os.path.join(DELIVERY_ROOT, "refine.py")
    spec = importlib.util.spec_from_file_location("cad_refine", path)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the CLI
        raise _refuse("refine_policy", "refine.py is missing next to the CLI (%s)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_refine"] = module
    spec.loader.exec_module(module)
    return module


def _nmsbox():
    """Import ``nmsbox.py`` next to this file (kept lazy so the default ``plain``
    rule never touches numpy or the box path)."""

    import importlib.util  # noqa: PLC0415

    module = sys.modules.get("cad_nmsbox")
    if module is not None:
        return module
    path = os.path.join(DELIVERY_ROOT, "nmsbox.py")
    spec = importlib.util.spec_from_file_location("cad_nmsbox", path)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the CLI
        raise _refuse("selection_policy", "nmsbox.py is missing next to the CLI (%s)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_nmsbox"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# stage-2 blend selector: the frozen 30-feature readout
# --------------------------------------------------------------------------- #
BLEND_RULE = "blend"
#: The shipped selector (``weights/blend_full_20261124.pkl``): 30-feature contract (26 base
#: + 4 decoded-box features dbox_ratio/dbox_iou/dbox_w/dbox_h) and 6-learner Blend family,
#: re-fit on the FULL train_final relabel set (3700 images / 167877 candidates / 28702 GT)
#: with delivered-mask labels under the gate-64 decoded head-box readout.  Measured end to
#: end on the 300 held-out stems at >= 0.70 / tau 0.10 / cap 60: PQ 0.3740 (TP 1090 / FP 681
#: / FN 1426), up from 0.3708 for the 26-feature selector on the same pipeline (+0.0032;
#: TP +4 / FP -18 / FN -4).  The previous selector (``weights/blend_full_20261123.pkl``) is
#: superseded.
DEFAULT_BLEND_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(DELIVERY_ROOT), "weights", "blend_full_20261124.pkl")
)
DEFAULT_BLEND_THR = 0.70
DEFAULT_BLEND_TAU = 0.10
DEFAULT_BLEND_CAP = 60
BLEND_FEATURE_COUNT = 30

_BLEND_MODEL_CACHE = {}


def _features26():
    """Import ``features26.py`` next to this file.

    Kept lazy and private for the same reason as ``_nmsbox``: the default rule must
    not touch numpy-heavy helpers it never uses.
    """

    import importlib.util  # noqa: PLC0415

    module = sys.modules.get("cad_features26")
    if module is not None:
        return module
    path = os.path.join(DELIVERY_ROOT, "features26.py")
    spec = importlib.util.spec_from_file_location("cad_features26", path)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the CLI
        raise _refuse("selection_policy", "features26.py is missing next to the CLI (%s)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_features26"] = module
    spec.loader.exec_module(module)
    return module


def _mask_source():
    """Import ``mask_source.py`` next to this file.

    Lazy like the helpers above: ``--mask-source mask`` (the default) never loads
    the stroke head, its checkpoint or torch's UNet code.
    """

    import importlib.util  # noqa: PLC0415

    module = sys.modules.get("cad_mask_source")
    if module is not None:
        return module
    path = os.path.join(DELIVERY_ROOT, "mask_source.py")
    spec = importlib.util.spec_from_file_location("cad_mask_source", path)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the CLI
        raise _refuse("selection_policy", "mask_source.py is missing next to the CLI (%s)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_mask_source"] = module
    spec.loader.exec_module(module)
    return module


def _blend_model(path):
    """Load the pickled blend selector (once per process) and check its shape.

    The pickle names ``models.Blend``, so ``delivery/`` must be importable before
    unpickling; it is the CLI's own directory, i.e. ``sys.path[0]`` under a normal
    run.  The 26-column check is the same one ``infer.py: v3_blockers()`` makes:
    a selector fitted on another feature order cannot be scored silently.
    """

    key = os.path.abspath(str(path))
    cached = _BLEND_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if not os.path.isfile(key):
        raise _refuse("selection_policy", "blend selector model missing: %s" % key)
    if DELIVERY_ROOT not in sys.path:
        sys.path.insert(0, DELIVERY_ROOT)
    import pickle  # noqa: PLC0415

    try:
        with open(key, "rb") as handle:
            model = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - report as a refusal, never a traceback
        raise _refuse(
            "selection_policy", "blend selector unreadable: %s: %s" % (type(exc).__name__, exc)
        ) from None
    parts = list(getattr(model, "parts", None) or [])
    if not parts:
        raise _refuse("selection_policy", "blend selector %s holds no fitted parts" % key)
    n_features = int(getattr(parts[0], "n_features_in_", 0))
    if n_features != BLEND_FEATURE_COUNT:
        raise _refuse(
            "selection_policy",
            "blend selector expects %d features, this CLI's features26.py builds %d"
            % (n_features, BLEND_FEATURE_COUNT),
        )
    _BLEND_MODEL_CACHE[key] = model
    return model


def validate_blend(
    select_rule=None,
    blend_model=None,
    blend_thr=None,
    blend_tau=None,
    blend_cap=None,
):
    """Validate ``--select-rule blend`` and its knobs; returns the block only for blend.

    ``blend`` keeps the pool decode exactly as is (the mode flags pass score 0.50 /
    mask 0.40 / cap 300) and *adds* the frozen selector after refinement: 26
    features of the refined mask (``delivery/features26.py``), one probability per
    candidate (``delivery/models.py: Blend`` = HGB + ExtraTrees average), then
    Things only (class < 30), p >= ``--blend-thr``, greedy same-class box IoU
    NMS inside the box threshold tau, global cap, per-image cap.  That rule is
    ``nmsbox.select_by_box_nms`` applied to the probabilities instead of the
    detection scores — the same function, not a re-implementation.
    """

    rule = DEFAULT_SELECT_RULE if select_rule is None else str(select_rule)
    if rule != BLEND_RULE:
        return None

    model = os.path.abspath(DEFAULT_BLEND_MODEL if blend_model is None else str(blend_model))
    if not os.path.isfile(model):
        raise _refuse("selection_policy", "blend selector model not found: %s" % model)
    thr = _selection_float(
        "--blend-thr", DEFAULT_BLEND_THR if blend_thr is None else blend_thr, 0.0, 1.0, high_open=True
    )
    tau = _selection_float(
        "--blend-tau",
        DEFAULT_BLEND_TAU if blend_tau is None else blend_tau,
        0.0,
        1.0,
        high_open=True,
    )
    cap = _selection_int(
        "--blend-cap", DEFAULT_BLEND_CAP if blend_cap is None else blend_cap, 1, DECODER_MAX_DETECTIONS
    )
    module = _nmsbox()
    return {
        "rule": BLEND_RULE,
        "selector": "Blend (HistGradientBoosting + ExtraTrees average), p = predict_proba(x)[:, 1]",
        "model": model,
        "model_bytes": os.path.getsize(model),
        "threshold": thr,
        "box_iou": tau,
        "per_image_cap": cap,
        "max_detections": DECODER_MAX_DETECTIONS,
        "stuff_dropped_from": module.STUFF_CLASS_START,
        "features": {
            "source": "delivery/features26.py compute_features",
            "columns": BLEND_FEATURE_COUNT,
            "mask_source": "ink_box_pad2",
            "boxes": "bbox_from_mask of the REFINED mask, [x_min, y_min, x_max+1, y_max+1]",
            "ink": "refine.ink_from_png(--refine-ink), the same raster refinement used",
        },
        "order": (
            "candidates (pool decode: mask threshold, score threshold, <= 300) -> a6 refinement "
            "-> 26 features of the refined mask -> blend probability -> Things only (class < 30) "
            "-> p >= threshold (inclusive) -> greedy in probability-descending order, ties keep "
            "decode order; a kept box suppresses same-class boxes with IoU >= box_iou (inclusive) "
            "-> global max-detections cap (survivor order kept) -> per-image cap (top-N by "
            "probability, ties keep decode order)"
        ),
        "comparisons": {
            "score": "blend probability >= threshold (inclusive)",
            "nms": "same-class box IoU >= box_iou (inclusive)",
            "cap": "survivor order kept (probability-descending); ties keep decode order",
            "per_image_cap": "top-N by probability, ties keep decode order",
        },
        "replaces": "the delivered readout only; the plain pool selection still runs first because "
        "the selector scores the refined candidates it produced",
        "applied_in": "cad_mrcnn_delivery.DeliveryRun.emit -> _blend_stage -> "
        "nmsbox.select_by_box_nms, i.e. after refinement and before the square-canvas pad and "
        "the emitter",
        "measured": {
            "space": "the 387 held-out records, native raster, strict IoU > 0.5, micro PQ "
            "(tools/score_local.py; a local held-out number, never an official score)",
            "points": {
                "blend(0.70, 0.15, cap 28)": 0.247198,
                "blend(0.60, 0.20, cap 20)": 0.240847,
                "wide nmsbox(0.97, 0.10, cap 40)": 0.163523,
            },
            "chosen": "blend(0.70, 0.15, cap 28): TP 763 / FP 518 / FN 2281, SumIoU 534.57, "
            "SQ 0.7006, RQ 0.3528",
            "recipe": "tools/blend_eval.py extract + score; evidence/blend-deployable/REPORT.md",
            "note": "a local held-out measurement; this CLI does not re-measure it",
        },
    }


SELECT_RULES = ("plain", "nmsbox", "blend")
DEFAULT_SELECT_RULE = "plain"
DEFAULT_NMS_BOX_IOU = 0.10
DEFAULT_NMS_BOX_THR = 0.97


def validate_box_nms(select_rule=None, nms_box_iou=None, nms_box_thr=None):
    """Validate ``--select-rule``; returns the block only for the non-default rule.

    ``plain`` (the default) is the measured path this module has always run: the
    same-class *mask* IoU NMS, the strict score threshold, then the caps.  It
    returns ``None``, so the default policy dict, the selection signature and
    every receipt stay byte-identical.

    ``nmsbox`` replaces that stage with the wide-pool readout: raw score
    >= ``--nms-box-thr`` (0.97), greedy same-class *box* IoU suppression at
    ``--nms-box-iou`` (0.10), then the caps.  It needs the decoded boxes, which
    the run carries from ``predict`` into ``emit``; nothing else changes -- mask
    threshold, refinement and emitter are the same.
    """

    rule = DEFAULT_SELECT_RULE if select_rule is None else str(select_rule)
    if rule not in SELECT_RULES:
        raise _refuse(
            "selection_policy",
            "--select-rule refused: expected one of %r, got %r" % (list(SELECT_RULES), select_rule),
        )
    if rule == DEFAULT_SELECT_RULE:
        return None
    if rule == BLEND_RULE:
        # The blend rule is a *post-refinement* stage (validate_blend); the pool
        # selection that runs before it stays the measured plain rule.
        return None

    iou = _selection_float(
        "--nms-box-iou",
        DEFAULT_NMS_BOX_IOU if nms_box_iou is None else nms_box_iou,
        0.0,
        1.0,
        high_open=True,
    )
    thr = _selection_float(
        "--nms-box-thr",
        DEFAULT_NMS_BOX_THR if nms_box_thr is None else nms_box_thr,
        0.0,
        1.0,
        high_open=True,
    )
    module = _nmsbox()
    return {
        "rule": "nmsbox",
        "box_iou": iou,
        "box_threshold": thr,
        "stuff_dropped_from": module.STUFF_CLASS_START,
        "order": module.BOX_SELECT_ORDER,
        "comparisons": {
            "score": "score >= box_threshold (inclusive)",
            "nms": "same-class box IoU >= box_iou (inclusive)",
            "cap": "survivor order kept (score-descending); ties keep decode order",
            "per_image_cap": "top-N by score, ties keep decode order",
        },
        "replaces": "the same-class mask IoU NMS and the strict score threshold of the plain rule",
        "inputs": "decoded xyxy boxes carried from the predictor into the selection stage",
        "applied_in": "cad_mrcnn_delivery.select_instances -> nmsbox.select_by_box_nms",
        "measured": {
            "space": "feature space (delivery/v2lib.py eval_keep), 387 held-out records, 1000px "
                     "raster, strict IoU > 0.5, micro PQ",
            "points": {
                "plain(0.99, cap 20)": 0.147478,
                "nmsbox(0.80, 0.20, cap 20)": 0.119681,
                "nmsbox(0.97, 0.10, cap 40)": 0.166234,
            },
            "chosen": "nmsbox(0.97, 0.10, cap 40): TP 691 / FP 1957 / FN 2353, emitted 2648 "
                      "(~6.8 per image); the grid is flat from tau 0.05 to 0.20",
            "calibration": "probs_blend.pkl self-check: blend at .60/.20/20 measured 0.219650 vs "
                           "canonical 0.220984 with equal emitted counts at three points, i.e. a "
                           "systematic offset of -0.0013..-0.0015; canonical estimate for the chosen "
                           "point ~0.1676",
            "note": "a local held-out measurement, never an official or competition score; this CLI "
                    "does not re-measure it",
        },
    }


STUFF_CLASSES = (30, 31, 32, 33, 34)


def validate_drop_stuff(drop_stuff=None):
    """Validate ``--drop-stuff``; returns the block only when the flag is on.

    The 36-class vocabulary carries five classes (30..34) that the measured
    readout drops before scoring: they are the "stuff" tail of the label set and
    every one of them costs more in FP than it earns in TP at the shipping
    operating point.  The emitter paints a Stuff class as ONE merged
    ``instanceId == -1`` group and never as a box, so the drop is applied to the
    decoded instances just before the delivery selection: a Stuff detection then
    cannot consume a ``--per-image-cap`` slot that a Thing could use, and the
    surviving Things are renumbered densely by the emitter.
    """

    if not drop_stuff:
        return None
    return {
        "classes": list(STUFF_CLASSES),
        "rule": "drop every decoded instance whose class id is in classes, before the selection",
        "stage": "after the decoder's nms/threshold, before threshold -> max-detections -> per-image-cap",
        "renumbering": "dense, by the emitter (instanceIds/obj_cts renumbered in order)",
        "not_a_filter_on_ink": True,
        "measured_28": {"pq_without": 0.120714704164, "pq_with": 0.125862769467},
        "measured_112": {"pq_without": 0.137443379498, "pq_with": None},
        "why": "classes 30..34 are the stuff tail of the vocabulary: dropping them removes more "
        "false positives than true positives at this operating point, and a delivered primitive "
        "of a stuff class is never a glyph",
    }


def validate_mask_paste(mode=None, logit_pad=None):
    """Validate ``--mask-paste`` / ``--mask-logit-pad``; ``None`` for the default.

    Only a non-default mode enters the policy, so the default policy dict, the
    selection signature, every document and the receipt stay byte-identical to
    the runs that predate the flag.
    """

    value = DEFAULT_MASK_PASTE if mode is None else mode
    if not isinstance(value, str) or value not in mask_paste.MASK_PASTE_MODES:
        raise _refuse(
            "selection_policy",
            "--mask-paste must be one of %r (got %r)"
            % (list(mask_paste.MASK_PASTE_MODES), value),
        )
    if value == DEFAULT_MASK_PASTE:
        if logit_pad is not None:
            raise _refuse(
                "selection_policy",
                "--mask-logit-pad only applies to --mask-paste %r (got %r with %r)"
                % (mask_paste.MASK_PASTE_LOGITS, logit_pad, value),
            )
        return None
    pad = DEFAULT_MASK_LOGIT_PAD if logit_pad is None else logit_pad
    if isinstance(pad, bool) or not isinstance(pad, (int, float)):
        raise _refuse("selection_policy", "--mask-logit-pad must be a number, got %r" % (pad,))
    pad = float(pad)
    if not math.isfinite(pad) or pad > -1.0:
        raise _refuse(
            "selection_policy",
            "--mask-logit-pad must be finite and <= -1.0, got %r: it is the logit of the cell "
            "torchvision pads around the mask, and the probability path's 0.0 stands for certain "
            "background, not for p=0.5" % (pad,),
        )
    return {
        "mode": value,
        "logit_pad": pad,
        "logit_zero_probability": 0.5,
        "boundary": "logit >= 0, i.e. sigmoid(logit) >= 0.5: the decision boundary of the "
        "probability paste is unchanged because sigmoid is monotone",
        "stage": "torchvision.models.detection: the mask sigmoid is deferred from "
        "roi_heads.maskrcnn_inference to after the native-canvas bilinear paste, and "
        "roi_heads.expand_masks pads with logit_pad instead of 0.0",
        "unchanged": [
            "predicted-class mask channel selection",
            "expand_boxes(boxes, scale).to(torch.int64) box integerisation",
            "canvas clipping and therefore the instance count",
            "the lane's mask_threshold test (probabilities >= 0.5)",
        ],
        "applied_in": "cad_mrcnn_delivery.DeliveryRun.predict wrapper (mask_paste.wrap_predict)",
    }


def validate_ink_box(min_side=None, stage=None):
    """Validate ``--mask-ink-min-side`` / ``--mask-ink-stage``.

    ``min_side`` is the box side (px) above which a selected instance's mask is
    replaced by the PNG ink inside its box widened by ``INK_BOX_PAD`` px.  ``0``
    returns ``None`` (no policy block, no stage: the documents are what they
    always were), which is the previous behaviour and the escape hatch for A/B
    runs.  ``stage`` is ``post`` (after selection: the selector is untouched,
    only the emitted document changes -- the measured placement) or ``pre``
    (before the blend selector, whose boxes and 26 features then read ink boxes).
    """

    value = DEFAULT_INK_MIN_SIDE if min_side is None else min_side
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse(
            "selection_policy",
            "--mask-ink-min-side must be an integer number of pixels, got %r" % (value,),
        )
    low, high = INK_MIN_SIDE_RANGE
    if value < low or value > high:
        raise _refuse(
            "selection_policy",
            "--mask-ink-min-side must be in %d..%d (0 disables the rule), got %r"
            % (low, high, value),
        )
    placement = DEFAULT_INK_STAGE if stage is None else str(stage)
    if placement not in INK_STAGES:
        raise _refuse(
            "selection_policy",
            "--mask-ink-stage must be one of %s, got %r" % (", ".join(INK_STAGES), stage),
        )
    if value == 0:
        return None
    return {
        "min_side": int(value),
        "crop_pad": int(INK_BOX_PAD),
        "stage_placement": placement,
        "rule": "max(box_w, box_h) > min_side -> mask = ink & crop(box widened by crop_pad px, "
        "clamped to the raster); the candidate's own predicted mask is ignored",
        "empty_fallback": "an empty ink crop keeps the candidate's refined mask",
        "box": "bbox_from_mask of the refined mask, [x_min, y_min, x_max + 1, y_max + 1], the "
        "same box the selection stage and the 26 features use",
        "ink": "the run's ink raster (--refine-ink), the same map the refinement used",
        "stage": (
            "after selection and before the square-canvas pad: the blend selector and its 26 "
            "features read the refined masks exactly as before, and only the selected instances "
            "whose box exceeds min_side reach the emitter as ink"
            if placement == "post"
            else "after the a6 refinement, before the blend selector builds the 26 features and "
            "before the square-canvas pad"
        ),
        "applied_in": "cad_mrcnn_delivery.DeliveryRun.emit -> _ink_box_stage -> "
        "cad_mrcnn_delivery.ink_box_masks (%s)" % ("after_selection" if placement == "post"
                                                  else "after_refine_before_blend"),
        "may_reduce": (
            "the emitted box of a replaced instance shrinks to its ink (the mask is the ink "
            "inside the old box); selection, counts and features are unaffected"
            if placement == "post"
            else "the emitted box of a replaced instance shrinks to its ink (the mask is the "
            "ink inside the old box), so the blend selector's box IoU is measured on ink boxes"
        ),
        "measured": INK_BOX_MEASURED,
        "measured_in_delivery": INK_BOX_DELIVERY_MEASURED,
    }


def validate_mask_source(source=None, min_side=None, tau=None, gate=None, checkpoint=None):
    """Validate ``--mask-source`` and the 256^2 head's knobs; return the policy block.

    ``mask`` (the default) is the shipped behaviour and returns ``None``: no block,
    no stage, no model load, and the documents are byte-identical to earlier runs.
    ``head`` replaces the emitted mask of every selected instance whose box side
    exceeds ``min_side`` with the stroke head's prediction (``ink >= gate`` and
    ``prob >= tau`` inside its own window).  ``ink`` selects the measured
    ink-in-box arm instead, which reads ``--mask-ink-min-side`` /
    ``--mask-ink-stage``; the block only records that choice.
    """

    choice = DEFAULT_MASK_SOURCE if source is None else str(source)
    if choice not in MASK_SOURCES:
        raise _refuse(
            "selection_policy",
            "--mask-source must be one of %s, got %r" % (", ".join(MASK_SOURCES), source),
        )
    if choice != "head":
        if choice == "mask":
            return None
        return {
            "source": "ink",
            "rule": "the measured ink-in-box arm: --mask-ink-min-side / --mask-ink-stage",
            "applied_in": "cad_mrcnn_delivery.DeliveryRun.emit -> _ink_box_stage -> "
            "cad_mrcnn_delivery.ink_box_masks",
            "measured": INK_BOX_DELIVERY_MEASURED,
        }

    side = _selection_int(
        "--mask-head-min-side",
        DEFAULT_MASK_HEAD_MIN_SIDE if min_side is None else min_side,
        0,
        4096,
    )
    threshold = _selection_float(
        "--mask-head-tau",
        DEFAULT_MASK_HEAD_TAU if tau is None else tau,
        0.0,
        1.0,
        high_open=True,
    )
    level = _selection_int(
        "--mask-head-gate",
        DEFAULT_MASK_HEAD_GATE if gate is None else gate,
        1,
        255,
    )
    module = _mask_source()
    if checkpoint is None:
        path = module.default_checkpoint()
        origin = "weights" if os.path.normcase(path) == os.path.normcase(module.WEIGHTS_CKPT) \
            else "runs"
    else:
        path = os.path.abspath(str(checkpoint))
        origin = "flag"
    if not os.path.isfile(path):
        raise _refuse(
            "selection_policy",
            "--mask-head-ckpt: no stroke-head checkpoint at %s (pass --mask-head-ckpt)" % (path,),
        )
    return {
        "source": "head",
        "min_side": int(side),
        "tau": float(threshold),
        "gate": int(level),
        "checkpoint": path,
        "checkpoint_origin": origin,
        "size": int(module.HEAD_SIZE),
        "width": int(module.HEAD_WIDTH),
        "parameters": int(module.HEAD_PARAMETERS),
        "rule": "max(box_w, box_h) > min_side -> mask = (ink >= gate) AND (prob >= tau) of the "
        "256^2 stroke head, pasted back into its own window; the candidate's own mask is ignored",
        "empty_fallback": "an empty head prediction keeps the candidate's refined mask",
        "window": "side = max(box_side, 64) + 8 px, centred on the refined mask's box, "
        "zero-padded outside the raster",
        "stage": "after selection and before the square-canvas pad: the blend selector and its "
        "26 features read the refined masks exactly as before, and only the selected instances "
        "whose box exceeds min_side reach the emitter as head masks",
        "applied_in": "cad_mrcnn_delivery.DeliveryRun.emit -> _mask_head_stage -> "
        "delivery/mask_source.head_masks (after_selection)",
        "measured": MASK_HEAD_MEASURED,
    }


def validate_selection(
    *,
    score_threshold=None,
    mask_threshold=None,
    nms_iou=None,
    max_detections=None,
    per_image_cap=None,
    refine_mask=None,
    refine_ink=None,
    refine_ratio=None,
    drop_stuff=None,
    mask_paste=None,
    mask_logit_pad=None,
    mask_ink_min_side=None,
    mask_ink_stage=None,
    mask_source=None,
    mask_head_min_side=None,
    mask_head_tau=None,
    mask_head_gate=None,
    mask_head_ckpt=None,
    select_rule=None,
    nms_box_iou=None,
    nms_box_thr=None,
    blend_model=None,
    blend_thr=None,
    blend_tau=None,
    blend_cap=None,
):
    """Validate the five selection knobs and the mask refinement; return the policy.

    Called before any checkpoint, PNG or inputs access.  Defaults are the model's
    measured decode settings, so the flags are additive.  The refinement knobs are
    carried in their own ``refine`` block: ``signature`` stays the five-number
    selection signature the receipts and the frozen tests pin.  ``--select-rule
    nmsbox`` is carried in its own ``box_nms`` block for the same reason, and
    ``--select-rule blend`` in a ``blend`` block: it leaves this pre-refinement
    stage alone and adds the frozen post-refinement readout.
    """

    score = _selection_float(
        "--score-threshold",
        DEFAULT_SCORE_THRESHOLD if score_threshold is None else score_threshold,
        0.0,
        1.0,
        high_open=True,
    )
    mask = _selection_float(
        "--mask-threshold",
        DEFAULT_MASK_THRESHOLD if mask_threshold is None else mask_threshold,
        0.0,
        1.0,
        high_open=True,
    )
    iou = _selection_float(
        "--nms-iou",
        DEFAULT_NMS_IOU if nms_iou is None else nms_iou,
        0.0,
        1.0,
        high_open=False,
    )
    cap = _selection_int(
        "--max-detections",
        DEFAULT_MAX_DETECTIONS if max_detections is None else max_detections,
        1,
        DECODER_MAX_DETECTIONS,
    )
    per_image = None
    if per_image_cap is not None:
        per_image = _selection_int("--per-image-cap", per_image_cap, 1, cap)

    if not any(abs(mask - choice) <= 1e-12 for choice in MASK_THRESHOLD_CHOICES):
        raise _refuse(
            "selection_policy",
            "--mask-threshold refused: the frozen decoder binarises every mask at this value "
            "before the selection stage sees it, so only the measured choices %r are accepted "
            "(got %r)" % (list(MASK_THRESHOLD_CHOICES), mask),
        )

    paste = validate_mask_paste(mask_paste, mask_logit_pad)
    head_block = validate_mask_source(
        mask_source, mask_head_min_side, mask_head_tau, mask_head_gate, mask_head_ckpt
    )
    ink_side = mask_ink_min_side
    if head_block is not None and head_block.get("source") == "ink" and (
        ink_side is None or int(ink_side) == 0
    ):
        # ``--mask-source ink`` names the measured arm, whose threshold is 32 and
        # whose placement is post; an explicit --mask-ink-min-side still wins.
        ink_side = 32
    ink_box = validate_ink_box(ink_side, mask_ink_stage)
    if head_block is not None and head_block.get("source") == "ink":
        head_block["min_side"] = int((ink_box or {}).get("min_side") or 0)
        head_block["stage_placement"] = (ink_box or {}).get("stage_placement")
    refine = validate_refine(refine_mask, refine_ink, refine_ratio)
    stuff = validate_drop_stuff(drop_stuff)
    box_nms = validate_box_nms(select_rule, nms_box_iou, nms_box_thr)
    blend = validate_blend(select_rule, blend_model, blend_thr, blend_tau, blend_cap)

    policy = {
        "order": SELECTION_ORDER,
        "score_threshold": score,
        "mask_threshold": mask,
        "nms_iou": iou,
        "max_detections": cap,
        "per_image_cap": per_image,
        "comparisons": {
            "score": "score > score_threshold (strict)",
            "nms": "same-class mask IoU > nms_iou (strict)",
            "cap": "decode order kept; ties keep decode order",
            "per_image_cap": "top-N by score, ties keep decode order, output stays in decode order",
        },
        "applied_in": "CLI selection stage, after decode and before emit",
        "refine": refine,
        "decoder_fixed": {
            "box_score_thresh": DEFAULT_SCORE_THRESHOLD,
            "box_nms_thresh": DEFAULT_NMS_IOU,
            "box_detections_per_img": DECODER_MAX_DETECTIONS,
            "mask_threshold": DEFAULT_MASK_THRESHOLD,
        },
        "signature": [score, mask, iou, cap, per_image],
    }
    if stuff is not None:
        # Only a non-default knob enters the policy, so the default policy dict --
        # and the documents, selection report and signature built from it -- stay
        # exactly what earlier runs produced.
        policy["drop_stuff"] = stuff
    if paste is not None:
        policy["mask_paste"] = paste
    if ink_box is not None:
        # Same rule as the blocks above: 0 (disabled) enters nothing, so the
        # previous policy dict, receipts and documents stay byte-identical.
        policy["ink_box"] = ink_box
    if head_block is not None:
        # ``--mask-source head`` / ``ink``: a post-selection replacement of the
        # emitted mask.  ``mask`` (the default) enters nothing at all.
        policy["mask_source"] = head_block
    if box_nms is not None:
        # Same rule as the blocks above: only a non-default rule enters the
        # policy, so the default policy dict stays exactly what earlier runs
        # produced.  The stage order is replaced, so ``order`` follows suit.
        policy["box_nms"] = box_nms
        policy["order"] = box_nms["order"]
    if blend is not None:
        # ``blend`` adds a post-refinement stage instead of replacing the rule
        # above: the plain pool selection still runs first (its counts are kept in
        # the per-image ``blend.pool_counts``), and the blend rule decides what is
        # emitted.  ``order`` therefore describes the delivered readout.
        policy["blend"] = blend
        policy["order"] = blend["order"]
    return policy


def selection_signature(policy):
    return tuple(policy["signature"])


def validate_mask_head(mask_head=None, chunk=None):
    """Validate ``--mask-head`` / ``--mask-chunk``; returns the mask-head block.

    Nothing here touches torch, the checkpoint or a PNG: the head is a build-time
    choice, validated with the other knobs.  ``--mask-chunk`` is only used by the
    112 head; on the 28 head it is accepted and recorded as ``applied: false`` so a
    shared command line stays valid.
    """

    head = DEFAULT_MASK_HEAD if mask_head is None else str(mask_head)
    if head not in MASK_HEADS:
        raise _refuse(
            "mask_head",
            "--mask-head refused: expected one of %s, got %r" % (", ".join(MASK_HEADS), mask_head),
        )
    size = MASK_HEAD_OUTPUT[head]
    try:
        chunk_value = int(DEFAULT_MASK_CHUNK if chunk is None else chunk)
    except (TypeError, ValueError):
        raise _refuse(
            "selection_policy", "--mask-chunk refused: integer required, got %r" % (chunk,)
        ) from None
    chunk_value = _selection_int("--mask-chunk", chunk_value, *MASK_CHUNK_RANGE)
    applied = head != DEFAULT_MASK_HEAD
    return {
        "head": head,
        "output_size": size,
        "pool_size": 14,
        "deconv_stages": 3,
        "parameters": 796452 if applied else None,
        "chunk": chunk_value if applied else None,
        "chunk_requested": chunk_value,
        "chunk_applied": applied,
        "checkpoint_must_declare": (
            "fine_mask.output_size == 112" if applied else "the frozen 28x28 mask predictor"
        ),
        "applied_in": (
            "loader only: the model is built with the fine predictor, loaded strictly and wrapped "
            "in the chunked mask runner; selection, refinement, paste and emit are the same code "
            "as the 28 head"
        ),
        "measured": dict(MASK_HEAD_MEASURED),
    }


def _fine_head():
    """Import ``fine_head.py`` next to this file (lazy: torch-free knob validation)."""

    import importlib.util  # noqa: PLC0415

    module = sys.modules.get("cad_fine_head")
    if module is not None:
        return module
    path = os.path.join(DELIVERY_ROOT, "fine_head.py")
    spec = importlib.util.spec_from_file_location("cad_fine_head", path)
    if spec is None or spec.loader is None:  # pragma: no cover - ships with the CLI
        raise _refuse("mask_head", "fine_head.py is missing next to the CLI (%s)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_fine_head"] = module
    spec.loader.exec_module(module)
    return module


def fine_head_loader(chunk=None):
    """A ``loader(checkpoint, device) -> context`` that installs the 112 head.

    Mirrors the fine-head lane's loader exactly: build through the project
    builder, install the fine predictor, ``load_state_dict(strict=True)``, move to
    the device, ``eval()``, set the inference transform, then wrap the mask branch
    in the chunked runner.  ``instance_eval.batch``'s driver, predict, emit and the
    whole selection/refinement stage are the same code the 28 head uses.
    """

    chunk_value = int(DEFAULT_MASK_CHUNK if chunk is None else chunk)

    def load(checkpoint, device):
        head = _fine_head()
        import torch  # noqa: PLC0415 - only the 112 head path needs torch here

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise _refuse("device_unavailable", "CUDA requested but unavailable")
        try:
            model, config, info, payload = head.load_fine_checkpoint(
                checkpoint, device=target, model_root=MODEL_ROOT
            )
        except RuntimeError as exc:
            raise _refuse(
                "mask_head",
                "--mask-head 112 could not load %s: %s (the checkpoint must carry a 112x112 "
                "mask predictor)" % (checkpoint, exc),
            ) from None
        chunk_report = head.install_chunked_mask(model, chunk=chunk_value)
        info = dict(info)
        info["mask_chunk"] = {key: value for key, value in chunk_report.items() if key != "runner"}
        runner = chunk_report.get("runner")
        mask_head = head.head_report(payload, info.get("mask_predictor") or {})
        mask_head["chunk"] = chunk_value
        mask_head["chunked"] = bool(chunk_report.get("installed"))
        return {
            "infer": _batch()._infer_module(),
            "model": model,
            "config": config,
            "info": info,
            "payload": payload,
            "device": target,
            "max_detections": int(getattr(model.roi_heads, "detections_per_img", 0) or 0),
            "mask_head": mask_head,
            "mask_runner": runner,
        }

    return load


# --------------------------------------------------------------------------- #
# selection arithmetic
# --------------------------------------------------------------------------- #
def _mask_span(mask):
    """Tight bbox of a boolean mask as (x0, y0, x1, y1) with exclusive right/bottom.

    Restricting the AND to the bbox intersection cannot change the count: a pixel
    cannot be set in two masks outside both of their bounding boxes.
    """

    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _roi_intersection(mask, span, previous, previous_span):
    import numpy as np

    x0 = max(span[0], previous_span[0])
    y0 = max(span[1], previous_span[1])
    x1 = min(span[2], previous_span[2])
    y1 = min(span[3], previous_span[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1] & previous[y0:y1, x0:x1]))


def select_instances(
    masks,
    class_ids,
    scores,
    *,
    score_threshold,
    nms_iou,
    max_detections,
    per_image_cap=None,
    boxes=None,
    select_rule=None,
    nms_box_iou=None,
    nms_box_thr=None,
):
    """Apply the delivery selection to one image's decoded instances.

    ``masks`` are the binarised per-instance masks in decode order, ``class_ids``
    are canonical class ids 0..34 and ``scores`` are the detector confidences.
    Returns the surviving decode indices plus the stage counts.

    ``select_rule="nmsbox"`` replaces the mask-IoU NMS and the score threshold
    with the wide-pool box rule (``nmsbox.select_by_box_nms``) and needs
    ``boxes`` (xyxy, decode order); the caps behave the same way.  Every other
    value -- including the default -- runs the plain mask path and ignores
    ``boxes`` entirely.
    """

    import numpy as np

    if str(select_rule or DEFAULT_SELECT_RULE) != DEFAULT_SELECT_RULE:
        if str(select_rule) not in SELECT_RULES:
            raise _refuse(
                "selection_policy",
                "--select-rule refused: expected one of %r, got %r"
                % (list(SELECT_RULES), select_rule),
            )
        if boxes is None:
            raise _refuse(
                "box_nms_boxes",
                "--select-rule nmsbox needs the decoded boxes, but none were supplied",
            )
        return _nmsbox().select_by_box_nms(
            boxes,
            class_ids,
            scores,
            score_threshold=DEFAULT_NMS_BOX_THR if nms_box_thr is None else float(nms_box_thr),
            box_iou=DEFAULT_NMS_BOX_IOU if nms_box_iou is None else float(nms_box_iou),
            max_detections=max_detections,
            per_image_cap=per_image_cap,
        )

    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    labels = [int(value) for value in class_ids]
    values = [float(value) for value in scores]
    if not (len(masks) == len(labels) == len(values)):
        raise ValueError(
            "selection input mismatch: %d masks, %d class ids, %d scores"
            % (len(masks), len(labels), len(values))
        )
    for index, mask in enumerate(masks):
        if mask.ndim != 2:
            raise ValueError("selection mask %d is not 2-D: shape %r" % (index, mask.shape))

    candidates = len(masks)
    spans = [_mask_span(mask) for mask in masks]
    areas = [int(np.count_nonzero(mask)) for mask in masks]

    kept = []
    nms_suppressed = 0
    for index in range(candidates):
        span = spans[index]
        suppressed = False
        if span is not None and areas[index] > 0:
            mask = masks[index]
            for other in kept:
                other_span = spans[other]
                if other_span is None or areas[other] <= 0:
                    continue
                if labels[other] != labels[index]:
                    continue
                intersection = _roi_intersection(mask, span, masks[other], other_span)
                if intersection == 0:
                    continue
                union = areas[index] + areas[other] - intersection
                if union > 0 and (intersection / union) > nms_iou:
                    suppressed = True
                    break
        if suppressed:
            nms_suppressed += 1
        else:
            kept.append(index)

    over_threshold = [index for index in kept if values[index] > score_threshold]
    threshold_dropped = len(kept) - len(over_threshold)

    capped = over_threshold[:max_detections]
    cap_dropped = len(over_threshold) - len(capped)

    per_image_dropped = 0
    selected = capped
    if per_image_cap is not None and len(capped) > per_image_cap:
        top = set(sorted(capped, key=lambda index: (-values[index], index))[:per_image_cap])
        selected = [index for index in capped if index in top]
        per_image_dropped = len(capped) - len(selected)

    lowest = min((values[index] for index in selected), default=None)
    counts = {
        "candidates": candidates,
        "after_nms": len(kept),
        "after_threshold": len(over_threshold),
        "after_cap": len(capped),
        "after_per_image_cap": len(selected),
        "selected": len(selected),
        "nms_suppressed": nms_suppressed,
        "threshold_dropped": threshold_dropped,
        "cap_dropped": cap_dropped,
        "per_image_cap_dropped": per_image_dropped,
    }
    return {"indices": selected, "counts": counts, "lowest_selected_score": lowest}


# --------------------------------------------------------------------------- #
# path / knob checks that need the filesystem (stat only, never a content read)
# --------------------------------------------------------------------------- #
def _under_official_datasets(path):
    """True when the parent chain of ``path`` holds a directory named ``datasets``."""
    parts = [part for part in str(path).replace("\\", "/").lower().split("/") if part]
    return OFFICIAL_INPUT_DIRNAME in parts[:-1]


def _check_sealed(path, label, allow_official_inputs=False):
    text = str(path).replace("\\", "/").lower()
    for token in SEALED_TOKENS:
        if token in text:
            if allow_official_inputs and _under_official_datasets(path):
                # the one documented exception: an input PNG of the official
                # corpora, for an offline self-check run
                continue
            raise _refuse("sealed_path", "%s resolves into protected data: %s" % (label, path))


def validate_delivery_paths(
    *,
    out,
    checkpoint,
    png=None,
    inputs=None,
    device=None,
    allow_official_inputs=False,
    resume=False,
):
    """Stat-only validation of every path and the device string.

    ``resume`` skips the fresh-directory rule only; the directory's contents are
    then checked against :data:`RESUME_BOOKKEEPING` by ``validate_resume_out``
    once the sample list is known.
    """

    if (png is None) == (inputs is None):
        raise _refuse("inputs_choice", "give exactly one of --png or --inputs")
    device = DEFAULT_DEVICE if device is None else str(device)
    if not DEVICE_RE.match(device):
        raise _refuse("device", "--device refused: expected cpu or cuda[:N], got %r" % (device,))

    out = os.path.abspath(str(out))
    _check_sealed(out, "--out")

    checkpoint = os.path.abspath(str(checkpoint))
    _check_sealed(checkpoint, "--checkpoint")
    if not os.path.isfile(checkpoint):
        raise _refuse("checkpoint_missing", "--checkpoint not found: %s" % checkpoint)

    if png is not None:
        png = os.path.abspath(str(png))
        _check_sealed(png, "--png", allow_official_inputs)
        if not png.lower().endswith(".png"):
            raise _refuse("png_required", "--png must be a .png: %s" % png)
        if not os.path.isfile(png):
            raise _refuse("png_missing", "--png not found: %s" % png)
    else:
        inputs = os.path.abspath(str(inputs))
        _check_sealed(inputs, "--inputs")
        if not os.path.isfile(inputs):
            raise _refuse("inputs_missing", "--inputs not found: %s" % inputs)

    if os.path.isfile(out):
        raise _refuse("out_not_directory", "--out is a file: %s" % out)
    if os.path.isdir(out) and not resume:
        existing = sorted(os.listdir(out))
        if existing:
            raise _refuse(
                "out_not_empty",
                "--out already holds %d entries (%s); delivery writes into a fresh directory"
                % (len(existing), ", ".join(existing[:5])),
            )

    return {
        "out": out,
        "checkpoint": checkpoint,
        "device": device,
        "png": png,
        "inputs": inputs,
    }


# --------------------------------------------------------------------------- #
# image-only inputs
# --------------------------------------------------------------------------- #
def _png_stem(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def image_only_inputs(
    *, png=None, inputs=None, sample_id=None, group_id=None, allow_official_inputs=False
):
    """Build the image-only inputs payload used for prediction.

    Only ``sample_id``, ``group_id`` and ``png_path`` are carried over.  Every
    other row key (``svg_path``, ``record_index``, ``position`` and anything a
    competitor might add, ground truth included) is dropped and merely counted in
    ``ignored_row_keys`` so the audit shows it was seen and not used.
    """

    rows = []
    derived = {"png": None, "inputs": None, "inputs_sha256": None}
    ignored = set()

    if png is not None:
        path = os.path.abspath(str(png))
        rows.append(
            {
                "sample_id": sample_id or _png_stem(path),
                "group_id": group_id,
                "png_path": path,
            }
        )
        derived["png"] = path
    else:
        source = os.path.abspath(str(inputs))
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            schema = payload.get("schema")
            if schema is not None and schema != SCHEMA_INPUTS:
                raise _refuse("inputs_schema", "--inputs schema must be %r, got %r" % (SCHEMA_INPUTS, schema))
            source_rows = payload.get("samples")
        else:
            source_rows = payload
        if not isinstance(source_rows, list) or not source_rows:
            raise _refuse("inputs_shape", "--inputs must hold a non-empty 'samples' list")
        derived["inputs"] = source
        derived["inputs_sha256"] = sha256_file(source)
        for index, row in enumerate(source_rows):
            if not isinstance(row, dict):
                raise _refuse("inputs_shape", "sample %d is not an object" % index)
            raw = row.get("png_path")
            if not isinstance(raw, str) or not raw.strip():
                raise _refuse("inputs_png", "sample %d has no png_path" % index)
            path = os.path.abspath(raw)
            if not path.lower().endswith(".png"):
                raise _refuse("png_required", "sample %d png_path is not a .png: %s" % (index, path))
            _check_sealed(path, "sample %d png_path" % index, allow_official_inputs)
            if not os.path.isfile(path):
                raise _refuse("png_missing", "sample %d png_path not found: %s" % (index, path))
            row_sample_id = row.get("sample_id")
            if row_sample_id in (None, ""):
                row_sample_id = _png_stem(path)
            if not isinstance(row_sample_id, str) or not row_sample_id.strip():
                raise _refuse("inputs_sample_id", "sample %d has an empty sample_id" % index)
            row_group_id = row.get("group_id")
            if row_group_id is not None and not isinstance(row_group_id, str):
                row_group_id = None
            for key in row:
                if key not in ("sample_id", "group_id", "png_path"):
                    ignored.add(str(key))
            rows.append(
                {"sample_id": row_sample_id, "group_id": row_group_id, "png_path": path}
            )

    seen_ids = set()
    stems = {}
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise _refuse("inputs_duplicate", "duplicate sample_id %r" % row["sample_id"])
        seen_ids.add(row["sample_id"])
        stem = _png_stem(row["png_path"])
        if stem in stems and stems[stem] != row["png_path"]:
            raise _refuse(
                "results_stem_collision",
                "two inputs share the results stem %r: %s vs %s"
                % (stem, stems[stem], row["png_path"]),
            )
        stems[stem] = row["png_path"]

    return {
        "schema": SCHEMA_INPUTS,
        "created_utc": _utc_now(),
        "dataset": "cad-png-delivery-mrcnn01",
        "role": "competitor_delivery_image_only",
        "image_only": True,
        "field_note": "sample_id, group_id, png_path only; no ground truth, no svg_path",
        "derived_from": derived,
        "ignored_row_keys": sorted(ignored),
        "count": len(rows),
        "samples": rows,
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _write_failure(out, stage, exc):
    """Best-effort durable failure record; never masks the original error."""

    try:
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = os.path.join(
            os.path.abspath(str(out)), "failures", "failure_%s_%s.json" % (stamp, stage)
        )
        atomic_json(
            target,
            {
                "schema": "cad-png-delivery-mrcnn01.failure/1",
                "created_utc": _utc_now(),
                "stage": stage,
                "error_type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "detail": str(exc),
                "argv": list(sys.argv),
            },
        )
        return target
    except Exception:  # pragma: no cover - diagnostics only
        return None


def validate_resume_out(out, samples, results_dir=None):
    """Refuse ``--resume`` over a directory holding anything but this run's output.

    Allowed: the run's own documents (``<stem>.json`` under ``results/``, and the
    flat copies ``infer.py`` mirrors beside them), :data:`RESUME_BOOKKEEPING`, and
    the run directories of :data:`RESUME_RUN_DIRS`.  A PNG, a stray JSON or an
    unknown directory means the directory is not ours, so the run refuses instead
    of treating a stranger's file as a finished document.
    """

    out = os.path.abspath(str(out))
    results_dir = results_dir or os.path.join(out, "results")
    stems = {_png_stem(row["png_path"]) for row in samples}
    doc_names = {stem + ".json" for stem in stems}
    foreign = []
    if not os.path.isdir(out):
        return {"foreign": foreign, "stems": stems}
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name)
        if os.path.isdir(path):
            if name in RESUME_RUN_DIRS:
                if name == "results":
                    for inner in sorted(os.listdir(path)):
                        if inner in doc_names and os.path.isfile(os.path.join(path, inner)):
                            continue
                        foreign.append("results/" + inner)
                continue
            foreign.append(name + "/")
            continue
        if name in RESUME_BOOKKEEPING or name in doc_names:
            continue
        foreign.append(name)
    if foreign:
        raise _refuse(
            "out_resume_foreign",
            "--resume: %s holds %d entries this run did not write (%s); only the run's own "
            "documents, %s and the %s/ directories may be present"
            % (
                out,
                len(foreign),
                ", ".join(foreign[:6]),
                "/".join(RESUME_BOOKKEEPING),
                "/".join(RESUME_RUN_DIRS),
            ),
        )
    return {"foreign": foreign, "stems": stems}


def _reusable_documents(out, results_dir, samples, *, contract=None):
    """The documents ``--resume`` may skip: on disk, parseable, exact fields, valid."""

    C = contract or _contract()
    reused = []
    for row in samples:
        stem = _png_stem(row["png_path"])
        path = os.path.join(results_dir, stem + ".json")
        if not os.path.isfile(path):
            continue
        try:
            document = C.load_json(path)
        except (OSError, ValueError) as exc:
            raise _refuse(
                "resume_document_unreadable", "--resume: %s: %s" % (path, exc)
            ) from None
        if not isinstance(document, dict) or tuple(sorted(document)) != tuple(
            sorted(DOCUMENT_FIELDS)
        ):
            raise _refuse(
                "resume_document_fields",
                "--resume: %s does not hold exactly the %d contract fields"
                % (path, len(DOCUMENT_FIELDS)),
            )
        try:
            C.validate_prediction_document(document, row["sample_id"])
        except C.GuardError as exc:
            raise _refuse(
                "resume_document_invalid",
                "--resume: %s: %s" % (path, getattr(exc, "detail", exc)),
            ) from None
        reused.append(
            {
                "sample_id": row["sample_id"],
                "png_path": row["png_path"],
                "png_sha256": sha256_file(row["png_path"]),
                "result": os.path.relpath(path, out).replace("\\", "/"),
                "document_sha256": C.sha256_value(document),
                "primitives": len(document["semanticIds"]),
                "reused": True,
            }
        )
    return reused


def _publish_document(out, results_dir, png_path, document, *, contract=None):
    """Validate one document and write it now; returns the export facts.

    The same checks the export stage has always run, moved to the moment the
    emitter returns: each document is durable before the next image is read, so a
    run that dies mid-batch keeps everything already emitted.  ``C.write_json``
    writes atomically and keeps the exact bytes earlier runs produced.
    """

    C = contract or _contract()
    stem = _png_stem(png_path)
    fields = tuple(sorted(document or {}))
    if fields != tuple(sorted(DOCUMENT_FIELDS)):
        raise DeliveryRefused("document_fields", "%s: fields %r" % (stem, fields))
    text = json.dumps(document, ensure_ascii=False)
    if '"score"' in text:
        raise DeliveryRefused("document_score", "%s: a score field reached the document" % stem)
    try:
        C.validate_prediction_document(document, stem)
    except C.GuardError as exc:
        raise DeliveryRefused(
            "document_violation", "%s: %s" % (stem, getattr(exc, "detail", exc))
        ) from None
    violations = document_report(document)
    if violations:
        raise DeliveryRefused("document_violation", "%s: %r" % (stem, violations[:6]))
    target = os.path.join(results_dir, stem + ".json")
    C.write_json(target, document)
    return {
        "result": os.path.relpath(target, out).replace("\\", "/"),
        "document_sha256": C.sha256_value(document),
        "primitives": len(document["semanticIds"]),
    }


class _Progress:
    """A long run's stderr progress: one line per ``PROGRESS_EVERY`` images.

    stdout carries the CLI's final JSON and nothing else, so every line here goes
    to stderr.  An image that took longer than ``SLOW_IMAGE_S`` also gets a line,
    which is what tells an operator the run is alive but slow.  ``done`` starts at
    the reused count so the percentage is of the whole input list.
    """

    def __init__(self, total, *, done=0, clock=time.perf_counter, stream=None):
        self.total = int(total)
        self.done = int(done)
        self.clock = clock
        self.start = clock()
        self.stream = stream

    def tick(self, label="", seconds=None):
        self.done += 1
        elapsed = self.clock() - self.start
        rate = (elapsed / self.done) if self.done else 0.0
        eta = rate * max(self.total - self.done, 0)
        stream = self.stream if self.stream is not None else sys.stderr
        lines = []
        if self.done % PROGRESS_EVERY == 0 or self.done == self.total:
            lines.append(
                "progress %d/%d elapsed=%.1f rate=%.3f eta=%.1f"
                % (self.done, self.total, elapsed, rate, eta)
            )
        if seconds is not None and float(seconds) > SLOW_IMAGE_S:
            lines.append(
                "progress: slow image %s %.1fs (image %d/%d)"
                % (label, float(seconds), self.done, self.total)
            )
        if not lines:
            return
        try:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
        except Exception:  # pragma: no cover - a closed stderr must not kill the run
            pass


def _lane_import_paths():
    for entry in (EVAL_DIR, MODEL_ROOT, SRC_DIR):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _contract():
    _lane_import_paths()
    import instance_eval.contract as C  # noqa: PLC0415 - deliberate lazy import

    return C


def _batch():
    _lane_import_paths()
    import instance_eval.batch as batch  # noqa: PLC0415

    return batch


# --------------------------------------------------------------------------- #
# open-read audit (records every read performed during the run)
# --------------------------------------------------------------------------- #
class OpenAudit:
    """Audit hook that records read opens and classifies them by role.

    Declared roles (checkpoint, pngs, inputs) and interpreter roots
    (``sys.path``) are allowed.  Anything else that looks like a data file is a
    violation; a read of ``.svg`` is always a violation.  This is how the run
    receipt can state that no ground-truth or SVG artefact was opened.
    """

    def __init__(self, roles=None, roots=None):
        self.roles = {os.path.normcase(os.path.abspath(k)): v for k, v in (roles or {}).items()}
        self.roots = [os.path.normcase(os.path.abspath(root)) for root in (roots or []) if root]
        self.reads = {}
        self.violations = []
        self.installed = False
        self._busy = False

    def install(self):
        if self.installed:
            return self
        roots = list(self.roots)
        for entry in sys.path:
            if entry and os.path.isdir(entry):
                roots.append(os.path.normcase(os.path.abspath(entry)))
        self.roots = roots
        sys.addaudithook(self._hook)
        self.installed = True
        return self

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _is_read(mode, flags):
        if isinstance(mode, str):
            return ("w" not in mode) and ("a" not in mode) and ("x" not in mode)
        if isinstance(flags, int):
            return not (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        return True

    def _hook(self, event, args):  # pragma: no cover - exercised through tests
        if event != "open" or self._busy or not args:
            return
        path = args[0]
        if not isinstance(path, (str, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        if not self._is_read(mode, flags):
            return
        try:
            self._busy = True
            self._record(str(path))
        except Exception:
            pass
        finally:
            self._busy = False

    def _record(self, path):
        key = os.path.normcase(os.path.abspath(path))
        if key in self.reads:
            return
        role = self.roles.get(key)
        if role is None:
            lowered = key.replace("\\", "/")
            for root in self.roots:
                if lowered.startswith(root.replace("\\", "/") + "/") or lowered == root:
                    role = "runtime"
                    break
        if role is None:
            suffix = os.path.splitext(key)[1].lower()
            if suffix == ".svg":
                role = "violation:svg"
            elif suffix in DATA_SUFFIXES:
                role = "violation:data"
            else:
                role = "other"
        self.reads[key] = role
        if role.startswith("violation"):
            self.violations.append({"path": path, "role": role})

    def report(self):
        counts = {}
        for role in self.reads.values():
            counts[role] = counts.get(role, 0) + 1
        return {
            "installed": self.installed,
            "reads_total": len(self.reads),
            "roles": dict(sorted(counts.items())),
            "violations": list(self.violations),
        }


# --------------------------------------------------------------------------- #
# timed stage wrappers around the lane's driver
# --------------------------------------------------------------------------- #
class DeliveryRun:
    """Wraps loader / png_loader / predict / emit and records per-image work.

    ``loader_impl`` / ``predict_impl`` / ``png_impl`` let the CPU tests replace the
    model access; the timing, bookkeeping and selection stage stay this CLI's own.
    """

    def __init__(
        self,
        policy,
        clock=time.perf_counter,
        loader_impl=None,
        predict_impl=None,
        png_impl=None,
        on_document=None,
        dump_pool=None,
        head_box="decoded",
    ):
        self.policy = policy
        self.clock = clock
        self.loader_impl = loader_impl
        self.predict_impl = predict_impl
        self.png_impl = png_impl
        # Called with each finished per-image record, i.e. from inside the
        # batch loop: the document is written there, not after the last image.
        self.on_document = on_document
        # Debug only: directory to dump the refined candidate pool into
        # (post a6 refinement, pre blend selector). None = disabled.
        self.dump_pool = dump_pool
        # B1 experiment: which box the stroke head window uses (mask | decoded).
        self.head_box = head_box
        # Debug only: decoded box-head boxes aligned with the pool (set in emit).
        self._pool_decoded_boxes = None
        # Debug/experiment: decoded boxes aligned with the blend-SELECTED masks
        # (set in emit after the blend stage; used by the stroke-head window).
        self._selected_decoded_boxes = None
        self.load_s = None
        self.decode = None
        self.mask_head = None
        self.images = []
        self._open = None
        self._last_png = None
        self._loader_ink = None
        self._last_boxes = None
        self.refine_totals = None
        self._head_model = None

    # -- loader ------------------------------------------------------------ #
    def _record_context(self, context, load_s):
        self.load_s = load_s
        model = context.get("model")
        heads = getattr(model, "roi_heads", None)
        reported = context.get("mask_head")
        self.mask_head = dict(reported) if reported else {"output_size": 28, "chunked": False}
        self.decode = {
            "device": str(context.get("device")),
            "max_detections": int(context.get("max_detections") or 0),
            "roi_score_thresh": _as_float(getattr(heads, "score_thresh", None)),
            "roi_nms_thresh": _as_float(getattr(heads, "nms_thresh", None)),
            "roi_detections_per_img": _as_float(getattr(heads, "detections_per_img", None)),
            "config": _jsonable(context.get("config")),
        }
        return self.decode

    def loader(self, checkpoint, device):
        underlying = self.loader_impl or _batch().default_loader
        start = self.clock()
        context = dict(underlying(checkpoint, device))
        self._record_context(context, self.clock() - start)
        return context

    # -- png --------------------------------------------------------------- #
    def png_loader(self, path):
        underlying = self.png_impl or _batch().default_png_loader
        start = self.clock()
        image, ink, meta = underlying(path)
        read_s = self.clock() - start
        self._last_png = path
        self._loader_ink = ink
        self._open = {
            "png_path": str(path),
            "read_s": read_s,
            "png_meta": dict(meta or {}),
            "ink_pixels": int(ink.sum()) if hasattr(ink, "sum") else None,
        }
        return image, ink, meta

    # -- predict ----------------------------------------------------------- #
    def predict(self, context, image):
        underlying = self.predict_impl or _batch().default_predict
        start = self.clock()
        result = underlying(context, image)
        predict_s = self.clock() - start
        # The box-selection rule (--select-rule nmsbox) scores the *decoded
        # boxes*, so they are carried from here into emit() as-is.  Nothing else
        # reads them: the plain rule ignores them completely.
        self._last_boxes = (result or {}).get("boxes") if isinstance(result, dict) else None
        if self._open is not None:
            self._open["predict_s"] = predict_s
        return result

    # -- emit (the selection stage) ---------------------------------------- #
    def emit(self, masks, class_ids, scores, *, height, width, dense=False):
        batch = _batch()
        start = self.clock()

        # Box rule (--select-rule nmsbox): the decoded xyxy boxes are aligned with
        # the mask list here, i.e. before the Stuff drop, and filtered with it so
        # index i means the same instance in both lists.  ``None`` on the plain
        # rule, which never looks at boxes.
        box_block = self.policy.get("box_nms")
        boxes = None
        if box_block:
            raw_boxes = self._last_boxes
            if raw_boxes is None:
                raise _refuse(
                    "box_nms_boxes",
                    "--select-rule nmsbox needs the decoded boxes, but the predictor returned none",
                )
            boxes = [[float(value) for value in list(row)[:4]] for row in list(raw_boxes)][: len(masks)]
            if len(boxes) != len(masks):
                raise _refuse(
                    "box_nms_boxes",
                    "box/mask count mismatch: %d boxes, %d masks" % (len(boxes), len(masks)),
                )

        # Stuff drop: on the decoded instances, before the delivery selection, so a
        # Stuff detection never consumes a --per-image-cap slot that a Thing could
        # use.  The emitter paints a Stuff class as one merged instanceId == -1
        # group, so this filter is exactly "drop the stuff tail" -- it removes no
        # Thing mask and, being class-only, cannot change any surviving mask.
        stuff_record = None
        stuff_block = self.policy.get("drop_stuff")
        if stuff_block:
            stuff_ids = [i for i, value in enumerate(class_ids) if int(value) in STUFF_CLASSES]
            dropped_by_class = {}
            for i in stuff_ids:
                key = str(int(class_ids[i]))
                dropped_by_class[key] = dropped_by_class.get(key, 0) + 1
            if stuff_ids:
                gone = set(stuff_ids)
                masks = [m for i, m in enumerate(masks) if i not in gone]
                class_ids = [c for i, c in enumerate(class_ids) if i not in gone]
                scores = [s for i, s in enumerate(scores) if i not in gone]
                if boxes is not None:
                    boxes = [b for i, b in enumerate(boxes) if i not in gone]
            stuff_record = {
                "classes": list(STUFF_CLASSES),
                "decoded_in": len(class_ids) + len(stuff_ids),
                "dropped": len(stuff_ids),
                "dropped_by_class": dropped_by_class,
                "decoded_kept": len(class_ids),
            }

        # Decoded box-head boxes aligned with the (post-stuff-drop) mask list:
        # ``self._last_boxes`` is one row per decoded instance (pre-stuff-drop),
        # so apply the identical stuff filter.  Used only by the debug pool dump;
        # the shipped selection never reads these on the blend rule.
        decoded_boxes = None
        if self._last_boxes is not None:
            last = list(self._last_boxes)
            if stuff_block and stuff_ids:
                gone = set(stuff_ids)
                decoded_boxes = [b for i, b in enumerate(last) if i not in gone]
            else:
                decoded_boxes = last

        selected = select_instances(
            masks,
            class_ids,
            scores,
            score_threshold=self.policy["score_threshold"],
            nms_iou=self.policy["nms_iou"],
            max_detections=self.policy["max_detections"],
            per_image_cap=self.policy["per_image_cap"],
            boxes=boxes,
            select_rule=(box_block or {}).get("rule"),
            nms_box_iou=(box_block or {}).get("box_iou"),
            nms_box_thr=(box_block or {}).get("box_threshold"),
        )
        select_s = self.clock() - start
        keep = selected["indices"]
        kept_masks = [masks[index] for index in keep]
        kept_classes = [int(class_ids[index]) for index in keep]
        kept_scores = [float(scores[index]) for index in keep]
        self._pool_decoded_boxes = (
            [decoded_boxes[index] for index in keep] if decoded_boxes is not None else None
        )

        if stuff_record is not None:
            stuff_record["selected_in"] = len(keep)
            stuff_record["kept"] = len(kept_classes)

        # Ink refinement, on the native raster: the square-canvas pad for a
        # non-square record happens afterwards, so both the mask and the PNG ink
        # are still at the image's own resolution here.  That ordering is
        # measured; do not move the pad before refinement.
        refine_start = self.clock()
        refine_record = self._refine_stage(kept_masks, int(width), int(height))
        kept_masks = refine_record.pop("masks")
        refine_s = self.clock() - refine_start
        # Debug hook: dump the refined pool (post a6, pre blend selector).
        self._dump_pool(kept_masks, kept_classes, kept_scores, int(width), int(height))

        # Ink-in-box masks for instances larger than the 112^2 head can resolve
        # (--mask-ink-min-side, off when the policy has no block).  ``pre`` runs
        # here, before the blend selector, so the selector's boxes and 26 features
        # are read off ink boxes; ``post`` (the default and the measured placement)
        # runs after selection, so nothing upstream of the emitter -- features,
        # selection, counts, instance ids -- ever sees a replaced mask.
        ink_box_block = self.policy.get("ink_box") or {}
        ink_box_post = bool(ink_box_block) and ink_box_block.get("stage_placement") == "post"
        ink_box_record = None
        ink_box_s = 0.0
        if ink_box_block and not ink_box_post:
            ink_box_start = self.clock()
            ink_box_out = self._ink_box_stage(kept_masks, int(width), int(height))
            kept_masks = ink_box_out.pop("masks")
            ink_box_record = ink_box_out.pop("record")
            ink_box_s = self.clock() - ink_box_start

        # Blend readout: the frozen 26-feature selector scores the *refined*
        # candidates (its boxes, fill and ink overlap are measured on the refined
        # mask), so it sits here -- after ``_refine_stage`` and before the
        # square-canvas pad, which would otherwise move the boxes.  The pool
        # selection above still ran, unchanged, to produce these candidates: its
        # counts are kept as ``blend.pool_counts`` and the blend rule decides what
        # is emitted.
        blend_block = self.policy.get("blend")
        blend_record = None
        blend_s = 0.0
        pool_selected = len(keep)
        if blend_block:
            blend_start = self.clock()
            outcome = self._blend_stage(kept_masks, kept_classes, kept_scores, int(width), int(height))
            chosen = list(outcome["indices"])
            probs = list(outcome["probabilities"])
            kept_masks = [kept_masks[index] for index in chosen]
            kept_classes = [int(kept_classes[index]) for index in chosen]
            kept_scores = [float(probs[index]) for index in chosen]
            blend_record = outcome["record"]
            blend_record["pool_counts"] = dict(selected["counts"])
            blend_record["pool_selected"] = pool_selected
            blend_s = self.clock() - blend_start
            # Decoded boxes for the blend-SELECTED instances (B1 experiment: the
            # stroke head uses these as its window instead of the mask bbox).
            self._selected_decoded_boxes = (
                [self._pool_decoded_boxes[index] for index in chosen]
                if self._pool_decoded_boxes is not None
                else None
            )
        else:
            self._selected_decoded_boxes = self._pool_decoded_boxes

        if ink_box_block and ink_box_post:
            # After selection: every mask here is one the emitter will draw, so the
            # replacement changes exactly the delivered document and nothing else.
            ink_box_start = self.clock()
            ink_box_out = self._ink_box_stage(kept_masks, int(width), int(height))
            kept_masks = ink_box_out.pop("masks")
            ink_box_record = ink_box_out.pop("record")
            ink_box_s = self.clock() - ink_box_start

        # Stroke-mask head (--mask-source head): the same post-selection placement
        # as the ink rule above, so features, selection, counts and instance ids are
        # exactly what the shipped settings produce; only the drawn mask changes.
        mask_source_block = self.policy.get("mask_source") or {}
        mask_source_record = None
        mask_source_s = 0.0
        if mask_source_block.get("source") == "head":
            mask_source_start = self.clock()
            head_out = self._mask_head_stage(kept_masks, int(width), int(height))
            kept_masks = head_out.pop("masks")
            mask_source_record = head_out.pop("record")
            mask_source_s = self.clock() - mask_source_start

        pad_record = canvas_padding_record(width, height)
        kept_masks, side, padded = square_canvas_masks(kept_masks, int(width), int(height))
        emit_height = int(side) if padded else int(height)
        emit_width = int(side) if padded else int(width)
        emit_start = self.clock()
        emitted = batch.default_emit(
            kept_masks,
            kept_classes,
            kept_scores,
            height=emit_height,
            width=emit_width,
            dense=bool(dense),
        )
        emit_s = self.clock() - emit_start

        record = dict(self._open or {})
        counts = selected["counts"]
        lowest = selected["lowest_selected_score"]
        if blend_record is not None:
            # The delivered readout is the blend one: its counts and its lowest
            # surviving probability are what the reports and the exports carry.
            counts = blend_record["counts"]
            lowest = blend_record["lowest_surviving_score"]
        record.update(
            {
                "width": int(width),
                "height": int(height),
                "decoded": len(masks),
                "counts": counts,
                "lowest_surviving_score": lowest,
                "select_s": select_s,
                "blend_s": blend_s,
                "refine_s": refine_s,
                "refine": refine_record,
                "emit_s": emit_s,
                "emitted_instances": int((emitted.get("diagnostics") or {}).get("emitted_instances") or 0),
                "document": emitted.get("document"),
                "diagnostics": dict(emitted.get("diagnostics") or {}),
                "emit_canvas": {
                    "width": emit_width,
                    "height": emit_height,
                    "padded_to_square": bool(padded),
                    "crop_residual_px": pad_record["crop_residual_px"],
                    "canvas_padding": dict(pad_record["canvas_padding"]),
                    "unusual_aspect": bool(padded and pad_record["unusual_aspect"]),
                    "note": pad_record["note"],
                },
            }
        )
        record.setdefault("predict_s", None)
        if stuff_record is not None:
            record["stuff_drop"] = stuff_record
        if box_block is not None:
            record["box_nms"] = {
                "rule": box_block["rule"],
                "box_iou": box_block["box_iou"],
                "box_threshold": box_block["box_threshold"],
                "boxes_supplied": boxes is not None,
            }
        record["total_s"] = (
            record.get("read_s", 0.0) + (record.get("predict_s") or 0.0) + select_s + refine_s + emit_s
        )
        if blend_record is not None:
            record["blend"] = blend_record
            record["total_s"] = record["total_s"] + blend_s
        if ink_box_record is not None:
            # Provenance for the non-default knob only: with the rule off the
            # record keeps exactly the keys it has always had.
            record["ink_box"] = ink_box_record
            record["ink_box_s"] = ink_box_s
            record["total_s"] = record["total_s"] + ink_box_s
        if mask_source_record is not None:
            record["mask_source"] = mask_source_record
            record["mask_source_s"] = mask_source_s
            record["total_s"] = record["total_s"] + mask_source_s
        self.images.append(record)
        self._open = None
        self._last_boxes = None
        if self.on_document is not None:
            # The per-image document is validated and written here, before the
            # next PNG is read, so an interrupted run keeps every finished one.
            self.on_document(record)
        return emitted

    # -- mask refinement (refine.py) --------------------------------------- #
    def _refine_stage(self, masks, width, height):
        """Refine the surviving masks; returns a per-image record with the masks.

        The record is JSON-ready except for ``masks``, which the caller pops.
        """

        refine_policy = self.policy.get("refine") or {}
        rule = refine_policy.get("rule") or REFINE_RULES[0]
        coverage = refine_policy.get("component_coverage")
        if coverage is None:
            coverage = DEFAULT_REFINE_RATIO
        record = {
            "rule": rule,
            "ink": refine_policy.get("ink") or DEFAULT_REFINE_INK,
            "component_coverage": coverage,
            "applied_in": "after selection, before the square-canvas pad and the emitter",
            "native": [int(width), int(height)],
            "instances_before": len(masks),
            "instances_nonempty_before": 0,
            "instances_nonempty_after": 0,
        }
        if rule == REFINE_RULES[0]:
            nonempty = _count_nonempty(masks)
            record["source"] = None
            record["stats"] = _empty_refine_stats()
            record["instances_nonempty_before"] = nonempty
            record["instances_nonempty_after"] = nonempty
            record["masks"] = list(masks)
            return record

        module = _refine()
        ink, ink_info = self._ink_map(module, int(width), int(height))
        record.update(ink_info)
        stats = module.new_stats()
        refined = module.refine_masks(masks, ink, rule=rule, stats=stats, coverage=coverage)
        if len(refined) != len(masks):  # pragma: no cover - refine_masks is 1:1
            raise _refuse(
                "refine_policy",
                "refinement changed the instance count: %d -> %d" % (len(masks), len(refined)),
            )
        before = _count_nonempty(masks)
        after = _count_nonempty(refined)
        if after != before:  # pragma: no cover - refine.py guarantees at least the original mask
            raise _refuse(
                "refine_policy",
                "refinement emptied %d instance(s) (%d -> %d non-empty); the emitted instance "
                "count must not move" % (before - after, before, after),
            )
        stats["dropped_instances"] = 0
        record["instances_nonempty_before"] = before
        record["instances_nonempty_after"] = after
        record["stats"] = stats
        record["masks"] = refined
        if self.refine_totals is None:
            self.refine_totals = module.new_stats()
            self.refine_totals["images"] = 0
            self.refine_totals["ink_pixels"] = 0
        self.refine_totals["images"] = int(self.refine_totals.get("images", 0)) + 1
        self.refine_totals["ink_pixels"] = int(self.refine_totals.get("ink_pixels", 0)) + int(
            record.get("png_pixels") or 0
        )
        for key, value in stats.items():
            self.refine_totals[key] = int(self.refine_totals.get(key, 0)) + int(value)
        return record

    def _dump_pool(self, masks, class_ids, scores, width, height):
        """Debug only: write the refined candidate pool to ``--dump-pool``.

        One ``<stem>.npz`` per image: the a6-refined masks (sparse COO), their
        boxes (``bbox_from_mask``), detection scores and predicted classes, as
        the blend selector sees them.  Inert when ``dump_pool`` is unset; this
        is a self-check hook, never part of the delivered documents.
        """

        if not self.dump_pool:
            return
        import numpy as np  # noqa: PLC0415

        features = _features26()
        boxes: list = []
        decoded: list = []
        keep_scores: list = []
        keep_classes: list = []
        ys: list = []
        xs: list = []
        offsets = [0]
        decoded_all = self._pool_decoded_boxes
        for idx, (mask, label, score) in enumerate(zip(masks, class_ids, scores)):
            array = np.asarray(mask, dtype=bool)
            if not array.any():
                continue
            box = features.bbox_from_mask(array)
            if box is None:  # pragma: no cover - non-empty mask always has a box
                continue
            boxes.append(box)
            if decoded_all is not None and idx < len(decoded_all):
                decoded.append([float(v) for v in list(decoded_all[idx])[:4]])
            keep_scores.append(float(score))
            keep_classes.append(int(label))
            yy, xx = np.nonzero(array)
            ys.append(yy.astype(np.int32))
            xs.append(xx.astype(np.int32))
            offsets.append(offsets[-1] + int(yy.size))
        stem = _png_stem(self._last_png) if self._last_png else "unknown"
        os.makedirs(self.dump_pool, exist_ok=True)
        path = os.path.join(self.dump_pool, stem + ".npz")
        save_kwargs = dict(
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            scores=np.asarray(keep_scores, dtype=np.float32),
            classes=np.asarray(keep_classes, dtype=np.int32),
            mask_ys=np.concatenate(ys) if ys else np.zeros(0, dtype=np.int32),
            mask_xs=np.concatenate(xs) if xs else np.zeros(0, dtype=np.int32),
            mask_off=np.asarray(offsets, dtype=np.int64),
            width=int(width),
            height=int(height),
        )
        if decoded:
            save_kwargs["decoded_boxes"] = np.asarray(decoded, dtype=np.float32).reshape(-1, 4)
        np.savez_compressed(path, **save_kwargs)

    def _ink_map(self, module, width, height):
        """The ink map to refine against, from the PNG itself when it is readable."""

        import numpy as np  # noqa: PLC0415

        ink_rule = (self.policy.get("refine") or {}).get("ink") or DEFAULT_REFINE_INK
        info = {"ink_rule": ink_rule, "ink_definition": module.INK_DEFINITIONS[ink_rule], "source": None}
        path = self._last_png
        note = None
        if path is not None and os.path.isfile(str(path)):
            try:
                ink, (png_w, png_h) = module.ink_from_png(path, ink_rule)
            except Exception as exc:  # noqa: BLE001 - a stub PNG in the tests is not readable
                note = "declared PNG unreadable (%s: %s)" % (type(exc).__name__, exc)
            else:
                if (int(png_w), int(png_h)) != (int(width), int(height)):
                    raise _refuse(
                        "refine_policy",
                        "the PNG raster %dx%d does not match the emitter canvas %dx%d; refusing to "
                        "refine against a different raster" % (png_w, png_h, width, height),
                    )
                info["source"] = "png"
                info["png_path"] = str(path)
                info["png_pixels"] = int(np.count_nonzero(ink))
                if note is not None:  # pragma: no cover
                    info["note"] = note
                return ink, info

        if self._loader_ink is None:
            raise _refuse(
                "refine_policy",
                "no ink map available for refinement: the PNG could not be opened (%s) and the "
                "loader returned none" % (note or "not attempted"),
            )
        fallback = np.asarray(self._loader_ink, dtype=bool)
        if fallback.shape != (int(height), int(width)):
            raise _refuse(
                "refine_policy",
                "ink shape %r does not match the emitter canvas %dx%d"
                % (fallback.shape, width, height),
            )
        info["source"] = "loader"
        info["png_pixels"] = int(np.count_nonzero(fallback))
        info["note"] = (note or "the declared PNG was not opened") + (
            "; using the loader's ink map, which follows model/infer.py (alpha over white, "
            "min(R,G,B) < 254) rather than the named ink rule"
        )
        return fallback, info

    def _ink_box_stage(self, masks, width, height):
        """Replace the mask of every instance whose box side exceeds the threshold.

        The rule itself is ``ink_box_masks`` (module level, unit-tested): the
        mask becomes the PNG ink inside the refined box widened by
        ``INK_BOX_PAD`` px, clamped to the raster, and an empty crop keeps the
        refined mask.  The ink map is the one the refinement used
        (``--refine-ink``), read from the same PNG at the same native raster.

        Returns ``{"masks": [...], "record": {...}}``; the caller pops ``masks``.
        """

        block = self.policy.get("ink_box") or {}
        min_side = int(block.get("min_side") or 0)
        placement = block.get("stage_placement") or DEFAULT_INK_STAGE
        ink, ink_info = self._ink_map(_refine(), int(width), int(height))
        masks, record = ink_box_masks(masks, ink, min_side, int(width), int(height))
        record["applied_in"] = (
            "after selection and before the square-canvas pad (after_selection)"
            if placement == "post"
            else "after the a6 refinement, before the blend features and the pad "
            "(after_refine_before_blend)"
        )
        record["stage_placement"] = placement
        record["ink"] = ink_info
        return {"masks": masks, "record": record}

    def _continuous_ink(self, width, height):
        """The continuous 0..255 ink map the stroke head gates at ``--mask-head-gate``.

        Read from the same PNG and at the same native raster the refinement used;
        when the PNG is not readable the loader's boolean ink is used as 0/255,
        which makes ``ink >= gate`` the composite128 test the refinement applied,
        so the gate can never admit a pixel the refinement did not see.
        """

        import numpy as np  # noqa: PLC0415

        module = _mask_source()
        info = {"source": None}
        path = self._last_png
        note = None
        if path is not None and os.path.isfile(str(path)):
            try:
                ink = module.continuous_ink_from_png(path)
            except Exception as exc:  # noqa: BLE001 - a stub PNG in the tests is not readable
                note = "declared PNG unreadable (%s: %s)" % (type(exc).__name__, exc)
            else:
                if tuple(int(v) for v in ink.shape) != (int(height), int(width)):
                    raise _refuse(
                        "selection_policy",
                        "the PNG raster %dx%d does not match the emitter canvas %dx%d; refusing "
                        "to run the stroke head on a different raster"
                        % (ink.shape[1], ink.shape[0], width, height),
                    )
                info["source"] = "png"
                info["png_path"] = str(path)
                info["continuous"] = True
                if note is not None:  # pragma: no cover
                    info["note"] = note
                return ink, info

        boolean, ink_info = self._ink_map(_refine(), int(width), int(height))
        ink = np.where(np.asarray(boolean, dtype=bool), np.uint8(255), np.uint8(0))
        info["source"] = ink_info.get("source")
        info["ink_rule"] = ink_info.get("ink_rule")
        info["continuous"] = False
        info["note"] = (note or "the declared PNG was not opened") + (
            "; gating the loader's boolean ink as 0/255, i.e. the gate test is the same "
            "composite128 test the refinement applied"
        )
        return ink, info

    def _head(self):
        """``(model, meta)`` of the stroke head, loaded once per run."""

        if self._head_model is not None:
            return self._head_model

        import torch  # noqa: PLC0415

        block = self.policy.get("mask_source") or {}
        module = _mask_source()
        device = str((self.decode or {}).get("device") or "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():  # pragma: no cover
            device = "cpu"
        try:
            model, meta = module.load_head(block.get("checkpoint"), device=device)
        except FileNotFoundError as exc:
            raise _refuse("selection_policy", "--mask-head-ckpt: %s" % (exc,))
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint is a refusal, not a traceback
            raise _refuse(
                "selection_policy",
                "--mask-head-ckpt: %s could not be loaded (%s: %s)"
                % (block.get("checkpoint"), type(exc).__name__, exc),
            )
        model.eval()
        self._head_model = (model, meta)
        return self._head_model

    def _mask_head_stage(self, masks, width, height):
        """Replace the mask of every selected instance above the head's threshold.

        The rule is ``delivery/mask_source.head_masks``: the 256^2 stroke head runs
        on the ink of each selected instance whose box side exceeds
        ``--mask-head-min-side`` and the mask becomes ``ink >= gate`` and
        ``prob >= tau`` pasted back into the head's own window; an empty prediction
        keeps the refined mask.  Placement is the post-selection one: features,
        selection, counts and instance ids are untouched.

        Returns ``{"masks": [...], "record": {...}}``; the caller pops ``masks``.
        """

        import numpy as np  # noqa: PLC0415

        module = _mask_source()
        block = self.policy.get("mask_source") or {}
        min_side = int(block.get("min_side") or 0)
        tau = float(block.get("tau") if block.get("tau") is not None else module.HEAD_TAU)
        gate = int(block.get("gate") or module.HEAD_GATE)
        ink, ink_info = self._continuous_ink(int(width), int(height))
        counts = {key: 0 for key in module.HEAD_STAT_KEYS}
        counts["instances"] = len(masks)
        boxes, positions = [], []
        decoded_sel = self._selected_decoded_boxes if self.head_box == "decoded" else None
        for position, mask in enumerate(masks):
            array = np.asarray(mask, dtype=bool)
            if not array.any():
                counts["empty_masks"] += 1
                continue
            box = None
            if decoded_sel is not None and position < len(decoded_sel):
                row = decoded_sel[position]
                if row is not None:
                    box = [float(v) for v in list(row)[:4]]
            if box is None:
                box = _mask_box(array)
            if box is None:  # pragma: no cover - a non-empty mask always has a box
                counts["empty_masks"] += 1
                continue
            if max(box[2] - box[0], box[3] - box[1]) <= min_side:
                counts["below_threshold"] += 1
                continue
            boxes.append(box)
            positions.append(int(position))

        out = list(masks)
        loaded = None
        if boxes:
            model, meta = self._head()
            produced = module.head_masks(
                ink,
                boxes,
                model,
                size=int(meta.get("size") or module.HEAD_SIZE),
                tau=tau,
                gate=gate,
                device=str(meta.get("device") or "cpu"),
                pad=int(module.HEAD_PAD),
                min_side=int(module.HEAD_MIN_SIDE),
            )
            loaded = {
                "parameters": int(meta.get("parameters") or 0),
                "device": str(meta.get("device")),
                "size": int(meta.get("size") or module.HEAD_SIZE),
                "epoch": meta.get("epoch"),
            }
            for position, head in zip(positions, produced):
                head = np.asarray(head, dtype=bool)
                if head.shape != (int(height), int(width)):  # pragma: no cover - guarded above
                    raise _refuse(
                        "selection_policy",
                        "stroke-head mask %r does not match the emitter canvas %dx%d"
                        % (head.shape, width, height),
                    )
                if not head.any():
                    counts["empty_head_kept_original"] += 1
                    continue
                out[position] = head
                counts["replaced"] += 1

        record = {
            "applied": True,
            "source": "head",
            "min_side": min_side,
            "tau": tau,
            "gate": gate,
            "checkpoint": block.get("checkpoint"),
            "checkpoint_origin": block.get("checkpoint_origin"),
            "size": int(block.get("size") or module.HEAD_SIZE),
            "width": int(block.get("width") or module.HEAD_WIDTH),
            "parameters": int(block.get("parameters") or module.HEAD_PARAMETERS),
            "boxes": len(boxes),
            "loaded": loaded,
            "ink": ink_info,
            "counts": counts,
            "applied_in": "after selection and before the square-canvas pad (after_selection)",
            "note": "only the selected instances whose box side exceeds min_side are replaced; "
            "an empty prediction keeps the refined mask, so every emitted mask stays non-empty",
        }
        return {"masks": out, "record": record}

    def _blend_stage(self, masks, class_ids, scores, width, height):
        """Score the refined candidates with the frozen blend and select.

        This is the ``--select-rule blend`` readout.  It reproduces the measured
        arm exactly: the 26 columns of ``delivery/features26.py`` computed on the
        *refined* mask (``bbox_from_mask`` box, ``ink_box_pad2`` ink source), one
        probability per candidate from ``delivery/models.py: Blend``, then
        ``nmsbox.select_by_box_nms`` with the blend probabilities in place of the
        detection scores (``--blend-thr`` / ``--blend-tau`` / ``--blend-cap``).
        Candidates whose refined mask is empty carry no box and are dropped before
        the features are built, exactly as ``tools/blend_eval.py`` did when it
        produced the arm.

        Returns the chosen mask positions, a probability per input mask (0.0 for a
        dropped candidate, which is never selected) and a JSON-ready record.  The
        caller applies ``indices``; nothing is reordered here.
        """

        import numpy as np  # noqa: PLC0415

        block = self.policy["blend"]
        selector = _nmsbox()
        features = _features26()
        model = _blend_model(block["model"])
        ink, ink_info = self._ink_map(_refine(), int(width), int(height))
        ink = np.asarray(ink, dtype=bool)

        decoded_all = self._pool_decoded_boxes
        positions, boxes, cand_scores, cand_labels, cand_masks, cand_dboxes = [], [], [], [], [], []
        skipped_empty = 0
        for position, (mask, label, score) in enumerate(zip(masks, class_ids, scores)):
            array = np.asarray(mask, dtype=bool)
            if not array.any():
                skipped_empty += 1
                continue
            box = features.bbox_from_mask(array)
            if box is None:  # pragma: no cover - a non-empty mask always has a box
                skipped_empty += 1
                continue
            positions.append(int(position))
            boxes.append(box)
            cand_scores.append(float(score))
            cand_labels.append(int(label))
            cand_masks.append(array)
            if decoded_all is not None and position < len(decoded_all):
                cand_dboxes.append([float(v) for v in list(decoded_all[position])[:4]])
            else:
                cand_dboxes.append([float(v) for v in box])

        probabilities = [0.0] * len(masks)
        indices = []
        counts = {
            "candidates": len(boxes),
            "after_nms": 0,
            "after_threshold": 0,
            "after_cap": 0,
            "after_per_image_cap": 0,
            "selected": 0,
            "nms_suppressed": 0,
            "threshold_dropped": len(boxes),
            "cap_dropped": 0,
            "per_image_cap_dropped": 0,
        }
        lowest = None
        if boxes:
            table = features.compute_features(
                boxes, cand_scores, cand_labels, cand_masks, ink, int(width), int(height),
                decoded_boxes=cand_dboxes,
            )
            scored = np.asarray(
                model.predict_proba(np.asarray(table, dtype=np.float64))[:, 1], dtype=np.float64
            )
            values = [float(value) for value in scored.reshape(-1)]
            for index, value in enumerate(values):
                probabilities[positions[index]] = value
            chosen = selector.select_by_box_nms(
                boxes,
                cand_labels,
                values,
                score_threshold=float(block["threshold"]),
                box_iou=float(block["box_iou"]),
                max_detections=int(block["max_detections"]),
                per_image_cap=int(block["per_image_cap"]),
            )
            indices = [positions[index] for index in chosen["indices"]]
            counts = dict(chosen["counts"])
            lowest = chosen["lowest_selected_score"]

        record = {
            "rule": BLEND_RULE,
            "model": block["model"],
            "threshold": block["threshold"],
            "box_iou": block["box_iou"],
            "per_image_cap": block["per_image_cap"],
            "features": {
                "source": "delivery/features26.py compute_features",
                "columns": len(features.FEATURE_NAMES),
                "mask_source": "ink_box_pad2",
                "candidates_with_box": len(boxes),
                "skipped_empty_masks": skipped_empty,
            },
            "counts": counts,
            "lowest_surviving_score": lowest,
            "ink": ink_info,
        }
        return {"indices": indices, "probabilities": probabilities, "record": record}


REFINE_STAT_KEYS = (
    "instances",
    "changed",
    "components",
    "kept_components",
    "instances_without_component",
    "a6_to_a1",
    "a1_to_original",
    "a1_rule_empty",
    "dropped_instances",
)


def _empty_refine_stats():
    return {key: 0 for key in REFINE_STAT_KEYS}


def _count_nonempty(masks):
    import numpy as np  # noqa: PLC0415

    return sum(1 for mask in masks if bool(np.any(mask)))


INK_BOX_STAT_KEYS = (
    "instances",
    "replaced",
    "below_threshold",
    "empty_ink_kept_original",
    "empty_masks",
)


def _mask_box(array):
    """``[x0, y0, x1, y1]`` (exclusive upper bound) of a 2-D boolean array."""

    import numpy as np  # noqa: PLC0415

    rows = np.flatnonzero(array.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(array.any(axis=0))
    return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]


def ink_box_masks(masks, ink, min_side, width, height, pad=INK_BOX_PAD):
    """Replace the mask of every candidate whose box side exceeds ``min_side``.

    For such a candidate the returned mask is ``ink`` restricted to the box
    widened by ``pad`` px on every side and clamped to the raster; the
    candidate's own mask is ignored.  A candidate whose box side is at or below
    ``min_side`` keeps its mask untouched, and so does one whose ink crop is
    empty (keeping a non-empty mask non-empty is what the emitter and the
    contract rely on).  ``bbox_from_mask`` of the *input* mask is the box, the
    pool's ``[x_min, y_min, x_max + 1, y_max + 1]`` convention, so the crop
    reaches ``pad`` pixels past the last mask row.

    Returns ``(masks, record)``: the new masks (1:1, same order) and a
    JSON-ready count record.  ``min_side <= 0`` returns the input masks and a
    record with ``applied: False``, so the caller can keep the stage off.
    """

    import numpy as np  # noqa: PLC0415

    ink_array = np.asarray(ink, dtype=bool)
    if ink_array.ndim != 2 or ink_array.shape != (int(height), int(width)):
        raise _refuse(
            "refine_policy",
            "ink raster %r does not match the declared %dx%d; refusing to build ink-box masks "
            "on a different raster" % (ink_array.shape, width, height),
        )
    counts = {key: 0 for key in INK_BOX_STAT_KEYS}
    counts["instances"] = len(masks)
    record = {
        "applied": bool(int(min_side) > 0),
        "min_side": int(min_side),
        "crop_pad": int(pad),
        "native": [int(width), int(height)],
        "counts": counts,
    }
    if int(min_side) <= 0:
        return list(masks), record

    out = []
    for mask in masks:
        array = np.asarray(mask, dtype=bool)
        box = _mask_box(array)
        if box is None:
            counts["empty_masks"] += 1
            out.append(array)
            continue
        x0, y0, x1, y1 = box
        if max(x1 - x0, y1 - y0) <= int(min_side):
            counts["below_threshold"] += 1
            out.append(array)
            continue
        top = max(0, y0 - int(pad))
        left = max(0, x0 - int(pad))
        bottom = min(int(height), y1 + int(pad))
        right = min(int(width), x1 + int(pad))
        crop = ink_array[top:bottom, left:right]
        if not crop.any():
            counts["empty_ink_kept_original"] += 1
            out.append(array)
            continue
        replacement = np.zeros((int(height), int(width)), dtype=bool)
        replacement[top:bottom, left:right] = crop
        counts["replaced"] += 1
        out.append(replacement)
    return out, record


def refine_summary(policy, exports):
    """The receipt / selection-report refinement block, from the per-image records.

    Every branch of the rule is counted so a reviewer can see how often it fired:
    ``a6_no_kept_component`` (nothing past the 50% coverage test) -> ``a6_to_a1``
    (mask & ink used instead) -> ``a1_to_original`` (the original mask, the last
    and only never-empty branch).  ``dropped_instances`` must stay 0.
    """

    refine_policy = policy.get("refine") or {}
    per_image = [dict(item.get("refine") or {}) for item in exports]
    stats_total = {key: 0 for key in REFINE_STAT_KEYS}
    for entry in per_image:
        for key in REFINE_STAT_KEYS:
            stats_total[key] += int((entry.get("stats") or {}).get(key, 0))
    sources = {}
    for entry in per_image:
        name = entry.get("source") or "none"
        sources[name] = sources.get(name, 0) + 1
    observed = sorted(
        {float(entry["component_coverage"]) for entry in per_image if entry.get("component_coverage") is not None}
    )
    policy_coverage = refine_policy.get("component_coverage")
    if policy_coverage is None:  # 0.0 is a real ratio, never "missing"
        policy_coverage = DEFAULT_REFINE_RATIO
    return {
        "rule": refine_policy.get("rule"),
        "ink": refine_policy.get("ink"),
        "ink_definition": refine_policy.get("ink_definition"),
        "component_coverage": refine_policy.get("component_coverage"),
        "component_coverage_default": refine_policy.get("component_coverage_default"),
        "component_coverage_range": refine_policy.get("component_coverage_range"),
        "component_coverage_rule": refine_policy.get("component_coverage_rule"),
        "component_coverage_observed": observed,
        "component_coverage_consistent": observed in ([], [float(policy_coverage)]),
        "component_connectivity": refine_policy.get("component_connectivity"),
        "applied_in": refine_policy.get("applied_in"),
        "stage_order": refine_policy.get("stage_order"),
        "fallbacks": refine_policy.get("fallbacks"),
        "never_drops_an_instance": True,
        "images": len(per_image),
        "images_refined": sum(1 for entry in per_image if entry.get("source")),
        "ink_sources": sources,
        "instances_selected": sum(int(entry.get("instances_before") or 0) for entry in per_image),
        "instances_refined": stats_total["instances"],
        "instances_changed": stats_total["changed"],
        "instances_unchanged": stats_total["instances"] - stats_total["changed"],
        "instances_nonempty_before": sum(
            int(entry.get("instances_nonempty_before") or 0) for entry in per_image
        ),
        "instances_nonempty_after": sum(
            int(entry.get("instances_nonempty_after") or 0) for entry in per_image
        ),
        "ink_pixels": sum(int(entry.get("png_pixels") or 0) for entry in per_image),
        "components_in_instance_crops": stats_total["components"],
        "kept_components": stats_total["kept_components"],
        "fallback_counts": {
            "a6_no_kept_component": stats_total["instances_without_component"],
            "a6_to_a1": stats_total["a6_to_a1"],
            "a1_to_original": stats_total["a1_to_original"],
            "a1_rule_empty": stats_total["a1_rule_empty"],
            "dropped_instances": stats_total["dropped_instances"],
        },
        "measured_on_val387": refine_policy.get("measured_on_val387"),
        "measured_note": refine_policy.get("measured_note"),
        "measured_at_component_coverage": refine_policy.get("measured_at_component_coverage"),
        "ratio_sweep": refine_policy.get("ratio_sweep"),
        "per_image": per_image,
    }


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        try:
            import dataclasses

            return _jsonable(dataclasses.asdict(value))
        except Exception:
            return repr(value)
    return repr(value)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def environment_block():
    modules = sys.modules
    torch = modules.get("torch")
    torchvision = modules.get("torchvision")
    numpy = modules.get("numpy")
    pillow = modules.get("PIL.Image")
    block = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "prefix": sys.prefix,
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "env": {
            key: os.environ.get(key)
            for key in (
                "PYTHONUTF8",
                "PYTHONIOENCODING",
                "PYTHONNOUSERSITE",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
            if os.environ.get(key) is not None
        },
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "torchvision": getattr(torchvision, "__version__", None),
        "torchvision_file": getattr(torchvision, "__file__", None),
        "numpy": getattr(numpy, "__version__", None),
        "pillow": getattr(pillow, "__version__", None),
        "cuda_available": None,
        "cuda_devices": [],
        "cudnn": None,
    }
    if torch is not None:
        try:
            block["cuda_available"] = bool(torch.cuda.is_available())
            if block["cuda_available"]:
                block["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            block["cudnn"] = getattr(getattr(torch, "backends", None), "cudnn", None) and torch.backends.cudnn.version()
        except Exception:
            pass
    return block


def lane_sources():
    """Hashes of the frozen code the delivery runs on."""

    relative = {
        "run_batch_predict.py": os.path.join(EVAL_DIR, "run_batch_predict.py"),
        "instance_eval/batch.py": os.path.join(EVAL_DIR, "instance_eval", "batch.py"),
        "instance_eval/contract.py": os.path.join(EVAL_DIR, "instance_eval", "contract.py"),
        "model/infer.py": os.path.join(MODEL_ROOT, "model", "infer.py"),
        "model/emit.py": os.path.join(MODEL_ROOT, "model", "emit.py"),
        "src/cadvec/recognition_submission.py": os.path.join(SRC_DIR, "cadvec", "recognition_submission.py"),
        "src/cadvec/png_raster_vectorize.py": os.path.join(SRC_DIR, "cadvec", "png_raster_vectorize.py"),
    }
    sources = {}
    for name, path in relative.items():
        entry = {"path": path, "exists": os.path.isfile(path), "sha256": None, "bytes": None}
        if entry["exists"]:
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = os.path.getsize(path)
        sources[name] = entry
    sources["cad_mrcnn_delivery.py"] = {
        "path": os.path.abspath(__file__),
        "exists": True,
        "sha256": sha256_file(os.path.abspath(__file__)),
        "bytes": os.path.getsize(os.path.abspath(__file__)),
    }
    return sources


# --------------------------------------------------------------------------- #
# independent document checker (the lane's validate_document does not enforce
# width/height==100 or the viewBox, so the delivery contract needs its own)
# --------------------------------------------------------------------------- #
VIEWBOX = (0.0, 0.0, 100.0, 100.0)
STUFF_CLASS_START = 30
IGNORE_CLASS = 35
BOUND_TOL = 1e-6

# The scorer rasterises the submitted ``args`` through the 100 x 100 viewBox onto
# its own 1000 px canvas; it never looks at the PNG's own aspect ratio, and the
# PNG's dimensions matter only to the model.  So for any W x H raster the
# coordinate mapping is: ``side = max(W, H)``, offset 0, scale ``100.0 / side``
# -- the drawing is treated as having been rendered on a square ``side x side``
# canvas and cropped at the right/bottom.  That affine always lands inside the
# viewBox, so it can never trip the fail-closed document guard.
#
# In this dataset the non-square records really are that crop: 6000/6000 PNGs
# are 1000x1000 (5794), 1000x999 (125), 999x1000 (79) or 999x999 (2), and the
# FYrUh-class SVGs declare viewBox "0 0 100.0 99.99999999999994".  The lane's
# emitter, by contrast, letterboxes: submission.viewbox_transform(width, height)
# is min(w/100, h/100) plus a centring offset, so a 1000x999 raster becomes scale
# 9.99 with offset_x 0.5 and a mask touching column 0 lands at x = -0.05005,
# outside the viewBox.  Padding the mask back to its square canvas before emit
# restores the exact inverse: scale 10, no offset, coordinates 0..99.9.  Proof
# on the held-out pair: a mask at rows 100:200 cols 300:400 maps to
# x[30.0, 39.9] y[10.0, 19.9].
#
# A missing document scores zero for that image; a slightly mis-scaled document
# still scores something.  Refusing to emit is therefore the worst failure mode,
# so a non-square raster is NEVER refused.  Sides that differ by more than
# CROP_RESIDUAL_MAX_PX are an unusual aspect ratio: the receipt records
# ``crop_residual_px``, ``canvas_padding`` and a note, and the image is delivered
# anyway.  No per-axis stretch and no letterbox -- the image is cropped, not
# resized, and both alternatives measured worse.
#
# The pad is applied here, in the delivery lane; model/emit.py stays frozen.
CROP_RESIDUAL_MAX_PX = 2


def canvas_padding_record(width, height):
    """Describe the right/bottom zero-pad a raster needs to reach its square canvas.

    Square rasters get zero padding and no note; a non-square raster is treated
    as a square ``side x side`` canvas cropped at the right/bottom, so the pad is
    all on those two sides.  Sides further apart than ``CROP_RESIDUAL_MAX_PX``
    are flagged as an unusual aspect ratio -- flagged, never refused.
    """

    width, height = int(width), int(height)
    side = max(width, height)
    residual = abs(width - height)
    unusual = residual > CROP_RESIDUAL_MAX_PX
    return {
        "side": side,
        "crop_residual_px": residual,
        "canvas_padding": {
            "left": 0,
            "top": 0,
            "right": side - width,
            "bottom": side - height,
        },
        "unusual_aspect": unusual,
        "note": (
            "unusual aspect ratio %dx%d (sides differ by %d px): treated as a square %dx%d "
            "canvas cropped at the right/bottom and zero-padded, not stretched and not "
            "letterboxed; delivered rather than refused because a missing document scores zero"
            % (width, height, residual, side, side)
        )
        if unusual
        else None,
    }


def square_canvas_masks(masks, width, height):
    """Pad any non-square raster to its square canvas, right/bottom.

    Returns ``(masks, side, padded)``.  Square rasters come back untouched with
    ``side = None``.  A non-square raster is treated as a square ``side x side``
    canvas cropped at the right/bottom, so it is zero-padded (zeros are black,
    i.e. background: no ink and no mask) up to ``side = max(width, height)``.
    Padding is never refused; ``canvas_padding_record`` flags an unusual aspect
    ratio for the receipt instead.
    """

    width, height = int(width), int(height)
    side = max(width, height)
    if width == height:
        return masks, None, False
    import numpy  # noqa: PLC0415 - the CLI stays numpy-free until a non-square raster appears

    padded = []
    for mask in masks:
        canvas = numpy.zeros((side, side), dtype=numpy.bool_)
        canvas[:height, :width] = numpy.asarray(mask, dtype=bool)
        padded.append(canvas)
    return padded, side, True


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def document_report(document):
    """Check a delivery document against the twelve-field contract.

    Returns a list of violations; empty means the document may be delivered.
    """

    if not isinstance(document, dict):
        return ["document is not an object"]

    violations = []
    present = set(document)
    missing = sorted(set(DOCUMENT_FIELDS) - present)
    extra = sorted(present - set(DOCUMENT_FIELDS))
    if missing:
        violations.append("missing_fields:%s" % ",".join(missing))
    if extra:
        violations.append("unknown_fields:%s" % ",".join(extra))
    if missing:
        return violations

    for field in ("width", "height"):
        value = document[field]
        if not _is_number(value) or float(value) != VIEWBOX[2]:
            violations.append("%s:%r" % (field, value))

    arrays = {
        field: document[field]
        for field in ("commands", "args", "lengths", "widths", "semanticIds", "instanceIds", "layerIds", "rgb")
    }
    for field, value in arrays.items():
        if not isinstance(value, list):
            violations.append("%s:not_a_list" % field)
    if violations:
        return violations

    sizes = {field: len(value) for field, value in arrays.items()}
    if len(set(sizes.values())) != 1:
        violations.append("length_mismatch:%r" % (sizes,))
        return violations
    count = len(arrays["commands"])

    things = []
    for index in range(count):
        command = arrays["commands"][index]
        if not _is_int(command) or not 0 <= command <= 3:
            violations.append("commands[%d]:%r" % (index, command))
        argument = arrays["args"][index]
        if not isinstance(argument, list) or len(argument) != 8 or not all(_is_number(item) for item in argument):
            violations.append("args[%d]:%r" % (index, argument))
        else:
            low = min(argument)
            high = max(argument)
            if low < VIEWBOX[0] - BOUND_TOL or high > VIEWBOX[2] + BOUND_TOL:
                violations.append("args[%d]:outside_viewbox %r" % (index, argument))
        length = arrays["lengths"][index]
        if not _is_number(length) or length < 0:
            violations.append("lengths[%d]:%r" % (index, length))
        width = arrays["widths"][index]
        if not _is_number(width) or width < 0:
            violations.append("widths[%d]:%r" % (index, width))
        semantic = arrays["semanticIds"][index]
        instance = arrays["instanceIds"][index]
        if not _is_int(semantic) or not 0 <= semantic <= IGNORE_CLASS:
            violations.append("semanticIds[%d]:%r" % (index, semantic))
            continue
        if not _is_int(instance) or instance < -1:
            violations.append("instanceIds[%d]:%r" % (index, instance))
            continue
        if semantic < STUFF_CLASS_START:
            if instance < 0:
                violations.append("thing_instance_id[%d]:%r" % (index, instance))
            else:
                things.append(instance)
        elif instance != -1:
            violations.append("stuff_instance_id[%d]:%r" % (index, instance))
        rgb = arrays["rgb"][index]
        if not isinstance(rgb, list) or len(rgb) != 3 or not all(_is_int(channel) and 0 <= channel <= 255 for channel in rgb):
            violations.append("rgb[%d]:%r" % (index, rgb))
        layer = arrays["layerIds"][index]
        if not _is_int(layer) or layer < 1:
            violations.append("layerIds[%d]:%r" % (index, layer))

    unique = set(things)
    first_seen = list(dict.fromkeys(things))
    if first_seen != sorted(first_seen):
        violations.append("thing_instance_ids:not_ascending_by_first_appearance")

    centers = document["obj_cts"]
    boxes = document["boxes"]
    if not isinstance(centers, list) or not isinstance(boxes, list):
        return violations + ["obj_cts/boxes:not_a_list"]
    for index, row in enumerate(centers):
        if (
            not isinstance(row, list)
            or len(row) != 4
            or not _is_number(row[0])
            or not _is_number(row[1])
            or not _is_int(row[3])
        ):
            violations.append("obj_cts[%d]:%r" % (index, row))
    for index, row in enumerate(boxes):
        if (
            not isinstance(row, list)
            or len(row) != 5
            or not all(_is_number(item) for item in row[:4])
            or not _is_int(row[4])
        ):
            violations.append("boxes[%d]:%r" % (index, row))
            continue
        if not 0 <= row[4] <= IGNORE_CLASS:
            violations.append("boxes[%d]:semantic %r" % (index, row[4]))
        if row[4] < STUFF_CLASS_START:
            low = min(row[:4])
            high = max(row[:4])
            if low < VIEWBOX[0] - BOUND_TOL or high > VIEWBOX[2] + BOUND_TOL:
                violations.append("boxes[%d]:outside_viewbox %r" % (index, row))
    thing_rows = [row for row in boxes if isinstance(row, list) and len(row) == 5 and _is_int(row[4]) and row[4] < STUFF_CLASS_START]
    if len(thing_rows) != len(unique):
        violations.append("boxes:%d thing rows for %d thing instances" % (len(thing_rows), len(unique)))
    return violations


# --------------------------------------------------------------------------- #
# the delivery run
# --------------------------------------------------------------------------- #
def run_delivery(
    args,
    *,
    loader=None,
    predict=None,
    png_loader=None,
    emit=None,
    clock=time.perf_counter,
):
    """Execute one delivery run; returns the summary dict.

    ``loader`` / ``predict`` / ``png_loader`` / ``emit`` are injection points for
    the CPU tests (they replace model access, never the selection stage unless
    ``emit`` is overridden deliberately).
    """

    # 1. knobs first: nothing below this line before every knob is validated.
    policy = validate_selection(
        score_threshold=getattr(args, "score_threshold", None),
        mask_threshold=getattr(args, "mask_threshold", None),
        nms_iou=getattr(args, "nms_iou", None),
        max_detections=getattr(args, "max_detections", None),
        per_image_cap=getattr(args, "per_image_cap", None),
        refine_mask=getattr(args, "refine_mask", None),
        refine_ink=getattr(args, "refine_ink", None),
        refine_ratio=getattr(args, "refine_ratio", None),
        drop_stuff=getattr(args, "drop_stuff", False),
        mask_paste=getattr(args, "mask_paste", None),
        mask_logit_pad=getattr(args, "mask_logit_pad", None),
        mask_ink_min_side=getattr(args, "mask_ink_min_side", None),
        mask_ink_stage=getattr(args, "mask_ink_stage", None),
        mask_source=getattr(args, "mask_source", None),
        mask_head_min_side=getattr(args, "mask_head_min_side", None),
        mask_head_tau=getattr(args, "mask_head_tau", None),
        mask_head_gate=getattr(args, "mask_head_gate", None),
        mask_head_ckpt=getattr(args, "mask_head_ckpt", None),
        select_rule=getattr(args, "select_rule", None),
        nms_box_iou=getattr(args, "nms_box_iou", None),
        nms_box_thr=getattr(args, "nms_box_thr", None),
        blend_model=getattr(args, "blend_model", None),
        blend_thr=getattr(args, "blend_thr", None),
        blend_tau=getattr(args, "blend_tau", None),
        blend_cap=getattr(args, "blend_cap", None),
    )

    # 1b. mask head: a build-time choice, so it is validated here with the knobs.
    # Only the 112 head enters the policy block, which keeps the 28 head's policy,
    # selection signature and every document byte-identical to earlier runs.
    mask_head_policy = validate_mask_head(
        getattr(args, "mask_head", None), getattr(args, "mask_chunk", None)
    )
    if mask_head_policy["head"] != DEFAULT_MASK_HEAD:
        policy["mask_head"] = mask_head_policy

    # 2. paths (stat only)
    allow_official_inputs = bool(getattr(args, "allow_official_inputs", False))
    resume = bool(getattr(args, "resume", False))
    plan = validate_delivery_paths(
        out=getattr(args, "out", None),
        checkpoint=getattr(args, "checkpoint", None) or DEFAULT_CHECKPOINT,
        png=getattr(args, "png", None),
        inputs=getattr(args, "inputs", None),
        device=getattr(args, "device", None),
        allow_official_inputs=allow_official_inputs,
        resume=resume,
    )
    out = plan["out"]

    # 3. image-only inputs
    image_only = image_only_inputs(
        png=plan["png"],
        inputs=plan["inputs"],
        sample_id=getattr(args, "sample_id", None),
        group_id=getattr(args, "group_id", None),
        allow_official_inputs=allow_official_inputs,
    )

    summary = {
        "status": "planned",
        "out": out,
        "checkpoint": plan["checkpoint"],
        "device": plan["device"],
        "policy": policy,
        "images": [
            {"sample_id": row["sample_id"], "png_path": row["png_path"]} for row in image_only["samples"]
        ],
        "selection_report": os.path.abspath(getattr(args, "selection_report", None) or os.path.join(out, "selection_report.json")),
    }
    if allow_official_inputs:
        # Provenance for the non-default guard only: a normal run keeps exactly the
        # summary keys it has always had.
        summary["guards"] = {
            "allow_official_inputs": True,
            "relaxed": "input PNGs under a datasets/ directory (offline self-check only)",
        }
    if getattr(args, "dry_run", False):
        summary["status"] = "dry_run"
        summary["image_only_inputs"] = image_only
        if mask_head_policy["head"] != DEFAULT_MASK_HEAD:
            summary["mask_head"] = mask_head_policy
        return summary

    C = _contract()
    batch = _batch()

    # The 112 head swaps the loader only; run_batch, predict, emit and the whole
    # selection/refinement stage below stay the same code.
    head_loader = None
    if mask_head_policy["head"] != DEFAULT_MASK_HEAD:
        head_loader = fine_head_loader(mask_head_policy["chunk"])

    os.makedirs(out, exist_ok=True)
    results_dir = os.path.join(out, "results")

    # --resume: the directory may still hold the documents of a run that died.
    # Check its contents before anything is written into it.
    samples = image_only["samples"]
    reused = []
    if resume:
        validate_resume_out(out, samples, results_dir)
        reused = _reusable_documents(out, results_dir, samples, contract=C)
    reused_keys = {
        os.path.normcase(os.path.abspath(item["png_path"])) for item in reused
    }
    pending = [
        row
        for row in samples
        if os.path.normcase(os.path.abspath(row["png_path"])) not in reused_keys
    ]
    if resume:
        sys.stderr.write("resume: reused %d/%d\n" % (len(reused), len(samples)))
        sys.stderr.flush()

    inputs_path = atomic_json(os.path.join(out, "inputs", "image_only_inputs.json"), image_only)
    # Only the images that still need producing: run_batch reads this file to
    # decide what to run, and reused documents must not be emitted a second time
    # (that would be a rewrite, and C.write_json refuses to overwrite).
    pending_path = inputs_path
    if len(pending) != len(samples):
        pending_path = atomic_json(
            os.path.join(out, "inputs", "resume_pending_inputs.json"),
            {
                "schema": SCHEMA_INPUTS,
                "created_utc": _utc_now(),
                "derived_from": "resume",
                "count": len(pending),
                "samples": pending,
            },
        )
    predictions_dir = os.path.join(out, "predictions")
    if resume and os.path.isdir(predictions_dir) and os.listdir(predictions_dir):
        # run_batch writes into a fresh directory; an earlier attempt already
        # filled the first one, so the resumed attempt takes the next free slot.
        slot = 1
        while os.path.exists(os.path.join(predictions_dir, "resume-%02d" % slot)):
            slot += 1
        predictions_dir = os.path.join(predictions_dir, "resume-%02d" % slot)

    # Emit-time publication: _publish_document validates and writes one document
    # the moment its emit returns, so a run killed mid-batch keeps the finished
    # ones and --resume can skip them.  The facts are cached for the export stage.
    written = {}

    def publish(record):
        png_path = record["png_path"]
        facts = _publish_document(
            out, results_dir, png_path, record.get("document"), contract=C
        )
        written[os.path.normcase(os.path.abspath(png_path))] = facts
        progress.tick(_png_stem(png_path), record.get("total_s"))

    progress = _Progress(len(samples), done=len(reused), clock=clock)

    # --mask-paste logits wraps predict and nothing else: the lane's
    # predict_arrays, the driver, the selection stage, the refinement and the
    # emitter are all the same code.  The default prob paste wraps nothing, so
    # its documents are the ones earlier runs produced, byte for byte.
    predict_impl = predict
    paste_policy = policy.get("mask_paste")
    if paste_policy:
        predict_impl = mask_paste.wrap_predict(
            predict or batch.default_predict, paste_policy["logit_pad"]
        )

    run = DeliveryRun(
        policy,
        clock=clock,
        loader_impl=loader or head_loader,
        predict_impl=predict_impl,
        png_impl=png_loader,
        on_document=publish,
        dump_pool=getattr(args, "dump_pool", None),
        head_box=getattr(args, "head_box", "decoded"),
    )
    audit = OpenAudit(
        roles={
            plan["checkpoint"]: "checkpoint",
            inputs_path: "inputs",
            **({pending_path: "inputs"} if pending_path != inputs_path else {}),
            **{row["png_path"]: "png" for row in image_only["samples"]},
            **(
                {policy["blend"]["model"]: "selector"}
                if policy.get("blend")
                else {}
            ),
        }
    )

    wall_start = clock()
    stage = "predict"
    try:
        audit.install()
        if pending:
            result = batch.run_batch(
                predictions_dir,
                checkpoint=plan["checkpoint"],
                device=plan["device"],
                inputs=pending_path,
                sample_set="explicit",
                max_detections=None,  # the decode cap stays the model's own 300
                score_threshold=None,  # the decode threshold stays the checkpoint's 0.05
                # 0.5 keeps the decoder's own binarisation; a measured choice from
                # MASK_THRESHOLD_CHOICES is handed to the lane explicitly.
                mask_threshold=(
                    None
                    if policy["mask_threshold"] == DEFAULT_MASK_THRESHOLD
                    else policy["mask_threshold"]
                ),
                dense_diagnostic=False,
                loader=run.loader,
                predict=run.predict,
                png_loader=run.png_loader,
                emit=emit or run.emit,
            )
        else:
            # --resume over a directory that already holds every document: nothing
            # to infer, nothing to write, but the bookkeeping below still runs.
            result = {
                "schema": "cadvec.direct_instance.batch_predict.v1",
                "status": "no_pending_samples",
                "predictions_dir": predictions_dir,
                "predictions": [],
                "checkpoint": plan["checkpoint"],
                "checkpoint_sha256": sha256_file(plan["checkpoint"]),
                "device": plan["device"],
                "checkpoint_loads": 0,
                "png_reads": 0,
                "seal_written": False,
                "gt_read": False,
                "svg_read": False,
            }
    except C.GuardError as exc:
        _write_failure(out, stage, exc)
        raise DeliveryRefused(getattr(exc, "code", "lane_error"), str(exc.detail)) from None
    except DeliveryRefused:
        raise
    except Exception as exc:
        _write_failure(out, stage, exc)
        raise
    wall_s = clock() - wall_start
    audit_report = audit.report()

    # 4. results/<stem>.json, one document each
    stage = "export"
    try:
        records = list(result.get("predictions") or [])
        produced = list(run.images)
        if len(records) != len(produced):
            raise DeliveryRefused(
                "selection_alignment",
                "%d predictions but %d selection records" % (len(records), len(produced)),
            )
        exports = []
        for index, (record, selection) in enumerate(zip(records, produced)):
            if os.path.normcase(record["png_path"]) != os.path.normcase(selection["png_path"]):
                raise DeliveryRefused(
                    "selection_alignment",
                    "record %d png mismatch: %s vs %s"
                    % (index, record["png_path"], selection["png_path"]),
                )
            document = selection["document"]
            key = os.path.normcase(os.path.abspath(record["png_path"]))
            facts = written.get(key)
            if facts is None:
                # An injected emit bypassed DeliveryRun.emit, so nothing was
                # published for this image: validate and write it here instead.
                facts = _publish_document(
                    out, results_dir, record["png_path"], document, contract=C
                )
            stem = _png_stem(record["png_path"])
            timing = {
                key: selection.get(key)
                for key in (
                    "read_s",
                    "predict_s",
                    "select_s",
                    "blend_s",
                    "refine_s",
                    "emit_s",
                    "total_s",
                )
            }
            if selection.get("ink_box_s") is not None:
                timing["ink_box_s"] = selection.get("ink_box_s")
            if selection.get("mask_source_s") is not None:
                timing["mask_source_s"] = selection.get("mask_source_s")
            exports.append(
                {
                    "stem": stem,
                    "sample_id": record["sample_id"],
                    "png_path": record["png_path"],
                    "png_sha256": record["png_sha256"],
                    "result": facts["result"],
                    "document_sha256": facts["document_sha256"],
                    "primitives": facts["primitives"],
                    "emitted_instances": selection["emitted_instances"],
                    "candidates": selection["counts"]["candidates"],
                    "after_nms": selection["counts"]["after_nms"],
                    "after_threshold": selection["counts"]["after_threshold"],
                    "after_cap": selection["counts"]["after_cap"],
                    "after_per_image_cap": selection["counts"]["after_per_image_cap"],
                    "lowest_surviving_score": selection["lowest_surviving_score"],
                    "width": selection.get("width"),
                    "height": selection.get("height"),
                    "square": selection.get("width") == selection.get("height"),
                    "emit_canvas": dict(selection.get("emit_canvas") or {}),
                    "refine": dict(selection.get("refine") or {}),
                    "stuff_drop": selection.get("stuff_drop"),
                    "blend": selection.get("blend"),
                    "ink_box": selection.get("ink_box"),
                    "mask_source": selection.get("mask_source"),
                    "timing": timing,
                }
            )
    except Exception as exc:
        _write_failure(out, stage, exc)
        raise

    # 5. selection report
    refine_block = refine_summary(policy, exports)
    selection_block = {
        "schema": SCHEMA_SELECTION,
        "created_utc": _utc_now(),
        "policy": policy,
        "evidence": {
            "held_out_reference": "387 images / 361 groups / 3044 GT instances, PQ 0.005939695322 at the "
            "checkpoint defaults with no post-decode selection; ~296 detections/image at 0.05 "
            "against ~7.9 GT instances/image",
            "note": "the shipping score threshold is supplied by the caller; this CLI never tunes it",
        },
        "images": [
            {
                "sample_id": item["sample_id"],
                "png_path": item["png_path"],
                "result": item["result"],
                "candidates": item["candidates"],
                "after_nms": item["after_nms"],
                "after_threshold": item["after_threshold"],
                "after_cap": item["after_cap"],
                "after_per_image_cap": item["after_per_image_cap"],
                "selected": item["after_per_image_cap"],
                "emitted": item["emitted_instances"],
                "lowest_surviving_score": item["lowest_surviving_score"],
                "document_primitives": item["primitives"],
                "width": item["width"],
                "height": item["height"],
                "square": item["square"],
                "emit_canvas": item["emit_canvas"],
                "refine": item["refine"],
                "blend": item.get("blend"),
            }
            for item in exports
        ],
        "totals": {
            "images": len(exports),
            "candidates": sum(item["candidates"] for item in exports),
            "after_nms": sum(item["after_nms"] for item in exports),
            "after_threshold": sum(item["after_threshold"] for item in exports),
            "after_cap": sum(item["after_cap"] for item in exports),
            "selected": sum(item["after_per_image_cap"] for item in exports),
            "emitted_instances": sum(item["emitted_instances"] for item in exports),
            "non_square_images": sum(1 for item in exports if not item["square"]),
            "lowest_surviving_score": min(
                (item["lowest_surviving_score"] for item in exports if item["lowest_surviving_score"] is not None),
                default=None,
            ),
        },
        "refine_mask": refine_block,
    }
    if policy.get("drop_stuff"):
        for entry, item in zip(selection_block["images"], exports):
            entry["stuff_drop"] = item.get("stuff_drop")
        selection_block["totals"]["stuff_dropped_instances"] = sum(
            int((item.get("stuff_drop") or {}).get("dropped") or 0) for item in exports
        )
        selection_block["stuff_drop"] = dict(
            policy["drop_stuff"],
            total_selected=int(selection_block["totals"]["selected"]),
            total_emitted=int(selection_block["totals"]["emitted_instances"]),
        )
    if policy.get("ink_box"):
        # Provenance for the non-default knob only: entries keep their old keys.
        for entry, item in zip(selection_block["images"], exports):
            entry["ink_box"] = item.get("ink_box")
        selection_block["totals"]["ink_box_replaced"] = sum(
            int(((item.get("ink_box") or {}).get("counts") or {}).get("replaced") or 0)
            for item in exports
        )
    if policy.get("mask_source"):
        # Same rule: the block exists only for --mask-source head/ink.
        for entry, item in zip(selection_block["images"], exports):
            entry["mask_source"] = item.get("mask_source")
        selection_block["totals"]["mask_source_replaced"] = sum(
            int(((item.get("mask_source") or {}).get("counts") or {}).get("replaced") or 0)
            for item in exports
        )
    if policy.get("blend"):
        # Provenance for the non-default rule only (the measured readout): the pool
        # totals above stay the pool's own counts, the blend totals sit beside them.
        selection_block["totals"]["blend_candidates"] = sum(
            int(((item.get("blend") or {}).get("counts") or {}).get("candidates") or 0)
            for item in exports
        )
        selection_block["totals"]["blend_selected"] = sum(
            int(((item.get("blend") or {}).get("counts") or {}).get("selected") or 0)
            for item in exports
        )
        selection_block["selection_rule"] = dict(policy["blend"])
    if reused:
        # --resume provenance: the pool totals above count this run's produced
        # images only; reused documents kept the bytes an earlier run wrote.
        selection_block["resume"] = {
            "enabled": True,
            "reused": len(reused),
            "produced": len(exports),
            "reused_images": [
                {
                    "sample_id": item["sample_id"],
                    "png_path": item["png_path"],
                    "result": item["result"],
                    "document_sha256": item["document_sha256"],
                }
                for item in reused
            ],
        }
        selection_block["totals"]["reused_images"] = len(reused)
    selection_paths = [atomic_json(os.path.join(out, "selection_report.json"), selection_block)]
    extra_report = getattr(args, "selection_report", None)
    if extra_report:
        selection_paths.append(atomic_json(extra_report, selection_block))

    index = {
        "schema": SCHEMA_INDEX,
        "created_utc": _utc_now(),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "policy": policy,
        "images": {
            item["stem"]: {
                "sample_id": item["sample_id"],
                "png_path": item["png_path"],
                "png_sha256": item["png_sha256"],
                "result": item["result"],
                "document_sha256": item["document_sha256"],
                "validated_by": "instance_eval.contract.validate_prediction_document",
            }
            for item in exports
        },
        "counts": {
            "images": len(exports) + len(reused),
            "produced": len(exports),
            "reused": len(reused),
        },
    }
    for item in reused:
        # A --resume document is on disk and validated, so it belongs in the index
        # like a produced one; "reused": true says this run did not write it.
        index["images"][_png_stem(item["png_path"])] = {
            "sample_id": item["sample_id"],
            "png_path": item["png_path"],
            "png_sha256": item["png_sha256"],
            "result": item["result"],
            "document_sha256": item["document_sha256"],
            "validated_by": "instance_eval.contract.validate_prediction_document",
            "reused": True,
        }
    index_path = atomic_json(os.path.join(out, "results_index.json"), index)

    # 6. run receipt
    totals = {
        "images": len(exports),
        "checkpoint_load_s": run.load_s,
        "wall_s": wall_s,
        "read_s": sum(item["timing"]["read_s"] or 0.0 for item in exports),
        "predict_s": sum(item["timing"]["predict_s"] or 0.0 for item in exports),
        "select_s": sum(item["timing"]["select_s"] or 0.0 for item in exports),
        "blend_s": sum(item["timing"]["blend_s"] or 0.0 for item in exports),
        "refine_s": sum(item["timing"]["refine_s"] or 0.0 for item in exports),
        "emit_s": sum(item["timing"]["emit_s"] or 0.0 for item in exports),
    }
    per_image = [item["timing"]["total_s"] for item in exports if item["timing"]["total_s"]]
    ordered = sorted(per_image)
    if ordered:
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    else:
        median = None
    totals.update(
        {
            "per_image_mean_s": (sum(per_image) / len(per_image)) if per_image else None,
            "per_image_median_s": median,
            "per_image_max_s": ordered[-1] if ordered else None,
            "images_per_s": (len(per_image) / wall_s) if per_image and wall_s else None,
        }
    )

    receipt = {
        "schema": SCHEMA_RECEIPT,
        "status": "delivery_written",
        "created_utc": _utc_now(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "checkpoint": {
            "path": plan["checkpoint"],
            "sha256": result.get("checkpoint_sha256"),
            "bytes": os.path.getsize(plan["checkpoint"]),
        },
        "device": plan["device"],
        "decode": run.decode,
        "policy": policy,
        "selection_signature": list(selection_signature(policy)),
        "inputs": {
            "image_only_path": inputs_path,
            "image_only_sha256": sha256_file(inputs_path),
            "count": image_only["count"],
            "derived_from": image_only["derived_from"],
            "ignored_row_keys": image_only["ignored_row_keys"],
        },
        "outputs": {
            "out": out,
            "results_dir": results_dir,
            "predictions_dir": os.path.abspath(str(result.get("predictions_dir"))),
            "results_index": index_path,
            "selection_reports": selection_paths,
            "n_results": len(exports),
            "n_produced": len(exports),
            "n_reused": len(reused),
        },
        "counts": dict(index["counts"]),
        "guards": {
            "ground_truth_read": False,
            "svg_read": False,
            "checkpoint_loads": result.get("checkpoint_loads"),
            "png_reads": result.get("png_reads"),
            "document_fields_exact": True,
            "score_field_in_document": False,
            "selection_before_emit": True,
            "decode_settings_from_flags": bool(
                policy["mask_threshold"] != DEFAULT_MASK_THRESHOLD
            ),
            "all_images_square": all(item["square"] for item in exports),
            "refine_before_emit": True,
            "refine_rule": (policy.get("refine") or {}).get("rule"),
            "refine_component_coverage": (policy.get("refine") or {}).get("component_coverage"),
            "refine_component_coverage_consistent": refine_block["component_coverage_consistent"],
            "refine_kept_instance_count": (
                refine_block["instances_nonempty_after"] == refine_block["instances_nonempty_before"]
            ),
            "refine_dropped_instances": refine_block["fallback_counts"]["dropped_instances"],
        },
        "non_square": {
            "policy": "every raster is mapped to the 100 x 100 viewBox through side = max(width, "
            "height), offset 0, scale 100.0 / side: a non-square PNG is treated as a square "
            "side x side canvas cropped at the right/bottom, so the mask is zero-padded "
            "right/bottom to that square and emitted at side x side.  Square rasters are "
            "emitted unchanged.  Nothing is refused: a missing document scores zero, a "
            "mis-scaled one still scores something.  No stretch, no letterbox.",
            "assumption": "a non-square PNG is a square canvas cropped at the right/bottom, which "
            "is what the dataset's non-square records are (1000x999, 999x1000, 999x999)",
            "crop_residual_max_px": CROP_RESIDUAL_MAX_PX,
            "non_square_images": sum(1 for item in exports if not item["square"]),
            "padded_images": sum(1 for item in exports if (item.get("emit_canvas") or {}).get("padded_to_square")),
            "unusual_aspect_images": sum(
                1 for item in exports if (item.get("emit_canvas") or {}).get("unusual_aspect")
            ),
            "crop_residual_px_max": max(
                ((item.get("emit_canvas") or {}).get("crop_residual_px") or 0 for item in exports),
                default=0,
            ),
            "images": [
                {
                    "sample_id": item["sample_id"],
                    "native_width": item["width"],
                    "native_height": item["height"],
                    "emit_width": (item.get("emit_canvas") or {}).get("width"),
                    "emit_height": (item.get("emit_canvas") or {}).get("height"),
                    "crop_residual_px": (item.get("emit_canvas") or {}).get("crop_residual_px"),
                    "canvas_padding": dict((item.get("emit_canvas") or {}).get("canvas_padding") or {}),
                    "unusual_aspect": bool((item.get("emit_canvas") or {}).get("unusual_aspect")),
                    "note": (item.get("emit_canvas") or {}).get("note"),
                }
                for item in exports
                if not item["square"]
            ],
            "lane_files_modified": False,
            "deviation": "model/emit.py is untouched; the pad happens in this CLI before it calls model.emit",
        },
        "refine_mask": refine_block,
        "lane": {
            "driver": result.get("schema"),
            "gt_read_reported_by_driver": result.get("gt_read"),
            "svg_read_reported_by_driver": result.get("svg_read"),
            "seal_written": result.get("seal_written"),
            "batch_max_detections": result.get("max_detections"),
            "sources": lane_sources(),
        },
        "timings": {"per_image": [dict(item["timing"], stem=item["stem"], sample_id=item["sample_id"], width=item["width"], height=item["height"], square=item["square"]) for item in exports], "totals": totals},
        "environment": environment_block(),
        "open_audit": audit_report,
    }
    if mask_head_policy["head"] != DEFAULT_MASK_HEAD:
        # Provenance only for the non-default head: the 28 head's receipt keeps
        # exactly the keys it has always had.
        receipt["mask_head"] = dict(
            run.mask_head or {},
            head=mask_head_policy["head"],
            output_size=mask_head_policy["output_size"],
            chunk_requested=mask_head_policy["chunk_requested"],
            loader="fine_head.fine_head_loader",
        )
    if policy.get("box_nms"):
        # Provenance for the non-default rule only: the plain rule keeps exactly
        # the receipt keys it has always had.
        receipt["guards"]["select_rule"] = policy["box_nms"]["rule"]
        receipt["selection_rule"] = dict(policy["box_nms"])
    if policy.get("blend"):
        # Provenance for the non-default rule only, as above.
        receipt["guards"]["select_rule"] = policy["blend"]["rule"]
        receipt["guards"]["blend_after_refine"] = True
        receipt["selection_rule"] = dict(policy["blend"])
    if policy.get("drop_stuff"):
        receipt["guards"]["stuff_dropped_instances"] = sum(
            int((item.get("stuff_drop") or {}).get("dropped") or 0) for item in exports
        )
        receipt["stuff_drop"] = dict(
            policy["drop_stuff"],
            total_selected=int(selection_block["totals"]["selected"]),
            total_emitted=int(selection_block["totals"]["emitted_instances"]),
        )
    if allow_official_inputs:
        # Provenance for the non-default guard only (an offline self-check run).
        receipt["guards"]["allow_official_inputs"] = True
        receipt["official_inputs_guard"] = {
            "allow_official_inputs": True,
            "relaxed_rule": "sealed-token refusal skipped for an input PNG whose path holds a "
                            "datasets/ directory; every other path keeps the full refusal",
            "sealed_tokens": list(SEALED_TOKENS),
            "purpose": "offline self-check (dev_final / test_final); the platform run needs it off",
        }
    if paste_policy:
        # Provenance for the non-default paste only: the prob paste keeps exactly
        # the receipt keys it has always had.
        receipt["mask_paste"] = dict(paste_policy)
    if policy.get("ink_box"):
        # Provenance for the non-default knob only, as above: with
        # --mask-ink-min-side 0 the receipt keeps exactly the keys it has.
        ink_box_counts = {key: 0 for key in INK_BOX_STAT_KEYS}
        for item in exports:
            block = item.get("ink_box") or {}
            for key in INK_BOX_STAT_KEYS:
                ink_box_counts[key] += int((block.get("counts") or {}).get(key, 0))
        receipt["ink_box"] = dict(
            policy["ink_box"],
            totals=ink_box_counts,
            images=len(exports),
            images_with_replacement=sum(
                1 for item in exports if int(((item.get("ink_box") or {}).get("counts") or {}).get("replaced", 0))
            ),
        )
        receipt["guards"]["ink_box_stage"] = policy["ink_box"].get("stage_placement")
        receipt["guards"]["ink_box_after_refine"] = True
        receipt["guards"]["ink_box_after_selection"] = bool(
            policy["ink_box"].get("stage_placement") == "post" and policy.get("blend")
        )
        receipt["guards"]["ink_box_before_blend"] = bool(
            policy["ink_box"].get("stage_placement") == "pre" and policy.get("blend")
        )
        totals["ink_box_s"] = sum(item["timing"].get("ink_box_s") or 0.0 for item in exports)
    if policy.get("mask_source"):
        # Provenance for the non-default mask source only, as above; with
        # --mask-source mask the receipt keeps exactly the keys it has.
        mask_source_block = dict(policy["mask_source"])
        if mask_source_block.get("source") == "head":
            head_counts = {key: 0 for key in _mask_source().HEAD_STAT_KEYS}
            for item in exports:
                block = item.get("mask_source") or {}
                for key in _mask_source().HEAD_STAT_KEYS:
                    head_counts[key] += int((block.get("counts") or {}).get(key, 0))
            mask_source_block["totals"] = head_counts
        receipt["mask_source"] = dict(
            mask_source_block,
            images=len(exports),
            images_with_replacement=sum(
                1
                for item in exports
                if int(((item.get("mask_source") or {}).get("counts") or {}).get("replaced", 0))
            ),
        )
        receipt["guards"]["mask_source"] = policy["mask_source"].get("source")
        receipt["guards"]["mask_source_after_selection"] = bool(policy.get("blend"))
        receipt["guards"]["mask_source_before_emit"] = True
        if policy["mask_source"].get("source") == "head":
            totals["mask_source_s"] = sum(
                item["timing"].get("mask_source_s") or 0.0 for item in exports
            )
    if policy["mask_threshold"] != DEFAULT_MASK_THRESHOLD:
        receipt["mask_threshold"] = {
            "value": policy["mask_threshold"],
            "default": DEFAULT_MASK_THRESHOLD,
            "choices": list(MASK_THRESHOLD_CHOICES),
            "applied_in": "instance_eval.batch.run_batch(mask_threshold=...): the decoder's mask "
            "binarisation, probabilities >= value, before the selection stage",
            "decode_setting_from_flags": True,
        }
    receipt_path = atomic_json(os.path.join(out, "run_receipt.json"), receipt)

    if audit_report["violations"]:
        raise DeliveryRefused(
            "open_audit_violation",
            "results were written but the run read files it must not touch: %r"
            % (audit_report["violations"][:5],),
        )

    result_summary = {
        "status": "delivery_written",
        "out": out,
        "results_dir": results_dir,
        "results": [item["result"] for item in exports],
        "n_results": len(exports),
        "n_reused": len(reused),
        "n_documents": len(exports) + len(reused),
        "resume": bool(resume),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "device": plan["device"],
        "policy": policy,
        "refine_mask": {
            "rule": refine_block["rule"],
            "ink": refine_block["ink"],
            "component_coverage": refine_block["component_coverage"],
            "instances_changed": refine_block["instances_changed"],
            "instances_nonempty_before": refine_block["instances_nonempty_before"],
            "instances_nonempty_after": refine_block["instances_nonempty_after"],
            "fallback_counts": refine_block["fallback_counts"],
        },
        "selection_report": selection_paths,
        "run_receipt": receipt_path,
        "per_image_mean_s": totals["per_image_mean_s"],
        "wall_s": wall_s,
        "non_square_images": sum(1 for item in exports if not item["square"]),
        "open_audit_violations": len(audit_report["violations"]),
    }
    if mask_head_policy["head"] != DEFAULT_MASK_HEAD:
        result_summary["mask_head"] = receipt["mask_head"]
    if policy.get("blend"):
        result_summary["selection_rule"] = receipt["selection_rule"]
    if paste_policy:
        result_summary["mask_paste"] = receipt["mask_paste"]
    if policy.get("ink_box"):
        result_summary["ink_box"] = receipt["ink_box"]
    if policy.get("mask_source"):
        result_summary["mask_source"] = receipt["mask_source"]
    if policy["mask_threshold"] != DEFAULT_MASK_THRESHOLD:
        result_summary["mask_threshold"] = receipt["mask_threshold"]
    return result_summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="cad_mrcnn_delivery",
        description=(
            "PNG -> JSON delivery CLI for the Mask R-CNN direct-instance model. "
            "Writes <out>/results/<stem>.json with exactly the twelve contract fields."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--png", metavar="PATH", help="one input PNG")
    source.add_argument("--inputs", metavar="JSON", help="inputs JSON with a 'samples' list")
    parser.add_argument("--out", metavar="DIR", required=True, help="fresh output directory")
    parser.add_argument("--checkpoint", metavar="PATH", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help="keep instances scoring strictly above this")
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS,
                        help="global per-image cap after the threshold (max %d)" % DECODER_MAX_DETECTIONS)
    parser.add_argument("--per-image-cap", type=int, default=None,
                        help="optional per-image top-N by score after the global cap")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU,
                        help="same-class mask IoU suppression threshold (1.0 disables it)")
    parser.add_argument("--select-rule", choices=list(SELECT_RULES), default=DEFAULT_SELECT_RULE,
                        help="stage-2 rule: plain (default) = same-class mask IoU NMS + strict "
                             "score threshold + caps, the measured path; nmsbox = the frozen "
                             "wide-pool readout (Things only, raw score >= --nms-box-thr, greedy "
                             "same-class box IoU suppression >= --nms-box-iou, then the caps), "
                             "which reads the decoded boxes instead of the masks")
    parser.add_argument("--nms-box-iou", type=float, default=DEFAULT_NMS_BOX_IOU,
                        metavar="FLOAT",
                        help="with --select-rule nmsbox: same-class box IoU suppression threshold "
                             "(inclusive); must be > 0, 0.05..0.20 measured flat")
    parser.add_argument("--nms-box-thr", type=float, default=DEFAULT_NMS_BOX_THR, metavar="FLOAT",
                        help="with --select-rule nmsbox: the raw detection score a candidate must "
                             "reach to be kept (inclusive)")
    parser.add_argument("--blend-model", default=DEFAULT_BLEND_MODEL, metavar="PATH",
                        help="with --select-rule blend: the pickled 26-feature selector "
                             "(delivery/models.py: Blend); default %s" % (DEFAULT_BLEND_MODEL,))
    parser.add_argument("--blend-thr", type=float, default=DEFAULT_BLEND_THR, metavar="FLOAT",
                        help="with --select-rule blend: the blend probability a candidate must reach "
                             "to be kept (inclusive); default %r, the measured operating point"
                             % (DEFAULT_BLEND_THR,))
    parser.add_argument("--blend-tau", type=float, default=DEFAULT_BLEND_TAU, metavar="FLOAT",
                        help="with --select-rule blend: same-class box IoU at or above which a kept box "
                             "suppresses another (inclusive); default %r" % (DEFAULT_BLEND_TAU,))
    parser.add_argument("--blend-cap", type=int, default=DEFAULT_BLEND_CAP, metavar="N",
                        help="with --select-rule blend: per-image cap on the blended survivors, top-N by "
                             "blend probability; default %r" % (DEFAULT_BLEND_CAP,))
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD,
                        help="decoder mask binarisation, probabilities >= this, applied by the lane "
                             "before the selection stage: one of %r (0.5 is the frozen decoder's "
                             "own value)" % (list(MASK_THRESHOLD_CHOICES),))
    parser.add_argument("--mask-paste", choices=list(mask_paste.MASK_PASTE_MODES),
                        default=DEFAULT_MASK_PASTE,
                        help="where the mask sigmoid happens relative to the native-canvas bilinear "
                             "paste: prob (default) = the frozen path, sigmoid inside torchvision, "
                             "resize the probabilities, then >= threshold; logits = resize the mask "
                             "logits and sigmoid afterwards, which puts the 0.5 boundary exactly on "
                             "logit 0 while keeping the ROI peak that the probability paste averages "
                             "under the threshold on 1px strokes")
    parser.add_argument("--mask-logit-pad", type=float, default=None, metavar="FLOAT",
                        help="with --mask-paste logits: the logit used for torchvision's one-cell "
                             "mask padding; the probability path pads with 0.0 (certain background), "
                             "but logit 0 is p=0.5, so <= -1.0 is required (default %r)"
                             % (DEFAULT_MASK_LOGIT_PAD,))
    parser.add_argument("--mask-ink-min-side", type=int, default=DEFAULT_INK_MIN_SIDE,
                        metavar="PX",
                        help="how large a box has to be before its mask stops coming from the "
                             "112x112 head: an instance whose max(box_w, box_h) exceeds this value "
                             "gets mask = PNG ink inside its box widened by %d px (clamped to the "
                             "raster), the head's own mask ignored, so a 1 px stroke that the "
                             "head's bilinear paste turns into a fragment is delivered whole; an "
                             "empty crop keeps the refined mask (default %d = off: the rule is "
                             "only +0.0013 PQ end to end on the 387, below the +0.008 gate, so "
                             "the measured arm 32 is opt-in and 0 reproduces the previous "
                             "documents bit for bit)" % (INK_BOX_PAD, DEFAULT_INK_MIN_SIDE))
    parser.add_argument("--mask-ink-stage", choices=INK_STAGES, default=DEFAULT_INK_STAGE,
                        help="where the ink-in-box rule runs: post (default) replaces the masks "
                             "of the selected instances after the blend selector, so features, "
                             "selection, counts and instance ids are exactly what they were; pre "
                             "runs before the selector, whose boxes and 26 features then read ink "
                             "boxes (measured end to end at +0.000746 PQ; post, which keeps the "
                             "selector untouched, at +0.001306 PQ)")
    parser.add_argument("--mask-source", choices=MASK_SOURCES, default=DEFAULT_MASK_SOURCE,
                        help="where the emitted mask of an instance comes from: mask (default) = the "
                             "refined mask of the 112x112 branch, i.e. the shipped documents; head = "
                             "the 256^2 instance stroke head (delivery/mask_source.py) for every "
                             "selected instance whose box side exceeds --mask-head-min-side, with "
                             "features, selection, counts and instance ids untouched; ink = the "
                             "measured ink-in-box arm (--mask-ink-min-side / --mask-ink-stage)")
    parser.add_argument("--mask-head-min-side", type=int, default=DEFAULT_MASK_HEAD_MIN_SIDE,
                        metavar="PX",
                        help="box side (px) above which --mask-source head replaces an emitted mask "
                             "(default %d)" % (DEFAULT_MASK_HEAD_MIN_SIDE,))
    parser.add_argument("--mask-head-tau", type=float, default=DEFAULT_MASK_HEAD_TAU,
                        metavar="P",
                        help="probability threshold of --mask-source head, applied to the head's "
                             "own window after the bilinear paste (default %.2f)"
                             % (DEFAULT_MASK_HEAD_TAU,))
    parser.add_argument("--mask-head-gate", type=int, default=DEFAULT_MASK_HEAD_GATE,
                        metavar="G",
                        help="ink grey level (0..255) the head's prediction is clamped to; 96 is the "
                             "gate the arm was measured at and is slightly more permissive than the "
                             "composite128 (> 127) the refinement uses (default %d)"
                             % (DEFAULT_MASK_HEAD_GATE,))
    parser.add_argument("--mask-head-ckpt", metavar="PATH",
                        help="stroke-head checkpoint for --mask-source head (default: "
                             "weights/mask_head256_u1.pt when it exists, else "
                             "runs/instance_mask_head256/best.pt)")
    parser.add_argument("--drop-stuff", action="store_true",
                        help="drop every selected instance of the five stuff classes 30..34 after the "
                             "per-image cap and before refinement/emit (survivors are renumbered "
                             "densely); measured 28 head PQ 0.1207 -> 0.1259")
    parser.add_argument("--mask-head", choices=MASK_HEADS, default=DEFAULT_MASK_HEAD,
                        help="spatial resolution of the mask branch: 28 = the shipped head of the "
                             "frozen checkpoint (default, unchanged path), 112 = the fine head "
                             "(conv5_mask reused + conv6/conv7 deconvs, 796452 params) which needs a "
                             "matching checkpoint; measured PQ 0.1207 -> 0.1374 at the same selection")
    parser.add_argument("--mask-chunk", type=int, default=DEFAULT_MASK_CHUNK,
                        help="ROIs per chunk in the 112 head's chunked mask runner (VRAM lever, "
                             "range %d..%d); ignored by the 28 head" % MASK_CHUNK_RANGE)
    parser.add_argument("--refine-mask", choices=REFINE_RULES, default=DEFAULT_REFINE_RULE,
                        help="ground-truth-free mask refinement after selection: a6 = union of the "
                             "PNG ink components whose covered share reaches --refine-ratio (measured "
                             "+65.8%% micro PQ at ratio 0.5), a1 = mask & ink (threshold-robust), "
                             "none = the sealed baseline")
    parser.add_argument("--refine-ink", choices=REFINE_INK_RULES, default=DEFAULT_REFINE_INK,
                        help="ink definition for --refine-mask: nonzero = any RGB channel nonzero on "
                             "the raw RGBA PNG (the documented dataset convention), composite128 = "
                             "min(R,G,B) < 128 over white, the measuring lane's thinner operating point")
    parser.add_argument("--refine-ratio", type=float, default=DEFAULT_REFINE_RATIO,
                        metavar="FLOAT",
                        help="component-inclusion ratio for --refine-mask a6, 0.0..1.0 inclusive: an "
                             "8-connected ink component is kept whole when the share of its pixels "
                             "inside the mask is >= this value.  0.0 keeps every ink component the mask "
                             "touches and is the measured best arm (0.060834794 -> 0.116281992 micro PQ "
                             "on the 387 held-out records at score 0.99, cap 20, composite128 ink); "
                             "1.0 keeps only fully covered components.  No effect on a1/none")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu or cuda[:N]")
    parser.add_argument("--sample-id", default=None, help="with --png: override the sample id")
    parser.add_argument("--group-id", default=None, help="with --png: optional group id")
    parser.add_argument("--selection-report", metavar="PATH", default=None,
                        help="also write the per-image selection report here")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate knobs and paths, print the plan, touch no model and write nothing")
    parser.add_argument("--dump-pool", metavar="DIR", default=None,
                        help="debug only (self-check): write the refined candidate pool "
                             "(post a6 refinement, pre blend selector) as one <stem>.npz per "
                             "image into DIR; inert when unset and never part of the documents")
    parser.add_argument("--head-box", choices=("mask", "decoded"), default="decoded",
                        help="which box the stroke head (--mask-source head) uses as its window: "
                             "decoded (default) = the box-head regression box (B1: +1.6% PQ on "
                             "train_final); mask = bbox of the refined mask (the previous shipped "
                             "behaviour)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run: --out may then already hold the documents "
                             "an earlier attempt finished (results/<sample_id>.json, plus the "
                             "infer_report/results_index/run_receipt/selection_report JSONs and the "
                             "inputs/results/predictions/failures directories).  Every document that "
                             "parses and passes the contract is reused as-is and not re-emitted; the "
                             "rest are produced.  Any other file in --out is refused (exit 2).  "
                             "Each document is written as soon as its image finishes, so a run killed "
                             "mid-batch keeps everything already emitted")
    parser.add_argument("--allow-official-inputs", action="store_true",
                        help="仅用于离线自检 (offline self-check only), the platform run does not need it "
                             "and must not set it: admit input PNGs that sit under a datasets/ directory "
                             "(the official corpora, e.g. datasets/dev_final) by skipping that one "
                             "sealed-path refusal; --out, --checkpoint, --inputs and every non-PNG path "
                             "keep the full refusal")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    out = getattr(args, "out", None)
    try:
        summary = run_delivery(args)
    except DeliveryRefused as exc:
        # a durable failure record only once the run has created its directory,
        # so a pre-flight typo cannot poison a fresh --out
        if out and os.path.isdir(out):
            _write_failure(out, exc.code, exc)
        print(json.dumps({"status": "error", "code": exc.code, "detail": exc.detail}, ensure_ascii=False))
        exit_codes = {"document_violation": 3, "document_fields": 3, "document_score": 3, "open_audit_violation": 4}
        return exit_codes.get(exc.code, 2)
    except Exception as exc:  # unexpected: keep the trace for the operator
        if out and os.path.isdir(out):
            _write_failure(out, "unexpected", exc)
        print(json.dumps({"status": "error", "code": "unexpected", "detail": "%s: %s" % (type(exc).__name__, exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
