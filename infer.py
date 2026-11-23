#!/usr/bin/env python
"""infer.py — the single offline entry point for 图纸智析 (赛题 4).

The grader points this at a directory of PNGs and reads one 12-field JSON per PNG
out of ``--out``.  Everything below ``delivery/`` is reused, never re-implemented:

    cad_mrcnn_delivery.py   detector + 112x112 mask head + refine + vectorise + emit
    verify_delivery.py      independent contract check (optional, --verify)

Usage
-----
    python infer.py --inputs <DIR|inputs.json|single.png> --out <DIR> \\
                     [--device cpu|cuda:0] [--mode wide|v3|simple]

``--mode`` defaults to ``wide`` -- the platform command carries no ``--mode``, and
``wide`` is the runnable readout (``v3`` would fall back to ``simple`` here because
its selector artefacts are absent, so a default of ``v3`` would throw the wide pool
away silently).  ``--mode simple`` and ``--mode v3`` stay available and unchanged.

``--device cuda`` needs a CUDA-enabled torchvision.  If ``site-packages`` holds the
``+cpu`` wheel (``torchvision::nms`` then has no CUDA kernel and every cuda run dies
in the detector's post-processing), point ``--tv-overlay`` / ``CAD_TV_OVERLAY_DIR`` at
a ``pip install --target`` copy of the matching wheel; the probe below refuses the run
with that message instead of a traceback.  CPU needs nothing extra.

Modes
-----
``simple``   the frozen hard rule: pool at score >= 0.99 capped at 20 per image,
             Stuff dropped.  This is the shipped u250 deliverable (ladder rung
             "0.99 + cap 20", then 112^2 head + a6/composite128 refinement).

``v3``       intended: the wide pool (mask 0.40 / score 0.50 / cap 300) filtered by
             the 26-feature blend selector (``delivery/model_u250_blend.pkl``,
             p >= 0.60, same-class box NMS tau 0.20, cap 20, Stuff >= 30 dropped).
             That selector needs two artefacts this repository does not contain —
             see ``v3_blockers()`` below — so v3 refuses loudly and falls back to
             ``simple`` unless ``--strict`` is given.  The fallback is recorded in
             ``<out>/infer_report.json``; nothing is faked.

``wide``     **the default**: the deliverable readout of that same wide pool, with no
             deleted artefact behind it: raw detection score >= 0.97, greedy
             same-class box IoU NMS (tau 0.10) in score order, 40 per image, Stuff
             >= 30 dropped.  Selection rule = ``--select-rule nmsbox``
             (``nmsbox.py``), which reads the decoded boxes instead of the masks.
             Measured on the 387 held-out records in feature space (v2lib
             ``eval_keep``): 0.166234 micro PQ (a local held-out number, never an
             official score).

Output layout (``<out>`` must be new or empty, as the delivery CLI requires)
---------------------------------------------------------------------------
    <out>/results/<stem>.json   canonical documents (what verify_delivery.py reads)
    <out>/<stem>.json           byte-identical flat mirror (what the platform reads)
    <out>/results_index.json    CLI provenance index
    <out>/run_receipt.json      CLI run receipt
    <out>/selection_report.json per-image candidate/selection counts
    <out>/infer_report.json     this wrapper's own record (mode, fallback, timings)

No network, no external API, no ground truth: the CLI runs with an open-read audit
that fails the run if anything but the checkpoint, the PNGs and the inputs JSON
is opened.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DELIVERY_DIR = os.path.join(REPO_ROOT, "delivery")
DELIVERY_CLI = os.path.join(DELIVERY_DIR, "cad_mrcnn_delivery.py")
VERIFIER = os.path.join(DELIVERY_DIR, "verify_delivery.py")
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "weights", "fine_mask112_u550.pt")
# Stage-2 blend selector (the shipped readout, ``weights/blend_full_20261123.pkl``):
# the same 26-column contract and 6-learner Blend family, re-fit on the FULL train_final
# relabel set (3700 images / 167877 candidates / 28702 GT) with delivered-mask labels under
# the gate-64 decoded head-box pipeline.  Measured end to end on the 300 held-out stems at
# >= 0.70 / tau 0.10 / cap 60: PQ 0.3708 (TP 1086 / FP 699 / FN 1430), up from 0.3636 for
# the previous selector on the same pipeline (+0.0072); OOF retrain alone looked negative
# (delta -0.0025) but end to end flipped positive, so this pkl replaces the shipped one.
# Blend-threshold sweep on the same 300 stems suggested 0.75 (simulated 0.3679 vs 0.3602 at
# 0.70), but the real end-to-end run at 0.75 scored 0.3684 -- BELOW the 0.70 readout of 0.3708.
# The sweep's cached-pool / skip-empty-GT shortcuts skewed the high thresholds, so 0.70 stays.
# The previous shipped pkl stays on disk as ``SUPERSEDED_BLEND_MODEL`` (the older
# ``blend_sameclass_20261119.pkl`` it superseded is no longer on disk).  The frozen
# v3 note named the older ``delivery/model_u250_blend.pkl``, which is still accepted
# when neither deployable copy is present.
DEFAULT_BLEND_MODEL = os.path.join(REPO_ROOT, "weights", "blend_full_20261123.pkl")
SUPERSEDED_BLEND_MODEL = os.path.join(REPO_ROOT, "weights", "blend_head_20261120.pkl")
LEGACY_BLEND_MODEL = os.path.join(DELIVERY_DIR, "model_u250_blend.pkl")
BLEND_MODELS = (DEFAULT_BLEND_MODEL, SUPERSEDED_BLEND_MODEL, LEGACY_BLEND_MODEL)

INPUTS_SCHEMA = "cadvec.direct_instance.inputs.v1"
REPORT_SCHEMA = "cad-infer.report/1"

# Stage-1 policy shared by both modes: the 112^2 head and the measured refinement.
COMMON_FLAGS = [
    "--mask-head", "112",
    "--mask-chunk", "24",
    "--refine-mask", "a6",
    "--refine-ink", "composite128",
    "--refine-ratio", "0.0",
    "--mask-paste", "prob",
    "--max-detections", "300",
]

# Stage-1 policy per mode.  ``blend`` is the wide pool plus the frozen 26-feature
# selector (delivery/features26.py + delivery/models.py, readout: blend probability
# >= 0.70, same-class box IoU >= 0.10, 60 per image); ``wide`` is the same pool with
# the box-only readout; ``v3`` is the pool as the frozen v3 note declared it;
# ``simple`` is the shipped u250 deliverable.
MODE_FLAGS = {
    "blend": [
        "--mask-threshold", "0.35",
        "--score-threshold", "0.50",
        "--per-image-cap", "300",
        "--select-rule", "blend",
        "--blend-thr", "0.70",
        "--blend-tau", "0.10",
        "--blend-cap", "60",
        # The ink-in-box rule (--mask-ink-min-side) is OFF: measured end to end
        # on the 387 held-out records at +0.0013 PQ (post placement), below the
        # +0.008 gate, so the mode keeps the shipped documents and the knob stays
        # an explicit opt-in (evidence/ink-in-box-postselect-20261120.json).
        "--mask-ink-min-side", "0",
        # The 256^2 stroke head replaces the emitted mask of every selected instance
        # (``--mask-head-min-side 0``): 0.302224 micro PQ against 0.247198 shipped on the
        # 387 held-out records (+0.055), TP 856 / FP 425 / FN 2188, SQ 0.763504,
        # selection and instance count untouched (evidence/mask-head-delivery-20261120.json
        # and the min-side/tau sweep: 64/0.30 0.274271, 0/0.30 0.294351, 0/0.50 0.300494,
        # 0/0.70 0.302224, 0/0.85 0.289018 -- the density threshold peaks at 0.70).
        # ``--mask-source mask`` restores the shipped masks.
        "--mask-source", "head",
        "--mask-head-min-side", "0",
        "--mask-head-tau", "0.70",
        # Gate 64 (delivery default 96): measured end to end on the 300-stem
        # train_final subset with the decoded head-box window at +0.0087 PQ
        # (0.3549 -> 0.3636), TP +19 / FP -19 / FN -19, RQ 0.475 -> 0.490.
        # Lowering the ink gate admits darker ink, so the 256^2 stroke head
        # rescues more thin-stroke fragments (see _diag/run_gate64.sh).
        "--mask-head-gate", "64",
    ],
    "v3": [
        "--mask-threshold", "0.40",
        "--score-threshold", "0.50",
        "--per-image-cap", "300",
    ],
    # Wide pool + the box readout that needs no deleted artefact: raw score >= 0.97,
    # same-class box IoU NMS tau 0.10, 40 per image, Stuff dropped.  Measured on the
    # 387 held-out records in feature space (v2lib eval_keep): 0.166234 micro PQ.
    "wide": [
        "--mask-threshold", "0.40",
        "--score-threshold", "0.50",
        "--per-image-cap", "40",
        "--select-rule", "nmsbox",
        "--nms-box-iou", "0.10",
        "--nms-box-thr", "0.97",
    ],
    "simple": [
        "--mask-threshold", "0.50",
        "--score-threshold", "0.99",
        "--per-image-cap", "20",
    ],
}

# The platform runs ``python infer.py --inputs <dir> --out <dir>`` with no --mode, so
# the default must be the runnable readout: ``blend`` (the measured arm, 0.335961 micro
# PQ on the 387 held-out records with --mask-source head).  ``v3`` still falls back to
# ``simple`` here.
DEFAULT_MODE = "blend"

# The stroke-head arm's defaults (delivery --mask-head-*), recorded in the report even
# when the arm is off: the head itself is 256^2, gate 96, tau 0.30 (see
# evidence/instance-mask-head-256-20261120/REPORT.md).
DEFAULT_MASK_HEAD_MIN_SIDE = 64
DEFAULT_MASK_HEAD_TAU = 0.30
DEFAULT_MASK_HEAD_GATE = 96


# Selector v2 (stage 2) as declared, recorded in the report even when unusable.
V3_SELECTOR = {
    "model": os.path.relpath(LEGACY_BLEND_MODEL, REPO_ROOT).replace("\\", "/"),
    "features": "26 unscaled, FEATURES order of cad-verifier01/extract.py",
    "threshold": 0.60,
    "nms_tau": 0.20,
    "per_image_cap": 20,
    "stuff_dropped_from": 30,
}


# --------------------------------------------------------------------------- #
# v3 availability
# --------------------------------------------------------------------------- #
def v3_blockers():
    """Why stage 2 (the 26-feature blend selector) cannot run on unseen PNGs here.

    Checked by looking, not by assuming: each entry names the artefact that was
    searched for and where.  An empty list would mean v3 is runnable.

    The selector itself is no longer a blocker: ``--mode blend`` wires it into this
    entry point (``delivery/features26.py`` + ``delivery/models.py``), so the model
    counts as present when either the deployable ``weights/blend_full_20261123.pkl``,
    the superseded ``weights/blend_head_20261120.pkl`` or the older
    ``delivery/model_u250_blend.pkl`` loads with 26 features.  What still
    blocks ``v3`` is its own stage-2 emitter path (the extractor it names and the
    pre-extracted pool shards), not the model file.
    """
    problems = []

    # 1. the selector itself: any declared path that loads with 26 features is enough
    selector_problems = []
    for candidate in BLEND_MODELS:
        if not os.path.isfile(candidate):
            selector_problems.append("selector model missing: %s" % candidate)
            continue
        try:
            if DELIVERY_DIR not in sys.path:
                sys.path.insert(0, DELIVERY_DIR)
            import pickle  # noqa: PLC0415
            with open(candidate, "rb") as handle:
                blend = pickle.load(handle)
            n_features = int(getattr(blend.parts[0], "n_features_in_", 0))
        except Exception as exc:  # noqa: BLE001 - report, never raise here
            selector_problems.append(
                "selector model unreadable: %s: %s: %s" % (candidate, type(exc).__name__, exc)
            )
            continue
        if n_features != 26:
            selector_problems.append(
                "selector %s expects %d features, the frozen rule says 26" % (candidate, n_features)
            )
            continue
        selector_problems = []
        break
    problems.extend(selector_problems)

    # 2. the feature extractor the selector was fitted on (FEATURES order included)
    candidates = [
        os.environ.get("CAD_VERIFIER_DIR") or "",
        os.path.join(REPO_ROOT, "cad-verifier01"),
        DELIVERY_DIR,
        os.path.join(REPO_ROOT, "tools"),
    ]
    found = [os.path.join(d, "extract.py") for d in candidates if d and os.path.isfile(os.path.join(d, "extract.py"))]
    if not found:
        problems.append(
            "26-feature extractor not shipped (searched cad-verifier01/extract.py under %s); "
            "without FEATURES the feature order is unknown and any recomputation would be invented"
            % ", ".join(d for d in candidates if d)
        )

    # 3. the stage-2 emit path: v2_emit.py reads training shards + a probs pickle
    #    keyed by held-out sample_id, so it has no route for a new image.
    pool_dir = os.environ.get("CAD_POOL_DIR") or os.path.join(DELIVERY_DIR, "pool")
    shards = []
    for base in (pool_dir, os.path.join(pool_dir, "expand")):
        if os.path.isdir(base):
            shards.extend(n for n in os.listdir(base) if n.startswith("shard") and n.endswith(".pkl"))
    if not shards:
        problems.append(
            "stage-2 pool artefacts absent (no shard*of*.pkl under %s); v2_emit.py selects from "
            "pre-extracted shards and has no path for an unseen image" % pool_dir
        )
    if not os.path.isfile(os.path.join(DELIVERY_DIR, "v2_emit.py")):
        problems.append("stage-2 emitter missing: delivery/v2_emit.py")

    return problems


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def collect_directory(root):
    """Every *.png under ``root``, recursively; symlinks are never followed."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            rows.append(path)
    rows.sort(key=lambda p: (os.path.relpath(p, root).casefold(), os.path.relpath(p, root)))
    return rows


def sample_id_for(path):
    return os.path.splitext(os.path.basename(path))[0]


def check_stems(paths):
    seen = {}
    for path in paths:
        stem = sample_id_for(path)
        if stem in seen:
            raise SystemExit(
                "infer.py: two inputs share the output name %r.json: %s vs %s"
                % (stem, seen[stem], path)
            )
        seen[stem] = path


def write_inputs_manifest(paths, target):
    """The delivery CLI's image-only inputs payload: sample_id, group_id, png_path."""
    payload = {
        "schema": INPUTS_SCHEMA,
        "role": "competitor_delivery_image_only",
        "image_only": True,
        "count": len(paths),
        "samples": [
            {"sample_id": sample_id_for(p), "group_id": None, "png_path": os.path.abspath(p)}
            for p in paths
        ],
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    return payload


def resolve_inputs(raw):
    """('dir'|'inputs'|'png', [png paths], sample ids) for the three --inputs shapes."""
    path = os.path.abspath(raw)
    if not os.path.exists(path):
        raise SystemExit("infer.py: --inputs not found: %s" % path)
    if os.path.isdir(path):
        paths = collect_directory(path)
        if not paths:
            raise SystemExit("infer.py: no *.png under %s" % path)
        check_stems(paths)
        return "dir", paths
    if path.lower().endswith(".png"):
        return "png", [path]
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise SystemExit("infer.py: --inputs JSON holds no 'samples' list: %s" % path)
        paths = [str(row.get("png_path") or "") for row in rows]
        missing = [p for p in paths if not p or not os.path.isfile(p)]
        if missing:
            raise SystemExit(
                "infer.py: --inputs JSON points at missing PNGs (png_path resolves against the "
                "current directory): %s" % ", ".join(missing[:4])
            )
        check_stems(paths)
        return "inputs", paths
    raise SystemExit("infer.py: --inputs must be a directory, an inputs JSON or a *.png: %s" % path)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def build_command(mode, args, source, paths, manifest):
    # The child runs with cwd=REPO_ROOT (see ``run_cli``), so every path is handed over
    # absolute: a relative value would be resolved against the package directory and the
    # documents would land somewhere the caller never looks, while the child still exits 0.
    command = [sys.executable, DELIVERY_CLI]
    if source == "png":
        command += ["--png", os.path.abspath(paths[0])]
    else:
        command += ["--inputs", os.path.abspath(manifest)]
    command += ["--out", os.path.abspath(args.out), "--device", args.device]
    command += ["--checkpoint", os.path.abspath(args.checkpoint)]
    command += list(COMMON_FLAGS) + list(MODE_FLAGS[mode])
    if mode == "blend":
        # The model is a flag (default: weights/blend_full_20261123.pkl); the
        # absolute path is passed so the child's cwd cannot change its meaning.
        command += ["--blend-model", os.path.abspath(args.blend_model)]
    ink_min_side = getattr(args, "mask_ink_min_side", None)
    if ink_min_side is not None:
        # The mode table above carries the measured arm (32); an explicit caller
        # value is appended after it and, since the child's parser keeps the last
        # occurrence, wins over the mode's own value (--mask-ink-min-side 0 turns
        # the rule off for an A/B run).
        command += ["--mask-ink-min-side", str(int(ink_min_side))]
    ink_stage = getattr(args, "mask_ink_stage", None)
    if ink_stage is not None:
        # Same last-one-wins trick as above: --mask-ink-stage pre reproduces the
        # pre-selector placement for an A/B run.
        command += ["--mask-ink-stage", str(ink_stage)]
    mask_source = getattr(args, "mask_source", None)
    if mask_source is not None:
        # --mask-source head / ink: appended after the mode table, so the shipped
        # mode stays ``mask`` (nothing appended) and an opt-in arm wins.
        command += ["--mask-source", str(mask_source)]
        for flag, name in (
            ("--mask-head-min-side", "mask_head_min_side"),
            ("--mask-head-tau", "mask_head_tau"),
            ("--mask-head-gate", "mask_head_gate"),
            ("--mask-head-ckpt", "mask_head_ckpt"),
        ):
            value = getattr(args, name, None)
            if value is None:
                continue
            command += [flag, str(value)]
    command += ["--drop-stuff"]
    if getattr(args, "resume", False):
        # Every mode: the child is the one that decides which documents to reuse.
        command += ["--resume"]
    if getattr(args, "allow_official_inputs", False):
        command += ["--allow-official-inputs"]
    for flag, value in args.overrides or []:
        command += [flag, value]
    return command


def child_env(overlay=None):
    """Pipe the delivery CLI through UTF-8 so its Chinese paths survive the console code page.

    ``overlay`` is a directory placed ahead of site-packages on PYTHONPATH: the only way to
    give this venv a CUDA-enabled torchvision without touching the environment offline
    (pip install --target).  PyTorch's own CUDA build is irrelevant if torchvision is the
    +cpu wheel: torchvision::nms and torchvision::roi_align then have no CUDA kernel and
    every CUDA run dies inside the detector's post-processing.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if overlay:
        parts = [overlay] + [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


PROBE = r"""
import json
import torch
import torchvision
out = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "torchvision": torchvision.__version__, "torchvision_path": torchvision.__file__,
       "cuda_available": bool(torch.cuda.is_available()), "device_name": None,
       "nms_cuda": False, "roi_align_cuda": False, "error": None}
try:
    if out["cuda_available"]:
        out["device_name"] = torch.cuda.get_device_name(0)
        boxes = torch.rand(8, 4, device="cuda")
        scores = torch.rand(8, device="cuda")
        torchvision.ops.nms(boxes, scores, 0.5)
        out["nms_cuda"] = True
        rois = [torch.tensor([[0.0, 0.0, 4.0, 4.0]], device="cuda")]
        torchvision.ops.roi_align(torch.rand(1, 1, 8, 8, device="cuda"), rois, 2, 0.5)
        out["roi_align_cuda"] = True
except Exception as exc:  # noqa: BLE001 - the point is to report, not to raise
    out["error"] = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:300])
print(json.dumps(out))
"""


def probe_runtime(overlay, device):
    """What the delivery CLI will actually get on ``device``; also the preflight for cuda."""
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        env=child_env(overlay),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        info = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "detail": "runtime probe failed: %s" % (proc.stderr or proc.stdout)[-500:]}

    if not device.startswith("cuda"):
        info["ok"] = True
        return info
    if not info.get("cuda_available"):
        info["ok"] = False
        info["detail"] = "torch has no usable CUDA device"
        return info
    if not (info.get("nms_cuda") and info.get("roi_align_cuda")):
        info["ok"] = False
        info["detail"] = (
            "torchvision %s at %s is the CPU-only build: torchvision::nms / roi_align have no "
            "CUDA kernel, so a cuda run fails inside the detector's post-processing. Fix it with "
            "--tv-overlay <dir> (a pip install --target copy of the CUDA torchvision wheel built "
            "for torch CUDA %s) or run --device cpu. Probe error: %s"
            % (info.get("torchvision"), info.get("torchvision_path"), info.get("torch_cuda"),
               info.get("error"))
        )
        return info
    info["ok"] = True
    return info


def run_cli(command, cwd, overlay=None):
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=child_env(overlay),
        stdout=subprocess.PIPE,
        # stderr is inherited on purpose: the child's progress lines (one per ten
        # images, plus a line for an image that took over five seconds) must reach
        # the operator while the run is still going.  Captured and merged they
        # would only surface at the end, i.e. after twenty minutes of silence.
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, round(time.time() - started, 3)


def mirror_flat(out_dir, stems):
    """Byte-identical copy of results/<stem>.json to <out>/<stem>.json."""
    written = []
    for stem in stems:
        src = os.path.join(out_dir, "results", stem + ".json")
        dst = os.path.join(out_dir, stem + ".json")
        shutil.copyfile(src, dst)
        written.append(os.path.basename(dst))
    return written


def read_documents(out_dir, stems):
    facts = []
    for stem in stems:
        path = os.path.join(out_dir, "results", stem + ".json")
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, ValueError):
            facts.append({"stem": stem, "missing": True})
            continue
        semantic = doc.get("semanticIds") or []
        instance = doc.get("instanceIds") or []
        facts.append(
            {
                "stem": stem,
                "fields": len(doc),
                "primitives": len(semantic),
                "instances": len({i for s, i in zip(semantic, instance) if 0 <= s <= 29 and i >= 0}),
                "canvas": [doc.get("width"), doc.get("height")],
            }
        )
    return facts


def verify(out_dir):
    proc = subprocess.run(
        [sys.executable, VERIFIER, out_dir],
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    report = None
    try:
        report = json.loads(proc.stdout)
    except ValueError:
        pass
    return proc.returncode, report, proc.stdout


def say(text):
    """Write to stdout even when the console code page cannot encode every byte."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text.encode(encoding, "replace").decode(encoding))
    except (UnicodeError, LookupError):
        stream.write(text.encode("ascii", "replace").decode("ascii"))
    stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Offline PNG -> 12-field JSON entry point; one document per input PNG.",
    )
    parser.add_argument("--inputs", required=True, metavar="PATH",
                        help="directory of *.png (recursive), an inputs JSON with 'samples', or one *.png")
    parser.add_argument("--out", required=True, metavar="DIR",
                        help="output directory; must be new or empty")
    parser.add_argument("--device", default="cpu", metavar="DEV",
                        help="cpu or cuda[:N] (default: cpu)")
    parser.add_argument("--mode", choices=("blend", "v3", "wide", "simple"), default=DEFAULT_MODE,
                        help="default: %s. blend = the wide pool + the frozen 26-feature selector "
                             "(blend probability >= 0.70, same-class box IoU >= 0.10, cap 60, Stuff "
                             "dropped), the measured readout; wide = the same pool with the box-only "
                             "readout (score >= 0.97, box IoU 0.10, cap 40); v3 = the pool as the "
                             "frozen v3 note declared it (falls back to simple when its stage-2 "
                             "emitter artefacts are absent); simple = frozen 0.99/cap20 path"
                             % DEFAULT_MODE)
    parser.add_argument("--blend-model", default=DEFAULT_BLEND_MODEL, metavar="PATH",
                        help="with --mode blend: the pickled 26-feature selector (default: "
                             "weights/blend_full_20261123.pkl)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, metavar="PATH",
                        help="detector + 112^2 mask-head checkpoint (default: weights/fine_mask112_u550.pt)")
    parser.add_argument("--mask-ink-min-side", type=int, default=None, metavar="PX",
                        help="delivery --mask-ink-min-side: an instance whose box side exceeds this "
                             "many pixels gets its mask from the PNG ink inside the box (the 112^2 "
                             "head cannot resolve a stroke that thin); 0 turns the rule off. "
                             "Default: the mode's own value (blend: 0 = off)")
    parser.add_argument("--mask-ink-stage", choices=("pre", "post"), default=None,
                        help="delivery --mask-ink-stage: post (default) applies the ink-in-box rule "
                             "to the selected instances after the blend selector; pre applies it "
                             "before the selector, whose boxes and features then read ink boxes")
    parser.add_argument("--mask-source", choices=("mask", "head", "ink"), default=None,
                        help="delivery --mask-source: mask (default) = the refined 112^2 mask, i.e. "
                             "the shipped documents; head = the 256^2 instance stroke head for "
                             "selected instances above --mask-head-min-side; ink = the ink-in-box arm")
    parser.add_argument("--mask-head-min-side", type=int, default=None, metavar="PX",
                        help="delivery --mask-head-min-side: box side above which --mask-source head "
                             "replaces an emitted mask (delivery default 64)")
    parser.add_argument("--mask-head-tau", type=float, default=None, metavar="P",
                        help="delivery --mask-head-tau: probability threshold of the stroke head "
                             "(delivery default 0.30)")
    parser.add_argument("--mask-head-gate", type=int, default=None, metavar="G",
                        help="delivery --mask-head-gate: ink grey level the head is clamped to "
                             "(delivery default 96)")
    parser.add_argument("--mask-head-ckpt", default=None, metavar="PATH",
                        help="delivery --mask-head-ckpt: stroke-head checkpoint (default: "
                             "weights/mask_head256_u1.pt, else runs/instance_mask_head256/best.pt)")
    parser.add_argument("--tv-overlay", default=None, metavar="DIR",
                        help="directory holding a CUDA-enabled torchvision package, put ahead of "
                             "site-packages for the run (pip install --target, offline). Also read "
                             "from CAD_TV_OVERLAY_DIR. Required for --device cuda when the "
                             "installed torchvision is the +cpu wheel")
    parser.add_argument("--strict", action="store_true",
                        help="with --mode v3: refuse instead of falling back to simple")
    parser.add_argument("--no-flat", action="store_true",
                        help="do not mirror results/<stem>.json to <out>/<stem>.json")
    parser.add_argument("--verify", action="store_true",
                        help="run delivery/verify_delivery.py on --out after the run; with --resume it "
                             "still checks every document in results/, reused ones included")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run instead of refusing a non-empty --out: the "
                             "documents already in <out>/results/ (from an earlier attempt killed "
                             "mid-batch, e.g. by a restart or a platform timeout) are reused as-is and "
                             "only the missing images are produced. The delivery CLI rejects any file "
                             "in --out that the run did not write. Each document is written as soon as "
                             "its image finishes, so an interrupted run keeps everything already done; "
                             "progress lines go to stderr")
    parser.add_argument("--allow-official-inputs", action="store_true",
                        help="仅用于离线自检：--inputs 指向 datasets/ 下的官方语料（如 datasets/dev_final）"
                             "时放行输入 PNG，其余 sealed 护栏（--out/--checkpoint/--inputs、SVG 与 GT）"
                             "不变；平台运行不需要也不应打开。记录了该开关的 run 在 "
                             "infer_report.json 的 guards.allow_official_inputs 留痕")
    parser.add_argument("--set", action="append", nargs=2, metavar=("FLAG", "VALUE"),
                        dest="overrides",
                        help="override one delivery-CLI option, e.g. --set --score-threshold 0.30 "
                             "(debug only; recorded in the report as a policy deviation)")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"cpu|cuda(:\d+)?", args.device or ""):
        raise SystemExit("infer.py: --device must be cpu or cuda[:N], got %r" % args.device)
    if not os.path.isfile(DELIVERY_CLI):
        raise SystemExit("infer.py: delivery CLI missing: %s" % DELIVERY_CLI)
    if not os.path.isfile(args.checkpoint):
        raise SystemExit("infer.py: checkpoint not found: %s" % args.checkpoint)
    if args.mode == "blend" and not os.path.isfile(args.blend_model):
        raise SystemExit("infer.py: --blend-model not found: %s" % args.blend_model)

    overlay = args.tv_overlay or os.environ.get("CAD_TV_OVERLAY_DIR") or None
    if overlay:
        overlay = os.path.abspath(overlay)
        if not os.path.isdir(os.path.join(overlay, "torchvision")):
            raise SystemExit("infer.py: --tv-overlay has no torchvision package: %s" % overlay)
    runtime = probe_runtime(overlay, args.device)
    if not runtime.get("ok"):
        say(json.dumps({"status": "error", "code": "runtime_probe",
                        "detail": runtime.get("detail")}, ensure_ascii=False) + "\n")
        return 2

    out_dir = os.path.abspath(args.out)
    if os.path.isfile(out_dir):
        raise SystemExit("infer.py: --out is a file: %s" % out_dir)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.resume:
        raise SystemExit(
            "infer.py: --out must be empty (the delivery CLI writes into a fresh directory): %s\n"
            "  to continue an interrupted run instead, pass --resume: the documents already in "
            "results/ are reused and only the missing images are produced" % out_dir
        )

    # Freeze every path the child will see to an absolute one: ``--out`` reuses the
    # validated ``out_dir``, and the rest are made absolute here so ``build_command``
    # never forwards a value whose meaning depends on the child's cwd.  ``--inputs`` is
    # normalised before ``resolve_inputs`` so the manifest's ``png_path`` entries (and
    # the report) name absolute paths too.
    args.out = out_dir
    args.inputs = os.path.abspath(args.inputs)
    args.checkpoint = os.path.abspath(args.checkpoint)
    if getattr(args, "blend_model", None):
        args.blend_model = os.path.abspath(args.blend_model)

    mode = args.mode
    fallback = None
    blockers = []
    if mode == "v3":
        blockers = v3_blockers()
        if blockers:
            if args.strict:
                say(json.dumps({"status": "v3_unavailable", "reasons": blockers},
                               ensure_ascii=False) + "\n")
                return 3
            mode = "simple"
            fallback = (
                "v3 (wide pool + 26-feature blend selector) is not runnable on an unseen PNG in "
                "this repository; fell back to simple. Reasons: " + " | ".join(blockers)
            )
            warn = ["infer.py: WARNING: --mode v3 unavailable, using simple."]
            warn += ["  - " + reason for reason in blockers]
            sys.stderr.write("\n".join(warn) + "\n")
            sys.stderr.flush()

    source, paths = resolve_inputs(args.inputs)
    os.makedirs(out_dir, exist_ok=True)

    workdir = tempfile.mkdtemp(prefix="cad-infer-inputs-")
    manifest = None
    try:
        if source != "png":
            manifest = os.path.join(workdir, "inputs.json")
            write_inputs_manifest(paths, manifest)
        command = build_command(mode, args, source, paths, manifest)
        started = time.time()
        proc, wall_s = run_cli(command, REPO_ROOT, overlay)
        say(proc.stdout if proc.stdout.endswith("\n") else proc.stdout + "\n")
        if proc.returncode != 0:
            say(json.dumps({"status": "error", "exit": proc.returncode, "mode": mode},
                           ensure_ascii=False) + "\n")
            return proc.returncode
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # A child that exits 0 but leaves no document behind is a failure, never "ok": the
    # usual cause is an --out the child resolved against its own cwd, so the documents
    # exist but in a directory the caller never looks at.  Decide this before
    # mirror_flat (which would raise) and before --verify (which would read nothing).
    results_dir = os.path.join(out_dir, "results")
    produced = []
    if os.path.isdir(results_dir):
        produced = sorted(n for n in os.listdir(results_dir) if n.lower().endswith(".json"))
    if not produced:
        detail = (
            "the delivery CLI exited 0 but wrote no *.json to %s; it may have written them "
            "elsewhere (pass an absolute --out) or the inputs produced nothing" % results_dir
        )
        say(json.dumps({"status": "error", "code": "no_documents", "out": out_dir,
                        "detail": detail, "mode": mode}, ensure_ascii=False) + "\n")
        sys.stderr.write("infer.py: ERROR: " + detail + "\n")
        sys.stderr.flush()
        return 6

    stems = [sample_id_for(p) for p in paths]
    flat = [] if args.no_flat else mirror_flat(out_dir, stems)
    facts = read_documents(out_dir, stems)

    # --resume bookkeeping: the child's own counts (reused vs produced), read from
    # the index it wrote; absent on a run that produced everything in one go.
    resume_block = {"enabled": bool(args.resume), "counts": None}
    try:
        with open(os.path.join(out_dir, "results_index.json"), encoding="utf-8") as handle:
            resume_block["counts"] = (json.load(handle) or {}).get("counts")
    except (OSError, ValueError):
        pass

    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "mode_requested": args.mode,
        "mode_used": mode,
        "mode_is_default": bool(args.mode == DEFAULT_MODE),
        "mode_default": DEFAULT_MODE,
        "fallback": fallback,
        "v3_blockers": blockers,
        "v3_intended_policy": dict(V3_SELECTOR, stage1_flags=dict(zip(MODE_FLAGS["v3"][::2], MODE_FLAGS["v3"][1::2]))),
        "mode_flags": dict(zip(MODE_FLAGS[mode][::2], MODE_FLAGS[mode][1::2])),
        "guards": {
            "allow_official_inputs": bool(args.allow_official_inputs),
            "allow_official_inputs_scope": "input PNGs under a datasets/ directory only; --out, "
                                           "--checkpoint, --inputs and non-PNG paths keep the "
                                           "sealed-path refusal",
            "allow_official_inputs_purpose": "offline self-check (e.g. --inputs datasets/dev_final); "
                                             "the platform run does not need it and must not set it",
        },
        "device": args.device,
        "runtime": runtime,
        "tv_overlay": overlay,
        "checkpoint": os.path.abspath(args.checkpoint),
        "inputs": {"kind": source, "path": os.path.abspath(args.inputs), "count": len(paths),
                   "stems": stems},
        "command": command,
        "extra_args": [" ".join(pair) for pair in (args.overrides or [])],
        "flat_mirror": flat,
        "resume": resume_block,
        "wall_s": round(time.time() - started, 3),
        "delivery_wall_s": wall_s,
        "documents": facts,
        "documents_written": sum(1 for f in facts if not f.get("missing")),
        "note": "documents are the delivery CLI's own results/ files, copied byte for byte; "
                "no score of ours is quoted anywhere in them",
    }

    if mode == "wide":
        # The run's own selection stage, spelled out where the caller looks first.
        # The CLI records the same block as selection_report.json policy.box_nms;
        # --set overrides win over these defaults and are listed in extra_args.
        wide_flags = dict(zip(MODE_FLAGS["wide"][::2], MODE_FLAGS["wide"][1::2]))
        report["selection"] = {
            "select_rule": wide_flags["--select-rule"],
            "nms_box_iou": float(wide_flags["--nms-box-iou"]),
            "nms_box_thr": float(wide_flags["--nms-box-thr"]),
            "per_image_cap": int(wide_flags["--per-image-cap"]),
            "stuff_dropped_from": 30,
            "rule": "raw score >= nms_box_thr (inclusive), greedy in score-descending order with "
                    "ties in decode order, a kept box suppresses same-class boxes with box IoU >= "
                    "nms_box_iou (inclusive), then top-N by score up to per_image_cap",
            "applied_in": "delivery/cad_mrcnn_delivery.py select_instances -> delivery/nmsbox.py",
        }

    if mode == "blend":
        # The measured readout, spelled out where the caller looks first: the wide
        # pool, the a6 refinement, then the frozen 26-feature selector.  The CLI
        # records the same block as selection_report.json policy.blend; --set
        # overrides win over these defaults and are listed in extra_args.
        blend_flags = dict(zip(MODE_FLAGS["blend"][::2], MODE_FLAGS["blend"][1::2]))
        report["blend_model"] = os.path.abspath(args.blend_model)
        report["selection"] = {
            "select_rule": blend_flags["--select-rule"],
            "blend_thr": float(blend_flags["--blend-thr"]),
            "blend_tau": float(blend_flags["--blend-tau"]),
            "blend_cap": int(blend_flags["--blend-cap"]),
            "pool_per_image_cap": int(blend_flags["--per-image-cap"]),
            "features": "the 26 columns of delivery/features26.py, computed on the refined mask "
                        "(bbox_from_mask box, ink_box_pad2 ink source)",
            "stuff_dropped_from": 30,
            "rule": "blend probability >= blend_thr (inclusive), greedy in probability-descending "
                    "order with ties in decode order, a kept box suppresses same-class boxes with "
                    "box IoU >= blend_tau (inclusive), then top-N by probability up to blend_cap",
            "stage": "after the a6 refinement, before the square-canvas pad and the emitter",
            "applied_in": "delivery/cad_mrcnn_delivery.py DeliveryRun.emit -> _blend_stage -> "
                          "delivery/nmsbox.py select_by_box_nms",
            "mask_ink_min_side": int(
                blend_flags.get("--mask-ink-min-side", 0)
                if getattr(args, "mask_ink_min_side", None) is None
                else args.mask_ink_min_side
            ),
            "mask_ink_stage": str(
                getattr(args, "mask_ink_stage", None)
                or blend_flags.get("--mask-ink-stage", "post")
            ),
            "mask_ink_rule": "an instance whose box side exceeds mask_ink_min_side gets its mask from "
                             "the ink map inside its box (2 px pad, clamped) instead of the 112^2 head; "
                             "post = only the selected instances after the selector (the emitted "
                             "document changes, nothing upstream does), pre = before the selector; "
                             "0 = off, which is the default",
            "mask_source": str(
                getattr(args, "mask_source", None)
                or blend_flags.get("--mask-source", "mask")
            ),
            "mask_source_rule": "mask = the refined 112^2 mask (the shipped documents); head = the "
                                "256^2 instance stroke head (delivery/mask_source.py) for the selected "
                                "instances whose box side exceeds mask_head_min_side (the shipped mode); "
                                "ink = the ink-in-box arm above; the stage runs after selection, so "
                                "features, selection, counts and instance ids are unchanged",
            "mask_head_min_side": int(
                args.mask_head_min_side
                if getattr(args, "mask_head_min_side", None) is not None
                else DEFAULT_MASK_HEAD_MIN_SIDE
            ),
            "mask_head_tau": float(
                args.mask_head_tau
                if getattr(args, "mask_head_tau", None) is not None
                else DEFAULT_MASK_HEAD_TAU
            ),
            "mask_head_gate": int(
                args.mask_head_gate
                if getattr(args, "mask_head_gate", None) is not None
                else DEFAULT_MASK_HEAD_GATE
            ),
            "measured_held_out_pq_local": 0.247198,
            "measured_held_out_pq_local_shipped": 0.274271,
            "measured_held_out_pq_local_shipped_note": "the same 387 records with the shipped mode "
                                                      "(--mask-source head, min side 64, tau 0.30, gate 96): "
                                                      "TP 815 / FP 466 / FN 2229, SQ 0.7277, RQ 0.3769, 1281 "
                                                      "instances as shipped, 208 documents changed, +0.027073 "
                                                      "over the 112^2 masks; evidence/mask-head-delivery-20261120.json",
            "measured_held_out_pq_local_ink_box_off": 0.247198,
            "measured_held_out_pq_local_ink_box_post": 0.248504,
            "measured_held_out_pq_local_ink_box_post_note": "the same 387 records with "
                                                           "--mask-ink-min-side 32 --mask-ink-stage post "
                                                           "(TP 777 / FP 504 / FN 2267, 1281 instances as "
                                                           "shipped, 217 documents changed): +0.001306, short of "
                                                           "the +0.008 gate, so the mode keeps 0; "
                                                           "evidence/ink-in-box-postselect-20261120.json",
            "measured_held_out_pq_local_ink_box": 0.247944,
            "measured_held_out_pq_local_ink_box_note": "the same 387 records with --mask-ink-min-side 32 "
                                                      "in the pre-selector placement (TP 783 / FP 531 / FN 2261, "
                                                      "1314 instances, 220 documents changed): +0.000746, which "
                                                      "is below the +0.008 the rule was asked to clear; "
                                                      "evidence/ink-in-box-delivery-20261120.json",
            "measured_space": "387 held-out records, strict IoU > 0.5, micro PQ (tools/score_local.py); "
                              "a local held-out number, never an official score",
        }

    if args.verify:
        code, verifier_report, raw = verify(out_dir)
        report["verifier"] = {"exit": code, "ok": bool((verifier_report or {}).get("ok")),
                              "errors": (verifier_report or {}).get("errors", [])}
        if verifier_report is None:
            report["verifier"]["raw_tail"] = raw[-2000:]

    with open(os.path.join(out_dir, "infer_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")

    say(json.dumps({
        "status": "ok",
        "out": out_dir,
        "mode": mode,
        "fallback": bool(fallback),
        "documents": report["documents_written"],
        "instances": sum(f.get("instances", 0) for f in facts),
        "wall_s": report["wall_s"],
        "verifier_ok": (report.get("verifier") or {}).get("ok"),
    }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
