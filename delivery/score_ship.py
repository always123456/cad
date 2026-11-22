"""Parent-owned canonical scorer for the ship-387 CLI output.

Scores an emitted delivery directory against the 387 held-out GT with the canonical
raster scorer: raster 1000x1000, viewBox 100, width 1, per-class Hungarian, strict
IoU > 0.5, every unmatched prediction is FP (including classes with no GT in that
image), PQ = SumIoU / (TP + 0.5*FP + 0.5*FN).

NOTE: the repository path contains non-ASCII characters. It is written with unicode
escape sequences because files authored through this shell are encoded by the console
code page, which corrupts literal CJK. Never put raw CJK into a file this way.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANE = REPO + "/outputs/recognition-direct-instance-20261114-01"
sys.path.insert(0, os.path.join(LANE, "evaluation"))
sys.path.insert(0, os.path.join(REPO, "src"))

from instance_eval import contract as C  # noqa: E402
import cadvec.recognition_raster_data as RD  # noqa: E402

PRESETS = {
    "sealed": os.path.join(LANE, "runs/grouped_expanded_paste_consistent01/predictions"),
    # Historical ship-387 result dirs; supply them through the environment.
    "ship": os.environ.get("CAD_SHIP_DIR", ""),
    "ship2": os.environ.get("CAD_SHIP2_DIR", ""),
    "ship3": os.environ.get("CAD_SHIP3_DIR", ""),
}
BASELINE = {"TP": 156, "FP": 29789, "FN": 2888, "SumIoU": 97.972304487499}


def score(results_dir: str, limit: int | None) -> dict:
    inputs = json.load(open(os.path.join(LANE, "evaluation/grouped_val387_inputs.json"), encoding="utf-8"))
    samples = inputs["samples"]
    if limit:
        samples = samples[:limit]
    data = C.raster_data()
    raster = (1000, 1000, (100.0, 100.0))
    tp = fp = fn = 0
    sum_iou = 0.0
    per_image = []
    started = time.time()
    for k, sample in enumerate(samples):
        sid = sample["sample_id"]
        gt_doc = C.validate_gt_document(
            data.document_from_svg(sample["svg_path"], curve_steps=128, split=None), sid)
        stem = os.path.splitext(os.path.basename(sample["png_path"]))[0]
        candidates = [
            os.path.join(results_dir, stem + ".json"),
            os.path.join(results_dir, sid.replace(":", "_") + ".json"),
            os.path.join(results_dir, sid + ".json"),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise FileNotFoundError("no prediction document for %s under %s (tried %r)"
                                    % (sid, results_dir, [os.path.basename(c) for c in candidates]))
        pred_env = json.load(open(path, encoding="utf-8"))
        pred_doc, _ = C.unwrap_prediction(pred_env, sid)
        gt_seg = RD._segment_masks(gt_doc, *raster)
        pred_seg = RD._segment_masks(pred_doc, *raster)

        gt_by_class: dict[int, list] = {}
        pred_by_class: dict[int, list[int]] = {}
        for seg in gt_seg:
            gt_by_class.setdefault(int(seg["class_id"]), []).append(seg["mask"])
        for pos, seg in enumerate(pred_seg):
            pred_by_class.setdefault(int(seg["class_id"]), []).append(pos)

        itp = ifp = ifn = 0
        isum = 0.0
        for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
            gt_masks = gt_by_class.get(class_id, [])
            cols = pred_by_class.get(class_id, [])
            n_gt = len(gt_masks)
            n_pred = len(cols)
            ctp = 0
            if gt_masks and cols:
                iou = RD._iou_matrix(gt_masks, [pred_seg[c]["mask"] for c in cols])
                for row, col in RD._hungarian(-iou):
                    value = float(iou[row, col])
                    if value > 0.5:
                        ctp += 1
                        isum += value
            itp += ctp
            ifp += n_pred - ctp
            ifn += n_gt - ctp
        tp += itp
        fp += ifp
        fn += ifn
        sum_iou += isum
        per_image.append({"sample_id": sid, "TP": itp, "FP": ifp, "FN": ifn,
                          "SumIoU": round(isum, 9), "pred_segments": len(pred_seg),
                          "gt_segments": len(gt_seg)})
        if (k + 1) % 50 == 0:
            print("  %d/%d elapsed %.1fs tp=%d fp=%d" % (k + 1, len(samples), time.time() - started, tp, fp), flush=True)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "results_dir": results_dir,
        "images": len(samples),
        "elapsed_s": round(time.time() - started, 1),
        "TP": tp, "FP": fp, "FN": fn,
        "SumIoU": round(sum_iou, 9),
        "PQ": round(sum_iou / denom, 12) if denom else 0.0,
        "SQ": round(sum_iou / tp, 12) if tp else 0.0,
        "RQ": round(tp / denom, 12) if denom else 0.0,
        "recall": round(tp / (tp + fn), 9) if (tp + fn) else 0.0,
        "precision": round(tp / (tp + fp), 9) if (tp + fp) else 0.0,
        "per_image": per_image,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    preset, limit, expect, outpath = "ship", None, None, None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--preset":
            preset = args[i + 1]; i += 2
        elif a == "--limit":
            limit = int(args[i + 1]); i += 2
        elif a == "--expect-baseline":
            expect = "baseline"; i += 1
        elif a == "--out":
            outpath = args[i + 1]; i += 2
        else:
            raise SystemExit("unknown arg %r" % a)
    result = score(PRESETS[preset], limit)
    res = {k: v for k, v in result.items() if k != "per_image"}
    print(json.dumps(res, indent=2))
    if expect == "baseline":
        bad = [k for k in BASELINE if abs(float(result[k]) - BASELINE[k]) > 1e-6]
        print("BASELINE_MATCH" if not bad else "BASELINE_MISMATCH %r" % bad)
        if bad:
            sys.exit(1)
    if outpath:
        json.dump(result, open(outpath, "w", encoding="utf-8"), indent=2)
