"""Canonical score of one run restricted to an explicit sample list.

Same raster scorer as `cad-parent-ship-score01/score_ship.py` (raster 1000x1000,
viewBox 100, per-class Hungarian, strict IoU > 0.5, PQ = SumIoU/(TP+0.5FP+0.5FN)),
but the sample set comes from a subset inputs file so a 96-image cell is scored
against exactly its own 96 ground truths and nothing is counted as missing.

  python tools/score_subset.py <results_dir> --inputs runs/subset96-inputs.json \
      --out runs/ab-prob50/score_canonical.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# delivery/tools/x.py -> repository root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANE = REPO + "/outputs/recognition-direct-instance-20261114-01"
sys.path.insert(0, os.path.join(LANE, "evaluation"))
sys.path.insert(0, os.path.join(REPO, "src"))

from instance_eval import contract as C  # noqa: E402
import cadvec.recognition_raster_data as RD  # noqa: E402

FULL_INPUTS = os.path.join(LANE, "evaluation/grouped_val387_inputs.json")
RASTER = (1000, 1000, (100.0, 100.0))


def score(results_dir, sample_ids=None, inputs=None):
    with open(FULL_INPUTS, encoding="utf-8") as handle:
        samples = json.load(handle)["samples"]
    if inputs:
        with open(inputs, encoding="utf-8") as handle:
            wanted = [row["sample_id"] for row in json.load(handle)["samples"]]
        index = {sample["sample_id"]: sample for sample in samples}
        missing = [sid for sid in wanted if sid not in index]
        if missing:
            raise SystemExit("subset ids not in the held-out list: %r" % missing[:5])
        samples = [index[sid] for sid in wanted]
    if sample_ids:
        keep = set(sample_ids)
        samples = [sample for sample in samples if sample["sample_id"] in keep]

    data = C.raster_data()
    tp = fp = fn = 0
    sum_iou = 0.0
    per_image = []
    started = time.time()
    for k, sample in enumerate(samples):
        sid = sample["sample_id"]
        gt_doc = C.validate_gt_document(
            data.document_from_svg(sample["svg_path"], curve_steps=128, split=None), sid
        )
        stem = os.path.splitext(os.path.basename(sample["png_path"]))[0]
        candidates = [
            os.path.join(results_dir, stem + ".json"),
            os.path.join(results_dir, sid.replace(":", "_") + ".json"),
            os.path.join(results_dir, sid + ".json"),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise FileNotFoundError("no prediction document for %s under %s" % (sid, results_dir))
        with open(path, encoding="utf-8") as handle:
            pred_env = json.load(handle)
        pred_doc, _ = C.unwrap_prediction(pred_env, sid)
        gt_seg = RD._segment_masks(gt_doc, *RASTER)
        pred_seg = RD._segment_masks(pred_doc, *RASTER)

        gt_by_class = {}
        pred_by_class = {}
        for seg in gt_seg:
            gt_by_class.setdefault(int(seg["class_id"]), []).append(seg["mask"])
        for pos, seg in enumerate(pred_seg):
            pred_by_class.setdefault(int(seg["class_id"]), []).append(pos)

        itp = ifp = ifn = 0
        isum = 0.0
        for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
            gt_masks = gt_by_class.get(class_id, [])
            cols = pred_by_class.get(class_id, [])
            ctp = 0
            if gt_masks and cols:
                iou = RD._iou_matrix(gt_masks, [pred_seg[c]["mask"] for c in cols])
                for row, col in RD._hungarian(-iou):
                    value = float(iou[row, col])
                    if value > 0.5:
                        ctp += 1
                        isum += value
            itp += ctp
            ifp += len(cols) - ctp
            ifn += len(gt_masks) - ctp
        tp += itp
        fp += ifp
        fn += ifn
        sum_iou += isum
        per_image.append(
            {
                "sample_id": sid,
                "TP": itp,
                "FP": ifp,
                "FN": ifn,
                "SumIoU": round(isum, 9),
                "pred_segments": len(pred_seg),
                "gt_segments": len(gt_seg),
            }
        )
        if (k + 1) % 25 == 0:
            print(
                "  %d/%d elapsed %.1fs tp=%d fp=%d" % (k + 1, len(samples), time.time() - started, tp, fp),
                flush=True,
            )
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "results_dir": results_dir,
        "images": len(samples),
        "elapsed_s": round(time.time() - started, 1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SumIoU": round(sum_iou, 9),
        "PQ": round(sum_iou / denom, 12) if denom else 0.0,
        "SQ": round(sum_iou / tp, 12) if tp else 0.0,
        "RQ": round(tp / denom, 12) if denom else 0.0,
        "recall": round(tp / (tp + fn), 9) if (tp + fn) else 0.0,
        "precision": round(tp / (tp + fp), 9) if (tp + fp) else 0.0,
        "mean_matched_iou": round(sum_iou / tp, 9) if tp else 0.0,
        "images_emitting_nothing": sum(1 for row in per_image if row["pred_segments"] == 0),
        "per_image": per_image,
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = score(args.results_dir, inputs=args.inputs)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
    if args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
