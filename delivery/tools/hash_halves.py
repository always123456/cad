"""Split a canonical score by the deterministic hash halves.

  python tools/hash_halves.py runs/<cell>/score_canonical.json [--out halves.json]

Half of a sample is ``int(sha256('cad-generalization-op01|' + sample_id), 16) % 2``,
the split used for every generalisation claim in this project, so a micro number
can be read against the two disjoint subsets that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def half_of(sample_id):
    return int(hashlib.sha256(("cad-generalization-op01|" + sample_id).encode()).hexdigest(), 16) % 2


def summarise(rows):
    tp = sum(row["TP"] for row in rows)
    fp = sum(row["FP"] for row in rows)
    fn = sum(row["FN"] for row in rows)
    sum_iou = sum(row["SumIoU"] for row in rows)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "images": len(rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SumIoU": round(sum_iou, 9),
        "PQ": round(sum_iou / denom, 12) if denom else 0.0,
        "SQ": round(sum_iou / tp, 12) if tp else 0.0,
        "RQ": round(tp / denom, 12) if denom else 0.0,
        "images_emitting_nothing": sum(1 for row in rows if row["pred_segments"] == 0),
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("score_json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    with open(args.score_json, encoding="utf-8") as handle:
        score = json.load(handle)
    rows = score["per_image"]
    halves = {0: [], 1: []}
    for row in rows:
        halves[half_of(row["sample_id"])].append(row)
    report = {
        "score_json": args.score_json,
        "micro": summarise(rows),
        "half_0": summarise(halves[0]),
        "half_1": summarise(halves[1]),
    }
    report["both_halves_same_direction"] = None  # filled by the caller against a baseline
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
