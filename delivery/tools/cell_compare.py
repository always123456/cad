"""Where the instances a cell adds come from, image by image.

  python tools/cell_compare.py runs/ab-prob50/score_canonical.json \
      runs/ab-prob40/score_canonical.json [--out compare.json]

Joins two canonical scores on ``sample_id`` and splits the delta by image type:
images that emitted nothing before, images that already emitted, and the
instances added (``pred_segments`` delta) against what they turned into
(TP delta vs FP delta).  A recovered instance that matches a GT shows up as a TP;
one that matches nothing shows up as an FP.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        score = json.load(handle)
    return {row["sample_id"]: row for row in score["per_image"]}


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("cell")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    base = load(args.baseline)
    cell = load(args.cell)
    if set(base) != set(cell):
        raise SystemExit("the two scores cover different samples")

    started, existing, stopped = [], [], []
    delta_tp_new = delta_fp_new = delta_tp_old = delta_fp_old = 0
    for sid, row in cell.items():
        old = base[sid]
        dtp = row["TP"] - old["TP"]
        dfp = row["FP"] - old["FP"]
        if old["pred_segments"] == 0 and row["pred_segments"] > 0:
            started.append({"sample_id": sid, "TP": row["TP"], "FP": row["FP"],
                            "segments": row["pred_segments"]})
            delta_tp_new += dtp
            delta_fp_new += dfp
        elif old["pred_segments"] > 0 and row["pred_segments"] == 0:
            stopped.append(sid)
        elif old["pred_segments"] > 0:
            existing.append({"sample_id": sid, "dTP": dtp, "dFP": dfp,
                             "dsegments": row["pred_segments"] - old["pred_segments"]})
            delta_tp_old += dtp
            delta_fp_old += dfp

    report = {
        "baseline": args.baseline,
        "cell": args.cell,
        "images_started_emitting": len(started),
        "images_stopped_emitting": len(stopped),
        "started": started,
        "stopped": stopped,
        "delta_TP_on_images_that_started": delta_tp_new,
        "delta_FP_on_images_that_started": delta_fp_new,
        "delta_TP_on_images_already_emitting": delta_tp_old,
        "delta_FP_on_images_already_emitting": delta_fp_old,
        "images_with_more_TP": sum(1 for row in existing if row["dTP"] > 0),
        "images_with_more_FP": sum(1 for row in existing if row["dFP"] > 0),
        "images_with_fewer_TP": sum(1 for row in existing if row["dTP"] < 0),
        "existing_detail": existing,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("started", "existing_detail", "stopped")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
