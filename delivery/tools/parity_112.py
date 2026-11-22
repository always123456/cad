"""Compare a 112x112-head delivery run against the fine-head lane's own cell.

The lane (cad-finehead-readout728) built its cells with ``run_fine_infer.py`` +
``build_cell.py`` and post-processed them with ``strip_stuff.py``; this CLI runs the
same model, the same decode, the same a6/composite128/ratio-0.0 refinement and the
same emitter.  If the two are equivalent, then for every sample:

* the document must carry the same instances (``boxes``, ``obj_cts``),
* the same primitives (``args``/``commands``/``lengths``/``rgb``/``widths``/
  ``layerIds``/``semanticIds``/``instanceIds``) with identical values,
* and the canonical scorer must give the same per-image TP/FP/FN/SumIoU.

Usage:
  python parity_112.py <mine_results> <lane_results> [--score <lane_score.json>]
                       [--stems 2wsIh,Ep4Km,FUyVX] [--json <out.json>]

Exit code 0 only when every compared sample agrees.
"""

from __future__ import annotations

import json
import os
import sys


FIELDS = (
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


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def first_difference(left, right):
    """Return (field, index, left_value, right_value) for the first mismatch."""

    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        return ("fields", None, sorted(left_keys), sorted(right_keys))
    for field in FIELDS:
        a = left.get(field)
        b = right.get(field)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return (field, "len", len(a), len(b))
            for index, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return (field, index, x, y)
        return (field, None, a, b)
    return None


def compare(mine_dir, lane_dir, stems, score_path):
    lane_scores = {}
    if score_path and os.path.exists(score_path):
        for row in load(score_path).get("per_image") or []:
            lane_scores[row["sample_id"]] = row

    rows = []
    ok = True
    for stem in stems:
        mine_path = os.path.join(mine_dir, stem + ".json")
        lane_path = os.path.join(lane_dir, stem + ".json")
        if not os.path.exists(mine_path) or not os.path.exists(lane_path):
            rows.append({"stem": stem, "status": "missing",
                         "mine": os.path.exists(mine_path), "lane": os.path.exists(lane_path)})
            ok = False
            continue
        mine = load(mine_path)
        lane = load(lane_path)
        diff = first_difference(mine, lane)
        row = {
            "stem": stem,
            "status": "identical" if diff is None else "differs",
            "instances_mine": len(mine["boxes"]),
            "instances_lane": len(lane["boxes"]),
            "primitives_mine": len(mine["instanceIds"]),
            "primitives_lane": len(lane["instanceIds"]),
            "classes_mine": sorted({int(box[4]) for box in mine["boxes"]}),
            "empty_args_mine": not mine["args"],
            "empty_args_lane": not lane["args"],
            "first_difference": diff,
        }
        if diff is not None:
            ok = False
        rows.append(row)
    return ok, rows, lane_scores


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    mine_dir, lane_dir = argv[1], argv[2]
    score_path = None
    stems = None
    out_path = None
    rest = argv[3:]
    while rest:
        key = rest.pop(0)
        if key == "--score":
            score_path = rest.pop(0)
        elif key == "--stems":
            stems = [item.strip() for item in rest.pop(0).split(",") if item.strip()]
        elif key == "--json":
            out_path = rest.pop(0)
        else:
            raise SystemExit("unknown argument %r" % key)

    if stems is None:
        stems = sorted(
            os.path.splitext(name)[0] for name in os.listdir(lane_dir) if name.endswith(".json")
        )
    ok, rows, lane_scores = compare(mine_dir, lane_dir, stems, score_path)

    identical = sum(1 for row in rows if row["status"] == "identical")
    print(
        json.dumps(
            {
                "mine": mine_dir,
                "lane": lane_dir,
                "samples": len(rows),
                "identical": identical,
                "differing": len(rows) - identical,
                "instances_mine": sum(row.get("instances_mine") or 0 for row in rows),
                "instances_lane": sum(row.get("instances_lane") or 0 for row in rows),
                "primitives_mine": sum(row.get("primitives_mine") or 0 for row in rows),
                "primitives_lane": sum(row.get("primitives_lane") or 0 for row in rows),
                "verdict": "IDENTICAL" if ok else "MISMATCH",
            },
            indent=1,
        )
    )
    for row in rows:
        if row["status"] != "identical":
            print(json.dumps(row, indent=1))
        elif len(rows) <= 12:
            print(json.dumps(row))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "ok": ok, "lane_scores": lane_scores}, handle, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
