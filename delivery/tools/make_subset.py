"""Build the fixed stratified 96-image A/B subset and its image-only inputs file.

Strata (deterministic, ordered by sha256 of the sample id, never by outcome):

  non_square      every record whose PNG is not square
  zero_gt         records with no ground-truth segment
  emits_nothing   records whose shipped baseline document carries no primitive
  rest            filled by the hash order to reach the target size

Writes the inputs file for the delivery CLI plus a sidecar naming the strata, so
the subset can be rebuilt and audited without the model.

  python tools/make_subset.py --size 96
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

# delivery/tools/x.py -> repository root (same root as ROOT below)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANE = os.path.join(REPO, "outputs", "recognition-direct-instance-20261114-01")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "runs", "ship387-112-u250-03")
INPUTS_OUT = os.path.join(ROOT, "runs", "subset96-inputs.json")
SIDE_OUT = os.path.join(ROOT, "runs", "subset96.json")
TARGET = 96
# Quotas: the strata the task names are all represented, and the remainder stays
# productive (ground truth present, baseline document non-empty) so a recovered
# mask can actually show up as a TP.  The 65 documents that emit nothing are
# capped below their count for that reason; every one of the 7 non-square
# records is in.
ZERO_GT_QUOTA = 20
EMPTY_QUOTA = 30
SALT = "cad-mrcnn07-maskpaste-subset/v1|"


def png_size(path):
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("not a PNG: %s" % path)
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def order_key(sample_id):
    return hashlib.sha256((SALT + sample_id).encode("utf-8")).hexdigest()


def primitives(results_dir, sample_id, png_path):
    """Primitive count of the shipped document, or None when no document exists."""

    path = os.path.join(results_dir, os.path.basename(png_path)[:-4] + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    return len(document.get("semanticIds") or [])


def main(argv):
    size = TARGET
    if "--size" in argv:
        size = int(argv[argv.index("--size") + 1])

    with open(os.path.join(LANE, "evaluation", "grouped_val387_inputs.json"), encoding="utf-8") as handle:
        samples = json.load(handle)["samples"]
    with open(os.path.join(BASELINE, "score_canonical.json"), encoding="utf-8") as handle:
        per_image = json.load(handle)["per_image"]
    gt = {row["sample_id"]: int(row["gt_segments"]) for row in per_image}

    rows = []
    for sample in samples:
        sid = sample["sample_id"]
        width, height = png_size(sample["png_path"])
        rows.append(
            {
                "sample_id": sid,
                "group_id": sample["group_id"],
                "png_path": sample["png_path"],
                "width": width,
                "height": height,
                "rect": width * height,
                "gt_segments": gt.get(sid),
                "baseline_primitives": primitives(
                    os.path.join(BASELINE, "results"), sid, sample["png_path"]
                ),
            }
        )
    rows.sort(key=lambda row: order_key(row["sample_id"]))

    picked, strata = [], {}
    missing_gt = [row for row in rows if row["gt_segments"] is None]
    if missing_gt:
        raise SystemExit("no baseline row for %d samples" % len(missing_gt))

    def take(name, candidates):
        out = []
        for row in candidates:
            if row["sample_id"] in picked:
                continue
            row["stratum"] = name
            out.append(row)
            picked.append(row["sample_id"])
        strata[name] = [row["sample_id"] for row in out]
        return out

    take("non_square", [row for row in rows if row["width"] != row["height"]])
    take("zero_gt", [row for row in rows if row["gt_segments"] == 0][:ZERO_GT_QUOTA])
    take("emits_nothing", [row for row in rows if row["baseline_primitives"] == 0][:EMPTY_QUOTA])
    take(
        "rest",
        [
            row
            for row in rows
            if row["sample_id"] not in picked
            and row["gt_segments"] > 0
            and (row["baseline_primitives"] or 0) > 0
        ],
    )

    by_id = {row["sample_id"]: row for row in rows}
    # priority order is the stratum order above, so the quota-bearing strata are
    # never the ones truncated; `rest` (hash order) fills the remainder
    chosen = picked[:size]
    if len(chosen) < size:
        raise SystemExit("only %d candidates for a %d subset" % (len(chosen), size))
    leftover = [sid for sid in picked if sid not in chosen]
    if leftover:
        strata["not_picked"] = leftover
        for name in ("non_square", "zero_gt", "emits_nothing", "rest"):
            strata[name] = [sid for sid in strata[name] if sid in chosen]

    subset = [by_id[sid] for sid in chosen]
    inputs = {
        "samples": [
            {"sample_id": row["sample_id"], "group_id": row["group_id"], "png_path": row["png_path"]}
            for row in subset
        ]
    }
    side = {
        "schema": "cad-mrcnn07.subset/1",
        "salt": SALT,
        "size": len(subset),
        "strata": {name: strata[name] for name in strata},
        "counts": {name: len(strata[name]) for name in strata},
        "non_square": [row["sample_id"] for row in subset if row["width"] != row["height"]],
        "zero_gt": sum(1 for row in subset if row["gt_segments"] == 0),
        "baseline_empty_documents": sum(1 for row in subset if row["baseline_primitives"] == 0),
        "gt_segments": sum(int(row["gt_segments"]) for row in subset),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "stratum": row["stratum"],
                "width": row["width"],
                "height": row["height"],
                "gt_segments": row["gt_segments"],
                "baseline_primitives": row["baseline_primitives"],
                "key": order_key(row["sample_id"]),
            }
            for row in subset
        ],
    }
    for path, payload in ((INPUTS_OUT, inputs), (SIDE_OUT, side)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(json.dumps({"inputs": INPUTS_OUT, "sidecar": SIDE_OUT, **side["counts"],
                      "zero_gt": side["zero_gt"], "gt_segments": side["gt_segments"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
