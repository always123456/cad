"""One-image probe: what classes reach the selection / stuff filter?

  python tools/probe_classes.py [stem]
"""

from __future__ import annotations

import collections
import json
import os
import sys

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANE)

import cad_mrcnn_delivery as D  # noqa: E402

STEM = sys.argv[1] if len(sys.argv) > 1 else "2wsIh"
CHECKPOINT = os.environ.get(
    "CAD_FINE_MASK_HEAD_CHECKPOINT",
    "",
) or os.path.join("runs", "fine_mask112_upt250_snapshot.pt")
INPUTS = os.environ.get(
    "CAD_FULL387_INPUTS",
    "",
) or os.path.join("runs", "full387-inputs.json")


def main():
    rows = json.load(open(INPUTS, encoding="utf-8"))
    rows = rows["samples"] if isinstance(rows, dict) else rows
    row = [r for r in rows if os.path.splitext(os.path.basename(r["png_path"]))[0] == STEM][0]
    print("png", row["png_path"])

    loader = D.fine_head_loader(24)
    context = loader(CHECKPOINT, "cuda:0")
    image, height, width = D._batch().default_png_loader(row["png_path"])
    out = D._batch().default_predict(context, image)
    labels = [int(v) - 1 for v in out["labels"]]
    scores = [float(v) for v in out["scores"]]
    probs = out["mask_probabilities"]
    thr = float(out.get("mask_threshold") or 0.5)
    masks = [probs[i] >= thr for i in range(len(labels))]
    print("decoded", len(labels), "classes", sorted(collections.Counter(labels).items()))
    print("scores>=0.99", sum(1 for s in scores if float(s) > 0.99))
    over = [(int(l), round(float(s), 5)) for l, s in zip(labels, scores) if float(s) > 0.99]
    print("classes>=0.99", sorted(collections.Counter(l for l, _ in over).items()))

    sel = D.select_instances(
        masks, labels, scores, score_threshold=0.99, nms_iou=0.5, max_detections=300, per_image_cap=20
    )
    keep = sel["indices"]
    print("counts", sel["counts"])
    print("kept classes", sorted(collections.Counter(labels[i] for i in keep).items()))
    print("kept scores", [round(float(scores[i]), 5) for i in keep])
    empty = [labels[i] for i in keep if not masks[i].any()]
    print("kept with empty mask", len(empty), "of", len(keep), empty)
    for i in keep:
        if not masks[i].any():
            print("   empty keep idx", i, "class", labels[i], "score", round(float(scores[i]), 5),
                  "prob max", float(probs[i].max()))


if __name__ == "__main__":
    main()
