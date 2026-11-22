"""Emit one v2 arm from the wide pool: blend OOF probs + one rule cell -> documents.

  python v2_emit.py <arm> <shard> <nshard>

arm        name of a cell in cells.json / CELLS below (blend080, blendnms070, ...)
Documents land in cad-pool-expand01/expand_v2/<arm>/results/<stem>.json, identical in
shape to the other arms (strip_stuff applied), so score_314.py can score them.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time

import numpy as np

V2 = os.environ.get("CAD_V2_LIB_DIR", "")
MY = os.environ.get("CAD_POOL_DIR", "")
OUT = os.path.join(MY, "expand_v2")
sys.path.insert(0, V2)
from exp3b_nms import nms_keep  # noqa: E402
import v2lib as VL  # noqa: E402
import fl_pool as F  # noqa: E402

K = F.K
SS = F.SS

CELLS = {
    "blend080": (0.80, None, 20),
    "blendnms080": (0.80, 0.20, 20),
    "blendnms075": (0.75, 0.20, 20),
    "blendnms070": (0.70, 0.20, 20),
    "blendnms065": (0.65, 0.20, 20),
    "blendnms060": (0.60, 0.20, 20),
}

# labelled separately in the report: the point the INNER split picked (inner_check.py),
# per hash half; not the pre-declared operating point.
PER_HALF = {
    "blendinner": {0: (0.55, 0.30, 32), 1: (0.60, 0.20, 32)},
}


def rule_for(arm, sid):
    if arm in CELLS:
        return CELLS[arm]
    return PER_HALF[arm][VL.half_of(sid)]


def load_images():
    ims = []
    for f in sorted(glob.glob(os.path.join(MY, "expand", "shard*of*.pkl"))):
        ims.extend(pickle.load(open(f, "rb"))["images"])
    ims.sort(key=lambda im: im["sid"])
    return ims


def main(arm, shard, nshard):
    ims = {im["sid"]: im for im in load_images()}
    P = pickle.load(open(os.path.join(OUT, "probs_blend.pkl"), "rb"))
    pred_dir, sidecar_path = F.POOLS["expand"]
    _items, _maps, idx = F.sidecar_full(sidecar_path)
    out_dir = os.path.join(OUT, arm, "results")
    os.makedirs(out_dir, exist_ok=True)
    pol = F.emit_policy(None, 1e-9, 0.9999)
    samples = K.samples()[shard::nshard]
    tot = dict(files=0, skipped=0, instances=0, absorbed=0)
    t0 = time.time()
    for k, sample in enumerate(samples):
        sid = sample["sample_id"]
        png = sample["png_path"]
        stem = os.path.splitext(os.path.basename(png))[0]
        path = os.path.join(out_dir, stem + ".json")
        if F._valid_doc(path):
            tot["skipped"] += 1
            with open(path, encoding="utf-8") as fh:
                tot["instances"] += len(json.load(fh).get("boxes", []))
            continue
        im = ims[sid]
        thr, tau, cap = rule_for(arm, sid)
        p = np.asarray(P[sid], dtype=np.float64)
        if tau is None:
            cls = np.asarray(im["pred_cls"]).astype(np.int64)
            keep = np.flatnonzero((p >= thr) & (cls < 30))
            if keep.size > cap:
                keep = keep[np.argsort(-p[keep], kind="stable")[:cap]]
        else:
            keep = nms_keep(p, np.asarray(im["boxes"], dtype=np.float64),
                            np.asarray(im["pred_cls"]).astype(np.int64), thr, tau, cap)
        pred_path = K.find_pred(pred_dir, sid, png)
        segs, native, _doc = K.candidate_segments(pred_path)
        masks, classes, scores, _r = K.order_and_score(segs, idx[sid], sid)
        F._NATIVE[sid] = (int(native[0]), int(native[1]))
        out = F.emit_selected(sample, masks, classes, scores, keep, pol)
        doc = SS.strip(out["document"])[0]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        tot["files"] += 1
        tot["instances"] += int(len(doc["boxes"]))
        if (k + 1) % 10 == 0 or k + 1 == len(samples):
            print("[%s %d/%d] %.2f s/img inst=%d" % (arm, k + 1, len(samples),
                  (time.time() - t0) / (k + 1), tot["instances"]), flush=True)
    tot["elapsed_s"] = round(time.time() - t0, 1)
    tot["arm"], tot["shard"], tot["nshard"] = arm, shard, nshard
    print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
