"""Selector-v2 on the WIDE pool: blend probabilities + rule cells, feature space.

Reuses cad-selector-v2/{v2lib,models,exp3b_nms}.py read-only; features come from the
existing cad-pool-expand01/expand/shard*of4.pkl (no re-extraction).

  python v2_probs.py            compute OOF blend probs + every cell, write probs.pkl + cells.json
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
import v2lib as L  # noqa: E402
import models as M  # noqa: E402
from exp3b_nms import nms_keep  # noqa: E402

CFG = dict(max_depth=4, max_iter=400, lr=0.05, l2=1.0, min_samples_leaf=100,
           class_weight="balanced", early_stopping=False)
ET_KW = dict(n_estimators=300, min_samples_leaf=20, max_features=0.5)
SEEDS = (0, 1, 2)

# name -> (thr, tau or None, cap)
CELLS = {
    "blend080": (0.80, None, 20),
    "blendnms080": (0.80, 0.20, 20),
    "blendnms075": (0.75, 0.20, 20),
    "blendnms070": (0.70, 0.20, 20),
    "blendnms065": (0.65, 0.20, 20),
    "blendnms060": (0.60, 0.20, 20),
}


def load_images():
    ims = []
    for f in sorted(glob.glob(os.path.join(MY, "expand", "shard*of*.pkl"))):
        ims.extend(pickle.load(open(f, "rb"))["images"])
    ims.sort(key=lambda im: im["sid"])
    return ims


def keeps_for(ims, P, thr, tau, cap):
    out = []
    for im in ims:
        p = np.asarray(P[im["sid"]], dtype=np.float64)
        if tau is None:
            out.append(L.make_keep(p, im, thr, cap))
        else:
            out.append(nms_keep(p, np.asarray(im["boxes"], dtype=np.float64),
                                np.asarray(im["pred_cls"]).astype(np.int64), thr, tau, cap))
    return out


def main():
    t0 = time.time()
    ims = load_images()
    n_cand = int(sum(len(im["scores"]) for im in ims))
    print("images", len(ims), "candidates", n_cand,
          "gt", sum(len(im["gt_cls"]) for im in ims), flush=True)

    make = M.blend_make(0.5, CFG, ET_KW)
    t1 = time.time()
    P = M.oof_generic(ims, make, (), seeds=SEEDS)
    fit_s = time.time() - t1
    print("blend OOF fitted in %.1fs" % fit_s, flush=True)

    pickle.dump(P, open(os.path.join(OUT, "probs_blend.pkl"), "wb"), protocol=4)

    cells = {}
    for name, (thr, tau, cap) in CELLS.items():
        keeps = keeps_for(ims, P, thr, tau, cap)
        pooled = L.eval_keep(ims, keeps)
        row = dict(rule=dict(thr=thr, tau=tau, cap=cap), pooled=pooled)
        for h in (0, 1):
            ev = [im for im in ims if im["half"] == h]
            kk = [k for im, k in zip(ims, keeps) if im["half"] == h]
            row["h%d" % h] = L.eval_keep(ev, kk)
        cells[name] = row
        print("%-12s PQ %.6f TP %4d FP %4d FN %4d inst %5d | h0 %.6f h1 %.6f"
              % (name, pooled["PQ"], pooled["TP"], pooled["FP"], pooled["FN"],
                 pooled["emitted"],
                 row["h0"]["PQ"], row["h1"]["PQ"]), flush=True)

    doc = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               pool="cad-pool-expand01/expand (u550, score 0.50, cap 300)",
               images=len(ims), candidates=n_cand,
               gt=int(sum(len(im["gt_cls"]) for im in ims)),
               model=dict(kind="blend50 HGB(d4/400/lr.05/msl100,no-early)+ET300(msl20,mf0.5)",
                          weights_et=0.5, seeds=list(SEEDS), features="26, unscaled"),
               metric="feature space (v2lib.eval_keep; canonical offset ~+0.00095)",
               fit_seconds=round(fit_s, 1), elapsed_s=round(time.time() - t0, 1),
               cells=cells)
    with open(os.path.join(OUT, "cells.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print("elapsed %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
