"""Exp3b: refine the one rule family that beat the declared operating point.

exp3 showed same-class box NMS (r_nms, tau=0.2) with a *lower* threshold beats plain
thr+cap: NMS removes the duplicate FPs that a low threshold would otherwise buy, so recall
can be raised. Here the (thr, tau, cap) space is searched finer, plus a variant that also
suppresses near-identical boxes across classes (a duplicate GT sometimes gets two class
labels).

Honest protocol as in exp3: rule parameters are chosen per direction on the inner split of
the training half (model refit on inner-train only) and applied to the opposite half; the
pooled number is therefore not tuned on the half it is scored on. OOF-optimal parameters
are printed too, flagged optimistic.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2lib as L  # noqa: E402
from v2lib import _box_iou_mat  # noqa: E402
from exp3_rules import inner_probs  # noqa: E402

OUT = os.path.join(L.MY, "results", "exp3b_nms.json")
STUFF = L.STUFF


def nms_keep(p, boxes, cls, thr, tau, cap, tau_cross=None):
    things = cls < STUFF
    order = [int(j) for j in np.argsort(-p, kind="stable") if things[j] and p[j] >= thr]
    if not order:
        return np.zeros(0, dtype=np.int64)
    ov = _box_iou_mat(boxes)
    same = cls[:, None] == cls[None, :]
    sel = []
    alive = np.ones(len(p), dtype=bool)
    for j in order:
        if len(sel) >= cap:
            break
        if not alive[j]:
            continue
        sel.append(j)
        alive[j] = False
        block = (same[j] & (ov[j] >= tau))
        if tau_cross is not None:
            block |= ov[j] >= tau_cross
        alive &= ~block
        alive[j] = False
    return np.asarray(sel, dtype=np.int64)


def eval_rule(ims, probs, prm):
    return [nms_keep(np.asarray(probs[im["sid"]], float), np.asarray(im["boxes"], float),
                     np.asarray(im["pred_cls"]).astype(np.int64),
                     prm["thr"], prm["tau"], prm["cap"], prm.get("tau_cross"))
            for im in ims]


GRID = [dict(thr=t, tau=u, cap=c)
        for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
        for u in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
        for c in (8, 12, 16, 20, 32)]
GRID_X = [dict(d) | {"tau_cross": x} for d in GRID for x in (0.8, 0.9)]
GRID_X = [d for d in GRID_X if d["thr"] >= 0.6 and d["tau"] >= 0.15]


def main():
    t0 = time.time()
    ims = L.load_images("u250")
    P = pickle.load(open(os.path.join(L.MY, "results", "oof_bestcfg.pkl"), "rb"))
    ip = {d: inner_probs(ims, (), L.CFG_BEST, d) for d in (0, 1)}
    res = {"grid": len(GRID), "grid_x": len(GRID_X), "families": {}}

    for name, grid in (("nms_same", GRID), ("nms_cross", GRID_X)):
        chosen, keeps_all = {}, {}
        for d in (0, 1):
            iva = [im for im in ims if im["half"] == d and L.inner_half(im["sid"]) == 1]
            ev = [im for im in ims if im["half"] == 1 - d]
            best = None
            for prm in grid:
                pq = L.eval_keep(iva, eval_rule(iva, ip[d], prm))["PQ"]
                if best is None or pq > best[0]:
                    best = (pq, prm)
            chosen[d] = dict(inner_PQ=best[0], params=best[1])
            keeps_all[d] = (ev, eval_rule(ev, P, best[1]))
        ev0, k0 = keeps_all[0]
        ev1, k1 = keeps_all[1]
        ev_imgs = sorted(ev0 + ev1, key=lambda im: im["sid"])
        kmap = {im["sid"]: k for im, k in zip(ev0 + ev1, k0 + k1)}
        keeps = [kmap[im["sid"]] for im in ev_imgs]
        hon = L.eval_keep(ev_imgs, keeps)
        surf = []
        for prm in grid:
            m = L.eval_keep(ims, eval_rule(ims, P, prm))
            surf.append(dict(params=prm, PQ=m["PQ"], TP=m["TP"], FP=m["FP"],
                             inst=int(sum(len(k) for k in eval_rule(ims, P, prm)))))
        surf.sort(key=lambda r: -r["PQ"])
        res["families"][name] = dict(
            honest=dict(params_per_direction={str(d): chosen[d] for d in (0, 1)},
                        pooled=hon, instances=int(sum(len(k) for k in keeps))),
            oof_best=surf[:10])
        print("%-10s honest PQ %.6f TP %d FP %d FN %d inst %d"
              % (name, hon["PQ"], hon["TP"], hon["FP"], hon["FN"],
                 sum(len(k) for k in keeps)))
        for d in (0, 1):
            print("   dir %d innerPQ %.6f %s" % (d, chosen[d]["inner_PQ"], chosen[d]["params"]))
        print("   oof-best(optimistic):", surf[0]["PQ"], surf[0]["params"])
        json.dump(res, open(OUT, "w"), indent=1)
    print("elapsed %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
