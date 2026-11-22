"""Exp3: selection rules + probability calibration, on the winning OOF probabilities.

Every rule is applied per image and is GT-free at apply time. Honest protocol: for each
direction the rule parameters are chosen on an inner split of the *training* half
(probs from a model fit on that inner-train part only), then applied to the eval half.
The OOF-best parameterisation per family is also reported, clearly flagged optimistic.

  r_thr_cap   p >= thr, then top-cap by p                      (the shipped rule)
  r_mass      k = round(alpha * sum(p)), top-k with p >= floor
  r_count     k = round(beta * #{p >= 0.5}), top-k with p >= floor
  r_nms       greedy by p, same-class box-IoU < tau, p >= thr
  r_pen       greedy argmax p*(1-max_sameclass_iou_to_selected)^gamma, p >= thr

Calibration: decile reliability of the OOF probabilities, isotonic and Platt maps fitted
on the training half, and what each does to the operating point.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2lib as L  # noqa: E402

OUT = os.path.join(L.MY, "results", "exp3_rules.json")
STUFF = L.STUFF


def _iou_sameclass(boxes, cls):
    from v2lib import _box_iou_mat
    m = _box_iou_mat(boxes)
    return np.where(cls[:, None] == cls[None, :], m, 0.0)


def apply_rule(p, boxes, cls, rule, prm):
    n = len(p)
    things = cls < STUFF
    if rule == "r_thr_cap":
        ok = np.flatnonzero((p >= prm["thr"]) & things)
        if ok.size > prm["cap"]:
            ok = ok[np.argsort(-p[ok], kind="stable")[:prm["cap"]]]
        return ok
    if rule == "r_mass":
        k = int(round(prm["alpha"] * float(np.clip(p[things], 0, 1).sum())))
        ok = np.flatnonzero((p >= prm["floor"]) & things)
        if ok.size > k:
            ok = ok[np.argsort(-p[ok], kind="stable")[:k]]
        elif ok.size > prm["cap"]:
            ok = ok[np.argsort(-p[ok], kind="stable")[:prm["cap"]]]
        return ok
    if rule == "r_count":
        k = int(round(prm["beta"] * float(np.sum((p >= 0.5) & things))))
        ok = np.flatnonzero((p >= prm["floor"]) & things)
        if ok.size > k:
            ok = ok[np.argsort(-p[ok], kind="stable")[:k]]
        elif ok.size > prm["cap"]:
            ok = ok[np.argsort(-p[ok], kind="stable")[:prm["cap"]]]
        return ok
    if rule in ("r_nms", "r_pen"):
        ov = _iou_sameclass(boxes, cls)
        order = [int(j) for j in np.argsort(-p, kind="stable") if things[j]]
        sel = []
        cur = p.copy()
        alive = np.ones(n, dtype=bool)
        while order and len(sel) < prm["cap"]:
            if rule == "r_pen":
                cand = [j for j in order if alive[j] and p[j] >= prm["thr"]]
                if not cand:
                    break
                j = max(cand, key=lambda k: cur[k])
            else:
                cand = [j for j in order if p[j] >= prm["thr"]]
                if not cand:
                    break
                j = cand[0]
            sel.append(j)
            alive[j] = False
            order.remove(j)
            if rule == "r_pen":
                cur = cur * np.power(np.clip(1.0 - ov[j], 0.0, 1.0), prm["gamma"])
                cur[j] = -1.0
            else:
                # drop everything that overlaps the accepted one too much
                order = [k for k in order if ov[j, k] < prm["tau"]]
        return np.asarray(sel, dtype=np.int64)
    raise SystemExit("bad rule " + rule)


def eval_rule(ims, probs, rule, prm):
    keeps = []
    for im in ims:
        p = probs[im["sid"]]
        keeps.append(apply_rule(p, np.asarray(im["boxes"], float),
                                np.asarray(im["pred_cls"]).astype(np.int64), rule, prm))
    return keeps


GRIDS = {
    "r_thr_cap": [dict(thr=t, cap=c) for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
                  for c in (12, 20, 32)],
    "r_mass": [dict(alpha=a, cap=32, floor=f) for a in (0.3, 0.5, 0.8, 1.0, 1.5)
               for f in (0.5, 0.8)],
    "r_count": [dict(beta=b, cap=32, floor=f) for b in (0.5, 0.75, 1.0, 1.25, 1.5)
                for f in (0.5, 0.8)],
    "r_nms": [dict(tau=t, thr=h, cap=c) for t in (0.2, 0.3, 0.5, 0.7)
              for h in (0.5, 0.6, 0.8, 0.9) for c in (12, 20, 32)],
    "r_pen": [dict(gamma=g, thr=h, cap=c) for g in (0.5, 1.0, 2.0)
              for h in (0.5, 0.6, 0.8, 0.9) for c in (12, 20, 32)],
}


def inner_probs(ims, blocks, cfg, d):
    """Probs on the inner-val part of training half d, from a model fit on inner-train."""
    tr = [im for im in ims if im["half"] == d]
    itr = [im for im in tr if L.inner_half(im["sid"]) == 0]
    iva = [im for im in tr if L.inner_half(im["sid"]) == 1]
    prior, g = L.class_prior_table(itr)
    Xall, _ = L.build_X(ims, blocks, prior, g)
    starts = np.cumsum([0] + [len(im["scores"]) for im in ims])
    row_of = {im["sid"]: k for k, im in enumerate(ims)}
    idx = {name: np.concatenate([np.arange(starts[row_of[im["sid"]]],
                                           starts[row_of[im["sid"]]] + len(im["scores"]))
                                 for im in part])
           for name, part in (("tr", itr), ("va", iva))}
    y = np.concatenate([np.asarray(im["label"]).astype(np.int64) for im in itr])
    p, _ = L.fit_predict(dict(cfg), Xall[idx["tr"]], y, Xall[idx["va"]])
    out = {}
    pos = 0
    for im in iva:
        m = len(im["scores"])
        out[im["sid"]] = p[pos:pos + m]
        pos += m
    return out


def main(probs_path, blocks, cfg):
    ims = L.load_images("u250")
    P = pickle.load(open(probs_path, "rb"))
    res = dict(probs_path=os.path.basename(probs_path), blocks=list(blocks), cfg=cfg,
               families={})

    # inner probs per direction, needed for the honest choice
    ip = {d: inner_probs(ims, blocks, cfg, d) for d in (0, 1)}

    for rule, grid in GRIDS.items():
        # honest: pick params on the inner split of each direction, apply to eval half
        chosen = {}
        keeps_all = {}
        for d in (0, 1):
            iva = [im for im in ims if im["half"] == d and L.inner_half(im["sid"]) == 1]
            ev = [im for im in ims if im["half"] == 1 - d]
            best = None
            for prm in grid:
                k = eval_rule(iva, ip[d], rule, prm)
                pq = L.eval_keep(iva, k)["PQ"]
                if best is None or pq > best[0]:
                    best = (pq, prm)
            chosen[d] = dict(inner_PQ=best[0], params=best[1])
            keeps_all[d] = (ev, eval_rule(ev, P, rule, best[1]))
        ev0, k0 = keeps_all[0]
        ev1, k1 = keeps_all[1]
        ev_imgs = sorted(ev0 + ev1, key=lambda im: im["sid"])
        kmap = {im["sid"]: k for im, k in zip(ev0, k0)}
        kmap.update({im["sid"]: k for im, k in zip(ev1, k1)})
        keeps = [kmap[im["sid"]] for im in ev_imgs]
        hon = L.eval_keep(ev_imgs, keeps)

        # optimistic: best single parameterisation on the OOF itself
        best_oof = None
        for prm in grid:
            k = eval_rule(ims, P, rule, prm)
            pq = L.eval_keep(ims, k)["PQ"]
            if best_oof is None or pq > best_oof[0]:
                best_oof = (pq, prm)
        res["families"][rule] = dict(
            honest=dict(params_per_direction={str(d): chosen[d] for d in (0, 1)},
                        pooled=hon,
                        instances=int(sum(len(k) for k in keeps))),
            optimistic=dict(params=best_oof[1], PQ=best_oof[0]),
            grid_size=len(grid))
        print("%-10s honest PQ %.6f (TP %d FP %d FN %d inst %d) | optimistic %.6f %s"
              % (rule, hon["PQ"], hon["TP"], hon["FP"], hon["FN"],
                 sum(len(k) for k in keeps), best_oof[0], best_oof[1]), flush=True)

    # ---------------- calibration
    y = {im["sid"]: np.asarray(im["label"]).astype(np.int64) for im in ims}
    dec = []
    edges = np.linspace(0, 1, 11)
    allp = np.concatenate([P[im["sid"]] for im in ims])
    ally = np.concatenate([y[im["sid"]] for im in ims])
    for i in range(10):
        m = (allp >= edges[i]) & (allp < edges[i + 1] if i < 9 else allp <= 1.0)
        if m.sum():
            dec.append(dict(bin=i, lo=round(float(edges[i]), 2), n=int(m.sum()),
                            mean_p=round(float(allp[m].mean()), 6),
                            pos_rate=round(float(ally[m].mean()), 6)))
    cal = dict(reliability=dec, base_rate=round(float(ally.mean()), 6))
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    oof_pq = {}
    for d in (0, 1):
        tr = [im for im in ims if im["half"] == d]
        ev = [im for im in ims if im["half"] == 1 - d]
        ptr = np.concatenate([P[im["sid"]] for im in tr])
        ytr = np.concatenate([y[im["sid"]] for im in tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ptr, ytr)
        pl = LogisticRegression(C=1e6).fit(np.log(np.clip(ptr, 1e-6, 1 - 1e-6) /
                                                 (1 - np.clip(ptr, 1e-6, 1 - 1e-6)))[:, None], ytr)
        for name, fn in (("isotonic", iso.predict),
                         ("platt", lambda q: pl.predict_proba(
                             np.log(np.clip(q, 1e-6, 1 - 1e-6) /
                                    (1 - np.clip(q, 1e-6, 1 - 1e-6)))[:, None])[:, 1])):
            Pev = {im["sid"]: np.asarray(fn(P[im["sid"]]), float) for im in ev}
            keeps = [L.make_keep(Pev[im["sid"]], im, 0.8, 20) for im in ev]
            oof_pq.setdefault(name, []).extend(zip(ev, keeps))
    for name, pairs in oof_pq.items():
        pr = sorted(pairs, key=lambda t: t[0]["sid"])
        cal[name] = dict(fixed_0p8_cap20=L.eval_keep([t[0] for t in pr], [t[1] for t in pr]),
                         instances=int(sum(len(t[1]) for t in pr)))
    cal["uncalibrated_fixed_0p8_cap20"] = L.eval_keep(ims, L.keeps_from_probs(ims, P))
    res["calibration"] = cal
    print("calibration:", {k: (v.get("PQ") if isinstance(v, dict) else None)
                           for k, v in cal.items() if k != "reliability"})
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    path = sys.argv[1]
    meta = json.load(open(os.path.join(L.MY, "results", "exp2_meta.json")))
    main(path, tuple(meta["blocks"]), meta["cfg"])
