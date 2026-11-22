"""Selector v2 study: data loading, GT-free feature blocks, OOF harness, rules.

Read-only on cad-selector-fine01 (feature shards + picks) and the canonical scorer.
Everything written goes under cad-selector-v2/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np

# Study lanes live outside the repository; point these at local checkouts of the
# lanes to run the study.  Empty by default, so the module imports without them.
MY = os.environ.get("CAD_SELECTOR_V2_DIR", "")
FINE = os.environ.get("CAD_SELECTOR_FINE01_DIR", "")
VER = os.environ.get("CAD_VERIFIER01_DIR", "")
READOUT = os.environ.get("CAD_FINEHEAD_READOUT728_DIR", "")
SHIP = os.environ.get("CAD_PARENT_SHIP_SCORE01_DIR", "")
for _p in (FINE, VER, READOUT, SHIP):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

STUFF = 30
THR = 0.8
CAP = 20
BASE_FEATURES = None  # filled by load_images
BASELINE_PQ = 0.157333015481


# ------------------------------------------------------------------ data
def load_images(pool="u250"):
    ims = []
    for f in sorted(glob.glob(os.path.join(FINE, pool, "shard*of*.pkl"))):
        ims.extend(pickle.load(open(f, "rb"))["images"])
    ims.sort(key=lambda im: im["sid"])
    return ims


def half_of(sid):
    return int(hashlib.sha256(("cad-generalization-op01|" + sid).encode()).hexdigest(), 16) % 2


def inner_half(sid):
    return int(hashlib.sha256(("cad-verifier01-inner|" + sid).encode()).hexdigest(), 16) % 2


# ------------------------------------------------------------------ evaluation
def image_matrix(im):
    return (np.asarray(im["iou_b00"], dtype=np.float64), np.asarray(im["gt_cls"]).astype(np.int64),
            np.asarray(im["pred_cls"]).astype(np.int64))


def eval_keep(images, keep_lists):
    from scipy.optimize import linear_sum_assignment
    tp = fp = fn = 0
    siou = 0.0
    emitted = 0
    for im, idx in zip(images, keep_lists):
        iou, gt_cls, pred_cls = image_matrix(im)
        idx = np.asarray(idx, dtype=np.int64)
        emitted += len(idx)
        if not len(idx):
            fn += len(gt_cls)
            continue
        sub = iou[:, idx]
        pc = pred_cls[idx]
        itp = 0
        isiou = 0.0
        for cls in np.unique(np.concatenate([gt_cls, pc])):
            gi = np.flatnonzero(gt_cls == cls)
            pi = np.flatnonzero(pc == cls)
            if not gi.size or not pi.size:
                continue
            m = sub[np.ix_(gi, pi)]
            r, c = linear_sum_assignment(-m)
            v = m[r, c]
            hit = v > 0.5
            itp += int(hit.sum())
            isiou += float(v[hit].sum())
        tp += itp
        fp += len(idx) - itp
        fn += len(gt_cls) - itp
        siou += isiou
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(TP=tp, FP=fp, FN=fn, SumIoU=round(siou, 9),
                PQ=round(siou / den, 12) if den else 0.0,
                SQ=round(siou / tp, 12) if tp else 0.0,
                RQ=round(tp / den, 12) if den else 0.0,
                recall=round(tp / (tp + fn), 9) if tp + fn else 0.0,
                precision=round(tp / (tp + fp), 9) if tp + fp else 0.0,
                emitted=emitted, images=len(images))


def make_keep(p, im, thr=THR, cap=CAP):
    cls = np.asarray(im["pred_cls"]).astype(np.int64)
    ok = np.flatnonzero((p >= thr) & (cls < STUFF))
    if cap is not None and ok.size > cap:
        ok = ok[np.argsort(-p[ok], kind="stable")[:cap]]
    return ok


def split_metrics(images, keeps):
    out = {"oof": eval_keep(images, keeps)}
    for h in (0, 1):
        sel = [k for im, k in zip(images, keeps) if im["half"] == h]
        sub = [im for im in images if im["half"] == h]
        out["h%d" % h] = eval_keep(sub, sel)
    out["instances"] = int(sum(len(k) for k in keeps))
    return out


# ------------------------------------------------------------------ models
def make_model(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_depth=cfg["max_depth"], max_iter=cfg["max_iter"],
              learning_rate=cfg["lr"], l2_regularization=cfg.get("l2", 1.0),
              random_state=cfg.get("random_state", 0),
              min_samples_leaf=cfg.get("min_samples_leaf", 20))
    if cfg.get("class_weight"):
        kw["class_weight"] = cfg["class_weight"]
    early = cfg.get("early_stopping", "auto")
    if early != "auto":
        kw["early_stopping"] = early
    return HistGradientBoostingClassifier(**kw)


BASE_CFG = dict(max_depth=3, max_iter=200, lr=0.05, l2=1.0,
                min_samples_leaf=20, class_weight="balanced")


def fit_predict(cfg, Xtr, ytr, Xev):
    clf = make_model(cfg)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1], clf


# ------------------------------------------------------------------ feature blocks
def class_prior_table(train_ims):
    """positives / candidates per predicted class, smoothed; global fallback."""
    from collections import defaultdict
    pos, tot = defaultdict(float), defaultdict(float)
    gp = gt = 0.0
    for im in train_ims:
        lab = np.asarray(im["label"]).astype(np.int64)
        cls = np.asarray(im["pred_cls"]).astype(np.int64)
        for c in np.unique(cls):
            m = cls == c
            pos[int(c)] += float(lab[m].sum())
            tot[int(c)] += float(m.sum())
        gp += float(lab.sum())
        gt += float(len(lab))
    g = (gp + 1.0) / (gt + 20.0)
    tab = {}
    for c in range(64):
        tab[c] = (pos.get(c, 0.0) + 10.0 * g) / (tot.get(c, 0.0) + 10.0)
    return tab, g


def _box_iou_mat(boxes):
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = a[:, None] + a[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou


def block_arrays(im, prior_tab=None, prior_global=0.03):
    """GT-free engineered blocks for one image -> dict(name -> (n,k) float64)."""
    n = len(im["scores"])
    scores = np.asarray(im["scores"], dtype=np.float64)
    cls = np.asarray(im["pred_cls"]).astype(np.int64)
    boxes = np.asarray(im["boxes"], dtype=np.float64)
    F = np.asarray(im["feats"], dtype=np.float64)  # 26 base columns, decode order
    feats_model = list(BASE_FEATURES or im["features"])
    col = {name: i for i, name in enumerate(feats_model)}
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    nms = np.asarray(im["nms_keep"], dtype=np.int64)
    in_nms = np.zeros(n, dtype=bool)
    in_nms[nms] = True

    # ---- B1 score-shape within image
    srt = scores[order]
    next_higher = np.zeros(n)
    for k in range(1, n):
        next_higher[order[k]] = scores[order[k - 1]]
    margin_up = next_higher - scores
    mx = float(scores.max()) if n else 1.0
    mu, sd = float(scores.mean()), float(scores.std() + 1e-9)
    b1 = np.column_stack([
        rank.astype(np.float64) / 20.0,
        np.log1p(rank.astype(np.float64)),
        margin_up * 100.0,
        (next_higher - scores) / (next_higher + 1e-6),
        scores / (mx + 1e-9),
        (scores - mu) / sd,
        np.log(np.clip(scores, 1e-6, None)),
        np.log1p(np.maximum(scores - 0.8, 0.0) * 100.0),
    ])

    # ---- B2 box geometry / overlap (box-level, no masks)
    bw = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    bh = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    diag = np.hypot(bw, bh)
    iou = _box_iou_mat(boxes)
    same = (cls[:, None] == cls[None, :])
    ov_same = np.where(same, iou, 0.0)
    max_iou_all = iou.max(axis=1) if n > 1 else np.zeros(n)
    max_iou_same = ov_same.max(axis=1) if n > 1 else np.zeros(n)
    cnt30 = (iou > 0.3).sum(axis=1).astype(np.float64)
    cnt30s = (ov_same > 0.3).sum(axis=1).astype(np.float64)
    # distance to nearest strictly higher-scoring candidate that overlaps it
    dist_dup = np.ones(n)
    order_pos = {int(j): k for k, j in enumerate(order)}
    for i in range(n):
        best = 1e18
        for j in np.flatnonzero(ov_same[i] > 0.3):
            if order_pos[int(j)] < order_pos[i]:
                cx = (boxes[i, 0] + boxes[i, 2] - boxes[j, 0] - boxes[j, 2]) / 2.0
                cy = (boxes[i, 1] + boxes[i, 3] - boxes[j, 1] - boxes[j, 3]) / 2.0
                d = float(np.hypot(cx, cy))
                if d < best:
                    best = d
        dist_dup[i] = 0.0 if best > 1e17 else min(best / 1414.2, 2.0)
    b2 = np.column_stack([
        np.log1p(diag), diag / 1414.2,
        np.abs(np.log((bw + 1.0) / (bh + 1.0))),
        max_iou_all, max_iou_same, cnt30, cnt30s, dist_dup,
        in_nms.astype(np.float64),
        (np.clip(bw, 0, None) * np.clip(bh, 0, None)) / 1e6,
    ])

    # ---- B3 mask morphology ratios (from base columns + refined mask stats)
    area = np.maximum(np.expm1(F[:, col["mask_area"]]), 1.0)
    ncomp = F[:, col["ncomp_total"]]
    nge25 = F[:, col["ncomp_ge25"]]
    perim = np.maximum(F[:, col["mask_perim"]], 1.0)
    nink = np.expm1(np.clip(F[:, col["ink_in_mask"]], 0, 30))
    b3 = np.column_stack([
        ncomp / (area / 100.0 + 1.0),
        nge25 / (area / 100.0 + 1.0),
        area / (perim * perim + 1e-9) * 100.0,
        nink / (area + 1.0),
        np.log1p(ncomp),
        np.log1p(perim),
    ])

    # ---- B4 class prior (training half only)
    if prior_tab is None:
        prior_tab = {}
    pr = np.asarray([prior_tab.get(int(c), prior_global) for c in cls])
    m = cls != 0
    b4 = np.column_stack([
        pr, np.log(pr / (1.0 - pr + 1e-9)),
        (cls == 0).astype(np.float64),
    ])

    # ---- B5 relative rank within own class in the image
    cls_rank = np.zeros(n)
    for c in np.unique(cls):
        m = np.flatnonzero(cls == c)
        if m.size:
            r = np.argsort(-scores[m], kind="stable")
            tmp = np.empty(m.size)
            tmp[r] = np.arange(m.size)
            cls_rank[m] = tmp
    b5 = np.column_stack([
        cls_rank / 10.0,
        np.asarray([np.sum(cls == c) for c in cls], dtype=np.float64) / 20.0,
        (cls_rank == 0).astype(np.float64),
    ])

    # ---- B6 class-level raw-score context inside the image (GT-free)
    max_of_cls = np.zeros(n)
    hi_of_cls = np.zeros(n)
    for c in np.unique(cls):
        m = cls == c
        mx_c = float(scores[m].max())
        max_of_cls[m] = mx_c
        hi_of_cls[m] = float(np.sum(scores[m] > 0.8))
    b6 = np.column_stack([
        max_of_cls,
        scores / (max_of_cls + 1e-9),
        np.log1p(hi_of_cls),
        (max_of_cls >= 0.8).astype(np.float64),
        np.asarray([np.sum(cls == c) for c in cls], dtype=np.float64),
        np.asarray([np.sum(cls == c) for c in cls], dtype=np.float64) / max(1.0, n),
    ])
    return {"B1_score": b1, "B2_box": b2, "B3_morph": b3, "B4_prior": b4,
            "B5_clsrank": b5, "B6_clsctx": b6}


BLOCK_DIMS = {"B1_score": 8, "B2_box": 10, "B3_morph": 6, "B4_prior": 3,
              "B5_clsrank": 3, "B6_clsctx": 6}


_X_CACHE = {}


def build_X(images, blocks=(), prior_tab=None, prior_global=0.03):
    """Return (X, names) with base 26 columns + requested blocks, row order = image order."""
    global BASE_FEATURES
    if BASE_FEATURES is None:
        BASE_FEATURES = list(images[0]["features"])
    key = (id(images), tuple(blocks), id(prior_tab))
    if key in _X_CACHE:
        return _X_CACHE[key]
    names = list(BASE_FEATURES)
    for b in blocks:
        names += ["%s_%d" % (b, i) for i in range(BLOCK_DIMS[b])]
    mats = []
    for im in images:
        parts = [np.asarray(im["feats"], dtype=np.float64)]
        if blocks:
            bl = block_arrays(im, prior_tab, prior_global)
            parts += [bl[b] for b in blocks]
        mats.append(np.column_stack(parts))
    X = np.concatenate(mats) if mats else np.zeros((0, len(names)))
    _X_CACHE[key] = (X, names)
    return X, names


def offsets(images):
    return np.cumsum([0] + [len(im["scores"]) for im in images])


def per_image_probs(images, probs):
    off = offsets(images)
    return [probs[off[k]:off[k + 1]] for k in range(len(images))]


# ------------------------------------------------------------------ OOF driver
def winner_labels(im):
    """1 for the single best-IoU candidate of each GT that is recovered at all (IoU>0.5),
    0 for every other candidate. Aligns training with 'emit each GT once' instead of
    'any candidate with IoU>0.5 is positive' (which teaches duplicate FPs)."""
    gt_of = np.asarray(im["gt_of"]).astype(np.int64)
    iou_best = np.asarray(im["iou_best"], dtype=np.float64)
    lab = np.zeros(len(gt_of), dtype=np.int64)
    for g in np.unique(gt_of[gt_of >= 0]):
        idx = np.flatnonzero(gt_of == g)
        k = int(idx[np.argmax(iou_best[idx])])
        if iou_best[k] > 0.5:
            lab[k] = 1
    return lab


def oof_fit_predict(images, blocks=(), cfg=None, seeds=(0,), labels=None):
    """Strict OOF: for each direction fit on one half, predict the other.
    Returns {sid: prob array}. GT-free at predict time."""
    cfg = dict(BASE_CFG if cfg is None else cfg)
    X, _names = build_X(images, blocks)
    starts = np.cumsum([0] + [len(im["scores"]) for im in images])
    row_of = {im["sid"]: k for k, im in enumerate(images)}
    out = {}
    for d in (0, 1):
        tr = [im for im in images if im["half"] == d]
        ev = [im for im in images if im["half"] == 1 - d]
        idx_tr = np.concatenate([np.arange(starts[row_of[im["sid"]]],
                                           starts[row_of[im["sid"]]] + len(im["scores"]))
                                 for im in tr])
        idx_ev = np.concatenate([np.arange(starts[row_of[im["sid"]]],
                                           starts[row_of[im["sid"]]] + len(im["scores"]))
                                 for im in ev])
        ytr = np.concatenate([np.asarray(im["label"] if labels is None else labels[im["sid"]])
                              .astype(np.int64) for im in tr])
        ps = []
        for s in seeds:
            c = dict(cfg); c["random_state"] = s
            p, _ = fit_predict(c, X[idx_tr], ytr, X[idx_ev])
            ps.append(p)
        p = np.mean(ps, axis=0)
        pos = 0
        for im in ev:
            m = len(im["scores"])
            out[im["sid"]] = p[pos:pos + m]
            pos += m
    return out


_START = {}


def _start(images, im):
    key = id(images)
    if key not in _START:
        pos = {}
        o = 0
        for x in images:
            pos[x["sid"]] = o
            o += len(x["scores"])
        _START[key] = pos
    return _START[key][im["sid"]]


def keeps_from_probs(images, prob_map):
    return [make_keep(prob_map[im["sid"]], im) for im in images]
