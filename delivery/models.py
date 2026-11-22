"""Model factories + generic OOF/inner-split probability computation.

A model spec is a callable make(seed) -> estimator with fit/predict_proba. HGB keeps the
shipped protocol; ExtraTrees is added for diversity; a blend averages the two probability
vectors (that average, not a stacked meta-model, is what shipped experiments showed works).
"""
from __future__ import annotations

import numpy as np

import v2lib as L


def hgb_make(cfg):
    def mk(seed):
        c = dict(cfg)
        c["random_state"] = seed
        return L.make_model(c)
    return mk


def et_make(n_estimators=300, min_samples_leaf=20, max_features=0.5):
    def mk(seed):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    class_weight="balanced_subsample",
                                    max_features=max_features, n_jobs=6, random_state=seed)
    return mk


class Blend:
    def __init__(self, parts, weights):
        self.parts, self.weights = parts, weights

    def fit(self, X, y):
        for p in self.parts:
            p.fit(X, y)
        return self

    def predict_proba(self, X):
        acc = None
        for w, p in zip(self.weights, self.parts):
            q = p.predict_proba(X)[:, 1] * w
            acc = q if acc is None else acc + q
        s = np.clip(acc, 0.0, 1.0)
        return np.stack([1.0 - s, s], axis=1)


def blend_make(w_et, cfg, et_kw=None):
    et_kw = et_kw or {}

    def mk(seed):
        return Blend([hgb_make(cfg)(seed), et_make(**et_kw)(seed)], [1.0 - w_et, w_et])
    return mk


def _row_index(ims):
    starts = np.cumsum([0] + [len(im["scores"]) for im in ims])
    row_of = {im["sid"]: k for k, im in enumerate(ims)}
    return starts, row_of


def _rows(ims, starts, row_of, part):
    return np.concatenate([np.arange(starts[row_of[im["sid"]]],
                                     starts[row_of[im["sid"]]] + len(im["scores"]))
                           for im in part])


def _unflatten(part, p):
    out, pos = {}, 0
    for im in part:
        n = len(im["scores"])
        out[im["sid"]] = p[pos:pos + n]
        pos += n
    return out


def fit_predict_parts(ims, X, tr, va, make, seeds=(0, 1, 2)):
    starts, row_of = _row_index(ims)
    itr, iva = _rows(ims, starts, row_of, tr), _rows(ims, starts, row_of, va)
    y = np.concatenate([np.asarray(im["label"]).astype(np.int64) for im in tr])
    acc = None
    for s in seeds:
        m = make(s)
        m.fit(X[itr], y)
        p = m.predict_proba(X[iva])[:, 1]
        acc = p if acc is None else acc + p
    return _unflatten(va, acc / len(seeds))


def oof_generic(ims, make, blocks=(), seeds=(0, 1, 2)):
    prior, g = L.class_prior_table(ims)
    X, _ = L.build_X(ims, blocks, prior, g)
    P = {}
    for d in (0, 1):
        tr = [im for im in ims if im["half"] == d]
        va = [im for im in ims if im["half"] == 1 - d]
        P.update(fit_predict_parts(ims, X, tr, va, make, seeds))
    return P


def inner_generic(ims, make, d, blocks=(), seeds=(0, 1, 2)):
    """Probs on the inner-val part of half d, from a model fitted on that half's inner-train."""
    prior, g = L.class_prior_table(ims)
    X, _ = L.build_X(ims, blocks, prior, g)
    tr = [im for im in ims if im["half"] == d]
    itr = [im for im in tr if L.inner_half(im["sid"]) == 0]
    iva = [im for im in tr if L.inner_half(im["sid"]) == 1]
    return fit_predict_parts(ims, X, itr, iva, make, seeds)
