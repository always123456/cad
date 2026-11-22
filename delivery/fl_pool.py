"""Refit the learned candidate selector on the fine-head (112^2 mask) candidate pool.

CPU only.  Nothing outside this lane is written.

Reused verbatim (read-only, imported):
  cad-verifier01/extract.py      26 GT-free features, B00 refinement, GT IoU (the protocol)
  cad-verifier01/verify2.py      group-aware 2-fold CV, HGB(3,200,0.05,l2=1), grid 0.2..0.9 x 12/20/32
  cad-finehead-readout728/kcommon.py   candidate loader (shipped rasteriser) + shipped
                                select->refine->pad->vectorise emit path (a6/composite128/0.0)
  cad-finehead-readout728/strip_stuff.py  document-level stuff/background drop
  cad-parent-ship-score01/score_ship.py   canonical scorer

Sub-commands
  check                      fidelity micro-checks before any long run
  pass   <pool> <shard> <n>  per-image features/labels (extract) + baseline emission
  cv     <pool>              strict OOF picks (hgb), writes picks_<pool>.pkl
  emit   <pool> <arm> <shard> <n>   emit selected candidates -> <pool>/<arm>/results
  score  <pool> <arm> [...]  canonical score + per-half aggregate
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

# Copy of cad-selector-fine01/fl.py (sha256 D668BBD7...CC69). Only this line and the
# POOLS entry below differ: the frozen lane is imported, never written.
# Lane roots are machine-specific: pass them through the environment, never in code.
MY = os.environ.get("CAD_POOL_DIR", "")
READOUT = os.environ.get("CAD_READOUT_DIR", "")
VER = os.environ.get("CAD_VERIFIER_DIR", "")
MASKR = os.environ.get("CAD_MASKR_DIR", "")
SHIP = os.environ.get("CAD_SHIP_SCORE_DIR", "")

for _p in (MY, READOUT, VER, SHIP):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import kcommon as K          # noqa: E402  (imports the shipped delivery package + scorer)
import extract as E          # noqa: E402
import verify2 as V          # noqa: E402
import strip_stuff as SS     # noqa: E402

P, F, C, RD, D = E.P, E.F, E.C, E.RD, K.D
FEATURES = E.FEATURES
STUFF = 30
CAP_MAXDET = 300
POL_BASELINE = None  # built lazily via K.policy

POOLS = {
    "u250": (os.path.join(MASKR, "predictions_fine_u250"),
             os.path.join(MASKR, "scores_sidecar_fine_u250.json")),
    "u728": (os.path.join(READOUT, "predictions_fine_u728"),
             os.path.join(READOUT, "scores_sidecar_fine_u728.json")),
    # widened pool: score 0.50 / cap 300 on the u550 fine checkpoint (run_pool.py)
    "expand": (os.path.join(MY, "pool", "predictions"),
               os.path.join(MY, "pool", "sidecar.json")),
}


# ---------------------------------------------------------------- sidecar / masks
def sidecar_full(path):
    """(items, maps, idx): extract.py's sidecar dict, per-instance records, and
    kcommon's (score, prediction_index) table, all keyed by sample_id."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items, maps, idx = {}, {}, {}
    for item in data:
        sid = item["sample_id"]
        items[sid] = item
        tab = {}
        for inst in item["instances"]:
            tab[(int(inst["class_id"]), int(inst["instance_id"]))] = inst
        maps[sid] = tab
        idx[sid] = {k: (float(v["score"]), int(v["prediction_index"])) for k, v in tab.items()}
    return items, maps, idx


def nms_kept(raw_masks, classes, nms_iou=0.5):
    """Decode-order indices surviving the delivery's same-class mask-IoU NMS.

    Byte-for-byte the arithmetic of cad_mrcnn_delivery.select_instances (strict >).
    """
    spans = [D._mask_span(m) for m in raw_masks]
    areas = [int(np.count_nonzero(m)) for m in raw_masks]
    kept = []
    for i, m in enumerate(raw_masks):
        sp = spans[i]
        sup = False
        if sp is not None and areas[i] > 0:
            for o in kept:
                osp = spans[o]
                if osp is None or areas[o] <= 0 or classes[o] != classes[i]:
                    continue
                inter = D._roi_intersection(m, sp, raw_masks[o], osp)
                if inter == 0:
                    continue
                union = areas[i] + areas[o] - inter
                if union > 0 and (inter / union) > nms_iou:
                    sup = True
                    break
        if not sup:
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def shipped_keep(scores, nms_idx, thr=0.99, cap=20, maxdet=CAP_MAXDET):
    """The unlearned baseline: NMS survivors -> score > thr -> maxdet cap -> per-image cap."""
    over = [int(i) for i in nms_idx if float(scores[i]) > thr]
    capped = over[:maxdet]
    if cap is not None and len(capped) > cap:
        top = set(sorted(capped, key=lambda i: (-float(scores[i]), i))[:cap])
        capped = [i for i in capped if i in top]
    return np.asarray(capped, dtype=np.int64)


def emit_policy(per_image_cap, score_threshold, nms_iou):
    return K.policy(per_image_cap, 0.0, ink="composite128", rule="a6",
                    score_threshold=score_threshold, nms_iou=nms_iou,
                    max_detections=CAP_MAXDET)


# ---------------------------------------------------------------- one image
def one_image(sample, pred_dir, sidecars, per, prior, baseline_dir, pol_emit):
    """Features/labels for every candidate (DECODE order) + this image's baseline emit.

    extract.py walks the document in raster order; the delivery path emits in decode
    order (prediction_index). The u250 sidecar's prediction_index is not always
    monotone, so the two orders disagree on those images. Everything is returned in
    decode order: features are permuted, and NMS runs on the permuted masks.
    """
    sid = sample["sample_id"]
    png = sample["png_path"]
    items, maps, idx = sidecars
    pred_path = K.find_pred(pred_dir, sid, png)
    segs, native, _doc = K.candidate_segments(pred_path)
    n = len(segs)
    keys = [(int(s["class_id"]), int(s["instance_id"])) for s in segs]
    pi = np.asarray([idx[sid][k][1] for k in keys], dtype=np.int64)
    perm = np.argsort(pi, kind="stable")
    reordered = not np.array_equal(perm, np.arange(n))

    masks_r = [np.asarray(s["mask"], dtype=bool) for s in segs]
    classes_r = [int(s["class_id"]) for s in segs]

    fake = dict(sample)
    fake["prediction"] = pred_path
    rec = E.extract_image(fake, items, {sid: {"nms_keep": []}}, per, prior)
    if len(rec["scores"]) != n:
        raise RuntimeError("%s candidate count %d != %d" % (sid, n, len(rec["scores"])))
    if reordered:
        rec["feats"] = np.asarray(rec["feats"])[perm]
        rec["boxes"] = np.asarray(rec["boxes"])[perm]
        for key in ("label", "gt_of", "iou_best", "pred_cls", "scores"):
            rec[key] = np.asarray(rec[key])[perm]
        rec["iou_b00"] = np.asarray(rec["iou_b00"])[:, perm]
        rec["nms_keep"] = []
    masks = [masks_r[i] for i in perm]
    classes = [classes_r[i] for i in perm]
    scores = np.asarray(rec["scores"], dtype=np.float64)
    check_masks, check_classes, check_scores, _r = K.order_and_score(segs, idx[sid], sid)
    if not np.allclose(scores, np.asarray(check_scores, float)):
        raise RuntimeError("%s candidate scores differ between loaders" % sid)
    if classes != list(check_classes):
        raise RuntimeError("%s candidate classes differ between loaders" % sid)
    del check_masks

    nms_idx = nms_kept(masks, classes)
    rec["nms_keep"] = nms_idx.tolist()

    out = None
    keep_base = shipped_keep(scores, nms_idx, 0.99, 20)
    if baseline_dir:
        out = emit_selected(sample, masks, classes, scores, keep_base, pol_emit)
        stem = os.path.splitext(os.path.basename(png))[0]
        doc = SS.strip(out["document"])[0]
        with open(os.path.join(baseline_dir, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
    rec["baseline_instances"] = int(len(keep_base))
    rec["reordered"] = bool(reordered)
    return rec, out, masks, classes, scores, nms_idx


def emit_selected(sample, masks, classes, scores, keep, pol):
    sel_m = [masks[int(i)] for i in keep]
    sel_c = [int(classes[int(i)]) for i in keep]
    sel_s = [float(scores[int(i)]) for i in keep]
    out, rec = K.emit(sel_m, sel_c, sel_s, _native(sample), sample["png_path"], pol)
    selected = int(rec["counts"]["selected"])
    if selected != len(keep):
        # The frozen emit path may legitimately absorb a kept candidate: an empty
        # post-refine mask vectorises to nothing, and a duplicate that survives the
        # pool's own NMS (identical mask twice in the delivered document) is merged
        # by the policy's NMS.  Both are the delivery's own filters, not a leak.
        cnt = rec["counts"]
        explained = (int(cnt.get("nms_suppressed", 0)) + int(cnt.get("threshold_dropped", 0))
                     + int(cnt.get("cap_dropped", 0)) + int(cnt.get("per_image_cap_dropped", 0))
                     + len([m for m in sel_m if int(np.asarray(m).sum()) == 0]))
        if explained == len(keep) - selected:
            print("emit absorbed %s: %d of %d kept candidates (counts %s)"
                  % (sample["sample_id"], len(keep) - selected, len(keep),
                     json.dumps({k: cnt.get(k) for k in ("nms_suppressed", "threshold_dropped",
                                                         "cap_dropped", "per_image_cap_dropped")},
                                sort_keys=True)), flush=True)
        else:
            raise RuntimeError("%s: emit kept %d of %d (selection not bypassed; explained %d)"
                               % (sample["sample_id"], selected, len(keep), explained))
    return out


def _native(sample):
    # native size is read from the prediction envelope header by the caller path;
    # K.emit needs it only for the pad, and candidate_segments already parsed it.
    return _NATIVE[sample["sample_id"]]


_NATIVE = {}


# ---------------------------------------------------------------- pass
def cmd_pass(pool, shard, nshard):
    pred_dir, sidecar_path = POOLS[pool]
    sidecars = sidecar_full(sidecar_path)
    per, prior = E.load_colour()
    samples = K.samples()[shard::nshard]
    base_dir = os.path.join(MY, pool, "baseline", "results")
    os.makedirs(base_dir, exist_ok=True)
    pol_emit = emit_policy(None, 1e-9, 0.9999)
    out = []
    t0 = time.time()
    for k, sample in enumerate(samples):
        sid = sample["sample_id"]
        png = sample["png_path"]
        pred_path = K.find_pred(pred_dir, sid, png)
        _segs, native, _doc = K.candidate_segments(pred_path)
        _NATIVE[sid] = (int(native[0]), int(native[1]))
        rec, _o, _m, _c, _s, nms_idx = one_image(sample, pred_dir, sidecars, per, prior,
                                                 base_dir, pol_emit)
        rec["nms_keep"] = nms_idx
        out.append(rec)
        if (k + 1) % 5 == 0 or k + 1 == len(samples):
            rate = (time.time() - t0) / (k + 1)
            print("[%d/%d] %.2f s/img eta %.1f min" % (k + 1, len(samples), rate,
                  rate * (len(samples) - k - 1) / 60.0), flush=True)
    path = os.path.join(MY, pool, "shard%dof%d.pkl" % (shard, nshard))
    with open(path, "wb") as fh:
        pickle.dump(dict(images=out, features=list(FEATURES), shard=shard, total=nshard,
                         elapsed=round(time.time() - t0, 1)), fh, protocol=4)
    print("written", len(out), "images", round((time.time() - t0) / 60.0, 2), "min", flush=True)


def load_images(pool):
    ims = []
    for f in sorted(glob.glob(os.path.join(MY, pool, "shard*of*.pkl"))):
        ims.extend(pickle.load(open(f, "rb"))["images"])
    ims.sort(key=lambda im: im["sid"])
    return ims


# ---------------------------------------------------------------- cv
def cmd_cv(pool):
    images = load_images(pool)
    print("images", len(images), "candidates", sum(len(im["scores"]) for im in images),
          "gt", sum(len(im["gt_cls"]) for im in images), flush=True)
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               pool=pool, images=len(images),
               candidates=int(sum(len(im["scores"]) for im in images)),
               gt=int(sum(len(im["gt_cls"]) for im in images)),
               features=list(FEATURES), grid=dict(THRS=V.THRS, CAPS=V.CAPS),
               operating_point="fixed 0.8 / cap 20 (pre-declared; inner CV reported only)",
               variants={})
    picks, picks_inner = {}, {}
    for d in (0, 1):
        tr = [im for im in images if im["half"] == d]
        ev = [im for im in images if im["half"] == 1 - d]
        r = V._run(images, tr, ev, "hgb")
        keep80 = [V.make_keep(p, im, 0.8, 20) for p, im in zip(r["probs"], r["ev"])]
        for im, keep in zip(r["ev"], keep80):
            picks[im["sid"]] = keep
        for im, keep in zip(r["ev"], r["keeps"]):
            picks_inner[im["sid"]] = keep
        ev80 = V.eval_keep(r["ev"], keep80)
        out["variants"]["dir%d_train%d_eval%d" % (d, d, 1 - d)] = dict(
            threshold=r["chosen"]["threshold"], cap=r["chosen"]["cap"],
            inner_PQ=r["chosen"]["inner_PQ"], pool_eval=r["eval"],
            fixed_0p8_cap20=ev80, inner_table=r["inner_table"],
            records=len(r["ev"]))
        print("dir%d train h%d -> eval h%d: inner chose thr %.1f cap %d (inner PQ %.12f; "
              "eval PQ %.12f) | fixed 0.8/20 eval PQ %.12f TP %d FP %d FN %d"
              % (d, d, 1 - d, r["chosen"]["threshold"], r["chosen"]["cap"],
                 r["chosen"]["inner_PQ"], r["eval"]["PQ"], ev80["PQ"],
                 ev80["TP"], ev80["FP"], ev80["FN"]), flush=True)
    out["variants"]["oof_fixed_0p8_cap20"] = V.eval_keep(
        images, [picks[im["sid"]] for im in images])
    out["variants"]["oof_inner_chosen"] = V.eval_keep(
        images, [picks_inner[im["sid"]] for im in images])
    out["instances_fixed_0p8_cap20"] = int(sum(len(v) for v in picks.values()))
    out["instances_inner_chosen"] = int(sum(len(v) for v in picks_inner.values()))
    with open(os.path.join(MY, pool, "picks_%s.pkl" % pool), "wb") as fh:
        pickle.dump(dict(picks=picks, picks_inner=picks_inner, info=out), fh, protocol=4)
    with open(os.path.join(MY, pool, "picks_%s.json" % pool), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name in ("oof_fixed_0p8_cap20", "oof_inner_chosen"):
        v = out["variants"][name]
        print("%-20s PQ %.12f TP %d FP %d FN %d" % (name, v["PQ"], v["TP"], v["FP"], v["FN"]),
              flush=True)
    print("instances fixed %.0f  inner %d"
          % (out["instances_fixed_0p8_cap20"], out["instances_inner_chosen"]), flush=True)


# ---------------------------------------------------------------- emit
def _valid_doc(path):
    """True iff `path` holds a complete emitted document (not a truncated kill)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (ValueError, OSError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("boxes"), list)


def cmd_emit(pool, arm, shard, nshard):
    """arm: 'sel' / 'fixed08' (pre-declared 0.8/cap20 OOF picks) or 'sel_inner' (inner-chosen)."""
    pred_dir, sidecar_path = POOLS[pool]
    _items, _maps, idx = sidecar_full(sidecar_path)
    blob = pickle.load(open(os.path.join(MY, pool, "picks_%s.pkl" % pool), "rb"))
    picks = blob["picks_inner"] if arm == "sel_inner" else blob["picks"]
    samples = K.samples()[shard::nshard]
    out_dir = os.path.join(MY, pool, arm, "results")
    os.makedirs(out_dir, exist_ok=True)
    pol_emit = emit_policy(None, 1e-9, 0.9999)
    tot = dict(files=0, skipped=0, instances=0, kept=0)
    t0 = time.time()
    for k, sample in enumerate(samples):
        sid = sample["sample_id"]
        png = sample["png_path"]
        stem = os.path.splitext(os.path.basename(png))[0]
        out_path = os.path.join(out_dir, stem + ".json")
        if _valid_doc(out_path):
            # already emitted by an earlier (killed) run of this shard: skip, but
            # keep the counters honest by re-reading the document's counts.
            with open(out_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            tot["skipped"] += 1
            tot["instances"] += int(len(prev.get("boxes", [])))
            continue
        pred_path = K.find_pred(pred_dir, sid, png)
        segs, native, _doc = K.candidate_segments(pred_path)
        masks, classes, scores, _r = K.order_and_score(segs, idx[sid], sid)
        keep = np.asarray(picks[sid], dtype=np.int64)
        _NATIVE[sid] = (int(native[0]), int(native[1]))
        out = emit_selected(sample, masks, classes, scores, keep, pol_emit)
        doc = SS.strip(out["document"])[0]
        stem = os.path.splitext(os.path.basename(png))[0]
        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        tot["files"] += 1
        tot["kept"] += int(len(keep))
        tot["instances"] += int(len(doc["boxes"]))
        if (k + 1) % 10 == 0 or k + 1 == len(samples):
            print("[%d/%d] %.2f s/img" % (k + 1, len(samples), (time.time() - t0) / (k + 1)),
                  flush=True)
    print(json.dumps(tot), flush=True)


# ---------------------------------------------------------------- score
def _agg(rows):
    tp = sum(r["TP"] for r in rows)
    fp = sum(r["FP"] for r in rows)
    fn = sum(r["FN"] for r in rows)
    siou = sum(r["SumIoU"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(TP=tp, FP=fp, FN=fn, SumIoU=round(siou, 9), emitted=tp + fp,
                PQ=round(siou / den, 12) if den else 0.0,
                SQ=round(siou / tp, 12) if tp else 0.0,
                RQ=round(tp / den, 12) if den else 0.0,
                recall=round(tp / (tp + fn), 9) if tp + fn else 0.0,
                precision=round(tp / (tp + fp), 9) if tp + fp else 0.0,
                records=len(rows), gt=sum(r["gt_segments"] for r in rows))


def cmd_score(pool, arms):
    from score_ship import score as canonical_score
    out = dict(pool=pool, metric="cad-parent-ship-score01/score_ship.py (unmodified)", arms={})
    for arm in arms:
        d = os.path.join(MY, pool, arm, "results")
        files = len([f for f in os.listdir(d) if f.endswith(".json")])
        t0 = time.time()
        res = canonical_score(d, None)
        rows = res["per_image"]
        halves = {h: _agg([r for r in rows if int(hashlib.sha256(
            ("cad-generalization-op01|" + r["sample_id"]).encode()).hexdigest(), 16) % 2 == h])
            for h in (0, 1)}
        entry = dict(results_dir=d, documents=files, pooled=_agg(rows),
                     per_half={"eval_h%d" % h: v for h, v in halves.items()},
                     elapsed_s=round(time.time() - t0, 1))
        out["arms"][arm] = entry
        p = entry["pooled"]
        print("%-8s docs %d  TP %d FP %d FN %d SumIoU %.9f PQ %.12f SQ %.9f RQ %.9f "
              "(%.1f s)" % (arm, files, p["TP"], p["FP"], p["FN"], p["SumIoU"], p["PQ"],
                            p["SQ"], p["RQ"], entry["elapsed_s"]), flush=True)
        for k, v in entry["per_half"].items():
            print("   %-8s PQ %.12f TP %d FP %d FN %d SumIoU %.9f"
                  % (k, v["PQ"], v["TP"], v["FP"], v["FN"], v["SumIoU"]), flush=True)
    with open(os.path.join(MY, pool, "score_%s.json" % "_".join(arms)), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pass":
        cmd_pass(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "cv":
        cmd_cv(sys.argv[2])
    elif cmd == "emit":
        cmd_emit(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3:])
