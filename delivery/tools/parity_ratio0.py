"""CPU parity check: this package's --refine-ratio 0.0 arm vs the sweep lane's.

Both records are 1000x1000 natives, so the document affine is scale 10 / offset 0
and the 1000x1000 rasterised document mask *is* the native refined mask: the
comparison below is a bit-exact native-raster comparison.

Reference side: the sweep lane's own functions (pass1.decompose + pass1.mask_at at
ratio 0.00 with the composite128 ink, then fastemit.raster_mask, which is validated
bit-exact against batch.default_emit + the canonical rasteriser).

This package's side: the real CLI path (run_delivery -> select_instances ->
_refine_stage -> square-canvas pad -> the lane's default_emit) with model access
injected (no torch forward, no GPU) and the real PNG as the ink source.

    python tools/parity_ratio0.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TESTS = os.path.join(PKG, "tests")
SWEEP = os.environ.get("CAD_INKREFINE_SWEEP_DIR", "")
for path in (PKG, TESTS, SWEEP):
    if path and path not in sys.path:
        sys.path.insert(0, path)

import cad_mrcnn_delivery as delivery  # noqa: E402
import fastemit as F  # noqa: E402
import pass1 as P  # noqa: E402
import support  # noqa: E402
import sweep as S  # noqa: E402

RATIO = 0.00
THRESHOLD = 0.99
CAP = 20
TARGETS = ("train_final:3ZCLT", "train_final:TqNpv")
OUT_ROOT = os.path.join(os.path.dirname(SWEEP), "cad-inkrefine-parity01")


def seal_samples():
    with open(P.SEAL, encoding="utf-8") as handle:
        return {sample["sample_id"]: sample for sample in json.load(handle)["samples"]}


def sidecar_scores():
    with open(P.SIDECAR, encoding="utf-8") as handle:
        return {item["sample_id"]: item for item in json.load(handle)}


def reference_side(sample):
    """The sweep lane's arm: candidates, selection, ratio-0.00 refinement, rasters."""

    with open(sample["prediction"], encoding="utf-8") as handle:
        envelope = json.load(handle)
    document = envelope["document"]
    native = (envelope.get("source") or {}).get("native_size") or {}
    width, height = int(native.get("width") or 1000), int(native.get("height") or 1000)

    native_segments = P.raster_runs(document, width, height)
    keys = [(int(seg["class_id"]), int(seg["instance_id"])) for seg in native_segments]
    sidecar = sidecar_scores()[sample["sample_id"]]
    instances = {(int(i["class_id"]), int(i["instance_id"])): i for i in sidecar["instances"]}
    scores = np.array([float(instances[key]["score"]) for key in keys], dtype=np.float64)

    chosen = S.select({"scores": scores}, THRESHOLD, CAP)
    _ink_a, ink_b = P.ink_maps(sample["png_path"])
    assert ink_b.shape == (height, width), (ink_b.shape, height, width)

    affine = F.emit_affine(width, height)
    stats = {key: 0 for key in (
        "B00_orig_empty", "B00_a1_empty", "B00_components", "B00_kept_components",
        "B00_grown_empty", "B00_a1_empty_used",
    )}
    masks, tiers = [], []
    for index in chosen:
        segment = native_segments[index]
        orig = segment["mask"]
        decomposition = P.decompose(orig, ink_b)
        refined, tier = P.mask_at(decomposition, orig, RATIO, stats, "B00")
        masks.append(F.raster_mask(refined, affine))
        tiers.append(tier)
    return {
        "native": (width, height),
        "classes": [int(native_segments[i]["class_id"]) for i in chosen],
        "scores": [float(scores[i]) for i in chosen],
        "native_masks": [native_segments[i]["mask"] for i in chosen],
        "masks": masks,
        "tiers": tiers,
        "stats": stats,
        "candidates": native_segments,
        "candidate_scores": scores,
        "png": sample["png_path"],
        "sample_id": sample["sample_id"],
    }


def png_loader(path):
    from PIL import Image

    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    channels = np.ascontiguousarray(rgba[..., :3].transpose(2, 0, 1)).astype(np.float32) / 255.0
    ink, (width, height) = delivery._refine().ink_from_png(path, "composite128")
    return channels, ink, {"path": str(path), "width": width, "height": height}


def mine(sample_id, reference, out_dir):
    labels_all = np.array([int(seg["class_id"]) + 1 for seg in reference["candidates"]], dtype=np.int64)
    scores_all = np.array(reference["candidate_scores"], dtype=np.float64)
    masks_all = [np.asarray(seg["mask"], dtype=bool) for seg in reference["candidates"]]

    chosen = S.select({"scores": scores_all}, THRESHOLD, CAP)
    labels = labels_all[chosen]
    scores = scores_all[chosen].astype(np.float32)
    probabilities = np.stack([masks_all[i].astype(np.float32) for i in chosen])

    def predict(_context, _image):
        return {
            "boxes": np.zeros((len(labels), 4), dtype=np.float32),
            "labels": labels,
            "scores": scores,
            "mask_probabilities": probabilities,
            "score_threshold": 0.05,
            "mask_threshold": 0.5,
        }

    checkpoint = os.path.join(OUT_ROOT, "checkpoint.pt")
    if not os.path.isfile(checkpoint):
        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(checkpoint, "wb") as handle:
            handle.write(b"parity fixture, no weights")
    namespace = argparse.Namespace(
        png=reference["png"],
        inputs=None,
        out=out_dir,
        checkpoint=checkpoint,
        score_threshold=THRESHOLD,
        max_detections=300,
        per_image_cap=CAP,
        nms_iou=0.5,
        mask_threshold=0.5,
        refine_mask="a6",
        refine_ink="composite128",
        refine_ratio=RATIO,
        device="cpu",
        sample_id=sample_id,
        group_id=None,
        selection_report=None,
        dry_run=False,
    )
    summary = delivery.run_delivery(
        namespace,
        loader=support.fake_loader,
        predict=predict,
        png_loader=png_loader,
    )
    emitted = []
    for relative in summary["results"]:
        with open(os.path.join(summary["out"], relative), encoding="utf-8") as handle:
            payload = json.load(handle)
        document = payload.get("document", payload)
        for segment in P.raster_runs(document, 1000, 1000):
            emitted.append((int(segment["class_id"]), segment["mask"]))
    return summary, emitted


def compare(sample_id):
    samples = seal_samples()
    reference = reference_side(samples[sample_id])
    out_dir = os.path.join(OUT_ROOT, sample_id.replace(":", "_"))
    if os.path.isdir(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    summary, emitted = mine(sample_id, reference, out_dir)

    reference_classes = list(reference["classes"])
    mine_classes = [cls for cls, _mask in emitted]
    print("%s: native %s" % (sample_id, reference["native"]))
    print("  sweep lane arm : %d instances, classes %s" % (len(reference_classes), reference_classes))
    print("  this package   : %d instances, classes %s" % (len(mine_classes), mine_classes))
    print("  tiers          : %s" % (sorted(set(reference["tiers"]))))
    receipt_refine = summary["refine_mask"]
    print("  receipt        : rule=%s ink=%s ratio=%r changed=%d before=%d after=%d"
          % (receipt_refine["rule"], receipt_refine["ink"], receipt_refine["component_coverage"],
             receipt_refine["instances_changed"], receipt_refine["instances_nonempty_before"],
             receipt_refine["instances_nonempty_after"]))
    result = {"sample_id": sample_id, "reference_count": len(reference_classes),
              "package_count": len(mine_classes), "classes_mine": mine_classes,
              "classes_reference": reference_classes,
              "receipt_ratio": receipt_refine["component_coverage"]}

    if sorted(reference_classes) != sorted(mine_classes) or len(emitted) != len(reference["masks"]):
        result["verdict"] = "INSTANCE SETS DIFFER"
        print("  VERDICT: instance sets differ -- class multisets do not match")
        return result

    if reference_classes == mine_classes:
        ordering = "document order"
        pairs = list(zip(mine_classes, [mask for _cls, mask in emitted], reference["masks"]))
    else:
        # model/emit.py emits things first and stuff (class >= 30) last, grouped by
        # class; the sweep lane's selection order is score-descending.  Same
        # instances, the emitter's canonical order -- so pair by class and content.
        ordering = "class-sorted (the lane's emitter puts stuff last)"
        mine_sorted = sorted(emitted, key=lambda item: (item[0], item[1].tobytes()))
        ref_sorted = sorted(zip(reference_classes, reference["masks"]),
                            key=lambda item: (item[0], item[1].tobytes()))
        pairs = [(mine_cls, mine_mask, ref_mask)
                 for (mine_cls, mine_mask), (_ref_cls, ref_mask) in zip(mine_sorted, ref_sorted)]

    only_ref = only_mine = 0
    per_instance = []
    for index, (cls, mask, want) in enumerate(pairs):
        extra = int(np.count_nonzero(mask & ~want))
        missing = int(np.count_nonzero(want & ~mask))
        only_mine += extra
        only_ref += missing
        per_instance.append({"index": index, "class_id": cls, "only_package": extra,
                             "only_reference": missing,
                             "reference_px": int(want.sum()), "package_px": int(mask.sum())})
    identical = only_ref == 0 and only_mine == 0
    result.update({
        "identical": identical,
        "ordering": ordering,
        "pixels_only_in_reference": only_ref,
        "pixels_only_in_package": only_mine,
        "per_instance": per_instance,
        "verdict": "BIT-EXACT" if identical else "DIFFERS",
    })
    print("  ordering       : %s" % ordering)
    print("  per-instance   : %s" % ", ".join(
        "#%d/%d ref=%d pkg=%d d+=%d d-=%d" % (item["index"], item["class_id"], item["reference_px"],
                                              item["package_px"], item["only_package"],
                                              item["only_reference"]) for item in per_instance))
    print("  pixels only in reference mask: %d" % only_ref)
    print("  pixels only in package   mask: %d" % only_mine)
    print("  VERDICT: %s" % result["verdict"])
    return result


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = [compare(sample_id) for sample_id in TARGETS]
    path = os.path.join(OUT_ROOT, "parity_ratio0.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("wrote %s" % path)
    return 0 if all(item.get("identical") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
