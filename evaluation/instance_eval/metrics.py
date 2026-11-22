"""Instance metrics derived from the canonical document scorer.

The scorer itself is never reimplemented here: every TP/FP/FN/SumIoU number comes
from ``cadvec.recognition_raster_data.score_documents`` with the canonical scale
(100 x 100 canvas, 1000 px raster).  This module only *derives* the ratios the
project asks for (recall, precision, F1, Thing vs Stuff, per image) and audits
the denominators.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import contract as C

THING_RANGE = range(0, 30)
STUFF_RANGE = range(30, 35)


def _ratio(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _canonical(gt_documents: Sequence[Mapping[str, Any]], pred_documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    data = C.raster_data()
    result = data.score_documents(
        list(gt_documents),
        list(pred_documents),
        raster_size=C.RASTER_SIZE,
        canvas_size=C.CANVAS_SIZE,
    )
    return dict(result)


def _block(tp: int, fp: int, fn: int, sum_iou: float) -> dict[str, Any]:
    tp, fp, fn = int(tp), int(fp), int(fn)
    sum_iou = float(sum_iou)
    pq_denominator = tp + 0.5 * fp + 0.5 * fn
    recall_denominator = tp + fn
    precision_denominator = tp + fp

    pq = None if pq_denominator <= 0 else sum_iou / pq_denominator
    sq = None if tp <= 0 else sum_iou / tp
    rq = None if pq_denominator <= 0 else tp / pq_denominator
    recall = None if recall_denominator <= 0 else tp / recall_denominator
    precision = None if precision_denominator <= 0 else tp / precision_denominator
    f1_denominator = 2 * tp + fp + fn
    f1 = None if f1_denominator <= 0 else 2.0 * tp / f1_denominator

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SumIoU": round(sum_iou, 12),
        "PQ": _ratio(pq),
        "SQ": _ratio(sq),
        "RQ": _ratio(rq),
        "recall": _ratio(recall),
        "precision": _ratio(precision),
        "f1": _ratio(f1),
        "denominators": {
            "recall": recall_denominator,
            "precision": precision_denominator,
            "f1": f1_denominator,
            "pq": round(pq_denominator, 12),
        },
    }


def segment_key_counts(document: Mapping[str, Any]) -> dict[str, int]:
    """Count the segment keys the canonical scorer will see, and those it skips.

    Applies the canonical key rule only (class 35 dropped, Thing with instance
    < 0 dropped, Stuff merged per class) so the reported denominators can be
    audited against the scorer's TP+FN / TP+FP.
    """

    keys: set[tuple[int, int]] = set()
    skipped = 0
    for semantic, instance in zip(document["semanticIds"], document["instanceIds"]):
        semantic, instance = int(semantic), int(instance)
        if semantic == 35:
            skipped += 1
            continue
        if 0 <= semantic <= 29:
            if instance < 0:
                skipped += 1
                continue
            keys.add((semantic, instance))
        elif 30 <= semantic <= 34:
            keys.add((semantic, -1))
        else:
            skipped += 1
    return {"keys": len(keys), "skipped": skipped}


def _per_class(raw: Mapping[Any, Any]) -> dict[int, dict[str, Any]]:
    per_class: dict[int, dict[str, Any]] = {}
    for key, value in dict(raw or {}).items():
        semantic = int(key)
        block = _block(value.get("TP", 0), value.get("FP", 0), value.get("FN", 0), value.get("SumIoU", 0.0))
        block["group"] = "thing" if semantic in THING_RANGE else "stuff"
        per_class[semantic] = block
    return per_class


def _group(per_class: Mapping[int, Mapping[str, Any]], classes: Iterable[int]) -> dict[str, Any]:
    tp = fp = fn = 0
    sum_iou = 0.0
    members: list[int] = []
    for semantic in classes:
        block = per_class.get(int(semantic))
        if block is None:
            continue
        members.append(int(semantic))
        tp += int(block["TP"])
        fp += int(block["FP"])
        fn += int(block["FN"])
        sum_iou += float(block["SumIoU"])
    group = _block(tp, fp, fn, sum_iou)
    group["classes_present"] = members
    return group


def instance_report(
    gt_documents: Sequence[Mapping[str, Any]],
    pred_documents: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Full instance report for aligned GT/prediction document lists."""

    gt_documents = list(gt_documents)
    pred_documents = list(pred_documents)
    if len(gt_documents) != len(pred_documents):
        raise C.GuardError(
            "document_count",
            "%d gt vs %d prediction documents" % (len(gt_documents), len(pred_documents)),
        )
    if not gt_documents:
        raise C.GuardError("no_images", "nothing to score")
    if sample_ids is not None and len(sample_ids) != len(gt_documents):
        raise C.GuardError("sample_id_count", "%d ids for %d images" % (len(sample_ids), len(gt_documents)))
    ids = list(sample_ids) if sample_ids is not None else [str(index) for index in range(len(gt_documents))]

    result = _canonical(gt_documents, pred_documents)
    tp = int(result.get("TP", 0))
    fp = int(result.get("FP", 0))
    fn = int(result.get("FN", 0))
    sum_iou = float(result.get("SumIoU", 0.0))
    overall = _block(tp, fp, fn, sum_iou)
    overall["PQ"] = _ratio(result.get("PQ"))
    overall["SQ"] = _ratio(result.get("SQ"))
    overall["RQ"] = _ratio(result.get("RQ"))
    overall["SumIoU"] = round(sum_iou, 12)

    per_class = _per_class(result.get("per_class") or {})
    thing = _group(per_class, THING_RANGE)
    stuff = _group(per_class, STUFF_RANGE)

    per_image: list[dict[str, Any]] = []
    gt_scored_expected = 0
    pred_scored_expected = 0
    gt_skipped = 0
    pred_skipped = 0
    for index, (gt_document, pred_document) in enumerate(zip(gt_documents, pred_documents)):
        gt_counts = segment_key_counts(gt_document)
        pred_counts = segment_key_counts(pred_document)
        gt_scored_expected += gt_counts["keys"]
        pred_scored_expected += pred_counts["keys"]
        gt_skipped += gt_counts["skipped"]
        pred_skipped += pred_counts["skipped"]
        single = _canonical([gt_document], [pred_document])
        block = _block(
            single.get("TP", 0),
            single.get("FP", 0),
            single.get("FN", 0),
            single.get("SumIoU", 0.0),
        )
        block["PQ"], block["SQ"], block["RQ"] = (
            _ratio(single.get("PQ")),
            _ratio(single.get("SQ")),
            _ratio(single.get("RQ")),
        )
        block["image_index"] = index
        block["sample_id"] = ids[index]
        block["gt_keys"] = gt_counts["keys"]
        block["gt_skipped"] = gt_counts["skipped"]
        block["pred_keys"] = pred_counts["keys"]
        per_image.append(block)

    pq_denominator = tp + 0.5 * fp + 0.5 * fn
    expected_pq = None if pq_denominator <= 0 else round(sum_iou / pq_denominator, 12)
    class_tp = sum(int(block["TP"]) for block in per_class.values())
    class_fp = sum(int(block["FP"]) for block in per_class.values())
    class_fn = sum(int(block["FN"]) for block in per_class.values())
    consistency = {
        "gt_keys": gt_scored_expected,
        "pred_keys": pred_scored_expected,
        "gt_skipped": gt_skipped,
        "pred_skipped": pred_skipped,
        "gt_tp_plus_fn": tp + fn,
        "pred_tp_plus_fp": tp + fp,
        "gt_denominator_ok": bool(tp + fn == gt_scored_expected),
        "pred_denominator_ok": bool(tp + fp == pred_scored_expected),
        "class_totals_ok": bool(class_tp == tp and class_fp == fp and class_fn == fn),
        "pq_formula_ok": bool(
            (expected_pq is None and overall["PQ"] is None)
            or (expected_pq is not None and overall["PQ"] is not None and abs(expected_pq - overall["PQ"]) <= 1e-9)
        ),
        "thing_plus_stuff_ok": bool(
            thing["TP"] + stuff["TP"] == tp
            and thing["FP"] + stuff["FP"] == fp
            and thing["FN"] + stuff["FN"] == fn
        ),
    }
    consistency["ok"] = all(
        bool(consistency[key])
        for key in (
            "gt_denominator_ok",
            "pred_denominator_ok",
            "class_totals_ok",
            "pq_formula_ok",
            "thing_plus_stuff_ok",
        )
    )

    return {
        "n_images": len(gt_documents),
        "raster_size": list(result.get("raster_size") or [C.RASTER_SIZE, C.RASTER_SIZE]),
        "canvas_size": list(result.get("canvas_size", C.CANVAS_SIZE)),
        "policy": result.get("policy"),
        "iou_threshold": C.IOU_THRESHOLD,
        "assignment": "hungarian",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SumIoU": overall["SumIoU"],
        "PQ": overall["PQ"],
        "SQ": overall["SQ"],
        "RQ": overall["RQ"],
        "recall": overall["recall"],
        "precision": overall["precision"],
        "f1": overall["f1"],
        "denominators": overall["denominators"],
        "thing": thing,
        "stuff": stuff,
        "per_class": {str(semantic): block for semantic, block in sorted(per_class.items())},
        "per_image": per_image,
        "consistency": consistency,
    }


__all__ = ["instance_report", "segment_key_counts"]
