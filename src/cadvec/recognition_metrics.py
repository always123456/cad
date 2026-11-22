"""Pure prediction validation and recognition metrics.

The module is intentionally independent of file I/O, RF/GraphSAGE and the
eval_pool.  It validates one prediction record per current candidate, separates
semantic/instance supervision, and returns JSON-safe metrics with explicit
empty denominators.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

from .recognition_contract import METRICS_SCHEMA, PREDICTION_SCHEMA


class MetricsValidationError(ValueError):
    code = "metrics_validation_error"

    def __init__(self, reason: str, detail: Optional[Mapping[str, Any]] = None):
        self.reason = reason
        self.detail = dict(detail or {})
        super().__init__(f"{reason}: {json.dumps(self.detail, ensure_ascii=False, sort_keys=True)}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _points(value: Any) -> Tuple[Tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise MetricsValidationError("invalid_geometry", {"value": value})
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise MetricsValidationError("invalid_geometry_point", {"point": point})
        x, y = point
        if not _is_number(x) or not _is_number(y):
            raise MetricsValidationError("non_finite_geometry", {"point": point})
        result.append((float(x), float(y)))
    return tuple(result)


def _candidate_points(candidate: Mapping[str, Any]) -> Tuple[Tuple[float, float], ...]:
    geometry = candidate.get("geometry")
    if not isinstance(geometry, Mapping):
        raise MetricsValidationError("candidate_geometry_missing", {"candidate": candidate.get("candidate_uid")})
    raw = geometry.get("points_px")
    if raw is None:
        raw = geometry.get("points")
    return _points(raw)


def _same_points(left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(float(a[0]), float(b[0]), rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(float(a[1]), float(b[1]), rel_tol=1e-12, abs_tol=1e-9)
        for a, b in zip(left, right)
    )


def _classes_list(classes_raw: Iterable[Any]) -> List[int]:
    if isinstance(classes_raw, (str, bytes)) or not isinstance(classes_raw, Iterable):
        raise MetricsValidationError("classes_raw_invalid")
    values = []
    for value in classes_raw:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 35:
            raise MetricsValidationError("classes_raw_invalid", {"value": value})
        values.append(value)
    if len(values) != len(set(values)):
        raise MetricsValidationError("classes_raw_duplicate")
    return sorted(values)


def validate_predictions(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require exactly one valid prediction for every current candidate."""

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise MetricsValidationError("candidates_not_a_sequence")
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise MetricsValidationError("predictions_not_a_sequence")
    by_uid: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise MetricsValidationError("candidate_not_a_mapping")
        uid = candidate.get("candidate_uid")
        if not isinstance(uid, str) or not uid:
            raise MetricsValidationError("candidate_uid_invalid", {"candidate_uid": uid})
        if uid in by_uid:
            raise MetricsValidationError("duplicate_candidate_uid", {"candidate_uid": uid})
        sample_id = candidate.get("sample_id")
        source_kind = candidate.get("source_kind")
        if not isinstance(sample_id, str) or not sample_id:
            raise MetricsValidationError("candidate_sample_id_invalid", {"candidate_uid": uid})
        if source_kind not in ("png_line", "svg_primitive"):
            raise MetricsValidationError("candidate_source_kind_invalid", {"candidate_uid": uid, "value": source_kind})
        if "schema" in candidate and candidate.get("schema") != "cadvec.recognition.candidate/1":
            raise MetricsValidationError("candidate_schema_invalid", {"candidate_uid": uid})
        if "version" in candidate and candidate.get("version") != 1:
            raise MetricsValidationError("candidate_version_invalid", {"candidate_uid": uid})
        by_uid[uid] = {
            "candidate_uid": uid,
            "sample_id": sample_id,
            "source_kind": source_kind,
            "points": _candidate_points(candidate),
            "candidate": candidate,
        }

    pred_by_uid: Dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise MetricsValidationError("prediction_not_a_mapping")
        if prediction.get("schema") != PREDICTION_SCHEMA:
            raise MetricsValidationError("prediction_schema_invalid", {"candidate_uid": prediction.get("candidate_uid")})
        if prediction.get("version") != 1:
            raise MetricsValidationError("prediction_version_invalid", {"candidate_uid": prediction.get("candidate_uid")})
        uid = prediction.get("candidate_uid")
        if not isinstance(uid, str) or not uid:
            raise MetricsValidationError("prediction_uid_invalid", {"candidate_uid": uid})
        if uid in pred_by_uid:
            raise MetricsValidationError("duplicate_prediction_uid", {"candidate_uid": uid})
        if uid not in by_uid:
            raise MetricsValidationError("extra_prediction_uid", {"candidate_uid": uid})
        candidate = by_uid[uid]
        if prediction.get("sample_id") != candidate["sample_id"]:
            raise MetricsValidationError("prediction_sample_mismatch", {"candidate_uid": uid})
        if prediction.get("source_kind") != candidate["source_kind"]:
            raise MetricsValidationError("prediction_source_mismatch", {"candidate_uid": uid})
        if "target" not in prediction or prediction.get("target") is not None:
            raise MetricsValidationError("prediction_target_not_null", {"candidate_uid": uid})
        if "association" in prediction and prediction.get("association") is not None:
            raise MetricsValidationError("prediction_association_not_null", {"candidate_uid": uid})
        predicted = prediction.get("semantic_raw_pred")
        if not isinstance(predicted, int) or isinstance(predicted, bool) or not 1 <= predicted <= 35:
            raise MetricsValidationError("semantic_raw_pred_invalid", {"candidate_uid": uid, "value": predicted})
        probability = prediction.get("semantic_proba")
        if probability is not None and (not _is_number(probability) or not 0.0 <= float(probability) <= 1.0):
            raise MetricsValidationError("semantic_proba_invalid", {"candidate_uid": uid, "value": probability})
        group = prediction.get("instance_group")
        if group is not None and (not isinstance(group, (int, str)) or isinstance(group, bool)):
            raise MetricsValidationError("instance_group_invalid", {"candidate_uid": uid, "value": group})
        pred_points = _points(prediction.get("geometry_px"))
        if not _same_points(pred_points, candidate["points"]):
            raise MetricsValidationError("prediction_geometry_mismatch", {"candidate_uid": uid})
        pred_by_uid[uid] = prediction

    missing = sorted(set(by_uid) - set(pred_by_uid))
    if missing:
        raise MetricsValidationError("missing_predictions", {"candidate_uids": missing})
    return {
        "candidate_count": len(by_uid),
        "prediction_count": len(pred_by_uid),
        "candidate_by_uid": by_uid,
        "prediction_by_uid": pred_by_uid,
    }


def _records(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    classes_raw: Sequence[int],
) -> List[Dict[str, Any]]:
    validation = validate_predictions(candidates, predictions)
    classes = set(classes_raw)
    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        uid = candidate["candidate_uid"]
        prediction = validation["prediction_by_uid"][uid]
        association = candidate.get("association") or {}
        supervision = candidate.get("supervision") or {}
        semantic_raw = supervision.get("semantic_raw")
        semantic_ok = (
            association.get("status") == "unique"
            and supervision.get("semantic_status") == "ok"
            and isinstance(semantic_raw, int)
            and not isinstance(semantic_raw, bool)
            and 1 <= semantic_raw <= 35
        )
        instance_raw = supervision.get("instance_raw")
        instance_ok = (
            association.get("status") == "unique"
            and supervision.get("instance_status") == "ok"
            and isinstance(instance_raw, int)
            and not isinstance(instance_raw, bool)
            and instance_raw >= 0
            and isinstance(semantic_raw, int)
            and 1 <= semantic_raw <= 30
        )
        predicted = int(prediction["semantic_raw_pred"])
        records.append(
            {
                "candidate_uid": uid,
                "sample_id": candidate["sample_id"],
                "group_id": candidate.get("group_id"),
                "pair_id": candidate.get("pair_id") or candidate["sample_id"],
                "unique_png_id": candidate.get("unique_png_id") or candidate["sample_id"],
                "primitive_uid": candidate.get("primitive_uid"),
                "source_kind": candidate["source_kind"],
                "association_status": association.get("status"),
                "target_semantic_raw": semantic_raw if semantic_ok else None,
                "target_valid": semantic_ok,
                "predicted_semantic_raw": predicted,
                "predicted_known": predicted in classes,
                "predicted_proba": prediction.get("semantic_proba"),
                "target_instance_raw": instance_raw if instance_ok else None,
                "instance_valid": instance_ok,
                "predicted_instance_group": prediction.get("instance_group"),
            }
        )
    return records


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _semantic_metrics(records: Sequence[Mapping[str, Any]], classes_raw: Sequence[int]) -> Dict[str, Any]:
    classes = set(classes_raw)
    evaluable = [row for row in records if row["target_valid"]]
    support = Counter(int(row["target_semantic_raw"]) for row in evaluable)
    eval_classes = sorted(support)
    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in sorted(classes | set(eval_classes)):
        tp = sum(
            1
            for row in evaluable
            if row["predicted_known"]
            and row["target_semantic_raw"] == cls
            and row["predicted_semantic_raw"] == cls
        )
        predicted = sum(
            1
            for row in evaluable
            if row["predicted_known"] and row["predicted_semantic_raw"] == cls
        )
        fp = predicted - tp
        fn = support.get(cls, 0) - tp
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support.get(cls, 0))
        if support.get(cls, 0) == 0:
            class_f1 = None
        elif predicted == 0:
            class_f1 = 0.0
        else:
            class_f1 = _f1(precision, recall)
            if class_f1 is None:
                class_f1 = 0.0
        per_class[str(cls)] = {
            "support": support.get(cls, 0),
            "predicted": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
        }
    unknown = sum(1 for row in evaluable if not row["predicted_known"])
    tp_total = sum(entry["tp"] for entry in per_class.values())
    predicted_known = sum(entry["predicted"] for entry in per_class.values())
    support_total = len(evaluable)
    micro_precision = _safe_div(tp_total, predicted_known + unknown)
    micro_recall = _safe_div(tp_total, support_total)
    macro_values = [per_class[str(cls)]["f1"] for cls in eval_classes]
    macro = sum(value for value in macro_values if value is not None) / len(macro_values) if macro_values else None
    train_supported = [cls for cls in eval_classes if cls in classes]
    val_only = [cls for cls in eval_classes if cls not in classes]
    train_only = sorted(cls for cls in classes if cls not in support)
    return {
        "status": "ok" if evaluable and predicted_known + unknown else ("partial" if evaluable else "blocked"),
        "classes_raw": sorted(classes),
        "eval_classes": eval_classes,
        "train_supported": train_supported,
        "train_only": train_only,
        "val_only": val_only,
        "unknown": unknown,
        "counts": {
            "evaluable": len(evaluable),
            "support_total": support_total,
            "predicted_known": predicted_known,
            "unknown_predictions": unknown,
            "unknown_fp": unknown,
            "tp": tp_total,
        },
        "macro_f1": macro,
        "micro_f1": _f1(micro_precision, micro_recall),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "per_class": per_class,
    }


def _pred_key(row: Mapping[str, Any], mode: str = "prediction") -> Hashable:
    if mode == "singleton":
        return ("item", row["sample_id"], row["candidate_uid"])
    if mode == "allmerge":
        return ("all", row["sample_id"])
    group = row.get("predicted_instance_group")
    if group is None:
        return ("singleton", row["sample_id"], row["candidate_uid"])
    return ("prediction", row["sample_id"], group)


def _gt_key(row: Mapping[str, Any]) -> Hashable:
    return ("gt", row["sample_id"], int(row["target_instance_raw"]))


def _grouping_metrics(items: Sequence[Mapping[str, Any]], pred_key) -> Dict[str, Any]:
    if not items:
        return {
            "b3": {"precision": None, "recall": None, "f1": None},
            "pairwise": {"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0},
            "merge_split": {"merged_pred_clusters": 0, "merged_gt_pairs": 0, "split_gt_clusters": 0},
            "counts": {"items": 0, "pred_clusters": 0, "gt_clusters": 0},
            "empty_denominator": True,
        }
    pred_members: Dict[Hashable, List[Mapping[str, Any]]] = defaultdict(list)
    gt_members: Dict[Hashable, List[Mapping[str, Any]]] = defaultdict(list)
    cross: Counter = Counter()
    for row in items:
        p = pred_key(row)
        g = _gt_key(row)
        pred_members[p].append(row)
        gt_members[g].append(row)
        cross[(p, g)] += 1
    precisions = []
    recalls = []
    for row in items:
        p = pred_key(row)
        g = _gt_key(row)
        intersection = cross[(p, g)]
        precisions.append(intersection / len(pred_members[p]))
        recalls.append(intersection / len(gt_members[g]))
    b3_precision = sum(precisions) / len(precisions)
    b3_recall = sum(recalls) / len(recalls)
    tp = 0
    for p, members in pred_members.items():
        by_gt = Counter(_gt_key(row) for row in members)
        tp += sum(count * (count - 1) // 2 for count in by_gt.values())
    pred_pairs = sum(len(members) * (len(members) - 1) // 2 for members in pred_members.values())
    gt_pairs = sum(len(members) * (len(members) - 1) // 2 for members in gt_members.values())
    fp = pred_pairs - tp
    fn = gt_pairs - tp
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    merged_pred_clusters = 0
    merged_gt_pairs = 0
    for members in pred_members.values():
        count = len({_gt_key(row) for row in members})
        if count >= 2:
            merged_pred_clusters += 1
            merged_gt_pairs += count * (count - 1) // 2
    split_gt_clusters = sum(
        1 for members in gt_members.values() if len({pred_key(row) for row in members}) >= 2
    )
    return {
        "b3": {"precision": b3_precision, "recall": b3_recall, "f1": _f1(b3_precision, b3_recall)},
        "pairwise": {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "empty_denominator": tp + fp == 0 or tp + fn == 0,
        },
        "merge_split": {
            "merged_pred_clusters": merged_pred_clusters,
            "merged_gt_pairs": merged_gt_pairs,
            "split_gt_clusters": split_gt_clusters,
        },
        "counts": {
            "items": len(items),
            "pred_clusters": len(pred_members),
            "gt_clusters": len(gt_members),
        },
        "empty_denominator": False,
    }


def _instance_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    items = [row for row in records if row["instance_valid"]]
    if not items:
        return {
            "status": "blocked",
            "reason": "empty_instance_denominator",
            "b3": {"precision": None, "recall": None, "f1": None},
            "pairwise": {"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0},
            "merge_split": {"merged_pred_clusters": 0, "merged_gt_pairs": 0, "split_gt_clusters": 0},
            "baselines": {"singleton": None, "allmerge": None},
            "counts": {"items": 0, "gt_clusters": 0, "pred_clusters": 0},
            "empty_denominator": True,
        }
    primary = _grouping_metrics(items, _pred_key)
    singleton = _grouping_metrics(items, lambda row: _pred_key(row, "singleton"))
    allmerge = _grouping_metrics(items, lambda row: _pred_key(row, "allmerge"))
    return {
        "status": "ok",
        "reason": None,
        "b3": primary["b3"],
        "pairwise": primary["pairwise"],
        "merge_split": primary["merge_split"],
        "baselines": {"singleton": singleton, "allmerge": allmerge},
        "counts": primary["counts"],
        "empty_denominator": False,
    }


def _denominators(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def count(level: str, *, evaluable: bool) -> int:
        selected = [row for row in records if (not evaluable or row["target_valid"])]
        if level == "candidate":
            return len(selected)
        if level == "primitive":
            return len(
                {
                    (row["sample_id"], row["primitive_uid"])
                    for row in selected
                    if row["primitive_uid"] is not None
                }
            )
        if level == "pair":
            return len({(row["sample_id"], row["pair_id"]) for row in selected})
        if level == "unique_png":
            return len({(row["sample_id"], row["unique_png_id"]) for row in selected})
        raise MetricsValidationError("unknown_denominator_level", {"level": level})

    return {
        "candidate": {"total": count("candidate", evaluable=False), "semantic_eval": count("candidate", evaluable=True)},
        "primitive": {"total": count("primitive", evaluable=False), "semantic_eval": count("primitive", evaluable=True)},
        "pair": {"total": count("pair", evaluable=False), "semantic_eval": count("pair", evaluable=True)},
        "unique_png": {"total": count("unique_png", evaluable=False), "semantic_eval": count("unique_png", evaluable=True)},
    }


def compute_metrics(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    classes_raw: Iterable[Any],
    coverage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Strictly validate predictions and return JSON-safe semantic/instance metrics."""

    classes = _classes_list(classes_raw)
    validation = validate_predictions(candidates, predictions)
    records = _records(candidates, predictions, classes)
    semantic = _semantic_metrics(records, classes)
    instance = _instance_metrics(records)
    denominators = _denominators(records)
    coverage_counts = {
        "selected": int((coverage or {}).get("selected", len(records))),
        "geometry_incomplete": int((coverage or {}).get("geometry_incomplete", 0)),
        "mapping_unavailable": int((coverage or {}).get("mapping_unavailable", 0)),
        "association_excluded": sum(1 for row in records if row["association_status"] == "excluded"),
        "association_ambiguous": sum(1 for row in records if row["association_status"] == "ambiguous"),
        "association_unmatched": sum(1 for row in records if row["association_status"] == "unmatched"),
    }
    semantic_counts = semantic.get("counts") or {}
    instance_counts = instance.get("counts") or {}
    denominators["coverage"] = coverage_counts
    denominators["semantic"] = {
        "evaluable": semantic_counts.get("evaluable", 0),
        "support_total": semantic_counts.get("support_total", 0),
        "unknown_predictions": semantic_counts.get("unknown_predictions", 0),
    }
    denominators["instance"] = {
        "items": instance_counts.get("items", 0),
        "gt_clusters": instance_counts.get("gt_clusters", 0),
        "pred_clusters": instance_counts.get("pred_clusters", 0),
    }
    reasons = []
    if not records:
        status = "blocked"
        reasons.append("no_candidates")
    elif semantic["status"] != "ok":
        status = "partial"
        reasons.append(f"semantic_{semantic['status']}")
    elif instance["status"] != "ok":
        status = "partial"
        reasons.append("instance_empty_denominator")
    else:
        status = "ok"
    payload = {
        "schema": METRICS_SCHEMA,
        "version": 1,
        "status": status,
        "status_reason": reasons,
        "counts": {
            "candidates": validation["candidate_count"],
            "predictions": validation["prediction_count"],
            "semantic_evaluable": semantic_counts.get("evaluable", 0),
            "instance_evaluable": instance_counts.get("items", 0),
            "unknown_predictions": semantic_counts.get("unknown_predictions", 0),
            **coverage_counts,
        },
        "coverage": coverage_counts,
        "denominators": denominators,
        "semantic": semantic,
        "instance": instance,
    }
    # Fail early if a future edit introduces NaN/Infinity or non-JSON values.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload
