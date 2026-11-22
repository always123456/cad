"""Shared fit-only RF/GraphSAGE evaluation on the permanent tune set.

The tune PNG candidates are materialized exactly once, projected without
supervision/association, and passed as the same objects to every model.  All
scores go through ``recognition_metrics.compute_metrics`` so both models share
the same candidate denominator, coverage accounting and metric code.  A model
that has no label-free instance grouping keeps ``instance_group=null`` and its
instance metrics are explicitly reported as unsupported rather than silently
scored as singletons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import recognition_baseline as baseline
from . import recognition_candidates as candidate_mod
from . import recognition_gnn as gnn
from . import recognition_metrics as metrics_mod
from .recognition_contract import (
    ROLE_FIT,
    ROLE_TUNE,
    STUFF_MAX,
    STUFF_MIN,
    THING_MAX,
    THING_MIN,
    BlockedError,
    ContractError,
    RecognitionError,
    RoleAccessError,
)

TUNE_COMPARISON_SCHEMA = "cadvec.recognition.tune_comparison/1"
INSTANCE_UNSUPPORTED = "unsupported_no_label_free_grouping"
INSTANCE_SCORED = "scored_label_free_grouping"


@dataclass(frozen=True)
class ModelEvaluator:
    """A model prediction callable plus whether it emits label-free grouping."""

    name: str
    predict: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]
    label_free_grouping: bool


@dataclass
class TuneCandidateSet:
    """The single permanent tune candidate set and its coverage accounting."""

    candidates: List[Dict[str, Any]]
    coverage: Dict[str, Any]
    group_ids: List[str]
    sample_ids: List[str]


def label_free_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model-visible projection: identity + geometry, no labels."""

    return gnn.label_free_candidate(candidate)


def _require_tune_only(
    candidates: Sequence[Mapping[str, Any]],
    *,
    allowed_sample_ids: Optional[Sequence[str]] = None,
    allowed_group_ids: Optional[Sequence[str]] = None,
) -> None:
    if not candidates:
        raise BlockedError("no_tune_candidates")
    samples = set(allowed_sample_ids) if allowed_sample_ids is not None else None
    groups = set(allowed_group_ids) if allowed_group_ids is not None else None
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ContractError("tune_candidate_not_a_mapping")
        uid = candidate.get("candidate_uid")
        if not isinstance(uid, str) or not uid:
            raise ContractError("tune_candidate_uid_invalid")
        if uid in seen:
            raise ContractError(f"duplicate_tune_candidate_uid:{uid}")
        seen.add(uid)
        if candidate.get("role") != ROLE_TUNE:
            raise RoleAccessError(f"tune_candidate_role_mismatch:{uid}")
        sample_id = candidate.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError(f"tune_candidate_sample_invalid:{uid}")
        if samples is not None and sample_id not in samples:
            raise RoleAccessError(f"tune_candidate_outside_tune_samples:{uid}")
        group_id = candidate.get("group_id")
        if groups is not None and (group_id is None or group_id not in groups):
            raise RoleAccessError(f"tune_candidate_outside_tune_groups:{uid}")


def _assert_prediction_contract(
    predictions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    label_free_grouping: bool,
) -> None:
    candidate_uids = [candidate["candidate_uid"] for candidate in candidates]
    expected = set(candidate_uids)
    seen = set()
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise metrics_mod.MetricsValidationError("prediction_not_a_mapping")
        uid = prediction.get("candidate_uid")
        if not isinstance(uid, str) or not uid:
            raise metrics_mod.MetricsValidationError("prediction_uid_invalid", {"candidate_uid": uid})
        if uid in seen:
            raise metrics_mod.MetricsValidationError("duplicate_prediction_uid", {"candidate_uid": uid})
        if uid not in expected:
            raise metrics_mod.MetricsValidationError("extra_prediction_uid", {"candidate_uid": uid})
        seen.add(uid)
        if "target" not in prediction or prediction.get("target") is not None:
            raise metrics_mod.MetricsValidationError("prediction_target_not_null", {"candidate_uid": uid})
        if "association" not in prediction or prediction.get("association") is not None:
            raise metrics_mod.MetricsValidationError(
                "prediction_association_not_null", {"candidate_uid": uid}
            )
        if "supervision" in prediction:
            raise metrics_mod.MetricsValidationError(
                "prediction_contains_supervision",
                {"candidate_uid": uid, "field": "supervision"},
            )
        group = prediction.get("instance_group")
        predicted = prediction.get("semantic_raw_pred")
        predicted_thing = (
            isinstance(predicted, int)
            and not isinstance(predicted, bool)
            and THING_MIN <= predicted <= THING_MAX
        )
        predicted_stuff = (
            isinstance(predicted, int)
            and not isinstance(predicted, bool)
            and STUFF_MIN <= predicted <= STUFF_MAX
        )
        if label_free_grouping and group is None and predicted_thing:
            raise metrics_mod.MetricsValidationError(
                "instance_group_missing_for_label_free_model", {"candidate_uid": uid}
            )
        if label_free_grouping and group is not None and predicted_stuff:
            raise metrics_mod.MetricsValidationError(
                "instance_group_for_stuff_prediction", {"candidate_uid": uid}
            )
        if not label_free_grouping and group is not None:
            raise metrics_mod.MetricsValidationError(
                "instance_group_must_be_null", {"candidate_uid": uid}
            )
    missing = sorted(expected - seen)
    if missing:
        raise metrics_mod.MetricsValidationError("missing_predictions", {"candidate_uids": missing})


def attach_geometry_to_predictions(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach candidate pixel geometry to prediction records that lack it.

    GraphSAGE predictions are scored per graph and do not carry geometry; the
    strict metrics contract requires it.  Geometry always comes from the shared
    candidate set, never from labels or the model.
    """

    if any(not isinstance(prediction, Mapping) for prediction in predictions):
        raise metrics_mod.MetricsValidationError("prediction_not_a_mapping")
    try:
        return gnn.attach_geometry(candidates, predictions)
    except gnn.GraphSchemaError as exc:
        raise metrics_mod.MetricsValidationError(str(exc)) from exc


def _instance_block(metrics: Mapping[str, Any], label_free_grouping: bool) -> Dict[str, Any]:
    instance = metrics.get("instance") or {}
    denominators = (metrics.get("denominators") or {}).get("instance") or {}
    if label_free_grouping:
        return {
            "status": INSTANCE_SCORED,
            "reported": True,
            "empty": False,
            "reason": None,
            "source": "label_free_model_grouping",
            "counts": dict(instance.get("counts") or {}),
            "denominator": dict(denominators),
            "b3": instance.get("b3"),
            "pairwise": instance.get("pairwise"),
            "merge_split": instance.get("merge_split"),
        }
    return {
        "status": INSTANCE_UNSUPPORTED,
        "reported": False,
        "empty": True,
        "reason": "instance_group_is_null",
        "source": None,
        "counts": dict(instance.get("counts") or {}),
        "denominator": dict(denominators),
        "b3": None,
        "pairwise": None,
        "merge_split": None,
    }


def evaluate_models(
    candidates: Sequence[Mapping[str, Any]],
    models: Sequence[ModelEvaluator],
    *,
    classes_raw: Sequence[Any],
    coverage: Mapping[str, Any],
    allowed_sample_ids: Optional[Sequence[str]] = None,
    allowed_group_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score every model on one candidate set with the strict shared metrics."""

    candidate_list = list(candidates)
    _require_tune_only(
        candidate_list,
        allowed_sample_ids=allowed_sample_ids,
        allowed_group_ids=allowed_group_ids,
    )
    if not models:
        raise BlockedError("no_models_to_evaluate")
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ContractError("duplicate_model_name")
    class_list = list(classes_raw)

    # Both models see the same label-free projection of the same candidate set.
    model_input = [label_free_candidate(candidate) for candidate in candidate_list]
    sections: Dict[str, Dict[str, Any]] = {}
    for model in models:
        predictions = model.predict(model_input)
        if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
            raise metrics_mod.MetricsValidationError("predictions_not_a_sequence")
        prediction_list = list(predictions)
        _assert_prediction_contract(
            prediction_list,
            candidate_list,
            label_free_grouping=model.label_free_grouping,
        )
        strict = baseline._compute_evaluation_metrics(
            candidate_list,
            prediction_list,
            classes_raw=class_list,
            coverage=coverage,
        )
        sections[model.name] = {
            "model": model.name,
            "label_free_grouping": model.label_free_grouping,
            "prediction_count": len(prediction_list),
            "prediction_target_all_null": all(
                row.get("target") is None for row in prediction_list
            ),
            "prediction_association_all_null": all(
                row.get("association") is None for row in prediction_list
            ),
            "metrics": strict,
            "instance": _instance_block(strict, model.label_free_grouping),
        }

    return {
        "schema": TUNE_COMPARISON_SCHEMA,
        "version": 1,
        "status": "complete",
        "role": ROLE_TUNE,
        "candidates_shared": True,
        "model_input_label_free": True,
        "candidate_count": len(candidate_list),
        "classes_raw": sorted(int(value) for value in class_list),
        "coverage": dict(coverage),
        "primary_comparison": "semantic",
        "models": sections,
    }


def build_tune_candidates(
    manifest: Mapping[str, Any],
    *,
    train_scale: int = 32,
    train_root: Optional[str] = None,
) -> TuneCandidateSet:
    """Materialize the permanent tune PNG candidate set exactly once."""

    baseline.validate_manifest(manifest)
    if train_scale not in (32, 64, 128, 256):
        raise ContractError("train_scale must be 32/64/128/256")
    reader = baseline.RoleReader(manifest, ROLE_TUNE, root=train_root)
    groups = reader.group_ids(train_scale)
    rows = baseline._rows_for_groups(manifest, groups, role=ROLE_TUNE)
    rows = baseline.validated_train_file_paths(manifest, rows)
    from . import png_frontend, svg_parse

    coverage: Dict[str, Any] = {
        "selected": len(groups),
        "pairs": len(rows),
        "processed_pairs": 0,
        "frontend_failed": 0,
        "mapping_unavailable": 0,
        "geometry_incomplete": 0,
        "candidates": 0,
    }
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        document = svg_parse.parse_svg(
            row["svg_path"], split="train_final", record_commands=True, curve_steps=128
        )
        frontend = png_frontend.predict_png(
            row["png_path"],
            png_frontend.FrontendConfig(
                max_lines=4000,
                max_pixels=40_000_000,
                max_file_bytes=64 * 1024 * 1024,
            ),
        )
        if frontend.get("status") not in ("ok", "empty_input", "no_lines_detected"):
            coverage["frontend_failed"] += 1
            continue
        gt, mapping = baseline._pixel_gt_for_frontend(
            document, row["sample_id"], frontend
        )
        if mapping.get("status") != "ok":
            coverage["mapping_unavailable"] += 1
            continue
        if any(not primitive.complete for primitive in gt):
            coverage["geometry_incomplete"] += 1
            continue
        current = candidate_mod.build_png_candidates(
            frontend,
            row["sample_id"],
            group_id=row.get("group_id"),
            role=ROLE_TUNE,
        )
        candidate_mod.associate_png_candidates(current, gt)
        candidates.extend(current)
        coverage["processed_pairs"] += 1

    coverage["candidates"] = len(candidates)
    tune_set = TuneCandidateSet(
        candidates=candidates,
        coverage=coverage,
        group_ids=list(groups),
        sample_ids=[row["sample_id"] for row in rows],
    )
    _require_tune_only(
        tune_set.candidates,
        allowed_sample_ids=tune_set.sample_ids,
        allowed_group_ids=tune_set.group_ids,
    )
    return tune_set


def load_rf_evaluator(
    rf_model_manifest: str,
    *,
    train_scale: int,
) -> tuple[ModelEvaluator, Dict[str, Any], List[int]]:
    """Load the fit-role RF semantic artifact and expose a label-free predictor."""

    semantic, _, info = baseline._load_model(rf_model_manifest)
    if info.get("role") != ROLE_FIT or info.get("stage") != ROLE_FIT:
        raise RoleAccessError("comparison_requires_fit_role_rf_model")
    try:
        declared_scale = int(info.get("train_scale"))
    except (TypeError, ValueError) as exc:
        raise ContractError("rf_model_train_scale_missing") from exc
    if declared_scale != int(train_scale):
        raise ContractError("rf_model_train_scale_mismatch")
    classes_raw = sorted(int(value) for value in info.get("classes_raw") or [])
    if not classes_raw:
        raise ContractError("rf_classes_raw_missing")

    def predict(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return baseline._predict_candidates(semantic, candidates)

    return ModelEvaluator("rf", predict, False), dict(info), classes_raw


def _checkpoint_normalisation(bundle: Mapping[str, Any]):
    return gnn.checkpoint_normalisation(bundle)


def _normalise_batch(batch: gnn.GraphBatch, stats: Any) -> gnn.GraphBatch:
    return gnn.normalise_batch(batch, stats)


def _label_free_batch(
    sample_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> gnn.GraphBatch:
    """Build a geometry-only graph batch; labels/association never enter."""

    return gnn.build_label_free_batch(sample_id, candidates)


def _predict_graphsage(
    bundle: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Predict from geometry only via the shared C1 inference path."""

    return gnn.predict_candidates(bundle, candidates)


def edge_ap_diagnostic(
    scored_by_sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Optional supervised edge diagnostic; never part of primary comparison.

    Scores are C1 edge-head logits; average precision only depends on ranking,
    so the monotone sigmoid would not change it.
    """

    from . import recognition_gnn_train as gnn_train

    by_uid = {candidate["candidate_uid"]: candidate for candidate in candidates}
    labels: List[int] = []
    scores: List[float] = []
    for payload in scored_by_sample.values():
        uids = list(payload.get("candidate_uids") or [])
        if any(uid not in by_uid for uid in uids):
            raise ContractError("edge_diagnostic_candidate_mismatch")
        group = [by_uid[uid] for uid in uids]
        index_by_uid = {uid: index for index, uid in enumerate(uids)}
        edges = list(payload.get("edges") or [])
        edge_index = [
            (index_by_uid[left], index_by_uid[right]) for left, right, _ in edges
        ]
        edge_labels, edge_mask = gnn_train._edge_labels(group, edge_index)
        for label, valid, (_, _, score) in zip(edge_labels, edge_mask, edges):
            if valid:
                labels.append(int(label))
                scores.append(float(score))
    positive = sum(labels)
    result = {
        "status": "blocked_insufficient_class_support",
        "average_precision": None,
        "support": len(labels),
        "positive_support": positive,
        "negative_support": len(labels) - positive,
        "note": "separate diagnostic; not the primary model comparison",
    }
    if labels and len(set(labels)) > 1:
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            result["status"] = "blocked_missing_sklearn"
            return result
        result["status"] = "ok"
        result["average_precision"] = float(average_precision_score(labels, scores))
    return result


def load_graphsage_evaluator(
    checkpoint_path: str,
    *,
    train_scale: int,
) -> tuple[ModelEvaluator, Dict[str, Any], List[int], Dict[str, Any]]:
    """Load the C1 checkpoint and expose a label-free predictor."""

    bundle = gnn.load_checkpoint(checkpoint_path)
    payload = dict(bundle.get("manifest") or {})
    declared_role = payload.get("role") or payload.get("stage")
    if declared_role is not None and declared_role not in (ROLE_FIT, "train"):
        raise RoleAccessError("comparison_requires_fit_role_gnn_checkpoint")
    if payload.get("train_scale") is not None:
        try:
            declared_scale = int(payload["train_scale"])
        except (TypeError, ValueError) as exc:
            raise ContractError("gnn_checkpoint_train_scale_invalid") from exc
        if declared_scale != int(train_scale):
            raise ContractError("gnn_checkpoint_train_scale_mismatch")
    classes_raw = sorted(int(value) for value in bundle.get("classes_raw") or [])
    if not classes_raw:
        raise ContractError("gnn_classes_raw_missing")
    captured: Dict[str, Any] = {}

    def predict(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        predictions, scored = _predict_graphsage(bundle, candidates)
        captured["scored_by_sample"] = scored
        return predictions

    return ModelEvaluator("graphsage", predict, True), payload, classes_raw, captured


def compare_tune(
    manifest_path: str,
    out: str,
    *,
    rf_model_manifest: str,
    gnn_checkpoint: str,
    train_scale: int = 32,
    train_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare fit-trained RF and GraphSAGE on one permanent tune set."""

    manifest = baseline._read_json(manifest_path)
    baseline.validate_manifest(manifest)
    tune_set = build_tune_candidates(
        manifest, train_scale=train_scale, train_root=train_root
    )
    if not tune_set.candidates:
        raise BlockedError("no_tune_candidates")

    rf_model, rf_info, rf_classes = load_rf_evaluator(
        rf_model_manifest, train_scale=train_scale
    )
    gnn_model, gnn_info, gnn_classes, captured = load_graphsage_evaluator(
        gnn_checkpoint, train_scale=train_scale
    )
    classes_raw = sorted(set(rf_classes) | set(gnn_classes))
    result = evaluate_models(
        tune_set.candidates,
        [rf_model, gnn_model],
        classes_raw=classes_raw,
        coverage=tune_set.coverage,
        allowed_sample_ids=tune_set.sample_ids,
        allowed_group_ids=tune_set.group_ids,
    )
    result["train_scale"] = train_scale
    result["tune_group_count"] = len(tune_set.group_ids)
    result["tune_sample_count"] = len(tune_set.sample_ids)
    result["model_artifacts"] = {
        "rf": {
            "role": rf_info.get("role"),
            "stage": rf_info.get("stage"),
            "train_scale": rf_info.get("train_scale"),
            "classes_raw": rf_classes,
        },
        "graphsage": {
            "role": gnn_info.get("role") or gnn_info.get("stage"),
            "train_scale": gnn_info.get("train_scale"),
            "classes_raw": gnn_classes,
            "config_fingerprint": gnn_info.get("config_fingerprint"),
        },
    }
    result["edge_ap_diagnostic"] = edge_ap_diagnostic(
        captured.get("scored_by_sample") or {},
        tune_set.candidates,
    )
    baseline._write_json(out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cadvec.recognition_tune_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare-tune")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--rf-model-manifest", required=True)
    compare.add_argument("--gnn-checkpoint", required=True)
    compare.add_argument("--train-scale", type=int, default=32)
    compare.add_argument("--train-root", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare-tune":
            result = compare_tune(
                args.manifest,
                args.out,
                rf_model_manifest=args.rf_model_manifest,
                gnn_checkpoint=args.gnn_checkpoint,
                train_scale=args.train_scale,
                train_root=args.train_root,
            )
        else:  # pragma: no cover
            raise ContractError("unknown command")
    except (RecognitionError, OSError, ValueError) as exc:
        print(f"error: {getattr(exc, 'code', type(exc).__name__)}: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
