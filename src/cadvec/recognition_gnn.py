"""GraphSAGE C1 preparation and dependency-guarded synthetic smoke.

This module intentionally imports PyTorch/PyG lazily.  When they are absent,
callers get an explicit ``MissingDependencyError`` or a ``blocked`` smoke
result; nothing is installed and no training is attempted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import recognition_features as feature_mod
from .recognition_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    PREDICTION_SCHEMA,
    RAW_SEMANTIC_MAX,
    RAW_SEMANTIC_MIN,
    THING_MAX,
    THING_MIN,
    ContractError,
    MissingDependencyError,
    RecognitionError,
    is_raw_semantic,
)

C1_MODEL_ID = "graphsage-c1-v1"
C1_SCHEMA = "cadvec.recognition.gnn/1"
C1_CONFIG = {
    "layers": 2,
    "hidden": 128,
    "aggr": "mean",
    "root_weight": True,
    "normalize": False,
    "dropout": 0.2,
    "edge_head": "abs_diff+sum+edge_features",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
    "seed": 20261108,
}
C1_FINGERPRINT = hashlib.sha256(
    json.dumps(C1_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
C1_CLASS_MIN = RAW_SEMANTIC_MIN
C1_CLASS_MAX = RAW_SEMANTIC_MAX
EDGE_THRESHOLD = 0.5


class GnnError(RecognitionError):
    code = "gnn_error"


class GraphSchemaError(GnnError, ValueError):
    code = "graph_schema_error"


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """Label-free node/edge tensors/features for one synthetic or real graph."""

    node_features: Tuple[Tuple[float, ...], ...]
    edge_index: Tuple[Tuple[int, int], ...]
    edge_features: Tuple[Tuple[float, ...], ...]
    candidate_uids: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    source_kinds: Tuple[str, ...]


def dependency_status() -> Dict[str, Any]:
    """Check availability without importing or installing either package."""
    result: Dict[str, Any] = {
        "torch": importlib.util.find_spec("torch") is not None,
        "torch_geometric": importlib.util.find_spec("torch_geometric") is not None,
        "available": False,
        "versions": {},
    }
    if result["torch"] and result["torch_geometric"]:
        try:
            import torch
            import torch_geometric
            result["versions"] = {
                "torch": getattr(torch, "__version__", None),
                "torch_geometric": getattr(torch_geometric, "__version__", None),
            }
            result["available"] = True
        except Exception as exc:  # pragma: no cover - broken install
            result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def require_torch():
    status = dependency_status()
    if not status["available"]:
        missing = [name for name in ("torch", "torch_geometric") if not status[name]]
        raise MissingDependencyError(
            "GraphSAGE requires " + ", ".join(missing)
        )
    import torch
    from torch_geometric.nn import SAGEConv
    return torch, SAGEConv


def validate_classes_raw(classes_raw: Sequence[Any]) -> Tuple[int, ...]:
    values = tuple(int(value) for value in classes_raw)
    if not values:
        raise GraphSchemaError("classes_raw must not be empty")
    if len(set(values)) != len(values):
        raise GraphSchemaError("classes_raw must be unique")
    if any(not is_raw_semantic(value) for value in values):
        raise GraphSchemaError("classes_raw must lie in raw 1..35")
    return values


def build_model(classes_raw: Sequence[Any], config: Optional[Mapping[str, Any]] = None):
    """Construct C1 modules; raises MissingDependencyError when PyG is absent."""
    torch, SAGEConv = require_torch()
    values = validate_classes_raw(classes_raw)
    cfg = dict(C1_CONFIG)
    if config:
        cfg.update(config)
    if cfg != C1_CONFIG:
        raise GraphSchemaError("only the frozen C1 config is supported")
    encoder = torch.nn.ModuleList(
        [
            SAGEConv(
                len(NODE_FEATURE_NAMES),
                int(cfg["hidden"]),
                aggr=cfg["aggr"],
                root_weight=cfg["root_weight"],
                normalize=cfg["normalize"],
            ),
            SAGEConv(
                int(cfg["hidden"]),
                int(cfg["hidden"]),
                aggr=cfg["aggr"],
                root_weight=cfg["root_weight"],
                normalize=cfg["normalize"],
            ),
        ]
    )
    node_head = torch.nn.Linear(int(cfg["hidden"]), len(values))
    edge_head = torch.nn.Sequential(
        torch.nn.Linear(2 * int(cfg["hidden"]) + len(EDGE_FEATURE_NAMES), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    return {
        "schema": C1_SCHEMA,
        "version": 1,
        "model_id": C1_MODEL_ID,
        "config": cfg,
        "config_fingerprint": C1_FINGERPRINT,
        "classes_raw": list(values),
        "feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "encoder": encoder,
        "node_head": node_head,
        "edge_head": edge_head,
    }


def forward_batch(
    bundle: Mapping[str, Any], batch: GraphBatch, *, device: Optional[str] = None,
    pair_index: Sequence[Sequence[int]] = (),
    pair_features: Sequence[Sequence[float]] = (),
) -> Dict[str, Any]:
    """Forward the unchanged graph adjacency and optionally score independent pairs."""
    torch, _ = require_torch()
    cfg = bundle["config"]
    x = torch.tensor(batch.node_features, dtype=torch.float32)
    edge_index = torch.tensor(batch.edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(batch.edge_features, dtype=torch.float32)
    if device:
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
    h = x
    for layer in bundle["encoder"]:
        h = layer(h, edge_index)
        h = torch.relu(h)
        h = torch.nn.functional.dropout(h, p=float(cfg["dropout"]), training=bundle.get("training", False))
    node_logits = bundle["node_head"](h)
    if edge_index.numel() == 0:
        edge_scores = torch.empty((0,), dtype=torch.float32)
    else:
        src, dst = edge_index[0], edge_index[1]
        symmetric = torch.cat(
            [torch.abs(h[src] - h[dst]), h[src] + h[dst], edge_attr], dim=1
        )
        edge_scores = bundle["edge_head"](symmetric).squeeze(-1)
    if pair_index:
        pair_edges = torch.tensor(pair_index, dtype=torch.long).t().contiguous()
        pair_attr = torch.tensor(pair_features, dtype=torch.float32)
        if device:
            pair_edges = pair_edges.to(device)
            pair_attr = pair_attr.to(device)
        src_pair, dst_pair = pair_edges[0], pair_edges[1]
        pair_input = torch.cat(
            [torch.abs(h[src_pair] - h[dst_pair]), h[src_pair] + h[dst_pair], pair_attr], dim=1
        )
        pair_scores = bundle["edge_head"](pair_input).squeeze(-1)
    else:
        pair_scores = torch.empty((0,), dtype=torch.float32)
    return {
        "node_logits": node_logits,
        "edge_scores": edge_scores,
        "pair_scores": pair_scores,
        "classes_raw": list(bundle["classes_raw"]),
    }


def _sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid, saturating without overflow."""

    score = float(value)
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exp = math.exp(score)
    return exp / (1.0 + exp)


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(float(value) - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps] if total else [0.0] * len(values)


def predictions_from_scores(
    *,
    candidate_uids: Sequence[str],
    sample_ids: Sequence[str],
    source_kinds: Sequence[str],
    classes_raw: Sequence[Any],
    semantic_logits: Sequence[Sequence[float]],
    edge_index: Sequence[Sequence[int]] = (),
    edge_scores: Sequence[float] = (),
    threshold: float = EDGE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Build strict prediction records without accepting labels/supervision.

    ``edge_scores`` are the C1 edge head's raw logits (training uses
    ``binary_cross_entropy_with_logits``); each logit is converted with a stable
    sigmoid and the probability is compared against the probability threshold.

    Instance grouping follows the inference contract: an edge may join two
    candidates only when its probability reaches ``threshold``, both endpoints are
    predicted Thing raw semantics (1..30), and both share the same predicted
    raw semantic.  Stuff candidates (31..35) never receive an instance group
    and never take part in a grouping edge.
    """
    classes = validate_classes_raw(classes_raw)
    if not (len(candidate_uids) == len(sample_ids) == len(source_kinds) == len(semantic_logits)):
        raise GraphSchemaError("prediction arrays must have equal length")
    for row in semantic_logits:
        if len(row) != len(classes):
            raise GraphSchemaError("semantic logits width must equal classes_raw")
    probabilities_by_index: List[List[float]] = []
    predicted_raw: List[int] = []
    predicted_index: List[int] = []
    for row in semantic_logits:
        probabilities = _softmax(row)
        class_index = max(range(len(classes)), key=lambda item: probabilities[item])
        probabilities_by_index.append(probabilities)
        predicted_index.append(class_index)
        predicted_raw.append(int(classes[class_index]))
    parent = list(range(len(candidate_uids)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for (left, right), score in zip(edge_index, edge_scores):
        left_index, right_index = int(left), int(right)
        if left_index == right_index:
            continue
        if _sigmoid(score) < threshold:
            continue
        left_class = predicted_raw[left_index]
        right_class = predicted_raw[right_index]
        if left_class != right_class:
            continue
        if not (THING_MIN <= left_class <= THING_MAX):
            continue
        union(left_index, right_index)
    root_members: Dict[int, List[int]] = {}
    for index in range(len(candidate_uids)):
        root_members.setdefault(find(index), []).append(index)
    canonical = {
        member: min(members)
        for members in root_members.values()
        for member in members
    }
    groups = {index: canonical[index] for index in range(len(candidate_uids))}
    output: List[Dict[str, Any]] = []
    for index, uid in enumerate(candidate_uids):
        probabilities = probabilities_by_index[index]
        raw = predicted_raw[index]
        output.append(
            {
                "schema": PREDICTION_SCHEMA,
                "version": 1,
                "candidate_uid": str(uid),
                "sample_id": str(sample_ids[index]),
                "source_kind": str(source_kinds[index]),
                "semantic_raw_pred": raw,
                "semantic_proba": float(probabilities[predicted_index[index]]),
                "instance_group": f"g{groups[index]}" if THING_MIN <= raw <= THING_MAX else None,
                "target": None,
                "association": None,
            }
        )
    return output


LABEL_FIELDS = ("supervision", "association")


def label_free_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Model-visible projection: identity + geometry only, no labels."""

    return {key: value for key, value in candidate.items() if key not in LABEL_FIELDS}


def label_free_nodes(
    candidates: Sequence[Mapping[str, Any]],
) -> List["feature_mod.NodeGeometry"]:
    """Geometry-only node view; supervision/association are dropped here."""

    return [feature_mod.node_geometry_from_candidate(item) for item in candidates]


def build_label_free_batch(
    sample_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> GraphBatch:
    """The single label-free graph builder used by training and inference.

    Only identity and pixel geometry enter; supervision/association records are
    ignored even when present on the candidate mappings.
    """

    nodes = label_free_nodes(candidates)
    node_features = feature_mod.extract_node_features(nodes)
    edges, _ = feature_mod.build_local_graph(nodes)
    edge_features = feature_mod.extract_edge_features(nodes, edges)
    index_by_uid = {
        candidate["candidate_uid"]: index for index, candidate in enumerate(candidates)
    }
    return GraphBatch(
        node_features=tuple(
            tuple(feature[name] for name in NODE_FEATURE_NAMES) for feature in node_features
        ),
        edge_index=tuple(
            (index_by_uid[edge.source_uid], index_by_uid[edge.target_uid]) for edge in edges
        ),
        edge_features=tuple(
            tuple(feature[name] for name in EDGE_FEATURE_NAMES) for feature in edge_features
        ),
        candidate_uids=tuple(candidate["candidate_uid"] for candidate in candidates),
        sample_ids=tuple(sample_id for _ in candidates),
        source_kinds=tuple(str(candidate.get("source_kind") or "") for candidate in candidates),
    )


def checkpoint_normalisation(bundle: Mapping[str, Any]):
    """Read the fit-only normalisation recorded inside a checkpoint."""

    document = (bundle.get("manifest") or {}).get("normalisation")
    if document is None:
        return None
    if isinstance(document, Mapping):
        if "mean" in document and "scale" in document:
            return document["mean"], document["scale"]
    if isinstance(document, (list, tuple)) and len(document) == 2:
        return document[0], document[1]
    raise ContractError("gnn_checkpoint_normalisation_invalid")


def normalise_batch(batch: GraphBatch, stats: Any) -> GraphBatch:
    """Apply checkpoint normalisation to node features only; adjacency is frozen."""

    if stats is None:
        return batch
    means, scales = stats
    if len(means) != len(NODE_FEATURE_NAMES) or len(scales) != len(NODE_FEATURE_NAMES):
        raise ContractError("gnn_checkpoint_normalisation_width_mismatch")
    rows = tuple(
        tuple(
            (float(value) - float(means[index])) / float(scales[index])
            for index, value in enumerate(row)
        )
        for row in batch.node_features
    )
    return GraphBatch(
        node_features=rows,
        edge_index=batch.edge_index,
        edge_features=batch.edge_features,
        candidate_uids=batch.candidate_uids,
        sample_ids=batch.sample_ids,
        source_kinds=batch.source_kinds,
    )


def attach_geometry(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach candidate pixel geometry; geometry never comes from the model."""

    by_uid = {candidate["candidate_uid"]: candidate for candidate in candidates}
    output: List[Dict[str, Any]] = []
    for prediction in predictions:
        record = dict(prediction)
        candidate = by_uid.get(record.get("candidate_uid"))
        if candidate is None:
            raise GraphSchemaError(f"extra_prediction_uid:{record.get('candidate_uid')}")
        if record.get("sample_id") != candidate.get("sample_id"):
            raise GraphSchemaError(f"prediction_sample_mismatch:{record.get('candidate_uid')}")
        if record.get("source_kind") != candidate.get("source_kind"):
            raise GraphSchemaError(f"prediction_source_mismatch:{record.get('candidate_uid')}")
        if "geometry_px" not in record:
            geometry = candidate.get("geometry") or {}
            record["geometry_px"] = geometry.get("points_px") or geometry.get("points")
        output.append(record)
    return output


def predict_batch(
    bundle: Mapping[str, Any],
    batch: GraphBatch,
    *,
    threshold: float = EDGE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Forward one graph and turn its scores into strict prediction records."""

    output = forward_batch(bundle, batch)
    scores = output["edge_scores"].detach().cpu().tolist()
    records = predictions_from_scores(
        candidate_uids=batch.candidate_uids,
        sample_ids=batch.sample_ids,
        source_kinds=batch.source_kinds,
        classes_raw=bundle["classes_raw"],
        semantic_logits=output["node_logits"].detach().cpu().tolist(),
        edge_index=batch.edge_index,
        edge_scores=scores,
        threshold=threshold,
    )
    return records, [float(score) for score in scores]


def predict_candidates(
    bundle: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    threshold: float = EDGE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The single label-free inference path: per sample graph -> records.

    Returns predictions ordered like ``candidates`` plus per-sample edge scores
    for diagnostics.  Edge entries keep the raw logits (monotone ranking for the
    edge-AP diagnostic); ``edge_probabilities`` carries the threshold-space
    probabilities.  No manifest, role, label or paired file is touched.
    """

    torch, _ = require_torch()
    by_sample: Dict[str, List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        by_sample.setdefault(str(candidate["sample_id"]), []).append(
            label_free_candidate(candidate)
        )
    stats = checkpoint_normalisation(bundle)
    if isinstance(bundle, dict):
        bundle["training"] = False
    predictions_by_uid: Dict[str, Dict[str, Any]] = {}
    scored_by_sample: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for sample_id, group in by_sample.items():
            batch = normalise_batch(build_label_free_batch(sample_id, group), stats)
            records, scores = predict_batch(bundle, batch, threshold=threshold)
            for record in attach_geometry(group, records):
                predictions_by_uid[record["candidate_uid"]] = record
            scored_by_sample[sample_id] = {
                "candidate_uids": list(batch.candidate_uids),
                "edge_score_kind": "logit",
                "edges": [
                    (
                        batch.candidate_uids[int(left)],
                        batch.candidate_uids[int(right)],
                        float(score),
                    )
                    for (left, right), score in zip(batch.edge_index, scores)
                ],
                "edge_probabilities": [
                    _sigmoid(score) for score in scores
                ],
            }

    ordered: List[Dict[str, Any]] = []
    for candidate in candidates:
        record = predictions_by_uid.pop(candidate["candidate_uid"], None)
        if record is None:
            raise GraphSchemaError(f"missing_predictions:{candidate['candidate_uid']}")
        ordered.append(record)
    if predictions_by_uid:
        raise GraphSchemaError("extra_prediction_uid:" + ",".join(sorted(predictions_by_uid)))
    return ordered, scored_by_sample


def synthetic_batch() -> GraphBatch:
    """Three collinear/adjacent segments, using the frozen feature widths."""
    nodes = (
        tuple(0.0 for _ in NODE_FEATURE_NAMES),
        tuple(0.0 for _ in NODE_FEATURE_NAMES),
        tuple(0.0 for _ in NODE_FEATURE_NAMES),
    )
    return GraphBatch(
        node_features=nodes,
        edge_index=((0, 1), (1, 2)),
        edge_features=tuple(tuple(0.0 for _ in EDGE_FEATURE_NAMES) for _ in range(2)),
        candidate_uids=("s#l0", "s#l1", "s#l2"),
        sample_ids=("s", "s", "s"),
        source_kinds=("png_line", "png_line", "png_line"),
    )


def synthetic_smoke() -> Dict[str, Any]:
    """Dependency-guarded smoke; never installs or trains."""
    status = dependency_status()
    if not status["available"]:
        return {
            "schema": C1_SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "missing_dependency",
            "missing": [name for name in ("torch", "torch_geometric") if not status[name]],
            "dependency_status": status,
        }
    bundle = build_model([1, 2, 31, 35])
    batch = synthetic_batch()
    started = time.perf_counter()
    output = forward_batch(bundle, batch)
    predictions = predictions_from_scores(
        candidate_uids=batch.candidate_uids,
        sample_ids=batch.sample_ids,
        source_kinds=batch.source_kinds,
        classes_raw=bundle["classes_raw"],
        semantic_logits=output["node_logits"].detach().cpu().tolist(),
        edge_index=batch.edge_index,
        edge_scores=output["edge_scores"].detach().cpu().tolist(),
    )
    return {
        "schema": C1_SCHEMA,
        "version": 1,
        "status": "ok",
        "model_id": C1_MODEL_ID,
        "config_fingerprint": C1_FINGERPRINT,
        "dependency_status": status,
        "prediction_count": len(predictions),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "target_all_null": all(item["target"] is None for item in predictions),
    }


def save_checkpoint(bundle: Mapping[str, Any], path: str, *, manifest_path: Optional[str] = None) -> Dict[str, Any]:
    torch, _ = require_torch()
    payload = {
        "schema": C1_SCHEMA,
        "version": 1,
        "model_id": C1_MODEL_ID,
        "config": dict(bundle["config"]),
        "config_fingerprint": C1_FINGERPRINT,
        "classes_raw": list(bundle["classes_raw"]),
        "feature_names": list(bundle["feature_names"]),
        "edge_feature_names": list(bundle["edge_feature_names"]),
        "encoder": [layer.state_dict() for layer in bundle["encoder"]],
        "node_head": bundle["node_head"].state_dict(),
        "edge_head": bundle["edge_head"].state_dict(),
        "dependency_versions": dependency_status()["versions"],
        "seed": C1_CONFIG["seed"],
    }
    torch.save(payload, path)
    result = {
        "schema": payload["schema"],
        "version": payload["version"],
        "model_id": payload["model_id"],
        "config": payload["config"],
        "config_fingerprint": payload["config_fingerprint"],
        "classes_raw": payload["classes_raw"],
        "feature_names": payload["feature_names"],
        "edge_feature_names": payload["edge_feature_names"],
        "dependency_versions": payload["dependency_versions"],
        "seed": payload["seed"],
        "artifact_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    }
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return result


def load_checkpoint(path: str, classes_raw: Optional[Sequence[Any]] = None):
    torch, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bundle = build_model(classes_raw or payload["classes_raw"], payload.get("config"))
    for layer, state in zip(bundle["encoder"], payload["encoder"]):
        layer.load_state_dict(state)
    bundle["node_head"].load_state_dict(payload["node_head"])
    bundle["edge_head"].load_state_dict(payload["edge_head"])
    bundle["manifest"] = payload
    return bundle
