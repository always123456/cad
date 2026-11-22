"""GraphSAGE C1 train/tune runner with strict train-only role reuse.

The current Windows venv has no PyTorch/PyG.  This module therefore exposes a
real, dependency-guarded train/tune path but returns ``blocked`` before reading
training sources when the stack is absent.  It never calls refit/freeze/
evaluate and never opens eval_pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import recognition_baseline as baseline
from . import recognition_candidates as candidates
from . import recognition_gnn as gnn
from . import recognition_edge_sampler as edge_sampler
from .recognition_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    ROLE_FIT,
    ROLE_TUNE,
    EvalSealedError,
    MissingDependencyError,
    RecognitionError,
    RoleAccessError,
    instance_mask,
    semantic_mask,
)

SCHEMA = "cadvec.recognition.gnn_train/1"
CHECKPOINT_SCHEMA = "cadvec.recognition.gnn_checkpoint/1"
DEFAULT_RUN_ROOT = os.path.join("outputs", "recognition-gnn")
EDGE_THRESHOLD = 0.5


class TrainerError(RecognitionError):
    code = "gnn_trainer_error"


@dataclass(frozen=True)
class SupervisionGraph:
    """A graph plus independent pair supervision kept outside GraphBatch."""

    batch: gnn.GraphBatch
    node_labels: Tuple[Optional[int], ...]
    node_mask: Tuple[bool, ...]
    edge_labels: Tuple[Optional[int], ...]
    edge_mask: Tuple[bool, ...]
    supervision_pairs: Tuple[Mapping[str, Any], ...] = ()
    pair_audit: Mapping[str, Any] = None


def dependency_status() -> Dict[str, Any]:
    return gnn.dependency_status()


def trainer_smoke() -> Dict[str, Any]:
    """Dependency/interface smoke; does not read data or train."""
    status = dependency_status()
    if not status["available"]:
        return {
            "schema": SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "missing_dependency",
            "missing": [name for name in ("torch", "torch_geometric") if not status[name]],
            "dependency_status": status,
            "model_id": gnn.C1_MODEL_ID,
            "config_fingerprint": gnn.C1_FINGERPRINT,
            "trained": False,
        }
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "ready",
        "model_id": gnn.C1_MODEL_ID,
        "config_fingerprint": gnn.C1_FINGERPRINT,
        "dependency_status": status,
        "trained": False,
    }


def _node_labels(candidates_for_sample: Sequence[Mapping[str, Any]]) -> Tuple[List[Optional[int]], List[bool]]:
    labels: List[Optional[int]] = []
    mask: List[bool] = []
    for candidate in candidates_for_sample:
        supervision = candidate.get("supervision") or {}
        association = candidate.get("association") or {}
        valid = semantic_mask(
            semantic_present=bool(supervision.get("semantic_raw") is not None),
            semantic_status=str(supervision.get("semantic_status") or ""),
            semantic_raw=supervision.get("semantic_raw"),
            geometry_ok=True,
            association_status=str(association.get("status") or ""),
        )
        labels.append(int(supervision["semantic_raw"]) if valid and supervision.get("semantic_raw") is not None else None)
        mask.append(bool(valid))
    return labels, mask


def _edge_labels(
    candidates_for_sample: Sequence[Mapping[str, Any]],
    edge_index: Sequence[Tuple[int, int]],
) -> Tuple[List[Optional[int]], List[bool]]:
    labels: List[Optional[int]] = []
    mask: List[bool] = []
    for left, right in edge_index:
        a = candidates_for_sample[left]
        b = candidates_for_sample[right]
        sa, sb = a.get("supervision") or {}, b.get("supervision") or {}
        aa, ab = a.get("association") or {}, b.get("association") or {}
        valid = (
            aa.get("status") == "unique"
            and ab.get("status") == "unique"
            and sa.get("semantic_raw") == sb.get("semantic_raw")
            and isinstance(sa.get("semantic_raw"), int)
            and 1 <= sa["semantic_raw"] <= 30
            and instance_mask(
                instance_present=bool(sa.get("instance_raw") is not None),
                instance_status=str(sa.get("instance_status") or ""),
                instance_raw=sa.get("instance_raw"),
                semantic_raw=sa.get("semantic_raw"),
                geometry_ok=True,
                association_status="unique",
            )
            and instance_mask(
                instance_present=bool(sb.get("instance_raw") is not None),
                instance_status=str(sb.get("instance_status") or ""),
                instance_raw=sb.get("instance_raw"),
                semantic_raw=sb.get("semantic_raw"),
                geometry_ok=True,
                association_status="unique",
            )
        )
        labels.append(int(sa["instance_raw"] == sb["instance_raw"]) if valid else None)
        mask.append(bool(valid))
    return labels, mask


def build_supervision_graph(
    sample_id: str,
    candidates_for_sample: Sequence[Mapping[str, Any]],
    *,
    role: Optional[str] = None,
) -> SupervisionGraph:
    batch = gnn.build_label_free_batch(sample_id, candidates_for_sample)
    node_labels, node_mask = _node_labels(candidates_for_sample)
    edge_labels, edge_mask = _edge_labels(candidates_for_sample, batch.edge_index)
    pair_result = edge_sampler.sample_edge_pairs(
        candidates_for_sample, gnn.label_free_nodes(candidates_for_sample), role=role
    )
    return SupervisionGraph(
        batch=batch,
        node_labels=tuple(node_labels),
        node_mask=tuple(node_mask),
        edge_labels=tuple(edge_labels),
        edge_mask=tuple(edge_mask),
        supervision_pairs=tuple(pair_result["pairs"]),
        pair_audit=pair_result["audit"],
    )


def edge_supervision_status(graphs: Sequence[SupervisionGraph]) -> Dict[str, Any]:
    labels = [int(pair["label"]) for graph in graphs for pair in graph.supervision_pairs]
    if not labels:
        labels = [label for graph in graphs for label, valid in zip(graph.edge_labels, graph.edge_mask) if valid]
    if not labels:
        return {"status": "blocked_no_edge_supervision", "valid_edges": 0, "classes": []}
    classes = sorted(set(labels))
    if len(classes) < 2:
        return {"status": "blocked_single_class", "valid_edges": len(labels), "classes": classes}
    return {"status": "available", "valid_edges": len(labels), "classes": classes}


def selected_source_rows(
    manifest: Mapping[str, Any],
    role: str,
    *,
    train_scale: int = 32,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Metadata-only role/source rows; called before any PNG/SVG read."""
    if role not in (ROLE_FIT, ROLE_TUNE):
        raise EvalSealedError("GNN trainer only accepts fit/tune")
    reader = baseline.RoleReader(manifest, role)
    groups = reader.group_ids(train_scale)
    rows = baseline._rows_for_groups(manifest, groups, role=role)
    return groups, rows


def build_graph_batches(
    manifest: Mapping[str, Any],
    role: str,
    *,
    train_scale: int = 32,
) -> List[SupervisionGraph]:
    """Reuse baseline's source rehash/path and role reader; no eval role."""
    _, rows = selected_source_rows(manifest, role, train_scale=train_scale)
    rows = baseline.validated_train_file_paths(manifest, rows)
    from . import png_frontend, svg_parse

    graphs: List[SupervisionGraph] = []
    for row in rows:
        document = svg_parse.parse_svg(
            row["svg_path"], split="train_final", record_commands=True, curve_steps=128
        )
        frontend = png_frontend.predict_png(
            row["png_path"],
            png_frontend.FrontendConfig(
                max_lines=4000, max_pixels=40_000_000, max_file_bytes=64 * 1024 * 1024
            ),
        )
        if frontend.get("status") not in ("ok", "empty_input", "no_lines_detected"):
            continue
        gt, mapping = baseline._pixel_gt_for_frontend(document, row["sample_id"], frontend)
        if mapping.get("status") != "ok" or any(not primitive.complete for primitive in gt):
            continue
        sample_candidates = candidates.build_png_candidates(
            frontend, row["sample_id"], group_id=row.get("group_id"), role=role
        )
        candidates.associate_png_candidates(sample_candidates, gt)
        if sample_candidates:
            graphs.append(build_supervision_graph(row["sample_id"], sample_candidates, role=role))
    return graphs


def _normalisation(graphs: Sequence[SupervisionGraph]) -> Optional[Tuple[List[float], List[float]]]:
    rows = [node for graph in graphs for node in graph.batch.node_features]
    if not rows:
        return None
    width = len(NODE_FEATURE_NAMES)
    means = [sum(row[i] for row in rows) / len(rows) for i in range(width)]
    variances = [sum((row[i] - means[i]) ** 2 for row in rows) / len(rows) for i in range(width)]
    return means, [math.sqrt(value) if value > 0 else 1.0 for value in variances]


def normalise_graph(graph: SupervisionGraph, stats: Optional[Tuple[Sequence[float], Sequence[float]]]) -> SupervisionGraph:
    if stats is None:
        return graph
    batch = gnn.normalise_batch(graph.batch, stats)
    return SupervisionGraph(batch, graph.node_labels, graph.node_mask, graph.edge_labels, graph.edge_mask, graph.supervision_pairs, graph.pair_audit)


def _torch_metrics() -> Dict[str, Any]:
    return dependency_status()


def build_checkpoint_manifest(
    *,
    classes_raw: Sequence[int],
    normalisation_stats: Optional[Tuple[Sequence[float], Sequence[float]]],
    artifact_sha256: Optional[str],
    dependency_versions: Optional[Mapping[str, Any]] = None,
    best_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "model_id": gnn.C1_MODEL_ID,
        "config": dict(gnn.C1_CONFIG),
        "config_fingerprint": gnn.C1_FINGERPRINT,
        "classes_raw": sorted(int(value) for value in classes_raw),
        "feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "normalisation_fit_only": (
            {
                "mean": [float(value) for value in normalisation_stats[0]],
                "scale": [float(value) for value in normalisation_stats[1]],
            }
            if normalisation_stats
            else None
        ),
        "dependency_versions": dict(dependency_versions or {}),
        "artifact_sha256": artifact_sha256,
        "best_epoch": best_epoch,
        "trained": artifact_sha256 is not None,
    }


def train_c1(
    manifest_path: str,
    run_dir: str,
    *,
    train_scale: int = 32,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train C1 on fit, early-stop on tune; never reads eval or refits."""
    status = trainer_smoke()
    if status["status"] == "blocked" and not os.path.isfile(manifest_path):
        return {
            **status,
            "run_dir": run_dir,
            "train_scale": train_scale,
            "trained": False,
        }
    manifest = baseline._read_json(manifest_path)
    baseline.validate_manifest(manifest)
    # Count real source rows from role/group metadata before any file read.
    fit_groups, fit_rows = selected_source_rows(manifest, ROLE_FIT, train_scale=train_scale)
    tune_groups, tune_rows = selected_source_rows(manifest, ROLE_TUNE, train_scale=train_scale)
    selected_source_pairs = len(fit_rows) + len(tune_rows)
    if selected_source_pairs > baseline.RUN_CAPS["max_pairs"]:
        return {
            "schema": SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "budget_exceeded",
            "message": f"selected_source_pairs>{baseline.RUN_CAPS['max_pairs']}",
            "train_scale": train_scale,
            "fit_groups": len(fit_groups),
            "tune_groups": len(tune_groups),
            "selected_source_pairs": selected_source_pairs,
            "processed_pairs": 0,
            "graphs": 0,
            "trained": False,
            "no_eval_access": True,
        }
    if status["status"] == "blocked":
        return {**status, "run_dir": run_dir, "train_scale": train_scale, "trained": False}
    fit_graphs = build_graph_batches(manifest, ROLE_FIT, train_scale=train_scale)
    tune_graphs = build_graph_batches(manifest, ROLE_TUNE, train_scale=train_scale)
    if any(graph.pair_audit and graph.pair_audit.get("status") == "blocked_pair_cap" for graph in fit_graphs + tune_graphs):
        return {"schema": SCHEMA, "version": 1, "status": "blocked", "reason": "budget_exceeded", "message": "edge_supervision_positive_pairs_exceed_per_sample_cap", "trained": False, "no_eval_access": True}
    if not fit_graphs or not tune_graphs:
        return {"schema": SCHEMA, "version": 1, "status": "blocked", "reason": "no_train_or_tune_graphs", "selected_source_pairs": selected_source_pairs, "processed_pairs": selected_source_pairs, "graphs": 0, "trained": False}
    stats = _normalisation(fit_graphs)
    fit_graphs = [normalise_graph(graph, stats) for graph in fit_graphs]
    tune_graphs = [normalise_graph(graph, stats) for graph in tune_graphs]
    edge_status = edge_supervision_status(fit_graphs)
    classes_raw = sorted({label for graph in fit_graphs for label, valid in zip(graph.node_labels, graph.node_mask) if valid})
    if not classes_raw:
        return {"schema": SCHEMA, "version": 1, "status": "blocked", "reason": "no_valid_node_semantic_rows", "selected_source_pairs": selected_source_pairs, "processed_pairs": selected_source_pairs, "graphs": len(fit_graphs) + len(tune_graphs), "trained": False}
    started = time.monotonic()
    run_counts = {
        "pairs": selected_source_pairs,
        "processed_pairs": selected_source_pairs,
        "graphs": len(fit_graphs) + len(tune_graphs),
        "candidates_total": sum(len(graph.batch.candidate_uids) for graph in fit_graphs + tune_graphs),
        "edges_total": sum(len(graph.batch.edge_index) for graph in fit_graphs + tune_graphs),
        "candidates_image": max((len(graph.batch.candidate_uids) for graph in fit_graphs + tune_graphs), default=0),
        "edges_image": max((len(graph.batch.edge_index) for graph in fit_graphs + tune_graphs), default=0),
    }
    try:
        caps_snapshot = baseline.enforce_run_caps(run_counts, started=started)
    except baseline.BudgetExceeded as exc:
        return {
            "schema": SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "budget_exceeded",
            "message": str(exc),
            "counts": run_counts,
        }
    torch, _ = gnn.require_torch()
    torch.manual_seed(int(gnn.C1_CONFIG["seed"]))
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return {"schema": SCHEMA, "version": 1, "status": "blocked", "reason": "cuda_unavailable"}
        torch.cuda.manual_seed_all(int(gnn.C1_CONFIG["seed"]))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    bundle = gnn.build_model(classes_raw)
    if device != "cpu":
        for module in (bundle["encoder"], bundle["node_head"], bundle["edge_head"]):
            module.to(device)
    bundle["device"] = device
    optimizer = torch.optim.AdamW(
        [parameter for module in (bundle["encoder"], bundle["node_head"], bundle["edge_head"]) for parameter in module.parameters()],
        lr=gnn.C1_CONFIG["lr"],
        weight_decay=gnn.C1_CONFIG["weight_decay"],
    )
    class_counts = {value: sum(1 for graph in fit_graphs for label, valid in zip(graph.node_labels, graph.node_mask) if valid and label == value) for value in classes_raw}
    weights = torch.tensor(
        [len(classes_raw) / max(class_counts[value], 1) for value in classes_raw],
        dtype=torch.float32,
        device=bundle["device"],
    )
    class_index = {value: index for index, value in enumerate(classes_raw)}
    best = {"epoch": -1, "macro_f1": -1.0, "edge_ap": None, "state": None}
    stale = 0
    history: List[Dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(min(max_epochs, gnn.C1_CONFIG["max_epochs"])):
        bundle["training"] = True
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=bundle["device"])
        for graph in fit_graphs:
            output = gnn.forward_batch(
                bundle, graph.batch, device=bundle["device"],
                pair_index=[pair["pair_index"] for pair in graph.supervision_pairs],
                pair_features=[[pair["features"][name] for name in EDGE_FEATURE_NAMES] for pair in graph.supervision_pairs],
            )
            node_targets = [class_index[label] for label, valid in zip(graph.node_labels, graph.node_mask) if valid]
            if node_targets:
                node_logits = output["node_logits"][[i for i, valid in enumerate(graph.node_mask) if valid]]
                total_loss = total_loss + torch.nn.functional.cross_entropy(
                    node_logits,
                    torch.tensor(node_targets, device=bundle["device"]),
                    weight=weights,
                )
            pair_targets = [int(pair["label"]) for pair in graph.supervision_pairs]
            if pair_targets and edge_status["status"] == "available":
                total_loss = total_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    output["pair_scores"],
                    torch.tensor(pair_targets, dtype=torch.float32, device=bundle["device"]),
                )
        total_loss.backward()
        optimizer.step()
        epoch_metrics = _evaluate_graphs(bundle, tune_graphs)
        history.append({"epoch": epoch, "loss": float(total_loss.detach()), **epoch_metrics})
        if epoch_metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "epoch": epoch,
                "macro_f1": epoch_metrics["macro_f1"],
                "edge_ap": epoch_metrics["edge_ap"],
                "state": {
                    "encoder": [
                        {key: value.detach().clone() for key, value in layer.state_dict().items()}
                        for layer in bundle["encoder"]
                    ],
                    "node_head": {key: value.detach().clone() for key, value in bundle["node_head"].state_dict().items()},
                    "edge_head": {key: value.detach().clone() for key, value in bundle["edge_head"].state_dict().items()},
                },
            }
            stale = 0
        else:
            stale += 1
        if stale >= min(patience, gnn.C1_CONFIG["patience"]):
            break
    os.makedirs(run_dir, exist_ok=True)
    checkpoint = os.path.join(run_dir, "graphsage_c1_best.pt")
    if os.path.exists(checkpoint):
        raise TrainerError("run_dir_already_has_checkpoint; use a new run nonce")
    state = best["state"] or {
        "encoder": [layer.state_dict() for layer in bundle["encoder"]],
        "node_head": bundle["node_head"].state_dict(),
        "edge_head": bundle["edge_head"].state_dict(),
    }
    torch.save(
        {
            "config": gnn.C1_CONFIG,
            "classes_raw": classes_raw,
            "encoder": state["encoder"],
            "node_head": state["node_head"],
            "edge_head": state["edge_head"],
            "normalisation": stats,
        },
        checkpoint,
    )
    digest = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
    result = {
        "schema": SCHEMA,
        "version": 1,
        "status": "partial" if edge_status["status"] != "available" else "complete",
        "model_id": gnn.C1_MODEL_ID,
        "config": dict(gnn.C1_CONFIG),
        "config_fingerprint": gnn.C1_FINGERPRINT,
        "feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "train_scale": train_scale,
        "device": device,
        "selected_source_pairs": selected_source_pairs,
        "processed_pairs": selected_source_pairs,
        "graphs": len(fit_graphs) + len(tune_graphs),
        "fit_groups": len(fit_groups),
        "tune_groups": len(tune_groups),
        "fit_graphs": len(fit_graphs),
        "tune_graphs": len(tune_graphs),
        "classes_raw": classes_raw,
        "edge_supervision": edge_status,
        "edge_pair_audit": {
            "fit": [dict(graph.pair_audit or {}) for graph in fit_graphs],
            "tune": [dict(graph.pair_audit or {}) for graph in tune_graphs],
        },
        "best": {key: value for key, value in best.items() if key != "state"},
        "history": history,
        "elapsed_s": round(time.monotonic() - started, 3),
        "caps": dict(baseline.RUN_CAPS),
        "caps_snapshot": caps_snapshot,
        "resource_peak": {
            **baseline.resource_snapshot(started),
            "peak_gpu_memory_mb": (
                round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 3)
                if device.startswith("cuda")
                else None
            ),
            "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "cuda_version": getattr(torch.version, "cuda", None),
        },
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "dependency_versions": status.get("dependency_status", {}).get("versions", {}),
        "normalisation_fit_only": bool(stats is not None),
        "normalisation": (
            {"mean": [float(value) for value in stats[0]], "scale": [float(value) for value in stats[1]]}
            if stats
            else None
        ),
        "seed": gnn.C1_CONFIG["seed"],
        "no_eval_access": True,
    }
    with open(os.path.join(run_dir, "train_manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def _evaluate_graphs(bundle: Mapping[str, Any], graphs: Sequence[SupervisionGraph]) -> Dict[str, Any]:
    # Torch-dependent metrics; kept separate so synthetic/blocked paths need no torch.
    torch, _ = gnn.require_torch()
    from sklearn.metrics import average_precision_score
    y_true, y_pred = [], []
    edge_y, edge_p = [], []
    for graph in graphs:
        output = gnn.forward_batch(
            bundle, graph.batch, device=bundle.get("device"),
            pair_index=[pair["pair_index"] for pair in graph.supervision_pairs],
            pair_features=[[pair["features"][name] for name in EDGE_FEATURE_NAMES] for pair in graph.supervision_pairs],
        )
        node_scores = output["node_logits"].detach().cpu()
        pred = node_scores.argmax(dim=1).tolist()
        for index, (label, valid) in enumerate(zip(graph.node_labels, graph.node_mask)):
            if valid:
                y_true.append(label)
                y_pred.append(output["classes_raw"][pred[index]])
        for index, pair in enumerate(graph.supervision_pairs):
            edge_y.append(int(pair["label"]))
            edge_p.append(float(output["pair_scores"][index].detach().cpu()))
    classes = sorted(set(y_true))
    f1_values = []
    for value in classes:
        tp = sum(1 for truth, predicted in zip(y_true, y_pred) if truth == value and predicted == value)
        fp = sum(1 for truth, predicted in zip(y_true, y_pred) if truth != value and predicted == value)
        fn = sum(1 for truth, predicted in zip(y_true, y_pred) if truth == value and predicted != value)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    edge_ap = None
    if edge_y and len(set(edge_y)) > 1:
        edge_ap = float(average_precision_score(edge_y, edge_p))
    return {"macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0, "edge_ap": edge_ap}


PREDICT_SCHEMA = "cadvec.recognition.gnn_predict_png/1"
FRONTEND_LIMITS = {
    "max_lines": 4000,
    "max_pixels": 40_000_000,
    "max_file_bytes": 64 * 1024 * 1024,
}
PNG_ONLY_MESSAGE = "predict-png accepts exactly one PNG file"


def single_png_path(png: Any) -> str:
    """Enforce one existing ``.png`` file; SVG/paired/multiple inputs are refused."""

    if isinstance(png, (list, tuple)):
        # The CLI collects repeated --png flags; exactly one is allowed.
        if len(png) != 1:
            raise TrainerError(PNG_ONLY_MESSAGE)
        png = png[0]
    if not isinstance(png, str) or not png.strip():
        raise TrainerError(PNG_ONLY_MESSAGE)
    parts = [part.strip() for part in re.split(r"[,;]", png) if part.strip()]
    if len(parts) != 1:
        raise TrainerError(PNG_ONLY_MESSAGE)
    path = parts[0]
    if os.path.isdir(path):
        raise TrainerError(PNG_ONLY_MESSAGE)
    if not os.path.isfile(path):
        raise TrainerError(f"png_not_found: {path}")
    if os.path.splitext(path)[1].lower() != ".png":
        raise TrainerError("predict-png accepts only a .png file; no SVG or paired input")
    return path


def predict_png(checkpoint: str, png: Any, out: str) -> Dict[str, Any]:
    """PNG-only C1 inference using just the checkpoint manifest.

    Reads one PNG, builds PNG candidates without any GT/association pass and
    runs the shared label-free graph/inference path.  No public manifest,
    role reader, SVG parse or label source is touched.
    """

    path = single_png_path(png)
    from . import png_frontend

    result = png_frontend.predict_png(path, png_frontend.FrontendConfig(**FRONTEND_LIMITS))
    if result.get("status") not in ("ok", "empty_input", "no_lines_detected"):
        raise TrainerError(f"frontend status: {result.get('status')}")
    sample_id = "png:" + os.path.basename(path)
    sample_candidates = candidates.build_png_candidates(result, sample_id)
    model_input = [gnn.label_free_candidate(item) for item in sample_candidates]
    bundle = gnn.load_checkpoint(checkpoint)
    predictions, _ = gnn.predict_candidates(bundle, model_input)
    baseline._write_jsonl(out, predictions)
    return {
        "schema": PREDICT_SCHEMA,
        "version": 1,
        "status": "ok",
        "png_only": True,
        "checkpoint": checkpoint,
        "model_id": bundle.get("model_id", gnn.C1_MODEL_ID),
        "config_fingerprint": bundle.get("config_fingerprint", gnn.C1_FINGERPRINT),
        "classes_raw": sorted(int(value) for value in bundle["classes_raw"]),
        "sample_id": sample_id,
        "candidate_count": len(sample_candidates),
        "prediction_count": len(predictions),
        "target_all_null": all(row["target"] is None for row in predictions),
        "association_all_null": all(row["association"] is None for row in predictions),
        "out": out,
        "no_eval_access": True,
    }


def checkpoint_roundtrip_smoke() -> Dict[str, Any]:
    """Prove save -> load -> predict is bit-identical on the real C1 model."""

    status = trainer_smoke()
    if status["status"] != "ready":
        return status
    torch, _ = gnn.require_torch()
    torch.manual_seed(int(gnn.C1_CONFIG["seed"]))
    bundle = gnn.build_model([1, 2, 31, 35])
    batch = gnn.synthetic_batch()
    with torch.no_grad():
        before, _ = gnn.predict_batch(bundle, batch)
        with tempfile.TemporaryDirectory(prefix="gnn-roundtrip-") as tmp:
            checkpoint = os.path.join(tmp, "graphsage_c1_smoke.pt")
            gnn.save_checkpoint(bundle, checkpoint)
            loaded = gnn.load_checkpoint(checkpoint)
            after, _ = gnn.predict_batch(loaded, batch)
    identical = before == after
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "ok" if identical else "mismatch",
        "model_id": gnn.C1_MODEL_ID,
        "config_fingerprint": gnn.C1_FINGERPRINT,
        "save_load_predict": True,
        "prediction_count": len(after),
        "predictions_identical": identical,
        "classes_preserved": list(loaded["classes_raw"]) == list(bundle["classes_raw"]),
        "target_all_null": all(row["target"] is None for row in after),
        "association_all_null": all(row["association"] is None for row in after),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cadvec.recognition_gnn_train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    sub.add_parser("roundtrip")
    predict = sub.add_parser("predict-png")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--png", required=True, action="append", help="exactly one PNG file")
    predict.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--train-scale", type=int, default=32)
    train.add_argument("--device", default="cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = trainer_smoke()
        elif args.command == "roundtrip":
            result = checkpoint_roundtrip_smoke()
        elif args.command == "predict-png":
            result = predict_png(args.checkpoint, args.png, args.out)
        else:
            result = train_c1(
                args.manifest, args.run_dir, train_scale=args.train_scale, device=args.device
            )
    except (RecognitionError, OSError, ValueError) as exc:
        print(f"error: {getattr(exc, 'code', type(exc).__name__)}: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("status") in ("ready", "complete", "partial", "ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
