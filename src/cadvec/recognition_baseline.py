"""Gate3B public recognition base: roles, candidates, RF smoke and isolation.

The module is intentionally conservative: it never opens eval_pool before a
frozen model exists, never parses labels in ``prepare``, and reports blocked
state instead of fabricating a model or metrics when dependencies/data are
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import recognition_candidates as candidate_mod
from . import recognition_features as feature_mod
from . import recognition_metrics as metrics_mod
from .recognition_contract import (
    AUDIT_GROUP_IDS,
    CANDIDATE_SCHEMA,
    EVAL_POOL_SIZE,
    FIT_SEQUENCE_SIZE,
    MODEL_SCHEMA,
    PREDICTION_SCHEMA,
    ROLE_EVAL,
    ROLE_FIT,
    ROLE_RESERVE,
    ROLE_TUNE,
    SCHEMA,
    SCHEMA_VERSION,
    SEED,
    TUNE_SEQUENCE_SIZE,
    BlockedError,
    BudgetExceeded,
    ContractError,
    EvalSealedError,
    MissingDependencyError,
    RecognitionError,
    RoleAccessError,
    assign_permanent_roles,
    classes_raw_from_values,
    instance_mask,
    semantic_mask,
    train_prefixes,
    validate_manifest,
)

TRAIN_ROOT = "train_artifacts"
EVAL_ROOT = "eval_artifacts"
FROZEN_FILE = "frozen.json"
MANIFEST_FILE = "recognition_public_manifest.json"
MODEL_FILE = "semantic.joblib"
EDGE_MODEL_FILE = "edge.joblib"
RUN_CAPS = {
    "max_pairs": 512,
    "max_primitives": 142_086,
    "max_candidates_per_image": 4_000,
    "max_candidates_total": 200_000,
    "max_edges_per_image": 50_000,
    "max_edges_total": 1_000_000,
    "max_elapsed_s": 1800,
    "max_rss_mb": 4_096,
}

REGISTRY = {
    "semantic": {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": SEED,
        "n_jobs": 2,
    },
    "edge": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": 2,
        "threshold": 0.5,
    },
}
REGISTRY_FINGERPRINT = hashlib.sha256(
    json.dumps(REGISTRY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _is_eval_path(path: Optional[str]) -> bool:
    if not path:
        return False
    parts = [part.lower() for part in os.path.normpath(path).split(os.sep)]
    return any(part in ("eval", "eval_artifacts", "evaluation") for part in parts)


class RoleReader:
    """Role-scoped metadata reader; train roles cannot touch eval roots."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        role: str,
        *,
        root: Optional[str] = None,
        allow_eval: bool = False,
        cloud_pack: bool = False,
    ) -> None:
        validate_manifest(manifest)
        if role not in (ROLE_FIT, ROLE_TUNE, ROLE_EVAL, ROLE_RESERVE):
            raise ContractError(f"unknown role: {role!r}")
        if role == ROLE_EVAL and not allow_eval:
            raise EvalSealedError("eval_pool is sealed until freeze")
        if cloud_pack and (role == ROLE_EVAL or _is_eval_path(root)):
            raise RoleAccessError("cloud_pack_eval_forbidden")
        if role != ROLE_EVAL and _is_eval_path(root):
            raise RoleAccessError("train_reader_cannot_read_eval_root")
        if role == ROLE_EVAL and root is not None and not _is_eval_path(root):
            raise RoleAccessError("evaluate_reader_requires_eval_root")
        self.manifest = manifest
        self.role = role
        self.root = root
        self.allow_eval = allow_eval
        self.cloud_pack = cloud_pack

    def group_ids(self, scale: Optional[int] = None) -> List[str]:
        roles = self.manifest["role_assignment"]
        if self.role == ROLE_EVAL:
            return list(roles["eval_pool"])
        if self.role == ROLE_RESERVE:
            return list(roles["reserve"])
        if scale is None:
            return list(roles[f"{self.role}_sequence"])
        return list(train_prefixes(roles, scale)[self.role])


def assert_cloud_pack_boundary(
    manifest: Mapping[str, Any],
    packaged_paths: Sequence[str],
    *,
    allowlist: Sequence[str] = (),
) -> None:
    """Cloud packing requires an explicit realpath allowlist.

    A path name is not sufficient evidence: the resolved path must be listed,
    and a listed file whose SHA-256 is an eval-role file hash is rejected.
    Without an allowlist cloud packing remains blocked rather than claiming
    isolation by naming convention.
    """

    validate_manifest(manifest)
    if manifest.get("include_eval") or manifest.get("eval_artifacts"):
        raise RoleAccessError("cloud_pack_eval_manifest_forbidden")
    if not allowlist:
        raise BlockedError("cloud_pack_allowlist_required")
    allowed = {os.path.realpath(path) for path in allowlist}
    _, _, _, eval_digests = _source_index(manifest)
    for path in packaged_paths:
        resolved = os.path.realpath(path)
        if resolved not in allowed:
            raise RoleAccessError("cloud_pack_unlisted_path")
        if _is_eval_path(resolved):
            raise RoleAccessError("cloud_pack_eval_forbidden")
        if os.path.isfile(resolved) and _sha256_file(resolved).lower() in eval_digests:
            raise RoleAccessError("cloud_pack_eval_file_hash")


def _source_paths(manifest: Mapping[str, Any]) -> Dict[str, str]:
    source = manifest.get("source_run02") or {}
    if not isinstance(source, dict):
        raise ContractError("source_run02 must be a mapping")
    raw = {key: str(value) for key, value in source.items() if isinstance(value, str)}
    aliases = {
        "sample_rows": ("sample_rows", "sample_rows.json"),
        "duplicate_groups": ("duplicate_groups", "duplicate_groups.json"),
        "frozen_selection": ("frozen_selection", "frozen_selection.json"),
    }
    result: Dict[str, str] = {}
    for key, names in aliases.items():
        for name in names:
            if name in raw:
                result[key] = raw[name]
                break
        else:
            raise ContractError(f"source_run02 missing {key}")
    return result


def _role_key(role: str) -> str:
    return {
        ROLE_FIT: "fit_sequence",
        ROLE_TUNE: "tune_sequence",
        ROLE_EVAL: "eval_pool",
        ROLE_RESERVE: "reserve",
    }[role]


def _role_for_group(manifest: Mapping[str, Any], group_id: str) -> str:
    if group_id in set(manifest.get("exclusions") or []):
        return "excluded"
    hits = [
        role
        for role in (ROLE_FIT, ROLE_TUNE, ROLE_EVAL, ROLE_RESERVE)
        if group_id in set(manifest["role_assignment"][_role_key(role)])
    ]
    if len(hits) != 1:
        raise ContractError(f"group role is not unique: {group_id!r} -> {hits!r}")
    return hits[0]


def _source_index(
    manifest: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, str],
    set[str],
]:
    paths = _source_paths(manifest)
    sample_rows = _read_json(paths["sample_rows"])
    duplicate_groups = _read_json(paths["duplicate_groups"])
    frozen = _read_json(paths["frozen_selection"])
    for key, document in (
        ("sample_rows", sample_rows),
        ("duplicate_groups", duplicate_groups),
        ("frozen_selection", frozen),
    ):
        expected = (manifest.get("source_sha256") or {}).get(key)
        if expected:
            actual = _sha256_file(paths[key])
            if actual != expected:
                raise ContractError(f"content_changed:{key}")
    group_members: Dict[str, List[Dict[str, Any]]] = {}
    sample_to_group: Dict[str, str] = {}
    for group in duplicate_groups.get("groups", []):
        group_id = str(group["group_id"])
        members = list(group.get("members", []))
        group_members[group_id] = members
        _role_for_group(manifest, group_id)
        for member in members:
            sample_id = member.get("sample_id")
            if not isinstance(sample_id, str):
                raise ContractError(f"group {group_id!r} has member without sample_id")
            if sample_id in sample_to_group:
                raise ContractError(f"sample_id appears in multiple groups: {sample_id!r}")
            sample_to_group[sample_id] = group_id
    rows_by_id = {row.get("sample_id"): row for row in sample_rows.get("rows", [])}
    if any(sample_id not in rows_by_id for sample_id in sample_to_group):
        raise ContractError("source group member missing from sample_rows")
    if any(sample_id not in sample_to_group for sample_id in rows_by_id):
        raise ContractError("source row missing a frozen group role")
    eval_digests: set[str] = set()
    for group_id in manifest["role_assignment"]["eval_pool"]:
        for member in group_members.get(group_id, []):
            row = rows_by_id.get(member.get("sample_id")) or {}
            for key in ("png_digest", "svg_digest"):
                digest = row.get(key)
                if isinstance(digest, str) and digest:
                    eval_digests.add(digest.lower())
    return sample_rows, group_members, sample_to_group, eval_digests


def _rows_for_groups(
    manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sample_rows, _, sample_to_group, eval_digests = _source_index(manifest)
    by_id = {row["sample_id"]: row for row in sample_rows.get("rows", [])}
    selected = set(group_ids)
    rows: List[Dict[str, Any]] = []
    for group_id in group_ids:
        actual_role = _role_for_group(manifest, group_id)
        if role is not None and actual_role != role:
            raise RoleAccessError(
                f"role_mismatch: group {group_id!r} is {actual_role!r}, reader is {role!r}"
            )
    # Reload membership without accepting a caller-supplied group_id label.
    _, group_members, source_to_group, _ = _source_index(manifest)
    for group_id in group_ids:
        for member in group_members.get(group_id, []):
            sample_id = member.get("sample_id")
            row = by_id.get(sample_id)
            if row is None:
                raise ContractError(f"selected source row missing: {sample_id!r}")
            mapped = source_to_group.get(sample_id)
            if mapped != group_id:
                raise RoleAccessError(
                    f"source_row_role_mismatch: {sample_id!r} maps to {mapped!r}, expected {group_id!r}"
                )
            if role != ROLE_EVAL and any(
                str(row.get(key) or "").lower() in eval_digests
                for key in ("png_digest", "svg_digest")
            ):
                raise RoleAccessError("train_reader_selected_eval_file")
            row = dict(row)
            row["group_id"] = group_id
            rows.append(row)
    return rows


def _train_dir(manifest: Mapping[str, Any]) -> str:
    root = manifest.get("datasets_root")
    if not isinstance(root, str) or not root:
        raise ContractError("manifest datasets_root missing")
    return os.path.join(root, "train_final")


def resource_snapshot(started: float) -> Dict[str, Any]:
    """Observable time/RSS snapshot; psutil is optional and never installed."""
    elapsed = max(0.0, time.monotonic() - started)
    rss_mb = None
    source = "unknown"
    try:
        import psutil  # type: ignore
        rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        source = "psutil"
    except ImportError:
        pass
    except Exception:
        source = "psutil_error"
    return {"elapsed_s": round(elapsed, 3), "rss_mb": round(rss_mb, 1) if rss_mb is not None else None, "rss_source": source}


def enforce_run_caps(
    counts: Mapping[str, Any],
    *,
    started: float,
    caps: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Raise BudgetExceeded for the first exhausted hard cap."""
    limits = dict(RUN_CAPS)
    if caps:
        limits.update(caps)
    snapshot = resource_snapshot(started)
    reasons: List[str] = []
    for key in ("pairs", "candidates_total", "edges_total"):
        if key == "pairs":
            value = int(counts.get("pairs", 0))
            limit = int(limits["max_pairs"])
        elif key == "candidates_total":
            value = int(counts.get("candidates_total", 0))
            limit = int(limits["max_candidates_total"])
        else:
            value = int(counts.get("edges_total", 0))
            limit = int(limits["max_edges_total"])
        if value > limit:
            reasons.append(f"{key}>{limit}")
    if int(counts.get("candidates_image", 0)) > int(limits["max_candidates_per_image"]):
        reasons.append(f"candidates_image>{limits['max_candidates_per_image']}")
    if int(counts.get("edges_image", 0)) > int(limits["max_edges_per_image"]):
        reasons.append(f"edges_image>{limits['max_edges_per_image']}")
    if snapshot["elapsed_s"] > float(limits["max_elapsed_s"]):
        reasons.append(f"elapsed_s>{limits['max_elapsed_s']}")
    if snapshot["rss_mb"] is not None and snapshot["rss_mb"] > float(limits["max_rss_mb"]):
        reasons.append(f"rss_mb>{limits['max_rss_mb']}")
    if reasons:
        error = BudgetExceeded(";".join(reasons))
        error.details = {"counts": dict(counts), "snapshot": snapshot, "caps": limits}
        raise error
    return snapshot


def validated_train_file_paths(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Realpath-bounded, actual-hash-verified train file pairs.

    This is deliberately called before any parser/frontend read in fit/tune.
    """
    root = os.path.realpath(_train_dir(manifest))
    root_cmp = os.path.normcase(root)
    validated: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        for kind in ("png", "svg"):
            name = str(record.get(f"{kind}_name") or "")
            if not name or os.path.basename(name) != name or name in (".", ".."):
                raise RoleAccessError(f"invalid_source_name:{kind}:{name!r}")
            path = os.path.realpath(os.path.join(root, name))
            try:
                inside = os.path.commonpath([root_cmp, os.path.normcase(path)]) == root_cmp
            except ValueError:
                inside = False
            if not inside:
                raise RoleAccessError(f"source_path_outside_train_root:{path}")
            if not os.path.isfile(path):
                raise ContractError(f"missing_source_file:{path}")
            expected = str(record.get(f"{kind}_digest") or "").lower()
            if not expected:
                raise ContractError(f"missing_source_digest:{kind}:{name}")
            actual = _sha256_file(path)
            if actual != expected:
                raise RoleAccessError(f"source_digest_mismatch:{kind}:{name}")
            record[f"{kind}_path"] = path
            record[f"{kind}_actual_digest"] = actual
        validated.append(record)
    return validated


def prepare(run02: str, out: str, *, train_scale: int = 32) -> Dict[str, Any]:
    """Freeze label-free roles from run02 metadata; never parse SVG labels."""

    run02 = os.path.abspath(run02)
    out = os.path.abspath(out)
    if train_scale not in (32, 64, 128, 256):
        raise ContractError("train_scale must be 32/64/128/256")
    source_order = (
        "sample_rows.json",
        "duplicate_groups.json",
        "frozen_selection.json",
        "prepare_manifest.json",
        "inventory.json",
        "files.csv",
    )
    source_paths = {name: os.path.join(run02, name) for name in source_order}
    for path in source_paths.values():
        if not os.path.isfile(path):
            raise BlockedError(f"missing run02 artifact: {path}")
    source_sha = {name: _sha256_file(path) for name, path in source_paths.items()}
    prepare_manifest = _read_json(source_paths["prepare_manifest.json"])
    expected_artifacts = prepare_manifest.get("artifacts_sha256") or {}
    json_artifacts = {"sample_rows.json", "duplicate_groups.json", "frozen_selection.json", "locked_config.json"}
    for name, expected in expected_artifacts.items():
        path = source_paths.get(name)
        if not path:
            continue
        actual = _fingerprint(_read_json(path)) if name in json_artifacts else _sha256_file(path)
        if actual != expected:
            raise ContractError(f"run02 artifact hash mismatch: {name}")
    sample_rows = _read_json(source_paths["sample_rows.json"])
    duplicate_groups = _read_json(source_paths["duplicate_groups.json"])
    frozen = _read_json(source_paths["frozen_selection.json"])
    excluded = {group["group_id"] for group in frozen.get("groups", [])}
    excluded |= set(AUDIT_GROUP_IDS)
    all_groups = [group["group_id"] for group in duplicate_groups.get("groups", [])]
    roles = assign_permanent_roles(all_groups, seed=SEED, exclude=excluded)
    train_dir = prepare_manifest.get("train_dir")
    datasets_root = os.path.dirname(train_dir) if isinstance(train_dir, str) else None
    manifest = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "train_scale": train_scale,
        "datasets_root": datasets_root,
        "source_run02": source_paths,
        "source_sha256": source_sha,
        "exclusions": sorted(excluded),
        "role_assignment": roles,
        "train_scales": {str(scale): dict(zip(("fit", "tune"), sizes)) for scale, sizes in ((32, (24, 8)), (64, (48, 16)), (128, (96, 32)), (256, (192, 64)))},
        "counts": {
            "groups": len(all_groups),
            "excluded_groups": len(excluded),
            "available_groups": len(all_groups) - len(excluded),
            "fit_sequence": len(roles["fit_sequence"]),
            "tune_sequence": len(roles["tune_sequence"]),
            "eval_pool": len(roles["eval_pool"]),
            "reserve": len(roles["reserve"]),
        },
        "caps": dict(RUN_CAPS),
    }
    validate_manifest(manifest)
    os.makedirs(os.path.join(out, TRAIN_ROOT), exist_ok=True)
    os.makedirs(os.path.join(out, EVAL_ROOT), exist_ok=True)
    _write_json(os.path.join(out, MANIFEST_FILE), manifest)
    return {"status": "complete", "manifest": os.path.join(out, MANIFEST_FILE), "counts": manifest["counts"]}


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        import joblib  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(f"RF dependencies unavailable: {exc}") from exc
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    return RandomForestClassifier, joblib


def semantic_rf():
    RandomForestClassifier, _ = _require_sklearn()
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=2,
    )


def edge_rf():
    RandomForestClassifier, _ = _require_sklearn()
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=2,
    )


def train_semantic_model(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise BlockedError("no semantic training rows")
    RandomForestClassifier, joblib = _require_sklearn()
    feature_names = list(rows[0]["features"].keys())
    X = [[float(row["features"][name]) for name in feature_names] for row in rows]
    y = [int(row["semantic_raw"]) for row in rows]
    model = semantic_rf()
    model.fit(X, y)
    return {
        "model": model,
        "feature_names": feature_names,
        "classes_raw": classes_raw_from_values(y),
        "rows": len(rows),
    }


def train_edge_model(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    labels = {int(row["label"]) for row in rows}
    if len(labels) < 2:
        return None
    feature_names = list(rows[0]["features"].keys())
    X = [[float(row["features"][name]) for name in feature_names] for row in rows]
    y = [int(row["label"]) for row in rows]
    model = edge_rf()
    model.fit(X, y)
    return {"model": model, "feature_names": feature_names, "rows": len(rows)}


def semantic_training_rows(
    candidates: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate, feature in zip(candidates, features):
        supervision = candidate.get("supervision") or {}
        association = candidate.get("association") or {}
        status = association.get("status")
        if status != "unique":
            continue
        if not semantic_mask(
            semantic_present=bool(supervision.get("semantic_raw") is not None),
            semantic_status=str(supervision.get("semantic_status") or ""),
            semantic_raw=supervision.get("semantic_raw"),
            geometry_ok=True,
            association_status=status,
        ):
            continue
        rows.append(
            {
                "role": role,
                "sample_id": candidate.get("sample_id"),
                "candidate_uid": candidate["candidate_uid"],
                "semantic_raw": int(supervision["semantic_raw"]),
                "features": dict(feature),
            }
        )
    return rows


def edge_training_rows(
    candidates: Sequence[Mapping[str, Any]],
    nodes: Sequence[feature_mod.NodeGeometry],
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from .recognition_edge_sampler import sample_edge_pairs

    return sample_edge_pairs(candidates, nodes, role=role)["pairs"]


def _preserve_aspect_ratio_raw(document: Mapping[str, Any]) -> Optional[str]:
    root = document.get("root") or {}
    attributes = root.get("attributes") if isinstance(root, Mapping) else {}
    value = attributes.get("preserveAspectRatio") if isinstance(attributes, Mapping) else None
    return value if isinstance(value, str) else None


def _pixel_gt_for_frontend(
    document: Mapping[str, Any],
    sample_id: str,
    frontend: Mapping[str, Any],
) -> Tuple[List[candidate_mod.GTPrimitive], Dict[str, Any]]:
    """Map all GT to the actual frontend PNG size before association."""

    image = frontend.get("input") or {}
    return candidate_mod.build_pixel_gt_primitives(
        document,
        sample_id,
        png_width=image.get("width"),
        png_height=image.get("height"),
        preserve_aspect_ratio_raw=_preserve_aspect_ratio_raw(document),
        viewport_mode="full_svg_viewport",
        curve_steps=128,
    )


def _training_rows(
    manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    role: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    from . import png_frontend, svg_parse

    started = time.monotonic()
    rows = _rows_for_groups(manifest, group_ids, role=role)
    rows = validated_train_file_paths(manifest, rows)
    semantic_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    accounting = {
        "groups": len(group_ids),
        "pairs": len(rows),
        "primitives_total": 0,
        "geometry_incomplete": 0,
        "mapping_unavailable": 0,
        "frontend_failed": 0,
        "quantized_candidates": 0,
        "candidates_total": 0,
        "candidates_image": 0,
        "edges_total": 0,
        "edges_image": 0,
        "semantic_rows": 0,
        "edge_rows": 0,
        "caps": dict(RUN_CAPS),
    }
    enforce_run_caps(accounting, started=started)
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
            accounting["frontend_failed"] += 1
            continue
        gt, mapping = _pixel_gt_for_frontend(document, row["sample_id"], frontend)
        if mapping.get("status") != "ok":
            accounting["mapping_unavailable"] += 1
            continue
        accounting["primitives_total"] += len(gt)
        if accounting["primitives_total"] > RUN_CAPS["max_primitives"]:
            raise BudgetExceeded(f"primitives_total>{RUN_CAPS['max_primitives']}")
        if any(not primitive.complete for primitive in gt):
            accounting["geometry_incomplete"] += 1
            continue
        candidates = candidate_mod.build_png_candidates(
            frontend, row["sample_id"], group_id=row.get("group_id"), role=role
        )
        accounting["quantized_candidates"] += len(candidates)
        accounting["candidates_total"] += len(candidates)
        accounting["candidates_image"] = len(candidates)
        enforce_run_caps(accounting, started=started)
        candidate_mod.associate_png_candidates(candidates, gt)
        nodes = [feature_mod.node_geometry_from_candidate(item) for item in candidates]
        if not nodes:
            continue
        node_features = feature_mod.extract_node_features(nodes)
        edges, _ = feature_mod.build_local_graph(nodes)
        accounting["edges_image"] = len(edges)
        accounting["edges_total"] += len(edges)
        enforce_run_caps(accounting, started=started)
        semantic_rows.extend(semantic_training_rows(candidates, node_features, role=role))
        from .recognition_edge_sampler import sample_edge_pairs
        pair_result = sample_edge_pairs(candidates, nodes, role=role)
        accounting.setdefault("edge_pair_audits", []).append(pair_result["audit"])
        if pair_result["audit"]["status"] == "blocked_pair_cap":
            raise BudgetExceeded("edge_supervision_positive_pairs_exceed_per_sample_cap")
        edge_rows.extend(pair_result["pairs"])
    accounting["semantic_rows"] = len(semantic_rows)
    accounting["edge_rows"] = len(edge_rows)
    pair_audits = accounting.get("edge_pair_audits", [])
    accounting["edge_pair_audit"] = {
        "schema": "cadvec.recognition.edge_pair_audit/1",
        "policy_fingerprint": (
            pair_audits[0].get("policy_fingerprint") if pair_audits else None
        ),
        "samples": pair_audits,
        "sample_count": len(pair_audits),
        "eligible_positive_before_cap": sum(item.get("eligible_positive_before_cap", 0) for item in pair_audits),
        "eligible_negative_before_cap": sum(item.get("eligible_negative_before_cap", 0) for item in pair_audits),
        "selected_positive": sum(item.get("selected_positive", 0) for item in pair_audits),
        "selected_negative": sum(item.get("selected_negative", 0) for item in pair_audits),
        "cap_hit": any(item.get("cap_hit", False) for item in pair_audits),
        "selected_pair_fingerprint": hashlib.sha256(json.dumps(
            [[row.get("sample_id"), row.get("pair"), row.get("label")] for row in edge_rows],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
    }
    accounting.update(resource_snapshot(started))
    return semantic_rows, edge_rows, accounting


def _fit_or_tune(manifest_path: str, out: str, role: str, train_scale: int) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    if train_scale not in (32, 64, 128, 256):
        raise ContractError("train_scale must be 32/64/128/256")
    reader = RoleReader(manifest, role, root=os.path.join(out, TRAIN_ROOT))
    groups = reader.group_ids(train_scale)
    semantic_rows, edge_rows, accounting = _training_rows(manifest, groups, role=role)
    if not semantic_rows:
        result = {
            "schema": MODEL_SCHEMA,
            "version": 1,
            "status": "blocked",
            "reason": "no_unique_semantic_training_rows",
            "role": role,
            "train_scale": train_scale,
            "accounting": accounting,
        }
        _write_json(os.path.join(out, TRAIN_ROOT, f"{role}_blocked.json"), result)
        return result
    semantic = train_semantic_model(semantic_rows)
    edge = train_edge_model(edge_rows)
    _, joblib = _require_sklearn()
    model_dir = os.path.join(out, TRAIN_ROOT)
    os.makedirs(model_dir, exist_ok=True)
    stem = f"{role}_s{train_scale}"
    semantic_path = os.path.join(model_dir, f"{stem}_semantic.joblib")
    joblib.dump(semantic, semantic_path)
    edge_path = None
    if edge is not None:
        edge_path = os.path.join(model_dir, f"{stem}_edge.joblib")
        joblib.dump(edge, edge_path)
    manifest_out = {
        "schema": MODEL_SCHEMA,
        "version": 1,
        "status": "complete" if edge is not None else "partial",
        "model_id": "rf-semantic-v1" if edge is None else "rf-semantic-v1+rf-edge-v1",
        "stage": role,
        "role": role,
        "train_scale": train_scale,
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "classes_raw": semantic["classes_raw"],
        "feature_names": semantic["feature_names"],
        "edge_feature_names": edge["feature_names"] if edge else None,
        "threshold": 0.5,
        "seed": SEED,
        "rows": semantic["rows"],
        "edge_rows": edge["rows"] if edge else 0,
        "accounting": accounting,
        "edge_pair_audit": accounting.get("edge_pair_audit"),
        "semantic_model": semantic_path,
        "semantic_model_sha256": _sha256_file(semantic_path),
        "edge_model": edge_path,
        "edge_model_sha256": _sha256_file(edge_path) if edge_path else None,
        "note": "RF baseline only; no eval_pool read.",
    }
    _write_json(os.path.join(model_dir, f"{stem}_model_manifest.json"), manifest_out)
    return manifest_out


def fit(manifest: str, out: str, train_scale: int = 32) -> Dict[str, Any]:
    return _fit_or_tune(manifest, out, ROLE_FIT, train_scale)


def tune(manifest: str, out: str, train_scale: int = 32) -> Dict[str, Any]:
    return _fit_or_tune(manifest, out, ROLE_TUNE, train_scale)


def refit_group_ids(manifest: Mapping[str, Any], train_scale: int) -> Dict[str, Any]:
    """Return the immutable fit+tune refit prefix, rejecting any other source."""
    validate_manifest(manifest)
    prefixes = train_prefixes(manifest["role_assignment"], train_scale)
    fit_groups = list(prefixes["fit"])
    tune_groups = list(prefixes["tune"])
    combined = fit_groups + tune_groups
    eval_groups = set(manifest["role_assignment"]["eval_pool"])
    if set(combined) & eval_groups:
        raise BlockedError("refit_group_overlaps_eval_pool")
    if len(combined) != len(set(combined)):
        raise BlockedError("refit_group_duplicate")
    return {
        "fit": fit_groups,
        "tune": tune_groups,
        "combined": combined,
        "fit_group_fingerprint": _fingerprint(fit_groups),
        "tune_group_fingerprint": _fingerprint(tune_groups),
        "combined_group_fingerprint": _fingerprint(combined),
    }


def refit(manifest: str, out: str, train_scale: int = 32) -> Dict[str, Any]:
    manifest_data = _read_json(manifest)
    groups = refit_group_ids(manifest_data, train_scale)
    combined = groups["combined"]
    semantic_rows, edge_rows, accounting = _training_rows(manifest_data, combined, role=None)
    if not semantic_rows:
        return {"schema": MODEL_SCHEMA, "version": 1, "status": "blocked", "reason": "no_unique_semantic_training_rows", "accounting": accounting}
    semantic = train_semantic_model(semantic_rows)
    edge = train_edge_model(edge_rows)
    _, joblib = _require_sklearn()
    model_dir = os.path.join(out, TRAIN_ROOT)
    os.makedirs(model_dir, exist_ok=True)
    stem = f"refit_s{train_scale}"
    semantic_path = os.path.join(model_dir, f"{stem}_semantic.joblib")
    joblib.dump(semantic, semantic_path)
    edge_path = None
    if edge is not None:
        edge_path = os.path.join(model_dir, f"{stem}_edge.joblib")
        joblib.dump(edge, edge_path)
    payload = {
        "schema": MODEL_SCHEMA,
        "version": 1,
        "status": "complete" if edge is not None else "partial",
        "model_id": "rf-refit",
        "stage": "refit",
        "role": "refit",
        "train_scale": train_scale,
        "fit_group_fingerprint": groups["fit_group_fingerprint"],
        "tune_group_fingerprint": groups["tune_group_fingerprint"],
        "combined_group_fingerprint": groups["combined_group_fingerprint"],
        "group_counts": {"fit": len(groups["fit"]), "tune": len(groups["tune"]), "combined": len(combined)},
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_manifest_sha256": _sha256_file(manifest),
        "classes_raw": semantic["classes_raw"],
        "feature_names": semantic["feature_names"],
        "edge_feature_names": edge["feature_names"] if edge else None,
        "threshold": 0.5,
        "seed": SEED,
        "rows": semantic["rows"],
        "edge_rows": edge["rows"] if edge else 0,
        "semantic_model": semantic_path,
        "semantic_model_sha256": _sha256_file(semantic_path),
        "edge_model": edge_path,
        "edge_model_sha256": _sha256_file(edge_path) if edge_path else None,
        "accounting": accounting,
        "edge_pair_audit": accounting.get("edge_pair_audit"),
        "note": "RF baseline refit on fit+tune only; no eval_pool read.",
    }
    _write_json(os.path.join(model_dir, f"{stem}_model_manifest.json"), payload)
    return payload


def freeze(run_dir: str, train_scale: int = 32) -> Dict[str, Any]:
    if train_scale not in (32, 64, 128, 256):
        raise ContractError("train_scale must be 32/64/128/256")
    train_dir = os.path.join(run_dir, TRAIN_ROOT)
    paths = {
        role: os.path.join(train_dir, f"{role}_s{train_scale}_model_manifest.json")
        for role in ("fit", "tune", "refit")
    }
    missing = [role for role, path in paths.items() if not os.path.isfile(path)]
    if missing:
        raise BlockedError(
            "freeze_requires_fit_tune_refit_manifests:" + ",".join(missing)
        )
    payloads = {role: _read_json(path) for role, path in paths.items()}
    source_hashes = {role: payload.get("source_manifest_sha256") for role, payload in payloads.items()}
    if len(set(source_hashes.values())) != 1 or not next(iter(source_hashes.values())):
        raise BlockedError("freeze_source_manifest_mismatch")
    train_root = os.path.realpath(train_dir)
    train_root_cmp = os.path.normcase(train_root)
    for role, payload in payloads.items():
        if payload.get("registry_fingerprint") != REGISTRY_FINGERPRINT:
            raise BlockedError(f"freeze_registry_mismatch:{role}")
        if payload.get("stage") != role or payload.get("role") != role:
            raise BlockedError(f"freeze_stage_role_mismatch:{role}")
        if payload.get("train_scale") != train_scale:
            raise BlockedError(f"freeze_scale_mismatch:{role}")
        if role == "refit" and (payload.get("status") != "complete" or not payload.get("edge_model")):
            raise BlockedError("freeze_requires_complete_refit_with_edge_model")
        for key in ("semantic_model", "edge_model"):
            declared = payload.get(key)
            if not declared:
                if key == "semantic_model" or role == "refit":
                    raise BlockedError(f"freeze_missing_{key}:{role}")
                continue
            resolved = os.path.realpath(str(declared))
            try:
                inside = os.path.commonpath([train_root_cmp, os.path.normcase(resolved)]) == train_root_cmp
            except ValueError:
                inside = False
            if not inside:
                raise BlockedError(f"freeze_model_outside_train_root:{role}:{key}")
            if not os.path.isfile(resolved):
                raise BlockedError(f"freeze_model_missing:{role}:{key}")
            expected = payload.get(f"{key}_sha256")
            if not expected or _sha256_file(resolved) != expected:
                raise BlockedError(f"freeze_model_hash_mismatch:{role}:{key}")
    model_manifest = paths["refit"]
    payload = payloads["refit"]
    frozen = {
        "schema": "cadvec.recognition.frozen/1",
        "version": 1,
        "status": "frozen",
        "model_manifest": model_manifest,
        "model_sha256": _sha256_file(model_manifest),
        "classes_raw": payload.get("classes_raw"),
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "train_scale": train_scale,
        "seed": SEED,
        "eval_root": os.path.join(run_dir, EVAL_ROOT),
    }
    _write_json(os.path.join(run_dir, FROZEN_FILE), frozen)
    return frozen


def _load_model(model_manifest_path: str) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    _, joblib = _require_sklearn()
    manifest = _read_json(model_manifest_path)
    semantic = joblib.load(manifest["semantic_model"])
    edge = joblib.load(manifest["edge_model"]) if manifest.get("edge_model") else None
    return semantic, edge, manifest


def _predict_candidates(semantic: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    nodes = [feature_mod.node_geometry_from_candidate(item) for item in candidates]
    feature_rows = feature_mod.extract_node_features(nodes)
    names = list(semantic["feature_names"])
    model = semantic["model"]
    X = [[float(row[name]) for name in names] for row in feature_rows]
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    classes = list(model.classes_)
    out: List[Dict[str, Any]] = []
    for candidate, pred, prob in zip(candidates, preds, probs if probs is not None else [None] * len(candidates)):
        probability = None
        if prob is not None:
            index = classes.index(pred)
            probability = float(prob[index])
        out.append(
            {
                "schema": PREDICTION_SCHEMA,
                "version": 1,
                "candidate_uid": candidate["candidate_uid"],
                "sample_id": candidate["sample_id"],
                "source_kind": candidate["source_kind"],
                "geometry_px": (candidate.get("geometry") or {}).get("points_px")
                or (candidate.get("geometry") or {}).get("points"),
                "semantic_raw_pred": int(pred),
                "semantic_proba": probability,
                "instance_group": None,
                "target": None,
                "association": None,
            }
        )
    return out


def predict_png(model_manifest: str, png: str) -> List[Dict[str, Any]]:
    from . import png_frontend

    if os.path.isdir(png):
        raise BlockedError("predict-png accepts exactly one PNG file")
    result = png_frontend.predict_png(
        png,
        png_frontend.FrontendConfig(
            max_lines=4000, max_pixels=40_000_000, max_file_bytes=64 * 1024 * 1024
        ),
    )
    if result.get("status") not in ("ok", "empty_input", "no_lines_detected"):
        raise BlockedError(f"frontend status: {result.get('status')}")
    candidates = candidate_mod.build_png_candidates(result, "png:" + os.path.basename(png))
    semantic, _, _ = _load_model(model_manifest)
    return _predict_candidates(semantic, candidates)


def smoke(model_manifest: str, png: str, out: str) -> Dict[str, Any]:
    started = time.perf_counter()
    predictions = predict_png(model_manifest, png)
    payload = {
        "schema": "cadvec.recognition.smoke/1",
        "version": 1,
        "status": "ok",
        "png_only": True,
        "model_manifest": model_manifest,
        "png": os.path.basename(png),
        "prediction_count": len(predictions),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "target_all_null": all(item.get("target") is None for item in predictions),
    }
    _write_json(out, payload)
    return payload


def _compute_evaluation_metrics(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    classes_raw: Sequence[int],
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Single bridge from evaluate to the strict metrics module."""

    return metrics_mod.compute_metrics(
        candidates,
        predictions,
        classes_raw=classes_raw,
        coverage=coverage,
    )


def evaluate(run_dir: str, out: str, predictions: Optional[str] = None) -> Dict[str, Any]:
    frozen_path = os.path.join(run_dir, FROZEN_FILE)
    if not os.path.isfile(frozen_path):
        raise EvalSealedError("freeze must be completed before evaluate")
    frozen = _read_json(frozen_path)
    manifest = _read_json(os.path.join(run_dir, MANIFEST_FILE))
    reader = RoleReader(
        manifest, ROLE_EVAL, root=os.path.join(run_dir, EVAL_ROOT), allow_eval=True
    )
    groups = reader.group_ids()
    from . import png_frontend, svg_parse

    model_manifest = frozen["model_manifest"]
    semantic, _, model_info = _load_model(model_manifest)
    classes = sorted({int(value) for value in model_info.get("classes_raw", [])})
    external_predictions: Optional[List[Mapping[str, Any]]] = None
    if predictions:
        external_predictions = []
        with open(predictions, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise metrics_mod.MetricsValidationError(
                            "prediction_not_a_mapping"
                        )
                    external_predictions.append(row)
    all_candidates: List[Dict[str, Any]] = []
    generated_predictions: List[Dict[str, Any]] = []
    coverage = {
        "selected": len(groups),
        "geometry_incomplete": 0,
        "mapping_unavailable": 0,
        "frontend_failed": 0,
    }
    for row in _rows_for_groups(manifest, groups, role=ROLE_EVAL):
        svg_path = os.path.join(_train_dir(manifest), str(row["svg_name"]))
        png_path = os.path.join(_train_dir(manifest), str(row["png_name"]))
        document = svg_parse.parse_svg(svg_path, split="train_final", record_commands=True, curve_steps=128)
        frontend = png_frontend.predict_png(
            png_path,
            png_frontend.FrontendConfig(max_lines=4000, max_pixels=40_000_000, max_file_bytes=64 * 1024 * 1024),
        )
        if frontend.get("status") not in ("ok", "empty_input", "no_lines_detected"):
            coverage["frontend_failed"] += 1
            continue
        gt, mapping = _pixel_gt_for_frontend(document, row["sample_id"], frontend)
        if mapping.get("status") != "ok":
            coverage["mapping_unavailable"] += 1
            continue
        if any(not primitive.complete for primitive in gt):
            coverage["geometry_incomplete"] += 1
            continue
        candidates = candidate_mod.build_png_candidates(frontend, row["sample_id"])
        candidate_mod.associate_png_candidates(candidates, gt)
        all_candidates.extend(candidates)
        if external_predictions is None:
            generated_predictions.extend(_predict_candidates(semantic, candidates))

    prediction_rows: Sequence[Mapping[str, Any]] = (
        external_predictions if external_predictions is not None else generated_predictions
    )
    payload = _compute_evaluation_metrics(
        all_candidates,
        prediction_rows,
        classes_raw=classes,
        coverage=coverage,
    )
    payload["eval_pool_sha256"] = _sha256_file(os.path.join(run_dir, MANIFEST_FILE))
    payload["coverage"]["frontend_failed"] = coverage["frontend_failed"]
    payload["counts"]["frontend_failed"] = coverage["frontend_failed"]
    payload["counts"]["semantic_valid"] = payload["semantic"]["counts"]["evaluable"]
    payload["counts"]["geometry_incomplete"] = coverage["geometry_incomplete"]
    payload["counts"]["mapping_unavailable"] = coverage["mapping_unavailable"]
    payload["counts"]["association_excluded"] = payload["coverage"]["association_excluded"]
    payload["counts"]["association_ambiguous"] = payload["coverage"]["association_ambiguous"]
    payload["counts"]["association_unmatched"] = payload["coverage"]["association_unmatched"]
    payload["counts"]["excluded"] = coverage["frontend_failed"]
    payload["counts"]["unknown"] = payload["semantic"]["unknown"]
    payload["denominators"]["semantic_score"] = {
        "valid": payload["semantic"]["counts"]["evaluable"]
    }
    payload["denominators"]["instance_score"] = {
        "valid": payload["instance"]["counts"]["items"]
    }
    payload["denominators"]["excluded"] = {
        "image_count": coverage["frontend_failed"],
        "candidate_count": payload["coverage"]["association_excluded"],
        "rate": coverage["frontend_failed"] / len(groups) if groups else None,
    }
    payload["grouping"] = payload["instance"]
    _write_json(out, payload)
    return payload


def compare_tune(
    manifest_path: str,
    out: str,
    *,
    rf_model_manifest: str,
    gnn_checkpoint: str,
    train_scale: int = 32,
    train_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Delegate to the shared fit-only tune evaluator."""

    from . import recognition_tune_eval

    return recognition_tune_eval.compare_tune(
        manifest_path,
        out,
        rf_model_manifest=rf_model_manifest,
        gnn_checkpoint=gnn_checkpoint,
        train_scale=train_scale,
        train_root=train_root,
    )


def control_svg(manifest_path: str, out: str, train_scale: int = 32) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    reader = RoleReader(manifest, ROLE_FIT, root=os.path.join(out, TRAIN_ROOT))
    groups = reader.group_ids(train_scale)
    from . import svg_parse

    records: List[Dict[str, Any]] = []
    for row in _rows_for_groups(manifest, groups, role=ROLE_FIT):
        svg_path = os.path.join(_train_dir(manifest), str(row["svg_name"]))
        document = svg_parse.parse_svg(svg_path, split="train_final", record_commands=True, curve_steps=128)
        built = candidate_mod.build_svg_candidates(document, row["sample_id"], group_id=row.get("group_id"), role="control")
        records.extend(built["candidates"])
    _write_jsonl(out, records)
    return {"status": "complete", "count": len(records), "path": out}


def _version_block() -> Dict[str, Any]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        metadata = None
    names = ("numpy", "scipy", "scikit-learn", "networkx", "joblib", "opencv-python-headless")
    versions = {}
    if metadata is not None:
        for name in names:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    return {"python": __import__("sys").version.split()[0], "packages": versions}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cadvec.recognition_baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run02", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("fit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("tune")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("refit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("freeze")
    p.add_argument("--run", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", default=None)
    p = sub.add_parser("predict-png")
    p.add_argument("--model", required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--model", required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("control-svg")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p = sub.add_parser("compare-tune")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rf-model-manifest", required=True)
    p.add_argument("--gnn-checkpoint", required=True)
    p.add_argument("--train-scale", type=int, default=32)
    p.add_argument("--train-root", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.run02, args.out, train_scale=args.train_scale)
        elif args.command == "fit":
            result = fit(args.manifest, args.out, args.train_scale)
        elif args.command == "tune":
            result = tune(args.manifest, args.out, args.train_scale)
        elif args.command == "refit":
            result = refit(args.manifest, args.out, args.train_scale)
        elif args.command == "freeze":
            result = freeze(args.run, args.train_scale)
        elif args.command == "evaluate":
            result = evaluate(args.run, args.out, predictions=args.predictions)
        elif args.command == "predict-png":
            result = {"status": "ok", "predictions": predict_png(args.model, args.png)}
            _write_jsonl(args.out, result["predictions"])
        elif args.command == "smoke":
            result = smoke(args.model, args.png, args.out)
        elif args.command == "control-svg":
            result = control_svg(args.manifest, args.out, args.train_scale)
        elif args.command == "compare-tune":
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
