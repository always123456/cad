"""Shared recognition contracts for the Gate3B public base.

This module is deliberately standard-library only.  It freezes the raw label
space, role names, masks, feature order and small validation helpers used by
the candidate, feature and baseline layers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

SCHEMA = "cadvec.recognition.public_manifest/1"
SCHEMA_VERSION = 1
CANDIDATE_SCHEMA = "cadvec.recognition.candidate/1"
MODEL_SCHEMA = "cadvec.recognition.model_manifest/1"
METRICS_SCHEMA = "cadvec.recognition.metrics/1"
PREDICTION_SCHEMA = "cadvec.recognition.predictions/1"

SEED = 20261108
RAW_SEMANTIC_MIN = 1
RAW_SEMANTIC_MAX = 35
THING_MIN = 1
THING_MAX = 30
STUFF_MIN = 31
STUFF_MAX = 35
ASSOCIATION_TOLERANCE_PX = 4.0
EDGE_THRESHOLD = 0.5
EDGE_PAIR_SCHEMA = "cadvec.recognition.edge_pair/1"
EDGE_PAIR_POLICY = {
    "max_pairs_per_sample": 1024,
    "positives": "retain_all_or_block",
    "negatives": "sha256_rank_up_to_positives_and_remaining_capacity",
    "order": "sample_id_then_canonical_candidate_uid_pair",
}

ROLE_FIT = "fit"
ROLE_TUNE = "tune"
ROLE_EVAL = "eval"
ROLE_RESERVE = "reserve"
ROLES = (ROLE_FIT, ROLE_TUNE, ROLE_EVAL, ROLE_RESERVE)

FIT_SEQUENCE_SIZE = 192
TUNE_SEQUENCE_SIZE = 64
EVAL_POOL_SIZE = 128
TRAIN_SCALES = {
    32: (24, 8),
    64: (48, 16),
    128: (96, 32),
    256: (192, 64),
}

AUDIT_GROUP_IDS = frozenset(
    {
        "g-d65d190f08660615",
        "g-aec0b246a5673957",
        "g-47b9e03dc9e4c4d8",
        "g-adb80f1af62e3dd2",
        "g-14000d273fdad139",
        "g-59c5cb6ce9e67532",
        "g-15f55f09e9b97b7b",
        "g-aec2467290adc7ad",
        "g-1c2d1f00d7eb110c",
        "g-46552f512fd79df5",
        "g-2aa8c5ea11fc89d1",
        "g-011a1d2e421a15c6",
    }
)

NODE_FEATURE_NAMES = (
    "length_px",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_area_px",
    "aspect_log",
    "orientation_sin",
    "orientation_cos",
    "straightness",
    "midpoint_x_norm",
    "midpoint_y_norm",
    "nearest_distance_px",
    "knn4_distance_mean_px",
    "local_count_24px",
    "orientation_coherence",
    "length_ratio_nearest",
)

EDGE_FEATURE_NAMES = (
    "endpoint_gap_min_px",
    "midpoint_distance_px",
    "angle_diff_abs_deg",
    "length_ratio_log",
    "bbox_gap_px",
    "bbox_overlap_ratio",
    "parallel_offset_px",
    "nearest_endpoint_distance_px",
)


class RecognitionError(Exception):
    """Base class for explicit, reportable failures."""

    code = "recognition_error"


class ContractError(RecognitionError, ValueError):
    code = "contract_error"


class RoleAccessError(RecognitionError):
    code = "role_access_denied"


class EvalSealedError(RecognitionError):
    code = "eval_sealed"


class MissingDependencyError(RecognitionError, ImportError):
    code = "missing_dependency"


class BlockedError(RecognitionError):
    code = "blocked"


class BudgetExceeded(RecognitionError):
    code = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class Supervision:
    """Labels/masks kept physically separate from NodeGeometry."""

    candidate_uid: str
    semantic_raw: Optional[int]
    semantic_status: str
    instance_raw: Optional[int]
    instance_status: str


def is_raw_semantic(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and RAW_SEMANTIC_MIN <= value <= RAW_SEMANTIC_MAX
    )


def semantic_kind(value: Any) -> Optional[str]:
    if not is_raw_semantic(value):
        return None
    return "Thing" if THING_MIN <= value <= THING_MAX else "Stuff"


def semantic_mask(
    *,
    semantic_present: bool,
    semantic_status: str,
    semantic_raw: Optional[int],
    geometry_ok: bool,
    association_status: str,
) -> bool:
    return bool(
        semantic_present
        and semantic_status == "ok"
        and is_raw_semantic(semantic_raw)
        and geometry_ok
        and association_status in ("unique", "not_applicable")
    )


def instance_mask(
    *,
    instance_present: bool,
    instance_status: str,
    instance_raw: Optional[int],
    semantic_raw: Optional[int],
    geometry_ok: bool,
    association_status: str,
) -> bool:
    return bool(
        instance_present
        and instance_status == "ok"
        and isinstance(instance_raw, int)
        and not isinstance(instance_raw, bool)
        and instance_raw >= 0
        and is_raw_semantic(semantic_raw)
        and THING_MIN <= semantic_raw <= THING_MAX
        and geometry_ok
        and association_status in ("unique", "not_applicable")
    )


def classes_raw_from_values(values: Iterable[Any]) -> list[int]:
    return sorted(
        {int(value) for value in values if is_raw_semantic(value)}
    )


def group_rank_key(group_id: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}\x00{group_id}".encode("utf-8")).hexdigest()


def rank_groups(group_ids: Iterable[str], seed: int = SEED) -> list[str]:
    return sorted(set(group_ids), key=lambda value: (group_rank_key(value, seed), value))


def assign_permanent_roles(
    group_ids: Iterable[str],
    *,
    seed: int = SEED,
    exclude: Iterable[str] = (),
) -> Dict[str, list[str]]:
    """Freeze fit/tune/eval/reserve once from label-free group IDs."""

    excluded = set(exclude)
    available = rank_groups(set(group_ids) - excluded, seed)
    return {
        "fit_sequence": available[:FIT_SEQUENCE_SIZE],
        "tune_sequence": available[FIT_SEQUENCE_SIZE : FIT_SEQUENCE_SIZE + TUNE_SEQUENCE_SIZE],
        "eval_pool": available[
            FIT_SEQUENCE_SIZE
            + TUNE_SEQUENCE_SIZE : FIT_SEQUENCE_SIZE
            + TUNE_SEQUENCE_SIZE
            + EVAL_POOL_SIZE
        ],
        "reserve": available[FIT_SEQUENCE_SIZE + TUNE_SEQUENCE_SIZE + EVAL_POOL_SIZE :],
    }


def train_prefixes(roles: Mapping[str, Sequence[str]], scale: int) -> Dict[str, list[str]]:
    if scale not in TRAIN_SCALES:
        raise ContractError(f"train scale must be one of {sorted(TRAIN_SCALES)}")
    if set(roles.get("fit_sequence", ())) & set(roles.get("eval_pool", ())):
        raise ContractError("fit/eval role overlap")
    if set(roles.get("tune_sequence", ())) & set(roles.get("eval_pool", ())):
        raise ContractError("tune/eval role overlap")
    fit_n, tune_n = TRAIN_SCALES[scale]
    fit = list(roles.get("fit_sequence", ()))
    tune = list(roles.get("tune_sequence", ()))
    if len(fit) < fit_n or len(tune) < tune_n:
        raise ContractError(
            f"role pools too small for scale {scale}: "
            f"fit={len(fit)} tune={len(tune)}"
        )
    return {"fit": fit[:fit_n], "tune": tune[:tune_n]}


def validate_roles(roles: Mapping[str, Sequence[str]]) -> None:
    required = ("fit_sequence", "tune_sequence", "eval_pool", "reserve")
    if any(name not in roles for name in required):
        raise ContractError("role_assignment must contain fit_sequence/tune_sequence/eval_pool/reserve")
    pools = {name: list(roles[name]) for name in required}
    for left in required:
        for right in required:
            if left >= right:
                continue
            if set(pools[left]) & set(pools[right]):
                raise ContractError(f"roles overlap: {left}/{right}")
    if len(pools["eval_pool"]) > EVAL_POOL_SIZE:
        raise ContractError("eval_pool exceeds fixed size")
    for scale, (fit_n, tune_n) in TRAIN_SCALES.items():
        if len(pools["fit_sequence"]) < fit_n or len(pools["tune_sequence"]) < tune_n:
            continue


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ContractError("public manifest schema mismatch")
    if manifest.get("version") != SCHEMA_VERSION:
        raise ContractError("public manifest version mismatch")
    roles = manifest.get("role_assignment")
    if not isinstance(roles, Mapping):
        raise ContractError("public manifest role_assignment missing")
    validate_roles(roles)
    if manifest.get("seed") != SEED:
        raise ContractError("public manifest seed mismatch")
