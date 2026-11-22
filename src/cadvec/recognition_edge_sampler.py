"""Shared deterministic RF/GNN supervision pairs; standard library only.

Pairs are separate from the geometry graph. Labels select targets only; feature
vectors are computed solely from NodeGeometry with the frozen edge feature API.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Sequence

from . import recognition_features as features
from .recognition_contract import (
    EDGE_FEATURE_NAMES, EDGE_PAIR_POLICY, EDGE_PAIR_SCHEMA, ContractError,
    instance_mask,
)

PAIR_SCHEMA = EDGE_PAIR_SCHEMA
PAIR_CAP_PER_SAMPLE = int(EDGE_PAIR_POLICY["max_pairs_per_sample"])


def _fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PAIR_POLICY_FINGERPRINT = _fingerprint(EDGE_PAIR_POLICY)


def _choose2(n: int) -> int:
    return n * (n - 1) // 2


def _semantic_valid(candidate: Mapping[str, Any]) -> bool:
    # Pair eligibility needs the raw Thing class, not semantic-loss validity.
    raw = (candidate.get("supervision") or {}).get("semantic_raw")
    return isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= 30


def _instance_valid(candidate: Mapping[str, Any]) -> bool:
    supervision = candidate.get("supervision") or {}
    raw = supervision.get("instance_raw")
    return instance_mask(
        instance_present=bool(supervision.get("instance_present", raw is not None)),
        instance_status=str(supervision.get("instance_status") or ""),
        instance_raw=raw, semantic_raw=supervision.get("semantic_raw"),
        geometry_ok=True, association_status="unique",
    )


def sample_edge_pairs(
    candidates: Sequence[Mapping[str, Any]],
    nodes: Sequence[features.NodeGeometry],
    *,
    role: str | None = None,
    max_pairs_per_sample: int = PAIR_CAP_PER_SAMPLE,
) -> Dict[str, Any]:
    """Retain every positive or block; SHA256-rank balanced bounded negatives.

    Audit exclusions are exclusive, in sample/association/semantic/instance
    order. Counting is combinatorial and only selected negatives are retained in
    memory. A smaller cap may be supplied for fixtures; the contract cap cannot
    be increased. Any overflowing sample blocks the entire returned selection.
    """
    if len(candidates) != len(nodes):
        raise ContractError("candidate/node arrays must align")
    if not isinstance(max_pairs_per_sample, int) or isinstance(max_pairs_per_sample, bool) or not 0 <= max_pairs_per_sample <= PAIR_CAP_PER_SAMPLE:
        raise ContractError("supervision pair cap must be within fixed 0..256")
    by_sample = defaultdict(list)
    uid_to_index = {}
    for index, (candidate, node) in enumerate(zip(candidates, nodes)):
        uid, sid = candidate.get("candidate_uid"), candidate.get("sample_id")
        if not isinstance(uid, str) or not uid or uid in uid_to_index:
            raise ContractError("supervision pair candidate_uid must be unique and nonempty")
        if not isinstance(sid, str) or not sid or node.sample_id != sid or node.candidate_uid != uid:
            raise ContractError("supervision pair candidate/node identity mismatch")
        uid_to_index[uid] = index
        by_sample[sid].append(candidate)

    audits = []
    selected = []
    total_within = sum(_choose2(len(items)) for items in by_sample.values())
    for sid, items in sorted(by_sample.items()):
        unique = [item for item in items if (item.get("association") or {}).get("status") == "unique"]
        same_class = defaultdict(list)
        for item in unique:
            if _semantic_valid(item):
                same_class[item["supervision"]["semantic_raw"]].append(item)
        semantic_pair_count = sum(_choose2(len(group)) for group in same_class.values())
        eligible_classes = {
            raw: sorted((item for item in group if _instance_valid(item)), key=lambda item: item["candidate_uid"])
            for raw, group in same_class.items()
        }
        eligible_pair_count = sum(_choose2(len(group)) for group in eligible_classes.values())
        positive_count = sum(
            _choose2(count)
            for group in eligible_classes.values()
            for count in Counter(item["supervision"]["instance_raw"] for item in group).values()
        )
        negative_count = eligible_pair_count - positive_count
        blocked = positive_count > max_pairs_per_sample
        negative_limit = 0 if blocked else min(positive_count, max_pairs_per_sample - positive_count, negative_count)

        def negative_pairs():
            for _, group in sorted(eligible_classes.items()):
                for a, b in itertools.combinations(group, 2):
                    if a["supervision"]["instance_raw"] != b["supervision"]["instance_raw"]:
                        pair = (sid, a["candidate_uid"], b["candidate_uid"], 0)
                        # Rank depends only on sample/candidate identities, never labels/features.
                        yield (_fingerprint(pair[:3]), pair)

        if not blocked:
            for group in eligible_classes.values():
                by_instance = defaultdict(list)
                for item in group:
                    by_instance[item["supervision"]["instance_raw"]].append(item)
                for members in by_instance.values():
                    selected.extend((sid, a["candidate_uid"], b["candidate_uid"], 1) for a, b in itertools.combinations(members, 2))
            if negative_limit:
                selected.extend(pair for _, pair in heapq.nsmallest(negative_limit, negative_pairs()))
        audits.append({
            "sample_id": sid,
            "candidate_pairs": _choose2(len(items)),
            "eligible_positive_before_cap": positive_count,
            "eligible_negative_before_cap": negative_count,
            "excluded_nonunique_association": _choose2(len(items)) - _choose2(len(unique)),
            "excluded_semantic": _choose2(len(unique)) - semantic_pair_count,
            "excluded_instance": semantic_pair_count - eligible_pair_count,
            "selected_positive": 0 if blocked else positive_count,
            "selected_negative": negative_limit,
            "negatives_removed_balance": max(0, negative_count - positive_count),
            "negatives_removed_cap": 0 if blocked else max(0, min(negative_count, positive_count) - negative_limit),
            "cap_per_sample": max_pairs_per_sample,
            "cap_hit": blocked or (not blocked and min(negative_count, positive_count) > negative_limit),
            "status": "blocked_pair_cap" if blocked else "available",
        })
    blocked = any(audit["status"] == "blocked_pair_cap" for audit in audits)
    if blocked:
        selected = []
    selected.sort()
    edges = [features.Edge(left, right) for _, left, right, _ in selected]
    vectors = features.extract_edge_features(nodes, edges) if edges else []
    records = [{
        "schema": PAIR_SCHEMA, "version": 1, "sample_id": sid,
        "pair": [left, right], "edge": [left, right],
        "pair_index": [uid_to_index[left], uid_to_index[right]],
        "features": {name: float(vector[name]) for name in EDGE_FEATURE_NAMES},
        "label": label, "role": role,
    } for (sid, left, right, label), vector in zip(selected, vectors)]
    count_keys = (
        "eligible_positive_before_cap", "eligible_negative_before_cap",
        "excluded_nonunique_association", "excluded_semantic", "excluded_instance",
        "negatives_removed_balance", "negatives_removed_cap",
    )
    audit = {
        "schema": PAIR_SCHEMA, "policy": dict(EDGE_PAIR_POLICY),
        "policy_fingerprint": PAIR_POLICY_FINGERPRINT,
        "samples": audits, "candidate_pairs": _choose2(len(candidates)),
        "excluded_cross_sample": _choose2(len(candidates)) - total_within,
        **{key: sum(item[key] for item in audits) for key in count_keys},
        "selected_positive": sum(row["label"] == 1 for row in records),
        "selected_negative": sum(row["label"] == 0 for row in records),
        "cap_per_sample": max_pairs_per_sample,
        "cap_hit": any(item["cap_hit"] for item in audits),
        "status": "blocked_pair_cap" if blocked else "available",
        "selected_pair_fingerprint": _fingerprint(selected),
    }
    return {"schema": PAIR_SCHEMA, "version": 1, "pairs": records, "audit": audit}
