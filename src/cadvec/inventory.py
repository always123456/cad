"""File-level inventory of the dataset splits.

Only names, sizes, digests and PNG/SVG pairing are computed.  File contents
are read as bytes for hashing, never interpreted as annotations — which is why
the same code is allowed to run on ``test_final``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .policy import (
    ROLE_IGNORED,
    ROLE_PNG,
    ROLE_SVG,
    SPLIT_POLICIES,
    classify_name,
    resolve_policy,
    stem_of,
)

HASH_ALGORITHMS = ("sha256", "size", "none")
DEFAULT_SPLITS = ("train_final", "test_final", "dev_final")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _digest(path: str, algorithm: str) -> Optional[str]:
    if algorithm == "sha256":
        return sha256_file(path)
    if algorithm == "size":
        return f"size:{os.path.getsize(path)}"
    return None


def scan_split(
    directory: str,
    split: str,
    hash_algorithm: str = "sha256",
) -> Dict[str, Any]:
    """Inventory one split directory without touching file contents."""

    if hash_algorithm not in HASH_ALGORITHMS:
        raise ValueError(f"hash_algorithm must be one of {HASH_ALGORITHMS}")

    policy = resolve_policy(split)
    if policy is None:
        raise ValueError(f"unknown split {split!r}")

    entry: Dict[str, Any] = {
        "path": directory,
        "present": os.path.isdir(directory),
        "policy": policy.to_dict(),
        "annotation_access": policy.annotation_access,
    }
    if not entry["present"]:
        entry["files"] = {
            "total": 0,
            "excluded_resource_forks": 0,
            "samples": 0,
            "png": 0,
            "svg": 0,
            "other": 0,
            "bytes": 0,
            "sample_bytes": 0,
        }
        entry["pairing"] = {
            "png_stems": 0,
            "svg_stems": 0,
            "paired": 0,
            "png_only": 0,
            "svg_only": 0,
            "duplicate_stems": 0,
        }
        entry["digests"] = {"algorithm": hash_algorithm, "hashed": 0}
        entry["file_records"] = []
        entry["notes"] = ["split directory missing"]
        return entry

    names = sorted(os.listdir(directory))
    counts = Counter()
    records: List[Dict[str, Any]] = []
    png_stems: List[str] = []
    svg_stems: List[str] = []
    other_names: List[str] = []
    excluded_names: List[str] = []
    total_bytes = 0
    sample_bytes = 0

    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        role = classify_name(name)
        size = os.path.getsize(path)
        counts["total"] += 1
        total_bytes += size
        if role == ROLE_IGNORED:
            counts["excluded_resource_forks"] += 1
            excluded_names.append(name)
            continue
        if role == ROLE_PNG:
            counts["png"] += 1
            counts["samples"] += 1
            png_stems.append(stem_of(name))
            sample_bytes += size
        elif role == ROLE_SVG:
            counts["svg"] += 1
            counts["samples"] += 1
            svg_stems.append(stem_of(name))
            sample_bytes += size
        else:
            counts["other"] += 1
            other_names.append(name)

        records.append(
            {
                "split": split,
                "name": name,
                "role": role,
                "extension": os.path.splitext(name)[1].lower(),
                "stem": stem_of(name),
                "size_bytes": size,
                "digest": _digest(path, hash_algorithm),
            }
        )

    png_set = set(png_stems)
    svg_set = set(svg_stems)
    paired = png_set & svg_set
    png_only = sorted(png_set - svg_set)
    svg_only = sorted(svg_set - png_set)

    png_dupes = sorted(stem for stem, count in Counter(png_stems).items() if count > 1)
    svg_dupes = sorted(stem for stem, count in Counter(svg_stems).items() if count > 1)

    # Windows filesystems are case-insensitive; record accidental case merges.
    case_map: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        role = classify_name(name)
        if role in (ROLE_PNG, ROLE_SVG):
            case_map[name.lower()].append(name)
    case_collisions = sorted(
        sorted(group) for group in case_map.values() if len(group) > 1
    )

    digests: Dict[str, Any] = {"algorithm": hash_algorithm}
    if hash_algorithm != "none":
        by_digest: Dict[str, List[str]] = defaultdict(list)
        for record in records:
            if record["digest"]:
                by_digest[record["digest"]].append(record["name"])
        duplicate_groups = {
            digest: sorted(files) for digest, files in by_digest.items() if len(files) > 1
        }
        role_by_name = {record["name"]: record["role"] for record in records}
        groups_by_role = {"png": 0, "svg": 0, "mixed": 0}
        unique_by_role = {"png": 0, "svg": 0, "other": 0}
        for digest, names_in_group in by_digest.items():
            roles = {role_by_name.get(name, "other") for name in names_in_group}
            if len(roles) == 1:
                role = next(iter(roles))
                unique_by_role[role] = unique_by_role.get(role, 0) + 1
                if len(names_in_group) > 1 and role in groups_by_role:
                    groups_by_role[role] += 1
            else:
                unique_by_role["other"] = unique_by_role.get("other", 0) + 1
                if len(names_in_group) > 1:
                    groups_by_role["mixed"] += 1
        digests.update(
            {
                "hashed": sum(len(files) for files in by_digest.values()),
                "unique_sample_digests": len(by_digest),
                "unique_digests_by_role": unique_by_role,
                "duplicate_groups": len(duplicate_groups),
                "duplicate_groups_by_role": groups_by_role,
                "duplicate_examples": [
                    files for _, files in sorted(duplicate_groups.items())[:10]
                ],
            }
        )
    else:
        digests["hashed"] = 0

    notes: List[str] = []
    if policy.name == "test_final":
        notes.append(
            "file-level inventory only: no SVG content or label attribute was parsed"
        )
    if policy.name == "dev_final" and counts["svg"] == 0:
        notes.append("PNG-only split; no local ground truth and no local PQ")
    if other_names:
        notes.append("non-sample files present: " + ", ".join(other_names))

    entry.update(
        {
            "files": {
                "total": counts["total"],
                "excluded_resource_forks": counts["excluded_resource_forks"],
                "samples": counts["samples"],
                "png": counts["png"],
                "svg": counts["svg"],
                "other": counts["other"],
                "bytes": total_bytes,
                "sample_bytes": sample_bytes,
            },
            "pairing": {
                "png_stems": len(png_set),
                "svg_stems": len(svg_set),
                "paired": len(paired),
                "png_only": len(png_only),
                "svg_only": len(svg_only),
                "duplicate_png_stems": len(png_dupes),
                "duplicate_svg_stems": len(svg_dupes),
                "duplicate_stems": len(png_dupes) + len(svg_dupes),
                "case_collisions": case_collisions[:20],
                "unpaired_examples": {
                    "png_only": png_only[:10],
                    "svg_only": svg_only[:10],
                },
            },
            "digests": digests,
            "file_records": records,
            "notes": notes,
        }
    )
    return entry


def build_inventory(
    root: str,
    splits: Sequence[str] = DEFAULT_SPLITS,
    hash_algorithm: str = "sha256",
) -> Dict[str, Any]:
    """Inventory all requested splits below *root*."""

    split_reports: Dict[str, Any] = {}
    file_records: List[Dict[str, Any]] = []
    for split in splits:
        directory = os.path.join(root, split)
        report = scan_split(directory, split, hash_algorithm)
        file_records.extend(report.pop("file_records"))
        split_reports[split] = report

    totals = {
        "files": 0,
        "excluded_resource_forks": 0,
        "samples": 0,
        "png": 0,
        "svg": 0,
        "other": 0,
        "bytes": 0,
        "paired": 0,
        "unpaired_png": 0,
        "unpaired_svg": 0,
    }
    for report in split_reports.values():
        split_files = report["files"]
        pairing = report["pairing"]
        totals["files"] += split_files["total"]
        totals["excluded_resource_forks"] += split_files["excluded_resource_forks"]
        for key in ("samples", "png", "svg", "other", "bytes"):
            totals[key] += split_files[key]
        totals["paired"] += pairing["paired"]
        totals["unpaired_png"] += pairing["png_only"]
        totals["unpaired_svg"] += pairing["svg_only"]

    return {
        "schema": "cadvec.inventory/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": root,
        "exclusion_rules": [
            "basename starts with '._' (macOS resource fork)",
            "basename is '.DS_Store' or starts with '.DS_Store'",
        ],
        "hash_algorithm": hash_algorithm,
        "splits": split_reports,
        "totals": totals,
        "policies": {name: policy.to_dict() for name, policy in SPLIT_POLICIES.items()},
        "notes": [
            "file hashes read bytes only; no SVG label attribute is interpreted here",
            "pairing uses file-name stems within a split (no manifest paths)",
            "dev_final is PNG-only and has no local ground truth, so no local PQ",
            "test_final is file-level only; do not read its semantic/instance labels",
        ],
        "file_records": file_records,
    }


def inventory_output_paths(out_dir: str) -> Dict[str, str]:
    """Final inventory file paths; callers guard both before any write."""

    return {
        "inventory_json": os.path.join(out_dir, "inventory.json"),
        "files_csv": os.path.join(out_dir, "files.csv"),
    }


def write_inventory(report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """Write ``inventory.json`` plus ``files.csv``; return the paths."""

    os.makedirs(out_dir, exist_ok=True)
    paths = inventory_output_paths(out_dir)
    json_path = paths["inventory_json"]

    file_records = report.pop("file_records", [])
    payload = dict(report)
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")

    csv_path = paths["files_csv"]
    fieldnames = [
        "split",
        "name",
        "role",
        "extension",
        "stem",
        "size_bytes",
        "digest",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in file_records:
            writer.writerow({key: record.get(key) for key in fieldnames})

    report["file_records"] = file_records
    return paths
