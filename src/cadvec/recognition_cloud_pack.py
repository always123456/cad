"""Auditable A100/GNN cloud-pack preparation.

No network, cloud API, upload, or data copy happens by default.  The pack
contains metadata, source-file hashes and a launch template; raw PNG/SVG files
are listed separately and copied only when explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import recognition_baseline as baseline
from . import recognition_gnn as gnn
from .recognition_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    ROLE_EVAL,
    ROLE_FIT,
    ROLE_RESERVE,
    ROLE_TUNE,
    SEED,
    train_prefixes,
)

SCHEMA = "cadvec.recognition.cloud_pack/1"
VERSION = 1
DEFAULT_SCALE = 32
DEFAULT_OUTPUT = os.path.join("outputs", "recognition-gate3c-a100-pack-20261108-01")
RAW_POLICY = "raw_png_svg_not_included_default"


class CloudPackError(ValueError):
    code = "cloud_pack_error"


class PackRejected(CloudPackError):
    code = "cloud_pack_rejected"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _selected_rows(manifest: Mapping[str, Any], scale: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline.validate_manifest(manifest)
    fit_groups = train_prefixes(manifest["role_assignment"], scale)["fit"]
    tune_groups = train_prefixes(manifest["role_assignment"], scale)["tune"]
    if set(fit_groups) & set(tune_groups):
        raise PackRejected("fit_tune_overlap")
    eval_groups = set(manifest["role_assignment"]["eval_pool"])
    reserve_groups = set(manifest["role_assignment"]["reserve"])
    if set(fit_groups + tune_groups) & eval_groups:
        raise PackRejected("eval_group_selected")
    if set(fit_groups + tune_groups) & reserve_groups:
        raise PackRejected("reserve_group_selected")
    fit_rows = baseline._rows_for_groups(manifest, fit_groups, role=ROLE_FIT)
    tune_rows = baseline._rows_for_groups(manifest, tune_groups, role=ROLE_TUNE)
    return fit_rows, tune_rows


def _source_entries(rows: Sequence[Mapping[str, Any]], role: str) -> List[Dict[str, Any]]:
    entries = []
    for row in rows:
        for kind in ("png", "svg"):
            path = row.get(f"{kind}_path")
            if not path:
                raise CloudPackError(f"missing validated {kind}_path")
            real = os.path.realpath(str(path))
            entries.append(
                {
                    "role": role,
                    "sample_id": row["sample_id"],
                    "group_id": row.get("group_id"),
                    "kind": kind,
                    "name": os.path.basename(real),
                    "realpath": real,
                    "sha256": _sha256_file(real),
                    "bytes": os.path.getsize(real),
                    "upload_required_for_training": True,
                    "included_in_pack": False,
                }
            )
    return entries


def _assert_no_eval_text(values: Iterable[str], eval_groups: set[str], eval_digests: set[str]) -> None:
    for value in values:
        text = str(value)
        if any(group in text for group in eval_groups):
            raise PackRejected("eval_group_in_pack_metadata")
        for digest in eval_digests:
            if digest and digest.lower() in text.lower():
                raise PackRejected("eval_digest_in_pack_metadata")


def build_pack(
    manifest_path: str,
    out_dir: str,
    *,
    train_scale: int = DEFAULT_SCALE,
    include_raw: bool = False,
    run_nonce: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = os.path.abspath(manifest_path)
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    fit_rows, tune_rows = _selected_rows(manifest, train_scale)
    fit_rows = baseline.validated_train_file_paths(manifest, fit_rows)
    tune_rows = baseline.validated_train_file_paths(manifest, tune_rows)
    entries = _source_entries(fit_rows, ROLE_FIT) + _source_entries(tune_rows, ROLE_TUNE)
    selected_paths = [entry["realpath"] for entry in entries]
    # Existing baseline boundary: explicit realpath allowlist plus eval hash check.
    baseline.assert_cloud_pack_boundary(manifest, selected_paths, allowlist=selected_paths)
    eval_groups = set(manifest["role_assignment"]["eval_pool"])
    _, _, _, eval_digests = baseline._source_index(manifest)
    _assert_no_eval_text(selected_paths, eval_groups, eval_digests)
    if include_raw:
        for entry in entries:
            destination = os.path.join(out_dir, "raw", entry["role"], entry["name"])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(entry["realpath"], destination)
            entry["included_in_pack"] = True
            entry["pack_path"] = os.path.relpath(destination, out_dir).replace(os.sep, "/")
    nonce = run_nonce or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pack_manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "metadata_ready",
        "run_nonce": nonce,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest": manifest_path,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "seed": SEED,
        "train_scale": train_scale,
        "raw_policy": RAW_POLICY if not include_raw else "raw_png_svg_included_explicitly",
        "raw_upload_required": True,
        "fit_groups": train_prefixes(manifest["role_assignment"], train_scale)["fit"],
        "tune_groups": train_prefixes(manifest["role_assignment"], train_scale)["tune"],
        "counts": {
            "fit_groups": len(fit_rows),
            "tune_groups": len(tune_rows),
            "pairs": len(fit_rows) + len(tune_rows),
            "files": len(entries),
            "raw_included": sum(1 for entry in entries if entry["included_in_pack"]),
        },
        "excluded": {
            "eval_pool": len(eval_groups),
            "reserve": len(manifest["role_assignment"]["reserve"]),
        },
        "file_allowlist": "file_allowlist.json",
        "sha256sums": "sha256sums.txt",
        "run_config": "run_config.json",
        "launch_template": "launch_template.sh",
        "no_network_or_cloud_creation": True,
    }
    file_allowlist = {
        "schema": "cadvec.recognition.cloud_file_allowlist/1",
        "version": 1,
        "manifest_sha256": pack_manifest["source_manifest_sha256"],
        "entries": entries,
    }
    run_config = {
        "schema": "cadvec.recognition.cloud_run_config/1",
        "version": 1,
        "model_id": gnn.C1_MODEL_ID,
        "model_config": dict(gnn.C1_CONFIG),
        "config_fingerprint": gnn.C1_FINGERPRINT,
        "node_features": list(NODE_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
        "classes_raw_policy": "derived from train_fit raw semantic 1..35 at training time",
        "python_candidates": ["3.11", "3.12"],
        "torch_candidate": "2.7.0+cu126",
        "torch_geometric_candidate": "2.8.1",
        "candidate_status": "unverified_on_platform",
        "gpu": "NVIDIA A100-1-80G (4 vCPU / 16G RAM / 80Gi)",
        "batch": "by graph; no cross-graph edges",
        "normalization": "fit-only statistics reused for tune",
        "early_stop": {"max_epochs": 100, "patience": 10, "metric": "tune macro-F1 and edge AP when available"},
        "checkpoint": {"every_seconds": 600, "every_steps": None, "run_nonce": nonce},
        "budget": {
            "platform_points_current": None,
            "screenshot_estimate_points": 480,
            "screenshot_estimate_hours": 8,
            "price_note": "Screenshot estimate only; not a live quote or platform price.",
            "stop_lines": {
                "smoke_max_hours": 1,
                "short_train_max_hours": 4,
                "decision_after": "review validation and remaining budget before extending",
            },
        },
        "eval_access": False,
    }
    sums_lines = [
        f"{entry['sha256']}  {entry['role']}/{entry['name']}"
        for entry in entries
    ]
    readme = f"""# A100 GNN cloud pack (metadata)

This pack was generated without logging into a cloud platform, uploading data,
paying, or creating an instance. It contains metadata and source-file hashes.

- source manifest: `{manifest_path}`
- source manifest SHA256: `{pack_manifest['source_manifest_sha256']}`
- train_scale: `{train_scale}` (fit {len(fit_rows)} / tune {len(tune_rows)})
- raw PNG/SVG: {'included explicitly' if include_raw else 'NOT included; upload separately only after approval'}
- eval_pool/reserve: not selected or read
- no final evaluate, refit, freeze, predict, or platform submission is included.

Manual sequence (not automated here): review this pack, obtain approval and a
live platform quote, verify the Python/torch/PyG wheels, create the instance
manually, upload only the allowlisted files, then run the launch template.
"""
    launch = """#!/usr/bin/env bash
set -euo pipefail
# No data is embedded. Supply an approved, hash-verified pack and run nonce.
: "${RUN_NONCE:?set RUN_NONCE}"
: "${DATA_ROOT:?set DATA_ROOT to the approved upload root}"
test -f "$DATA_ROOT/pack_manifest.json"
# Verify hashes before any training:
(cd "$DATA_ROOT" && sha256sum -c sha256sums.txt)
python -m cadvec.recognition_gnn_train train \\\n  --manifest "$DATA_ROOT/recognition_public_manifest.json" \\\n  --run-dir "$DATA_ROOT/runs/$RUN_NONCE" \\\n  --train-scale 32 --device cuda
"""
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "pack_manifest.json"), pack_manifest)
    _write_json(os.path.join(out_dir, "file_allowlist.json"), file_allowlist)
    _write_json(os.path.join(out_dir, "run_config.json"), run_config)
    with open(os.path.join(out_dir, "sha256sums.txt"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(sums_lines) + "\n")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
    with open(os.path.join(out_dir, "launch_template.sh"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(launch)
    return {"status": "metadata_ready", "out_dir": out_dir, "pack_manifest": pack_manifest, "file_allowlist": file_allowlist}


def expand_archive(path: str) -> List[Dict[str, Any]]:
    """Expand a zip/tar archive into (name, sha256) entries; fails closed."""
    entries: List[Dict[str, Any]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                entries.append({"name": info.filename, "sha256": _sha256_bytes(archive.read(info)), "bytes": info.file_size})
        return entries
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                handle = archive.extractfile(info)
                data = handle.read() if handle else b""
                entries.append({"name": info.name, "sha256": _sha256_bytes(data), "bytes": len(data)})
        return entries
    raise CloudPackError("not a supported archive")


def validate_archive_boundary(manifest: Mapping[str, Any], path: str, *, allowlist: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Reject archive members with eval path/name, eval digest or unlisted alias."""
    entries = expand_archive(path)
    _, _, _, eval_digests = baseline._source_index(manifest)
    eval_groups = set(manifest["role_assignment"]["eval_pool"])
    allowed = {os.path.basename(item) for item in allowlist}
    for entry in entries:
        name = entry["name"].replace("\\", "/")
        parts = [part.lower() for part in name.split("/")]
        if any(part in ("eval", "eval_artifacts", "evaluation") for part in parts):
            raise PackRejected(f"archive_eval_path:{name}")
        _assert_no_eval_text([name], eval_groups, eval_digests)
        if entry["sha256"].lower() in eval_digests:
            raise PackRejected(f"archive_eval_digest:{name}")
        if allowed and os.path.basename(name) not in allowed:
            raise PackRejected(f"archive_unlisted_alias:{name}")
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cadvec.recognition_cloud_pack")
    parser.add_argument("build", nargs="?")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--run-nonce", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_pack(
            args.manifest,
            args.out,
            train_scale=args.train_scale,
            include_raw=args.include_raw,
            run_nonce=args.run_nonce,
        )
    except (OSError, ValueError, CloudPackError) as exc:
        print(f"error: {getattr(exc, 'code', type(exc).__name__)}: {exc}")
        return 2
    print(json.dumps(result["pack_manifest"], ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
