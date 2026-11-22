"""PNG diagnostic runner: prepare, preview and evaluate.

This is the thin orchestration layer over the already-validated modules:

* :mod:`cadvec.inventory` rebuilds a train-only file inventory,
* :mod:`cadvec.png_diagnostic_samples` loads/group/freezes metadata,
* :mod:`cadvec.png_frontend` predicts one PNG at a time,
* :mod:`cadvec.png_diagnostic_adapter` turns a parse document into GT polylines
  and the viewBox into an explicit pixel mapping,
* :mod:`cadvec.geometric_eval` computes the distances/coverage.

The runner never re-implements GT reconstruction, coordinate mapping, duplicate
grouping or distance maths.  It only orchestrates, guards output paths, enforces
compute/wall-clock budgets and writes diagnostic artifacts.

CLI::

    python -m cadvec.png_diagnostic_runner prepare --datasets D:/data --out RUN
    python -m cadvec.png_diagnostic_runner preview --run-dir RUN --out RUN/preview
    python -m cadvec.png_diagnostic_runner evaluate --run-dir RUN --out RUN/eval-dev \\
        --subset development [--alignment-review REVIEW.json]

All JSON is strict (``allow_nan=False``) and written through a same-directory
temporary file + ``os.replace``.  The last file written is ``completion.json``;
a partial or failed run never gets a success marker.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__
from . import geometric_eval, inventory, policy, png_diagnostic_adapter, png_frontend
from . import png_diagnostic_samples as samples
from .cli import _guard_write_targets  # private compatibility, see docs
from .png_frontend import FrontendConfig

SCHEMA = "cadvec.png_diagnostic_runner/1"
SCHEMA_VERSION = 1
ALIGNMENT_REVIEW_SCHEMA = "cadvec.png_diagnostic_alignment_review/1"
ALIGNMENT_REVIEW_VERSION = 1

DEFAULT_SAMPLE_COUNT = 24
DEFAULT_SEED = 20261108
DEFAULT_DEVELOPMENT_FRACTION = 0.5
DEFAULT_STEP_PX = 1.0
DEFAULT_MAX_SAMPLES = 20000
DEFAULT_TOLERANCES = (1.0, 2.0, 4.0)
DEFAULT_DUPLICATE_QUANTUM = 0.5
DEFAULT_CURVE_STEPS = 128
DEFAULT_CIRCLE_STEPS = 128
DEFAULT_COMPUTE_BUDGET = 10_000_000
DEFAULT_IMAGE_TIMEOUT_S = 60.0
DEFAULT_RUN_BUDGET_S = 1800.0
DEFAULT_MAX_PREVIEW = 2
DEFAULT_MAX_OVERLAYS = 6
DEFAULT_VIEWPORT_MODE = "full_svg_viewport"

RUN_FILES = (
    "inventory.json",
    "files.csv",
    "sample_rows.json",
    "duplicate_groups.json",
    "frozen_selection.json",
    "locked_config.json",
    "prepare_manifest.json",
    "completion.json",
)
PREVIEW_FILES = ("preview.json", "preview_samples.csv", "conclusion.md", "completion.json")
EVALUATE_FILES = (
    "statuses.csv",
    "summary.json",
    "summary.csv",
    "conclusion.md",
    "evaluate_manifest.json",
    "completion.json",
)

WRITER_COUNTER = 0
_clock = time.monotonic


class RunnerError(Exception):
    """Configuration, guard or input failure before a deliverable exists."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: str, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise RunnerError(f"cannot read {label} {path!r}: {type(exc).__name__}: {exc}")
    except ValueError as exc:
        raise RunnerError(f"{label} {path!r} is not valid JSON: {exc}")


def _code_fingerprints() -> Dict[str, str]:
    directory = os.path.dirname(os.path.abspath(__file__))
    names = (
        "png_diagnostic_runner.py",
        "png_diagnostic_adapter.py",
        "png_diagnostic_samples.py",
        "png_frontend.py",
        "geometric_eval.py",
        "inventory.py",
        "policy.py",
        "svg_parse.py",
        "geometry.py",
    )
    fingerprints = {}
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            fingerprints[name] = _sha256_file(path)
    return fingerprints


def _versions() -> Dict[str, Any]:
    return {
        "cadvec": __version__,
        "runner_schema": f"{SCHEMA}/v{SCHEMA_VERSION}",
        "adapter_schema": png_diagnostic_adapter.SCHEMA,
        "samples_schema": samples.SCHEMA,
        "frontend_schema": png_frontend.SCHEMA,
        "evaluation_schema": geometric_eval.SCHEMA,
    }


def _is_inside(path: str, root: str) -> bool:
    path_c = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    root_c = os.path.normcase(os.path.realpath(os.path.abspath(root)))
    return path_c == root_c or path_c.startswith(root_c + os.sep)


def _guard(
    datasets_root: str,
    out_dir: str,
    targets: Sequence[str],
    protected_inputs: Sequence[str],
) -> None:
    """Guard write targets before any mkdir/open.

    The existing CLI guard is reused via its private name (documented
    compatibility dependency).  Extra runner checks reject an output tree that
    contains the dataset/input root, and inputs that live below the output.
    """

    if _is_inside(out_dir, datasets_root) or _is_inside(datasets_root, out_dir):
        raise RunnerError(
            "output directory must not be inside, equal to, or an ancestor of the "
            "dataset root"
        )
    if policy.split_of_path(out_dir) in ("test_final", "dev_final"):
        raise RunnerError(
            "output directory resolves into a restricted test/dev split"
        )
    for item in protected_inputs:
        if _is_inside(item, out_dir):
            raise RunnerError(
                f"input {item!r} is inside the output directory; refusing to shadow it"
            )
    try:
        _guard_write_targets(datasets_root, list(targets), list(protected_inputs))
    except (ValueError, OSError) as exc:
        raise RunnerError(str(exc)) from exc


def _ensure_run_dir(out_dir: str, datasets_root: str, inputs: Sequence[str]) -> str:
    absolute = os.path.abspath(out_dir)
    if os.path.exists(absolute):
        if not os.path.isdir(absolute):
            raise RunnerError(f"output path exists and is not a directory: {out_dir}")
        if os.listdir(absolute):
            raise RunnerError(
                f"output directory already exists and is not empty: {out_dir}"
            )
    _guard(datasets_root, absolute, [absolute], inputs)
    os.makedirs(absolute, exist_ok=True)
    return absolute


class _Writer:
    """Strict-JSON / CSV writer with guard + atomic replace."""

    def __init__(self, out_dir: str, guard) -> None:
        self.out_dir = out_dir
        self.guard = guard
        self.counter = 0

    def _path(self, name: str) -> str:
        if os.path.isabs(name):
            raise RunnerError(f"unsafe output file name: {name!r}")
        parts = [part for part in name.replace("\\", "/").split("/") if part]
        if not parts or any(part in (".", "..") for part in parts):
            raise RunnerError(f"unsafe output file name: {name!r}")
        path = os.path.join(self.out_dir, *parts)
        if not _is_inside(path, self.out_dir):
            raise RunnerError(f"output path escapes the run directory: {name!r}")
        return path

    def _atomic(self, name: str, data: bytes) -> str:
        global WRITER_COUNTER
        path = self._path(name)
        WRITER_COUNTER += 1
        base = os.path.basename(path)
        temp = os.path.join(
            os.path.dirname(path),
            f".{base}.tmp-{os.getpid()}-{WRITER_COUNTER}",
        )
        self.guard([path, temp])
        try:
            with open(temp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self.guard([path])
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
        return path

    def json(self, name: str, payload: Any) -> str:
        return self._atomic(name, _json_bytes(payload))

    def csv(self, name: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> str:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        return self._atomic(name, buffer.getvalue().encode("utf-8"))

    def bytes(self, name: str, data: bytes) -> str:
        return self._atomic(name, data)


def _flatten_selected(frozen: Dict[str, Any], wanted: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    allowed = set(wanted) if wanted is not None else None
    for group in frozen.get("groups", []):
        split = group.get("split")
        if allowed is not None and split not in allowed:
            continue
        for member in group.get("members", []):
            row = copy.deepcopy(member)
            row["_group_id"] = group.get("group_id")
            row["_split"] = split
            rows.append(row)
    rows.sort(key=lambda item: item["sample_id"])
    return rows


def _selected_rows_for(frozen: Dict[str, Any], subset: str) -> List[Dict[str, Any]]:
    if subset == "development":
        return _flatten_selected(frozen, ("development",))
    if subset == "holdout":
        return _flatten_selected(frozen, ("holdout",))
    if subset == "all":
        return _flatten_selected(frozen)
    raise RunnerError(f"subset must be development, holdout or all, got {subset!r}")


def _unique_png_count(rows: Sequence[Dict[str, Any]]) -> int:
    return len(
        {
            row.get("png_digest") or row.get("png_name") or row.get("sample_id")
            for row in rows
        }
    )


def _distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"development": 0, "holdout": 0}
    for row in rows:
        split = row.get("_split") or row.get("split")
        if split in counts:
            counts[split] += 1
    return counts


def _polyline_stats(polylines: Sequence[Sequence[Sequence[float]]]) -> Dict[str, Any]:
    positive = 0
    length = 0.0
    for polyline in polylines:
        for index in range(1, len(polyline)):
            x1, y1 = polyline[index - 1]
            x2, y2 = polyline[index]
            segment = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
            if segment > 0.0:
                positive += 1
                length += segment
    return {"positive_segments": positive, "length": length}


def _outside_viewport(polylines: Sequence[Sequence[Sequence[float]]], viewport: Dict[str, Any]) -> bool:
    if not isinstance(viewport, dict):
        return True
    try:
        x0 = float(viewport["x"])
        y0 = float(viewport["y"])
        x1 = x0 + float(viewport["width"])
        y1 = y0 + float(viewport["height"])
    except (KeyError, TypeError, ValueError):
        return True
    eps = 1e-6
    for polyline in polylines:
        for point in polyline:
            x, y = float(point[0]), float(point[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return True
            if x < x0 - eps or x > x1 + eps or y < y0 - eps or y > y1 + eps:
                return True
    return False


def _eval_config(config: Dict[str, Any]) -> Tuple[float, int, Sequence[float], float]:
    evaluation = config.get("evaluation") or {}
    step = float(evaluation.get("step_px", DEFAULT_STEP_PX))
    max_samples = int(evaluation.get("max_samples", DEFAULT_MAX_SAMPLES))
    tolerances = tuple(float(item) for item in evaluation.get("tolerances", DEFAULT_TOLERANCES))
    quantum = float(evaluation.get("duplicate_quantum_px", DEFAULT_DUPLICATE_QUANTUM))
    return step, max_samples, tolerances, quantum


def _validate_budget_values(
    compute_budget: int, image_timeout: float, run_budget: float
) -> Tuple[int, float, float]:
    if isinstance(compute_budget, bool) or not isinstance(compute_budget, int):
        raise RunnerError("compute_budget must be an integer")
    if compute_budget <= 0:
        raise RunnerError("compute_budget must be positive")
    for name, value in (("image_timeout", image_timeout), ("run_budget", run_budget)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RunnerError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise RunnerError(f"{name} must be a finite positive number")
    return compute_budget, float(image_timeout), float(run_budget)


def _locked_config(
    selection_fingerprint: str,
    selection_content_sha256: str,
    frontend_config: FrontendConfig,
    *,
    compute_budget: int = DEFAULT_COMPUTE_BUDGET,
    image_timeout: float = DEFAULT_IMAGE_TIMEOUT_S,
    run_budget: float = DEFAULT_RUN_BUDGET_S,
) -> Dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "selection_content_sha256": selection_content_sha256,
        "frontend": frontend_config.to_dict(),
        "adapter": {
            "curve_steps": DEFAULT_CURVE_STEPS,
            "circle_steps": DEFAULT_CIRCLE_STEPS,
            "viewport_mode": DEFAULT_VIEWPORT_MODE,
        },
        "evaluation": {
            "step_px": DEFAULT_STEP_PX,
            "max_samples": DEFAULT_MAX_SAMPLES,
            "tolerances": list(DEFAULT_TOLERANCES),
            "duplicate_quantum_px": DEFAULT_DUPLICATE_QUANTUM,
        },
        "budgets": {
            "compute_budget_per_scope": compute_budget,
            "image_timeout_s": image_timeout,
            "run_budget_s": run_budget,
            "compute_budget_method": (
                "upper_bound max_samples * (gt_positive_segments + "
                "pred_positive_segments) * 2 directions; may trigger early limited"
            ),
        },
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _train_paths(row: Dict[str, Any], train_dir: str) -> Tuple[str, str]:
    png_path = os.path.join(train_dir, row["png_name"])
    svg_path = os.path.join(train_dir, row["svg_name"])
    train_real = os.path.realpath(train_dir)
    for path in (png_path, svg_path):
        if not _is_inside(path, train_real):
            raise RunnerError(
                f"sample file escapes the train directory: {os.path.basename(path)!r}"
            )
    return png_path, svg_path


def prepare_run(
    datasets: str,
    out: str,
    *,
    manifest: Optional[str] = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    development_fraction: float = DEFAULT_DEVELOPMENT_FRACTION,
    compute_budget: int = DEFAULT_COMPUTE_BUDGET,
    image_timeout: float = DEFAULT_IMAGE_TIMEOUT_S,
    run_budget: float = DEFAULT_RUN_BUDGET_S,
) -> Tuple[Dict[str, Any], int]:
    started = time.monotonic()
    compute_budget, image_timeout, run_budget = _validate_budget_values(
        compute_budget, image_timeout, run_budget
    )
    datasets_abs = os.path.abspath(datasets)
    if not os.path.isdir(datasets_abs):
        raise RunnerError(f"dataset root is missing or not a directory: {datasets!r}")
    if policy.split_of_path(datasets_abs) in ("test_final", "dev_final"):
        raise RunnerError("dataset root resolves into a restricted split")
    train_dir = os.path.join(datasets_abs, "train_final")
    if not os.path.isdir(train_dir):
        raise RunnerError(f"train_final directory is missing under {datasets_abs!r}")
    if policy.split_of_path(train_dir) in ("test_final", "dev_final"):
        raise RunnerError("train_final path resolves into a restricted split")
    if os.path.realpath(train_dir) != os.path.realpath(
        os.path.join(datasets_abs, "train_final")
    ):
        raise RunnerError("train_final must not be a symlink to another split")

    manifest_path = manifest
    if manifest_path is None:
        candidate = os.path.join(train_dir, "manifest.csv")
        manifest_path = candidate if os.path.isfile(candidate) else None
    if manifest_path is not None:
        manifest_path = os.path.abspath(manifest_path)
        if policy.split_of_path(manifest_path) in ("test_final", "dev_final"):
            raise RunnerError("manifest resolves into a restricted split")
        if not os.path.isfile(manifest_path):
            raise RunnerError(f"manifest is missing: {manifest_path!r}")

    target_names = RUN_FILES
    targets = [os.path.join(os.path.abspath(out), name) for name in target_names]
    protected = [manifest_path] if manifest_path else []
    out_abs = _ensure_run_dir(out, datasets_abs, protected)
    _guard(datasets_abs, out_abs, targets, protected)
    guard = lambda extra: _guard(datasets_abs, out_abs, list(extra), protected)
    writer = _Writer(out_abs, guard)

    report = inventory.build_inventory(
        datasets_abs, splits=("train_final",), hash_algorithm="sha256"
    )
    train_report = report["splits"]["train_final"]
    train_counts = {
        "files": train_report["files"]["total"],
        "samples": train_report["files"]["samples"],
        "png": train_report["files"]["png"],
        "svg": train_report["files"]["svg"],
        "paired_stems": train_report["pairing"]["paired"],
        "unpaired_png": train_report["pairing"]["png_only"],
        "unpaired_svg": train_report["pairing"]["svg_only"],
    }
    inventory_payload = copy.deepcopy(report)
    file_records = inventory_payload.pop("file_records", [])
    writer.json("inventory.json", inventory_payload)
    writer.csv(
        "files.csv",
        file_records,
        ("split", "name", "role", "extension", "stem", "size_bytes", "digest"),
    )

    loaded = samples.load_sample_rows(
        os.path.join(out_abs, "files.csv"), manifest_path
    )
    writer.json("sample_rows.json", loaded)
    if not loaded["rows"]:
        completion = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "command": "prepare",
            "status": "failed",
            "status_reason": "no_paired_train_samples",
            "generated_at": _now(),
            "train_counts": train_counts,
            "versions": _versions(),
            "code_fingerprints": _code_fingerprints(),
            "deliverable_complete": False,
            "metrics_evaluable": False,
        }
        writer.json("completion.json", completion)
        return completion, 2

    missing_digest_rows = [
        row for row in loaded["rows"] if not row.get("png_digest") or not row.get("svg_digest")
    ]
    if missing_digest_rows or loaded["accounting"].get("missing_digest_rows") or loaded["accounting"].get("invalid_digest_rows"):
        completion = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "command": "prepare",
            "status": "failed",
            "status_reason": "missing_or_invalid_digest",
            "generated_at": _now(),
            "train_counts": train_counts,
            "versions": _versions(),
            "code_fingerprints": _code_fingerprints(),
            "deliverable_complete": False,
            "metrics_evaluable": False,
        }
        writer.json("completion.json", completion)
        return completion, 2

    grouped = samples.group_exact_duplicates(loaded["rows"])
    writer.json("duplicate_groups.json", grouped)
    frozen = samples.freeze_selection(
        grouped,
        sample_count=sample_count,
        seed=seed,
        development_fraction=development_fraction,
    )
    writer.json("frozen_selection.json", frozen)
    frontend_config = FrontendConfig()
    selection_content_sha256 = _fingerprint(frozen)
    locked = _locked_config(
        frozen["fingerprint"],
        selection_content_sha256,
        frontend_config,
        compute_budget=compute_budget,
        image_timeout=image_timeout,
        run_budget=run_budget,
    )
    writer.json("locked_config.json", locked)
    artifacts_sha256 = {
        "sample_rows.json": _fingerprint(loaded),
        "duplicate_groups.json": _fingerprint(grouped),
        "frozen_selection.json": selection_content_sha256,
        "locked_config.json": _fingerprint(locked),
        "files.csv": _sha256_file(os.path.join(out_abs, "files.csv")),
        "inventory.json": _sha256_file(os.path.join(out_abs, "inventory.json")),
    }

    status = "complete"
    reasons: List[str] = []
    if frozen["status"] == "split_limited":
        status = "split_limited"
        reasons.append("fewer_than_two_groups")
    elif frozen["status"] == "selection_limited":
        status = "selection_limited"
        reasons.extend(warning["code"] for warning in frozen["warnings"])
    elif frozen["warnings"]:
        reasons.extend(warning["code"] for warning in frozen["warnings"])

    manifest_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "prepare",
        "status": status,
        "status_reasons": reasons,
        "generated_at": _now(),
        "datasets_root": datasets_abs,
        "train_dir": train_dir,
        "manifest": manifest_path,
        "selection_fingerprint": frozen["fingerprint"],
        "selection_content_sha256": selection_content_sha256,
        "locked_config_fingerprint": locked["fingerprint"],
        "frontend_config_fingerprint": _fingerprint(frontend_config.to_dict()),
        "artifacts_sha256": artifacts_sha256,
        "budget_source": "prepare_locked",
        "train_counts": train_counts,
        "selection_counts": frozen["counts"],
        "selection_status": frozen["status"],
        "loaded_accounting": loaded["accounting"],
        "issue_counts": _issue_counts(loaded["issues"]),
        "versions": _versions(),
        "code_fingerprints": _code_fingerprints(),
        "elapsed_s": round(_clock() - started, 3),
        "notes": [
            "inventory digests are a read-time snapshot; preview/evaluate re-hash "
            "the selected files and stop on drift",
            "manifest source paths are control-side metadata only and are never opened",
            "no SVG was parsed during prepare",
            "selection is group-atomic; every annotation variant is retained",
        ],
    }
    writer.json("prepare_manifest.json", manifest_payload)
    prepare_manifest_sha256 = _fingerprint(manifest_payload)

    completion = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "prepare",
        "status": status,
        "status_reasons": reasons,
        "generated_at": _now(),
        "selection_fingerprint": frozen["fingerprint"],
        "selection_content_sha256": selection_content_sha256,
        "locked_config_fingerprint": locked["fingerprint"],
        "artifacts_sha256": artifacts_sha256,
        "prepare_manifest_sha256": prepare_manifest_sha256,
        "counts": frozen["counts"],
        "train_counts": train_counts,
        "deliverable_complete": True,
        "metrics_evaluable": False,
        "versions": _versions(),
        "code_fingerprints": _code_fingerprints(),
        "elapsed_s": round(_clock() - started, 3),
    }
    writer.json("completion.json", completion)
    exit_code = 0 if status == "complete" else 1
    return completion, exit_code


def _issue_counts(issues: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for issue in issues:
        code = str(issue.get("code"))
        counts[code] = counts.get(code, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# shared input loading and hash verification
# ---------------------------------------------------------------------------


def _load_run(run_dir: str) -> Dict[str, Any]:
    absolute = os.path.abspath(run_dir)
    if not os.path.isdir(absolute):
        raise RunnerError(f"run directory is missing: {run_dir!r}")
    for name in RUN_FILES:
        if not os.path.isfile(os.path.join(absolute, name)):
            raise RunnerError(
                f"content_changed: preparation artifact {name!r} is missing; "
                "rerun prepare"
            )
    frozen = _read_json(os.path.join(absolute, "frozen_selection.json"), "frozen selection")
    locked = _read_json(os.path.join(absolute, "locked_config.json"), "locked config")
    manifest = _read_json(os.path.join(absolute, "prepare_manifest.json"), "prepare manifest")
    sample_rows = _read_json(os.path.join(absolute, "sample_rows.json"), "sample rows")
    grouped = _read_json(os.path.join(absolute, "duplicate_groups.json"), "duplicate groups")
    completion = _read_json(os.path.join(absolute, "completion.json"), "prepare completion")

    for label, document in (
        ("locked_config", locked),
        ("prepare_manifest", manifest),
        ("completion", completion),
    ):
        if not isinstance(document, dict):
            raise RunnerError(f"content_changed: {label} is not a JSON object")
        if document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
            raise RunnerError(
                f"content_changed: {label} schema/version mismatch; rerun prepare"
            )
    for label, document in (
        ("frozen_selection", frozen),
        ("sample_rows", sample_rows),
        ("duplicate_groups", grouped),
    ):
        if not isinstance(document, dict):
            raise RunnerError(f"content_changed: {label} is not a JSON object")
        if (
            document.get("schema") != samples.SCHEMA
            or document.get("schema_version") != samples.SCHEMA_VERSION
        ):
            raise RunnerError(
                f"content_changed: {label} schema/version mismatch; rerun prepare"
            )
    if completion.get("command") != "prepare":
        raise RunnerError("content_changed: completion is not a prepare completion")
    if completion.get("deliverable_complete") is not True:
        raise RunnerError("content_changed: prepare did not complete its deliverables")
    if completion.get("status") not in ("complete", "selection_limited", "split_limited"):
        raise RunnerError("content_changed: prepare completion is failed or unknown")
    if sample_rows.get("schema") != samples.SCHEMA:
        raise RunnerError("content_changed: sample_rows schema mismatch")
    if grouped.get("schema") != samples.SCHEMA:
        raise RunnerError("content_changed: duplicate_groups schema mismatch")
    if frozen.get("schema") != samples.SCHEMA:
        raise RunnerError("content_changed: frozen selection schema mismatch")

    locked_content = {k: locked[k] for k in locked if k != "fingerprint"}
    if _fingerprint(locked_content) != locked.get("fingerprint"):
        raise RunnerError("content_changed: locked_config fingerprint mismatch")
    if manifest.get("locked_config_fingerprint") != locked.get("fingerprint"):
        raise RunnerError(
            "content_changed: prepare manifest locked config identity mismatch"
        )
    if manifest.get("frontend_config_fingerprint") != _fingerprint(
        locked.get("frontend") or {}
    ):
        raise RunnerError(
            "content_changed: prepare manifest frontend config identity mismatch"
        )
    if manifest.get("selection_fingerprint") != frozen.get("fingerprint"):
        raise RunnerError("content_changed: manifest selection fingerprint mismatch")
    if locked.get("selection_fingerprint") != frozen.get("fingerprint"):
        raise RunnerError("content_changed: locked selection fingerprint mismatch")

    # Rebuild the complete candidate pool and the frozen selection from the
    # persisted provenance, then compare canonical content.  The frozen file's
    # self-reported fingerprint is never used as proof on its own.
    rows = sample_rows.get("rows")
    if not isinstance(rows, list):
        raise RunnerError("content_changed: sample_rows has no rows array")
    recomputed_groups = samples.group_exact_duplicates(rows)
    if _fingerprint(recomputed_groups) != _fingerprint(grouped):
        raise RunnerError(
            "content_changed: duplicate groups do not match persisted sample rows"
        )
    config = frozen.get("config") or {}
    try:
        recomputed_frozen = samples.freeze_selection(
            recomputed_groups,
            sample_count=config["sample_count"],
            seed=config["seed"],
            development_fraction=config["development_fraction"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(
            f"content_changed: frozen selection config is invalid ({exc}); rerun prepare"
        )
    if _fingerprint(recomputed_frozen) != _fingerprint(frozen):
        raise RunnerError(
            "content_changed: frozen selection does not match its provenance"
        )
    selection_content_sha256 = _fingerprint(frozen)
    for label, document in (
        ("locked_config", locked),
        ("prepare_manifest", manifest),
        ("completion", completion),
    ):
        if document.get("selection_content_sha256") != selection_content_sha256:
            raise RunnerError(
                f"content_changed: {label} selection content digest mismatch"
            )

    artifacts = manifest.get("artifacts_sha256") or {}
    expected_artifacts = {
        "sample_rows.json": _fingerprint(sample_rows),
        "duplicate_groups.json": _fingerprint(grouped),
        "frozen_selection.json": selection_content_sha256,
        "locked_config.json": _fingerprint(locked),
        "files.csv": _sha256_file(os.path.join(absolute, "files.csv")),
        "inventory.json": _sha256_file(os.path.join(absolute, "inventory.json")),
    }
    if artifacts != expected_artifacts:
        raise RunnerError("content_changed: prepare artifact digests mismatch")
    if completion.get("artifacts_sha256") != artifacts:
        raise RunnerError("content_changed: completion artifact anchor mismatch")
    if completion.get("prepare_manifest_sha256") != _fingerprint(manifest):
        raise RunnerError("content_changed: prepare manifest digest mismatch")

    datasets_root = manifest.get("datasets_root")
    train_dir = manifest.get("train_dir")
    if not isinstance(datasets_root, str) or not isinstance(train_dir, str):
        raise RunnerError("content_changed: prepare manifest lacks dataset paths")
    datasets_real = os.path.realpath(datasets_root)
    train_real = os.path.realpath(train_dir)
    expected_train = os.path.realpath(os.path.join(datasets_real, "train_final"))
    if train_real != expected_train:
        raise RunnerError(
            "content_changed: train_dir no longer resolves to datasets_root/train_final"
        )
    if not os.path.isdir(datasets_real) or not os.path.isdir(train_real):
        raise RunnerError("content_changed: dataset or train directory is missing")
    if policy.split_of_path(datasets_real) in ("test_final", "dev_final"):
        raise RunnerError("content_changed: dataset root resolves into a restricted split")
    if policy.split_of_path(train_real) in ("test_final", "dev_final"):
        raise RunnerError("content_changed: train directory resolves into a restricted split")

    current_code = _code_fingerprints()
    if manifest.get("code_fingerprints") != current_code:
        raise RunnerError(
            "code_changed: runner/module source differs from prepare; rerun prepare "
            "for a new run"
        )
    if manifest.get("budget_source") != "prepare_locked":
        raise RunnerError("content_changed: budget source is not prepare-locked")

    return {
        "run_dir": absolute,
        "frozen": frozen,
        "locked": locked,
        "manifest": manifest,
        "completion": completion,
        "sample_rows": sample_rows,
        "grouped": grouped,
        "selection_content_sha256": selection_content_sha256,
        "code_fingerprints": current_code,
    }


def _verify_files(rows: Sequence[Dict[str, Any]], train_dir: str) -> Dict[str, Dict[str, str]]:
    """Hash every selected PNG/SVG and compare with the frozen metadata."""

    cache: Dict[str, str] = {}
    verified: Dict[str, Dict[str, str]] = {}
    for row in rows:
        png_path, svg_path = _train_paths(row, train_dir)
        entry = {"png": "ok", "svg": "ok", "png_digest": "", "svg_digest": ""}
        for kind, path, expected in (
            ("png", png_path, row.get("png_digest")),
            ("svg", svg_path, row.get("svg_digest")),
        ):
            try:
                if path not in cache:
                    cache[path] = _sha256_file(path)
                actual = cache[path]
            except OSError as exc:
                entry[kind] = f"unreadable:{type(exc).__name__}"
                continue
            entry[f"{kind}_digest"] = actual
            if not expected:
                entry[kind] = "digest_unavailable"
            elif actual != str(expected).lower():
                entry[kind] = "changed"
        if entry["png"] != "ok" or entry["svg"] != "ok":
            entry["status"] = "input_changed"
        else:
            entry["status"] = "ok"
        verified[row["sample_id"]] = entry
    return verified


def _subset_for_row(row: Dict[str, Any]) -> str:
    return row.get("_split") or row.get("split") or "development"


# ---------------------------------------------------------------------------
# worker (spawn-safe)
# ---------------------------------------------------------------------------


def _worker_file(payload_path: str, result_path: str) -> None:
    try:
        with open(payload_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = _run_task(payload)
    except BaseException as exc:  # noqa: BLE001 - worker must report, not crash silently
        result = {
            "status": "failed",
            "status_reason": f"worker_error:{type(exc).__name__}",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = payload["mode"]
    config_dict = payload["frontend_config"]
    config = FrontendConfig.from_dict(config_dict)
    locked_config = payload.get("locked_config") or {}
    adapter_config = locked_config.get("adapter") or {}
    curve_steps = adapter_config.get("curve_steps")
    circle_steps = adapter_config.get("circle_steps")
    viewport_mode = adapter_config.get("viewport_mode")
    if (
        isinstance(curve_steps, bool)
        or not isinstance(curve_steps, int)
        or curve_steps <= 0
        or isinstance(circle_steps, bool)
        or not isinstance(circle_steps, int)
        or circle_steps <= 0
        or viewport_mode != DEFAULT_VIEWPORT_MODE
    ):
        return {
            "status": "failed",
            "status_reason": "unsupported_locked_adapter_config",
            "error": "locked adapter configuration is missing or unsupported",
        }
    started = time.monotonic()
    frontend_result = png_frontend.predict_png(payload["png_path"], config)
    timings = {"predict_s": round(time.monotonic() - started, 3)}

    if frontend_result.get("status") in (
        png_frontend.STATUS_INPUT_ERROR,
        png_frontend.STATUS_DECODE_ERROR,
        png_frontend.STATUS_UNSUPPORTED,
    ):
        return {
            "status": "failed",
            "status_reason": frontend_result.get("status_reason") or "frontend_failed",
            "frontend": frontend_result,
            "timings": timings,
        }

    width = (frontend_result.get("input") or {}).get("width")
    height = (frontend_result.get("input") or {}).get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return {
            "status": "failed",
            "status_reason": "frontend_missing_dimensions",
            "frontend": frontend_result,
            "timings": timings,
        }

    pred_result = png_diagnostic_adapter.pred_from_frontend(frontend_result)
    pred_polylines = pred_result.get("polylines") or []

    variants = []
    parse_started = time.monotonic()
    for variant in payload["variants"]:
        try:
            doc = _parse_svg_document(variant["svg_path"], curve_steps)
        except Exception as exc:  # noqa: BLE001 - reported per variant
            variants.append(
                {
                    "sample_id": variant["sample_id"],
                    "status": "failed",
                    "status_reason": f"parse_error:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            continue
        gt = png_diagnostic_adapter.gt_from_document(
            doc, curve_steps=curve_steps, circle_steps=circle_steps
        )
        preserve = None
        root = doc.get("root") or {}
        if isinstance(root, dict):
            preserve = (root.get("attributes") or {}).get("preserveAspectRatio")
        mapping = png_diagnostic_adapter.map_viewbox_to_pixels(
            doc.get("view_box"), width, height, preserve,
            viewport_mode=viewport_mode,
        )
        entry = {
            "sample_id": variant["sample_id"],
            "status": "ok",
            "gt": gt,
            "mapping": mapping,
            "gt_pixels": {},
            "gt_stats": {},
        }
        if gt.get("status") == "ok" and mapping.get("status") == "ok":
            for scope, key in (
                ("all_strokes", "polylines_all"),
                ("straight_only", "polylines_straight"),
            ):
                try:
                    mapped = png_diagnostic_adapter.apply_pixel_mapping(
                        gt[key], mapping
                    )
                except png_diagnostic_adapter.AdapterInputError as exc:
                    entry["status"] = "failed"
                    entry["status_reason"] = f"mapping_input_error:{exc}"
                    break
                entry["gt_pixels"][scope] = mapped
                entry["gt_stats"][scope] = _polyline_stats(mapped)
        variants.append(entry)
    timings["gt_s"] = round(time.monotonic() - parse_started, 3)

    if mode == "preview":
        return {
            "status": "ok",
            "frontend": frontend_result,
            "pred": pred_result,
            "variants": variants,
            "input": {"width": width, "height": height},
            "timings": timings,
        }

    # evaluate
    step, max_samples, tolerances, quantum = _eval_config(payload["locked_config"])
    compute_budget = int(locked_config["budgets"]["compute_budget_per_scope"])
    review_applies = bool(payload.get("review_applies"))
    scope_results: Dict[str, Any] = {}
    overall_pending = None

    for variant in variants:
        if variant["status"] != "ok":
            continue
        gt = variant["gt"]
        mapping = variant["mapping"]
        if gt.get("status") == "excluded":
            variant["status"] = "excluded"
            variant["status_reason"] = "unresolved_transform"
            continue
        if gt.get("status") == "failed":
            variant["status"] = "failed"
            variant["status_reason"] = "gt_rebuild_failed"
            continue
        if gt.get("complete") is False:
            variant["status"] = "alignment_pending"
            variant["status_reason"] = "gt_incomplete_unsupported"
            continue
        if mapping.get("status") != "ok":
            variant["status"] = "alignment_pending"
            variant["status_reason"] = "mapping_pending:" + ",".join(mapping.get("reasons") or [])
            continue
        if _outside_viewport(variant["gt_pixels"].get("all_strokes", []), mapping.get("viewport") or {}):
            variant["status"] = "alignment_pending"
            variant["status_reason"] = "gt_outside_viewport"
            continue
        if not review_applies:
            variant["status"] = "alignment_pending"
            variant["status_reason"] = "alignment_unverified_no_review"
            continue
        if pred_result.get("status") == "failed":
            variant["status"] = "failed"
            variant["status_reason"] = "prediction_failed"
            continue
        if pred_result.get("status") == "evaluation_limited":
            variant["status"] = "evaluation_limited"
            variant["status_reason"] = "prediction_limited"
            continue

        per_scope: Dict[str, Any] = {}
        for scope, key in (
            ("all_strokes", "all_strokes"),
            ("straight_only", "straight_only"),
        ):
            gt_polylines = variant["gt_pixels"].get(key, [])
            gt_stats = variant["gt_stats"].get(key, {"positive_segments": 0, "length": 0.0})
            estimate = int(max_samples) * (
                int(gt_stats["positive_segments"]) + len(pred_polylines)
            ) * 2
            budget = {
                "estimated_point_segment_operations": estimate,
                "compute_budget": compute_budget,
                "method": "upper_bound_two_directions",
                "may_limit_early": True,
                "gt_positive_segments": gt_stats["positive_segments"],
                "gt_length_px": gt_stats["length"],
                "pred_positive_segments": len(pred_polylines),
            }
            if estimate > compute_budget:
                per_scope[scope] = {
                    "status": "evaluation_limited",
                    "status_reason": "compute_budget_exceeded",
                    "support_scope": scope,
                    "budget": budget,
                    "gt": {"positive_segments": gt_stats["positive_segments"], "length": gt_stats["length"]},
                    "pred": {"positive_segments": len(pred_polylines)},
                    "directions": {},
                }
                continue
            try:
                per_scope[scope] = geometric_eval.evaluate_pair(
                    gt_polylines,
                    pred_polylines,
                    image_id=variant["sample_id"],
                    support_scope=scope,
                    step_px=step,
                    max_samples=max_samples,
                    tolerances=tolerances,
                    duplicate_quantum=quantum,
                )
                per_scope[scope]["budget"] = budget
            except geometric_eval.EvaluationInputError as exc:
                per_scope[scope] = {
                    "status": "failed",
                    "status_reason": f"evaluation_input_error:{exc}",
                    "support_scope": scope,
                    "budget": budget,
                    "directions": {},
                }
        variant["scopes"] = per_scope

    # aggregate overall status across variants/scopes
    statuses = []
    for variant in variants:
        if variant.get("status") == "ok":
            scopes = variant.get("scopes") or {}
            scope_statuses = [doc.get("status") for doc in scopes.values()]
            if not scope_statuses:
                statuses.append("alignment_pending")
            elif any(item == "failed" for item in scope_statuses):
                statuses.append("failed")
            elif any(item == "evaluation_limited" for item in scope_statuses):
                statuses.append("evaluation_limited")
            elif any(item == "ok" for item in scope_statuses):
                statuses.append("ok")
            else:
                statuses.append("empty_pair")
        else:
            statuses.append(variant.get("status"))
    status = "ok"
    for candidate in ("failed", "excluded", "alignment_pending", "evaluation_limited", "empty_pair"):
        if candidate in statuses:
            status = candidate
            break
    if statuses and all(item == "ok" for item in statuses):
        status = "ok"
    timings["total_s"] = round(time.monotonic() - started, 3)
    if len(variants) == 1 and variants[0].get("status") == "ok":
        variants[0]["status"] = status
        variants[0]["status_reason"] = None
    return {
        "status": status,
        "frontend": frontend_result,
        "pred": pred_result,
        "variants": variants,
        "input": {"width": width, "height": height},
        "timings": timings,
    }


def _parse_svg_document(path: str, curve_steps: int) -> Dict[str, Any]:
    from .svg_parse import parse_svg

    return parse_svg(
        path,
        split="train_final",
        curve_steps=curve_steps,
        record_commands=True,
    )


def _run_worker(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Run one image task in a controlled subprocess and really stop on timeout."""

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tempdir = tempfile.mkdtemp(prefix="cadvec-runner-task-")
    payload_path = os.path.join(tempdir, "payload.json")
    result_path = os.path.join(tempdir, "result.json")
    with open(payload_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "cadvec.png_diagnostic_runner",
        "__worker__",
        payload_path,
        result_path,
    ]
    timeout = max(0.001, float(timeout_s))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            return {
                "status": "evaluation_limited",
                "status_reason": "image_timeout",
                "timeout_s": timeout_s,
                "worker_stderr_tail": stderr.decode("utf-8", "replace")[-400:] if stderr else "",
            }
        if not os.path.isfile(result_path):
            return {
                "status": "failed",
                "status_reason": "worker_exited_without_result",
                "exit_code": process.returncode,
                "worker_stderr_tail": stderr.decode("utf-8", "replace")[-800:] if stderr else "",
            }
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            return {
                "status": "failed",
                "status_reason": f"worker_result_unreadable:{type(exc).__name__}",
                "worker_stderr_tail": stderr.decode("utf-8", "replace")[-800:] if stderr else "",
            }
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            process.kill()
            process.communicate()
        shutil.rmtree(tempdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


def _overlay_kinds(value: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not value:
        return ("combined",)
    allowed = {"combined", "gt", "pred"}
    kinds = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not kinds or any(kind not in allowed for kind in kinds):
        raise RunnerError("overlay kinds must be combined, gt and/or pred")
    return kinds


def _draw_overlays(
    out_dir: str,
    items: Sequence[Dict[str, Any]],
    kinds: Sequence[str],
    limit: int,
    guard,
    writer: Optional[_Writer] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        import cv2
    except ImportError:
        return [], "image_dependency_missing"
    if limit <= 0:
        return [], "limit_zero"
    overlay_dir = os.path.join(out_dir, "overlays")
    targets = [overlay_dir]
    guard(targets)
    os.makedirs(overlay_dir, exist_ok=True)
    produced: List[Dict[str, Any]] = []
    for item in items:
        if len(produced) >= limit:
            break
        try:
            with open(item["png_path"], "rb") as handle:
                data = handle.read()
            import numpy as np

            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 - overlay is optional diagnostics
            continue
        if image is None:
            continue
        for kind in kinds:
            if len(produced) >= limit:
                break
            canvas = image.copy()
            if kind in ("combined", "gt"):
                for polyline in item.get("gt_polylines") or []:
                    points = _int_points(polyline)
                    if len(points) >= 2:
                        cv2.polylines(canvas, [points], False, (0, 190, 0), 1)
            if kind in ("combined", "pred"):
                for segment in item.get("pred_segments") or []:
                    cv2.line(
                        canvas,
                        (int(segment[0]), int(segment[1])),
                        (int(segment[2]), int(segment[3])),
                        (0, 0, 255),
                        1,
                    )
            cv2.putText(canvas, "GT" if kind == "gt" else ("PRED" if kind == "pred" else "GT/PRED"),
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
            ok, buffer = cv2.imencode(".png", canvas)
            if not ok:
                continue
            digest = hashlib.sha256(item["sample_id"].encode("utf-8")).hexdigest()[:16]
            name = f"ov-{digest}-{kind}.png"
            if writer is not None:
                writer.bytes(f"overlays/{name}", buffer.tobytes())
            else:  # pragma: no cover - defensive fallback
                path = os.path.join(overlay_dir, name)
                guard([path])
                with open(path, "wb") as handle:
                    handle.write(buffer.tobytes())
            produced.append(
                {
                    "file": f"overlays/{name}",
                    "sample_id": item["sample_id"],
                    "kind": kind,
                    "width": item["width"],
                    "height": item["height"],
                    "legend": {"gt": "green", "pred": "red", "background": "original PNG"},
                }
            )
    return produced, None


def _int_points(polyline: Sequence[Sequence[float]]) -> Any:
    import numpy as np

    points = []
    for point in polyline:
        x, y = float(point[0]), float(point[1])
        if math.isfinite(x) and math.isfinite(y):
            points.append([int(round(x)), int(round(y))])
    if not points:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(points, dtype=np.int32)


def preview_run(
    run_dir: str,
    out: str,
    *,
    max_preview: int = DEFAULT_MAX_PREVIEW,
    overlay_kinds: Optional[Sequence[str]] = None,
    max_overlays: int = DEFAULT_MAX_OVERLAYS,
) -> Tuple[Dict[str, Any], int]:
    started = time.monotonic()
    if max_preview <= 0:
        raise RunnerError("max_preview must be positive")
    if max_overlays < 0:
        raise RunnerError("max_overlays must be non-negative")
    loaded = _load_run(run_dir)
    frozen = loaded["frozen"]
    locked = loaded["locked"]
    manifest = loaded["manifest"]
    datasets_root = manifest.get("datasets_root")
    train_dir = manifest.get("train_dir")
    if not datasets_root or not train_dir:
        raise RunnerError("prepare manifest lacks dataset paths")
    dev_rows = _selected_rows_for(frozen, "development")
    if not dev_rows:
        raise RunnerError("frozen selection has no development samples")

    run_abs = loaded["run_dir"]
    out_abs = os.path.abspath(out)
    protected = [os.path.join(run_abs, name) for name in RUN_FILES if os.path.isfile(os.path.join(run_abs, name))]
    targets = [out_abs] + [os.path.join(out_abs, name) for name in PREVIEW_FILES]
    targets.append(os.path.join(out_abs, "overlays"))
    _ensure_run_dir(out, datasets_root, protected)
    out_abs = os.path.abspath(out)
    _guard(datasets_root, out_abs, targets, protected)
    guard = lambda extra: _guard(datasets_root, out_abs, list(extra), protected)
    writer = _Writer(out_abs, guard)

    if not png_frontend.image_dependencies_available():
        completion = _completion_doc("preview", "failed", "image_dependency_missing", loaded, False, False)
        writer.json("completion.json", completion)
        return completion, 2

    verified = _verify_files(dev_rows, train_dir)
    chosen: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for row in dev_rows:
        if len(chosen) >= max_preview:
            break
        if verified[row["sample_id"]]["status"] != "ok":
            continue
        key = row.get("png_digest") or row.get("png_name")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        chosen.append(row)

    preview_entries: Dict[str, Dict[str, Any]] = {}
    for row in dev_rows:
        if verified[row["sample_id"]]["status"] != "ok":
            preview_entries[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "status": "input_changed",
                "status_reason": "digest_or_file_drift",
                "verification": verified[row["sample_id"]],
            }
        else:
            preview_entries[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "status": "not_selected_for_preview",
                "status_reason": "preview_limit_or_group_not_chosen",
            }
    overlay_items: List[Dict[str, Any]] = []
    timings: Dict[str, Any] = {}
    for row in chosen:
        key = row.get("png_digest") or row.get("png_name")
        variants = [
            candidate
            for candidate in dev_rows
            if (candidate.get("png_digest") or candidate.get("png_name")) == key
        ]
        payload_variants = []
        for variant in variants:
            if verified[variant["sample_id"]]["status"] != "ok":
                preview_entries[variant["sample_id"]] = {
                    "sample_id": variant["sample_id"],
                    "status": "input_changed",
                    "status_reason": "digest_or_file_drift",
                    "verification": verified[variant["sample_id"]],
                }
                continue
            png_path, svg_path = _train_paths(variant, train_dir)
            payload_variants.append(
                {
                    "sample_id": variant["sample_id"],
                    "svg_path": svg_path,
                    "svg_digest": variant.get("svg_digest"),
                }
            )
        if not payload_variants:
            continue
        png_path, _ = _train_paths(row, train_dir)
        payload = {
            "mode": "preview",
            "frontend_config": locked["frontend"],
            "locked_config": locked,
            "png_path": png_path,
            "variants": payload_variants,
        }
        result = _run_worker(
            payload, float(locked["budgets"]["image_timeout_s"])
        )
        timings[row["sample_id"]] = result.get("timings", {})
        if result.get("status") != "ok":
            for variant in payload_variants:
                preview_entries[variant["sample_id"]] = {
                    "sample_id": variant["sample_id"],
                    "status": "failed",
                    "status_reason": result.get("status_reason") or result.get("error") or "preview_failed",
                }
            continue
        frontend = result.get("frontend") or {}
        width = (result.get("input") or {}).get("width")
        height = (result.get("input") or {}).get("height")
        for variant in result.get("variants", []):
            entry = {
                "sample_id": variant["sample_id"],
                "status": "previewed",
                "frontend_status": frontend.get("status"),
                "frontend_reason": frontend.get("status_reason"),
                "pred_line_count": frontend.get("line_count"),
                "gt_status": (variant.get("gt") or {}).get("status"),
                "mapping_status": (variant.get("mapping") or {}).get("status"),
                "mapping_reasons": (variant.get("mapping") or {}).get("reasons"),
                "alignment_verification": "unverified",
                "gt_counts": (variant.get("gt") or {}).get("counts"),
                "complexity": {
                    "pred_lines": frontend.get("line_count"),
                    "gt_polylines_all": len(
                        (variant.get("gt") or {}).get("polylines_all") or []
                    ),
                    "gt_polylines_straight": len(
                        (variant.get("gt") or {}).get("polylines_straight") or []
                    ),
                },
                "verification": verified.get(variant["sample_id"]),
                "width": width,
                "height": height,
            }
            if variant.get("status") != "ok":
                entry["status"] = "failed"
                entry["status_reason"] = variant.get("status_reason")
            preview_entries[variant["sample_id"]] = entry
            if variant.get("gt_pixels"):
                overlay_items.append(
                    {
                        "sample_id": variant["sample_id"],
                        "png_path": png_path,
                        "width": width,
                        "height": height,
                        "gt_polylines": variant["gt_pixels"].get("all_strokes", []),
                        "pred_segments": _pred_segments(frontend),
                    }
                )

    for row in dev_rows:
        if row["sample_id"] not in preview_entries:
            preview_entries[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "status": "not_selected_for_preview",
                "status_reason": "preview_limit_or_group_not_chosen",
            }

    overlays, overlay_skip = _draw_overlays(
        out_abs, overlay_items, _overlay_kinds(overlay_kinds), max_overlays, guard,
        writer=writer,
    )

    entries = [preview_entries[row["sample_id"]] for row in dev_rows]
    current_code = loaded.get("code_fingerprints") or _code_fingerprints()
    preview_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "preview",
        "generated_at": _now(),
        "run_dir": run_abs,
        "selection_fingerprint": frozen.get("fingerprint"),
        "selection_content_sha256": loaded.get("selection_content_sha256"),
        "locked_config_fingerprint": locked.get("fingerprint"),
        "max_preview": max_preview,
        "samples": entries,
        "overlays": overlays,
        "overlay_skipped": overlay_skip,
        "timings": timings,
        "versions": _versions(),
        "code_fingerprints": current_code,
        "notes": [
            "only the selected preview PNGs were predicted; other development "
            "samples are not_selected_for_preview",
            "mathematical mapping being ok does not verify real PNG/SVG alignment",
            "no coverage or distance metric is computed in preview",
        ],
    }
    preview_content_sha256 = _fingerprint(preview_payload)
    preview_payload["content_sha256"] = preview_content_sha256
    writer.json("preview.json", preview_payload)
    writer.csv(
        "preview_samples.csv",
        entries,
        (
            "sample_id",
            "status",
            "status_reason",
            "frontend_status",
            "gt_status",
            "mapping_status",
            "width",
            "height",
        ),
    )
    _write_conclusion(out_abs, writer, "preview", entries, overlays)
    failed = sum(1 for entry in entries if entry["status"] in ("failed", "input_changed"))
    complete = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "preview",
        "status": "complete" if failed == 0 else "partial",
        "generated_at": _now(),
        "selection_fingerprint": frozen.get("fingerprint"),
        "selection_content_sha256": loaded.get("selection_content_sha256"),
        "locked_config_fingerprint": locked.get("fingerprint"),
        "preview_content_sha256": preview_content_sha256,
        "counts": {
            "selected_development": len(entries),
            "previewed": sum(1 for entry in entries if entry["status"] == "previewed"),
            "not_selected_for_preview": sum(
                1 for entry in entries if entry["status"] == "not_selected_for_preview"
            ),
            "input_changed": sum(1 for entry in entries if entry["status"] == "input_changed"),
            "failed": sum(1 for entry in entries if entry["status"] == "failed"),
            "overlays": len(overlays),
            "unique_png_count": _unique_png_count(dev_rows),
            "preview_predictions": len(timings),
        },
        "deliverable_complete": True,
        "metrics_evaluable": False,
        "elapsed_s": round(_clock() - started, 3),
        "versions": _versions(),
        "code_fingerprints": current_code,
    }
    writer.json("completion.json", complete)
    return complete, 0 if failed == 0 else 1


def _pred_segments(frontend: Dict[str, Any]) -> List[List[float]]:
    result = []
    for line in frontend.get("lines") or []:
        result.append([line.get("x1"), line.get("y1"), line.get("x2"), line.get("y2")])
    return result


def _completion_doc(
    command: str,
    status: str,
    reason: str,
    loaded: Dict[str, Any],
    deliverable: bool,
    metrics: bool,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": status,
        "status_reason": reason,
        "generated_at": _now(),
        "selection_fingerprint": loaded["frozen"].get("fingerprint"),
        "locked_config_fingerprint": loaded["locked"].get("fingerprint"),
        "deliverable_complete": deliverable,
        "metrics_evaluable": metrics,
        "versions": _versions(),
        "code_fingerprints": _code_fingerprints(),
    }


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _load_alignment_review(
    path: str,
    *,
    frozen: Dict[str, Any],
    locked: Dict[str, Any],
    dev_rows: Sequence[Dict[str, Any]],
    verified: Dict[str, Dict[str, str]],
    preview_dir: str,
    code_fingerprints: Dict[str, str],
) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        review = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunnerError(f"alignment review is not valid UTF-8 JSON: {exc}")
    if not isinstance(review, dict):
        raise RunnerError("alignment review must be a JSON object")
    if review.get("schema") != ALIGNMENT_REVIEW_SCHEMA:
        raise RunnerError("alignment review schema mismatch")
    if review.get("schema_version") != ALIGNMENT_REVIEW_VERSION:
        raise RunnerError("alignment review schema_version mismatch")
    if review.get("selection_fingerprint") != frozen.get("fingerprint"):
        raise RunnerError("alignment review selection fingerprint mismatch")
    if review.get("locked_config_fingerprint") != locked.get("fingerprint"):
        raise RunnerError("alignment review locked config fingerprint mismatch")
    if review.get("viewport_mode") != DEFAULT_VIEWPORT_MODE:
        raise RunnerError("alignment review viewport_mode must be full_svg_viewport")
    if review.get("decision") != "development_reviewed_assumption":
        raise RunnerError("alignment review decision must be development_reviewed_assumption")
    for field in ("reviewer", "evidence"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise RunnerError(f"alignment review needs a non-empty {field}")
    reviewed = review.get("reviewed_samples")
    if not isinstance(reviewed, list) or not reviewed:
        raise RunnerError("alignment review needs at least one reviewed sample")

    preview_path = os.path.join(preview_dir, "preview.json")
    preview = _read_json(preview_path, "preview output")
    preview_completion_path = os.path.join(preview_dir, "completion.json")
    preview_completion = _read_json(preview_completion_path, "preview completion")
    if preview.get("schema") != SCHEMA or preview.get("schema_version") != SCHEMA_VERSION:
        raise RunnerError("preview schema/version mismatch")
    if preview.get("command") != "preview":
        raise RunnerError("preview document is not a preview command result")
    if preview_completion.get("command") != "preview":
        raise RunnerError("preview completion is not a preview completion")
    if preview_completion.get("deliverable_complete") is not True:
        raise RunnerError("preview did not complete its deliverables")
    if preview_completion.get("status") not in ("complete", "partial"):
        raise RunnerError("preview completion status is unknown")
    if preview.get("selection_fingerprint") != frozen.get("fingerprint"):
        raise RunnerError("preview does not belong to this frozen selection")
    if preview.get("locked_config_fingerprint") != locked.get("fingerprint"):
        raise RunnerError("preview was produced with a different locked config")
    if preview.get("code_fingerprints") != code_fingerprints:
        raise RunnerError("preview was produced with different source code")
    stored_preview_hash = preview.get("content_sha256")
    preview_without_hash = {k: v for k, v in preview.items() if k != "content_sha256"}
    preview_content_sha256 = _fingerprint(preview_without_hash)
    if stored_preview_hash != preview_content_sha256:
        raise RunnerError("preview content digest mismatch")
    if preview_completion.get("preview_content_sha256") != preview_content_sha256:
        raise RunnerError("preview completion content digest mismatch")
    if preview_completion.get("locked_config_fingerprint") != locked.get("fingerprint"):
        raise RunnerError("preview completion locked config mismatch")
    if preview_completion.get("selection_fingerprint") != frozen.get("fingerprint"):
        raise RunnerError("preview completion selection mismatch")
    if preview_completion.get("code_fingerprints") != code_fingerprints:
        raise RunnerError("preview completion code identity mismatch")

    preview_samples = {
        entry.get("sample_id"): entry for entry in preview.get("samples") or []
    }
    dev_by_id = {row["sample_id"]: row for row in dev_rows}
    accepted = []
    for item in reviewed:
        if not isinstance(item, dict):
            raise RunnerError("reviewed_samples entries must be objects")
        sample_id = item.get("sample_id")
        if sample_id not in dev_by_id:
            raise RunnerError(
                f"reviewed sample {sample_id!r} is not a frozen development sample"
            )
        row = dev_by_id[sample_id]
        if item.get("png_digest") != row.get("png_digest") or item.get("svg_digest") != row.get("svg_digest"):
            raise RunnerError(f"reviewed digests do not match the frozen row for {sample_id!r}")
        verification = verified.get(sample_id) or {}
        if verification.get("status") != "ok":
            raise RunnerError(f"reviewed sample {sample_id!r} has changed or unreadable files")
        if (
            verification.get("png_digest") != row.get("png_digest")
            or verification.get("svg_digest") != row.get("svg_digest")
        ):
            raise RunnerError(f"reviewed sample {sample_id!r} current digests differ")
        entry = preview_samples.get(sample_id)
        if not entry or entry.get("status") != "previewed":
            raise RunnerError(
                f"reviewed sample {sample_id!r} was not actually previewed"
            )
        entry_verification = entry.get("verification") or {}
        if entry_verification.get("status") != "ok":
            raise RunnerError(
                f"preview for {sample_id!r} did not verify the sample files"
            )
        if (
            entry_verification.get("png_digest") != row.get("png_digest")
            or entry_verification.get("svg_digest") != row.get("svg_digest")
        ):
            raise RunnerError(
                f"preview evidence digests do not match the frozen row for {sample_id!r}"
            )
        if entry.get("gt_status") != "ok" or entry.get("mapping_status") != "ok":
            raise RunnerError(
                f"reviewed sample {sample_id!r} does not have a usable GT/mapping assumption"
            )
        if entry.get("frontend_status") not in (
            "ok",
            "empty_input",
            "no_lines_detected",
        ):
            raise RunnerError(
                f"reviewed sample {sample_id!r} frontend status is not preview-usable"
            )
        accepted.append(sample_id)
    return {
        "review": review,
        "review_raw": raw,
        "review_file_sha256": _sha256_bytes(raw),
        "review_content_sha256": _fingerprint(review),
        "preview_content_sha256": preview_content_sha256,
        "accepted_sample_ids": accepted,
    }


def _locked_budgets(locked: Dict[str, Any]) -> Tuple[int, float, float]:
    budgets = locked.get("budgets") or {}
    compute = budgets.get("compute_budget_per_scope")
    image = budgets.get("image_timeout_s")
    run = budgets.get("run_budget_s")
    if isinstance(compute, bool) or not isinstance(compute, int) or compute <= 0:
        raise RunnerError("content_changed: locked compute budget is invalid")
    for name, value in (("image_timeout_s", image), ("run_budget_s", run)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise RunnerError(f"content_changed: locked {name} is invalid")
    return compute, float(image), float(run)


def _resolve_budget(
    name: str, provided: Any, locked_value: Any
) -> Any:
    if provided is None:
        return locked_value
    if isinstance(provided, bool) or not isinstance(provided, (int, float)):
        raise RunnerError(f"{name} must be a number or omitted")
    if isinstance(locked_value, int):
        if isinstance(provided, bool) or not isinstance(provided, int):
            raise RunnerError(f"{name} must match the prepare-locked integer budget")
        if provided != locked_value:
            raise RunnerError(
                f"{name}={provided!r} differs from the prepare-locked value "
                f"{locked_value!r}; create a new prepare run with the new budget"
            )
        return provided
    if not math.isfinite(float(provided)) or float(provided) != float(locked_value):
        raise RunnerError(
            f"{name}={provided!r} differs from the prepare-locked value "
            f"{locked_value!r}; create a new prepare run with the new budget"
        )
    return float(provided)


def evaluate_run(
    run_dir: str,
    out: str,
    *,
    subset: str = "development",
    alignment_review: Optional[str] = None,
    preview_dir: Optional[str] = None,
    compute_budget: Optional[int] = None,
    image_timeout: Optional[float] = None,
    run_budget: Optional[float] = None,
    overlay_kinds: Optional[Sequence[str]] = None,
    max_overlays: int = DEFAULT_MAX_OVERLAYS,
) -> Tuple[Dict[str, Any], int]:
    started = _clock()
    if subset not in ("development", "holdout", "all"):
        raise RunnerError(f"unknown subset: {subset!r}")
    if max_overlays < 0:
        raise RunnerError("max_overlays must be non-negative")

    loaded = _load_run(run_dir)
    frozen = loaded["frozen"]
    locked = loaded["locked"]
    manifest = loaded["manifest"]
    locked_compute, locked_image_timeout, locked_run_budget = _locked_budgets(locked)
    compute_budget = _resolve_budget(
        "compute_budget", compute_budget, locked_compute
    )
    image_timeout = _resolve_budget(
        "image_timeout", image_timeout, locked_image_timeout
    )
    run_budget = _resolve_budget("run_budget", run_budget, locked_run_budget)
    deadline = started + float(run_budget)
    datasets_root = manifest.get("datasets_root")
    train_dir = manifest.get("train_dir")
    if not datasets_root or not train_dir:
        raise RunnerError("prepare manifest lacks dataset paths")

    rows = _selected_rows_for(frozen, subset)
    dev_rows = _selected_rows_for(frozen, "development")
    if not rows:
        raise RunnerError(f"frozen selection has no samples for subset {subset!r}")

    run_abs = loaded["run_dir"]
    preview_dir = os.path.abspath(preview_dir or os.path.join(run_abs, "preview"))
    protected = [
        os.path.join(run_abs, name)
        for name in RUN_FILES
        if os.path.isfile(os.path.join(run_abs, name))
    ]
    if os.path.isdir(preview_dir):
        protected.extend(
            os.path.join(preview_dir, name)
            for name in PREVIEW_FILES
            if os.path.isfile(os.path.join(preview_dir, name))
        )
    if alignment_review:
        protected.append(os.path.abspath(alignment_review))

    targets = [os.path.abspath(out)] + [
        os.path.join(os.path.abspath(out), name) for name in EVALUATE_FILES
    ]
    targets.append(os.path.join(os.path.abspath(out), "overlays"))
    targets.append(os.path.join(os.path.abspath(out), "images"))
    if alignment_review:
        targets.append(os.path.join(os.path.abspath(out), "alignment_review.json"))
    _ensure_run_dir(out, datasets_root, protected)
    out_abs = os.path.abspath(out)
    # image files are named after a stable hash in a second step
    _guard(datasets_root, out_abs, targets, protected)
    guard = lambda extra: _guard(datasets_root, out_abs, list(extra), protected)
    writer = _Writer(out_abs, guard)

    if not png_frontend.image_dependencies_available():
        completion = _completion_doc("evaluate", "failed", "image_dependency_missing", loaded, False, False)
        writer.json("completion.json", completion)
        return completion, 2

    verified = _verify_files(rows, train_dir)
    dev_verified = _verify_files(dev_rows, train_dir) if subset != "development" else verified
    review_info = None
    review_valid = False
    if alignment_review:
        review_info = _load_alignment_review(
            alignment_review,
            frozen=frozen,
            locked=locked,
            dev_rows=dev_rows,
            verified=dev_verified,
            preview_dir=preview_dir,
            code_fingerprints=loaded["code_fingerprints"],
        )
        review_valid = True
        writer.bytes("alignment_review.json", review_info["review_raw"])

    image_docs: List[Dict[str, Any]] = []
    overlay_items: List[Dict[str, Any]] = []
    timings: Dict[str, Any] = {}
    review_applied = bool(review_info)
    review_scope = (
        "development_assumption_applied_to_selected_splits"
        if review_applied
        else "unverified"
    )
    review_evidence_split = "development" if review_applied else None
    review_accepted_ids = (
        review_info["accepted_sample_ids"] if review_info else []
    )

    for row in rows:
        sample_id = row["sample_id"]
        verification = verified[sample_id]
        split = row.get("_split") or "development"
        doc: Dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "command": "evaluate",
            "sample_id": sample_id,
            "group_id": row.get("_group_id"),
            "split": split,
            "subset": subset,
            "verification": verification,
            "alignment_review": {
                "applied": review_applied,
                "review_scope": review_scope,
                "evidence_split": review_evidence_split,
                "applied_to_split": split if review_applied else None,
                "accepted_sample_ids": review_accepted_ids,
                "mapping_verification": (
                    "development_reviewed_assumption" if review_applied else "unverified"
                ),
            },
            "input": {
                "png": row.get("png_name"),
                "svg": row.get("svg_name"),
                "png_digest": row.get("png_digest"),
                "svg_digest": row.get("svg_digest"),
            },
        }
        if verification["status"] != "ok":
            doc.update(
                {
                    "status": "input_changed",
                    "status_reason": "frozen_digest_mismatch_or_unreadable",
                    "frontend": None,
                    "gt": None,
                    "mapping": None,
                    "scopes": {},
                }
            )
            image_docs.append(doc)
            continue
        remaining_budget = deadline - _clock()
        if remaining_budget <= 0.0:
            doc.update(
                {
                    "status": "evaluation_limited",
                    "status_reason": "overall_time_budget_exhausted",
                    "frontend": None,
                    "gt": None,
                    "mapping": None,
                    "scopes": {},
                }
            )
            image_docs.append(doc)
            continue
        timeout = min(float(image_timeout), max(0.001, remaining_budget))
        png_path, svg_path = _train_paths(row, train_dir)
        payload = {
            "mode": "evaluate",
            "frontend_config": locked["frontend"],
            "locked_config": locked,
            "png_path": png_path,
            "variants": [
                {"sample_id": sample_id, "svg_path": svg_path, "svg_digest": row.get("svg_digest")}
            ],
            "review_applies": bool(review_valid),
        }
        task_started = time.monotonic()
        result = _run_worker(payload, timeout)
        timings[sample_id] = {
            "wall_s": round(time.monotonic() - task_started, 3),
            **(result.get("timings") or {}),
        }
        if result.get("status") in ("evaluation_limited",) and result.get("status_reason") == "image_timeout":
            doc.update(
                {
                    "status": "evaluation_limited",
                    "status_reason": "image_timeout",
                    "timeout_s": timeout,
                    "frontend": None,
                    "gt": None,
                    "mapping": None,
                    "scopes": {},
                }
            )
            image_docs.append(doc)
            continue
        if result.get("status") in ("failed", "evaluation_limited") and not result.get("variants"):
            doc.update(
                {
                    "status": result.get("status"),
                    "status_reason": result.get("status_reason") or result.get("error"),
                    "frontend": result.get("frontend"),
                    "gt": None,
                    "mapping": None,
                    "scopes": {},
                }
            )
            image_docs.append(doc)
            continue
        variant = (result.get("variants") or [{}])[0]
        frontend = result.get("frontend") or {}
        pred = result.get("pred") or {}
        doc.update(
            {
                "status": variant.get("status") or result.get("status") or "failed",
                "status_reason": variant.get("status_reason") or result.get("status_reason"),
                "frontend": {
                    "status": frontend.get("status"),
                    "status_reason": frontend.get("status_reason"),
                    "line_count": frontend.get("line_count"),
                    "lines": frontend.get("lines") or [],
                    "pred_adapter": {
                        "status": pred.get("status"),
                        "reasons": pred.get("reasons"),
                    },
                },
                "gt": {
                    "status": (variant.get("gt") or {}).get("status"),
                    "reasons": (variant.get("gt") or {}).get("reasons"),
                    "complete": (variant.get("gt") or {}).get("complete"),
                    "counts": (variant.get("gt") or {}).get("counts"),
                    "approximation": (variant.get("gt") or {}).get("approximation"),
                    "excluded": (variant.get("gt") or {}).get("excluded"),
                    "gt_stats": variant.get("gt_stats"),
                },
                "mapping": variant.get("mapping"),
                "scopes": variant.get("scopes") or {},
            }
        )
        image_docs.append(doc)
        if variant.get("gt_pixels"):
            overlay_items.append(
                {
                    "sample_id": sample_id,
                    "png_path": png_path,
                    "width": (result.get("input") or {}).get("width"),
                    "height": (result.get("input") or {}).get("height"),
                    "gt_polylines": variant["gt_pixels"].get("all_strokes", []),
                    "pred_segments": _pred_segments(frontend),
                }
            )

    budget_overrun = _clock() >= deadline
    if budget_overrun:
        overlays, overlay_skip = [], "overall_time_budget_exhausted"
    else:
        overlays, overlay_skip = _draw_overlays(
            out_abs, overlay_items, _overlay_kinds(overlay_kinds), max_overlays, guard,
            writer=writer,
        )
        if _clock() >= deadline:
            budget_overrun = True
            overlay_skip = overlay_skip or "overall_time_budget_exhausted"

    images_dir = os.path.join(out_abs, "images")
    guard([images_dir])
    os.makedirs(images_dir, exist_ok=True)
    hashed_names: Dict[str, str] = {}
    for doc in image_docs:
        digest = hashlib.sha256(doc["sample_id"].encode("utf-8")).hexdigest()[:16]
        name = f"img-{digest}.json"
        if name in hashed_names and hashed_names[name] != doc["sample_id"]:
            raise RunnerError("sample_id hash collision in per-image file names")
        hashed_names[name] = doc["sample_id"]
        doc["image_file"] = f"images/{name}"
        writer.json(os.path.join("images", name), doc)

    status_rows = [
        {
            "sample_id": doc["sample_id"],
            "split": doc["split"],
            "status": doc["status"],
            "status_reason": doc.get("status_reason"),
            "frontend_status": (doc.get("frontend") or {}).get("status"),
            "gt_status": (doc.get("gt") or {}).get("status"),
            "mapping_status": (doc.get("mapping") or {}).get("status"),
            "image_file": doc.get("image_file"),
        }
        for doc in image_docs
    ]
    writer.csv(
        "statuses.csv",
        status_rows,
        (
            "sample_id",
            "split",
            "status",
            "status_reason",
            "frontend_status",
            "gt_status",
            "mapping_status",
            "image_file",
        ),
    )

    summaries: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    for split_name in ("development", "holdout"):
        split_docs = [doc for doc in image_docs if doc["split"] == split_name]
        if not split_docs:
            continue
        for scope in ("all_strokes", "straight_only"):
            records = []
            for doc in split_docs:
                scope_doc = (doc.get("scopes") or {}).get(scope)
                if doc["status"] in ("ok", "empty_pair", "evaluation_limited", "failed"):
                    if scope_doc and scope_doc.get("status") in (
                        "ok", "empty_pair", "evaluation_limited", "failed"
                    ):
                        records.append(scope_doc)
                        continue
                records.append(
                    {
                        "status": _summary_status(doc["status"]),
                        "status_reason": doc.get("status_reason"),
                    }
                )
            summary = geometric_eval.summarise(records)
            summaries.setdefault(split_name, {})[scope] = summary
            gt_direction = summary["directions"]["gt_to_pred"]
            pred_direction = summary["directions"]["pred_to_gt"]
            row = {
                "subset": subset,
                "split": split_name,
                "scope": scope,
                "selected": summary["counts"]["selected"],
                "eligible": summary["counts"]["eligible"],
                "excluded": summary["counts"]["excluded"],
                "failed": summary["counts"]["failed"],
                "alignment_pending": summary["counts"]["alignment_pending"],
                "evaluation_limited": summary["counts"]["evaluation_limited"],
                "empty_pair": summary["counts"]["empty_pair"],
            }
            for tolerance in ("1", "2", "4"):
                row[f"gt_coverage_macro_{tolerance}"] = _coverage_value(
                    gt_direction, "coverage_macro", tolerance
                )
                row[f"gt_coverage_micro_{tolerance}"] = _coverage_value(
                    gt_direction, "coverage_micro", tolerance
                )
                row[f"pred_coverage_macro_{tolerance}"] = _coverage_value(
                    pred_direction, "coverage_macro", tolerance
                )
                row[f"pred_coverage_micro_{tolerance}"] = _coverage_value(
                    pred_direction, "coverage_micro", tolerance
                )
            summary_rows.append(row)

    summary_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "evaluate",
        "generated_at": _now(),
        "subset": subset,
        "selection_fingerprint": frozen.get("fingerprint"),
        "locked_config_fingerprint": locked.get("fingerprint"),
        "alignment_review": (
            {
                "used": review_valid,
                "review_scope": review_scope,
                "evidence_split": review_evidence_split,
                "applied_to_splits": sorted({doc["split"] for doc in image_docs})
                if review_valid
                else [],
                "accepted_sample_ids": review_accepted_ids,
                "decision": (review_info["review"].get("decision") if review_info else None),
                "reviewer": (review_info["review"].get("reviewer") if review_info else None),
                "evidence": (review_info["review"].get("evidence") if review_info else None),
                "review_file": "alignment_review.json" if review_info else None,
                "review_file_sha256": (
                    review_info["review_file_sha256"] if review_info else None
                ),
                "review_content_sha256": (
                    review_info["review_content_sha256"] if review_info else None
                ),
                "preview_content_sha256": (
                    review_info["preview_content_sha256"] if review_info else None
                ),
                "verification": "development_reviewed_assumption" if review_valid else "unverified",
            }
        ),
        "summaries": summaries,
        "run_budget": {
            "run_budget_s": run_budget,
            "deadline_exceeded": budget_overrun,
        },
        "notes": [
            "scopes and development/holdout are summarised separately; no mixing",
            "empty predictions keep zero recall in macro and micro coverage",
            "input_changed and timeout records enter the summary as failed/limited",
            "no global median/P95 is synthesised; only core summary values are copied",
            "a valid development review applies the locked full_svg_viewport assumption "
            "to the selected splits; it is not per-image alignment proof",
        ],
    }
    writer.json("summary.json", summary_payload)
    writer.csv(
        "summary.csv",
        summary_rows,
        (
            "subset",
            "split",
            "scope",
            "selected",
            "eligible",
            "excluded",
            "failed",
            "alignment_pending",
            "evaluation_limited",
            "empty_pair",
            "gt_coverage_macro_1",
            "gt_coverage_micro_1",
            "gt_coverage_macro_2",
            "gt_coverage_micro_2",
            "gt_coverage_macro_4",
            "gt_coverage_micro_4",
            "pred_coverage_macro_1",
            "pred_coverage_micro_1",
            "pred_coverage_macro_2",
            "pred_coverage_micro_2",
            "pred_coverage_macro_4",
            "pred_coverage_micro_4",
        ),
    )

    counts = {
        "selected": len(image_docs),
        "ok": sum(1 for doc in image_docs if doc["status"] == "ok"),
        "empty_pair": sum(1 for doc in image_docs if doc["status"] == "empty_pair"),
        "evaluation_limited": sum(1 for doc in image_docs if doc["status"] == "evaluation_limited"),
        "failed": sum(1 for doc in image_docs if doc["status"] == "failed"),
        "excluded": sum(1 for doc in image_docs if doc["status"] == "excluded"),
        "alignment_pending": sum(1 for doc in image_docs if doc["status"] == "alignment_pending"),
        "input_changed": sum(1 for doc in image_docs if doc["status"] == "input_changed"),
        "development": sum(1 for doc in image_docs if doc["split"] == "development"),
        "holdout": sum(1 for doc in image_docs if doc["split"] == "holdout"),
        "unique_png_count": _unique_png_count(rows),
        "variant_count": len(rows),
        "worker_predictions": len(timings),
    }
    if _clock() >= deadline:
        budget_overrun = True
    _write_conclusion(out_abs, writer, "evaluate", image_docs, overlays, summaries)
    manifest_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "evaluate",
        "generated_at": _now(),
        "subset": subset,
        "selection_fingerprint": frozen.get("fingerprint"),
        "selection_content_sha256": loaded.get("selection_content_sha256"),
        "locked_config_fingerprint": locked.get("fingerprint"),
        "compute_budget": compute_budget,
        "image_timeout_s": image_timeout,
        "run_budget_s": run_budget,
        "budget_source": "prepare_locked",
        "deadline_exceeded": budget_overrun,
        "prediction_boundary": {
            "evaluate": "one worker/prediction per annotation variant row",
            "preview": "one prediction per unique PNG shared by its variants",
            "worker_tasks": len(timings),
            "unique_png_count": _unique_png_count(rows),
            "variant_count": len(rows),
        },
        "overlays": overlays,
        "overlay_skipped": overlay_skip,
        "alignment_review": (
            {
                "used": review_valid,
                "review_scope": review_scope,
                "evidence_split": review_evidence_split,
                "applied_to_splits": sorted({doc["split"] for doc in image_docs})
                if review_valid
                else [],
                "accepted_sample_ids": review_accepted_ids,
                "reviewer": (review_info["review"].get("reviewer") if review_info else None),
                "evidence": (review_info["review"].get("evidence") if review_info else None),
                "decision": (review_info["review"].get("decision") if review_info else None),
                "review_file": "alignment_review.json" if review_info else None,
                "review_file_sha256": (
                    review_info["review_file_sha256"] if review_info else None
                ),
                "review_content_sha256": (
                    review_info["review_content_sha256"] if review_info else None
                ),
                "preview_content_sha256": (
                    review_info["preview_content_sha256"] if review_info else None
                ),
            }
        ),
        "versions": _versions(),
        "code_fingerprints": loaded["code_fingerprints"],
        "timings": timings,
        "notes": [
            "compute budget is a conservative upper bound and may trigger early "
            "evaluation_limited",
            "wall-clock timeouts terminate the worker process; no truncated result "
            "is reported as complete",
            "per-image files are named by a stable sample_id hash, never by a "
            "manifest source path",
            "the run deadline includes input verification, workers, overlays and "
            "necessary summary/completion writes; writes after the deadline are "
            "limited cleanup, not a hard real-time guarantee",
        ],
    }
    writer.json("evaluate_manifest.json", manifest_payload)
    if _clock() >= deadline:
        budget_overrun = True
    if budget_overrun and not summary_payload["run_budget"]["deadline_exceeded"]:
        summary_payload["run_budget"]["deadline_exceeded"] = True
        writer.json("summary.json", summary_payload)
    if budget_overrun and manifest_payload.get("deadline_exceeded") is not True:
        manifest_payload["deadline_exceeded"] = True
        writer.json("evaluate_manifest.json", manifest_payload)

    deliverable_complete = True
    timeout_count = sum(
        1
        for doc in image_docs
        if doc["status"] == "evaluation_limited"
        and doc.get("status_reason") in ("image_timeout", "overall_time_budget_exhausted")
    )
    metrics_evaluable = (
        not budget_overrun
        and counts["ok"] > 0
        and counts["failed"] == 0
        and counts["input_changed"] == 0
        and counts["alignment_pending"] == 0
        and counts["evaluation_limited"] == 0
        and counts["excluded"] == 0
    )
    failed_any = counts["failed"] + counts["input_changed"]
    completion_status = (
        "complete"
        if failed_any == 0 and timeout_count == 0 and not budget_overrun
        else "partial"
    )
    completion = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "evaluate",
        "status": completion_status,
        "status_reason": (
            "overall_time_budget_exhausted" if budget_overrun else None
        ),
        "generated_at": _now(),
        "subset": subset,
        "selection_fingerprint": frozen.get("fingerprint"),
        "selection_content_sha256": loaded.get("selection_content_sha256"),
        "locked_config_fingerprint": locked.get("fingerprint"),
        "counts": counts,
        "summaries_written": sorted(summaries),
        "deliverable_complete": deliverable_complete,
        "metrics_evaluable": metrics_evaluable,
        "timeouts": timeout_count,
        "run_budget_exhausted": budget_overrun,
        "elapsed_s": round(_clock() - started, 3),
        "versions": _versions(),
        "code_fingerprints": loaded["code_fingerprints"],
    }
    writer.json("completion.json", completion)
    if _clock() >= deadline:
        budget_overrun = True
        completion["status"] = "partial"
        completion["status_reason"] = "overall_time_budget_exhausted"
        completion["run_budget_exhausted"] = True
        completion["metrics_evaluable"] = False
        completion["elapsed_s"] = round(_clock() - started, 3)
        if not summary_payload["run_budget"]["deadline_exceeded"]:
            summary_payload["run_budget"]["deadline_exceeded"] = True
            writer.json("summary.json", summary_payload)
        if manifest_payload.get("deadline_exceeded") is not True:
            manifest_payload["deadline_exceeded"] = True
            writer.json("evaluate_manifest.json", manifest_payload)
        writer.json("completion.json", completion)
    return (
        completion,
        0
        if failed_any == 0
        and timeout_count == 0
        and not budget_overrun
        else 1,
    )


def _summary_status(status: str) -> str:
    if status in ("ok", "empty_pair", "evaluation_limited", "failed"):
        return status
    if status == "excluded":
        return "excluded"
    if status == "alignment_pending":
        return "alignment_pending"
    return "failed"


def _coverage_value(direction: Dict[str, Any], field: str, tolerance: str) -> Optional[float]:
    values = direction.get(field) or {}
    value = values.get(tolerance)
    return None if value is None else round(float(value), 6)


def _write_conclusion(
    out_dir: str,
    writer: _Writer,
    command: str,
    entries: Sequence[Dict[str, Any]],
    overlays: Sequence[Dict[str, Any]],
    summaries: Optional[Dict[str, Any]] = None,
) -> None:
    counts: Dict[str, int] = {}
    for entry in entries:
        status = entry.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    lines = [f"# {command} conclusion", "", "Synthetic/diagnostic run; not a real-data result.", ""]
    lines.append("Status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if summaries:
        lines.append("")
        lines.append("Summaries are split by development/holdout and scope; pending "
                     "or limited items stay in their denominators.")
    if overlays:
        lines.append("")
        lines.append(f"Overlays: {len(overlays)} (green GT, red prediction, original PNG background).")
    lines.append("")
    lines.append("No semantic/instance labels or platform IDs are produced; no PQ is computed.")
    writer.bytes("conclusion.md", ("\n".join(lines) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.png_diagnostic_runner",
        description="Prepare frozen PNG diagnostic samples, preview them, and evaluate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="train-only inventory + frozen selection")
    prepare.add_argument("--datasets", required=True)
    prepare.add_argument("--manifest", default=None)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--development-fraction", type=float, default=DEFAULT_DEVELOPMENT_FRACTION)
    prepare.add_argument("--compute-budget", type=int, default=DEFAULT_COMPUTE_BUDGET)
    prepare.add_argument("--image-timeout", type=float, default=DEFAULT_IMAGE_TIMEOUT_S)
    prepare.add_argument("--run-budget", type=float, default=DEFAULT_RUN_BUDGET_S)

    preview = subparsers.add_parser("preview", help="predict a few development PNGs and draw overlays")
    preview.add_argument("--run-dir", required=True)
    preview.add_argument("--out", required=True)
    preview.add_argument("--max-preview", type=int, default=DEFAULT_MAX_PREVIEW)
    preview.add_argument("--max-overlays", type=int, default=DEFAULT_MAX_OVERLAYS)
    preview.add_argument("--overlay-kinds", default="combined")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen subset")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--subset", default="development", choices=("development", "holdout", "all"))
    evaluate.add_argument("--alignment-review", default=None)
    evaluate.add_argument("--preview-dir", default=None)
    evaluate.add_argument("--compute-budget", type=int, default=None)
    evaluate.add_argument("--image-timeout", type=float, default=None)
    evaluate.add_argument("--run-budget", type=float, default=None)
    evaluate.add_argument("--max-overlays", type=int, default=DEFAULT_MAX_OVERLAYS)
    evaluate.add_argument("--overlay-kinds", default="combined")

    worker = subparsers.add_parser("__worker__", help=argparse.SUPPRESS)
    worker.add_argument("payload")
    worker.add_argument("result")
    return parser


def _overlay_kinds_arg(value: str) -> Tuple[str, ...]:
    return _overlay_kinds([part.strip() for part in value.split(",") if part.strip()])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "__worker__":
        _worker_file(args.payload, args.result)
        return 0
    try:
        if args.command == "prepare":
            completion, code = prepare_run(
                args.datasets,
                args.out,
                manifest=args.manifest,
                sample_count=args.sample_count,
                seed=args.seed,
                development_fraction=args.development_fraction,
                compute_budget=args.compute_budget,
                image_timeout=args.image_timeout,
                run_budget=args.run_budget,
            )
        elif args.command == "preview":
            completion, code = preview_run(
                args.run_dir,
                args.out,
                max_preview=args.max_preview,
                overlay_kinds=_overlay_kinds_arg(args.overlay_kinds),
                max_overlays=args.max_overlays,
            )
        else:
            completion, code = evaluate_run(
                args.run_dir,
                args.out,
                subset=args.subset,
                alignment_review=args.alignment_review,
                preview_dir=args.preview_dir,
                compute_budget=args.compute_budget,
                image_timeout=args.image_timeout,
                run_budget=args.run_budget,
                overlay_kinds=_overlay_kinds_arg(args.overlay_kinds),
                max_overlays=args.max_overlays,
            )
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "command": completion.get("command"),
                "status": completion.get("status"),
                "counts": completion.get("counts") or completion.get("train_counts"),
                "selection_fingerprint": completion.get("selection_fingerprint"),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
