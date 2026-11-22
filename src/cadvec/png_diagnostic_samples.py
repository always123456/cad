"""Sample metadata, exact-duplicate grouping and frozen selection.

This is the pure-stdlib metadata lane that sits between the inventory output and
any future evaluation runner.  It deliberately:

* reads only the CSV files it is given (no directory listing, no PNG/SVG bytes,
  no manifest-driven file opening, no test_final/dev_final annotations);
* never imports NumPy/OpenCV/Pillow and never calls the PNG predictor, the SVG
  parser or the geometric evaluator;
* treats inventory digests as **metadata only**: they are copied from the input
  CSV and marked ``metadata_only_unverified`` rather than re-hashed here;
* keeps every annotation variant of an exactly repeated image instead of
  choosing a representative.

Three public functions are frozen for this stage:

``load_sample_rows(files_csv, manifest_csv=None, *, split="train_final")``
    Build paired train sample rows with explicit issues and accounting.

``group_exact_duplicates(rows, *, use_source_proxy=False)``
    Transitive closure over equal PNG/SVG SHA-256 digests, with optional,
    explicitly unproven ``source`` proxy edges.

``freeze_selection(grouped, *, sample_count=24, seed=20261108,
development_fraction=0.5)``
    Deterministically assign whole duplicate groups to development/holdout and
    select a finite, reproducible list.

Output schema: ``cadvec.png_diagnostic_samples`` version ``1``.  All results are
strict-JSON safe and the inputs are never mutated.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import policy

SCHEMA = "cadvec.png_diagnostic_samples"
SCHEMA_VERSION = 1
ALLOWED_SPLIT = "train_final"

FILES_COLUMNS = (
    "split",
    "name",
    "role",
    "extension",
    "stem",
    "size_bytes",
    "digest",
)
MANIFEST_COLUMNS = (
    "dest_stem",
    "source",
    "stem",
    "size_class",
    "png_bytes",
    "png_dimensions",
    "svg",
    "png",
)

VERIFICATION = "metadata_only_unverified"

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DIMENSIONS_RE = re.compile(r"^\s*(\d+)\s*[xX\u00d7]\s*(\d+)\s*$")


class SampleTableError(ValueError):
    """Raised when an input CSV/configuration cannot be used safely."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_single_filename(name: Any) -> bool:
    """True only for a plain file name that cannot traverse or escape."""

    if not isinstance(name, str) or not name:
        return False
    if "\x00" in name:
        return False
    if "/" in name or "\\" in name:
        return False
    if os.path.splitdrive(name)[0]:
        return False
    if os.path.isabs(name):
        return False
    if name in (".", ".."):
        return False
    if name != os.path.basename(name):
        return False
    return True


def _canonical_digest(value: Any) -> Tuple[Optional[str], str]:
    """Return ``(digest_or_None, status)`` for a SHA-256 cell.

    ``status`` is ``ok`` / ``missing`` / ``invalid``.  A non-SHA-256 value such
    as ``size:123`` is invalid, never a pseudo-unique hash.
    """

    if value is None:
        return None, "missing"
    text = str(value).strip()
    if not text:
        return None, "missing"
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        return None, "invalid"
    return text.lower(), "ok"


def _parse_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number >= 0 else None


def _parse_dimensions(value: Any) -> Optional[Dict[str, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _DIMENSIONS_RE.match(text)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return {"width": width, "height": height}


def _manifest_basename(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] or None


def _issue(code: str, message: str, **context: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {"code": code, "message": message}
    for key, value in context.items():
        if value is not None:
            record[key] = value
    return record


def _guard_input_path(path: Any, label: str, split: str) -> str:
    """Reject CSV paths that physically resolve into a restricted split."""

    if not isinstance(path, (str, os.PathLike)):
        raise SampleTableError(f"{label} must be a path-like string")
    text = os.fspath(path)
    if not isinstance(text, str) or not text.strip():
        raise SampleTableError(f"{label} must be a non-empty path")
    resolved = policy.split_of_path(text)
    if resolved in ("test_final", "dev_final"):
        raise SampleTableError(
            f"{label} {text!r} resolves into restricted split {resolved!r}; "
            "this stage only reads train_final metadata"
        )
    if resolved is not None and resolved != split:
        raise SampleTableError(
            f"{label} {text!r} resolves to split {resolved!r}, expected {split!r}"
        )
    return text


def _guard_manifest_value(value: Any, field: str, split: str) -> None:
    """Reject manifest metadata that points into a restricted split.

    Only path parsing is used; nothing named here is opened.
    """

    if not isinstance(value, str) or not value.strip():
        return
    resolved = policy.split_of_path(value.strip())
    if resolved in ("test_final", "dev_final"):
        raise SampleTableError(
            f"manifest field {field!r}={value!r} points into restricted split "
            f"{resolved!r}; refusing to build a train sample table"
        )
    if resolved is not None and resolved != split:
        raise SampleTableError(
            f"manifest field {field!r}={value!r} declares split {resolved!r}, "
            f"expected {split!r}"
        )


def _read_csv(path: str, required: Sequence[str], label: str):
    try:
        handle = open(path, "r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise SampleTableError(
            f"cannot open {label} {path!r}: {type(exc).__name__}: {exc}"
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise SampleTableError(
                f"{label} {path!r} is missing required column(s): "
                + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def load_sample_rows(
    files_csv: Any,
    manifest_csv: Any = None,
    *,
    split: str = ALLOWED_SPLIT,
) -> Dict[str, Any]:
    """Build paired ``train_final`` sample rows from inventory CSV metadata.

    Only ``files_csv`` (and optional ``manifest_csv``) are opened.  Rows of
    other splits are counted and ignored; they never enter the sample table.
    Every input row is validated, and problems are returned as traceable
    ``issues`` rather than dropped silently.  Inventory digests are copied and
    marked unverified; this function does not hash files or claim the CSV
    reflects the current data tree.
    """

    if not isinstance(split, str) or split.strip() != ALLOWED_SPLIT:
        raise SampleTableError(
            f"load_sample_rows only accepts split={ALLOWED_SPLIT!r}, got {split!r}"
        )
    files_path = _guard_input_path(files_csv, "files_csv", ALLOWED_SPLIT)
    manifest_path = None
    if manifest_csv is not None:
        manifest_path = _guard_input_path(
            manifest_csv, "manifest_csv", ALLOWED_SPLIT
        )

    file_rows = _read_csv(files_path, FILES_COLUMNS, "files_csv")

    issues: List[Dict[str, Any]] = []
    accounting: Dict[str, Any] = {
        "files_csv_rows": len(file_rows),
        "files_csv_other_split_rows": 0,
        "files_csv_other_role_rows": 0,
        "files_csv_ignored_name_rows": 0,
        "files_csv_invalid_name_rows": 0,
        "files_csv_duplicate_name_rows": 0,
        "files_csv_conflicting_name_rows": 0,
        "train_png_rows": 0,
        "train_svg_rows": 0,
        "paired_stems": 0,
        "missing_png_stems": 0,
        "missing_svg_stems": 0,
        "duplicate_png_stems": 0,
        "duplicate_svg_stems": 0,
        "case_collision_stems": 0,
        "missing_digest_rows": 0,
        "invalid_digest_rows": 0,
        "manifest_rows": None,
        "manifest_attached_rows": 0,
        "manifest_unmatched_rows": 0,
        "manifest_filename_mismatch_rows": 0,
        "manifest_ambiguous_stems": 0,
    }

    records_by_name: Dict[str, List[Dict[str, Any]]] = {}
    invalid_names: set = set()
    png_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    svg_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    conflicted_stems: Dict[str, str] = {}
    case_map: Dict[str, set] = {}
    all_seen_stems: set = set()

    for row in file_rows:
        row_split = (row.get("split") or "").strip()
        if row_split != ALLOWED_SPLIT:
            accounting["files_csv_other_split_rows"] += 1
            continue

        name = row.get("name")
        role = (row.get("role") or "").strip()
        if role not in (policy.ROLE_PNG, policy.ROLE_SVG):
            accounting["files_csv_other_role_rows"] += 1
            continue

        if not _is_single_filename(name):
            accounting["files_csv_invalid_name_rows"] += 1
            issues.append(
                _issue(
                    "invalid_name",
                    "sample name must be a single file name without absolute "
                    "path, drive or path separator",
                    name=str(name),
                )
            )
            continue
        if policy.is_ignored_name(name):
            accounting["files_csv_ignored_name_rows"] += 1
            issues.append(
                _issue(
                    "ignored_resource_fork",
                    "resource-fork/.DS_Store name is never a sample",
                    name=name,
                )
            )
            continue

        derived_role = policy.classify_name(name)
        if derived_role != role:
            accounting["files_csv_invalid_name_rows"] += 1
            invalid_names.add(name)
            issues.append(
                _issue(
                    "role_extension_mismatch",
                    f"role {role!r} does not match extension of {name!r}",
                    name=name,
                )
            )
            continue

        declared_stem = (row.get("stem") or "").strip()
        derived_stem = policy.stem_of(name)
        if declared_stem and declared_stem != derived_stem:
            accounting["files_csv_invalid_name_rows"] += 1
            invalid_names.add(name)
            issues.append(
                _issue(
                    "stem_mismatch",
                    f"stem column {declared_stem!r} does not match name {name!r}",
                    name=name,
                )
            )
            continue
        declared_extension = (row.get("extension") or "").strip().lower()
        if declared_extension and declared_extension != os.path.splitext(name)[1].lower():
            accounting["files_csv_invalid_name_rows"] += 1
            invalid_names.add(name)
            issues.append(
                _issue(
                    "extension_mismatch",
                    f"extension column {declared_extension!r} does not match "
                    f"name {name!r}",
                    name=name,
                )
            )
            continue

        case_map.setdefault(name.lower(), set()).add(name)
        all_seen_stems.add(derived_stem)

        digest, digest_status = _canonical_digest(row.get("digest"))
        if digest_status == "missing":
            accounting["missing_digest_rows"] += 1
            issues.append(
                _issue(
                    "missing_digest",
                    "digest is empty; this row cannot contribute exact "
                    "duplicate edges and is not treated as unique",
                    name=name,
                )
            )
        elif digest_status == "invalid":
            accounting["invalid_digest_rows"] += 1
            issues.append(
                _issue(
                    "invalid_digest",
                    "digest is not a 64-character SHA-256 hex string; it is not "
                    "used as a hash",
                    name=name,
                )
            )

        size_bytes = _parse_size(row.get("size_bytes"))
        if size_bytes is None and str(row.get("size_bytes") or "").strip():
            issues.append(
                _issue(
                    "invalid_size_bytes",
                    "size_bytes is not a non-negative integer",
                    name=name,
                )
            )

        record = {
            "name": name,
            "role": role,
            "stem": derived_stem,
            "digest": digest,
            "digest_status": digest_status,
            "size_bytes": size_bytes,
        }
        if role == policy.ROLE_PNG:
            accounting["train_png_rows"] += 1
        else:
            accounting["train_svg_rows"] += 1
        records_by_name.setdefault(name, []).append(record)

    # Fail closed on duplicate file-name metadata.  Identical duplicate rows
    # may be de-duplicated explicitly; any identity-relevant difference (or an
    # invalid duplicate of the same name) excludes the whole stem instead of
    # keeping whichever row happened to come first.
    for name in sorted(records_by_name):
        records = records_by_name[name]
        stem = records[0]["stem"]
        duplicated = len(records) > 1 or name in invalid_names
        if not duplicated:
            record = records[0]
            if record["role"] == policy.ROLE_PNG:
                png_by_stem.setdefault(stem, []).append(record)
            else:
                svg_by_stem.setdefault(stem, []).append(record)
            continue

        accounting["files_csv_duplicate_name_rows"] += 1
        signatures = {
            (record["role"], record["stem"], record["digest"], record["size_bytes"])
            for record in records
        }
        identical = len(signatures) == 1 and name not in invalid_names
        if identical:
            issues.append(
                _issue(
                    "duplicate_name_row_identical",
                    "identical duplicate rows for the same file name were "
                    "de-duplicated explicitly; no distinct content was lost",
                    name=name,
                    duplicate_count=len(records),
                )
            )
            record = records[0]
            if record["role"] == policy.ROLE_PNG:
                png_by_stem.setdefault(stem, []).append(record)
            else:
                svg_by_stem.setdefault(stem, []).append(record)
            continue

        accounting["files_csv_conflicting_name_rows"] += 1
        conflicted_stems[stem] = "duplicate_name_conflict"
        fields = [
            field
            for field in ("role", "stem", "digest", "size_bytes")
            if len({record[field] for record in records}) > 1
        ]
        if name in invalid_names:
            fields.append("invalid_duplicate_row")
        issues.append(
            _issue(
                "duplicate_name_conflict",
                "the same file name has conflicting metadata; the whole stem "
                "is excluded and no candidate is kept",
                name=name,
                stem=stem,
                conflicting_fields=sorted(set(fields)),
                candidates=sorted(
                    [
                        {
                            "role": record["role"],
                            "digest": record["digest"],
                            "size_bytes": record["size_bytes"],
                            "digest_status": record["digest_status"],
                        }
                        for record in records
                    ],
                    key=lambda item: (
                        item["role"],
                        item["digest"] or "",
                        -1 if item["size_bytes"] is None else item["size_bytes"],
                    ),
                ),
            )
        )

    # Case collisions are ambiguous on case-insensitive filesystems; exclude
    # those stems instead of silently merging or picking one spelling.
    for lower_name, names in case_map.items():
        if len(names) > 1:
            stems = sorted({policy.stem_of(item) for item in names})
            accounting["case_collision_stems"] += len(stems)
            for stem in stems:
                conflicted_stems[stem] = "case_collision"
            issues.append(
                _issue(
                    "case_collision",
                    "file names differ only by case; pairing is ambiguous",
                    names=sorted(names),
                )
            )

    for stem, records in png_by_stem.items():
        if len(records) > 1:
            accounting["duplicate_png_stems"] += 1
            conflicted_stems[stem] = "duplicate_png"
            issues.append(
                _issue(
                    "duplicate_png_for_stem",
                    "more than one PNG shares this stem; no candidate is chosen",
                    stem=stem,
                    names=sorted(record["name"] for record in records),
                )
            )
    for stem, records in svg_by_stem.items():
        if len(records) > 1:
            accounting["duplicate_svg_stems"] += 1
            conflicted_stems[stem] = "duplicate_svg"
            issues.append(
                _issue(
                    "duplicate_svg_for_stem",
                    "more than one SVG shares this stem; no annotation version "
                    "is chosen",
                    stem=stem,
                    names=sorted(record["name"] for record in records),
                )
            )

    sample_stems = sorted(set(png_by_stem) | set(svg_by_stem))
    rows: List[Dict[str, Any]] = []
    all_stems = set(sample_stems) | set(all_seen_stems)
    for stem in sample_stems:
        if stem in conflicted_stems:
            continue
        pngs = png_by_stem.get(stem, [])
        svgs = svg_by_stem.get(stem, [])
        if not pngs:
            accounting["missing_png_stems"] += 1
            issues.append(
                _issue(
                    "missing_png",
                    "SVG has no same-stem PNG; no pair is fabricated",
                    stem=stem,
                )
            )
            continue
        if not svgs:
            accounting["missing_svg_stems"] += 1
            issues.append(
                _issue(
                    "missing_svg",
                    "PNG has no same-stem SVG annotation; it is not a paired sample",
                    stem=stem,
                )
            )
            continue
        png = pngs[0]
        svg = svgs[0]
        rows.append(
            {
                "sample_id": f"{ALLOWED_SPLIT}:{stem}",
                "split": ALLOWED_SPLIT,
                "stem": stem,
                "png_name": png["name"],
                "svg_name": svg["name"],
                "png_digest": png["digest"],
                "svg_digest": svg["digest"],
                "png_size_bytes": png["size_bytes"],
                "svg_size_bytes": svg["size_bytes"],
                "source_proxy": None,
                "manifest": None,
                "png_dimensions": None,
                "size_class": None,
                "verification": VERIFICATION,
            }
        )
        accounting["paired_stems"] = len(rows)

    # Optional manifest join: exact dest_stem plus file-name consistency only.
    manifest_used = False
    if manifest_path is not None:
        manifest_used = True
        manifest_rows = _read_csv(manifest_path, MANIFEST_COLUMNS, "manifest_csv")
        accounting["manifest_rows"] = len(manifest_rows)
        by_dest: Dict[str, List[Dict[str, Any]]] = {}
        for index, manifest_row in enumerate(manifest_rows):
            for field in ("source", "stem", "svg", "png"):
                _guard_manifest_value(manifest_row.get(field), field, ALLOWED_SPLIT)
            dest = (manifest_row.get("dest_stem") or "").strip()
            if dest:
                by_dest.setdefault(dest, []).append((index, manifest_row))

        rows_by_stem = {row["stem"]: row for row in rows}
        consumed: set = set()
        for stem, candidates in by_dest.items():
            sample = rows_by_stem.get(stem)
            if sample is None:
                continue
            if len(candidates) > 1:
                accounting["manifest_ambiguous_stems"] += 1
                issues.append(
                    _issue(
                        "manifest_duplicate_dest_stem",
                        "manifest has multiple rows for this dest_stem; the "
                        "source metadata is left unattached",
                        stem=stem,
                    )
                )
                continue
            index, manifest_row = candidates[0]
            png_base = _manifest_basename(manifest_row.get("png"))
            svg_base = _manifest_basename(manifest_row.get("svg"))
            if png_base != sample["png_name"] or svg_base != sample["svg_name"]:
                accounting["manifest_filename_mismatch_rows"] += 1
                issues.append(
                    _issue(
                        "manifest_filename_mismatch",
                        "manifest png/svg basenames do not match the inventory "
                        "pair; metadata is not attached",
                        stem=stem,
                        png=png_base,
                        svg=svg_base,
                    )
                )
                continue
            consumed.add(index)
            source = (manifest_row.get("source") or "").strip()
            size_class = (manifest_row.get("size_class") or "").strip()
            dimensions = _parse_dimensions(manifest_row.get("png_dimensions"))
            if (
                str(manifest_row.get("png_dimensions") or "").strip()
                and dimensions is None
            ):
                issues.append(
                    _issue(
                        "invalid_png_dimensions",
                        "png_dimensions is not 'WIDTHxHEIGHT'; the value is kept "
                        "raw but not parsed",
                        stem=stem,
                    )
                )
            sample["source_proxy"] = source or None
            sample["size_class"] = size_class or None
            sample["png_dimensions"] = dimensions
            sample["manifest"] = {
                name: manifest_row.get(name) for name in MANIFEST_COLUMNS
            }
            accounting["manifest_attached_rows"] += 1

        accounting["manifest_unmatched_rows"] = len(manifest_rows) - len(consumed)

    rows.sort(key=lambda item: item["sample_id"])
    accounting["paired_stems"] = len(rows)

    notes = [
        "only train_final rows enter the sample table; other split rows are counted and ignored",
        "digests are copied from the inventory CSV and marked metadata_only_unverified; "
        "they do not prove the current files are unchanged",
        "a same-stem PNG/SVG pair is a file-name pairing identity, not proof of "
        "independent source images",
        "manifest source/stem/size_class are control-side metadata descriptions, "
        "not features and not proven source identity",
        "no PNG/SVG bytes, labels or directory entries were read",
    ]
    if not manifest_used:
        notes.append(
            "no manifest supplied: source_proxy, png_dimensions and size_class stay "
            "null; representativeness is metadata-only"
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "split": ALLOWED_SPLIT,
        "verification": VERIFICATION,
        "inputs": {
            "files_csv": files_path,
            "manifest_csv": manifest_path,
        },
        "rows": rows,
        "issues": issues,
        "accounting": accounting,
        "excluded_stems": sorted(all_stems - {row["stem"] for row in rows}),
        "manifest_used": manifest_used,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 2. group_exact_duplicates
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _group_id(member_ids: Sequence[str]) -> str:
    payload = "\x00".join(sorted(member_ids)).encode("utf-8")
    return "g-" + hashlib.sha256(payload).hexdigest()[:16]


def group_exact_duplicates(
    rows: Sequence[Dict[str, Any]], *, use_source_proxy: bool = False
) -> Dict[str, Any]:
    """Group rows by equal PNG/SVG SHA-256 digests (transitive closure).

    Equal PNG digest and equal SVG digest are both strong edges.  A PNG digest
    shared by different SVG annotations therefore keeps *all* annotation
    variants in one group; no representative is chosen.  ``source_proxy`` is an
    optional, explicitly unproven merge key and is off by default.
    """

    if not isinstance(use_source_proxy, bool):
        raise TypeError("use_source_proxy must be a bool")
    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be a list of sample rows")

    working: List[Dict[str, Any]] = copy.deepcopy(list(rows))
    issues: List[Dict[str, Any]] = []
    invalid_digest_rows = 0
    missing_digest_rows = 0

    seen_ids: Dict[str, Dict[str, Any]] = {}
    unique_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(working):
        if not isinstance(row, dict):
            raise TypeError(f"row {index} is not a mapping")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SampleTableError(f"row {index} has no usable sample_id")
        if sample_id in seen_ids:
            if seen_ids[sample_id] == row:
                issues.append(
                    _issue(
                        "duplicate_sample_id_identical",
                        "identical duplicate row collapsed; no distinct data lost",
                        sample_id=sample_id,
                    )
                )
                continue
            raise SampleTableError(
                f"conflicting duplicate sample_id {sample_id!r}; refusing to "
                "merge different row contents"
            )
        seen_ids[sample_id] = row
        unique_rows.append(row)

    for row in unique_rows:
        for key in ("png_digest", "svg_digest"):
            digest, status = _canonical_digest(row.get(key))
            if status == "missing":
                missing_digest_rows += 1
            elif status == "invalid":
                invalid_digest_rows += 1
                issues.append(
                    _issue(
                        "invalid_digest",
                        f"{key} is not a canonical SHA-256 hex string and is not "
                        "used as an edge",
                        sample_id=row.get("sample_id"),
                        field=key,
                    )
                )
            row[key] = digest

    union = _UnionFind(len(unique_rows))
    edge_counts = {
        "exact_png_digest": 0,
        "exact_svg_digest": 0,
        "source_proxy": 0,
    }
    successful_edges: List[Tuple[str, int, int]] = []
    first_by_digest: Dict[Tuple[str, str], int] = {}
    for index, row in enumerate(unique_rows):
        for field, basis in (
            ("png_digest", "exact_png_digest"),
            ("svg_digest", "exact_svg_digest"),
        ):
            digest = row.get(field)
            if not digest:
                continue
            key = (basis, digest)
            if key in first_by_digest:
                if union.union(first_by_digest[key], index):
                    edge_counts[basis] += 1
                    successful_edges.append((basis, first_by_digest[key], index))
            else:
                first_by_digest[key] = index

    if use_source_proxy:
        first_by_source: Dict[str, int] = {}
        for index, row in enumerate(unique_rows):
            source = row.get("source_proxy")
            if not isinstance(source, str) or not source.strip():
                continue
            key = source.strip()
            if key in first_by_source:
                if union.union(first_by_source[key], index):
                    edge_counts["source_proxy"] += 1
                    successful_edges.append(
                        ("source_proxy", first_by_source[key], index)
                    )
            else:
                first_by_source[key] = index

    components: Dict[int, List[int]] = {}
    for index in range(len(unique_rows)):
        components.setdefault(union.find(index), []).append(index)
    component_of = {index: union.find(index) for index in range(len(unique_rows))}
    bases_by_component: Dict[int, set] = {}
    for basis, left, right in successful_edges:
        bases_by_component.setdefault(component_of[left], set()).add(basis)
        bases_by_component.setdefault(component_of[right], set()).add(basis)

    groups: List[Dict[str, Any]] = []
    for root, members in components.items():
        member_rows = sorted(
            (unique_rows[index] for index in members),
            key=lambda item: item["sample_id"],
        )
        member_ids = [row["sample_id"] for row in member_rows]
        basis: List[str] = sorted(bases_by_component.get(root, set()))
        png_digests = [
            row["png_digest"] for row in member_rows if row.get("png_digest")
        ]
        svg_digests = [
            row["svg_digest"] for row in member_rows if row.get("svg_digest")
        ]
        sources = [
            row.get("source_proxy")
            for row in member_rows
            if isinstance(row.get("source_proxy"), str)
            and row.get("source_proxy").strip()
        ]
        if len(set(png_digests)) < len(png_digests):
            basis.append("exact_png_digest")
        if len(set(svg_digests)) < len(svg_digests):
            basis.append("exact_svg_digest")
        if use_source_proxy and len(set(sources)) < len(sources):
            basis.append("source_proxy")

        svg_names = sorted(
            {row.get("svg_name") for row in member_rows if row.get("svg_name")}
        )
        risk_codes: List[str] = [
            "metadata_only_inventory_unverified",
            "exact_duplicates_do_not_imply_source_independence",
        ]
        if len(svg_names) > 1 and len(png_digests) > len(set(png_digests)):
            risk_codes.append("png_identical_annotation_variants_preserved")
        if len(svg_digests) > len(set(svg_digests)) and len(set(png_digests)) > 1:
            risk_codes.append("svg_identical_across_different_png")
        if "source_proxy" in basis:
            risk_codes.append("source_proxy_unverified")
        if any(row.get("png_digest") is None or row.get("svg_digest") is None for row in member_rows):
            risk_codes.append("missing_digest_not_grouped")

        groups.append(
            {
                "group_id": _group_id(member_ids),
                "basis": basis,
                "member_sample_ids": member_ids,
                "pair_count": len(member_rows),
                "members": member_rows,
                "annotation_variants": {
                    "svg_names": svg_names,
                    "svg_digests": sorted(set(svg_digests)),
                    "distinct_svg_name_count": len(svg_names),
                    "distinct_svg_digest_count": len(set(svg_digests)),
                    "distinct_png_digest_count": len(set(png_digests)),
                    "multiple_annotation_versions": len(svg_names) > 1,
                },
                "risk_codes": risk_codes,
            }
        )

    groups.sort(key=lambda item: item["group_id"])
    singleton_count = sum(1 for group in groups if group["pair_count"] == 1)
    multi_count = len(groups) - singleton_count

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "grouping": {
            "exact_png_digest": True,
            "exact_svg_digest": True,
            "source_proxy": use_source_proxy,
            "transitive_closure": True,
        },
        "groups": groups,
        "issues": issues,
        "accounting": {
            "input_row_count": len(working),
            "unique_row_count": len(unique_rows),
            "group_count": len(groups),
            "singleton_group_count": singleton_count,
            "multi_member_group_count": multi_count,
            "merged_png_digest_edges": edge_counts["exact_png_digest"],
            "merged_svg_digest_edges": edge_counts["exact_svg_digest"],
            "merged_source_proxy_edges": edge_counts["source_proxy"],
            "missing_digest_fields": missing_digest_rows,
            "invalid_digest_fields": invalid_digest_rows,
        },
        "risk_notes": [
            "exact digest equality is metadata-only and unverified against the current files",
            "exact duplicate isolation does not exclude near-duplicates or unproven "
            "same-source images",
            "a same-stem PNG/SVG pairing is not proof of source-image independence",
            "source_proxy merges use an unverified control-side label, never a "
            "proven source identity",
            "missing/invalid digests are never merged into one pseudo-group",
        ],
    }


# ---------------------------------------------------------------------------
# 3. freeze_selection
# ---------------------------------------------------------------------------


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SampleTableError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise SampleTableError(f"{name} must be positive, got {value!r}")
    return value


def _seeded_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}\x00{group_id}".encode("utf-8")).hexdigest()


def _deterministic_order(
    group_ids: Sequence[str],
    size_classes: Dict[str, List[str]],
    seed: int,
) -> List[str]:
    """Order groups deterministically, lightly stratified when classes exist.

    The seeded SHA-256 key replaces any dependence on input order or on
    Python's RNG implementation.  When every group has a size class, groups are
    round-robined across sorted class buckets; this is ordering-only
    stratification, not a claim about geometric representativeness.
    """

    ordered_by_class: Dict[str, List[str]] = {}
    classes_present = True
    for group_id in group_ids:
        classes = size_classes.get(group_id, [])
        if not classes:
            classes_present = False
            break
        ordered_by_class.setdefault(",".join(classes), []).append(group_id)

    if not classes_present:
        return sorted(group_ids, key=lambda gid: (_seeded_key(seed, gid), gid))

    buckets: List[List[str]] = []
    for class_key in sorted(ordered_by_class):
        buckets.append(
            sorted(
                ordered_by_class[class_key],
                key=lambda gid: (_seeded_key(seed, gid), gid),
            )
        )
    ordered: List[str] = []
    position = 0
    while True:
        added = False
        for bucket in buckets:
            if position < len(bucket):
                ordered.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return ordered


def _canonical_member(member: Dict[str, Any]) -> Dict[str, Any]:
    """Stable, JSON-safe member content used by the freeze fingerprint."""

    png_digest, _ = _canonical_digest(member.get("png_digest"))
    svg_digest, _ = _canonical_digest(member.get("svg_digest"))
    return {
        "sample_id": member.get("sample_id"),
        "split": member.get("split"),
        "stem": member.get("stem"),
        "png_name": member.get("png_name"),
        "svg_name": member.get("svg_name"),
        "png_digest": png_digest,
        "svg_digest": svg_digest,
        "png_size_bytes": member.get("png_size_bytes"),
        "svg_size_bytes": member.get("svg_size_bytes"),
        "source_proxy": member.get("source_proxy"),
        "png_dimensions": copy.deepcopy(member.get("png_dimensions")),
        "size_class": member.get("size_class"),
        "verification": member.get("verification"),
        "manifest": copy.deepcopy(member.get("manifest")),
    }


def _normalized_grouping(grouped: Dict[str, Any]) -> Dict[str, bool]:
    raw = grouped.get("grouping")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "exact_png_digest": bool(raw.get("exact_png_digest", True)),
        "exact_svg_digest": bool(raw.get("exact_svg_digest", True)),
        "source_proxy": bool(raw.get("source_proxy", False)),
        "transitive_closure": bool(raw.get("transitive_closure", True)),
    }


def _count_unique(members: Sequence[Dict[str, Any]], kind: str) -> int:
    if kind == "png":
        digest_key, name_key = "png_digest", "png_name"
    else:
        digest_key, name_key = "svg_digest", "svg_name"
    keys = set()
    for member in members:
        digest = member.get(digest_key)
        if isinstance(digest, str) and digest:
            keys.add(("digest", digest))
        else:
            keys.add(("name", member.get(name_key) or member.get("sample_id")))
    return len(keys)


def freeze_selection(
    grouped: Dict[str, Any],
    *,
    sample_count: int = 24,
    seed: int = 20261108,
    development_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Freeze a reproducible, group-atomic development/holdout selection.

    Groups are partitioned first, then selected whole.  Selection order uses a
    seed-derived SHA-256 key over sorted group IDs, so input order and the
    local RNG state cannot change the result.  The fingerprint covers the
    configuration, the complete group order, the partition and the selection,
    and contains no timestamp.
    """

    if not isinstance(grouped, dict) or not isinstance(grouped.get("groups"), list):
        raise TypeError("grouped must be the output of group_exact_duplicates")
    groups_input = grouped["groups"]

    count = _validate_positive_int(sample_count, "sample_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SampleTableError(f"seed must be an integer, got {seed!r}")
    if (
        isinstance(development_fraction, bool)
        or not isinstance(development_fraction, (int, float))
        or not math.isfinite(float(development_fraction))
    ):
        raise SampleTableError(
            f"development_fraction must be a finite number, got {development_fraction!r}"
        )
    fraction = float(development_fraction)
    if not 0.0 < fraction < 1.0:
        raise SampleTableError(
            "development_fraction must be in (0, 1) so both development and "
            f"holdout can exist, got {development_fraction!r}"
        )

    by_id: Dict[str, Dict[str, Any]] = {}
    size_classes: Dict[str, List[str]] = {}
    for index, group in enumerate(groups_input):
        if not isinstance(group, dict):
            raise TypeError(f"group {index} is not a mapping")
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise SampleTableError(f"group {index} has no usable group_id")
        if group_id in by_id:
            raise SampleTableError(f"duplicate group_id {group_id!r}")
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise SampleTableError(
                f"group {group_id!r} must carry a non-empty 'members' list"
            )
        for member in members:
            if not isinstance(member, dict) or not isinstance(
                member.get("sample_id"), str
            ):
                raise SampleTableError(
                    f"group {group_id!r} has a member without sample_id"
                )
        by_id[group_id] = group
        classes = sorted(
            {
                member.get("size_class")
                for member in members
                if isinstance(member.get("size_class"), str)
                and member.get("size_class").strip()
            }
        )
        size_classes[group_id] = classes

    all_member_ids = [
        member["sample_id"]
        for group in by_id.values()
        for member in group["members"]
    ]
    if len(all_member_ids) != len(set(all_member_ids)):
        raise SampleTableError(
            "the same sample_id appears in more than one group; grouped metadata "
            "is inconsistent"
        )

    group_ids = sorted(by_id)
    order = _deterministic_order(group_ids, size_classes, seed)
    total_groups = len(order)

    development_groups_all: List[str]
    holdout_groups_all: List[str]
    status = "ok"
    split_limited_reasons: List[str] = []
    if total_groups == 0:
        development_groups_all = []
        holdout_groups_all = []
        status = "split_limited"
        split_limited_reasons.append("no_groups")
    elif total_groups == 1:
        development_groups_all = order[:]
        holdout_groups_all = []
        status = "split_limited"
        split_limited_reasons.append("fewer_than_two_groups_no_independent_holdout")
    else:
        development_count = int(total_groups * fraction + 0.5)
        development_count = max(1, min(development_count, total_groups - 1))
        development_groups_all = order[:development_count]
        holdout_groups_all = order[development_count:]

    sizes = {group_id: len(by_id[group_id]["members"]) for group_id in group_ids}

    development_target = (
        max(1, int(count * fraction + 0.5)) if development_groups_all else 0
    )
    holdout_target = max(1, count - development_target) if holdout_groups_all else 0

    selected_development: List[str] = []
    development_pairs = 0
    for group_id in development_groups_all:
        if development_pairs >= development_target:
            break
        selected_development.append(group_id)
        development_pairs += sizes[group_id]

    selected_holdout: List[str] = []
    holdout_pairs = 0
    for group_id in holdout_groups_all:
        if holdout_pairs >= holdout_target:
            break
        selected_holdout.append(group_id)
        holdout_pairs += sizes[group_id]

    warnings: List[Dict[str, Any]] = []
    if status == "split_limited":
        warnings.append(
            {
                "code": "split_limited",
                "message": "; ".join(split_limited_reasons),
            }
        )
    if development_groups_all and development_pairs < development_target:
        status = "selection_limited" if status == "ok" else status
        warnings.append(
            {
                "code": "development_pool_exhausted",
                "message": (
                    f"development pool provided {development_pairs} pairs, below "
                    f"target {development_target}"
                ),
            }
        )
    if holdout_groups_all and holdout_pairs < holdout_target:
        status = "selection_limited" if status == "ok" else status
        warnings.append(
            {
                "code": "holdout_pool_exhausted",
                "message": (
                    f"holdout pool provided {holdout_pairs} pairs, below target "
                    f"{holdout_target}"
                ),
            }
        )

    pair_count = development_pairs + holdout_pairs
    if pair_count > count:
        warnings.append(
            {
                "code": "pair_count_overshoot",
                "message": (
                    f"whole-group selection produced {pair_count} pairs for a "
                    f"target of {count}; groups are never split"
                ),
            }
        )
    if pair_count < count and status == "ok":
        status = "selection_limited"
        warnings.append(
            {
                "code": "pair_count_below_target",
                "message": (
                    f"whole-group selection produced {pair_count} pairs, below "
                    f"target {count}"
                ),
            }
        )

    selected_ids = selected_development + selected_holdout
    selected_members: List[Dict[str, Any]] = []
    for group_id in selected_ids:
        selected_members.extend(by_id[group_id]["members"])

    selected_groups: List[Dict[str, Any]] = []
    for split_name, ids in (
        ("development", selected_development),
        ("holdout", selected_holdout),
    ):
        for group_id in ids:
            group = by_id[group_id]
            selected_groups.append(
                {
                    "group_id": group_id,
                    "split": split_name,
                    "basis": list(group.get("basis", [])),
                    "pair_count": len(group["members"]),
                    "member_sample_ids": [
                        member["sample_id"] for member in group["members"]
                    ],
                    "members": copy.deepcopy(group["members"]),
                    "annotation_variants": copy.deepcopy(
                        group.get("annotation_variants", {})
                    ),
                    "risk_codes": list(group.get("risk_codes", [])),
                }
            )

    config_block = {
        "sample_count": count,
        "seed": seed,
        "development_fraction": fraction,
        "rule": (
            "sort groups by seeded sha256 key; partition whole groups into "
            "development then holdout; select whole groups greedily in that "
            "order until each split target is reached"
        ),
        "size_class_stratification": "ordering_only_when_classes_present",
    }
    grouping_block = _normalized_grouping(grouped)
    content_groups = []
    for group_id in sorted(by_id):
        group = by_id[group_id]
        content_groups.append(
            {
                "group_id": group_id,
                "basis": sorted(group.get("basis", [])),
                "members": sorted(
                    (_canonical_member(member) for member in group["members"]),
                    key=lambda member: str(member.get("sample_id")),
                ),
            }
        )
    fingerprint_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "config": config_block,
        "grouping": grouping_block,
        "groups": content_groups,
        "group_order": order,
        "development_groups": development_groups_all,
        "holdout_groups": holdout_groups_all,
        "selected_development": selected_development,
        "selected_holdout": selected_holdout,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    limitations = [
        "inventory digests are metadata_only_unverified; rebuild a train-only "
        "inventory and verify hashes before treating this as a frozen data list",
        "no primitive counts or foreground-density statistics are available in "
        "this metadata lane, so representativeness is metadata-only",
        "exact duplicate isolation does not exclude near-duplicates or unproven "
        "same-source images",
        "same-stem PNG/SVG pairing is a file-name identity, not source independence",
        "size_class, where present, is used only to order groups; it is not a "
        "geometric stratification",
    ]

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "config": config_block,
        "fingerprint": fingerprint,
        "selection": {
            "development_group_ids": list(selected_development),
            "holdout_group_ids": list(selected_holdout),
            "untouched_group_ids": sorted(set(group_ids) - set(selected_ids)),
        },
        "groups": selected_groups,
        "counts": {
            "requested_pair_count": count,
            "pair_count": pair_count,
            "unique_png_count": _count_unique(selected_members, "png"),
            "unique_svg_count": _count_unique(selected_members, "svg"),
            "group_count": len(selected_groups),
            "available_group_count": total_groups,
            "available_pair_count": sum(sizes.values()),
            "development_pair_count": development_pairs,
            "holdout_pair_count": holdout_pairs,
            "development_group_count": len(selected_development),
            "holdout_group_count": len(selected_holdout),
            "selection_limited": status == "selection_limited",
            "split_limited": status == "split_limited",
            "target_overshoot": pair_count > count,
        },
        "warnings": warnings,
        "limitations": limitations,
    }
