"""Command line entry points.

Examples (from the repository root)::

    python -m cadvec inventory --datasets datasets --out data/processed
    python -m cadvec parse-train --datasets datasets --out data/processed \
        --sample 5
    python -m cadvec parse-svg tests/fixtures/train_like.svg

The CLI never writes below ``datasets/``; outputs go to ``data/processed``
by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .attributes import LabelConfig, load_label_config
from .inventory import (
    DEFAULT_SPLITS,
    build_inventory,
    inventory_output_paths,
    write_inventory,
)
from .policy import AnnotationAccessError
from .svg_parse import SVGParseError, iter_svg_names, parse_split, parse_svg

DEFAULT_OUT_DIR = os.path.join("data", "processed")


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass


def _write_json(path: str, payload: Any) -> None:
    """Serialise strictly first, so a failed write never leaves invalid JSON."""

    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")


def _write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records
    ]
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def _canonical_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _guard_write_targets(
    datasets_root: Optional[str],
    targets: Sequence[Optional[str]],
    protected_inputs: Sequence[str] = (),
) -> None:
    """Refuse unsafe write targets before any mkdir/open/write.

    Rejects: a target inside the dataset tree, a target equal to an input
    source file, and two targets that resolve to the same final path
    (relative/absolute spellings and symlink aliases included).  Symlinks and
    junctions are resolved before comparison.  Temporary directories and
    ``data/processed`` stay writable.
    """

    protected_roots = [_canonical_path(datasets_root)] if datasets_root else []
    protected_files = [_canonical_path(item) for item in protected_inputs]
    seen: Dict[str, str] = {}
    for raw in targets:
        if not raw:
            continue
        target = _canonical_path(raw)
        if target in seen:
            raise ValueError(
                f"refusing to write {raw!r}: it is the same final path as "
                f"{seen[target]!r}"
            )
        seen[target] = raw
        for root in protected_roots:
            if target == root or target.startswith(root + os.sep):
                raise ValueError(
                    f"refusing to write {raw!r}: it resolves inside the source "
                    f"dataset tree {datasets_root!r}"
                )
        for source in protected_files:
            if target == source:
                raise ValueError(
                    f"refusing to overwrite the input source file: {raw!r}"
                )


def _error_diagnostic_count(summary: Dict[str, Any]) -> int:
    return int(summary.get("by_severity", {}).get("error", 0))


def _load_config(path: Optional[str]) -> LabelConfig:
    if not path:
        return LabelConfig()
    return load_label_config(path)


def _inventory_summary(report: Dict[str, Any]) -> str:
    lines = ["inventory"]
    for split, entry in report["splits"].items():
        files = entry["files"]
        pairing = entry["pairing"]
        lines.append(
            f"  {split}: total={files['total']} "
            f"excluded={files['excluded_resource_forks']} "
            f"samples={files['samples']} png={files['png']} svg={files['svg']} "
            f"other={files['other']} paired={pairing['paired']} "
            f"png_only={pairing['png_only']} svg_only={pairing['svg_only']}"
        )
    totals = report["totals"]
    lines.append(
        f"  totals: files={totals['files']} samples={totals['samples']} "
        f"png={totals['png']} svg={totals['svg']} paired={totals['paired']}"
    )
    return "\n".join(lines)


def cmd_inventory(args: argparse.Namespace) -> int:
    splits = tuple(args.splits.split(",")) if args.splits else DEFAULT_SPLITS
    if not os.path.isdir(args.datasets):
        print(
            f"error: dataset root is missing or not a directory: {args.datasets}",
            file=sys.stderr,
        )
        return 2

    missing = [
        split
        for split in splits
        if not os.path.isdir(os.path.join(args.datasets, split))
    ]
    if len(missing) == len(splits):
        print(
            "error: none of the requested split directories exist under "
            f"{args.datasets}: {', '.join(splits)}",
            file=sys.stderr,
        )
        return 2

    paths = inventory_output_paths(args.out)
    # Guard both final files, not just the output directory: an existing
    # inventory.json / files.csv can itself be a symlink back into datasets.
    _guard_write_targets(
        args.datasets, [paths["inventory_json"], paths["files_csv"]]
    )

    report = build_inventory(args.datasets, splits=splits, hash_algorithm=args.hash)
    paths = write_inventory(report, args.out)
    print(_inventory_summary(report))
    if missing:
        print(
            f"warning: split directories missing: {', '.join(missing)}",
            file=sys.stderr,
        )
    print(f"  wrote {paths['inventory_json']}")
    print(f"  wrote {paths['files_csv']}")
    return 0


def cmd_parse_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    split_dir = os.path.join(args.datasets, args.split)
    if not os.path.isdir(split_dir):
        print(f"error: split directory not found: {split_dir}", file=sys.stderr)
        return 2

    summary_path = args.out if args.out and args.out.endswith(".json") else os.path.join(
        args.out, "train_parse_summary.json"
    )
    sample_path = None
    if args.sample > 0:
        sample_path = args.sample_out or os.path.join(
            os.path.dirname(os.path.abspath(summary_path)), "train_parse_samples.jsonl"
        )
    _guard_write_targets(
        args.datasets,
        [summary_path, sample_path],
        [os.path.join(split_dir, name) for name in iter_svg_names(split_dir)],
    )

    summary = parse_split(
        split_dir,
        split=args.split,
        limit=args.limit,
        allow_test_annotations=args.allow_test_annotations,
        config=config,
        curve_steps=args.curve_steps,
        record_commands=False,
        collect_samples=args.sample,
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Samples were parsed from the same --limit selection; no second pass.
    samples = summary.pop("samples", [])
    _write_json(summary_path, summary)

    if sample_path is not None and samples:
        _write_jsonl(sample_path, samples)
    else:
        sample_path = None

    primitives = summary["primitives"]
    files = summary["files"]
    print("parse-train")
    print(
        f"  files: parsed={files['parsed']} failed={files['failed']} "
        f"error_diagnostics={files['with_error_diagnostics']} "
        f"zero_primitives={files['with_zero_primitives']}"
    )
    print(
        f"  primitives: total={primitives['total']} paths={primitives['paths']} "
        f"circles={primitives['circles']} ellipses={primitives['ellipses']}"
    )
    print(
        f"  labels: semantic_present={primitives['labeled_semantic']} "
        f"missing={primitives['missing_semantic']} | "
        f"instance_present={primitives['labeled_instance']} "
        f"missing={primitives['missing_instance']}"
    )
    print(f"  diagnostics: {summary['diagnostics']['by_severity']}")
    print(
        f"  transform: primitives_with_unapplied_transform="
        f"{primitives['with_unapplied_transform']}"
    )
    print(f"  wrote {summary_path}")
    if sample_path:
        print(f"  wrote {sample_path} ({len(samples)} files)")
    else:
        print("  no sample file written (no selected file or --sample 0)")
    error_diagnostics = _error_diagnostic_count(summary["diagnostics"])
    if files["failed"] or error_diagnostics:
        print(
            f"  status: FAILED ({files['failed']} file parse failure(s), "
            f"{error_diagnostics} error diagnostic(s))",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_parse_svg(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    # parse-svg has no --datasets flag; the repository default tree is still
    # protected, and the file being parsed can never be overwritten.
    _guard_write_targets(
        getattr(args, "datasets", None) or "datasets",
        [args.out],
        [args.file],
    )
    document = parse_svg(
        args.file,
        split=args.split,
        allow_test_annotations=args.allow_test_annotations,
        config=config,
        curve_steps=args.curve_steps,
        record_commands=True,
        include_digest=True,
    )
    if args.out:
        _write_json(args.out, document)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    errors = _error_diagnostic_count(document["diagnostic_summary"])
    if errors:
        print(
            f"status: {errors} error diagnostic(s) in {args.file}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadvec",
        description="Offline inventory and SVG structure parser for the CAD "
        "floor-plan competition data (algorithm stage 1).",
    )
    parser.add_argument("--version", action="version", version=f"cadvec {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="file-level inventory (names, sizes, digests, pairing)"
    )
    inventory.add_argument("--datasets", default="datasets")
    inventory.add_argument("--out", default=DEFAULT_OUT_DIR)
    inventory.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    inventory.add_argument("--hash", choices=("sha256", "size", "none"), default="sha256")
    inventory.set_defaults(func=cmd_inventory)

    parse_train = subparsers.add_parser(
        "parse-train",
        help="parse SVGs of a split; test_final is blocked unless explicitly allowed",
    )
    parse_train.add_argument("--datasets", default="datasets")
    parse_train.add_argument("--split", default="train_final")
    parse_train.add_argument("--out", default=DEFAULT_OUT_DIR)
    parse_train.add_argument("--limit", type=int, default=None)
    parse_train.add_argument(
        "--sample",
        type=int,
        default=5,
        help="number of full-document samples to keep (default 5, 0 disables)",
    )
    parse_train.add_argument("--sample-out", default=None)
    parse_train.add_argument("--config", default=None)
    parse_train.add_argument("--curve-steps", type=int, default=8)
    parse_train.add_argument("--allow-test-annotations", action="store_true")
    parse_train.set_defaults(func=cmd_parse_train)

    parse_one = subparsers.add_parser("parse-svg", help="parse one SVG file")
    parse_one.add_argument("file")
    parse_one.add_argument("--out", default=None)
    parse_one.add_argument("--split", default=None)
    parse_one.add_argument("--config", default=None)
    parse_one.add_argument("--curve-steps", type=int, default=8)
    parse_one.add_argument("--allow-test-annotations", action="store_true")
    parse_one.set_defaults(func=cmd_parse_svg)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AnnotationAccessError as exc:
        print(f"policy error: {exc}", file=sys.stderr)
        return 3
    except (SVGParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # PermissionError, IsADirectoryError and friends: a usage/IO failure,
        # not a programming error.  Other exception types still surface.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
