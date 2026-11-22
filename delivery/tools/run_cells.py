"""Run several A/B cells back to back in one process (one GPU job at a time).

  python tools/run_cells.py --cells ab-logits50:logits:0.50 ab-prob45:prob:0.45 \
      ab-prob40:prob:0.40

Each spec is ``out_dir_suffix:mask_paste:mask_threshold``; every cell gets its own
``runs/<name>`` directory and ``cell_receipt.json`` from ``tools/run_cell.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_cell  # noqa: E402


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    codes = {}
    for spec in args.cells:
        name, mode, threshold = spec.split(":")
        cell_argv = [
            "--out", os.path.join(ROOT, "runs", name),
            "--mask-paste", mode,
            "--mask-threshold", threshold,
        ]
        if args.dry_run:
            cell_argv.append("--dry-run")
        print("=== cell %s (%s @ %s)" % (name, mode, threshold), flush=True)
        codes[name] = run_cell.main(cell_argv)
        print("=== cell %s exit=%s" % (name, codes[name]), flush=True)
    print({"cells": codes})
    return max(codes.values()) if codes else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
