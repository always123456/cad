"""Byte comparison of two run directories over a set of documents.

  python tools/compare_runs.py <dir_a> <dir_b> [--inputs runs/subset96-inputs.json]
                              [--results-only]

Compares ``results/<stem>.json`` (the delivered document) and, unless
``--results-only``, ``predictions/<sid>.json`` (the prediction envelope) byte for
byte, and prints the differences it found.  A non-zero exit means they differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def read(path):
    with open(path, "rb") as handle:
        return handle.read()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args(argv)

    if args.inputs:
        with open(args.inputs, encoding="utf-8") as handle:
            samples = json.load(handle)["samples"]
    else:
        samples = [
            {"sample_id": name[:-5].replace("_", ":", 1),
             "png_path": name}
            for name in sorted(os.listdir(os.path.join(args.left, "results")))
        ]

    report = {"left": args.left, "right": args.right, "documents": 0,
              "documents_identical": 0, "documents_differing": [],
              "predictions_compared": 0, "predictions_identical": 0,
              "predictions_differing": [], "missing": []}
    for sample in samples:
        sid = sample["sample_id"]
        stem = os.path.splitext(os.path.basename(sample["png_path"]))[0]
        for kind, folder, name in (
            ("documents", "results", stem + ".json"),
            ("predictions", "predictions", sid.replace(":", "_") + ".json"),
        ):
            if kind == "predictions" and args.results_only:
                continue
            left = os.path.join(args.left, folder, name)
            right = os.path.join(args.right, folder, name)
            if not (os.path.exists(left) and os.path.exists(right)):
                report["missing"].append([kind, name,
                                          os.path.exists(left), os.path.exists(right)])
                continue
            if kind == "documents":
                report["documents"] += 1
            else:
                report["predictions_compared"] += 1
            same = read(left) == read(right)
            if same:
                if kind == "documents":
                    report["documents_identical"] += 1
                else:
                    report["predictions_identical"] += 1
            else:
                report[kind + "_differing"].append(
                    {"name": name,
                     "left_sha256": hashlib.sha256(read(left)).hexdigest()[:16],
                     "right_sha256": hashlib.sha256(read(right)).hexdigest()[:16],
                     "left_bytes": os.path.getsize(left),
                     "right_bytes": os.path.getsize(right)}
                )
    print(json.dumps(report, indent=1))
    ok = (
        not report["documents_differing"]
        and not report["predictions_differing"]
        and not report["missing"]
        and report["documents"] == len(samples)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
