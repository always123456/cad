"""CLI for the model-independent instance evaluator.

    python run_instance_eval.py prepare       --predictions DIR --out DIR [--maps DIR] [--masks DIR]
    python run_instance_eval.py prepare-masks --masks DIR --out DIR [--masks-format png|npz]
    python run_instance_eval.py score         --out DIR --allow-gt [--report FILE] [--baseline FILE]

``prepare``/``prepare-masks`` never read GT.  ``score`` requires the explicit
``--allow-gt`` grant and verifies the prepared seal before the first GT read.
``--sample-set fit16`` swaps the default four exposed TUNE samples for a
capacity-16 FIT diagnostic set drawn disjointly from them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instance_eval import contract as C  # noqa: E402
from instance_eval import evaluator as E  # noqa: E402


def _emit(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _inputs(args: argparse.Namespace) -> object:
    if args.inputs:
        return args.inputs
    if args.sample_set == "fit16":
        return C.fit_inputs(args.manifest, count=16)
    return None


def _dataset(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset
    return "tune-fixed4" if args.sample_set == "fixed4" else "fit16-diagnostic"


def _prepare(args: argparse.Namespace) -> int:
    result = E.prepare(
        args.predictions,
        args.out,
        inputs=_inputs(args),
        maps_dir=args.maps,
        masks_dir=args.masks,
        mask_format=args.masks_format,
        mask_role=args.mask_role,
        dataset=_dataset(args),
        manifest=args.manifest,
    )
    _emit({"stage": "prepare", "status": "sealed", **result})
    return 0


def _prepare_masks(args: argparse.Namespace) -> int:
    result = E.prepare_from_masks(
        args.masks,
        args.out,
        format=args.masks_format,
        inputs=_inputs(args),
        dataset=_dataset(args),
        manifest=args.manifest,
    )
    _emit({"stage": "prepare-masks", "status": "sealed", **result})
    return 0


def _score(args: argparse.Namespace) -> int:
    report = E.score(
        args.out,
        allow_gt=args.allow_gt,
        reference_baseline=args.baseline,
        report_path=args.report,
    )
    metrics = report["metrics"]
    _emit(
        {
            "stage": "score",
            "status": report["success"]["status"],
            "passed": report["success"]["passed"],
            "report_path": report["report_path"],
            "n_images": metrics["n_images"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "PQ": metrics["PQ"],
            "SQ": metrics["SQ"],
            "RQ": metrics["RQ"],
            "thing": {
                "recall": metrics["thing"]["recall"],
                "PQ": metrics["thing"]["PQ"],
            },
            "stuff": {
                "recall": metrics["stuff"]["recall"],
                "PQ": metrics["stuff"]["PQ"],
            },
            "reference_baseline": report["reference_baseline"].get("status"),
        }
    )
    return 0 if report["success"]["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_instance_eval.py",
        description=(
            "Model-independent instance evaluation: a directory of 12-field result "
            "documents in, canonical PQ and instance recall out. No model, no inference."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="validate, hash and seal predictions before GT")
    prepare.add_argument("--predictions", required=True, help="directory of per-sample prediction JSON")
    prepare.add_argument("--out", required=True, help="lane-owned directory for the seal")
    prepare.add_argument("--maps", default=None, help="optional directory of per-sample .npz masks")
    prepare.add_argument(
        "--masks",
        default=None,
        help="optional directory of per-sample prediction PNGs (<slug>.semantic.png/.instance.png)",
    )
    prepare.add_argument(
        "--masks-format", default="png", choices=("png",), help="format for --masks (default png)"
    )
    prepare.add_argument(
        "--mask-role",
        default="authoritative",
        choices=("authoritative", "diagnostic"),
        help=(
            "authoritative: JSON must equal the document re-derived from the masks; "
            "diagnostic: masks are a winner-take-all rendering of overlapping instances, "
            "sealed and measured, disagreements recorded, JSON scored (default authoritative)"
        ),
    )
    prepare.add_argument(
        "--inputs",
        default=None,
        help="JSON list of {sample_id, png_path[, svg_path]}; default is the pinned four TUNE samples",
    )
    prepare.add_argument(
        "--sample-set",
        default="fixed4",
        choices=("fixed4", "fit16"),
        help="fixed4 (default, exposed TUNE) or fit16 (capacity-16 FIT diagnostic, disjoint)",
    )
    prepare.add_argument("--dataset", default=None, help="label recorded in the seal/report")
    prepare.add_argument("--manifest", default=C.MANIFEST_PATH, help="public manifest path")
    prepare.set_defaults(handler=_prepare)

    masks = subparsers.add_parser(
        "prepare-masks",
        help="derive 12-field documents from prediction masks (png/npz), then seal",
    )
    masks.add_argument("--masks", required=True, help="directory of per-sample prediction masks")
    masks.add_argument("--out", required=True, help="lane-owned directory for the seal")
    masks.add_argument(
        "--masks-format", default="png", choices=("png", "npz"), help="mask format (default png)"
    )
    masks.add_argument("--inputs", default=None, help="optional JSON sample list")
    masks.add_argument(
        "--sample-set",
        default="fixed4",
        choices=("fixed4", "fit16"),
        help="fixed4 (default) or fit16 diagnostic",
    )
    masks.add_argument("--dataset", default=None, help="label recorded in the seal/report")
    masks.add_argument("--manifest", default=C.MANIFEST_PATH, help="public manifest path")
    masks.set_defaults(handler=_prepare_masks)

    score = subparsers.add_parser("score", help="verify the seal, read GT once, report metrics")
    score.add_argument("--out", required=True, help="directory holding pre_gt_seal.json")
    score.add_argument(
        "--allow-gt",
        action="store_true",
        help="explicit parent grant to read GT (required)",
    )
    score.add_argument("--report", default=None, help="report path inside the lane")
    score.add_argument(
        "--baseline",
        default=None,
        help="optional reference baseline JSON carrying value, source and normalisation",
    )
    score.set_defaults(handler=_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except C.GuardError as error:
        sys.stderr.write(json.dumps({"status": "error", "code": error.code, "detail": error.detail}) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
