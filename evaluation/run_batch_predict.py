"""CLI: PNG-only batch predictions, then (separately) the pre-GT seal.

    python run_batch_predict.py \
        --inputs parent_capacity16_inputs.json \
        --checkpoint model/checkpoints/direct_instance.pt \
        --device cuda --predictions-dir runs/fit16/predictions

    python run_instance_eval.py prepare \
        --predictions runs/fit16/predictions --out runs/fit16/seal --sample-set explicit \
        --dataset fit16-diagnostic

This stage writes prediction envelopes only.  It never reads SVG, ground truth
or any protected split, and it never loads two checkpoints: one
``model.infer.load_checkpoint`` call serves the whole batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instance_eval import batch, contract as C  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions-dir", required=True, help="empty output directory for prediction JSONs")
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint loaded exactly once")
    parser.add_argument("--device", default=batch.DEFAULT_DEVICE, help="torch device name (default cpu)")
    parser.add_argument("--inputs", default=None, help="explicit inputs JSON (model-owned sample list and order)")
    parser.add_argument(
        "--sample-set",
        default=None,
        choices=("fixed4", "fit16", "explicit"),
        help="pinned set when --inputs is omitted (default fixed4)",
    )
    parser.add_argument("--manifest", default=C.MANIFEST_PATH, help="public manifest path")
    parser.add_argument("--max-detections", type=int, default=None, help="override only with a reason; default = official config")
    parser.add_argument("--dense-diagnostic", action="store_true", help="also compute the score-winner diagnostic map")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sample_set = args.sample_set or ("explicit" if args.inputs else "fixed4")
    try:
        result = batch.run_batch(
            args.predictions_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            inputs=args.inputs,
            sample_set=sample_set,
            manifest=args.manifest,
            max_detections=args.max_detections,
            dense_diagnostic=args.dense_diagnostic,
        )
    except C.GuardError as error:
        print(json.dumps({"status": "error", "code": error.code, "detail": error.detail}), file=sys.stderr)
        return 2
    summary = {
        "status": result["status"],
        "predictions_dir": result["predictions_dir"],
        "sample_ids": result["sample_ids"],
        "n_predictions": result["n_predictions"],
        "checkpoint_loads": result["checkpoint_loads"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "seal_written": result["seal_written"],
        "gt_read": result["gt_read"],
        "next": result["next"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
