"""Bounded CLI for the direct-instance Mask R-CNN lane.

Stages are deliberately separate:

* ``capacity`` -- decode the exact first-16 original FIT records, build targets,
  take deterministic native-384 crops, and write an inventory.  This measures
  memorisation capacity only, never submission recall.
* ``train`` -- build the pretrained topology, train with partial-label losses,
  and checkpoint.  It refuses to overwrite an existing checkpoint/log.
* ``predict`` -- one native PNG through ``model.infer``; no ground truth.

Run from ``outputs/recognition-direct-instance-20261114-01`` so the ``model``
package is importable::

    python -m model.train capacity --output-dir runs/capacity-...
    python -m model.train train --weights-file ... --output-dir runs/fit-...
    python -m model.train predict --checkpoint ... --png ... --output ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import data, infer, model as detector

OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    OUTPUTS_ROOT
    / "recognition-expanded-fit-20261114-01"
    / "manifest"
    / "expanded_train_manifest.json"
)
DEFAULT_SEED = detector.DEFAULT_SEED
DEFAULT_UPDATES = 600
DEFAULT_ACCUMULATE = 2
DEFAULT_CROPS = data.CAPACITY_CROPS


class TrainError(RuntimeError):
    pass


def _prepare_output_dir(path: str | os.PathLike[str], *, files: Sequence[str]) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = out / name
        if target.exists():
            raise TrainError("refusing to overwrite existing file: %r" % str(target))
    return out


def _seed_everything(seed: int) -> None:
    # Seed must be set BEFORE build_model so head initialisation is reproducible.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _crop_source(
    manifest: str | os.PathLike[str],
    *,
    limit: Optional[int],
    cache_size: int = 2,
) -> data.ExpandedFitSource:
    return data.ExpandedFitSource(manifest, limit=limit, cache_size=cache_size)


def run_capacity(args: argparse.Namespace) -> int:
    _seed_everything(int(args.seed))
    if int(args.crops) > len(data.CANONICAL_FIT16):
        raise TrainError(
            "capacity crops is bounded to the canonical first-%d list" % len(data.CANONICAL_FIT16)
        )
    source = _crop_source(args.manifest, limit=int(args.crops), cache_size=int(args.cache_size))
    report = data.capacity_report(
        source, crops=int(args.crops), seed=int(args.seed), size=detector.TRAIN_CROP_SIZE
    )
    report.update(
        {
            "full_image_submission_recall": {
                "measured": False,
                "why": "capacity is fixed-crop memorisation; full native-image recall requires inference plus evaluator",
            },
            "no_ground_truth_read_for_inference": True,
            "notes": [
                "The crop window is centered on the largest official instance with zero jitter.",
                "Keys whose bbox is clipped by the 384 window are retained with explicit retention metadata.",
                "Records are resolved from the expanded manifest's admitted fit order and checked against the canonical parent list; a mismatch refuses instead of silently substituting.",
            ],
        }
    )
    out = _prepare_output_dir(args.output_dir, files=("capacity.json",))
    path = out / "capacity.json"
    _write_json(path, report)
    print(
        json.dumps(
            {
                "capacity": str(path),
                "selected_records": report["selected_records"],
                "fit_inputs": [item["sample_id"] for item in report["fit_inputs"]],
                "metric_scope": report["metric_scope"],
                "submission_recall_measured": False,
            },
            allow_nan=False,
        )
    )
    return 0


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    backbone_parameters = list(model.backbone.parameters())
    return torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": 1e-4},
            {"params": backbone_parameters, "lr": 1e-5},
        ],
        weight_decay=0.01,
    )


def _cpu_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def run_train(args: argparse.Namespace) -> int:
    if int(args.updates) <= 0:
        raise TrainError("updates must be positive")
    if int(args.accumulate) <= 0:
        raise TrainError("accumulate must be positive")
    if args.weights == "coco" and not args.allow_download and not args.weights_file:
        raise TrainError(
            "refusing an implicit COCO download; pass --weights-file (asset owner) "
            "or explicitly --allow-download"
        )
    if args.weights_file and not Path(args.weights_file).is_file():
        raise TrainError("weights file does not exist: %r" % args.weights_file)

    out = _prepare_output_dir(
        args.output_dir,
        files=("checkpoint.pt", "training_log.jsonl", "train_report.json"),
    )
    seed = int(args.seed)
    _seed_everything(seed)

    config = detector.DetectorConfig(
        weights=str(args.weights),
        seed=seed,
        mask_target_mode=str(args.mask_target_mode),
        mask_target_aligned=bool(getattr(args, "mask_target_aligned", True)),
        mask_target_sampling_ratio=int(
            getattr(args, "mask_target_sampling_ratio", 2)
        ),
        mask_target_projection=str(
            getattr(args, "mask_target_projection", "legacy")
        ),
    )
    # Seed is set before construction; head initialisation is captured by config.seed.
    model, info = detector.build_model(config, weights_file=args.weights_file)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainError("CUDA requested but unavailable")
    model.to(device)

    source = _crop_source(
        args.manifest,
        limit=None if int(args.crops) <= 0 else int(args.crops),
        cache_size=int(args.cache_size),
    )
    if int(args.crops) > 0 and len(source) < int(args.crops):
        raise TrainError("manifest selected %d records, fewer than --crops" % len(source))
    optimizer = _optimizer(model)
    history: List[Dict[str, Any]] = []
    start = time.time()
    log_path = out / "training_log.jsonl"
    nonfinite = 0
    optimizer_steps = 0
    with log_path.open("w", encoding="utf-8") as log:
        for update in range(int(args.updates)):
            optimizer.zero_grad(set_to_none=True)
            aggregate: Dict[str, float] = {}
            counts: Dict[str, Any] = {}
            micro_meta: List[Dict[str, Any]] = []
            for micro in range(int(args.accumulate)):
                global_micro = update * int(args.accumulate) + micro
                index = global_micro % len(source)
                # Crop RNG is local to data.crop_for_update; torch RNG drives
                # anchor/ROI sampling and is intentionally seeded per micro-batch.
                torch.manual_seed(seed + global_micro)
                torch.cuda.manual_seed_all(seed + global_micro) if torch.cuda.is_available() else None
                crop = data.crop_for_update(
                    source,
                    index,
                    seed=seed,
                    update=global_micro,
                    jitter=int(args.jitter),
                    size=detector.TRAIN_CROP_SIZE,
                )
                target = detector.prepare_detector_targets(crop["target"], device=device)
                result = detector.training_losses(model, [crop["image"].to(device)], [target])
                losses = result["losses"]
                total = detector.total_loss(losses)
                if not bool(torch.isfinite(total)):
                    nonfinite += 1
                    raise TrainError(
                        "non-finite loss at update %d micro %d (not hidden)" % (update, micro)
                    )
                (total / float(args.accumulate)).backward()
                for name, value in losses.items():
                    aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu()) / int(args.accumulate)
                counts = result["counts"]
                micro_meta.append(crop["metadata"])
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise TrainError("non-finite gradient norm at update %d (not hidden)" % update)
            optimizer.step()
            optimizer_steps += 1
            record = {
                "update": int(update),
                "optimizer_steps": int(optimizer_steps),
                "losses": aggregate,
                "total": float(sum(aggregate.values())),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "counts": counts,
                "crops": micro_meta,
            }
            history.append(record)
            if len(history) > 50:
                history = history[-50:]
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if (update + 1) % int(args.log_every) == 0:
                print(
                    json.dumps(
                        {
                            "update": update + 1,
                            "updates": int(args.updates),
                            "total": record["total"],
                            "loss_mask": aggregate.get("loss_mask"),
                            "roi_positive": counts.get("roi_positive"),
                            "seconds": round(time.time() - start, 1),
                        },
                        allow_nan=False,
                    )
                )
            if args.save_every and (update + 1) % int(args.save_every) == 0:
                _save_checkpoint(
                    out,
                    model,
                    optimizer,
                    config,
                    info,
                    update + 1,
                    optimizer_steps,
                    source,
                    history,
                    args,
                    name="checkpoint-step-%06d.pt" % (update + 1),
                )

    checkpoint_path = _save_checkpoint(
        out,
        model,
        optimizer,
        config,
        info,
        int(args.updates),
        optimizer_steps,
        source,
        history,
        args,
    )
    report = {
        "stage": "train",
        "checkpoint": str(checkpoint_path),
        "updates": int(args.updates),
        "optimizer_steps": int(optimizer_steps),
        "accumulate": int(args.accumulate),
        "batch_size_per_forward": 1,
        "crop_size": detector.TRAIN_CROP_SIZE,
        "jitter": int(args.jitter),
        "seed": seed,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": source.manifest_sha256,
        "records_sha256": source.records_sha256,
        "selected_records": len(source),
        "nonfinite": int(nonfinite),
        "model_info": info,
        "config": config.to_dict(),
        "final_loss": history[-1]["losses"] if history else {},
        "metric_separation": {
            "fit_crop_memorisation": "training losses above",
            "tune": "not contacted",
            "full_image_submission_recall": "not measured here",
        },
    }
    _write_json(out / "train_report.json", report)
    print(json.dumps({"checkpoint": str(checkpoint_path), "updates": int(args.updates)}, allow_nan=False))
    return 0


def _save_checkpoint(
    out: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: detector.DetectorConfig,
    info: Dict[str, Any],
    update: int,
    optimizer_steps: int,
    source: data.ExpandedFitSource,
    history: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    name: str = "checkpoint.pt",
) -> Path:
    path = out / str(name)
    payload = {
        "schema": "cadvec.direct-instance-maskrcnn/1",
        "config": config.to_dict(),
        "model_info": info,
        "weights_source": info.get("weights_source"),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "update": int(update),
        "optimizer_steps": int(optimizer_steps),
        "seed": int(args.seed),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": source.manifest_sha256,
        "records_sha256": source.records_sha256,
        "selected_records": len(source),
        "history_tail": history[-10:],
        "contract": {
            "labels": "canonical class + 1; background 0; 36 outputs",
            "mask_roi_pool": int(info.get("mask_roi_pool_output_size", 0)),
            "mask_head_output": int(config.mask_roi_output_size),
            "train_transform": [config.train_min_size, config.train_max_size],
            "inference_transform": [config.inference_min_size, config.inference_max_size],
            "bn_running_stats": "frozen; affine trainable",
        },
    }
    torch.save(payload, path)
    return path


def run_predict(args: argparse.Namespace) -> int:
    prediction = infer.predict_document(
        args.checkpoint,
        args.png,
        device_name=args.device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        max_detections=args.max_detections,
        ink_intersection=bool(args.ink_intersection),
        make_dense_diagnostic=bool(args.dense_diagnostic),
        compute_roundtrip=not bool(args.no_roundtrip),
    )
    infer.write_prediction(prediction, args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "instances": len(prediction["instances"]),
                "roundtrip_min_iou": (
                    prediction["roundtrip"]["min_iou"] if prediction["roundtrip"] else None
                ),
            },
            allow_nan=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
        p.add_argument("--output-dir", required=True)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    cap = sub.add_parser("capacity")
    common(cap)
    cap.add_argument("--crops", type=int, default=DEFAULT_CROPS)
    cap.add_argument(
        "--cache-size",
        type=int,
        default=1,
        help="opt-in decoded-image/target LRU size; use 16 only after a measured memory preflight",
    )
    cap.set_defaults(func=run_capacity)

    train = sub.add_parser("train")
    common(train)
    train.add_argument("--weights", choices=("none", "coco"), default="none")
    train.add_argument("--weights-file", default=None)
    train.add_argument("--allow-download", action="store_true")
    train.add_argument("--crops", type=int, default=DEFAULT_CROPS)
    train.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    train.add_argument("--accumulate", type=int, default=DEFAULT_ACCUMULATE)
    train.add_argument("--jitter", type=int, default=detector.TRAIN_JITTER)
    train.add_argument("--mask-target-mode", choices=("soft", "hard"), default="soft")
    train.add_argument(
        "--mask-target-aligned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "ROIAlign aligned convention for GT/domain mask targets; defaults to "
            "the legacy True. Use --no-mask-target-aligned for a controlled diagnostic."
        ),
    )
    train.add_argument(
        "--mask-target-sampling-ratio",
        type=int,
        choices=(-1, 2),
        default=2,
        help="ROIAlign sampling ratio for GT/domain mask targets (-1 or 2)",
    )
    train.add_argument(
        "--mask-target-projection",
        choices=("legacy", "paste_consistent"),
        default="legacy",
        help=(
            "GT/domain ROI source; legacy keeps aligned/ratio, paste_consistent "
            "derives both from the exact paste geometry with effective False/-1"
        ),
    )
    train.add_argument("--device", default="cpu")
    train.add_argument("--cache-size", type=int, default=2)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every", type=int, default=0)
    train.set_defaults(func=run_train)

    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--png", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--score-threshold", type=float, default=None)
    predict.add_argument("--mask-threshold", type=float, default=None)
    predict.add_argument("--max-detections", type=int, default=None)
    predict.add_argument("--ink-intersection", action="store_true")
    predict.add_argument("--dense-diagnostic", action="store_true")
    predict.add_argument("--no-roundtrip", action="store_true")
    predict.set_defaults(func=run_predict)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
