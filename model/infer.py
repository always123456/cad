"""Direct PNG -> boxes / classes / scores / independent masks -> official 12-field document.

This module never reads a ground-truth annotation.  It loads a checkpoint produced
by ``model.train``, runs one native-resolution PNG, and emits one canonical
document by vectorising each selected independent mask separately (``emitter=
"runs"``) and merging the results.  A dense score-winner raster is diagnostic
only and is never the submitted document.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import emit, model as detector

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferenceError(RuntimeError):
    pass


def load_png(path: str | os.PathLike[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return native CHW float32 image, ink bool HW, and source metadata.

    Alpha is composited over white exactly like the canonical project loader;
    ink is ``min(R,G,B) < 254/255`` after that composite.  No resizing happens
    here.
    """

    png_path = Path(path)
    if not png_path.is_file():
        raise InferenceError("PNG does not exist: %r" % str(png_path))
    with Image.open(png_path) as handle:
        rgba = np.asarray(handle.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    rgb = np.clip(rgb, 0.0, 1.0)
    image = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
    ink = (rgb.min(axis=2) < (254.0 / 255.0)).astype(np.bool_)
    metadata = {
        "path": str(png_path.resolve()),
        "sha256": _sha256_file(png_path),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "alpha_composited_on_white": True,
        "ink_threshold": "min(R,G,B) < 254/255 after white composite",
    }
    return image, ink, metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    checkpoint: str | os.PathLike[str],
    *,
    device: torch.device,
) -> Tuple[torch.nn.Module, detector.DetectorConfig, Dict[str, Any], Dict[str, Any]]:
    """Rebuild the architecture without weights, load state strictly, move to device."""

    import torch

    path = Path(checkpoint)
    if not path.is_file():
        raise InferenceError("checkpoint does not exist: %r" % str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise InferenceError("checkpoint has no model_state: %r" % str(path))
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise InferenceError("checkpoint has no serialised config")
    config = dataclasses.replace(
        detector.DetectorConfig.from_dict(raw_config),
        weights="none",
    )
    # Build without any pretrained download; the checkpoint supplies every value.
    model, info = detector.build_model(config, weights_file=None)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=True)
    if missing or unexpected:
        raise InferenceError(
            "strict checkpoint load mismatch: missing=%r unexpected=%r"
            % (list(missing), list(unexpected))
        )
    model.to(device)
    model.eval()
    detector.set_inference_transform(model, config)
    return model, config, info, payload


def predict_arrays(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    device: torch.device,
    score_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the detector and return native boxes, labels, scores and probabilities."""

    import torch

    if image.ndim != 3 or image.shape[0] != 3:
        raise InferenceError("image must be CHW 3xHxW")
    score_threshold = float(
        score_threshold if score_threshold is not None else getattr(model.roi_heads, "score_thresh", 0.05)
    )
    mask_threshold = float(
        mask_threshold if mask_threshold is not None else getattr(model, "direct_instance_config").mask_threshold
    )
    model.roi_heads.score_thresh = score_threshold
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise InferenceError("image contains non-finite values")
    model.eval()
    with torch.no_grad():
        outputs = model([tensor.to(device)])
    output = outputs[0]
    boxes = output.get("boxes", torch.zeros((0, 4))).detach().cpu().numpy().astype(np.float32)
    labels = output.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().numpy()
    scores = output.get("scores", torch.zeros((0,))).detach().cpu().numpy().astype(np.float32)
    masks = output.get("masks", torch.zeros((0, 1, image.shape[1], image.shape[2]))).detach().cpu().numpy()
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise InferenceError("detector masks must be Nx1xHxW, got %r" % (tuple(masks.shape),))
    if masks.shape[0] != len(labels):
        raise InferenceError("mask count %d != labels %d" % (masks.shape[0], len(labels)))
    if masks.shape[2:] != image.shape[1:]:
        raise InferenceError(
            "detector masks are %r but native image is %r"
            % (tuple(masks.shape[2:]), tuple(image.shape[1:]))
        )
    if np.any(labels < 1) or np.any(labels > 35):
        raise InferenceError("model label outside 1..35: %r" % (np.unique(labels).tolist(),))
    return {
        "boxes": boxes,
        "labels": labels.astype(np.int64),
        "scores": scores,
        "mask_probabilities": masks[:, 0],
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
    }


def predict_document(
    checkpoint: str | os.PathLike[str],
    png: str | os.PathLike[str],
    *,
    device_name: str = "cpu",
    score_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    ink_intersection: bool = False,
    make_dense_diagnostic: bool = False,
    compute_roundtrip: bool = True,
) -> Dict[str, Any]:
    """Full direct-inference path for a single PNG."""

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA requested but unavailable")
    image, ink, image_meta = load_png(png)
    model, config, info, payload = load_checkpoint(checkpoint, device=device)
    if max_detections is not None:
        model.roi_heads.detections_per_img = int(max_detections)
    result = predict_arrays(
        model,
        image,
        device=device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )
    probabilities = result["mask_probabilities"]
    masks = probabilities >= float(result["mask_threshold"])
    if max_detections is not None and masks.shape[0] > int(max_detections):
        masks = masks[: int(max_detections)]
        for key in ("boxes", "labels", "scores", "mask_probabilities"):
            result[key] = result[key][: int(max_detections)]

    removed_pixels: List[int] = []
    if ink_intersection:
        # Explicit opt-in only.  Geometric mask support is never silently gated.
        for index in range(masks.shape[0]):
            before = int(masks[index].sum())
            masks[index] &= ink
            removed_pixels.append(before - int(masks[index].sum()))
    class_ids = [int(value) - 1 for value in result["labels"]]
    emitted = emit.emit_instances(
        [masks[i] for i in range(masks.shape[0])],
        class_ids,
        [float(v) for v in result["scores"]],
        height=int(image.shape[1]),
        width=int(image.shape[2]),
        prediction_indices=list(range(masks.shape[0])),
        emitter="runs",
        make_dense_diagnostic=make_dense_diagnostic,
    )
    roundtrip = None
    if compute_roundtrip:
        roundtrip = emit.roundtrip_iou(
            emitted["expected_masks"],
            emitted["document"],
            height=int(image.shape[1]),
            width=int(image.shape[2]),
        )
    diagnostics = dict(emitted["diagnostics"])
    diagnostics.update(
        {
            "score_threshold": float(result["score_threshold"]),
            "mask_threshold": float(result["mask_threshold"]),
            "ink_intersection": bool(ink_intersection),
            "ink_removed_pixels": removed_pixels,
            "raw_detections": int(masks.shape[0]),
            "roundtrip_native_only": True,
            "canonical_1000px_scoring_measured": False,
            "no_ground_truth_read": True,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(checkpoint)),
            "model_weights_source": payload.get("weights_source", info.get("weights_source")),
            "configuration": config.to_dict(),
            "configured_mask_roi_pool": int(info.get("mask_roi_pool_output_size", 0)),
            "configured_mask_head_output": int(config.mask_roi_output_size),
            "configured_max_detections": int(model.roi_heads.detections_per_img),
            "native_image": image_meta,
        }
    )
    instances = [dataclasses.asdict(instance) for instance in emitted["instances"]]
    return {
        "document": emitted["document"],
        "instances": instances,
        "roundtrip": roundtrip,
        "diagnostics": diagnostics,
        "boxes_native": result["boxes"].tolist(),
        "labels_model": result["labels"].tolist(),
        "class_ids_canonical": [int(v) for v in result["labels"] - 1],
        "scores": [float(v) for v in result["scores"]],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_prediction(prediction: Dict[str, Any], output_path: str | os.PathLike[str]) -> None:
    path = Path(output_path)
    if path.exists():
        raise InferenceError("refusing to overwrite existing output: %r" % str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(prediction), handle, indent=2, allow_nan=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--ink-intersection", action="store_true")
    parser.add_argument("--dense-diagnostic", action="store_true")
    parser.add_argument("--no-roundtrip", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prediction = predict_document(
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
    write_prediction(prediction, args.output)
    diagnostics = prediction["diagnostics"]
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "instances": len(prediction["instances"]),
                "raw_detections": diagnostics["raw_detections"],
                "roundtrip_min_iou": (
                    prediction["roundtrip"]["min_iou"] if prediction["roundtrip"] else None
                ),
                "canonical_1000px_scoring_measured": False,
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
