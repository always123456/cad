"""Shared fixtures for the delivery CLI tests (CPU only, no torch, no checkpoint)."""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cad_mrcnn_delivery as delivery  # noqa: E402


def args(**overrides):
    """A Namespace with the CLI defaults, overridable per test."""

    base = {
        "png": None,
        "inputs": None,
        "out": None,
        "checkpoint": None,
        "score_threshold": None,
        "max_detections": None,
        "per_image_cap": None,
        "nms_iou": None,
        "mask_threshold": None,
        "refine_mask": None,
        "refine_ink": None,
        "refine_ratio": None,
        "device": "cpu",
        "sample_id": None,
        "group_id": None,
        "selection_report": None,
        "dry_run": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def touch(path, payload=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        if payload is None:
            handle.write(b"fixture")
        elif isinstance(payload, bytes):
            handle.write(payload)
        else:
            handle.write(payload.encode("utf-8"))
    return path


def fake_context():
    """A stand-in for instance_eval.batch.default_loader's context."""

    from types import SimpleNamespace

    heads = SimpleNamespace(score_thresh=0.05, nms_thresh=0.5, detections_per_img=300)
    return {
        "infer": None,
        "model": SimpleNamespace(roi_heads=heads),
        "config": {"box_score_thresh": 0.05},
        "info": {},
        "payload": {},
        "device": "cpu",
        "max_detections": 300,
    }


def fake_loader(checkpoint, device):
    return fake_context()


def flat_png_loader(*_args, **_kwargs):
    """A PNG loader stand-in: a small square CHW float image and no ink."""

    import numpy as np

    image = np.zeros((3, 32, 32), dtype=np.float32)
    image[:, 4:12, 4:12] = 1.0
    ink = np.zeros((32, 32), dtype=bool)
    ink[4:12, 4:12] = True
    return image, ink, {"path": "fake", "width": 32, "height": 32, "fixture": True}


def detections(*specs):
    """Build a predict() result from (label, score, (y0, y1, x0, x1)) specs."""

    import numpy as np

    masks = []
    for _label, _score, box in specs:
        mask = np.zeros((32, 32), dtype=np.float32)
        y0, y1, x0, x1 = box
        mask[y0:y1, x0:x1] = 1.0
        masks.append(mask)
    probabilities = (
        np.stack(masks) if masks else np.zeros((0, 32, 32), dtype=np.float32)
    )
    return {
        "boxes": np.zeros((len(specs), 4), dtype=np.float32),
        "labels": np.array([spec[0] for spec in specs], dtype=np.int64),
        "scores": np.array([spec[1] for spec in specs], dtype=np.float32),
        "mask_probabilities": probabilities,
        "score_threshold": 0.05,
        "mask_threshold": 0.5,
    }


# label 1 twice (a duplicate that NMS must drop), label 33 -> canonical 32 (Stuff),
# and one weak detection below a raised threshold. A square 32x32 canvas keeps the
# emitter's viewBox affine exact (the lane's emit assumes a square raster, and the
# production path resizes to 1000x1000).
DEFAULT_SPECS = (
    (1, 0.90, (4, 12, 4, 12)),
    (1, 0.85, (4, 12, 4, 12)),
    (33, 0.60, (16, 24, 16, 24)),
    (2, 0.10, (18, 26, 2, 10)),
)


def default_predict(_context, _image):
    return detections(*DEFAULT_SPECS)


def fake_run(out, checkpoint, png=None, inputs=None, predict=None, **overrides):
    """Run the delivery CLI with fake model access (no torch, no checkpoint load)."""

    namespace = args(png=png, inputs=inputs, out=out, checkpoint=checkpoint, **overrides)
    return delivery.run_delivery(
        namespace,
        loader=fake_loader,
        predict=predict or default_predict,
        png_loader=flat_png_loader,
    )

