"""PNG-only batch prediction runner for the direct-instance lane.

One checkpoint load, one native-resolution PNG at a time, one official 12-field
document per sample:

    load_checkpoint(once) -> per PNG: load_png / predict_arrays / emit_instances

The emitted document is the independent-mask document from the shared emitter
(``emitter="runs"``): every selected mask is vectorised on its own in native
pixel space and the canonical viewBox affine is applied inside the emitter.
Overlapping instances therefore survive and Stuff classes merge per class.
There is no connected-component post-processing and no ink intersection.

This module never reads SVG, ground truth or any protected split, and it never
calls :func:`instance_eval.evaluator.prepare`: it writes prediction envelopes
only.  Every prediction is complete before the caller may seal them.

The model dependencies are injected.  Production defaults import
``model.infer``/``model.emit`` lazily, so importing this module (and its tests)
on a CPU-only box stays torch-free.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

from . import contract as C

DEFAULT_DEVICE = "cpu"
PNG_SUFFIXES = (".png",)
ENVELOPE_FIELDS = ("schema", "sample_id", "document", "diagnostics", "source")

# loader(checkpoint, device) -> context; context carries the model and device.
Loader = Callable[[str, str], Mapping[str, Any]]
# predict(context, image) -> {labels 1..35, scores, mask_probabilities NxHxW,
#                              mask_threshold, score_threshold}
Predict = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]
# emit(masks, class_ids, scores, height, width, dense) -> {document, instances, diagnostics, ...}
Emit = Callable[..., Mapping[str, Any]]
# png_loader(path) -> (image CHW float32, ink bool HW, metadata)
PngLoader = Callable[[str], tuple[Any, Any, Mapping[str, Any]]]


# --------------------------------------------------------------------------- #
# default adapters (lazy: nothing here imports torch at module import time)
# --------------------------------------------------------------------------- #
def _model_root() -> str:
    """The lane directory that holds ``model/`` (the parent of ``evaluation/``)."""

    parent = os.path.dirname(str(C.LANE_ROOT))
    if os.path.isdir(os.path.join(parent, "model")):
        return parent
    return str(C.LANE_ROOT)


def _model_module(name: str) -> Any:
    import importlib
    import sys

    lane_root = _model_root()
    if lane_root not in sys.path:
        sys.path.insert(0, lane_root)
    try:
        return importlib.import_module("model." + name)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise C.GuardError(
            "model_unavailable",
            "cannot import model.%s from %s: %s" % (name, lane_root, exc),
        ) from None


def _infer_module() -> Any:
    """``model.infer`` (imports torch); only touched when inference is needed."""

    return _model_module("infer")


def _emit_module() -> Any:
    """``model.emit``; numpy + cadvec only, so it works without torch."""

    return _model_module("emit")


def default_loader(checkpoint: str, device: str) -> dict[str, Any]:
    """Load the checkpoint exactly once and return the shared context."""

    infer = _infer_module()
    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise C.GuardError("device_unavailable", "CUDA requested but unavailable")
    model, config, info, payload = infer.load_checkpoint(checkpoint, device=target)
    return {
        "infer": infer,
        "model": model,
        "config": config,
        "info": info,
        "payload": payload,
        "device": target,
        "max_detections": int(getattr(model.roi_heads, "detections_per_img", 0) or 0),
    }


def default_predict(context: Mapping[str, Any], image: Any) -> dict[str, Any]:
    """Run the detector on one native image with the official config thresholds.

    ``run_batch`` may override a threshold explicitly; ``None`` keeps the values
    the checkpoint/config declares (never tuned here).
    """

    return context["infer"].predict_arrays(
        context["model"],
        image,
        device=context["device"],
        score_threshold=context.get("score_threshold"),
        mask_threshold=context.get("mask_threshold"),
    )


def default_emit(
    masks: Sequence[Any],
    class_ids: Sequence[int],
    scores: Sequence[float],
    *,
    height: int,
    width: int,
    dense: bool = False,
) -> dict[str, Any]:
    """Vectorise independent masks with the shared emitter (no copies)."""

    emit = _emit_module()
    return emit.emit_instances(
        list(masks),
        [int(value) for value in class_ids],
        [float(value) for value in scores],
        height=int(height),
        width=int(width),
        prediction_indices=list(range(len(masks))),
        emitter="runs",
        make_dense_diagnostic=bool(dense),
    )


def default_png_loader(path: str) -> tuple[Any, Any, Mapping[str, Any]]:
    infer = _infer_module()
    return infer.load_png(path)


# --------------------------------------------------------------------------- #
# inputs and output guards
# --------------------------------------------------------------------------- #
def resolve_inputs(
    inputs: Any = None,
    *,
    sample_set: str = "fixed4",
    manifest: str | os.PathLike[str] = C.MANIFEST_PATH,
) -> list[dict[str, Any]]:
    """Model-owned explicit inputs, or a pinned set from the public manifest."""

    if inputs is not None:
        records = C.normalize_inputs(inputs, manifest=manifest)
        if sample_set not in ("explicit", "fixed4", "fit16"):
            raise C.GuardError("sample_set", "unknown sample set %r" % (sample_set,))
        return records
    if sample_set == "fixed4":
        return C.default_inputs(manifest)
    if sample_set == "fit16":
        return C.fit_inputs(manifest)
    if sample_set == "explicit":
        raise C.GuardError("inputs_missing", "sample_set 'explicit' needs an inputs file")
    raise C.GuardError("sample_set", "unknown sample set %r" % (sample_set,))


def _refuse_existing_outputs(predictions_dir: str) -> str:
    target = os.path.abspath(predictions_dir)
    C.assert_no_sealed(target, "predictions_dir")
    if os.path.isfile(target):
        raise C.GuardError("predictions_exist", "predictions-dir is a file: %s" % target)
    if os.path.isdir(target):
        existing = sorted(os.listdir(target))
        if existing:
            raise C.GuardError(
                "predictions_exist",
                "predictions-dir already holds %d entries: %s" % (len(existing), target),
            )
    return target


def _png_target(record: Mapping[str, Any]) -> str:
    sample_id = str(record["sample_id"])
    path = str(record["png_path"])
    if not path.lower().endswith(PNG_SUFFIXES):
        raise C.GuardError(
            "batch_png_required", "%s: prediction input must be a PNG, got %s" % (sample_id, path)
        )
    if str(path).lower().endswith(".svg"):
        raise C.GuardError("batch_png_required", "%s: SVG input refused: %s" % (sample_id, path))
    if not os.path.isfile(path):
        raise C.GuardError("png_missing", "%s: %s" % (sample_id, path))
    C.assert_no_sealed(path, "png")
    return path


def _outside_viewbox(document: Mapping[str, Any], tolerance: float = 1e-6) -> int:
    """Count primitives with a vertex outside the canonical 0..100 viewBox.

    Mask content that was letterboxed out of the viewBox window transforms to
    coordinates outside the canvas; the shared emitter does not clamp it, so the
    batch records it instead of silently shrinking the document.
    """

    low, high = -tolerance, 100.0 + tolerance
    outside = 0
    for args in document.get("args") or []:
        for value in list(args)[0::2] + list(args)[1::2]:
            if float(value) < low or float(value) > high:
                outside += 1
                break
    return outside


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def run_batch(
    predictions_dir: str | os.PathLike[str],
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    device: str = DEFAULT_DEVICE,
    inputs: Any = None,
    sample_set: str = "fixed4",
    manifest: str | os.PathLike[str] = C.MANIFEST_PATH,
    max_detections: int | None = None,
    score_threshold: float | None = None,
    mask_threshold: float | None = None,
    dense_diagnostic: bool = False,
    loader: Loader | None = None,
    predict: Predict | None = None,
    emit: Emit | None = None,
    png_loader: PngLoader | None = None,
    trace: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Write one prediction envelope per PNG, then stop (no seal, no GT read)."""

    trace = trace if trace is not None else []
    records = resolve_inputs(inputs, sample_set=sample_set, manifest=manifest)
    if not records:
        raise C.GuardError("no_samples", "batch runner received no inputs")
    targets = [str(record["sample_id"]) for record in records]
    if len(set(targets)) != len(targets):
        raise C.GuardError("inputs_duplicate", "duplicate sample ids in batch inputs")

    out = _refuse_existing_outputs(str(predictions_dir))
    paths = [_png_target(record) for record in records]

    loader = loader or default_loader
    predict = predict or default_predict
    emit = emit or default_emit
    png_loader = png_loader or default_png_loader

    if checkpoint is None:
        raise C.GuardError("checkpoint_missing", "the batch runner needs --checkpoint")
    checkpoint_path = os.path.abspath(str(checkpoint))
    C.assert_no_sealed(checkpoint_path, "checkpoint")
    if not os.path.isfile(checkpoint_path):
        raise C.GuardError("checkpoint_missing", checkpoint_path)
    checkpoint_sha256 = C.sha256_file(checkpoint_path)

    # ONE load for the whole batch.
    context = dict(loader(checkpoint_path, str(device)))
    context.setdefault("score_threshold", score_threshold)
    context.setdefault("mask_threshold", mask_threshold)
    if score_threshold is not None:
        context["score_threshold"] = float(score_threshold)
    if mask_threshold is not None:
        context["mask_threshold"] = float(mask_threshold)
    trace.append(("checkpoint_loaded", checkpoint_path))
    batch_max = max_detections
    if batch_max is None:
        batch_max = context.get("max_detections")
    batch_max = int(batch_max) if batch_max else None

    envelopes: list[tuple[str, dict[str, Any]]] = []
    predictions: list[dict[str, Any]] = []
    for record, png_path in zip(records, paths):
        sample_id = str(record["sample_id"])
        image, _ink, meta = png_loader(png_path)
        trace.append(("png_read", png_path))
        result = predict(context, image)
        labels = [int(value) for value in result["labels"]]
        scores = [float(value) for value in result["scores"]]
        probabilities = result["mask_probabilities"]
        threshold = float(
            result.get("mask_threshold")
            if result.get("mask_threshold") is not None
            else mask_threshold
        )
        if mask_threshold is not None:
            threshold = float(mask_threshold)
        if len(labels) != len(scores) or len(labels) != len(probabilities):
            raise C.GuardError(
                "batch_detection_mismatch",
                "%s: %d labels, %d scores, %d masks"
                % (sample_id, len(labels), len(scores), len(probabilities)),
            )
        if any(label < 1 or label > 35 for label in labels):
            raise C.GuardError("batch_label_range", "%s: labels %r" % (sample_id, labels))
        masks = [probabilities[index] >= threshold for index in range(len(labels))]
        if batch_max is not None and len(masks) > batch_max:
            masks = masks[:batch_max]
            labels = labels[:batch_max]
            scores = scores[:batch_max]
        height, width = int(image.shape[1]), int(image.shape[2])
        emitted = emit(
            masks,
            [label - 1 for label in labels],
            scores,
            height=height,
            width=width,
            dense=bool(dense_diagnostic),
        )
        document = emitted["document"]
        C.validate_prediction_document(document, sample_id)
        outside = _outside_viewbox(document)
        trace.append(("predicted", sample_id))
        envelope = {
            "schema": C.SCHEMA_PREDICTION,
            "sample_id": sample_id,
            "document": document,
            "diagnostics": dict(emitted.get("diagnostics") or {}),
            "source": {
                "origin": "batch_predict",
                "png_path": png_path,
                "png_sha256": C.sha256_file(png_path),
                "native_size": {"height": height, "width": width},
                "png_meta": dict(meta or {}),
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "device": str(device),
                "emitter": "runs",
                "instance_masks": "independent",
                "ink_intersection": False,
                "connected_components": False,
                "score_threshold": result.get("score_threshold"),
                "mask_threshold": threshold,
                "max_detections": batch_max,
                "dense_diagnostic_only": True,
                "out_of_viewbox_primitives": outside,
                "no_ground_truth_read": True,
                "no_svg_read": True,
            },
        }
        envelopes.append((sample_id, envelope))
        predictions.append(
            {
                "sample_id": sample_id,
                "png_path": png_path,
                "png_sha256": envelope["source"]["png_sha256"],
                "document_sha256": C.sha256_value(document),
                "primitives": len(document["semanticIds"]),
                "instances": len(emitted.get("instances") or []),
                "classes": sorted({int(value) for value in document["semanticIds"]}),
                "detections": len(scores),
                "out_of_viewbox_primitives": outside,
            }
        )

    # Every prediction is complete in memory before any file is written, and all
    # target paths are checked before the first write: all-or-nothing.
    os.makedirs(out, exist_ok=True)
    file_targets = [os.path.join(out, C.slug(sample_id) + ".json") for sample_id in targets]
    clashes = [path for path in file_targets if os.path.exists(path)]
    if clashes:
        raise C.GuardError("prediction_exists", "refusing to overwrite %r" % clashes)
    for (sample_id, envelope), path in zip(envelopes, file_targets):
        C.write_json(path, envelope)
        trace.append(("prediction_written", path))

    return {
        "schema": "cadvec.direct_instance.batch_predict.v1",
        "status": "predictions_written",
        "predictions_dir": out,
        "sample_ids": targets,
        "n_predictions": len(file_targets),
        "predictions": predictions,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "checkpoint_loads": 1,
        "png_reads": len(paths),
        "max_detections": batch_max,
        "dense_diagnostic": bool(dense_diagnostic),
        "seal_written": False,
        "gt_read": False,
        "svg_read": False,
        "next": "instance_eval.evaluator.prepare(predictions_dir, out_dir=...)",
    }


__all__ = [
    "DEFAULT_DEVICE",
    "PNG_SUFFIXES",
    "default_emit",
    "default_loader",
    "default_png_loader",
    "default_predict",
    "resolve_inputs",
    "run_batch",
]
