#!/usr/bin/env python3
"""Bounded, offline, PNG-only tiled inference CLI (README section 4 JSON).

One explicit checkpoint plus one PNG file or a flat directory of PNGs produce
``<results>/<id>.json``, where ``<id>`` is exactly the five-character PNG stem.
Every document has exactly the twelve ``recognition_submission.RESULT_FIELDS``
fields and passes the local ``validate_document`` contract before it is
written; the written bytes are read back and re-validated as a batch.

Reused, unmodified project APIs (no re-implementation of model, decode,
vectorizer or contract):

* ``recognition_pretrained_pilot.load_checkpoint_model`` - the existing
  checkpoint loader (schema ``cadvec.recognition.pretrained_pilot_checkpoint/1``,
  ``role == "fit"``, 35-class pilot head). It builds the model with
  ``pretrained=False``, so no weights file is downloaded or resolved.
* ``recognition_pretrained_fullfit.predict_tiled_image`` - source-resolution
  overlap-averaged tiling at the training window, ``tile_size=256``,
  ``stride=128``, right/bottom white padding, never a resize.
* ``recognition_raster_infer._read_png`` / ``_check_png_input`` - PNG-only
  decode over white plus the existing input guard (``max_file_bytes=64 MiB``,
  ``max_pixels=40M`` from ``png_frontend.FrontendConfig``). The guard bounds
  input only; it is not a GPU memory guarantee.
* ``recognition_raster_infer.instance_ids`` - per-class 8-connected Thing
  components, Stuff left at ``-1``.
* ``recognition_raster_data._image_ink_support`` - min(R,G,B) rule, so pixels
  outside visible ink become ignore35 before components.
* ``recognition_raster_infer._viewbox_affine`` - the nominal
  ``xMidYMid meet`` PNG -> ``viewBox 0 0 100 100`` mapping. That mapping is an
  explicit unconfirmed assumption: inference receives no SVG metadata.
* ``png_raster_vectorize.vectorize_masks`` - the twelve-field line vectorizer.
* ``recognition_submission.validate_document`` - the local contract check.

Hard policies:

* No GT, no SVG sidecar, no label file and no manifest is read. Only ``*.png``
  entries of the requested input are opened; every other entry is ignored and
  counted.
* No download, no network call, no weight bundling, no data enumeration.
* No retry: each image is attempted exactly once and a failure is reported,
  never re-run with different settings.
* Deterministic: inputs are sorted by output id, JSON is written with
  ``sort_keys`` and ``\\n`` line endings, and no timestamp enters a result.
* A results directory stays clean: it may contain only ``<id>.json`` for the
  ids of the current batch. Anything else is a hard error before inference.
* The CLI does not rank or prefer any checkpoint. It requires the final
  checkpoint that the existing metadata marks as final; ``diagnostic_only``,
  non-final epochs, non-``fit`` roles and non-35-class payloads are rejected.

Official evaluator details, primitive-ID mapping and the official platform
schema remain unconfirmed; this module only enforces the local contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import png_raster_vectorize as vectorizer
from . import recognition_pretrained_fullfit as fullfit
from . import recognition_pretrained_pilot as pilot
from . import recognition_raster_data as raster_data
from . import recognition_raster_infer as raster_infer
from . import recognition_submission as submission
from .recognition_contract import RecognitionError

RUN_SCHEMA = "cadvec.recognition.tiled_submission_run/1"
METADATA_SCHEMA = "cadvec.recognition.tiled_submission_metadata/1"
DECODER = "semantic_cc8_v1"

# Fixed tiling contract: the inference window equals the 256 px training crop.
# The CLI exposes no tile/stride flags so the run cannot drift from the
# checkpoint's training geometry.
TILE_SIZE = fullfit.TILE_SIZE
TILE_STRIDE = fullfit.TILE_STRIDE
if (TILE_SIZE, TILE_STRIDE) != (256, 128):  # pragma: no cover - contract guard
    raise RuntimeError(
        "tiling contract changed upstream: expected tile 256 / stride 128, got "
        f"{TILE_SIZE}/{TILE_STRIDE}; refusing to guess a new geometry"
    )

EXPECTED_NUM_CLASSES = int(pilot.NUM_CLASSES)
IGNORE_CLASS = int(raster_data.IGNORE_CLASS)
VIEWBOX = tuple(submission.VIEWBOX)
RESULT_FIELDS = tuple(submission.RESULT_FIELDS)
DEFAULT_TOL_PX = 0.5
ID_PATTERN = re.compile(r"^[0-9A-Za-z]{5}$")
FINAL_SELECTION = "final_epoch_only"

INPUT_MAX_FILE_BYTES = int(raster_infer.INPUT_MAX_FILE_BYTES)
INPUT_MAX_PIXELS = int(raster_infer.INPUT_MAX_PIXELS)
INPUT_LIMITS_SOURCE = raster_infer._INPUT_LIMITS_SOURCE

VIEWBOX_MAPPING = (
    "uniform xMidYMid meet: s=min(W,H)/100, tx=(W-100s)/2, ty=(H-100s)/2, "
    "svg=(px-t)/s"
)
VIEWBOX_MAPPING_STATUS = "unconfirmed assumption; no SVG metadata at inference"

__all__ = [
    "RUN_SCHEMA",
    "METADATA_SCHEMA",
    "TILE_SIZE",
    "TILE_STRIDE",
    "RESULT_FIELDS",
    "TiledSubmissionError",
    "validate_checkpoint_payload",
    "checkpoint_provenance",
    "load_submission_checkpoint",
    "predict_document",
    "collect_inputs",
    "assign_output_ids",
    "run_job",
    "build_parser",
    "main",
]


class TiledSubmissionError(ValueError):
    """A bounded-inference contract violation with a readable reason."""

    code = "tiled_submission_error"


# ---------------------------------------------------------------------------
# checkpoint contract
# ---------------------------------------------------------------------------


def validate_checkpoint_payload(
    payload: Mapping[str, Any], path: str = "<checkpoint>"
) -> None:
    """Reject a checkpoint the existing metadata marks as unusable.

    Schema and ``role == "fit"`` are already enforced by
    ``pilot.load_checkpoint_model``. This adds the submission-side rules:
    35 classes, not a diagnostic snapshot, and the final epoch when the payload
    carries the metadata to decide it. Absent metadata is not invented.
    """

    problems: List[str] = []

    num_classes = payload.get("num_classes")
    if num_classes is not None:
        try:
            classes = int(num_classes)
        except (TypeError, ValueError):
            problems.append(f"num_classes_not_integer:{num_classes!r}")
        else:
            if classes != EXPECTED_NUM_CLASSES:
                problems.append(
                    f"num_classes={classes}!={EXPECTED_NUM_CLASSES}"
                )

    if bool(payload.get("diagnostic_only")):
        problems.append("diagnostic_only=true")

    if "selection_eligible" in payload and not bool(
        payload.get("selection_eligible")
    ):
        problems.append("selection_eligible=false")

    selection = payload.get("selection")
    if selection is not None and selection != FINAL_SELECTION:
        problems.append(f"selection={selection!r}")

    epoch = payload.get("epoch")
    total = payload.get("total_epochs", payload.get("epochs"))
    if epoch is not None and total is not None:
        try:
            epoch_value, total_value = int(epoch), int(total)
        except (TypeError, ValueError):
            problems.append(f"epoch_metadata_invalid:{epoch!r}/{total!r}")
        else:
            if epoch_value != total_value:
                problems.append(
                    f"epoch={epoch_value}!=total_epochs={total_value}"
                )

    if problems:
        raise TiledSubmissionError(
            "checkpoint_rejected:" + ",".join(problems) + f":{path}"
        )


def checkpoint_provenance(
    payload: Mapping[str, Any], path: str
) -> Dict[str, Any]:
    """Compact, JSON-safe provenance of the accepted checkpoint."""

    return {
        "path": os.path.abspath(path),
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "role": payload.get("role"),
        "epoch": payload.get("epoch"),
        "total_epochs": payload.get("total_epochs", payload.get("epochs")),
        "num_classes": payload.get("num_classes"),
        "model_id": payload.get("model_id"),
        "diagnostic_only": payload.get("diagnostic_only"),
        "selection": payload.get("selection"),
        "selection_eligible": payload.get("selection_eligible"),
    }


def load_submission_checkpoint(
    checkpoint_path: str, *, device: str = "cpu"
) -> Tuple[Any, Dict[str, Any]]:
    """Load one explicit checkpoint through the existing pilot loader.

    ``pretrained=False`` inside that loader means nothing is downloaded and no
    weights file is resolved. Any loader rejection is surfaced as
    ``checkpoint_rejected`` so the CLI can exit non-zero with one clear line.
    """

    if not os.path.isfile(checkpoint_path):
        raise TiledSubmissionError(f"checkpoint_not_found:{checkpoint_path}")
    try:
        model, payload = pilot.load_checkpoint_model(checkpoint_path, device=device)
    except (RecognitionError, OSError) as exc:
        raise TiledSubmissionError(
            f"checkpoint_rejected:{type(exc).__name__}:{exc}:{checkpoint_path}"
        ) from exc
    validate_checkpoint_payload(payload, checkpoint_path)
    return model, checkpoint_provenance(payload, checkpoint_path)


# ---------------------------------------------------------------------------
# one image -> one validated document
# ---------------------------------------------------------------------------


def _check_document(document: Any) -> None:
    if not isinstance(document, Mapping):
        raise TiledSubmissionError("document_not_object")
    fields = set(document)
    expected = set(RESULT_FIELDS)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        raise TiledSubmissionError(
            f"document_fields_invalid:missing={missing}:extra={extra}"
        )
    errors = submission.validate_document(document)
    if errors:
        raise TiledSubmissionError("document_invalid:" + ";".join(errors))


def predict_document(
    png_path: str,
    model: Any,
    *,
    device: str = "cpu",
    tol_px: float = DEFAULT_TOL_PX,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> Dict[str, Any]:
    """Tiled predict one PNG and return its document plus provenance.

    ``png_path`` is opened only for decoding; no sibling SVG, label, manifest
    or ground truth is read. The image is never resized: the tiled logits are
    restored to the native raster size before argmax.
    """

    import torch

    tol = float(tol_px)
    if not math.isfinite(tol) or tol < 0.0:
        raise TiledSubmissionError(f"tol_px_invalid:{tol_px!r}")

    image = raster_infer._read_png(png_path)
    height, width = int(image.shape[1]), int(image.shape[2])
    tiled = fullfit.predict_tiled_image(
        model, image, device=device, tile_size=int(tile_size), stride=int(stride)
    )
    logits = tiled["semantic_logits"]
    expected = (1, EXPECTED_NUM_CLASSES, height, width)
    if tuple(logits.shape) != expected:
        raise TiledSubmissionError(
            f"logits_shape_mismatch:{tuple(logits.shape)}!={expected}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise TiledSubmissionError("logits_non_finite")

    semantic = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.int32)
    support = raster_data._image_ink_support(image)
    semantic[~support] = IGNORE_CLASS
    instances = raster_infer.instance_ids(semantic)
    affine = raster_infer._viewbox_affine(width, height)
    document = vectorizer.vectorize_masks(
        semantic,
        instances,
        scale_x=affine["scale_x"],
        scale_y=affine["scale_y"],
        offset_x=affine["offset_x"],
        offset_y=affine["offset_y"],
        viewbox=VIEWBOX,
        tol_px=tol,
    )
    _check_document(document)

    thing_instances = sorted(
        {int(value) for value in document["instanceIds"] if int(value) >= 0}
    )
    metadata: Dict[str, Any] = {
        "png_path": os.path.abspath(png_path),
        "source_size": [width, height],
        "tile_size": int(tiled["tile_size"]),
        "stride": int(tiled["stride"]),
        "tile_count": int(tiled["tile_count"]),
        "tile_rows": [int(value) for value in tiled["tile_rows"]],
        "tile_columns": [int(value) for value in tiled["tile_columns"]],
        "padding": [int(value) for value in tiled["padding"]],
        "padded_size": [int(value) for value in tiled["padded_size"]],
        "primitive_count": int(len(document["commands"])),
        "thing_instance_count": int(len(thing_instances)),
        "ink_support_pixels": int(support.sum()),
        "decoder": DECODER,
        "tol_px": tol,
        "viewbox": [float(value) for value in VIEWBOX],
        "viewbox_mapping": VIEWBOX_MAPPING,
        "viewbox_mapping_status": VIEWBOX_MAPPING_STATUS,
        "affine": {
            "scale_x": float(affine["scale_x"]),
            "scale_y": float(affine["scale_y"]),
            "offset_x": float(affine["offset_x"]),
            "offset_y": float(affine["offset_y"]),
        },
        "input_guard": {
            "max_pixels": INPUT_MAX_PIXELS,
            "max_file_bytes": INPUT_MAX_FILE_BYTES,
            "source": INPUT_LIMITS_SOURCE,
            "scope": "input guard only; not a GPU memory guarantee",
        },
    }
    return {"document": document, "metadata": metadata}


# ---------------------------------------------------------------------------
# deterministic input enumeration
# ---------------------------------------------------------------------------


def collect_inputs(input_path: str) -> Tuple[List[str], int]:
    """Return sorted PNG paths plus the ignored-entry count.

    A directory is scanned one level deep; only entries whose suffix is
    ``.png`` (case-insensitive) are returned. SVG, JSON labels, manifests and
    subdirectories are never opened.
    """

    target = os.path.abspath(input_path)
    if os.path.isdir(target):
        names = sorted(os.listdir(target))
        pngs: List[str] = []
        ignored = 0
        for name in names:
            full = os.path.join(target, name)
            if os.path.isfile(full) and name.lower().endswith(".png"):
                pngs.append(full)
            else:
                ignored += 1
        if not pngs:
            raise TiledSubmissionError(f"no_png_inputs:{target}")
        return pngs, ignored
    if os.path.isfile(target):
        if not target.lower().endswith(".png"):
            raise TiledSubmissionError(f"input_not_png:{target}")
        return [target], 0
    raise TiledSubmissionError(f"input_not_found:{target}")


def assign_output_ids(paths: Sequence[str]) -> List[Tuple[str, str]]:
    """Map PNG paths to ``<id>.json`` ids, or reject the batch.

    The id must be exactly five ASCII alphanumerics, as the dataset README
    requires. Collisions are detected case-insensitively so a Windows
    filesystem case fold cannot silently overwrite one result with another.
    """

    assigned: Dict[str, str] = {}
    pairs: List[Tuple[str, str]] = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if not ID_PATTERN.fullmatch(stem):
            raise TiledSubmissionError(
                f"input_name_not_five_alphanumerics:{stem!r}:{path}"
            )
        key = stem.casefold()
        if key in assigned:
            raise TiledSubmissionError(
                f"duplicate_output_id:{stem!r}:{assigned[key]}:{path}"
            )
        assigned[key] = path
        pairs.append((stem, path))
    pairs.sort(key=lambda item: item[0].casefold())
    return pairs


# ---------------------------------------------------------------------------
# results directory contract
# ---------------------------------------------------------------------------


def _prepare_output_dir(
    output_dir: str, output_ids: Sequence[str], *, overwrite: bool
) -> str:
    target = os.path.abspath(output_dir)
    if os.path.exists(target) and not os.path.isdir(target):
        raise TiledSubmissionError(f"output_not_directory:{target}")
    os.makedirs(target, exist_ok=True)
    allowed = {f"{image_id}.json" for image_id in output_ids}
    foreign = sorted(name for name in os.listdir(target) if name not in allowed)
    if foreign:
        raise TiledSubmissionError(
            "results_dir_not_clean:"
            + ",".join(foreign)
            + f":{target}:use a fresh directory"
        )
    if not overwrite:
        existing = sorted(name for name in os.listdir(target) if name in allowed)
        if existing:
            raise TiledSubmissionError(
                "results_exist:" + ",".join(existing) + f":{target}:pass --overwrite"
            )
    return target


def _require_metadata_outside(metadata_path: str, output_dir: str) -> str:
    target = os.path.abspath(metadata_path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        inside = os.path.commonpath([target, output_dir]) == output_dir
    except ValueError:  # different drives on Windows
        inside = False
    if inside:
        raise TiledSubmissionError(
            f"metadata_inside_results:{target}:write provenance outside {output_dir}"
        )
    return target


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _read_back(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        document = submission.loads_strict(handle.read())
    _check_document(document)
    return document


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, TiledSubmissionError):
        return "contract"
    if isinstance(exc, raster_infer.RasterInferError):
        return "input"
    if isinstance(exc, vectorizer.VectorizeError):
        return "vectorize"
    if isinstance(exc, fullfit.FullFitError):
        return "predict"
    if isinstance(exc, submission.SubmissionError):
        return "document"
    return "error"


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def _validate_device(device: str) -> str:
    value = str(device).strip()
    if value == "cpu":
        return "cpu"
    if value == "cuda" or re.fullmatch(r"cuda:\d+", value):
        import torch

        if not torch.cuda.is_available():
            raise TiledSubmissionError(
                f"cuda_not_available:{value}:no CPU fallback is applied"
            )
        return value
    raise TiledSubmissionError(
        f"device_not_explicit:{value!r}:use 'cpu' or 'cuda' (no auto selection)"
    )


def run_job(
    *,
    input_path: str,
    output_dir: str,
    checkpoint_path: str,
    device: str = "cpu",
    tol_px: float = DEFAULT_TOL_PX,
    metadata_path: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run one bounded batch and return its deterministic summary."""

    resolved_device = _validate_device(device)
    # The explicit checkpoint is the primary gate: a diagnostic, non-final or
    # wrong-head file is rejected before any input is enumerated.
    model, checkpoint_info = load_submission_checkpoint(
        checkpoint_path, device=resolved_device
    )
    paths, ignored = collect_inputs(input_path)
    pairs = assign_output_ids(paths)
    results_dir = _prepare_output_dir(
        output_dir, [image_id for image_id, _ in pairs], overwrite=overwrite
    )
    metadata_target = (
        _require_metadata_outside(metadata_path, results_dir)
        if metadata_path
        else None
    )

    outputs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for image_id, png_path in pairs:
        target = os.path.join(results_dir, f"{image_id}.json")
        written = False
        try:
            result = predict_document(
                png_path, model, device=resolved_device, tol_px=tol_px
            )
            _write_json(target, result["document"])
            written = True
            _read_back(target)
            entry = {"id": image_id, "path": target}
            entry.update(result["metadata"])
            outputs.append(entry)
        except Exception as exc:  # noqa: BLE001 - reported, never retried
            if written and os.path.isfile(target):
                try:
                    os.remove(target)
                except OSError:  # pragma: no cover - defensive
                    pass
            failures.append(
                {
                    "id": image_id,
                    "path": os.path.abspath(png_path),
                    "error": f"{_failure_code(exc)}:{_single_line(exc)}",
                }
            )

    # Whole-batch re-validation: every JSON in the results directory is read
    # back from disk and checked, not just the one that was just written. A
    # document that fails here is removed so the directory holds only valid
    # twelve-field results.
    kept: List[Dict[str, Any]] = []
    for entry in outputs:
        target = entry["path"]
        try:
            _read_back(target)
        except Exception as exc:  # noqa: BLE001 - reported, never retried
            if os.path.isfile(target):
                try:
                    os.remove(target)
                except OSError:  # pragma: no cover - defensive
                    pass
            failures.append(
                {
                    "id": entry["id"],
                    "path": entry["png_path"],
                    "error": f"{_failure_code(exc)}:{_single_line(exc)}",
                }
            )
        else:
            kept.append(entry)
    outputs = kept
    failures.sort(key=lambda failure: failure["id"].casefold())

    summary: Dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "ok" if not failures else "failed",
        "input": os.path.abspath(input_path),
        "output_dir": results_dir,
        "checkpoint": checkpoint_info,
        "device": resolved_device,
        "tiling": {
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "overlap": TILE_SIZE - TILE_STRIDE,
            "combine": "arithmetic mean of overlapping logits",
            "padding": "right/bottom only, image value 1.0 (white)",
            "resize": None,
        },
        "decode": {
            "decoder": DECODER,
            "semantic": "argmax on 0..34 over PNG ink support only",
            "unknown": "outside ink support -> ignore35 before components",
            "instance": "Thing: per-class 8-connected components; Stuff: id -1",
            "ink_support_policy": dict(raster_data.INK_SUPPORT_POLICY),
        },
        "viewbox": {
            "viewbox": [float(value) for value in VIEWBOX],
            "mapping": VIEWBOX_MAPPING,
            "mapping_status": VIEWBOX_MAPPING_STATUS,
            "official_compatibility": "unconfirmed",
        },
        "input_guard": {
            "max_pixels": INPUT_MAX_PIXELS,
            "max_file_bytes": INPUT_MAX_FILE_BYTES,
            "source": INPUT_LIMITS_SOURCE,
            "scope": "input guard only; not a GPU memory guarantee",
        },
        "counts": {
            "requested": len(pairs),
            "written": len(outputs),
            "failed": len(failures),
            "ignored_non_png": int(ignored),
        },
        "outputs": outputs,
        "failures": failures,
        "results_dir_files": sorted(os.listdir(results_dir)),
        "labels_or_gt_read": False,
        "network_used": False,
        "retry": "none",
    }

    if metadata_target:
        _write_json(
            metadata_target,
            {"schema": METADATA_SCHEMA, "run": summary},
        )
        summary["metadata"] = metadata_target
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.recognition_tiled_submission",
        description=(
            "Offline PNG-only tiled inference: one explicit checkpoint plus a "
            "PNG file or flat PNG directory -> <output>/<id>.json with the "
            "twelve README section 4 fields."
        ),
        epilog=(
            "Tiling is fixed at tile 256 / stride 128 with no resize. GT, SVG "
            "sidecars, labels and manifests are never read; nothing is "
            "downloaded and a failed image is never retried."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="explicit pilot/fullfit checkpoint (final, 35 classes, role fit)",
    )
    parser.add_argument(
        "--input", required=True, help="one PNG file or a flat directory of PNGs"
    )
    parser.add_argument(
        "--output", required=True, help="results directory for <id>.json only"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="'cpu' or 'cuda'/'cuda:N'; no 'auto' and no silent fallback",
    )
    parser.add_argument(
        "--tol-px",
        type=float,
        default=DEFAULT_TOL_PX,
        help="RDP tolerance in source pixels (default 0.5)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="optional provenance JSON; must be outside the results directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing <id>.json files of this batch",
    )
    return parser


def _configure_console() -> None:
    """Make diagnostic output printable on any console codepage.

    Paths may contain non-ASCII characters; a legacy console encoding would
    otherwise raise ``UnicodeEncodeError`` while reporting a failure. JSON
    result files are always written as UTF-8 and are unaffected.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # e.g. StringIO under test
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):  # pragma: no cover - defensive
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_console()
    try:
        summary = run_job(
            input_path=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            device=args.device,
            tol_px=args.tol_px,
            metadata_path=args.metadata,
            overwrite=bool(args.overwrite),
        )
    except TiledSubmissionError as exc:
        print(f"error:{_single_line(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    for failure in summary["failures"]:
        print(
            f"failed:{failure['id']}:{failure['error']}",
            file=sys.stderr,
        )
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
