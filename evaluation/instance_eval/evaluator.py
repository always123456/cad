"""Two-stage, model-independent evaluation: ``prepare`` then ``score``.

``prepare``  reads predictions only (plus optional prediction masks) and writes a
             pre-GT seal: every artifact hashed and every document validated and
             re-derived *before* any GT exists in the process.
``score``    refuses without an explicit grant, re-verifies the seal and every
             artifact byte-for-byte, and only then reads GT.

A model is swapped by pointing ``prepare`` at a different prediction directory.
Nothing here imports a checkpoint, runs inference or touches the network.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

from . import contract as C
from . import metrics as M

SEAL_NAME = "pre_gt_seal.json"
REPORT_NAME = "evaluation_report.json"

GtBuilder = Callable[[Mapping[str, Any]], Mapping[str, Any]]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_artifact(directory: str, sample_id: str, suffix: str, code: str) -> str:
    names: list[str] = []
    for candidate in (sample_id + suffix, C.slug(sample_id) + suffix):
        if candidate not in names:
            names.append(candidate)
    tail = str(sample_id).split(":")[-1] + suffix
    if tail not in names:
        names.append(tail)
    hits: list[str] = []
    for name in names:
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path):
                hits.append(path)
        except OSError:  # pragma: no cover - illegal characters on Windows
            continue
    if not hits:
        raise C.GuardError(code, "%s: none of %r in %s" % (sample_id, names, directory))
    if len(hits) > 1:
        raise C.GuardError(
            "artifact_ambiguous",
            "%s: several candidates %r in %s" % (sample_id, hits, directory),
        )
    return hits[0]


def _document_drift(document: Mapping[str, Any], derived: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in C.DOCUMENT_FIELDS
        if C.canonical_json(document.get(field)) != C.canonical_json(derived.get(field))
    ]


def _default_gt_builder() -> GtBuilder:
    """Build GT from the SVG alone: the scorer only consumes ``gt_document``.

    ``build_sample`` would also decode the PNG and rasterise the dense training
    targets, which scoring never reads.  ``document_from_svg`` runs the exact
    same parse (``split=None``, ``curve_steps=128``, ``record_commands=True``)
    and the same ``document_from_parsed`` converter, so the document is
    identical without the image load or the dense targets.
    """

    data = C.raster_data()

    def build(record: Mapping[str, Any]) -> Mapping[str, Any]:
        sample_id = str(record["sample_id"])
        svg_path = record.get("svg_path")
        if not svg_path:
            raise C.GuardError("gt_source_missing", "%s: no svg_path recorded" % sample_id)
        C.assert_no_sealed(svg_path, "svg_path")
        C.assert_no_sealed(record["png_path"], "png_path")
        document = data.document_from_svg(str(svg_path), curve_steps=128, split=None)
        return {"gt_document": document}

    return build


def _verify_seal(out_dir: str) -> dict[str, Any]:
    path = os.path.join(out_dir, SEAL_NAME)
    if not os.path.isfile(path):
        raise C.GuardError("seal_missing", "run prepare first: %s" % path)
    payload = C.load_json(path)
    if not isinstance(payload, Mapping):
        raise C.GuardError("seal_schema", "seal is not an object")
    if payload.get("schema") != C.SCHEMA_SEAL:
        raise C.GuardError("seal_schema", "unexpected schema %r" % (payload.get("schema"),))
    if payload.get("stage") != "prepare":
        raise C.GuardError("seal_stage", "unexpected stage %r" % (payload.get("stage"),))
    if payload.get("gt_read") is not False:
        raise C.GuardError("seal_gt_read", "seal does not declare gt_read=false")
    for key in ("svg_read", "png_read"):
        if payload.get(key) is not False:
            raise C.GuardError("seal_gt_read", "seal does not declare %s=false" % key)
    lane = payload.get("lane")
    if lane is not None and os.path.normcase(os.path.abspath(str(lane))) != os.path.normcase(C.LANE_ROOT):
        raise C.GuardError("seal_lane", "seal belongs to %r" % (lane,))

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise C.GuardError("seal_samples", "seal carries no samples")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise C.GuardError("seal_files", "seal carries no file ledger")

    verified: dict[str, dict[str, Any]] = {}
    for rel, entry in files.items():
        if not isinstance(entry, Mapping):
            raise C.GuardError("seal_files", "ledger entry for %r is not an object" % (rel,))
        target = os.path.abspath(str(rel))
        if not os.path.isfile(target):
            raise C.GuardError("artifact_missing", target)
        size = os.path.getsize(target)
        if int(entry.get("size_bytes", -1)) != size:
            raise C.GuardError(
                "artifact_changed",
                "%s: size %d, sealed %r" % (target, size, entry.get("size_bytes")),
            )
        digest = C.sha256_file(target)
        if digest != entry.get("sha256"):
            raise C.GuardError("artifact_changed", "%s: sha256 %s" % (target, digest))
        verified[os.path.normcase(target)] = {"path": target, "sha256": digest}

    ids = [str(sample.get("sample_id")) for sample in samples if isinstance(sample, Mapping)]
    if ids != [str(value) for value in (payload.get("sample_ids") or [])]:
        raise C.GuardError("seal_ids", "sample list disagrees with sample_ids")
    for sample in samples:
        prediction = sample.get("prediction")
        if not isinstance(prediction, str):
            raise C.GuardError("seal_ledger", "sample %r has no prediction path" % (sample.get("sample_id"),))
        if os.path.normcase(os.path.abspath(prediction)) not in verified:
            raise C.GuardError("seal_ledger", "prediction %s is not in the file ledger" % prediction)
        map_path = sample.get("map")
        if map_path is not None and os.path.normcase(os.path.abspath(str(map_path))) not in verified:
            raise C.GuardError("seal_ledger", "map %s is not in the file ledger" % (map_path,))
        masks = sample.get("masks")
        if masks is not None:
            if not isinstance(masks, Mapping):
                raise C.GuardError("seal_ledger", "sample %r has a malformed masks block" % (sample.get("sample_id"),))
            role = sample.get("mask_role", payload.get("mask_role"))
            if role not in ("authoritative", "diagnostic"):
                raise C.GuardError(
                    "seal_ledger",
                    "sample %r has no valid mask_role (%r)" % (sample.get("sample_id"), role),
                )
            for kind, mask_path in masks.items():
                if os.path.normcase(os.path.abspath(str(mask_path))) not in verified:
                    raise C.GuardError(
                        "seal_ledger",
                        "%s mask %s is not in the file ledger" % (kind, mask_path),
                    )

    return {"path": path, "payload": payload, "sha256": C.sha256_file(path), "files": verified}


def _success(metrics: Mapping[str, Any], reference: Mapping[str, Any] | None, leak_free: bool) -> dict[str, Any]:
    recall = metrics.get("recall")
    recall_pass = None if recall is None else bool(float(recall) > C.RECALL_TARGET)

    pq = metrics.get("PQ")
    if reference is None:
        pq_pass: bool | None = None
        pq_status = "no_reference_baseline"
    elif reference.get("status") != "available" or not reference.get("comparable"):
        pq_pass = None
        pq_status = str(reference.get("status") or "reference_unavailable")
    elif pq is None:
        pq_pass = None
        pq_status = "canonical_pq_unavailable"
    else:
        pq_pass = bool(float(pq) >= float(reference["value"]))
        pq_status = "compared"

    if not leak_free:
        status, passed = "gt_leak_detected", False
    elif recall_pass is False:
        status, passed = "recall_below_target", False
    elif pq_status == "compared" and pq_pass is False:
        status, passed = "pq_below_reference", False
    elif recall_pass is True and pq_pass is True:
        status, passed = "passed", True
    elif recall_pass is True:
        status, passed = "recall_only_reference_unavailable", False
    else:
        status, passed = "incomplete", False

    return {
        "passed": passed,
        "status": status,
        "recall_target": C.RECALL_TARGET,
        "recall_rule": "TP/(TP+FN) > %.2f" % C.RECALL_TARGET,
        "recall_pass": recall_pass,
        "pq_reference_pass": pq_pass,
        "pq_reference_status": pq_status,
        "gt_leak_free": bool(leak_free),
        "requires": {
            "recall": "> %.2f" % C.RECALL_TARGET,
            "pq": ">= reference baseline value (explicit, provenance carrying, same normalisation)",
            "leak": "seal verified before the first GT read",
        },
    }


def _scope_role(dataset: Any) -> str:
    """Derive the honest sample role from the dataset label, never a constant."""

    label = str(dataset or "").strip().lower()
    if "fit" in label:
        return C.ROLE_FIT
    if "tune" in label or label.startswith("fixed4"):
        return C.ROLE
    return "unspecified"


def _scope_claim(dataset: Any, n_images: int, metrics: Mapping[str, Any]) -> str:
    gt_instances = int(metrics["TP"] or 0) + int(metrics["FN"] or 0)
    label = str(dataset or "tune-fixed4")
    role = _scope_role(dataset)
    if role == C.ROLE_FIT:
        return (
            "capacity FIT memorisation diagnostic on %d FIT samples the model was fitted on "
            "(%d GT instances); these are exposed training samples, so this measures capacity "
            "and memorisation, not held-out generalisation" % (n_images, gt_instances)
        )
    if role == C.ROLE:
        return (
            "scored on the four already exposed TUNE samples (%d GT instances); "
            "this is a diagnostic, not a pristine held-out result" % gt_instances
        )
    return (
        "scored on %d exposed samples of set %r (%d GT instances); sample exposure is "
        "not held-out isolation, so this is a diagnostic, not a pristine held-out result"
        % (n_images, label, gt_instances)
    )


# --------------------------------------------------------------------------- #
# stage 1: prepare (no GT)
# --------------------------------------------------------------------------- #
def prepare(
    predictions_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str] | None = None,
    *,
    inputs: Any = None,
    maps_dir: str | os.PathLike[str] | None = None,
    masks_dir: str | os.PathLike[str] | None = None,
    mask_format: str = "png",
    mask_role: str = "authoritative",
    dataset: str = "tune-fixed4",
    origin: str = "prediction_json",
    manifest: str | os.PathLike[str] = C.MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate and hash predictions, then write the pre-GT seal.

    Never opens a PNG pixel of the dataset, an SVG or any GT file: only
    prediction JSON, optional prediction masks (``maps_dir`` = per-sample
    ``.npz`` bundles, ``masks_dir`` = per-sample prediction PNGs) plus the
    public manifest.  ``dataset`` only labels the sample set in the seal.

    ``mask_role`` decides how much the optional masks are trusted:

    * ``"authoritative"`` -- the JSON must equal the document re-derived from
      the masks (the ``prepare_from_masks`` path), otherwise it is refused.
    * ``"diagnostic"`` -- the masks are a dense winner-take-all rendering of
      independent, possibly overlapping instance predictions.  They are sealed
      and measured, but disagreements with the JSON are recorded instead of
      refused, because overlap cannot survive a per-pixel map.  JSON stays the
      source of truth and no mask is ever required to score.
    """

    if out_dir is None:
        raise C.GuardError("out_dir_missing", "prepare needs a lane owned out_dir for the seal")
    out_dir = C.require_lane_owned(out_dir)
    predictions_dir = os.path.abspath(str(predictions_dir))
    C.assert_no_sealed(predictions_dir, "predictions_dir")
    if not os.path.isdir(predictions_dir):
        raise C.GuardError("predictions_missing", predictions_dir)
    if maps_dir is not None:
        maps_dir = os.path.abspath(str(maps_dir))
        C.assert_no_sealed(maps_dir, "maps_dir")
        if not os.path.isdir(maps_dir):
            raise C.GuardError("maps_missing", maps_dir)
    if masks_dir is not None:
        if mask_format not in ("png",):
            raise C.GuardError("mask_format", "unsupported mask format %r" % (mask_format,))
        if mask_role not in ("authoritative", "diagnostic"):
            raise C.GuardError(
                "mask_role", "mask_role must be 'authoritative' or 'diagnostic'; got %r" % (mask_role,)
            )
        masks_dir = os.path.abspath(str(masks_dir))
        C.assert_no_sealed(masks_dir, "masks_dir")
        if not os.path.isdir(masks_dir):
            raise C.GuardError("masks_missing", masks_dir)
    if not isinstance(dataset, str) or not dataset.strip():
        raise C.GuardError("dataset_label", "dataset label must be a non-empty string")

    seal_path = os.path.join(out_dir, SEAL_NAME)
    if os.path.exists(seal_path):
        raise C.GuardError("seal_exists", "%s already sealed; pick a new out_dir" % seal_path)

    records = C.normalize_inputs(inputs, manifest=manifest)
    files: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    caveats: list[str] = []
    totals = {"primitives": 0, "thing_instances": 0, "maps": 0, "masks": 0}

    for record in records:
        sample_id = record["sample_id"]
        prediction_path = _resolve_artifact(predictions_dir, sample_id, ".json", "prediction_missing")
        payload = C.load_json(prediction_path)
        document, meta = C.unwrap_prediction(payload, sample_id)
        C.validate_prediction_document(document, sample_id)
        stats = C.prediction_stats(document)

        size = os.path.getsize(prediction_path)
        digest = C.sha256_file(prediction_path)
        files[prediction_path] = {"sha256": digest, "size_bytes": size, "kind": "prediction"}

        canvas = [int(document["width"]), int(document["height"])]
        if canvas != [100, 100]:
            caveats.append(
                "%s declares canvas %r; the canonical scorer always uses 100x100" % (sample_id, canvas)
            )

        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "group_id": record.get("group_id"),
            "prediction": prediction_path,
            "prediction_sha256": digest,
            "prediction_size_bytes": size,
            "prediction_stats": stats,
            "document_sha256": C.sha256_value(document),
            "canvas_declared": canvas,
            "png_path": record["png_path"],
            "svg_path": record.get("svg_path"),
        }
        totals["primitives"] += stats["primitives"]
        totals["thing_instances"] += stats["thing_instances"]

        if maps_dir is not None:
            map_path = _resolve_artifact(maps_dir, sample_id, ".npz", "map_missing")
            semantic, instance, source_size = C.load_prediction_maps(map_path)
            derived, affine = C.document_from_maps(semantic, instance, source_size[0], source_size[1])
            drift = _document_drift(document, derived)
            if drift and mask_role == "authoritative":
                raise C.GuardError(
                    "document_mismatch",
                    "%s: JSON disagrees with its masks on %r" % (sample_id, drift),
                )
            if drift:
                caveats.append(
                    "%s: diagnostic npz map differs from the JSON on %r (expected for "
                    "overlapping instances; JSON is scored)" % (sample_id, drift)
                )
            declared_size = meta.get("source_size")
            if declared_size is not None and list(map(int, declared_size)) != list(source_size):
                raise C.GuardError(
                    "source_size_mismatch",
                    "%s: envelope %r vs masks %r" % (sample_id, declared_size, source_size),
                )
            map_digest = C.sha256_file(map_path)
            map_size = os.path.getsize(map_path)
            files[map_path] = {"sha256": map_digest, "size_bytes": map_size, "kind": "prediction_map"}
            sample["map"] = map_path
            sample["map_sha256"] = map_digest
            sample["map_role"] = mask_role
            sample["map_size_bytes"] = map_size
            sample["source_size"] = list(source_size)
            sample["affine"] = affine
            totals["maps"] += 1

        if masks_dir is not None:
            from . import provider as P  # local import keeps the cycle absent

            paths = P.mask_source_paths(masks_dir, sample_id)
            result = P.document_from_mask_files(paths["semantic"], paths["instance"])
            drift = _document_drift(document, result["document"])
            if drift and mask_role == "authoritative":
                raise C.GuardError(
                    "document_mismatch",
                    "%s: JSON disagrees with its PNG masks on %r" % (sample_id, drift),
                )
            if drift:
                caveats.append(
                    "%s: diagnostic winner map differs from the JSON on %r (expected for "
                    "overlapping instances; JSON is scored)" % (sample_id, drift)
                )
            declared_size = meta.get("source_size")
            if declared_size is not None and list(map(int, declared_size)) != list(result["source_size"]):
                if mask_role == "authoritative":
                    raise C.GuardError(
                        "source_size_mismatch",
                        "%s: envelope %r vs masks %r" % (sample_id, declared_size, result["source_size"]),
                    )
                caveats.append(
                    "%s: diagnostic mask source_size %r differs from the declared %r"
                    % (sample_id, list(result["source_size"]), list(map(int, declared_size)))
                )
            for kind, path in paths.items():
                files[path] = {
                    "sha256": C.sha256_file(path),
                    "size_bytes": os.path.getsize(path),
                    "kind": "prediction_mask_" + kind,
                }
            sample["masks"] = dict(paths)
            sample["mask_format"] = mask_format
            sample["mask_role"] = mask_role
            sample["mask_drift"] = list(drift)
            sample["mask_sha256"] = {kind: files[path]["sha256"] for kind, path in paths.items()}
            sample["source_size"] = list(result["source_size"])
            sample["affine"] = result["affine"]
            sample["diagnostics"] = result["diagnostics"]
            totals["masks"] += 1
            for warning, present in (
                ("prediction masks are empty", result["diagnostics"]["empty"]),
                (
                    "prediction pixels fall outside the letterboxed viewBox and are lost: %d"
                    % result["diagnostics"]["outside_letterbox_pixels"],
                    result["diagnostics"]["outside_letterbox_pixels"] > 0,
                ),
            ):
                if present:
                    caveats.append("%s: %s" % (sample_id, warning))

        samples.append(sample)

    seal = {
        "schema": C.SCHEMA_SEAL,
        "lane": C.LANE_ROOT,
        "created_utc": C.utc_now(),
        "stage": "prepare",
        "gt_read": False,
        "svg_read": False,
        "png_read": False,
        "predictions_dir": predictions_dir,
        "maps_dir": maps_dir,
        "masks_dir": masks_dir,
        "mask_format": mask_format if masks_dir is not None else None,
        "mask_role": mask_role if masks_dir is not None else None,
        "dataset": dataset.strip(),
        "origin": origin,
        "manifest": os.path.abspath(str(manifest)),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "samples": samples,
        "files": files,
        "counts": {
            "samples": len(samples),
            "predictions": len(samples),
            "maps": totals["maps"],
            "masks": totals["masks"],
            "primitives": totals["primitives"],
            "thing_instances": totals["thing_instances"],
        },
        "canvas": list(C.CANVAS_SIZE),
        "raster": C.RASTER_SIZE,
        "iou_threshold": C.IOU_THRESHOLD,
        "caveats": caveats,
        "policy": (
            "Predictions were validated with cadvec.recognition_submission.validate_document "
            "(strict: Thing instance < 0 is rejected) and hashed before any GT exists."
        ),
    }
    C.write_json(seal_path, seal)
    sealed = C.load_json(seal_path)
    return {
        "seal_path": seal_path,
        "seal_sha256": C.sha256_file(seal_path),
        "dataset": sealed["dataset"],
        "origin": sealed["origin"],
        "sample_ids": list(sealed["sample_ids"]),
        "counts": dict(sealed["counts"]),
        "caveats": list(caveats),
    }


def prepare_from_documents(
    documents: Mapping[str, Mapping[str, Any]],
    out_dir: str | os.PathLike[str],
    *,
    inputs: Any = None,
    dataset: str = "tune-fixed4",
    manifest: str | os.PathLike[str] = C.MANIFEST_PATH,
    origin: str = "documents",
) -> dict[str, Any]:
    """Seal documents that already live in memory (no prediction directory).

    Use this for a set that is still being staged -- for example a capacity-16
    FIT memorisation diagnostic, or a future TUNE set: pass the sample set and
    the documents, and the same pre-GT seal covers them.  The documents are
    written under ``<out_dir>/predictions`` first, so ``score`` re-verifies them
    from bytes exactly like any other prediction directory.
    """

    if not isinstance(documents, Mapping) or not documents:
        raise C.GuardError("documents_missing", "no documents supplied")
    out_dir = C.require_lane_owned(out_dir)
    records = C.normalize_inputs(inputs, manifest=manifest)
    wanted = [record["sample_id"] for record in records]
    missing = [sample_id for sample_id in wanted if sample_id not in documents]
    extra = [str(sample_id) for sample_id in documents if sample_id not in set(wanted)]
    if missing or extra:
        raise C.GuardError(
            "documents_mismatch",
            "missing %r, unexpected %r" % (missing, extra),
        )
    predictions_dir = os.path.join(out_dir, "predictions")
    if os.path.exists(os.path.join(out_dir, SEAL_NAME)):
        raise C.GuardError("seal_exists", os.path.join(out_dir, SEAL_NAME))
    for sample_id in wanted:
        payload = documents[sample_id]
        document, _ = C.unwrap_prediction(payload, sample_id)
        C.validate_prediction_document(document, sample_id)
        target = os.path.join(predictions_dir, C.slug(sample_id) + ".json")
        if os.path.exists(target):
            raise C.GuardError("prediction_exists", target)
        C.write_json(target, dict(document))
    return prepare(
        predictions_dir,
        out_dir,
        inputs=records,
        dataset=dataset,
        origin=origin,
        manifest=manifest,
    )


def prepare_from_masks(
    mask_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    format: str = "png",
    inputs: Any = None,
    dataset: str = "tune-fixed4",
    mask_role: str = "authoritative",
    manifest: str | os.PathLike[str] = C.MANIFEST_PATH,
) -> dict[str, Any]:
    """Derive documents from prediction masks, then seal them with the masks.

    The model owner writes only masks (``format="png"``: per-sample
    ``semantic``/``instance`` PNGs; ``format="npz"``: per-sample bundles).  The
    documents come from the canonical ``vectorize_masks(..., emitter="runs")``
    path, so a model that never emits the official JSON is still scorable.  Both
    the derived documents and the mask bytes are sealed before any GT read.
    """

    from . import provider as P

    out_dir = C.require_lane_owned(out_dir)
    records = C.normalize_inputs(inputs, manifest=manifest)
    provider = P.MaskProvider(mask_dir, format=format)
    results = provider.documents(records)
    predictions_dir = os.path.join(out_dir, "predictions")
    if os.path.exists(os.path.join(out_dir, SEAL_NAME)):
        raise C.GuardError("seal_exists", os.path.join(out_dir, SEAL_NAME))
    diagnostics: dict[str, Any] = {}
    for sample_id, item in results.items():
        target = os.path.join(predictions_dir, C.slug(sample_id) + ".json")
        if os.path.exists(target):
            raise C.GuardError("prediction_exists", target)
        C.write_json(target, dict(item["document"]))
        diagnostics[sample_id] = item["diagnostics"]
    result = prepare(
        predictions_dir,
        out_dir,
        inputs=records,
        masks_dir=mask_dir,
        mask_format=format,
        mask_role=mask_role,
        dataset=dataset,
        origin="prediction_masks",
        manifest=manifest,
    )
    result["diagnostics"] = diagnostics
    return result


# --------------------------------------------------------------------------- #
# stage 2: score (GT, explicit grant)
# --------------------------------------------------------------------------- #
def score(
    out_dir: str | os.PathLike[str],
    *,
    allow_gt: bool = False,
    gt_builder: GtBuilder | None = None,
    reference_baseline: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    trace: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify the seal, read GT once, and report canonical PQ plus recall/precision.

    ``allow_gt`` must be ``True`` (CLI ``--allow-gt``) as the explicit parent
    grant; anything else aborts before the first GT read.
    """

    if allow_gt is not True:
        raise C.GuardError(
            "gt_not_authorized",
            "score reads GT and needs the parent's explicit --allow-gt grant",
        )
    out_dir = C.require_lane_owned(out_dir)
    report_target = C.require_lane_owned(report_path or os.path.join(out_dir, REPORT_NAME))
    C.assert_no_sealed(report_target, "report_path")
    if os.path.exists(report_target):
        raise C.GuardError("report_exists", report_target)

    seal = _verify_seal(out_dir)
    payload = seal["payload"]
    if trace is not None:
        trace.append(("seal_verified", seal["sha256"]))

    records = payload["samples"]
    cached: list[dict[str, Any]] = []
    attachment_notes: list[str] = []
    for record in records:
        sample_id = str(record["sample_id"])
        prediction_path = os.path.abspath(str(record["prediction"]))
        if os.path.getsize(prediction_path) != int(record["prediction_size_bytes"]):
            raise C.GuardError("artifact_changed", prediction_path)
        if C.sha256_file(prediction_path) != record["prediction_sha256"]:
            raise C.GuardError("artifact_changed", prediction_path)
        document, _ = C.unwrap_prediction(C.load_json(prediction_path), sample_id)
        C.validate_prediction_document(document, sample_id)
        if record.get("map"):
            map_path = os.path.abspath(str(record["map"]))
            if C.sha256_file(map_path) != record.get("map_sha256"):
                raise C.GuardError("artifact_changed", map_path)
            semantic, instance, source_size = C.load_prediction_maps(map_path)
            derived, _ = C.document_from_maps(semantic, instance, source_size[0], source_size[1])
            drift = _document_drift(document, derived)
            if drift and str(record.get("map_role") or "authoritative") == "authoritative":
                raise C.GuardError(
                    "document_mismatch",
                    "%s: JSON no longer matches its sealed masks on %r" % (sample_id, drift),
                )
            if drift:
                attachment_notes.append("%s: diagnostic npz map drift %r" % (sample_id, drift))
        if record.get("masks"):
            from . import provider as P

            result = P.document_from_mask_files(
                record["masks"]["semantic"], record["masks"]["instance"]
            )
            drift = _document_drift(document, result["document"])
            if drift and str(record.get("mask_role") or "authoritative") == "authoritative":
                raise C.GuardError(
                    "document_mismatch",
                    "%s: JSON no longer matches its sealed PNG masks on %r" % (sample_id, drift),
                )
            if drift:
                attachment_notes.append("%s: diagnostic PNG mask drift %r" % (sample_id, drift))
        cached.append({"record": record, "document": document, "sample_id": sample_id})
        if trace is not None:
            trace.append(("prediction_verified", sample_id))

    builder = gt_builder or _default_gt_builder()
    gt_documents: list[Mapping[str, Any]] = []
    pred_documents: list[Mapping[str, Any]] = []
    sample_ids: list[str] = []
    for item in cached:
        sample_id = item["sample_id"]
        record = item["record"]
        sample = builder(
            {
                "sample_id": sample_id,
                "group_id": record.get("group_id"),
                "png_path": record["png_path"],
                "svg_path": record.get("svg_path"),
            }
        )
        if trace is not None:
            trace.append(("gt_built", sample_id))
        document = sample.get("gt_document") if isinstance(sample, Mapping) else None
        gt_documents.append(C.validate_gt_document(document, sample_id))
        pred_documents.append(item["document"])
        sample_ids.append(sample_id)

    report_metrics = M.instance_report(gt_documents, pred_documents, sample_ids=sample_ids)
    if not report_metrics["consistency"]["ok"]:
        raise C.GuardError(
            "metric_inconsistency",
            "canonical scorer disagrees with the document audit: %r"
            % ({key: value for key, value in report_metrics["consistency"].items() if value is False},),
        )

    reference = None
    if reference_baseline is not None:
        reference = C.load_reference_baseline(reference_baseline)

    leak_free = True  # seal verified above, before the first builder call
    report = {
        "schema": C.SCHEMA_REPORT,
        "lane": C.LANE_ROOT,
        "created_utc": C.utc_now(),
        "stage": "score",
        "grant": {"explicit": True, "flag": "--allow-gt"},
        "seal": {
            "path": seal["path"],
            "sha256": seal["sha256"],
            "created_utc": payload.get("created_utc"),
            "predictions_dir": payload.get("predictions_dir"),
            "maps_dir": payload.get("maps_dir"),
            "counts": payload.get("counts"),
            "caveats": payload.get("caveats"),
        },
        "samples": [
            {
                "sample_id": item["sample_id"],
                "group_id": item["record"].get("group_id"),
                "prediction": item["record"]["prediction"],
                "prediction_sha256": item["record"]["prediction_sha256"],
                "map": item["record"].get("map"),
                "map_sha256": item["record"].get("map_sha256"),
                "prediction_stats": item["record"].get("prediction_stats"),
            }
            for item in cached
        ],
        "metrics": report_metrics,
        "success": _success(report_metrics, reference, leak_free),
        "reference_baseline": reference if reference is not None else {"status": "not_provided"},
        "known_reference_baselines": list(C.KNOWN_REFERENCE_BASELINES),
        "gt_read": True,
        "scope": {
            "role": _scope_role(payload.get("dataset")),
            "role_source": "dataset label %r" % (payload.get("dataset"),),
            "dataset": payload.get("dataset"),
            "origin": payload.get("origin"),
            "sample_ids": sample_ids,
            "n_images": len(sample_ids),
            "gt_instances_scored": report_metrics["TP"] + report_metrics["FN"],
            "exposed_samples": True,
            "heldout": False,
            "claim": _scope_claim(payload.get("dataset"), len(sample_ids), report_metrics),
        },
        "attachment_notes": attachment_notes,
        "metric_definition": {
            "match": "same class, strict IoU > %.2f" % C.IOU_THRESHOLD,
            "assignment": "hungarian one-to-one",
            "recall": "TP/(TP+FN)",
            "precision": "TP/(TP+FP)",
            "f1": "2*TP/(2*TP+FP+FN) (the harmonic mean of recall and precision when both exist)",
            "pq": "SumIoU/(TP+0.5*FP+0.5*FN)",
            "sq": "SumIoU/TP",
            "rq": "TP/(TP+0.5*FP+0.5*FN)",
            "zero_denominator": None,
            "scorer": "cadvec.recognition_raster_data.score_documents (canonical)",
        },
    }
    C.write_json(report_target, report)
    report["report_path"] = report_target
    return report


__all__ = [
    "REPORT_NAME",
    "SEAL_NAME",
    "prepare",
    "prepare_from_documents",
    "prepare_from_masks",
    "score",
]
