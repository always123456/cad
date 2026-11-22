"""Guard rails, artifact helpers and document contracts for the direct-instance lane.

Every public helper here is model independent: the lane consumes already written
12-field result documents, so a future model can be swapped in without touching
the evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

PKG_DIR = os.path.dirname(os.path.abspath(__file__))       # .../evaluation/instance_eval
LANE_ROOT = os.path.dirname(PKG_DIR)                       # .../evaluation
OUTPUTS_DIR = os.path.dirname(os.path.dirname(LANE_ROOT))  # .../outputs
REPO_ROOT = os.path.dirname(OUTPUTS_DIR)                   # repo root
SRC_DIR = os.path.join(REPO_ROOT, "src")

MANIFEST_PATH = os.path.join(
    OUTPUTS_DIR,
    "recognition-gate3b-fit-tune-20261108-01",
    "recognition_public_manifest.json",
)

ROLE = "tune"
ROLE_FIT = "fit"
SCALE = 32
CAP = 4
SAMPLE_IDS = (
    "train_final:I8wfZ",
    "train_final:UtkbT",
    "train_final:9unYH",
    "train_final:O3ItP",
)
VIEWBOX = (0.0, 0.0, 100.0, 100.0)
CANVAS_SIZE = (100.0, 100.0)
RASTER_SIZE = 1000
TOL_PX = 0.5
IOU_THRESHOLD = 0.5          # canonical scorer matches only when iou > 0.5
RECALL_TARGET = 0.90         # strict: recall > 0.90

DOCUMENT_FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
    "obj_cts",
    "boxes",
    "rgb",
    "layerIds",
    "widths",
)
AFFINE_FIELDS = (
    "pixels_per_view_unit",
    "letterbox_x",
    "letterbox_y",
    "scale_x",
    "scale_y",
    "offset_x",
    "offset_y",
)
SEALED_TOKENS = ("test_final", "dev_final", "eval_pool", "reserve", ".runtime")

# Mirrors cadvec.recognition_submission (asserted in the test suite): Thing ids
# 0..29 carry instances, Stuff 30..34 is merged per class, 35 is ignore.
THING_MIN, THING_MAX = 0, 29
STUFF_MIN, STUFF_MAX = 30, 34
IGNORE_ID = 35
SCHEMA_SEAL = "cadvec.direct_instance.pre_gt_seal.v1"
SCHEMA_REPORT = "cadvec.direct_instance.evaluation_report.v1"
SCHEMA_PREDICTION = "cadvec.direct_instance.prediction.v1"
SCHEMA_INPUTS = "cadvec.direct_instance.inputs.v1"
SCHEMA_BASELINE = "cadvec.direct_instance.reference_baseline.v1"


class GuardError(RuntimeError):
    """Raised with a stable ``code`` so failures are machine readable."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (code, detail) if detail else code)
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- #
# filesystem
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], payload: Any) -> str:
    """Write atomically and refuse to overwrite an existing file."""

    path = os.path.abspath(str(path))
    if os.path.exists(path):
        raise GuardError("artifact_exists", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def slug(sample_id: str) -> str:
    return str(sample_id).replace(":", "_")


def require_lane_owned(path: str | os.PathLike[str]) -> str:
    """Refuse any write outside this lane's evaluation directory."""

    target = os.path.abspath(str(path))
    try:
        common = os.path.commonpath([target, LANE_ROOT])
    except ValueError:
        raise GuardError("outside_lane", target) from None
    if os.path.normcase(common) != os.path.normcase(LANE_ROOT):
        raise GuardError("outside_lane", target)
    return target


def assert_no_sealed(path: str | os.PathLike[str], label: str = "path") -> str:
    text = str(path).replace("\\", "/").lower()
    for token in SEALED_TOKENS:
        if token in text:
            raise GuardError("sealed_path", "%s resolves into protected data: %s" % (label, path))
    return str(path)


def ensure_repo_on_path() -> str:
    if SRC_DIR not in os.sys.path:
        os.sys.path.insert(0, SRC_DIR)
    return SRC_DIR


# --------------------------------------------------------------------------- #
# canonical project modules (imported lazily; no torch, no model)
# --------------------------------------------------------------------------- #
_CACHE: dict[str, Any] = {}


def _module(name: str) -> Any:
    if name not in _CACHE:
        ensure_repo_on_path()
        module = __import__("cadvec." + name, fromlist=["*"])
        _CACHE[name] = module
    return _CACHE[name]


def raster_data() -> Any:
    return _module("recognition_raster_data")


def submission() -> Any:
    return _module("recognition_submission")


def vectorizer() -> Any:
    return _module("png_raster_vectorize")


def require_numpy() -> Any:
    if "numpy" not in _CACHE:
        try:
            import numpy  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment guard
            raise GuardError("numpy_missing", str(exc)) from None
        _CACHE["numpy"] = numpy
    return _CACHE["numpy"]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def default_inputs(manifest: str | os.PathLike[str] = MANIFEST_PATH) -> list[dict[str, Any]]:
    """The pinned four exposed TUNE samples, resolved from the public manifest.

    Only the manifest JSON is opened: no PNG, no SVG, no mask.
    """

    assert_no_sealed(manifest, "manifest")
    data = raster_data()
    records = data.load_role_samples(str(manifest), ROLE, SCALE)
    if not records:
        raise GuardError("no_samples", "manifest produced no %s records" % ROLE)
    pinned = list(records[:CAP])
    ids = [record["sample_id"] for record in pinned]
    if len(ids) != CAP:
        raise GuardError("fixed4_cap", "expected %d samples, got %d" % (CAP, len(ids)))
    if tuple(ids) != SAMPLE_IDS:
        raise GuardError("fixed4_ids", "expected %r, got %r" % (list(SAMPLE_IDS), ids))
    groups = [record.get("group_id") for record in pinned]
    if len({str(group) for group in groups}) != CAP:
        raise GuardError("fixed4_groups", "sampled groups are not distinct: %r" % (groups,))
    return [
        {
            "sample_id": record["sample_id"],
            "group_id": record.get("group_id"),
            "png_path": record["png_path"],
            "svg_path": record["svg_path"],
        }
        for record in pinned
    ]


def select_inputs(
    manifest: str | os.PathLike[str] = MANIFEST_PATH,
    *,
    role: str = ROLE,
    count: int | None = None,
    exclude: Sequence[str] = (),
    scale: int = SCALE,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """Pick an explicit, ordered sample set from the public manifest.

    ``exclude`` drops ids that another set already owns -- a capacity-16 FIT
    diagnostic must stay disjoint from the fixed four TUNE samples.  Only ids
    and paths are read from the manifest; no PNG, SVG or mask is opened.
    """

    assert_no_sealed(manifest, "manifest")
    if count is not None and int(count) <= 0:
        raise GuardError("inputs_count", "count must be positive, got %r" % (count,))
    data = raster_data()
    records = data.load_role_samples(str(manifest), role, scale)
    if not records:
        raise GuardError("no_samples", "manifest produced no %r records" % (role,))
    banned = {str(value) for value in exclude}
    picked: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in banned or sample_id in {row["sample_id"] for row in picked}:
            continue
        picked.append(
            {
                "sample_id": sample_id,
                "group_id": record.get("group_id"),
                "png_path": record["png_path"],
                "svg_path": record["svg_path"],
            }
        )
        if count is not None and len(picked) >= int(count):
            break
    if count is not None and len(picked) != int(count):
        raise GuardError(
            "inputs_short",
            "%s: wanted %d samples, manifest offered %d after exclusions"
            % (label or role, int(count), len(picked)),
        )
    groups = [str(row.get("group_id")) for row in picked]
    if len(set(groups)) != len(groups):
        raise GuardError("inputs_groups", "%s: repeated group ids %r" % (label or role, groups))
    return picked


def fit_inputs(
    manifest: str | os.PathLike[str] = MANIFEST_PATH,
    *,
    count: int = 16,
) -> list[dict[str, Any]]:
    """Capacity-N FIT set (role ``fit``), explicitly disjoint from the fixed four.

    The FIT role and the TUNE role never share a group, so a memorisation
    diagnostic on this set cannot reuse a scored TUNE sample; the id exclusion is
    kept as a cheap, explicit assertion of that.
    """

    return select_inputs(
        manifest,
        role=ROLE_FIT,
        count=int(count),
        exclude=SAMPLE_IDS,
        label="fit%d" % int(count),
    )


def normalize_inputs(
    inputs: Any,
    *,
    manifest: str | os.PathLike[str] = MANIFEST_PATH,
) -> list[dict[str, Any]]:
    """Accept a list of records, a path to a JSON spec, or ``None`` for fixed4."""

    if inputs is None:
        return default_inputs(manifest)
    if isinstance(inputs, (str, os.PathLike)):
        assert_no_sealed(inputs, "inputs")
        payload = load_json(inputs)
        if isinstance(payload, Mapping):
            schema = payload.get("schema")
            if schema is not None and schema != SCHEMA_INPUTS:
                raise GuardError("inputs_schema", "unexpected schema %r" % (schema,))
            rows = payload.get("samples")
        else:
            rows = payload
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise GuardError("inputs_shape", "inputs file must hold a list of samples")
    else:
        rows = inputs
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise GuardError("inputs_shape", "inputs must be a list of samples")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GuardError("inputs_shape", "sample %d is not an object" % index)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise GuardError("inputs_sample_id", "sample %d has no sample_id" % index)
        if sample_id in seen:
            raise GuardError("inputs_duplicate", sample_id)
        seen.add(sample_id)
        png_path = row.get("png_path")
        if not isinstance(png_path, str) or not png_path.strip():
            raise GuardError("inputs_png", "%s has no png_path" % sample_id)
        png_path = os.path.abspath(png_path)
        assert_no_sealed(png_path, "png_path")
        if not os.path.isfile(png_path):
            raise GuardError("inputs_png_missing", png_path)
        svg_path = row.get("svg_path")
        if svg_path is not None:
            if not isinstance(svg_path, str) or not svg_path.strip():
                raise GuardError("inputs_svg", "%s has an empty svg_path" % sample_id)
            svg_path = os.path.abspath(svg_path)
            assert_no_sealed(svg_path, "svg_path")
        records.append(
            {
                "sample_id": sample_id,
                "group_id": row.get("group_id"),
                "png_path": png_path,
                "svg_path": svg_path,
            }
        )
    if not records:
        raise GuardError("inputs_empty", "no samples to evaluate")
    return records


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def unwrap_prediction(payload: Any, sample_id: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Accept a bare 12-field document or a ``{"document": ...}`` envelope."""

    if not isinstance(payload, Mapping):
        raise GuardError("prediction_schema", "%s: prediction is not an object" % sample_id)
    if "document" in payload:
        document = payload["document"]
        if not isinstance(document, Mapping):
            raise GuardError("prediction_schema", "%s: document is not an object" % sample_id)
        declared = payload.get("sample_id")
        if declared is not None and str(declared) != str(sample_id):
            raise GuardError(
                "prediction_sample_id",
                "%s: envelope declares %r" % (sample_id, declared),
            )
        meta = {
            key: payload[key]
            for key in ("schema", "sample_id", "source_size", "affine", "metadata")
            if key in payload
        }
        return document, meta
    return payload, {}


def validate_prediction_document(document: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    """Strict model output validation (rejects Thing ``instance < 0``)."""

    errors = submission().validate_document(document)
    if errors:
        raise GuardError(
            "prediction_invalid",
            "%s: %s" % (sample_id, "; ".join(str(error) for error in errors[:12])),
        )
    return document


def validate_gt_document(document: Any, sample_id: str) -> Mapping[str, Any]:
    """GT is the official reference: Thing ``instance == -1`` is legal there."""

    if not isinstance(document, Mapping):
        raise GuardError("gt_invalid", "%s: gt_document is not an object" % sample_id)
    missing = [field for field in DOCUMENT_FIELDS if field not in document]
    if missing:
        raise GuardError("gt_invalid", "%s: missing fields %r" % (sample_id, missing))
    commands = document["commands"]
    args = document["args"]
    lengths = document["lengths"]
    semantic = document["semanticIds"]
    instance = document["instanceIds"]
    sizes = {len(commands), len(lengths), len(semantic), len(instance)}
    if len(sizes) != 1:
        raise GuardError("gt_invalid", "%s: primitive arrays disagree in length" % sample_id)
    for index, arg in enumerate(args):
        if not isinstance(arg, Sequence) or len(arg) != 8:
            raise GuardError("gt_invalid", "%s: args[%d] is not 8 numbers" % (sample_id, index))
    for index, value in enumerate(semantic):
        if not isinstance(value, int) or not 0 <= value <= 35:
            raise GuardError("gt_invalid", "%s: semanticIds[%d]=%r" % (sample_id, index, value))
    for index, value in enumerate(instance):
        if not isinstance(value, int) or value < -1:
            raise GuardError("gt_invalid", "%s: instanceIds[%d]=%r" % (sample_id, index, value))
    return document


def prediction_stats(document: Mapping[str, Any]) -> dict[str, Any]:
    semantic = [int(value) for value in document["semanticIds"]]
    instance = [int(value) for value in document["instanceIds"]]
    thing = sorted({inst for sem, inst in zip(semantic, instance) if 0 <= sem <= 29 and inst >= 0})
    classes = sorted({sem for sem in semantic if 0 <= sem <= 34})
    stuff = [sem for sem in classes if 30 <= sem <= 34]
    return {
        "primitives": len(document["commands"]),
        "thing_instances": len(thing),
        "classes_present": classes,
        "stuff_present": stuff,
    }


# --------------------------------------------------------------------------- #
# optional prediction masks (only when the model emits masks)
# --------------------------------------------------------------------------- #
def validate_prediction_maps(
    semantic: Any,
    instance: Any,
    source_size: Sequence[int],
    label: str = "map",
) -> None:
    numpy = require_numpy()
    sem = numpy.asarray(semantic)
    ins = numpy.asarray(instance)
    if sem.ndim != 2 or ins.ndim != 2:
        raise GuardError("map_shape", "%s: masks must be 2D" % label)
    if sem.shape != ins.shape:
        raise GuardError("map_shape", "%s: semantic %r vs instance %r" % (label, sem.shape, ins.shape))
    if len(source_size) != 2:
        raise GuardError("map_source_size", "%s: source_size must be (width, height)" % label)
    width, height = int(source_size[0]), int(source_size[1])
    if width <= 0 or height <= 0:
        raise GuardError("map_source_size", "%s: non-positive %r" % (label, tuple(source_size)))
    if sem.shape != (height, width):
        raise GuardError(
            "map_source_size",
            "%s: masks %r disagree with source_size %r" % (label, sem.shape, (height, width)),
        )
    if not numpy.issubdtype(sem.dtype, numpy.integer) or not numpy.issubdtype(ins.dtype, numpy.integer):
        raise GuardError("map_dtype", "%s: masks must be integer arrays" % label)
    if sem.size:
        if int(sem.min()) < 0 or int(sem.max()) > 35:
            raise GuardError("map_semantic", "%s: semantic ids out of 0..35" % label)
        if int(ins.min()) < -1:
            raise GuardError("map_instance", "%s: instance ids below -1" % label)
    known = (sem >= 0) & (sem < 35)
    thing = known & (sem < 30)
    stuff = known & (sem >= 30)
    if bool((thing & (ins < 0)).any()):
        raise GuardError("map_identity", "%s: Thing pixels need an instance id" % label)
    if bool((stuff & (ins >= 0)).any()):
        raise GuardError("map_identity", "%s: Stuff pixels must use instance -1" % label)
    if bool((ins[~known] != -1).any()):
        raise GuardError("map_identity", "%s: non-primitive pixels must use instance -1" % label)


def document_from_maps(
    semantic: Any,
    instance: Any,
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Vectorise prediction masks exactly like the canonical raster pipeline."""

    validate_prediction_maps(semantic, instance, (width, height))
    transform = submission().viewbox_transform(int(width), int(height), VIEWBOX)
    scale = float(transform[0])
    letterbox_x = float(transform[1]) if len(transform) > 1 else (float(width) - VIEWBOX[2] * scale) / 2.0
    letterbox_y = float(transform[2]) if len(transform) > 2 else (float(height) - VIEWBOX[3] * scale) / 2.0
    # canonical pixel -> viewBox affine: x_view = x_px * scale_x + offset_x
    scale_x = scale_y = 1.0 / scale
    offset_x = -letterbox_x / scale
    offset_y = -letterbox_y / scale
    document = vectorizer().vectorize_masks(
        semantic,
        instance,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
        VIEWBOX,
        TOL_PX,
        emitter="runs",
    )
    validate_prediction_document(document, "derived")
    affine = {
        "pixels_per_view_unit": scale,
        "letterbox_x": letterbox_x,
        "letterbox_y": letterbox_y,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }
    return dict(document), affine


def load_prediction_maps(path: str | os.PathLike[str]) -> tuple[Any, Any, tuple[int, int]]:
    numpy = require_numpy()
    try:
        with numpy.load(str(path), allow_pickle=False) as bundle:
            semantic = numpy.asarray(bundle["semantic"])
            instance = numpy.asarray(bundle["instance"])
            size = numpy.asarray(bundle["source_size"]).reshape(-1)
    except GuardError:
        raise
    except Exception as exc:
        raise GuardError("map_unreadable", "%s: %s" % (path, exc)) from None
    if size.size != 2:
        raise GuardError("map_source_size", "%s: source_size is not a pair" % path)
    source_size = (int(size[0]), int(size[1]))
    validate_prediction_maps(semantic, instance, source_size, label=os.path.basename(str(path)))
    return semantic, instance, source_size


# --------------------------------------------------------------------------- #
# optional reference baseline (explicit, provenance carrying)
# --------------------------------------------------------------------------- #
REFERENCE_NORMALIZATION = {
    "canvas": list(CANVAS_SIZE),
    "raster": [RASTER_SIZE, RASTER_SIZE],
    "iou_threshold": IOU_THRESHOLD,
    "assignment": "hungarian",
    "aggregation": "micro",
    "same_class": True,
    "stroke_width": 1,
}


KNOWN_REFERENCE_BASELINES = (
    {
        "name": "FloorPlanCAD PanCADNet (original)",
        "metric": "PQ",
        "value": 0.561,
        "sq": 0.838,
        "rq": 0.660,
        "source": "arXiv:2105.07147 v2, Tables 5 and 6",
        "protocol": "30-class vector-entity length-weighted IoU, macro across classes, Hungarian, IoU 0.5",
        "discrepancy": "a later PQ 59.5 is an unverified protocol difference, not the official number",
        "comparable": False,
        "usable_as_gate": False,
        "why": (
            "this project scores proxy pixel IoU on a 100 canvas / %d px raster with 1 px "
            "stroke and micro aggregation; that normalisation differs, so the value must "
            "never be compared directly or turned into a pass threshold" % RASTER_SIZE
        ),
        "record": "baselines/floorplancad_pancadnet_pq.json",
        "metadata_only": True,
    },
)


def _normalization_mismatches(declared: Mapping[str, Any]) -> list[str]:
    mismatched: list[str] = []
    for key, expected in REFERENCE_NORMALIZATION.items():
        if key not in declared:
            mismatched.append("missing:%s" % key)
            continue
        actual = declared[key]
        if isinstance(expected, list):
            try:
                pair = [float(value) for value in actual]
            except (TypeError, ValueError):
                mismatched.append("type:%s" % key)
                continue
            reference = [float(value) for value in expected]
            if len(pair) != len(reference) or any(
                abs(a - b) > 1e-9 for a, b in zip(pair, reference)
            ):
                mismatched.append("value:%s" % key)
        elif isinstance(expected, bool):
            if bool(actual) is not expected:
                mismatched.append("value:%s" % key)
        elif isinstance(expected, (int, float)):
            try:
                if abs(float(actual) - float(expected)) > 1e-9:
                    mismatched.append("value:%s" % key)
            except (TypeError, ValueError):
                mismatched.append("type:%s" % key)
        elif str(actual).strip().lower() != str(expected):
            mismatched.append("value:%s" % key)
    extra = sorted(set(declared) - set(REFERENCE_NORMALIZATION))
    for key in extra:
        mismatched.append("extra:%s" % key)
    return mismatched


def load_reference_baseline(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a *user supplied* baseline. Never invents a value or a scale."""

    assert_no_sealed(path, "reference_baseline")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise GuardError("baseline_schema", "reference baseline must be an object")
    schema = payload.get("schema")
    if schema is not None and schema != SCHEMA_BASELINE:
        raise GuardError("baseline_schema", "unexpected schema %r" % (schema,))
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise GuardError("baseline_provenance", "reference baseline needs a source string")

    record: dict[str, Any] = {
        "path": os.path.abspath(str(path)),
        "sha256": sha256_file(path),
        "source": source,
        "metric": payload.get("metric", "PQ"),
        "value": payload.get("value"),
        "normalization": payload.get("normalization"),
        "notes": payload.get("notes"),
        "comparable": False,
        "status": "unavailable_value",
        "mismatches": [],
    }

    value = payload.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        record["status"] = "unavailable_value"
        return record
    record["value"] = float(value)

    declared = payload.get("normalization")
    if not isinstance(declared, Mapping):
        record["status"] = "unavailable_normalization"
        return record
    mismatches = _normalization_mismatches(declared)
    record["mismatches"] = mismatches
    if mismatches:
        record["status"] = "incomparable_normalization"
        return record
    record["status"] = "available"
    record["comparable"] = True
    return record


__all__ = [
    "AFFINE_FIELDS",
    "CANVAS_SIZE",
    "CAP",
    "DOCUMENT_FIELDS",
    "GuardError",
    "IOU_THRESHOLD",
    "KNOWN_REFERENCE_BASELINES",
    "LANE_ROOT",
    "ROLE_FIT",
    "MANIFEST_PATH",
    "RASTER_SIZE",
    "RECALL_TARGET",
    "REPO_ROOT",
    "ROLE",
    "SAMPLE_IDS",
    "SCALE",
    "SCHEMA_BASELINE",
    "SCHEMA_INPUTS",
    "SCHEMA_PREDICTION",
    "SCHEMA_REPORT",
    "SCHEMA_SEAL",
    "SRC_DIR",
    "TOL_PX",
    "VIEWBOX",
    "assert_no_sealed",
    "canonical_json",
    "default_inputs",
    "document_from_maps",
    "fit_inputs",
    "ensure_repo_on_path",
    "load_json",
    "load_prediction_maps",
    "load_reference_baseline",
    "normalize_inputs",
    "prediction_stats",
    "raster_data",
    "require_lane_owned",
    "require_numpy",
    "sha256_file",
    "sha256_value",
    "select_inputs",
    "slug",
    "submission",
    "utc_now",
    "validate_gt_document",
    "validate_prediction_document",
    "validate_prediction_maps",
    "vectorizer",
    "write_json",
]
