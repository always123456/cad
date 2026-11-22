"""Prediction providers: prediction masks in, official 12-field documents out.

A model owner does not have to emit the official document at all.  It may write
prediction masks (PNG, or the ``.npz`` bundle used by the canonical pipeline) and
this module turns them into the exact document the scorer consumes, using
``cadvec.png_raster_vectorize.vectorize_masks(..., emitter="runs")`` -- the same
vectorizer and the same canonical pixel -> viewBox affine as the rest of the
project.  Nothing here is a second scorer, a second vectorizer or an inference
step: it only reads prediction pixels the model already produced.

PNG convention (prediction side only, never GT)
-----------------------------------------------
``<slug>.semantic.png``   one 8/16 bit index image, values = official class ids
                          ``0..35`` (35 = ignore / background).
``<slug>.instance.png``   same size, one 8/16 bit index image; the value is the
                          Thing instance id (``0..``) and is ignored on Stuff
                          and ignore pixels, which become ``-1``.

The document is the canonical function of those two arrays, so it can be sealed
and re-derived byte-for-byte later.  Anything the vectorizer normalises (for
example renumbering Thing instance ids per class) is reported in ``diagnostics``
instead of being silently absorbed.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from . import contract as C

MASK_SUFFIXES = {"semantic": ".semantic.png", "instance": ".instance.png"}


# --------------------------------------------------------------------------- #
# PNG / npz loading
# --------------------------------------------------------------------------- #
def _pillow() -> Any:
    if "PIL.Image" not in C._CACHE:  # shared cache keeps a single import
        try:
            from PIL import Image  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment guard
            raise C.GuardError("pillow_missing", str(exc)) from None
        C._CACHE["PIL.Image"] = Image
    return C._CACHE["PIL.Image"]


def read_mask_png(path: str | os.PathLike[str], *, kind: str) -> Any:
    """Read one index PNG as an integer array of shape ``(h, w)``."""

    numpy = C.require_numpy()
    target = os.path.abspath(str(path))
    if not os.path.isfile(target):
        raise C.GuardError("mask_missing", "%s %s" % (kind, target))
    Image = _pillow()
    try:
        with Image.open(target) as image:
            image.load()
            mode = image.mode
            array = numpy.asarray(image)
    except C.GuardError:
        raise
    except Exception as exc:
        raise C.GuardError("mask_unreadable", "%s %s: %s" % (kind, target, exc)) from None
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise C.GuardError("mask_channels", "%s %s: shape %r" % (kind, target, array.shape))
        first = array[..., 0]
        for channel in range(1, array.shape[2]):
            if not bool(numpy.array_equal(first, array[..., channel])):
                raise C.GuardError(
                    "mask_not_index",
                    "%s %s: mode %s holds colours, not index ids" % (kind, target, mode),
                )
        array = first
    if array.ndim != 2:
        raise C.GuardError("mask_shape", "%s %s: %r" % (kind, target, array.shape))
    if array.dtype.kind == "f":
        raise C.GuardError("mask_dtype", "%s %s: float mask" % (kind, target))
    return array


def mask_source_paths(
    mask_dir: str | os.PathLike[str], sample_id: str
) -> dict[str, str]:
    """Resolve ``semantic``/``instance`` PNG paths for one sample."""

    directory = os.path.abspath(str(mask_dir))
    hits: dict[str, str] = {}
    for kind, suffix in MASK_SUFFIXES.items():
        names = [C.slug(sample_id) + suffix, str(sample_id).split(":")[-1] + suffix]
        found = [os.path.join(directory, name) for name in names]
        found = [path for path in dict.fromkeys(found) if os.path.isfile(path)]
        if not found:
            raise C.GuardError(
                "mask_missing", "%s: none of %r in %s" % (sample_id, names, directory)
            )
        if len(found) > 1:
            raise C.GuardError("mask_ambiguous", "%s: several %s files %r" % (sample_id, kind, found))
        hits[kind] = found[0]
    return hits


def load_mask_pngs(
    semantic_path: str | os.PathLike[str],
    instance_path: str | os.PathLike[str],
) -> tuple[Any, Any, tuple[int, int]]:
    """Load a semantic/instance PNG pair as scorer-ready arrays.

    Returns ``(semantic, instance, (width, height))`` where ``instance`` is
    ``-1`` on every pixel that is not a Thing, matching the canonical scorer.
    """

    numpy = C.require_numpy()
    semantic_raw = read_mask_png(semantic_path, kind="semantic")
    instance_raw = read_mask_png(instance_path, kind="instance")
    if semantic_raw.shape != instance_raw.shape:
        raise C.GuardError(
            "mask_shape_mismatch",
            "semantic %r vs instance %r" % (semantic_raw.shape, instance_raw.shape),
        )
    height, width = int(semantic_raw.shape[0]), int(semantic_raw.shape[1])
    semantic = semantic_raw.astype(numpy.int64)
    values = instance_raw.astype(numpy.int64)
    thing = (semantic >= C.THING_MIN) & (semantic <= C.THING_MAX)
    instance = numpy.full(semantic.shape, -1, dtype=numpy.int64)
    instance[thing] = values[thing]
    semantic = semantic.astype(numpy.uint8)
    instance = instance.astype(numpy.int32)
    C.validate_prediction_maps(semantic, instance, (width, height))
    return semantic, instance, (width, height)


# --------------------------------------------------------------------------- #
# derivation
# --------------------------------------------------------------------------- #
def mask_diagnostics(
    semantic: Any,
    instance: Any,
    document: Mapping[str, Any],
    affine: Mapping[str, Any],
    *,
    source_size: Sequence[int],
) -> dict[str, Any]:
    """Model-facing clues about how the masks survived the canonical affine."""

    numpy = C.require_numpy()
    height, width = int(source_size[1]), int(source_size[0])
    scale = float(affine["pixels_per_view_unit"])
    left = float(affine["letterbox_x"])
    top = float(affine["letterbox_y"])
    right = left + C.VIEWBOX[2] * scale
    bottom = top + C.VIEWBOX[3] * scale
    xs = numpy.arange(width)[None, :]
    ys = numpy.arange(height)[:, None]
    inside = (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)
    primitive = (semantic >= C.THING_MIN) & (semantic <= C.STUFF_MAX)
    outside = primitive & ~inside
    ids = [int(value) for value in instance[primitive]]
    emitted = [int(value) for value in document["instanceIds"] if int(value) >= 0]
    thing_keys = {
        (int(semantic_id), int(instance_id))
        for semantic_id, instance_id in zip(document["semanticIds"], document["instanceIds"])
        if C.THING_MIN <= int(semantic_id) <= C.THING_MAX and int(instance_id) >= 0
    }
    per_class: dict[str, int] = {}
    for class_id in sorted({int(value) for value in semantic[primitive]}):
        per_class[str(class_id)] = int((semantic == class_id).sum())
    return {
        "source_size": [int(source_size[0]), int(source_size[1])],
        "canvas": [int(document["width"]), int(document["height"])],
        "primitives": len(document["commands"]),
        "thing_instances": len(thing_keys),
        "thing_primitives": len(emitted),
        "classes": sorted(int(key) for key in per_class),
        "pixels_per_class": per_class,
        "primitive_pixels": int(primitive.sum()),
        "outside_letterbox_pixels": int(outside.sum()),
        "instance_ids_input": sorted({value for value in ids if value >= 0}),
        "instance_ids_emitted": sorted(set(emitted)),
        "instance_ids_renumbered": bool(
            sorted({value for value in ids if value >= 0}) != sorted(set(emitted))
        ),
        "empty": not document["commands"],
    }


def document_from_mask_files(
    semantic_path: str | os.PathLike[str],
    instance_path: str | os.PathLike[str],
    *,
    labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full provider result for one sample: document, affine, sizes, diagnostics."""

    semantic, instance, source_size = load_mask_pngs(semantic_path, instance_path)
    document, affine = C.document_from_maps(semantic, instance, source_size[0], source_size[1])
    return {
        "document": document,
        "affine": affine,
        "source_size": list(source_size),
        "semantic_path": os.path.abspath(str(semantic_path)),
        "instance_path": os.path.abspath(str(instance_path)),
        "labels": dict(labels or {}),
        "diagnostics": mask_diagnostics(
            semantic, instance, document, affine, source_size=source_size
        ),
    }


def document_from_npz(
    map_path: str | os.PathLike[str],
    *,
    labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    semantic, instance, source_size = C.load_prediction_maps(map_path)
    document, affine = C.document_from_maps(semantic, instance, source_size[0], source_size[1])
    return {
        "document": document,
        "affine": affine,
        "source_size": list(source_size),
        "map_path": os.path.abspath(str(map_path)),
        "labels": dict(labels or {}),
        "diagnostics": mask_diagnostics(
            semantic, instance, document, affine, source_size=source_size
        ),
    }


class MaskProvider:
    """Callable provider: ``provider(record) -> provider result``.

    ``format="png"`` (default) reads ``<slug>.semantic.png`` /
    ``<slug>.instance.png``; ``format="npz"`` reads ``<slug>.npz`` bundles.
    Records need only ``sample_id``.
    """

    def __init__(
        self,
        mask_dir: str | os.PathLike[str],
        *,
        format: str = "png",
    ) -> None:
        if format not in ("png", "npz"):
            raise C.GuardError("mask_format", "unsupported format %r" % (format,))
        self.directory = os.path.abspath(str(mask_dir))
        self.format = format
        if not os.path.isdir(self.directory):
            raise C.GuardError("masks_missing", self.directory)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MaskProvider(%r, format=%r)" % (self.directory, self.format)

    def __call__(self, record: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(record["sample_id"])
        labels = {"sample_id": sample_id, "group_id": record.get("group_id")}
        if self.format == "npz":
            path = os.path.join(self.directory, C.slug(sample_id) + ".npz")
            if not os.path.isfile(path):
                raise C.GuardError("mask_missing", "%s: %s" % (sample_id, path))
            return document_from_npz(path, labels=labels)
        paths = mask_source_paths(self.directory, sample_id)
        return document_from_mask_files(paths["semantic"], paths["instance"], labels=labels)

    def documents(self, records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """Derive one result per record, keyed by ``sample_id``."""

        results: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_id = str(record["sample_id"])
            if sample_id in results:
                raise C.GuardError("mask_duplicate", sample_id)
            results[sample_id] = self(record)
        return results


def derive_documents(
    mask_dir: str | os.PathLike[str],
    records: Sequence[Mapping[str, Any]],
    *,
    format: str = "png",
) -> dict[str, Mapping[str, Any]]:
    """``{sample_id: document}`` for models that emit masks only."""

    provider = MaskProvider(mask_dir, format=format)
    return {
        sample_id: result["document"]
        for sample_id, result in provider.documents(records).items()
    }


__all__ = [
    "MASK_SUFFIXES",
    "MaskProvider",
    "derive_documents",
    "document_from_mask_files",
    "document_from_npz",
    "load_mask_pngs",
    "mask_diagnostics",
    "mask_source_paths",
    "read_mask_png",
]
