"""Expanded-fit manifest reader and native-384 instance-centred crop builder.

This lane resolves the frozen expanded manifest independently.  It never calls a
historical restricted role loader; it filters the manifest's own admitted
``role='fit'`` records and reconstructs only
``<datasets_root>/train_final/<png_name|svg_name>``.

Dense images are loaded through a bounded LRU (2 by default) so a 3753-record
manifest does not imply 3753 decoded 1000x1000 rasters in memory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import contract


TRAIN_CROP_SIZE = 384
TRAIN_JITTER = 24
EXPECTED_SCHEMA = "cadvec.recognition.expanded_train_manifest/1"
EXPECTED_RECORDS = 3753
EXPECTED_GROUPS = 3528
FIT_RECORDS = 204
CAPACITY_CROPS = 16
CANONICAL_FIT16 = (
    "train_final:5Mnth",
    "train_final:M5Xaz",
    "train_final:WP1Nr",
    "train_final:E2isE",
    "train_final:4ENE3",
    "train_final:KR0Uf",
    "train_final:ZmGh0",
    "train_final:D70Oq",
    "train_final:Lpvc1",
    "train_final:4OOBV",
    "train_final:SdxLB",
    "train_final:Kpv63",
    "train_final:FYrUh",
    "train_final:H8qyg",
    "train_final:QLaAI",
    "train_final:WUf6V",
)


class ExpandedManifestError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: Any, name: str) -> str:
    text = str(value)
    if not text or text in (".", "..") or os.path.basename(text) != text:
        raise ExpandedManifestError("%s is not a bare filename: %r" % (name, value))
    if "/" in text or "\\" in text:
        raise ExpandedManifestError("%s contains a path separator: %r" % (name, value))
    return text


@dataclass(frozen=True)
class FitRecord:
    record_index: int
    sample_id: str
    group_id: str
    split: str
    source_role: str
    png_name: str
    svg_name: str
    png_digest: str

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "FitRecord":
        split = str(row.get("split", ""))
        if split != "train_final":
            raise ExpandedManifestError("record split %r is not train_final" % split)
        role = str(row.get("role", ""))
        if role != "fit":
            raise ExpandedManifestError("record role %r is not fit" % role)
        source_role = str(row.get("source_role", ""))
        if source_role not in ("fit", "reserve"):
            raise ExpandedManifestError("unsupported source_role %r" % source_role)
        return cls(
            record_index=int(row["record_index"]),
            sample_id=str(row["sample_id"]),
            group_id=str(row["group_id"]),
            split=split,
            source_role=source_role,
            png_name=_safe_component(row["png_name"], "png_name"),
            svg_name=_safe_component(row["svg_name"], "svg_name"),
            png_digest=str(row.get("png_digest", "")),
        )


class ExpandedFitSource:
    """Frozen admitted fit records with bounded decoded-image cache."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        limit: Optional[int] = None,
        first_source_role: Optional[str] = None,
        cache_size: int = 2,
        verify_records: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise ExpandedManifestError("manifest not found: %r" % str(self.manifest_path))
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != EXPECTED_SCHEMA:
            raise ExpandedManifestError("unexpected schema %r" % manifest.get("schema"))
        if int(manifest.get("version", 0)) != 1:
            raise ExpandedManifestError("unexpected manifest version %r" % manifest.get("version"))
        root = Path(str(manifest["datasets_root"]))
        if not root.is_dir():
            raise ExpandedManifestError("datasets_root is not a directory: %r" % str(root))
        self.datasets_root = root
        records_meta = manifest.get("records") or {}
        records_path = self.manifest_path.parent / str(records_meta.get("file", ""))
        if not records_path.is_file():
            raise ExpandedManifestError("records file not found: %r" % str(records_path))
        if verify_records and records_meta.get("sha256"):
            actual = _sha256(records_path)
            if actual != str(records_meta["sha256"]):
                raise ExpandedManifestError(
                    "records sha256 mismatch: %s != %s" % (actual, records_meta["sha256"])
                )
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.records_sha256 = _sha256(records_path)

        rows: List[FitRecord] = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record = FitRecord.from_json(row)
                if first_source_role is not None and record.source_role != first_source_role:
                    continue
                rows.append(record)
        rows.sort(key=lambda value: value.record_index)
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise ExpandedManifestError("no admitted fit records selected")
        self.records: Tuple[FitRecord, ...] = tuple(rows)
        self._cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[int, Tuple[Dict[str, Any], Dict[str, Any]]]" = OrderedDict()
        self.project = contract.project_modules()
        self.targets = self.project["targets"]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def sample_ids(self) -> List[str]:
        return [record.sample_id for record in self.records]

    def paths(self, index: int) -> Tuple[Path, Path]:
        record = self.records[index]
        split_dir = self.datasets_root / record.split
        png = split_dir / record.png_name
        svg = split_dir / record.svg_name
        for path in (png, svg):
            if not path.is_file():
                raise ExpandedManifestError("missing source file: %r" % str(path))
        return png, svg

    def _load(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        record = self.records[index]
        png, svg = self.paths(index)
        if record.png_digest:
            actual = _sha256(png)
            if actual != record.png_digest:
                raise ExpandedManifestError(
                    "png digest mismatch for %s: %s != %s"
                    % (record.sample_id, actual, record.png_digest)
                )
        sample = self.project["raster_data"].build_sample(
            str(png), str(svg), record.sample_id
        )
        target = self.targets.build_targets(sample["image"], sample["gt_document"])
        self._cache[index] = (sample, target)
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sample, target

    def get(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        return self._load(index)

    def __iter__(self) -> Iterator[FitRecord]:
        return iter(self.records)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def crop_for_update(
    source: ExpandedFitSource,
    index: int,
    *,
    seed: int,
    update: int,
    key_index: Optional[int] = None,
    jitter: int = TRAIN_JITTER,
    size: int = TRAIN_CROP_SIZE,
) -> Dict[str, Any]:
    """Build one reproducible instance-centred native-384 crop.

    The sampled official instance's bbox centre drives the window; jitter is
    bounded to +/-24 px.  When the bbox is larger than the window (or near an
    edge) clipping is recorded per key, never silently hidden.
    """

    sample, target = source.get(index)
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ExpandedManifestError("sample image must be CHW")
    height, width = int(image.shape[1]), int(image.shape[2])
    keys = list(target["keys"])
    masks = np.asarray(target["masks"], dtype=bool)
    boxes = np.asarray(target["boxes"], dtype=np.float32)

    rng = random.Random(int(seed) * 1000003 + int(index) * 9176 + int(update))
    if key_index is None and keys:
        key_index = rng.randrange(len(keys))
    if key_index is not None and keys:
        key_index = int(key_index) % len(keys)
        cx = float((boxes[key_index, 0] + boxes[key_index, 2]) * 0.5)
        cy = float((boxes[key_index, 1] + boxes[key_index, 3]) * 0.5)
    else:
        key_index = None
        cx, cy = width * 0.5, height * 0.5
    jitter_x = rng.uniform(-jitter, jitter) if jitter else 0.0
    jitter_y = rng.uniform(-jitter, jitter) if jitter else 0.0
    max_x = max(0, width - size)
    max_y = max(0, height - size)
    origin_x = int(round(cx + jitter_x - size * 0.5))
    origin_y = int(round(cy + jitter_y - size * 0.5))
    origin_x = max(0, min(max_x, origin_x))
    origin_y = max(0, min(max_y, origin_y))

    cropped = source.targets.crop_targets(
        image, target, origin=(origin_x, origin_y), size=size
    )
    crop_masks = np.asarray(cropped["masks"], dtype=bool)
    crop_boxes = np.asarray(cropped["boxes"], dtype=np.float32)
    crop_keys = list(cropped["keys"])
    visibility = np.asarray(cropped["visibility"], dtype=bool)
    domain = np.asarray(cropped["loss_domain"], dtype=bool)

    selected = visibility.copy()
    if selected.size:
        selected &= crop_masks.any(axis=(1, 2))
    selected_indices = np.where(selected)[0]
    selected_masks = crop_masks[selected_indices]
    selected_boxes = crop_boxes[selected_indices]
    selected_keys = [crop_keys[i] for i in selected_indices]
    union = selected_masks.any(axis=0) if selected_masks.size else np.zeros(domain.shape, dtype=bool)

    if "blank_negative" in cropped and cropped["blank_negative"] is not None:
        negative = np.asarray(cropped["blank_negative"], dtype=bool)
        if bool((negative & union).any()):
            raise ExpandedManifestError("contradictory blank_negative overlaps union(masks)")
    else:
        negative = domain & ~union

    clipping: List[Dict[str, Any]] = []
    if selected_indices.size:
        for local, original_index in enumerate(selected_indices):
            original_mask = masks[original_index]
            total = int(original_mask.sum())
            retained = int(crop_masks[original_index].sum())
            original_box = boxes[original_index]
            translated = np.array(
                [
                    original_box[0] - origin_x,
                    original_box[1] - origin_y,
                    original_box[2] - origin_x,
                    original_box[3] - origin_y,
                ],
                dtype=np.float32,
            )
            contained = bool(
                translated[0] >= 0
                and translated[1] >= 0
                and translated[2] <= size
                and translated[3] <= size
            )
            clipping.append(
                {
                    "key": [int(selected_keys[local][0]), int(selected_keys[local][1])],
                    "original_pixels": total,
                    "retained_pixels": retained,
                    "retention": _clip01(retained / total) if total else 1.0,
                    "bbox_fully_contained": contained,
                }
            )

    labels = [
        int(key[0]) + 1 for key in selected_keys
    ]  # detector labels from keys, never dense labels
    detector_target = {
        "boxes": torch.as_tensor(selected_boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(selected_masks, dtype=torch.bool),
        "keys": tuple((int(key[0]), int(key[1])) for key in selected_keys),
        "loss_domain": torch.as_tensor(domain, dtype=torch.bool),
        "blank_negative": torch.as_tensor(negative, dtype=torch.bool),
        "image_id": int(index),
    }
    metadata = {
        "record_index": int(index),
        "sample_id": source.records[index].sample_id,
        "group_id": source.records[index].group_id,
        "source_role": source.records[index].source_role,
        "origin": [int(origin_x), int(origin_y)],
        "size": int(size),
        "native_size": [height, width],
        "sampled_key_index": key_index,
        "target_count": int(len(keys)),
        "visible_count": int(selected_indices.size),
        "visible_keys": [[int(k[0]), int(k[1])] for k in selected_keys],
        "domain_pixels": int(domain.sum()),
        "negative_pixels": int(negative.sum()),
        "positive_pixels": int(union.sum()),
        "clipping": clipping,
        "jitter": [float(jitter_x), float(jitter_y)],
    }
    return {
        "image": torch.as_tensor(np.asarray(cropped["image"], dtype=np.float32)),
        "target": detector_target,
        "metadata": metadata,
        "crop_target_raw": cropped,
    }


def capacity_report(
    source: ExpandedFitSource,
    *,
    crops: int = CAPACITY_CROPS,
    seed: int = 20261108,
    size: int = TRAIN_CROP_SIZE,
) -> Dict[str, Any]:
    """Deterministic memorisation-capacity inventory; never a TUNE metric."""

    selected = min(int(crops), len(source))
    entries: List[Dict[str, Any]] = []
    fit_inputs: List[Dict[str, Any]] = []
    selected_ids = [source.records[i].sample_id for i in range(selected)]
    expected_ids = list(CANONICAL_FIT16[:selected])
    if selected_ids != expected_ids:
        raise ExpandedManifestError(
            "capacity records do not match canonical first-%d FIT order: %r != %r"
            % (selected, selected_ids, expected_ids)
        )
    for index in range(selected):
        # Capacity uses the largest official instance, no jitter, so the crop
        # decision is fully deterministic and independent of torch RNG.
        sample, target = source.get(index)
        png, svg = source.paths(index)
        fit_inputs.append(
            {
                "record_index": int(index),
                "sample_id": source.records[index].sample_id,
                "group_id": source.records[index].group_id,
                "png_path": str(png),
                "svg_path": str(svg),
            }
        )
        boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0]:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            key_index = int(np.argmax(areas))
        else:
            key_index = None
        result = crop_for_update(
            source,
            index,
            seed=seed,
            update=-1,
            key_index=key_index,
            jitter=0,
            size=size,
        )
        metadata = dict(result["metadata"])
        metadata.update(
            {
                "image_shape": [int(v) for v in result["image"].shape],
                "image_dtype": str(result["image"].dtype),
                "image_checksum": float(result["image"].double().sum().item()),
            }
        )
        entries.append(metadata)
    return {
        "contract": "direct-instance-maskrcnn-capacity/1",
        "manifest_path": str(source.manifest_path),
        "manifest_sha256": source.manifest_sha256,
        "records_sha256": source.records_sha256,
        "selected_records": selected,
        "requested_crops": int(crops),
        "crop_size": int(size),
        "seed": int(seed),
        "metric_scope": "fit_memorization_capacity",
        "tune_contact": False,
        "submission_recall": {
            "measured": False,
            "required_separately": "full-PNG inference + evaluator; crop memorization is not submission recall",
        },
        "fit_inputs": fit_inputs,
        "entries": entries,
    }
