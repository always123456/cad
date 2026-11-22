"""Class-agnostic ink-topology instance pilot (bounded CPU feasibility lane).

Fixed, deterministic, PNG-only topology proposal:

1. visible-ink support comes from the shared helper
   (:func:`cadvec.recognition_raster_data._image_ink_support`, i.e. the
   alpha-composited RGB rule ``min(R, G, B) < 254``); no SVG, no GT;
2. atoms are the 8-connected components of that ink support
   (class-agnostic: the semantic map never influences the partition);
3. pockets are the 4-connected components of the complement that touch no
   image border; every border-touching background component is *exterior* and
   is never a pocket;
4. every atom is assigned to the pocket with the largest boundary contact
   count (4-neighbour contact), ties to the smallest pocket id; an atom with no
   enclosed-pocket contact is an *orphan* and is omitted (abstention);
5. a group is ``pocket union assigned atoms``; its class is the majority over
   the saved semantic map restricted to known Thing votes (0..29).  A group
   with no known Thing vote is skipped -- no class is invented and unknown is
   never used as a negative;
6. Stuff (30..34) keeps the baseline full same-class merge unchanged
   (``instance == -1``), exactly like ``png_raster_vectorize.vectorize_masks``.

Everything is one fixed rule set: no size filter, no confidence cutoff, no
per-image cardinality filter, no sweep, no training, no GT.  The single gate is
a whole-run feasibility gate (median enclosed-pocket count per image, and "no
candidate pocket at all"), evaluated before any vectorisation.

Persisted handoff (evaluator contract, GT-free)::

    outputs/recognition-topology-instances-20261112-01/evaluation/
        feasibility.json            # cadvec.recognition.topology_feasibility/1
        predictions/<slug>.npz      # ``semantic`` only (int, -1..35)
        documents/<slug>.json       # top-level ``document`` (12 fields)
        predictions_manifest.json   # cadvec...topology_instances_eval_predictions/1
        groups.json                 # pilot-global group registry (descriptive)
        diagnostics.json            # per-image counters (descriptive)

Run ``python -m cadvec.recognition_topology_instances --help``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - environment dependent
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - environment dependent
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:  # pragma: no cover
    _ndimage = None  # type: ignore

from . import png_raster_vectorize as vectorizer
from . import recognition_raster_data as raster_data
from . import recognition_submission as submission

ARM = "topology_pocket_atoms_v1"
ROLE = "fit"
SAMPLE_PREFIX = "train_final:"
EXPECTED_SAMPLES = 24
TOL_PX = 0.5

ARM_MANIFEST_SCHEMA = "cadvec.recognition.joint_arm_manifest/1"
FEASIBILITY_SCHEMA = "cadvec.recognition.topology_feasibility/1"
DOCUMENT_SCHEMA = "cadvec.recognition.topology_arm_document/1"
PREDICTIONS_MANIFEST_SCHEMA = (
    "cadvec.recognition.topology_instances_eval_predictions/1"
)
GROUPS_SCHEMA = "cadvec.recognition.topology_groups/1"
DIAGNOSTICS_SCHEMA = "cadvec.recognition.topology_diagnostics/1"

#: Whole-run feasibility gate: median enclosed-pocket count per image.
KILL_MEDIAN_POCKETS = 150
#: Pocket-size diagnostic bucket in pixels; a counter only, never a filter.
SMALL_POCKET_PX = 200
#: Bounded exhaustive duplicate check (counter only, never a gate).
FULL_UNIQUENESS_MAX_POCKETS = 512

THING_IDS = tuple(range(0, raster_data.THING_MAX))  # 0..29
STUFF_IDS = tuple(range(raster_data.STUFF_MIN - 1, raster_data.STUFF_MAX))  # 30..34
IGNORE_ID = int(raster_data.IGNORE_CLASS)  # 35
_THING_SET = frozenset(THING_IDS)

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_ARM_DIR = os.path.join(
    _REPO_ROOT,
    "outputs",
    "recognition-joint-offsets-20261112-01",
    "evaluation",
    "arms",
    "joint_semantic_cc8",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "recognition-topology-instances-20261112-01"
)

_FOUR = None
_EIGHT = None


class TopologyError(RuntimeError):
    """Raised for inputs or state that this pilot refuses to handle."""


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - environment dependent
        raise TopologyError("numpy is required for the topology lane")
    if _ndimage is None:  # pragma: no cover - environment dependent
        raise TopologyError("scipy.ndimage is required for the topology lane")


def _four() -> Any:
    global _FOUR
    if _FOUR is None:
        _FOUR = np.array(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=bool,
        )
    return _FOUR


def _eight() -> Any:
    global _EIGHT
    if _EIGHT is None:
        _EIGHT = np.ones((3, 3), dtype=bool)
    return _EIGHT


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# manifest / record loading (fit role only)


def load_records(arm_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the 24 fit records of the joint semantic-cc8 arm manifest.

    Only fit samples (``train_final:`` ids) are accepted and the manifest must
    hold exactly 24 samples in frozen FIT-24 order.  PNG, prediction and
    document paths come straight from the manifest.
    """

    base = os.path.abspath(arm_dir or DEFAULT_ARM_DIR)
    manifest_path = os.path.join(base, "arm_manifest.json")
    if not os.path.isfile(manifest_path):
        raise TopologyError(f"arm manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise TopologyError("arm manifest must be an object")
    if manifest.get("schema") != ARM_MANIFEST_SCHEMA:
        raise TopologyError(
            f"unexpected arm manifest schema: {manifest.get('schema')!r}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise TopologyError(
            f"arm manifest must list exactly {EXPECTED_SAMPLES} samples; "
            f"got {len(samples) if isinstance(samples, list) else samples!r}"
        )

    records: List[Dict[str, Any]] = []
    seen = set()
    for entry in samples:
        sample_id = str(entry.get("sample_id"))
        if not sample_id.startswith(SAMPLE_PREFIX):
            raise TopologyError(f"fit-only lane refuses non-fit sample: {sample_id!r}")
        if sample_id in seen:
            raise TopologyError(f"duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        png_path = os.path.abspath(str(entry.get("png_path")))
        prediction = os.path.abspath(os.path.join(base, str(entry.get("prediction"))))
        document = os.path.abspath(os.path.join(base, str(entry.get("document"))))
        for label, path in (
            ("png", png_path),
            ("prediction", prediction),
            ("document", document),
        ):
            if not os.path.isfile(path):
                raise TopologyError(f"{sample_id}: missing {label} file {path!r}")
        records.append(
            {
                "sample_id": sample_id,
                "png_path": png_path,
                "prediction_path": prediction,
                "document_path": document,
                "group_id": entry.get("group_id"),
                "role": ROLE,
                "arm_dir": base,
            }
        )
    return records


def load_semantic(png_path: str, prediction_path: str) -> Any:
    """Load the saved PNG-derived semantic map for ``png_path``.

    The map must match the PNG raster exactly and hold class ids 0..35.  No
    logits, no GT, no validity mask is read.
    """

    _require_numpy()
    with open(prediction_path, "rb") as handle:
        payload = np.load(handle, allow_pickle=False)
        if "semantic" not in payload.files:
            raise TopologyError(
                f"prediction has no semantic array: {prediction_path!r}"
            )
        semantic = np.asarray(payload["semantic"])
    if semantic.ndim != 2 or not np.issubdtype(semantic.dtype, np.integer):
        raise TopologyError(
            f"semantic must be a 2D integer array: {prediction_path!r}"
        )
    if int(semantic.min()) < 0 or int(semantic.max()) > IGNORE_ID:
        raise TopologyError(
            f"semantic ids outside 0..{IGNORE_ID}: {prediction_path!r}"
        )
    image = raster_data._load_image(png_path)
    if tuple(semantic.shape) != (int(image.shape[1]), int(image.shape[2])):
        raise TopologyError(
            f"semantic raster {semantic.shape} does not match PNG "
            f"{tuple(image.shape[1:])}: {png_path!r}"
        )
    return semantic.astype(np.int16, copy=False)


def ink_support(png_path: str) -> Any:
    """Shared visible-ink support (PNG RGB only) as a boolean raster."""

    _require_numpy()
    return np.asarray(
        raster_data._image_ink_support(raster_data._load_image(png_path)),
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# topology: atoms, pockets, assignment


def label_atoms(ink: Any) -> Any:
    """8-connected ink components; labels 1..n in raster order, 0 = no ink."""

    _require_numpy()
    labels, _ = _ndimage.label(np.asarray(ink, dtype=bool), structure=_eight())
    return labels.astype(np.int32, copy=False)


def label_pockets(ink: Any) -> Tuple[Any, int]:
    """4-connected enclosed background components.

    Returns ``(pocket, count)``: ``pocket`` holds 0 for everything that is not
    an enclosed pocket (ink and border-connected background) and 1..count, in
    raster order, for enclosed pockets.
    """

    _require_numpy()
    ink_array = np.asarray(ink, dtype=bool)
    labels, total = _ndimage.label(~ink_array, structure=_four())
    if total == 0:
        return np.zeros(ink_array.shape, dtype=np.int32), 0
    border_ids = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    exterior = np.zeros(int(total) + 1, dtype=bool)
    exterior[border_ids] = True
    exterior[0] = True
    enclosed = ~exterior[labels]
    kept = np.unique(labels[enclosed])
    kept = kept[kept > 0]
    remap = np.zeros(int(total) + 1, dtype=np.int32)
    remap[kept] = np.arange(1, kept.size + 1, dtype=np.int32)
    pocket = np.where(enclosed, remap[labels], 0).astype(np.int32, copy=False)
    return pocket, int(kept.size)


def atom_pocket_contacts(atoms: Any, pocket: Any) -> Tuple[Any, Any, int, int]:
    """4-neighbour boundary contact counts between atoms and pockets.

    Returns ``(atom_ids, counts, atom_count, pocket_count)`` with ``counts`` an
    ``(len(atom_ids), pocket_count)`` int64 matrix of contact pixels.
    """

    _require_numpy()
    atom_array = np.asarray(atoms, dtype=np.int32)
    pocket_array = np.asarray(pocket, dtype=np.int32)
    pocket_count = int(pocket_array.max())
    atom_count = int(atom_array.max())
    rows: List[Any] = []
    cols: List[Any] = []
    for ink_side, pocket_side in (
        (atom_array[:-1, :], pocket_array[1:, :]),
        (atom_array[1:, :], pocket_array[:-1, :]),
        (atom_array[:, :-1], pocket_array[:, 1:]),
        (atom_array[:, 1:], pocket_array[:, :-1]),
    ):
        hit = (ink_side > 0) & (pocket_side > 0)
        if hit.any():
            rows.append(ink_side[hit])
            cols.append(pocket_side[hit])
    if not rows:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros((0, pocket_count), dtype=np.int64),
            atom_count,
            pocket_count,
        )
    keys = np.concatenate(rows).astype(np.int64) * (pocket_count + 1) + np.concatenate(
        cols
    ).astype(np.int64)
    flat = np.bincount(keys, minlength=(atom_count + 1) * (pocket_count + 1))
    matrix = flat.reshape(atom_count + 1, pocket_count + 1)[:, 1:]
    atom_ids = np.nonzero(matrix.sum(axis=1) > 0)[0].astype(np.int32)
    return atom_ids, matrix[atom_ids, :].astype(np.int64, copy=False), atom_count, pocket_count


def assign_atoms(
    atom_ids: Any, counts: Any, atom_count: int, pocket_count: int
) -> Any:
    """Atom -> pocket: max contact, ties to the smallest pocket id.

    Returns int32 of length ``atom_count + 1``; index 0 and every unassigned
    atom hold 0, otherwise a pocket id 1..pocket_count.
    """

    _require_numpy()
    assignment = np.zeros(int(atom_count) + 1, dtype=np.int32)
    if pocket_count == 0 or len(atom_ids) == 0:
        return assignment
    matrix = np.asarray(counts, dtype=np.int64)
    best = np.argmax(matrix, axis=1)  # first maximum == smallest pocket id
    best_counts = matrix[np.arange(matrix.shape[0]), best]
    keep = best_counts > 0
    assignment[np.asarray(atom_ids, dtype=np.int32)[keep]] = (best + 1).astype(
        np.int32
    )[keep]
    return assignment


# ---------------------------------------------------------------------------
# one-image topology


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 6)


def _percentile(values: Any, percent: float) -> Optional[float]:
    if np is None or values is None or len(values) == 0:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), percent)), 6)


def _majority(votes: Any) -> Tuple[int, int, float, bool]:
    """Return ``(class, total_votes, share, tie)``; class 0 when no vote."""

    row = np.asarray(votes, dtype=np.int64)
    total = int(row.sum())
    if total == 0:
        return 0, 0, 0.0, False
    top = int(row.max())
    winners = np.nonzero(row == top)[0]
    return int(winners[0]), total, float(top) / float(total), bool(winners.size > 1)


def topology_from_arrays(ink: Any, semantic: Any) -> Dict[str, Any]:
    """Deterministic class-agnostic pocket/atom grouping for one raster.

    Returns per-image evidence plus the arrays a candidate map and a report are
    built from.  No threshold, no size filter, no GT.
    """

    _require_numpy()
    ink_array = np.asarray(ink, dtype=bool)
    semantic_array = np.asarray(semantic)
    if semantic_array.shape != ink_array.shape:
        raise TopologyError("ink and semantic shapes differ")
    if not np.issubdtype(semantic_array.dtype, np.integer):
        raise TopologyError("semantic must be an integer array")
    if int(semantic_array.min()) < 0 or int(semantic_array.max()) > IGNORE_ID:
        raise TopologyError(f"semantic ids outside 0..{IGNORE_ID}")

    atoms = label_atoms(ink_array)
    atom_count = int(atoms.max())
    pocket, pocket_count = label_pockets(ink_array)
    atom_ids, counts, _, _ = atom_pocket_contacts(atoms, pocket)
    assignment = assign_atoms(atom_ids, counts, atom_count, pocket_count)

    touching = assignment[atoms]  # pocket id per ink pixel, 0 = orphan atom
    pocket_area = (
        np.bincount(pocket.flatten(), minlength=pocket_count + 1).astype(np.int64)
        if pocket_count
        else np.zeros(1, dtype=np.int64)
    )
    owning_pockets = np.unique(touching[touching > 0]) if pocket_count else np.zeros(
        0, dtype=np.int32
    )
    candidate_pockets = np.asarray(owning_pockets, dtype=np.int32)

    group_of_pocket = np.zeros(pocket_count + 1, dtype=np.int32)
    group_of_pocket[candidate_pockets] = np.arange(
        1, candidate_pockets.size + 1, dtype=np.int32
    )
    group_count = int(candidate_pockets.size)
    atom_group = np.where(assignment > 0, group_of_pocket[assignment], 0).astype(
        np.int32
    )
    owner = np.where(ink_array, atom_group[atoms], group_of_pocket[pocket]).astype(
        np.int32
    )

    group_pixels = np.zeros(group_count + 1, dtype=np.int64)
    if group_count:
        group_pixels = np.bincount(owner.flatten(), minlength=group_count + 1).astype(
            np.int64
        )
        group_pixels[0] = 0

    n_classes = len(THING_IDS)
    votes = np.zeros((group_count + 1, n_classes), dtype=np.int64)
    known_thing = (
        (owner > 0)
        & ink_array
        & (semantic_array >= 0)
        & (semantic_array <= THING_IDS[-1])
    )
    if group_count and known_thing.any():
        keys = owner[known_thing].astype(np.int64) * n_classes + semantic_array[
            known_thing
        ].astype(np.int64)
        flat = np.bincount(keys, minlength=(group_count + 1) * n_classes)
        votes = flat.reshape(group_count + 1, n_classes)

    atom_sizes = (
        np.bincount(atoms.flatten(), minlength=atom_count + 1)[1:].astype(np.int64)
        if atom_count
        else np.zeros(0, dtype=np.int64)
    )

    # Atom-level known-Thing votes (counter only): each atom casts one vote from
    # its own pixel majority, so a single giant wall atom cannot outvote many
    # small atoms; the difference is reported, not used to select anything.
    atom_votes = np.zeros((atom_count + 1, n_classes), dtype=np.int64)
    if atom_count and known_thing.any():
        keys = atoms[known_thing].astype(np.int64) * n_classes + semantic_array[
            known_thing
        ].astype(np.int64)
        flat = np.bincount(keys, minlength=(atom_count + 1) * n_classes)
        atom_votes = flat.reshape(atom_count + 1, n_classes)

    groups: List[Dict[str, Any]] = []
    for group_index in range(1, group_count + 1):
        pocket_id = int(candidate_pockets[group_index - 1])
        atom_members = np.nonzero(assignment == pocket_id)[0]
        thing_class, total_votes, share, tie = _majority(votes[group_index])
        atom_vote_classes = [
            _atom_class(atom_votes[atom_id]) for atom_id in atom_members.tolist()
        ]
        atom_mode = _mode(atom_vote_classes)
        groups.append(
            {
                "group_index": group_index,
                "pocket_id": pocket_id,
                "pocket_area": int(pocket_area[pocket_id]),
                "atom_ids": [int(value) for value in atom_members],
                "atom_count": int(atom_members.size),
                "atom_ink_pixels": int(
                    int(atom_sizes[atom_members - 1].sum()) if atom_members.size else 0
                ),
                "pixels": int(group_pixels[group_index]),
                "thing_class": int(thing_class),
                "thing_votes": int(total_votes),
                "thing_vote_share": round(float(share), 6),
                "thing_vote_tie": bool(tie),
                "atom_mode_class": int(atom_mode),
                "kept": bool(total_votes > 0),
            }
        )

    ink_pixels = int(ink_array.sum())
    largest_atom = int(atom_sizes.max()) if atom_count else 0
    orphan_atoms = np.nonzero(assignment[1:] == 0)[0] + 1
    orphan_ink = int(atom_sizes[orphan_atoms - 1].sum()) if orphan_atoms.size else 0
    multi = int(np.count_nonzero((counts > 0).sum(axis=1) >= 2)) if counts.size else 0
    pocket_sizes = pocket_area[1:]

    owned_pixels = int(group_pixels.sum())
    duplicates = 0
    fully_checked = False
    if 0 < group_count <= FULL_UNIQUENESS_MAX_POCKETS:
        multiplicity = np.zeros(ink_array.shape, dtype=np.uint8)
        for group_index in range(1, group_count + 1):
            pocket_id = int(candidate_pockets[group_index - 1])
            member = (pocket == pocket_id) | (
                ink_array & (atom_group[atoms] == group_index)
            )
            multiplicity[member] += 1
        duplicates = int(np.count_nonzero(multiplicity > 1))
        fully_checked = True

    kept_groups = [group for group in groups if group["kept"]]
    skipped_groups = [group for group in groups if not group["kept"]]
    largest_group = max((group["pixels"] for group in kept_groups), default=0)

    # Group-mask invariant: a group is "enclosed pocket union its assigned ink
    # atoms", and those two parts are disjoint (an atom pixel is ink, a pocket
    # pixel is background).  Its pixel count is therefore exactly
    # pocket_area + atom_ink_pixels -- and it deliberately exceeds the image's
    # ink pixels by the pocket interior, which is background but belongs to the
    # enclosed object.  Nothing off-ink is ever owned outside a group's own
    # pocket, and no pixel is owned twice.
    invariant_ok = all(
        int(group["pocket_area"]) + int(group["atom_ink_pixels"])
        == int(group["pixels"])
        for group in groups
    )
    if not invariant_ok:
        breakage = [
            {
                "group_index": group["group_index"],
                "pocket_area": group["pocket_area"],
                "atom_ink_pixels": group["atom_ink_pixels"],
                "pixels": group["pixels"],
            }
            for group in groups
            if int(group["pocket_area"]) + int(group["atom_ink_pixels"])
            != int(group["pixels"])
        ]
        raise TopologyError(f"group mask is not pocket union assigned atoms: {breakage[:3]}")
    owned_ink = int(sum(int(group["atom_ink_pixels"]) for group in groups))
    owned_background = int(sum(int(group["pocket_area"]) for group in groups))
    largest_group_ink = 0
    if kept_groups:
        biggest = max(kept_groups, key=lambda group: group["pixels"])
        largest_group_ink = int(biggest["atom_ink_pixels"])
    off_pocket_background = int(
        np.count_nonzero((owner > 0) & (~ink_array) & (pocket == 0))
    )
    if off_pocket_background:
        raise TopologyError(
            f"{off_pocket_background} owned pixels are background outside their own "
            "enclosed pocket"
        )

    stats = {
        "ink_pixels": ink_pixels,
        "atom_count": atom_count,
        "largest_atom_pixels": largest_atom,
        "largest_atom_ink_share": _ratio(largest_atom, ink_pixels),
        "pocket_count": int(pocket_count),
        "candidate_pocket_count": int(candidate_pockets.size),
        "pocket_area_p10": _percentile(pocket_sizes, 10),
        "pocket_area_median": _percentile(pocket_sizes, 50),
        "pocket_area_p90": _percentile(pocket_sizes, 90),
        "pockets_below_small": int(np.count_nonzero(pocket_sizes < SMALL_POCKET_PX)),
        "small_pocket_px": int(SMALL_POCKET_PX),
        "atoms_multi_pocket": multi,
        "orphan_atom_count": int(orphan_atoms.size),
        "orphan_ink_pixels": int(orphan_ink),
        "orphan_ink_share": _ratio(orphan_ink, ink_pixels),
        "group_count": int(len(kept_groups)),
        "skipped_group_count": int(len(skipped_groups)),
        "largest_group_pixels": int(largest_group),
        "largest_group_share": _ratio(largest_group, owned_pixels),
        "largest_group_share_note": (
            "denominator is owned_pixels (ink + enclosed pocket background), so a "
            "filled object's group legitimately exceeds its share of ink pixels"
        ),
        "largest_group_share_of_ink": _ratio(largest_group_ink, ink_pixels),
        "largest_group_ink_pixels": int(largest_group_ink),
        "owned_pixels": owned_pixels,
        "owned_ink_pixels": owned_ink,
        "owned_pocket_background_pixels": owned_background,
        "owned_ink_share_of_ink": _ratio(owned_ink, ink_pixels),
        "group_mask_invariant": "pixels == pocket_area + atom_ink_pixels",
        "pixel_duplicates": int(duplicates),
        "pixel_uniqueness_full_check": bool(fully_checked),
        "pixel_uniqueness_method": (
            "exhaustive per-group multiplicity"
            if fully_checked
            else "structural (owner arrays are single-valued; bounded check skipped)"
        ),
        "stuff_pixels_by_class": {
            str(class_id): int(np.count_nonzero((semantic_array == class_id) & ink_array))
            for class_id in STUFF_IDS
        },
    }

    return {
        "atoms": atoms,
        "pocket": pocket,
        "assignment": assignment,
        "owner": owner,
        "group_of_pocket": group_of_pocket,
        "groups": groups,
        "stats": stats,
    }


def _atom_class(row: Any) -> int:
    """Known-Thing pixel-majority class of one atom; -1 when it has no vote."""

    counts = np.asarray(row, dtype=np.int64)
    if int(counts.sum()) == 0:
        return -1
    return int(np.argmax(counts))


def _mode(values: Sequence[int]) -> int:
    kept = [value for value in values if value >= 0]
    if not kept:
        return -1
    counts = np.bincount(np.asarray(kept, dtype=np.int64), minlength=len(THING_IDS))
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# candidate map + document


def build_candidate(
    topology: Mapping[str, Any], semantic: Any, group_ids: Mapping[int, int]
) -> Dict[str, Any]:
    """Candidate semantic/instance rasters plus the vectorised document.

    ``group_ids`` maps a per-image ``group_index`` of a *kept* group to its
    pilot-global group id.  A kept group takes its majority Thing class over the
    whole group mask (pocket interior included, so a filled interior carries the
    enclosing object's class).  Skipped Thing groups and orphan atoms abstain to
    35 with ``instance == -1``; Stuff keeps its saved class with the baseline
    ``instance == -1`` merge.
    """

    _require_numpy()
    owner = np.asarray(topology["owner"], dtype=np.int32)
    semantic_saved = np.asarray(semantic, dtype=np.int16)
    kept = {int(group["group_index"]): group for group in topology["groups"] if group["kept"]}
    if set(group_ids) != set(kept):
        raise TopologyError(
            f"group id map {sorted(group_ids)} does not match kept groups {sorted(kept)}"
        )

    # Deterministic local ids in vectoriser emission order: class asc, then the
    # per-image group index asc (== ascending pocket id).
    ordered: List[Tuple[int, int, int]] = []
    for class_id in THING_IDS:
        members = sorted(
            index for index, group in kept.items() if group["thing_class"] == class_id
        )
        for group_index in members:
            ordered.append((class_id, group_index, len(ordered)))

    semantic_candidate = semantic_saved.copy()
    local_instance = np.full(owner.shape, -1, dtype=np.int32)
    for class_id, group_index, local_id in ordered:
        member = owner == group_index
        semantic_candidate[member] = class_id
        local_instance[member] = local_id

    kept_owner = np.isin(owner, list(kept))
    is_stuff = (semantic_saved >= STUFF_IDS[0]) & (semantic_saved <= STUFF_IDS[-1])
    abstain = (~kept_owner) & (~is_stuff)
    semantic_candidate[abstain] = IGNORE_ID

    local_to_global = {
        local_id: int(group_ids[group_index])
        for _, group_index, local_id in ordered
    }

    affine = _affine_terms(owner.shape[1], owner.shape[0])
    document = vectorizer.vectorize_masks(
        semantic_candidate,
        local_instance,
        scale_x=affine["scale_x"],
        scale_y=affine["scale_y"],
        offset_x=affine["offset_x"],
        offset_y=affine["offset_y"],
        viewbox=submission.VIEWBOX,
        tol_px=TOL_PX,
    )
    document = _remap_document(document, local_to_global, ordered)

    global_instance = np.full(owner.shape, -1, dtype=np.int32)
    for local_id, global_id in local_to_global.items():
        global_instance[local_instance == local_id] = global_id

    return {
        "document": document,
        "semantic": semantic_candidate,
        "semantic_saved": semantic_saved,
        "instance": global_instance,
        "local_instance": local_instance,
        "local_to_global": {int(k): int(v) for k, v in local_to_global.items()},
        "affine": affine,
    }


def _remap_document(
    document: Mapping[str, Any],
    local_to_global: Mapping[int, int],
    ordered: Sequence[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Rewrite vectoriser-local instance ids to pilot-global group ids."""

    instance_ids = [int(value) for value in document["instanceIds"]]
    present = sorted({value for value in instance_ids if value >= 0})
    if present != sorted(local_to_global):
        raise TopologyError(
            f"vectoriser instance ids {present} do not match candidate groups "
            f"{sorted(local_to_global)}"
        )
    expected_class = {local_id: class_id for class_id, _, local_id in ordered}
    for semantic, instance in zip(document["semanticIds"], instance_ids):
        if instance < 0:
            continue
        if expected_class[instance] != int(semantic):
            raise TopologyError(
                f"document/group class mismatch at local instance {instance}"
            )
    result = dict(document)
    result["instanceIds"] = [
        local_to_global[value] if value >= 0 else -1 for value in instance_ids
    ]
    result["obj_cts"] = [
        [
            entry[0],
            entry[1],
            entry[2],
            local_to_global[int(entry[3])] if int(entry[3]) >= 0 else -1,
        ]
        for entry in document["obj_cts"]
    ]
    errors = submission.validate_document(result)
    if errors:
        raise TopologyError("invalid_remapped_document:" + ",".join(errors))
    return result


def _affine_terms(width: int, height: int) -> Dict[str, float]:
    scale, letterbox_x, letterbox_y = submission.viewbox_transform(
        int(width), int(height), submission.VIEWBOX
    )
    inverse = 1.0 / scale
    return {
        "scale_x": inverse,
        "scale_y": inverse,
        "offset_x": -letterbox_x * inverse,
        "offset_y": -letterbox_y * inverse,
        "pixels_per_view_unit": float(scale),
        "letterbox_x": float(letterbox_x),
        "letterbox_y": float(letterbox_y),
    }


# ---------------------------------------------------------------------------
# gate


def feasibility_gate(pocket_counts: Sequence[int], candidate_total: int) -> Dict[str, Any]:
    """Whole-run cardinality gate; never a per-image filter, never a sweep."""

    values = [int(value) for value in pocket_counts]
    median = float(np.median(values)) if values else None
    inflated = bool(median is not None and median > KILL_MEDIAN_POCKETS)
    empty = int(candidate_total) <= 0
    return {
        "median_pockets": median,
        "pocket_count_values": values,
        "kill_median_pockets": float(KILL_MEDIAN_POCKETS),
        "killed_by_pocket_inflation": inflated,
        "candidate_pockets_total": int(candidate_total),
        "killed_by_no_candidate": empty,
        "killed": bool(inflated or empty),
        "scope": "single whole-run gate; no per-image filter, no sweep",
    }


# ---------------------------------------------------------------------------
# pilot


def run_pilot(
    arm_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    *,
    write: bool = True,
) -> Dict[str, Any]:
    """Analyse the 24 fit PNGs, gate, and build artifacts only if the gate passes."""

    _require_numpy()
    records = load_records(arm_dir)
    root = os.path.abspath(output_dir or DEFAULT_OUTPUT_DIR)
    eval_dir = os.path.join(root, "evaluation")

    prepared: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    for record in records:
        semantic = load_semantic(record["png_path"], record["prediction_path"])
        ink = ink_support(record["png_path"])
        topology = topology_from_arrays(ink, semantic)
        stats = dict(topology["stats"])
        stats["sample_id"] = record["sample_id"]
        stats["source_size"] = [int(topology["atoms"].shape[1]), int(topology["atoms"].shape[0])]
        stats["stuff_pixels_total"] = int(sum(stats["stuff_pixels_by_class"].values()))
        per_image.append(stats)
        prepared.append(
            {"record": record, "topology": topology, "semantic": semantic}
        )

    gate = feasibility_gate(
        [int(item["pocket_count"]) for item in per_image],
        sum(int(item["candidate_pocket_count"]) for item in per_image),
    )
    distribution = _distribution(per_image)

    if gate["killed"]:
        reason = (
            "pocket inflation: median enclosed pockets per image is above the "
            "whole-run kill threshold"
            if gate["killed_by_pocket_inflation"]
            else "no candidate pocket: no enclosed background component owns an ink atom"
        )
        report = {
            "schema": FEASIBILITY_SCHEMA,
            "created_utc": _utc_now(),
            "passed": False,
            "gt_read": False,
            "samples": len(records),
            "all_png_persisted": False,
            "pockets_total": int(distribution["pocket_count"]["total"] or 0),
            "pockets_majority_flip": 0,
            "class_changes": 0,
            "pixel_changes": 0,
            "arm": ARM,
            "role": ROLE,
            "killed": True,
            "gate": gate,
            "distribution": distribution,
            "stop_reason": reason,
            "diagnostics_note": _diagnostics_note(),
            "next": "stop; no vectorisation, no candidate maps or documents, no evaluation",
            "policy": _policy(),
            "images": per_image,
        }
        if write:
            _write_json(os.path.join(eval_dir, "feasibility.json"), report)
        return report

    # Gate passed: build the 24 candidates from PNG topology only.
    next_group_id = 0
    built: List[Dict[str, Any]] = []
    for item in prepared:
        topology = item["topology"]
        group_ids: Dict[int, int] = {}
        for group in topology["groups"]:
            if not group["kept"]:
                continue
            group_ids[int(group["group_index"])] = next_group_id
            next_group_id += 1
        candidate = build_candidate(topology, item["semantic"], group_ids)
        candidate["group_ids"] = group_ids
        built.append({**item, "candidate": candidate})

        stats = next(row for row in per_image if row["sample_id"] == item["record"]["sample_id"])
        stats["group_ids"] = [group_ids[int(g["group_index"])] for g in topology["groups"] if g["kept"]]
        stats["document_primitives"] = len(candidate["document"]["commands"])
        stats["document_instances"] = len(candidate["document"]["obj_cts"])
        stats["assigned_pixels"] = int((candidate["instance"] >= 0).sum())
        stats["pixel_changes"] = int(
            np.count_nonzero(candidate["semantic"] != candidate["semantic_saved"])
        )
        stats["semantic_rows_changed"] = int(
            np.count_nonzero(
                np.any(candidate["semantic"] != candidate["semantic_saved"], axis=1)
            )
        )
        stats["pockets_majority_flip"] = int(
            sum(
                1
                for g in topology["groups"]
                if g["kept"] and g["thing_vote_share"] < 0.5
            )
        )
        stats["class_changes"] = int(
            sum(
                1
                for g in topology["groups"]
                if g["kept"]
                and g["atom_mode_class"] >= 0
                and g["atom_mode_class"] != g["thing_class"]
            )
        )

    feasibility = {
        "schema": FEASIBILITY_SCHEMA,
        "created_utc": _utc_now(),
        "passed": True,
        "gt_read": False,
        "samples": len(records),
        "all_png_persisted": True,
        "pockets_total": int(sum(int(row["candidate_pocket_count"]) for row in per_image)),
        "pockets_majority_flip": int(sum(int(row["pockets_majority_flip"]) for row in per_image)),
        "class_changes": int(sum(int(row["class_changes"]) for row in per_image)),
        "pixel_changes": int(sum(int(row["pixel_changes"]) for row in per_image)),
        "arm": ARM,
        "role": ROLE,
        "killed": False,
        "gate": gate,
        "distribution": distribution,
        "policy": _policy(),
        "images": per_image,
    }

    if write:
        _persist(eval_dir, built, per_image)
        _write_json(os.path.join(eval_dir, "feasibility.json"), feasibility)
        _write_json(
            os.path.join(eval_dir, "diagnostics.json"),
            {
                "schema": DIAGNOSTICS_SCHEMA,
                "arm": ARM,
                "created_utc": _utc_now(),
                "gt_read": False,
                "distribution": distribution,
                "images": per_image,
            },
        )
    return feasibility


def _diagnostics_note() -> Dict[str, Any]:
    """How to read the per-image pixel counters in the killed report.

    A group is *enclosed pocket background union assigned ink atoms*, so its
    pixel count can exceed the image's ink pixels: the enclosed interior is
    background that belongs to the object.  ``largest_group_share`` divides by
    owned pixels (ink + pocket background); ``largest_group_share_of_ink`` and
    ``owned_ink_share_of_ink`` divide by ink pixels only.
    """

    return {
        "group_mask": "pocket_background UNION assigned ink atoms (disjoint parts)",
        "invariant": "group pixels == pocket_area + atom_ink_pixels, checked per group",
        "owned_pixels_can_exceed_ink": True,
        "off_pocket_background_owned": "always 0 (asserted; would raise TopologyError)",
        "shares": {
            "largest_group_share": "of owned pixels (ink + enclosed pocket background)",
            "largest_group_share_of_ink": "of image ink pixels",
            "owned_ink_share_of_ink": "of image ink pixels (atom part only)",
        },
        "status": "valid: the earlier reading of 'group pixels > ink' as a bug was a "
        "denominator misunderstanding, now labelled explicitly",
    }


def _policy() -> Dict[str, Any]:
    return {
        "atoms": "8-connected components of shared image ink support",
        "pockets": "4-connected background components touching no image border",
        "assignment": "max 4-neighbour boundary contact; ties to the smallest pocket id",
        "orphans": "an atom with no enclosed-pocket contact is omitted (abstention)",
        "class": "majority over saved semantic known Thing votes 0..29; no vote -> skip",
        "stuff": "baseline full same-class merge unchanged (instance -1)",
        "group_mask": (
            "enclosed pocket background union its assigned ink atoms; the "
            "interior background is part of the object, so a filled object's "
            "group exceeds its ink pixels"
        ),
        "gt_read": False,
        "svg_read": False,
        "thresholds": "none (no size, confidence or per-image cardinality filter)",
        "small_pocket_px": int(SMALL_POCKET_PX),
        "uniqueness": "each atom owns at most one pocket; each pixel one group",
    }


def _distribution(per_image: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def column(name: str) -> List[float]:
        return [float(item[name]) for item in per_image if item.get(name) is not None]

    pockets = [int(item["pocket_count"]) for item in per_image]
    shares = [float(item["orphan_ink_share"] or 0.0) for item in per_image]
    largest_atom = [float(item["largest_atom_ink_share"] or 0.0) for item in per_image]
    largest_group = [float(item["largest_group_share"] or 0.0) for item in per_image]
    largest_group_of_ink = [
        float(item["largest_group_share_of_ink"] or 0.0) for item in per_image
    ]
    owned_of_ink = [
        float(item["owned_ink_share_of_ink"] or 0.0) for item in per_image
    ]

    def summary(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"median": None, "max": None}
        return {
            "median": _percentile(values, 50),
            "max": round(float(max(values)), 6),
        }

    return {
        "pocket_count": {
            "min": int(min(pockets)) if pockets else None,
            "p10": _percentile(pockets, 10),
            "median": _percentile(pockets, 50),
            "p90": _percentile(pockets, 90),
            "max": int(max(pockets)) if pockets else None,
            "total": int(sum(pockets)),
        },
        "candidate_pockets_total": int(
            sum(int(item["candidate_pocket_count"]) for item in per_image)
        ),
        "group_count_total": int(sum(int(item["group_count"]) for item in per_image)),
        "skipped_group_count_total": int(
            sum(int(item["skipped_group_count"]) for item in per_image)
        ),
        "orphan_atom_total": int(
            sum(int(item["orphan_atom_count"]) for item in per_image)
        ),
        "orphan_ink_share": summary(shares),
        "largest_atom_ink_share": summary(largest_atom),
        "largest_group_share": summary(largest_group),
        "largest_group_share_of_ink": summary(largest_group_of_ink),
        "owned_ink_share_of_ink": summary(owned_of_ink),
        "share_denominators": {
            "largest_group_share": "owned pixels (ink + enclosed pocket background)",
            "largest_group_share_of_ink": "image ink pixels",
            "owned_ink_share_of_ink": "image ink pixels (atoms only)",
        },
        "pockets_below_small_total": int(sum(column("pockets_below_small"))),
        "atoms_multi_pocket_total": int(sum(column("atoms_multi_pocket"))),
        "pixel_duplicates_total": int(sum(column("pixel_duplicates"))),
        "full_uniqueness_checked_images": int(
            sum(1 for item in per_image if item["pixel_uniqueness_full_check"])
        ),
    }


def _persist(
    eval_dir: str,
    built: Sequence[Mapping[str, Any]],
    per_image: Sequence[Mapping[str, Any]],
) -> None:
    _require_numpy()
    predictions_dir = os.path.join(eval_dir, "predictions")
    documents_dir = os.path.join(eval_dir, "documents")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(documents_dir, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for item, stats in zip(built, per_image):
        record = item["record"]
        candidate = item["candidate"]
        topology = item["topology"]
        slug = record["sample_id"].replace(":", "_")
        npz_path = os.path.join(predictions_dir, slug + ".npz")
        document_path = os.path.join(documents_dir, slug + ".json")
        np.savez_compressed(
            npz_path,
            semantic=candidate["semantic"].astype(np.int16),
            semantic_saved=candidate["semantic_saved"].astype(np.int16),
            instance=candidate["instance"].astype(np.int32),
            atom=topology["atoms"].astype(np.int32),
            pocket=topology["pocket"].astype(np.int32),
            source_size=np.asarray(
                [topology["atoms"].shape[1], topology["atoms"].shape[0]],
                dtype=np.int32,
            ),
        )
        _write_json(
            document_path,
            {
                "schema": DOCUMENT_SCHEMA,
                "arm": ARM,
                "sample_id": record["sample_id"],
                "decoder": ARM,
                "semantic_source": (
                    "saved joint_semantic_cc8 prediction (PNG-derived) plus "
                    "pocket majority relabelling"
                ),
                "instance_source": "topology_pocket_atoms_v1 (PNG ink topology only)",
                "source_size": [
                    int(topology["atoms"].shape[1]),
                    int(topology["atoms"].shape[0]),
                ],
                "tol_px": TOL_PX,
                "affine": candidate["affine"],
                "document": candidate["document"],
            },
        )
        samples.append(
            {
                "sample_id": record["sample_id"],
                "png_path": record["png_path"],
                "document": os.path.join("documents", slug + ".json").replace("\\", "/"),
                "semantic_rows_changed": int(stats["semantic_rows_changed"]),
                "pockets": {
                    "pockets": int(stats["candidate_pocket_count"]),
                    "pockets_flipped": int(stats["pockets_majority_flip"]),
                    "changed_pixels": int(stats["pixel_changes"]),
                    "class_changes": int(stats["class_changes"]),
                },
            }
        )
        for group in topology["groups"]:
            if not group["kept"]:
                continue
            groups.append(
                {
                    "group_id": int(candidate["group_ids"][int(group["group_index"])]),
                    "sample_id": record["sample_id"],
                    "kind": "thing_pocket",
                    "pocket_id": int(group["pocket_id"]),
                    "pocket_area": int(group["pocket_area"]),
                    "atom_count": int(group["atom_count"]),
                    "atom_ids": list(group["atom_ids"]),
                    "atom_ink_pixels": int(group["atom_ink_pixels"]),
                    "pixels": int(group["pixels"]),
                    "thing_class": int(group["thing_class"]),
                    "thing_votes": int(group["thing_votes"]),
                    "thing_vote_share": float(group["thing_vote_share"]),
                    "thing_vote_tie": bool(group["thing_vote_tie"]),
                    "atom_mode_class": int(group["atom_mode_class"]),
                }
            )
        for class_id in STUFF_IDS:
            groups.append(
                {
                    "group_id": None,
                    "sample_id": record["sample_id"],
                    "kind": "stuff_class_unchanged",
                    "stuff_class": int(class_id),
                    "pixels": int(stats["stuff_pixels_by_class"][str(class_id)]),
                    "instance": -1,
                }
            )

    _write_json(
        os.path.join(eval_dir, "predictions_manifest.json"),
        {
            "schema": PREDICTIONS_MANIFEST_SCHEMA,
            "route": "topology_instances",
            "created_utc": _utc_now(),
            "gt_read_during_prediction": False,
            "saved_semantics_reused": True,
            "arm": ARM,
            "role": ROLE,
            "predictions_dir": "predictions",
            "documents_dir": "documents",
            "tol_px": TOL_PX,
            "vectoriser": "png_raster_vectorize.vectorize_masks (unchanged)",
            "samples": samples,
        },
    )
    _write_json(
        os.path.join(eval_dir, "groups.json"),
        {
            "schema": GROUPS_SCHEMA,
            "arm": ARM,
            "created_utc": _utc_now(),
            "group_id_scope": "pilot-global, unique across all 24 samples",
            "stuff_groups": "class-wide baseline merge; instance stays -1",
            "groups": groups,
        },
    )


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cadvec.recognition_topology_instances",
        description=(
            "Class-agnostic PNG ink-topology instance feasibility pilot "
            "(fit only, no GT, no training)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="24-image feasibility run")
    run_parser.add_argument("--arm-dir", default=DEFAULT_ARM_DIR)
    run_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the report without writing any file",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "run":
        return 2
    report = run_pilot(args.arm_dir, args.output_dir, write=not args.dry_run)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "samples": report["samples"],
                "pockets_total": report["pockets_total"],
                "gate": report["gate"],
                "pocket_count": report["distribution"]["pocket_count"],
                "group_count_total": report["distribution"]["group_count_total"],
                "orphan_ink_share": report["distribution"]["orphan_ink_share"],
                "largest_atom_ink_share": report["distribution"]["largest_atom_ink_share"],
                "largest_group_share": report["distribution"]["largest_group_share"],
                "pixel_duplicates_total": report["distribution"]["pixel_duplicates_total"],
                "stop_reason": report.get("stop_reason"),
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "ARM",
    "DEFAULT_ARM_DIR",
    "DEFAULT_OUTPUT_DIR",
    "FEASIBILITY_SCHEMA",
    "KILL_MEDIAN_POCKETS",
    "SMALL_POCKET_PX",
    "TopologyError",
    "load_records",
    "load_semantic",
    "ink_support",
    "label_atoms",
    "label_pockets",
    "atom_pocket_contacts",
    "assign_atoms",
    "topology_from_arrays",
    "build_candidate",
    "feasibility_gate",
    "run_pilot",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
