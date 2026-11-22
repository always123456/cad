"""Split policies and file-level exclusion rules.

Two rules are enforced here and nowhere else:

1. macOS resource forks (``._*``) and ``.DS_Store`` are never samples.
2. ``test_final`` is a file-level-only split: names, sizes, digests and
   PNG/SVG pairing may be inspected, its SVG label attributes may not be
   parsed without an explicit opt-in.

Both rules are data policies, not parser conveniences; the CLI and the tests
depend on them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

RESOURCE_FORK_PREFIX = "._"
IGNORED_BASENAMES = frozenset({".DS_Store"})
IGNORED_BASENAME_PREFIXES = (".DS_Store",)

PNG_EXT = ".png"
SVG_EXT = ".svg"

ROLE_IGNORED = "ignored"
ROLE_PNG = "png"
ROLE_SVG = "svg"
ROLE_OTHER = "other"


def is_ignored_name(name: str) -> bool:
    """True for macOS resource forks and ``.DS_Store`` variants."""

    if name.startswith(RESOURCE_FORK_PREFIX):
        return True
    if name in IGNORED_BASENAMES:
        return True
    return name.startswith(IGNORED_BASENAME_PREFIXES)


def classify_name(name: str) -> str:
    """Return one of ``ignored`` / ``png`` / ``svg`` / ``other``."""

    if is_ignored_name(name):
        return ROLE_IGNORED
    ext = os.path.splitext(name)[1].lower()
    if ext == PNG_EXT:
        return ROLE_PNG
    if ext == SVG_EXT:
        return ROLE_SVG
    return ROLE_OTHER


def stem_of(name: str) -> str:
    return os.path.splitext(name)[0]


class AnnotationAccessError(RuntimeError):
    """Raised when a split policy forbids reading SVG annotations."""


@dataclass(frozen=True)
class SplitPolicy:
    name: str
    description: str
    annotation_access: str
    file_level_inventory: bool = True
    svg_annotation_read: bool = False
    local_pq: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "annotation_access": self.annotation_access,
            "file_level_inventory": self.file_level_inventory,
            "svg_annotation_read": self.svg_annotation_read,
            "local_pq": self.local_pq,
            "notes": list(self.notes),
        }


SPLIT_POLICIES = {
    "train_final": SplitPolicy(
        name="train_final",
        description="Training split with SVG labels; may be parsed for training, "
        "in-split validation and error analysis.",
        annotation_access="full-svg (training use only)",
        svg_annotation_read=True,
        notes=(
            "raw semantic-id / instance-id values are facts of the local data, "
            "not platform prediction labels",
        ),
    ),
    "test_final": SplitPolicy(
        name="test_final",
        description="Held-out split; file-level inventory only.",
        annotation_access="file-level-only (names, sizes, digests, PNG/SVG pairing)",
        svg_annotation_read=False,
        notes=(
            "do not read semantic-id / instance-id, do not compute error, "
            "do not tune or select models against this split",
            "parser access requires an explicit allow_test_annotations opt-in",
        ),
    ),
    "dev_final": SplitPolicy(
        name="dev_final",
        description="PNG-only validation split; no local ground truth.",
        annotation_access="none (PNG only)",
        svg_annotation_read=False,
        local_pq=False,
        notes=(
            "no local PQ is computed: labels are not part of the distributed split",
        ),
    ),
}


def resolve_policy(split: Optional[str]) -> Optional[SplitPolicy]:
    if not split:
        return None
    key = split.strip().rstrip("/\\")
    key = os.path.basename(key)
    return SPLIT_POLICIES.get(key)


def _split_component(value: str) -> Optional[str]:
    """Canonical split name inside *value*, or ``None`` when unrecognised."""

    text = str(value).strip().rstrip("/\\")
    if not text:
        return None
    parts = [part for part in text.replace("/", os.sep).replace("\\", os.sep).split(os.sep) if part]
    for part in reversed(parts):
        if part in SPLIT_POLICIES:
            return part
    return None


def split_of_path(path: str) -> Optional[str]:
    """Split name implied by a filesystem path (no file reads).

    Symlinks and junctions are resolved first, so a link that lives in
    ``train_final`` but points into ``test_final`` is treated as
    ``test_final``.  Hard links cannot be detected from the path and remain a
    documented limitation.
    """

    absolute = os.path.abspath(path)
    try:
        resolved = os.path.realpath(absolute)
    except OSError:  # pragma: no cover - unreadable link chain
        resolved = absolute
    return _split_component(resolved)


def ensure_annotation_access(
    path: str,
    split: Optional[str] = None,
    allow_test_annotations: bool = False,
) -> Optional[str]:
    """Guard every SVG label read.

    The decision combines the split the path physically lives in with the
    split the caller declares.  A restricted path can never be unlocked by
    declaring a permissive split, and a path/declaration conflict fails
    closed.  ``allow_test_annotations`` is the only opt-in and it applies to
    ``test_final`` alone; it is an explicit act, never implied by a renamed
    split argument.

    Returns the resolved split name (or ``None`` for files outside every
    known split) and raises :class:`AnnotationAccessError` otherwise.
    """

    declared_raw = (split or "").strip()
    declared = _split_component(declared_raw) if declared_raw else None
    if declared_raw and declared is None:
        raise AnnotationAccessError(
            f"declared split {split!r} is not a recognised split "
            f"({', '.join(sorted(SPLIT_POLICIES))}); refusing to guess access "
            "from an unknown split name."
        )

    path_split = split_of_path(path)
    if path_split and declared and path_split != declared:
        raise AnnotationAccessError(
            f"declared split {declared!r} conflicts with the path split "
            f"{path_split!r}; refusing annotation access. Move the file or "
            "declare the split it actually belongs to."
        )

    resolved = path_split or declared
    policy = resolve_policy(resolved)
    if policy is None:
        return None
    if policy.svg_annotation_read:
        return resolved
    if allow_test_annotations and resolved == "test_final":
        return resolved
    raise AnnotationAccessError(
        f"split {resolved!r} is {policy.annotation_access}; "
        "reading SVG label attributes is blocked by the local data policy. "
        "Pass allow_test_annotations=True only after the competition organiser "
        "authorises annotation access."
    )


def iter_split_files(directory: str) -> Iterable[str]:
    """Yield file names in *directory* (non-recursive), unsorted."""

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                yield entry.name
