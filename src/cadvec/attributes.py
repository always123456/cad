"""Raw label-attribute reading and boundary validation.

Facts vs. policy
----------------
* Reading ``semantic-id`` / ``instance-id`` (and configured aliases) is a fact
  read from the local training SVG.
* Deciding that raw value 1 means platform class 0, that raw 35 means
  "background", or that a class is Thing vs. Stuff is *policy*.  None of that
  is hard-coded here.  ``LabelConfig`` is empty by default and every rule must
  be supplied through a JSON config file.

The default configuration also refuses to invent placeholders: a missing
attribute yields ``value = None`` plus a diagnostic, not 35 or -1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .diagnostics import Diagnostic

_INT_RE = re.compile(r"^[+-]?\d+$")

SEMANTIC = "semantic"
INSTANCE = "instance"

DEFAULT_SEMANTIC_ALIASES: Tuple[str, ...] = (
    "semantic-id",
    "semanticId",
    "semantic_id",
    "semantic",
)
DEFAULT_INSTANCE_ALIASES: Tuple[str, ...] = (
    "instance-id",
    "instanceId",
    "instance_id",
    "instance",
)

CATEGORY_THING = "thing"
CATEGORY_STUFF = "stuff"
CATEGORY_IGNORE = "ignore"
CATEGORY_VALUES = (CATEGORY_THING, CATEGORY_STUFF, CATEGORY_IGNORE)


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for label aliases and validation boundaries.

    Defaults are intentionally empty: no domain, no category policy, no
    mapping.  This keeps raw annotations raw.
    """

    semantic_aliases: Tuple[str, ...] = DEFAULT_SEMANTIC_ALIASES
    instance_aliases: Tuple[str, ...] = DEFAULT_INSTANCE_ALIASES
    semantic_domain: Optional[Tuple[int, int]] = None
    instance_min: Optional[int] = None
    missing_semantic_value: Optional[int] = None
    missing_instance_value: Optional[int] = None
    category_policy: Dict[int, str] = field(default_factory=dict)
    semantic_mapping: Optional[Dict[str, Any]] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_aliases": list(self.semantic_aliases),
            "instance_aliases": list(self.instance_aliases),
            "semantic_domain": list(self.semantic_domain)
            if self.semantic_domain
            else None,
            "instance_min": self.instance_min,
            "missing_semantic_value": self.missing_semantic_value,
            "missing_instance_value": self.missing_instance_value,
            "category_policy": {str(k): v for k, v in self.category_policy.items()},
            "semantic_mapping": self.semantic_mapping,
            "note": self.note,
        }


DEFAULT_CONFIG = LabelConfig()


def _as_tuple_of_str(value: Any, key: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def load_label_config(path: str) -> LabelConfig:
    """Load an explicit label config JSON file.

    Recognised keys: ``semantic_aliases``, ``instance_aliases``,
    ``semantic_domain`` (``[min, max]``), ``instance_min``,
    ``missing_semantic_value``, ``missing_instance_value``,
    ``category_policy`` (``{"<raw semantic id>": "thing"|"stuff"|"ignore"}``),
    ``semantic_mapping`` (for example ``{"mode": "offset", "offset": -1}``),
    ``note``.
    """

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("label config must be a JSON object")

    semantic_aliases = _as_tuple_of_str(
        payload.get("semantic_aliases", DEFAULT_SEMANTIC_ALIASES), "semantic_aliases"
    )
    instance_aliases = _as_tuple_of_str(
        payload.get("instance_aliases", DEFAULT_INSTANCE_ALIASES), "instance_aliases"
    )
    if not semantic_aliases:
        raise ValueError("semantic_aliases must not be empty")
    if not instance_aliases:
        raise ValueError("instance_aliases must not be empty")

    domain_raw = payload.get("semantic_domain")
    domain: Optional[Tuple[int, int]] = None
    if domain_raw is not None:
        if (
            not isinstance(domain_raw, (list, tuple))
            or len(domain_raw) != 2
            or not all(isinstance(item, int) for item in domain_raw)
        ):
            raise ValueError("semantic_domain must be [min, max] integers")
        if domain_raw[0] > domain_raw[1]:
            raise ValueError("semantic_domain min must not exceed max")
        domain = (int(domain_raw[0]), int(domain_raw[1]))

    instance_min = payload.get("instance_min")
    if instance_min is not None and not isinstance(instance_min, int):
        raise ValueError("instance_min must be an integer or null")

    for key in ("missing_semantic_value", "missing_instance_value"):
        value = payload.get(key)
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer or null")

    category_policy_raw = payload.get("category_policy") or {}
    if not isinstance(category_policy_raw, dict):
        raise ValueError("category_policy must be an object")
    category_policy: Dict[int, str] = {}
    for key, value in category_policy_raw.items():
        try:
            semantic_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"category_policy key {key!r} is not an integer") from exc
        if value not in CATEGORY_VALUES:
            raise ValueError(
                f"category_policy[{semantic_id!r}] must be one of {CATEGORY_VALUES}"
            )
        category_policy[semantic_id] = value

    mapping = payload.get("semantic_mapping")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("semantic_mapping must be an object")
        mode = mapping.get("mode", "none")
        if mode not in ("none", "offset"):
            raise ValueError("semantic_mapping.mode must be 'none' or 'offset'")
        if mode == "offset" and not isinstance(mapping.get("offset"), int):
            raise ValueError("semantic_mapping.offset must be an integer")

    return LabelConfig(
        semantic_aliases=semantic_aliases,
        instance_aliases=instance_aliases,
        semantic_domain=domain,
        instance_min=instance_min,
        missing_semantic_value=payload.get("missing_semantic_value"),
        missing_instance_value=payload.get("missing_instance_value"),
        category_policy=category_policy,
        semantic_mapping=mapping,
        note=str(payload.get("note", "")),
    )


@dataclass
class LabelValue:
    """One attribute read from an element, before any interpretation."""

    kind: str
    present: bool
    alias_used: Optional[str] = None
    raw: Optional[str] = None
    value: Optional[int] = None
    aliases_present: Dict[str, str] = field(default_factory=dict)
    status: str = "missing"  # ok | missing | non_integer | ambiguous

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "present": self.present,
            "raw": self.raw,
            "value": self.value,
            "alias": self.alias_used,
            "status": self.status,
        }
        if len(self.aliases_present) > 1:
            out["aliases_present"] = dict(self.aliases_present)
        return out


def read_label(attributes: Dict[str, str], aliases: Sequence[str], kind: str) -> LabelValue:
    """Read the first configured alias present in *attributes*.

    Attribute names in XML are case-sensitive, so aliases are matched exactly.
    Multiple aliases on one element are reported as ``ambiguous``.
    """

    found = {alias: attributes[alias] for alias in aliases if alias in attributes}
    if not found:
        return LabelValue(kind=kind, present=False, status="missing")

    alias_used = next(alias for alias in aliases if alias in found)
    raw = found[alias_used]
    value: Optional[int] = None
    status = "ok"
    stripped = raw.strip()
    if _INT_RE.match(stripped):
        value = int(stripped)
    else:
        status = "non_integer"

    distinct = {str(item).strip() for item in found.values()}
    if len(found) > 1 and (len(distinct) > 1 or status != "ok"):
        status = "ambiguous"

    return LabelValue(
        kind=kind,
        present=True,
        alias_used=alias_used,
        raw=raw,
        value=value,
        aliases_present=found,
        status=status,
    )


def validate_labels(
    semantic: LabelValue,
    instance: LabelValue,
    config: LabelConfig,
    *,
    file: Optional[str] = None,
    split: Optional[str] = None,
    element: Optional[str] = None,
    local_index: Optional[int] = None,
) -> Tuple[List[Diagnostic], Dict[str, Any]]:
    """Validate raw label values against *config* only.

    Returns diagnostics plus derived values that are explicitly labelled as
    derived (mapping / category), never silently substituted.
    """

    diagnostics: List[Diagnostic] = []
    derived: Dict[str, Any] = {}

    def diag(code: str, message: str, severity: str = "warning", attribute=None):
        diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                severity=severity,
                file=file,
                split=split,
                element=element,
                local_index=local_index,
                attribute=attribute,
            )
        )

    if not semantic.present:
        diag(
            "MISSING_SEMANTIC_ATTRIBUTE",
            "no configured semantic attribute found; value stays None "
            "(no placeholder is invented)",
            severity="info",
            attribute=config.semantic_aliases[0],
        )
    elif semantic.status == "non_integer":
        diag(
            "SEMANTIC_NON_INTEGER",
            f"semantic value {semantic.raw!r} is not an integer",
            severity="error",
            attribute=semantic.alias_used,
        )
    elif semantic.status == "ambiguous":
        diag(
            "DUPLICATE_SEMANTIC_ALIAS",
            "multiple semantic attribute aliases present with different values: "
            + repr(semantic.aliases_present),
            severity="error",
            attribute=semantic.alias_used,
        )
    elif (
        semantic.alias_used
        and config.semantic_aliases
        and semantic.alias_used != config.semantic_aliases[0]
    ):
        diag(
            "SEMANTIC_ALIAS_VARIANT",
            f"semantic label was read from variant attribute "
            f"{semantic.alias_used!r} (canonical: {config.semantic_aliases[0]!r})",
            severity="info",
            attribute=semantic.alias_used,
        )

    if not instance.present:
        diag(
            "MISSING_INSTANCE_ATTRIBUTE",
            "no configured instance attribute found; value stays None "
            "(no placeholder is invented)",
            severity="info",
            attribute=config.instance_aliases[0],
        )
    elif instance.status == "non_integer":
        diag(
            "INSTANCE_NON_INTEGER",
            f"instance value {instance.raw!r} is not an integer",
            severity="error",
            attribute=instance.alias_used,
        )
    elif instance.status == "ambiguous":
        diag(
            "DUPLICATE_INSTANCE_ALIAS",
            "multiple instance attribute aliases present with different values: "
            + repr(instance.aliases_present),
            severity="error",
            attribute=instance.alias_used,
        )
    elif (
        instance.alias_used
        and config.instance_aliases
        and instance.alias_used != config.instance_aliases[0]
    ):
        diag(
            "INSTANCE_ALIAS_VARIANT",
            f"instance label was read from variant attribute "
            f"{instance.alias_used!r} (canonical: {config.instance_aliases[0]!r})",
            severity="info",
            attribute=instance.alias_used,
        )

    if (
        semantic.status == "ok"
        and semantic.value is not None
        and config.semantic_domain is not None
    ):
        low, high = config.semantic_domain
        if not low <= semantic.value <= high:
            diag(
                "SEMANTIC_OUT_OF_DOMAIN",
                f"semantic value {semantic.value} outside configured domain [{low}, {high}]",
                severity="error",
                attribute=semantic.alias_used,
            )

    if (
        instance.status == "ok"
        and instance.value is not None
        and config.instance_min is not None
    ):
        if instance.value < config.instance_min:
            diag(
                "INSTANCE_OUT_OF_DOMAIN",
                f"instance value {instance.value} below configured minimum "
                f"{config.instance_min}",
                severity="error",
                attribute=instance.alias_used,
            )

    if (
        semantic.status == "ok"
        and semantic.value is not None
        and config.category_policy
    ):
        category = config.category_policy.get(semantic.value)
        if category is None:
            diag(
                "SEMANTIC_NOT_IN_CATEGORY_POLICY",
                f"semantic value {semantic.value} has no configured Thing/Stuff "
                "category; category stays unknown",
                severity="warning",
                attribute=semantic.alias_used,
            )
        else:
            derived["category"] = category
            if category == CATEGORY_THING:
                if instance.value is None or instance.value < 0:
                    diag(
                        "THING_WITHOUT_INSTANCE",
                        f"raw semantic {semantic.value} is configured as Thing but "
                        f"instance value is {instance.value!r}",
                        severity="error",
                        attribute=instance.alias_used,
                    )
            elif category in (CATEGORY_STUFF, CATEGORY_IGNORE):
                if instance.value is not None and instance.value != -1:
                    diag(
                        "STUFF_WITH_INSTANCE",
                        f"raw semantic {semantic.value} is configured as {category} "
                        f"but instance value is {instance.value}",
                        severity="error",
                        attribute=instance.alias_used,
                    )

    mapping = config.semantic_mapping
    if mapping and semantic.status == "ok" and semantic.value is not None:
        mode = mapping.get("mode", "none")
        if mode == "offset":
            offset = int(mapping["offset"])
            derived["mapped_semantic"] = semantic.value + offset
            derived["mapping_provenance"] = f"config:offset({offset:+d})"
        else:
            derived["mapping_provenance"] = "config:none"

    if config.missing_semantic_value is not None and semantic.value is None:
        derived["fallback_semantic"] = config.missing_semantic_value
        derived["fallback_semantic_provenance"] = "config:missing_semantic_value"
    if config.missing_instance_value is not None and instance.value is None:
        derived["fallback_instance"] = config.missing_instance_value
        derived["fallback_instance_provenance"] = "config:missing_instance_value"

    return diagnostics, derived
