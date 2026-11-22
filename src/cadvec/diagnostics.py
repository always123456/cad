"""Structured diagnostics shared by the inventory and the SVG parser.

Every non-fatal irregularity is recorded as a :class:`Diagnostic` instead of
being silently ignored, so that boundary problems (missing / duplicate /
out-of-range label attributes) can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SEVERITIES = ("error", "warning", "info")


@dataclass
class Diagnostic:
    """One auditable finding about a file or an element inside it."""

    code: str
    message: str
    severity: str = "warning"
    file: Optional[str] = None
    split: Optional[str] = None
    element: Optional[str] = None
    local_index: Optional[int] = None
    attribute: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        for key in ("file", "split", "element", "local_index", "attribute"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.detail:
            out["detail"] = dict(self.detail)
        return out


def summarise(diagnostics: List[Diagnostic]) -> Dict[str, Any]:
    """Aggregate diagnostics into severity / code counters."""

    by_severity: Dict[str, int] = {sev: 0 for sev in SEVERITIES}
    by_code: Dict[str, int] = {}
    for diag in diagnostics:
        by_severity[diag.severity] = by_severity.get(diag.severity, 0) + 1
        by_code[diag.code] = by_code.get(diag.code, 0) + 1
    return {
        "total": len(diagnostics),
        "by_severity": by_severity,
        "by_code": dict(sorted(by_code.items())),
    }
