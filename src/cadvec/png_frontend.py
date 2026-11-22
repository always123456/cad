"""PNG-only line-segment frontend (diagnostic, line-only).

This module is the optional image lane of :mod:`cadvec`.  It decodes exactly
one PNG (path or already-decoded array), composites alpha over an explicit or
recorded background, extracts a foreground mask and detects straight line
segments with OpenCV's probabilistic Hough transform.

Design boundaries enforced by this module:

* **line-only**: no arc/circle/ellipse type recovery, no semantic or instance
  labels, no "centreline" or CAD-primitive identity is claimed.
* **PNG-only, single file**: a path input opens only that file.  No directory
  pairing, no SVG, no manifest, no label JSON, no layer or source path is read.
* **raw segments**: Hough output is returned unmerged; collinear fragments stay
  separate.  Thick strokes and double edges can produce multiple segments.
* **explicit status**: a blank image is ``empty_input``; a foreground with no
  detected segment is ``no_lines_detected``; decode/input/unsupported errors
  and resource limits are distinct statuses, never a fake empty prediction.
* **optional dependency**: importing this module does not import NumPy or
  OpenCV.  They are imported per call, so the stdlib core keeps working when
  the ``image`` extra is not installed.

Schema: ``cadvec.png_frontend`` version ``1`` (see :data:`SCHEMA_VERSION`).
All returned values are strict-JSON safe (finite floats, no NaN/Inf).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Union

SCHEMA = "cadvec.png_frontend"
SCHEMA_VERSION = 1

#: Smallest/largest 8-bit sample values accepted for an explicit background.
MIN_BACKGROUND_VALUE = 0
MAX_BACKGROUND_VALUE = 255

#: Fixed PNG signature; anything else is rejected before decoding.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

STATUS_OK = "ok"
STATUS_EMPTY = "empty_input"
STATUS_NO_LINES = "no_lines_detected"
STATUS_LIMITED = "processing_limited"
STATUS_INPUT_ERROR = "input_error"
STATUS_DECODE_ERROR = "decode_error"
STATUS_UNSUPPORTED = "unsupported_image"

SUPPORTED_COLOR_TYPES = {
    0: "GRAY",  # greyscale
    2: "BGR",  # truecolour (decoded as BGR by OpenCV)
    3: "BGR",  # palette (decoded as BGR by OpenCV)
    4: "GRAY_ALPHA",  # greyscale + alpha
    6: "BGRA",  # truecolour + alpha
}

#: Background strings accepted by :class:`FrontendConfig`.
BACKGROUND_CHOICES = ("auto", "white", "black")


class PngFrontendError(Exception):
    """Base class for explicit frontend failures."""


class InvalidConfigError(PngFrontendError, ValueError):
    """A configuration value is illegal (range, type, NaN/Inf)."""


class MissingImageDependencyError(PngFrontendError, ImportError):
    """The optional ``image`` extra (NumPy + OpenCV) is not installed."""


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_int(name: str, value: Any, minimum: int, maximum: Optional[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidConfigError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise InvalidConfigError(f"{name} must be <= {maximum}, got {value!r}")
    return value


def _require_float(name: str, value: Any, minimum: float, maximum: Optional[float]) -> float:
    if not _is_finite_number(value):
        raise InvalidConfigError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if number < minimum:
        raise InvalidConfigError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and number > maximum:
        raise InvalidConfigError(f"{name} must be <= {maximum}, got {value!r}")
    return number


@dataclass(frozen=True)
class FrontendConfig:
    """Locked image-processing configuration for the PNG frontend.

    Every field is explicit and serialised into the result, so a prediction can
    always be reproduced from its own record.  Defaults are a CPU diagnostic
    baseline, not tuned competition thresholds.

    ``background`` accepts ``"auto"``, ``"white"``, ``"black"`` or an integer
    grey level ``0..255``.  The numeric level is used for every channel of the
    alpha composite, so a PNG whose background is mid-grey can be handled
    without guessing.

    ``polarity`` accepts ``"auto"``, ``"dark_on_light"`` or
    ``"light_on_dark"``.  ``"auto"`` compares dark and light deviation mass and
    records the choice; it is a recorded estimate, not a guarantee.
    """

    detector: str = "hough_p"
    polarity: str = "auto"
    background: Union[str, int] = "auto"
    foreground_threshold: int = 32

    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 40
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 5

    max_pixels: int = 40_000_000
    max_file_bytes: int = 64 * 1024 * 1024
    max_lines: int = 20_000

    def __post_init__(self) -> None:
        if self.detector != "hough_p":
            raise InvalidConfigError(
                "detector must be 'hough_p' in this version; LSD is capability-"
                f"checked only, got {self.detector!r}"
            )
        if self.polarity not in ("auto", "dark_on_light", "light_on_dark"):
            raise InvalidConfigError(
                "polarity must be 'auto', 'dark_on_light' or 'light_on_dark', "
                f"got {self.polarity!r}"
            )
        self._validate_background()
        _require_int("foreground_threshold", self.foreground_threshold, 1, 255)
        _require_float("hough_rho", self.hough_rho, 1e-6, None)
        _require_float("hough_theta_deg", self.hough_theta_deg, 1e-6, 180.0)
        _require_int("hough_threshold", self.hough_threshold, 1, None)
        _require_int("hough_min_line_length", self.hough_min_line_length, 1, None)
        _require_int("hough_max_line_gap", self.hough_max_line_gap, 0, None)
        _require_int("max_pixels", self.max_pixels, 1, None)
        _require_int("max_file_bytes", self.max_file_bytes, 1, None)
        _require_int("max_lines", self.max_lines, 1, None)

    def _validate_background(self) -> None:
        value = self.background
        if isinstance(value, bool):
            raise InvalidConfigError(
                f"background must be 'auto', 'white', 'black' or 0..255, got {value!r}"
            )
        if isinstance(value, int):
            if not MIN_BACKGROUND_VALUE <= value <= MAX_BACKGROUND_VALUE:
                raise InvalidConfigError(
                    "background grey level must be within "
                    f"{MIN_BACKGROUND_VALUE}..{MAX_BACKGROUND_VALUE}, got {value!r}"
                )
            return
        if value not in BACKGROUND_CHOICES:
            raise InvalidConfigError(
                "background must be 'auto', 'white', 'black' or 0..255, "
                f"got {value!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "polarity": self.polarity,
            "background": self.background,
            "foreground_threshold": self.foreground_threshold,
            "hough_rho": float(self.hough_rho),
            "hough_theta_deg": float(self.hough_theta_deg),
            "hough_threshold": self.hough_threshold,
            "hough_min_line_length": self.hough_min_line_length,
            "hough_max_line_gap": self.hough_max_line_gap,
            "max_pixels": self.max_pixels,
            "max_file_bytes": self.max_file_bytes,
            "max_lines": self.max_lines,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrontendConfig":
        if not isinstance(data, dict):
            raise InvalidConfigError("config must be a mapping")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise InvalidConfigError(f"unknown config key(s): {', '.join(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------
# Dependency and input helpers
# ---------------------------------------------------------------------------


def image_dependencies_available() -> bool:
    """Return True when NumPy and OpenCV can both be imported."""

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _import_deps():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised via subprocess test
        raise MissingImageDependencyError(
            "the PNG frontend needs the optional 'image' extra "
            "(numpy==2.5.4, opencv-python-headless==4.14.0.94, Python >= 3.12); "
            f"import failed: {exc}"
        ) from exc
    return np, cv2


def capability_block() -> Dict[str, Any]:
    """Static, schema-versioned statement of what this frontend can report."""

    return {
        "scope": "line_only",
        "line_only": True,
        "detector": "hough_p",
        "supports": ["line_segment"],
        "not_supported": [
            "arc",
            "circle",
            "ellipse",
            "curve_type_recovery",
            "semantic_label",
            "instance_id",
            "cad_primitive_identity",
            "centerline_precision",
        ],
        "notes": [
            "raw probabilistic-Hough segments; collinear fragments are not merged",
            "a straight local fit on a curved stroke does not recover its type",
            "no semantic or instance field is populated, ever",
            "LSD is installed and capability-checked but is not a second branch",
        ],
    }


def _error_result(
    status: str,
    code: str,
    message: str,
    config: FrontendConfig,
    *,
    file_name: Optional[str] = None,
    file_bytes: Optional[int] = None,
    file_bytes_source: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    elapsed_ms: float = 0.0,
) -> Dict[str, Any]:
    """Full-shaped result for a rejected input, never a fake empty prediction."""

    pixels = int(width * height) if width is not None and height is not None else None
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_reason": code,
        "capability": capability_block(),
        "config": config.to_dict(),
        "input": {
            "source": "path" if file_name is not None else "array",
            "file_name": file_name,
            "file_bytes": file_bytes,
            "file_bytes_source": file_bytes_source,
            "width": width,
            "height": height,
        },
        "background": None,
        "polarity": None,
        "foreground": None,
        "line_count": 0,
        "lines": [],
        "diagnostics": {"codes": [code], "notes": [message]},
        "limits": {
            "max_pixels": config.max_pixels,
            "max_file_bytes": config.max_file_bytes,
            "max_lines": config.max_lines,
            "pixels": pixels,
            "pixels_within_limit": None if pixels is None else pixels <= config.max_pixels,
            "line_limit_hit": False,
        },
        "timing_ms": round(float(elapsed_ms), 3),
        "error": {"code": code, "message": message},
    }


def _png_header(data: bytes) -> Optional[Dict[str, int]]:
    """Read width/height/bit depth/colour type from the IHDR chunk.

    Returns ``None`` when the chunk layout is invalid.  This runs before
    ``imdecode`` so an oversized or 16-bit PNG is rejected without decoding it.
    """

    if len(data) < 33 or data[12:16] != b"IHDR":
        return None
    length = int.from_bytes(data[8:12], "big")
    if length != 13:
        return None
    return {
        "width": int.from_bytes(data[16:20], "big"),
        "height": int.from_bytes(data[20:24], "big"),
        "bit_depth": data[24],
        "color_type": data[25],
        "interlace": data[28],
    }


def _compose_and_gray(np: Any, cv2: Any, image: Any, background_value: int, config: FrontendConfig):
    """Composite alpha over a grey background value and return uint8 grayscale.

    The returned alpha channel, when present, is kept in its **raw 0..255**
    scale so background estimation and composition share one convention.
    """

    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels == 1:
        return image, 1, None
    if channels == 2:
        grey = image[:, :, 0].astype(np.float32)
        alpha_raw = image[:, :, 1].astype(np.float32)
        alpha = alpha_raw / 255.0
        composed = grey * alpha + float(background_value) * (1.0 - alpha)
        return np.clip(np.rint(composed), 0, 255).astype(np.uint8), 2, alpha_raw
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 3, None
    if channels == 4:
        colour = image[:, :, :3].astype(np.float32)
        alpha_raw = image[:, :, 3].astype(np.float32)
        alpha3 = (alpha_raw / 255.0)[:, :, None]
        composed = colour * alpha3 + float(background_value) * (1.0 - alpha3)
        composed = np.clip(np.rint(composed), 0, 255).astype(np.uint8)
        return cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY), 4, alpha_raw
    raise InvalidConfigError(f"unsupported channel count: {channels}")


def _estimate_background(np: Any, grey: Any, alpha: Any) -> Dict[str, Any]:
    """Estimate the background grey level with an explicit, recorded basis.

    ``alpha`` is the raw 0..255 channel (never normalised), matching the scale
    used by composition.
    """

    if alpha is not None and float(alpha.min()) < 250.0:
        return {
            "mode": "auto",
            "selected": "white",
            "value": 255,
            "basis": "alpha transparency detected; auto default is white",
            "estimate": {
                "alpha_min": int(alpha.min()),
                "alpha_max": int(alpha.max()),
            },
            "ambiguous": True,
            "notes": [
                "transparent PNG: composition background defaults to white; pass "
                "an explicit background to remove this ambiguity",
            ],
        }

    values = grey
    if alpha is not None:
        opaque = alpha >= 250
        if bool(opaque.any()):
            values = grey[opaque]
    flat = values.reshape(-1)
    global_median = float(np.median(flat))
    height, width = grey.shape[:2]
    if height >= 3 and width >= 3:
        ring = np.concatenate(
            [
                grey[0, :].reshape(-1),
                grey[-1, :].reshape(-1),
                grey[1:-1, 0].reshape(-1),
                grey[1:-1, -1].reshape(-1),
            ]
        )
        if alpha is not None:
            ring_mask = np.concatenate(
                [
                    alpha[0, :].reshape(-1),
                    alpha[-1, :].reshape(-1),
                    alpha[1:-1, 0].reshape(-1),
                    alpha[1:-1, -1].reshape(-1),
                ]
            ) >= 250
            if bool(ring_mask.any()):
                ring = ring[ring_mask]
        border_median = float(np.median(ring))
    else:
        border_median = global_median

    selected_value = int(round(border_median))
    selected_value = max(0, min(255, selected_value))
    ambiguous = (
        96 <= selected_value <= 160
        or abs(border_median - global_median) > 32.0
    )
    if selected_value == 255:
        selected: Union[str, int] = "white"
    elif selected_value == 0:
        selected = "black"
    else:
        selected = selected_value
    return {
        "mode": "auto",
        "selected": selected,
        "value": selected_value,
        "basis": "median grey of the 1-pixel border ring (opaque pixels only)",
        "estimate": {
            "border_median": round(border_median, 3),
            "global_median": round(global_median, 3),
        },
        "ambiguous": ambiguous,
        "notes": [] if not ambiguous else [
            "background is near mid-grey or differs from the global median; "
            "confirm with an explicit background value",
        ],
    }


def _foreground_mask(np: Any, grey: Any, background_value: int, config: FrontendConfig):
    """Return (mask, polarity metadata) using recorded deviations from background."""

    grey_f = grey.astype(np.int16)
    dark_deviation = int(background_value) - grey_f
    light_deviation = grey_f - int(background_value)
    threshold = config.foreground_threshold
    dark_count = int(np.count_nonzero(dark_deviation > threshold))
    light_count = int(np.count_nonzero(light_deviation > threshold))

    if config.polarity == "dark_on_light":
        selected = "dark_on_light"
        mask = dark_deviation > threshold
        ambiguous = light_count > 0 and light_count * 4 >= max(dark_count, 1)
    elif config.polarity == "light_on_dark":
        selected = "light_on_dark"
        mask = light_deviation > threshold
        ambiguous = dark_count > 0 and dark_count * 4 >= max(light_count, 1)
    else:
        if dark_count >= light_count:
            selected = "dark_on_light"
            mask = dark_deviation > threshold
        else:
            selected = "light_on_dark"
            mask = light_deviation > threshold
        chosen = max(dark_count, light_count)
        other = min(dark_count, light_count)
        ambiguous = other > 0 and other * 4 >= max(chosen, 1)

    mask_u8 = mask.astype(np.uint8) * 255
    metadata = {
        "configured": config.polarity,
        "selected": selected,
        "ambiguous": bool(ambiguous),
        "dark_deviation_pixels": dark_count,
        "light_deviation_pixels": light_count,
        "threshold": threshold,
    }
    return mask_u8, metadata


def _detect_lines(np: Any, cv2: Any, mask: Any, width: int, height: int, config: FrontendConfig):
    theta = math.radians(float(config.hough_theta_deg))
    raw = cv2.HoughLinesP(
        mask,
        float(config.hough_rho),
        theta,
        int(config.hough_threshold),
        None,
        int(config.hough_min_line_length),
        int(config.hough_max_line_gap),
    )
    if raw is None:
        return []
    segments: List[Dict[str, Any]] = []
    for row in raw.reshape(-1, 4):
        x1, y1, x2, y2 = (float(int(value)) for value in row)
        x1 = min(max(x1, 0.0), float(width - 1))
        x2 = min(max(x2, 0.0), float(width - 1))
        y1 = min(max(y1, 0.0), float(height - 1))
        y2 = min(max(y2, 0.0), float(height - 1))
        length = math.hypot(x2 - x1, y2 - y1)
        if not (math.isfinite(length) and length > 0.0):
            continue
        segments.append(
            {
                "index": len(segments),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
                "length_px": round(length, 3),
            }
        )
    return segments


def _base_result(
    status: str,
    status_reason: Optional[str],
    config: FrontendConfig,
    input_block: Dict[str, Any],
    *,
    background: Optional[Dict[str, Any]] = None,
    foreground: Optional[Dict[str, Any]] = None,
    polarity: Optional[Dict[str, Any]] = None,
    lines: Optional[List[Dict[str, Any]]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
    elapsed_ms: float = 0.0,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_reason": status_reason,
        "capability": capability_block(),
        "config": config.to_dict(),
        "input": input_block,
        "background": background,
        "polarity": polarity,
        "foreground": foreground,
        "line_count": len(lines or []),
        "lines": lines or [],
        "diagnostics": diagnostics or {"codes": [], "notes": []},
        "limits": limits or {},
        "timing_ms": round(float(elapsed_ms), 3),
        "error": None,
    }


def _analyze(
    np: Any,
    cv2: Any,
    image: Any,
    config: FrontendConfig,
    *,
    input_block: Dict[str, Any],
    started: float,
) -> Dict[str, Any]:
    """Shared array analysis for both the path and array entry points."""

    height, width = image.shape[:2]
    input_block = dict(input_block)
    limits = {
        "max_pixels": config.max_pixels,
        "max_file_bytes": config.max_file_bytes,
        "max_lines": config.max_lines,
        "pixels": int(width * height),
        "pixels_within_limit": int(width * height) <= config.max_pixels,
        "line_limit_hit": False,
    }

    if width * height > config.max_pixels:
        return _base_result(
            STATUS_LIMITED,
            "pixel_limit_exceeded",
            config,
            input_block,
            limits=limits,
            diagnostics={
                "codes": ["pixel_limit_exceeded"],
                "notes": [
                    "image exceeds max_pixels; no resize or crop is applied and "
                    "no line result is claimed"
                ],
            },
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    # Resolve the composition background before touching alpha.
    if config.background == "auto":
        # A provisional grey view is enough for background estimation; alpha is
        # only ignored for that step, never dropped from the final composite.
        provisional, channels, alpha = _compose_and_gray(np, cv2, image, 255, config)
        background = _estimate_background(np, provisional, alpha)
    else:
        if config.background == "white":
            background_value = 255
            background = {
                "mode": "explicit",
                "selected": "white",
                "value": 255,
                "basis": "configured",
                "estimate": None,
                "ambiguous": False,
                "notes": [],
            }
        elif config.background == "black":
            background_value = 0
            background = {
                "mode": "explicit",
                "selected": "black",
                "value": 0,
                "basis": "configured",
                "estimate": None,
                "ambiguous": False,
                "notes": [],
            }
        else:
            background_value = int(config.background)
            background = {
                "mode": "explicit",
                "selected": background_value,
                "value": background_value,
                "basis": "configured",
                "estimate": None,
                "ambiguous": False,
                "notes": [],
            }
    if background["mode"] == "auto":
        background_value = int(background["value"])

    grey, channels, alpha = _compose_and_gray(np, cv2, image, background_value, config)
    input_block["channels"] = channels
    input_block["color_space"] = {
        1: "GRAY",
        2: "GRAY_ALPHA",
        3: "BGR",
        4: "BGRA",
    }[channels]

    mask, polarity = _foreground_mask(np, grey, background_value, config)
    foreground_count = int(np.count_nonzero(mask))
    foreground = {
        "pixel_count": foreground_count,
        "ratio": round(foreground_count / float(width * height), 6),
        "threshold": config.foreground_threshold,
    }

    codes: List[str] = ["no_collinear_merge", "double_edge_or_thick_stroke_possible"]
    notes: List[str] = [
        "HoughLinesP segments are returned raw; collinear fragments are not merged",
        "thick strokes and outlined strokes may yield several nearby segments",
        "arcs/circles/ellipses are not type-recovered even if locally line-like",
    ]
    if background.get("ambiguous"):
        codes.append("background_ambiguous")
        notes.extend(background.get("notes", []))
    if polarity.get("ambiguous"):
        codes.append("polarity_ambiguous")
        notes.append(
            "both dark and light deviation mass are present; confirm polarity or "
            "background explicitly"
        )
    if channels == 1:
        codes.append("single_channel_gray_input")

    diagnostics = {"codes": codes, "notes": notes}

    if foreground_count == 0:
        return _base_result(
            STATUS_EMPTY,
            "no_foreground_pixels",
            config,
            input_block,
            background=background,
            polarity=polarity,
            foreground=foreground,
            diagnostics=diagnostics,
            limits=limits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        lines = _detect_lines(np, cv2, mask, width, height, config)
    except cv2.error as exc:  # pragma: no cover - defensive
        result = _error_result(
            STATUS_DECODE_ERROR,
            "detection_failed",
            f"OpenCV line detection failed: {exc}",
            config,
            width=width,
            height=height,
        )
        result["timing_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    if len(lines) > config.max_lines:
        limits["line_limit_hit"] = True
        return _base_result(
            STATUS_LIMITED,
            "line_count_limit_exceeded",
            config,
            input_block,
            background=background,
            polarity=polarity,
            foreground=foreground,
            diagnostics={
                "codes": codes + ["line_count_limit_exceeded"],
                "notes": notes
                + [
                    f"detected {len(lines)} raw segments, above max_lines="
                    f"{config.max_lines}; results are withheld rather than silently "
                    "truncated"
                ],
            },
            limits=limits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    if not lines:
        return _base_result(
            STATUS_NO_LINES,
            "foreground_present_but_no_lines",
            config,
            input_block,
            background=background,
            polarity=polarity,
            foreground=foreground,
            diagnostics=diagnostics,
            limits=limits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    return _base_result(
        STATUS_OK,
        None,
        config,
        input_block,
        background=background,
        polarity=polarity,
        foreground=foreground,
        lines=lines,
        diagnostics=diagnostics,
        limits=limits,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def predict_png(path: Union[str, "os.PathLike[str]"], config: Optional[FrontendConfig] = None) -> Dict[str, Any]:
    """Predict raw line segments from exactly one PNG file.

    Only ``path`` is opened.  Nothing else in its directory is listed or read,
    so a PNG-only directory works and paired SVGs / manifests are invisible to
    this entry point.
    """

    if config is None:
        config = FrontendConfig()
    if not isinstance(config, FrontendConfig):
        raise TypeError("config must be a FrontendConfig instance")
    np, cv2 = _import_deps()
    started = time.perf_counter()

    file_name = os.path.basename(os.fspath(path))
    if os.path.isdir(path):
        return _error_result(
            STATUS_INPUT_ERROR, "not_a_file", f"path is a directory: {path}", config,
            file_name=file_name,
        )
    file_bytes: Optional[int] = None
    file_bytes_source: Optional[str] = None
    try:
        with open(path, "rb") as handle:
            try:
                stat_size = os.fstat(handle.fileno()).st_size
            except OSError:  # e.g. a stream without fstat: fall back to a bounded read
                stat_size = None
            if stat_size is not None and stat_size > config.max_file_bytes:
                return _error_result(
                    STATUS_LIMITED,
                    "file_size_limit_exceeded",
                    f"file stat size {stat_size} bytes is above max_file_bytes="
                    f"{config.max_file_bytes}; nothing was read or decoded",
                    config,
                    file_name=file_name,
                    file_bytes=stat_size,
                    file_bytes_source="stat",
                )
            # Bounded read: never pull more than one byte past the limit.
            data = handle.read(config.max_file_bytes + 1)
            file_bytes = len(data)
            if file_bytes > config.max_file_bytes:
                file_bytes_source = "observed_prefix"
            elif stat_size is not None:
                file_bytes_source = "stat"
            else:
                file_bytes_source = "read_to_eof"
    except FileNotFoundError:
        return _error_result(
            STATUS_INPUT_ERROR, "file_not_found", f"no such file: {path}", config,
            file_name=file_name,
        )
    except IsADirectoryError:
        return _error_result(
            STATUS_INPUT_ERROR, "not_a_file", f"path is a directory: {path}", config,
            file_name=file_name,
        )
    except OSError as exc:
        return _error_result(
            STATUS_INPUT_ERROR,
            "unreadable",
            f"cannot read {path}: {type(exc).__name__}: {exc}",
            config,
            file_name=file_name,
        )

    if file_bytes > config.max_file_bytes:
        return _error_result(
            STATUS_LIMITED,
            "file_size_limit_exceeded",
            f"file read prefix is {file_bytes} bytes, above max_file_bytes="
            f"{config.max_file_bytes}; it was not decoded",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
        )

    if not data.startswith(PNG_SIGNATURE):
        return _error_result(
            STATUS_INPUT_ERROR,
            "not_png",
            "file does not start with the PNG signature",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
        )

    header = _png_header(data)
    if header is None:
        return _error_result(
            STATUS_DECODE_ERROR,
            "invalid_ihdr",
            "PNG header chunk is missing or malformed",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
        )
    if header["bit_depth"] != 8:
        return _error_result(
            STATUS_UNSUPPORTED,
            "unsupported_bit_depth",
            f"only 8-bit PNG is supported, got bit depth {header['bit_depth']}",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )
    if header["color_type"] not in SUPPORTED_COLOR_TYPES:
        return _error_result(
            STATUS_UNSUPPORTED,
            "unsupported_color_type",
            f"unsupported PNG colour type {header['color_type']}",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )
    if header["width"] <= 0 or header["height"] <= 0:
        return _error_result(
            STATUS_DECODE_ERROR,
            "invalid_dimensions",
            f"PNG dimensions are not positive: {header['width']}x{header['height']}",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
        )
    if header["width"] * header["height"] > config.max_pixels:
        return _error_result(
            STATUS_LIMITED,
            "pixel_limit_exceeded",
            f"image {header['width']}x{header['height']} exceeds max_pixels="
            f"{config.max_pixels}; it was not decoded",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )

    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:  # pragma: no cover - defensive
        return _error_result(
            STATUS_DECODE_ERROR,
            "decode_failed",
            f"OpenCV could not decode the PNG: {exc}",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )
    if image is None:
        return _error_result(
            STATUS_DECODE_ERROR,
            "decode_failed",
            "OpenCV returned no image for these PNG bytes",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )
    if image.dtype != np.uint8:
        return _error_result(
            STATUS_UNSUPPORTED,
            "unsupported_dtype",
            f"decoded dtype is {image.dtype}, only uint8 is supported",
            config,
            file_name=file_name,
            file_bytes=file_bytes,
            file_bytes_source=file_bytes_source,
            width=header["width"],
            height=header["height"],
        )

    input_block = {
        "source": "path",
        "file_name": file_name,
        "file_bytes": file_bytes,
        "file_bytes_source": file_bytes_source,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "bit_depth": 8,
        "png_bit_depth": header["bit_depth"],
        "png_color_type": header["color_type"],
        "png_interlace": header["interlace"],
        "channels": None,
        "color_space": SUPPORTED_COLOR_TYPES[header["color_type"]],
    }
    return _analyze(np, cv2, image, config, input_block=input_block, started=started)


def predict_array(image: Any, config: Optional[FrontendConfig] = None) -> Dict[str, Any]:
    """Predict raw line segments from an already-decoded uint8 image array.

    Accepts grayscale (H,W), gray+alpha (H,W,2), BGR (H,W,3) and BGRA (H,W,4)
    uint8 arrays.  Other dtypes and channel counts are rejected explicitly.
    """

    if config is None:
        config = FrontendConfig()
    if not isinstance(config, FrontendConfig):
        raise TypeError("config must be a FrontendConfig instance")
    np, cv2 = _import_deps()
    started = time.perf_counter()

    if not hasattr(image, "shape") or not hasattr(image, "dtype"):
        raise TypeError("predict_array expects a NumPy array")
    if image.dtype != np.uint8:
        return _error_result(
            STATUS_UNSUPPORTED,
            "unsupported_dtype",
            f"array dtype is {image.dtype}, only uint8 is supported",
            config,
        )
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 2, 3, 4)):
        return _error_result(
            STATUS_UNSUPPORTED,
            "unsupported_channel_layout",
            f"unsupported array shape {image.shape}",
            config,
        )
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return _error_result(
            STATUS_INPUT_ERROR,
            "empty_array",
            f"array has no pixels: {image.shape}",
            config,
        )
    channels = 1 if image.ndim == 2 else image.shape[2]
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]  # (H, W, 1) is treated as plain grayscale
    input_block = {
        "source": "array",
        "file_name": None,
        "file_bytes": None,
        "file_bytes_source": None,
        "width": int(width),
        "height": int(height),
        "bit_depth": 8,
        "png_bit_depth": None,
        "png_color_type": None,
        "png_interlace": None,
        "channels": channels,
        "color_space": {1: "GRAY", 2: "GRAY_ALPHA", 3: "BGR", 4: "BGRA"}[channels],
    }
    return _analyze(np, cv2, image, config, input_block=input_block, started=started)


def predict(source: Any, config: Optional[FrontendConfig] = None) -> Dict[str, Any]:
    """Dispatch to :func:`predict_array` or :func:`predict_png`."""

    if hasattr(source, "shape") and hasattr(source, "dtype"):
        return predict_array(source, config)
    return predict_png(source, config)


def dumps(result: Dict[str, Any], *, compact: bool = False) -> str:
    """Strict JSON serialisation (rejects NaN/Inf, keeps Unicode)."""

    if compact:
        return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)


# ---------------------------------------------------------------------------
# Module-level CLI: python -m cadvec.png_frontend IMAGE
# ---------------------------------------------------------------------------


def _background_argument(value: str) -> Union[str, int]:
    if value in BACKGROUND_CHOICES:
        return value
    try:
        number = int(value)
    except ValueError:
        return value  # left for FrontendConfig to reject with a clear message
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadvec.png_frontend",
        description="PNG-only line-segment frontend (diagnostic, line-only). "
        "Prints strict JSON to stdout; writes no files.",
    )
    parser.add_argument("image", help="path to exactly one PNG file")
    parser.add_argument("--polarity", default="auto",
                        choices=("auto", "dark_on_light", "light_on_dark"))
    parser.add_argument("--background", default="auto",
                        help="auto | white | black | 0..255 grey level")
    parser.add_argument("--foreground-threshold", type=int, default=32)
    parser.add_argument("--hough-threshold", type=int, default=40)
    parser.add_argument("--hough-rho", type=float, default=1.0)
    parser.add_argument("--hough-theta-deg", type=float, default=1.0)
    parser.add_argument("--min-line-length", type=int, default=20)
    parser.add_argument("--max-line-gap", type=int, default=5)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--compact", action="store_true",
                        help="single-line JSON instead of indented JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = FrontendConfig(
            polarity=args.polarity,
            background=_background_argument(args.background),
            foreground_threshold=args.foreground_threshold,
            hough_rho=args.hough_rho,
            hough_theta_deg=args.hough_theta_deg,
            hough_threshold=args.hough_threshold,
            hough_min_line_length=args.min_line_length,
            hough_max_line_gap=args.max_line_gap,
            max_pixels=args.max_pixels,
            max_file_bytes=args.max_file_bytes,
            max_lines=args.max_lines,
        )
        result = predict_png(args.image, config)
    except MissingImageDependencyError as exc:
        print(f"dependency error: {exc}", file=sys.stderr)
        return 4
    except InvalidConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    sys.stdout.write(dumps(result, compact=args.compact) + "\n")
    status = result["status"]
    if status in (STATUS_OK, STATUS_EMPTY, STATUS_NO_LINES):
        return 0
    if status == STATUS_LIMITED:
        return 1
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
