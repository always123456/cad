"""SVG geometry helpers: path tokenisation, sampling, lengths.

Everything produced here is derived and traceable: the raw ``d`` string and
the raw command list are always preserved by the caller, while this module
only adds polylines, lengths, bounding boxes and the four arc-length sample
points used by the reference submission format.

Supported path commands: ``M L H V C S Q T A Z`` in absolute and relative
form, including implicit repeated argument groups.  Anything else raises
:class:`PathDataError` and is reported as a diagnostic by the parser.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

_PATH_COMMANDS = "MLHVCSQTAZ"
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


class PathDataError(ValueError):
    """Malformed or unsupported path data."""


class _Scanner:
    """Character scanner with number / arc-flag awareness."""

    __slots__ = ("text", "index", "length")

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.length = len(text)

    def eof(self) -> bool:
        return self.index >= self.length

    def skip_separators(self) -> None:
        text, length = self.text, self.length
        index = self.index
        while index < length and text[index] in " \t\r\n,":
            index += 1
        self.index = index

    def peek(self) -> str:
        return self.text[self.index] if self.index < self.length else ""

    def take_command(self) -> str:
        char = self.text[self.index]
        self.index += 1
        return char

    def number(self) -> float:
        self.skip_separators()
        match = _NUMBER_RE.match(self.text, self.index)
        if match is None:
            snippet = self.text[self.index : self.index + 12]
            raise PathDataError(f"expected a number at offset {self.index}: {snippet!r}")
        self.index = match.end()
        value = float(match.group(0))
        if not math.isfinite(value):
            raise PathDataError(
                f"non-finite number at offset {match.start()}: {match.group(0)!r}"
            )
        return value

    def flag(self) -> int:
        """Arc flags are single characters and may be written without separators."""

        self.skip_separators()
        char = self.peek()
        if char in "01":
            self.index += 1
            return int(char)
        value = self.number()
        if value not in (0.0, 1.0):
            raise PathDataError(f"arc flag must be 0 or 1, got {value!r}")
        return int(value)


def _cubic_points(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> List[Point]:
    out: List[Point] = []
    for step in range(1, steps + 1):
        t = step / steps
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3.0 * mt * mt * t
        c = 3.0 * mt * t * t
        d = t * t * t
        out.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            )
        )
    return out


def _quadratic_points(p0: Point, p1: Point, p2: Point, steps: int) -> List[Point]:
    out: List[Point] = []
    for step in range(1, steps + 1):
        t = step / steps
        mt = 1.0 - t
        a = mt * mt
        b = 2.0 * mt * t
        c = t * t
        out.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0],
                a * p0[1] + b * p1[1] + c * p2[1],
            )
        )
    return out


def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
    dot = ux * vx + uy * vy
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm == 0.0:
        return 0.0
    ratio = max(-1.0, min(1.0, dot / norm))
    angle = math.acos(ratio)
    if ux * vy - uy * vx < 0.0:
        angle = -angle
    return angle


def arc_points(
    start: Point,
    rx: float,
    ry: float,
    x_axis_rotation: float,
    large_arc: int,
    sweep: int,
    end: Point,
    steps: int = 8,
) -> List[Point]:
    """Sample an SVG elliptical arc (endpoint -> centre parameterisation)."""

    x1, y1 = start
    x2, y2 = end
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0:
        return [end]
    if rx * rx == 0.0 or ry * ry == 0.0:
        # The radius is representable but its square underflows, so the SVG
        # arc maths cannot be evaluated.  Refuse instead of fabricating a
        # chord or letting ZeroDivisionError escape.
        raise PathDataError(
            f"arc radius underflows float64 squaring (rx={rx!r}, ry={ry!r})"
        )

    phi = math.radians(x_axis_rotation % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    scale = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if scale > 1.0:
        factor = math.sqrt(scale)
        rx *= factor
        ry *= factor

    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        raise PathDataError("arc parameters overflow float64")
    if denominator == 0.0:
        return [end]
    sign = -1.0 if bool(large_arc) == bool(sweep) else 1.0
    coefficient = sign * math.sqrt(max(0.0, numerator / denominator))
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry

    if not all(math.isfinite(v) for v in (cxp, cyp, cx, cy, ux, uy, vx, vy)):
        raise PathDataError("arc geometry is not finite")

    theta1 = _angle(1.0, 0.0, ux, uy)
    delta = _angle(ux, uy, vx, vy)
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi

    out: List[Point] = []
    for step in range(1, steps + 1):
        theta = theta1 + delta * (step / steps)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        out.append(
            (
                cx + rx * cos_t * cos_phi - ry * sin_t * sin_phi,
                cy + rx * cos_t * sin_phi + ry * sin_t * cos_phi,
            )
        )
    return out


def _checked_points(points: List[Point], context: str) -> List[Point]:
    """Reject generated coordinates that overflowed to inf/nan."""

    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PathDataError(f"{context} produced non-finite coordinates")
    return points


def _finite_point(x: float, y: float, context: str) -> Point:
    """Reject an absolute coordinate that overflowed, e.g. relative addition."""

    if not (math.isfinite(x) and math.isfinite(y)):
        raise PathDataError(f"{context} produced non-finite coordinates")
    return (x, y)


def validate_curve_steps(curve_steps: int) -> int:
    if isinstance(curve_steps, bool) or not isinstance(curve_steps, int):
        raise ValueError(f"curve_steps must be an integer, got {curve_steps!r}")
    if curve_steps < 1:
        raise ValueError(f"curve_steps must be >= 1, got {curve_steps!r}")
    return curve_steps


def parse_path_data(d: str, curve_steps: int = 8) -> Dict[str, Any]:
    """Parse path data into absolute subpath polylines plus the raw command list."""

    validate_curve_steps(curve_steps)
    scanner = _Scanner(d)
    subpaths: List[List[Point]] = []
    commands: List[Dict[str, Any]] = []
    current: Optional[List[Point]] = None
    cmd: Optional[str] = None
    last_cmd: Optional[str] = None
    prev_cubic: Optional[Point] = None
    prev_quad: Optional[Point] = None
    cx = cy = 0.0
    sx = sy = 0.0

    def ensure_subpath() -> List[Point]:
        nonlocal current
        if current is None:
            current = [(cx, cy)]
            subpaths.append(current)
        return current

    while True:
        scanner.skip_separators()
        if scanner.eof():
            break
        char = scanner.peek()
        if char.isalpha():
            cmd = scanner.take_command()
            upper = cmd.upper()
            if upper not in _PATH_COMMANDS:
                raise PathDataError(f"unsupported path command {cmd!r}")
            if upper == "Z":
                commands.append({"command": cmd, "args": []})
                if current is not None and current:
                    if current[-1] != (sx, sy):
                        current.append((sx, sy))
                cx, cy = sx, sy
                last_cmd = "Z"
                prev_cubic = prev_quad = None
                current = None
                cmd = None
                continue
        elif cmd is None:
            raise PathDataError("path data must start with a command")

        assert cmd is not None
        upper = cmd.upper()
        relative = cmd.islower()

        if upper == "M":
            raw_x, raw_y = scanner.number(), scanner.number()
            x = cx + raw_x if relative else raw_x
            y = cy + raw_y if relative else raw_y
            _finite_point(x, y, f"moveto {cmd!r}")
            commands.append({"command": cmd, "args": [raw_x, raw_y]})
            current = [(x, y)]
            subpaths.append(current)
            cx, cy = x, y
            sx, sy = x, y
            prev_cubic = prev_quad = None
            last_cmd = "M"
            # Subsequent coordinate pairs after moveto are implicit linetos.
            cmd = "l" if relative else "L"
        elif upper == "L":
            raw_x, raw_y = scanner.number(), scanner.number()
            x = cx + raw_x if relative else raw_x
            y = cy + raw_y if relative else raw_y
            _finite_point(x, y, f"lineto {cmd!r}")
            path = ensure_subpath()
            path.append((x, y))
            commands.append({"command": cmd, "args": [raw_x, raw_y]})
            cx, cy = x, y
            prev_cubic = prev_quad = None
            last_cmd = "L"
        elif upper in ("H", "V"):
            value = scanner.number()
            if upper == "H":
                x = cx + value if relative else value
                y = cy
            else:
                x = cx
                y = cy + value if relative else value
            _finite_point(x, y, f"{upper} command")
            path = ensure_subpath()
            path.append((x, y))
            commands.append({"command": cmd, "args": [value]})
            cx, cy = x, y
            prev_cubic = prev_quad = None
            last_cmd = upper
        elif upper in ("C", "S"):
            if upper == "C":
                raw = [scanner.number() for _ in range(6)]
                if relative:
                    p1 = (cx + raw[0], cy + raw[1])
                    p2 = (cx + raw[2], cy + raw[3])
                    end = (cx + raw[4], cy + raw[5])
                else:
                    p1 = (raw[0], raw[1])
                    p2 = (raw[2], raw[3])
                    end = (raw[4], raw[5])
            else:
                raw = [scanner.number() for _ in range(4)]
                if prev_cubic is not None and last_cmd in ("C", "S", "c", "s"):
                    p1 = (2.0 * cx - prev_cubic[0], 2.0 * cy - prev_cubic[1])
                else:
                    p1 = (cx, cy)
                if relative:
                    p2 = (cx + raw[0], cy + raw[1])
                    end = (cx + raw[2], cy + raw[3])
                else:
                    p2 = (raw[0], raw[1])
                    end = (raw[2], raw[3])
            _finite_point(*p1, f"cubic control of {cmd!r}")
            _finite_point(*p2, f"cubic control of {cmd!r}")
            _finite_point(*end, f"cubic end of {cmd!r}")
            path = ensure_subpath()
            path.extend(
                _checked_points(
                    _cubic_points((cx, cy), p1, p2, end, curve_steps),
                    f"cubic segment of {cmd!r}",
                )
            )
            commands.append({"command": cmd, "args": raw})
            prev_cubic = p2
            prev_quad = None
            cx, cy = end
            last_cmd = cmd
        elif upper in ("Q", "T"):
            if upper == "Q":
                raw = [scanner.number() for _ in range(4)]
                if relative:
                    control = (cx + raw[0], cy + raw[1])
                    end = (cx + raw[2], cy + raw[3])
                else:
                    control = (raw[0], raw[1])
                    end = (raw[2], raw[3])
            else:
                raw = [scanner.number() for _ in range(2)]
                if prev_quad is not None and last_cmd in ("Q", "T", "q", "t"):
                    control = (2.0 * cx - prev_quad[0], 2.0 * cy - prev_quad[1])
                else:
                    control = (cx, cy)
                if relative:
                    end = (cx + raw[0], cy + raw[1])
                else:
                    end = (raw[0], raw[1])
            _finite_point(*control, f"quadratic control of {cmd!r}")
            _finite_point(*end, f"quadratic end of {cmd!r}")
            path = ensure_subpath()
            path.extend(
                _checked_points(
                    _quadratic_points((cx, cy), control, end, curve_steps),
                    f"quadratic segment of {cmd!r}",
                )
            )
            commands.append({"command": cmd, "args": raw})
            prev_quad = control
            prev_cubic = None
            cx, cy = end
            last_cmd = cmd
        elif upper == "A":
            rx, ry, rotation = scanner.number(), scanner.number(), scanner.number()
            large_arc, sweep = scanner.flag(), scanner.flag()
            raw_x, raw_y = scanner.number(), scanner.number()
            end = (cx + raw_x, cy + raw_y) if relative else (raw_x, raw_y)
            _finite_point(*end, f"arc end of {cmd!r}")
            path = ensure_subpath()
            arc = _checked_points(
                arc_points(
                    (cx, cy), rx, ry, rotation, large_arc, sweep, end, curve_steps
                ),
                f"arc segment of {cmd!r}",
            )
            path.extend(arc)
            commands.append(
                {
                    "command": cmd,
                    "args": [rx, ry, rotation, large_arc, sweep, raw_x, raw_y],
                }
            )
            prev_cubic = prev_quad = None
            cx, cy = end
            last_cmd = cmd
        else:  # pragma: no cover - guarded above
            raise PathDataError(f"unsupported path command {cmd!r}")

    return {"subpaths": subpaths, "commands": commands}


def polyline_length(points: Sequence[Point]) -> float:
    total = 0.0
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def sample_by_length(points: Sequence[Point], count: int = 4) -> List[List[float]]:
    """Evenly spaced samples by cumulative arc length (default: 4 points)."""

    if not points:
        return []
    if len(points) == 1 or count <= 1:
        return [[float(points[0][0]), float(points[0][1])] for _ in range(max(count, 1))]

    cumulative = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        total += math.hypot(x1 - x0, y1 - y0)
        cumulative.append(total)

    if total == 0.0:
        return [[float(points[0][0]), float(points[0][1])] for _ in range(count)]

    samples: List[List[float]] = []
    cursor = 0
    for step in range(count):
        target = total * step / (count - 1)
        while cursor < len(cumulative) - 2 and cumulative[cursor + 1] < target:
            cursor += 1
        span = cumulative[cursor + 1] - cumulative[cursor]
        ratio = 0.0 if span == 0.0 else (target - cumulative[cursor]) / span
        x0, y0 = points[cursor]
        x1, y1 = points[cursor + 1]
        samples.append([x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio])
    return samples


def subpath_lengths(subpaths: Sequence[Sequence[Point]]) -> List[float]:
    """Drawn length of each subpath; moveto-only subpaths have length 0."""

    return [polyline_length(subpath) for subpath in subpaths]


def _point_at_distance(points: Sequence[Point], distance: float) -> Point:
    """Point *distance* along one polyline; clamps to its endpoints."""

    if distance <= 0.0:
        return (float(points[0][0]), float(points[0][1]))
    travelled = 0.0
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        segment = math.hypot(x1 - x0, y1 - y0)
        if segment <= 0.0:
            continue
        if travelled + segment >= distance:
            ratio = (distance - travelled) / segment
            return (x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio)
        travelled += segment
    return (float(points[-1][0]), float(points[-1][1]))


def sample_subpaths_by_length(
    subpaths: Sequence[Sequence[Point]], count: int = 4
) -> List[List[float]]:
    """Evenly spaced samples over drawn length across several subpaths.

    Only real drawn segments contribute length.  The move between the end of
    one subpath and the start of the next is a pen-up move, not a connection,
    so no sample is ever interpolated across that gap.  A sample landing on a
    subpath boundary stays on the previous subpath; samples never fabricate a
    segment between separate subpaths.
    """

    count = max(int(count), 1)
    if count == 1:
        targets = [0.0]
    else:
        targets = [index / (count - 1) for index in range(count)]

    lengths = subpath_lengths(subpaths)
    # ``sum`` saturates to inf instead of raising OverflowError like fsum;
    # a non-finite total means the geometry is unusable, so return no samples
    # rather than interpolating across inf.
    total = sum(lengths)
    if not math.isfinite(total):
        return []
    drawable = [
        (subpath, length)
        for subpath, length in zip(subpaths, lengths)
        if length > 0.0 and len(subpath) >= 2
    ]

    if not drawable:
        for subpath in subpaths:
            if subpath:
                x, y = subpath[0]
                return [[float(x), float(y)] for _ in targets]
        return []

    samples: List[List[float]] = []
    for fraction in targets:
        target = min(max(total * fraction, 0.0), total)
        point: Optional[Point] = None
        if target <= 0.0:
            point = drawable[0][0][0]
        elif target >= total:
            point = drawable[-1][0][-1]
        else:
            consumed = 0.0
            for subpath, length in drawable:
                if target <= consumed + length:
                    point = _point_at_distance(subpath, target - consumed)
                    break
                consumed += length
        if point is None:  # pragma: no cover - target is clamped to total
            point = drawable[-1][0][-1]
        samples.append([float(point[0]), float(point[1])])
    return samples


def bbox_of(points: Sequence[Point]) -> Optional[List[float]]:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def circle_samples(cx: float, cy: float, r: float, count: int = 4) -> List[List[float]]:
    samples = []
    for step in range(count):
        angle = 2.0 * math.pi * step / max(count, 1)
        samples.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
    return samples


def ellipse_samples(
    cx: float, cy: float, rx: float, ry: float, count: int = 4
) -> List[List[float]]:
    samples = []
    for step in range(count):
        angle = 2.0 * math.pi * step / max(count, 1)
        samples.append([cx + rx * math.cos(angle), cy + ry * math.sin(angle)])
    return samples


def ellipse_perimeter(rx: float, ry: float) -> float:
    """Ramanujan II approximation (relative error < 1e-9 for sane radii)."""

    a, b = abs(rx), abs(ry)
    h = ((a - b) / (a + b)) ** 2 if (a + b) else 0.0
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
