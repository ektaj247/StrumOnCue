"""Instrument-frame geometry.

Metres; +x nut-to-bridge, +z above fretboard.
Adapted from the project's ek branch.
Defaults are hypotheses, not measurements.
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass
from math import hypot

STRING_NAMES = ("G", "C", "E", "A")
OPEN_MIDI = (67, 60, 64, 69)

CHORDS = {
    "C": (0, 0, 0, 3),
    "Am": (2, 0, 0, 0),
    "F": (2, 0, 1, 0),
    "G": (0, 2, 3, 2),
}


@dataclass(frozen=True)
class UkuleleParams:
    scale_length: float = 0.381
    nut_spread: float = 0.029
    saddle_spread: float = 0.040
    nut_height: float = 0.0016
    saddle_height: float = 0.0054
    string_radius: float = 0.00045
    neck_width: float = 0.044
    neck_depth: float = 0.020
    neck_length: float = 0.200

    # The current task uses five frets.
    n_frets: int = 5
    press_rule: str = "behind_fret"
    press_offset: float = 0.003
    fret_top: float = 0.001
    fret_radius: float = 0.0005

    def __post_init__(self):
        dimensions = [
            self.scale_length,
            self.nut_spread,
            self.saddle_spread,
            self.nut_height,
            self.saddle_height,
            self.string_radius,
            self.neck_width,
            self.neck_depth,
            self.neck_length,
            self.fret_top,
            self.fret_radius,
        ]
        if not all(0 < x < float("inf") for x in dimensions):
            raise ValueError("Dimensions must be finite and positive")

        if not isinstance(self.n_frets, int) or self.n_frets < 1:
            raise ValueError("n_frets must be a positive integer")

        if self.press_rule not in ("midpoint", "behind_fret"):
            raise ValueError("Unknown press rule")

        edges = [0.0] + fret_positions(self.scale_length, self.n_frets)

        if (
            edges[-1] > self.neck_length
            or self.neck_length >= self.scale_length
        ):
            raise ValueError("Displayed frets must fit on the neck")

        smallest_interval = min(
            b - a for a, b in zip(edges, edges[1:])
        )
        if (
            self.press_rule == "behind_fret"
            and not 0 < self.press_offset < smallest_interval
        ):
            raise ValueError("press_offset must fit inside every fret interval")

        neck_end_spread = (
            self.nut_spread
            + (self.saddle_spread - self.nut_spread)
            * self.neck_length
            / self.scale_length
        )
        if (
            max(self.nut_spread, neck_end_spread)
            + 2 * self.string_radius
            >= self.neck_width
        ):
            raise ValueError("Strings extend beyond the neck")

    def to_dict(self):
        return asdict(self)


def profile(name):
    if name == "ek":
        return UkuleleParams()

    if name == "baseline":
        return UkuleleParams(
            scale_length=0.380,
            nut_spread=0.027,
            saddle_spread=0.027,
            nut_height=0.002,
            saddle_height=0.002,
            string_radius=0.0006,
            neck_width=0.035,
            neck_depth=0.018,
            press_rule="midpoint",
            fret_top=0.0014,
            fret_radius=0.0006,
        )

    raise ValueError(f"Unknown profile: {name}")


def fret_positions(scale_length, n_frets):
    return [
        scale_length * (1 - 2 ** (-n / 12))
        for n in range(1, n_frets + 1)
    ]


def string_y(p, string, x):
    if string not in range(4):
        raise ValueError("string index must be 0..3")

    spread = (
        p.nut_spread
        + (p.saddle_spread - p.nut_spread) * x / p.scale_length
    )
    return (string / 3 - 0.5) * spread


def string_z(p, x):
    return (
        p.nut_height
        + (p.saddle_height - p.nut_height) * x / p.scale_length
    )


def press_point(p, fret, string):
    """Target on the string centreline; not a validated finger pose."""
    if fret not in range(1, p.n_frets + 1):
        raise ValueError("Fret must be within the displayed range")

    edges = [0.0] + fret_positions(p.scale_length, p.n_frets)

    if p.press_rule == "midpoint":
        x = (edges[fret - 1] + edges[fret]) / 2
    else:
        x = edges[fret] - p.press_offset

    return x, string_y(p, string, x), string_z(p, x)


def fingertip_target(p, fret, string):
    x, y, z = press_point(p, fret, string)
    return x, y, z + p.string_radius


def midi_pitch(p, string, fret):
    if string not in range(4) or fret not in range(p.n_frets + 1):
        raise ValueError("Invalid string/fret")

    return OPEN_MIDI[string] + fret


def classify_contact(p, point, tolerance=0.002):
    """Geometric string/fret label only.

    Callers must also check geometry IDs and contact force.
    Point must be in the instrument frame.
    Exact fret-wire boundaries return None.
    """
    x, y, z = point
    edges = [0.0] + fret_positions(p.scale_length, p.n_frets)

    if not 0 < x < edges[-1]:
        return None

    if any(abs(x - edge) < 1e-9 for edge in edges):
        return None

    string = min(
        range(4),
        key=lambda i: abs(y - string_y(p, i, x)),
    )

    distance = hypot(
        y - string_y(p, string, x),
        z - string_z(p, x),
    )
    if distance > p.string_radius + tolerance:
        return None

    return STRING_NAMES[string], bisect_right(edges, x)


def swept_crossings(
    p,
    start,
    end,
    height_tolerance=0.002,
    x_min=0.20,
):
    """Symbolic crossings of string lines by a swept fingertip point.

    Supports spreading strings and varying string height.
    No debounce, force test, acoustics, or scoring is implied.
    """
    x0, y0, z0 = start
    dx, dy, dz = (b - a for a, b in zip(start, end))
    events = []

    for i, name in enumerate(STRING_NAMES):
        slope = (
            (i / 3 - 0.5)
            * (p.saddle_spread - p.nut_spread)
            / p.scale_length
        )
        relative_dy = dy - slope * dx

        if abs(relative_dy) < 1e-12:
            continue

        t = (string_y(p, i, x0) - y0) / relative_dy

        if not 0 < t <= 1:
            continue

        x = x0 + t * dx
        z = z0 + t * dz

        if (
            x_min <= x <= p.scale_length
            and abs(z - string_z(p, x)) <= height_tolerance
        ):
            events.append(
                {
                    "string": name,
                    "fraction": t,
                    "direction": "D" if relative_dy > 0 else "U",
                }
            )

    return sorted(events, key=lambda event: event["fraction"])