"""Concert-ukulele geometry: fret positions, string lines, press points, pitches.

The instrument frame is the ukulele body frame:
  * +x runs along the strings, from the nut (x = 0) toward the saddle (x = L).
  * +y runs across the strings, from the G string to the A string.
  * +z points out of the fretboard (strings sit at small positive z).
"""

from __future__ import annotations

from dataclasses import dataclass, field

STRING_NAMES = ("G", "C", "E", "A")
OPEN_MIDI = (67, 60, 64, 69)  # re-entrant tuning: G4 C4 E4 A4


@dataclass(frozen=True)
class UkuleleParams:
    """Dimensions in metres for a concert ukulele, before scaling."""

    scale_length: float = 0.381     # nut to saddle (concert scale)
    nut_spread: float = 0.029       # centre-to-centre, outer strings, at the nut
    saddle_spread: float = 0.040    # centre-to-centre, outer strings, at the saddle
    nut_height: float = 0.0016      # string height above the fretboard at the nut
    saddle_height: float = 0.0054   # string height above the fretboard at the saddle
    fret_crown: float = 0.0010      # fret-wire height above the fretboard
    string_radius: float = 0.00045
    n_frets: int = 18
    press_offset: float = 0.003     # press point sits this far behind the fret wire
    neck_width: float = 0.044
    fretboard_thickness: float = 0.005
    neck_depth: float = 0.020
    body_length: float = 0.300
    lower_bout: float = 0.240
    upper_bout: float = 0.190
    body_depth: float = 0.070
    sound_hole_radius: float = 0.026
    scale_factor: float = 1.0       # uniform scale, a fallback if the fingers do not fit
    open_midi: tuple[int, ...] = field(default=OPEN_MIDI)

    def s(self, value: float) -> float:
        """Apply the uniform scale factor to a length."""
        return value * self.scale_factor

    @property
    def L(self) -> float:
        return self.s(self.scale_length)


def fret_positions(scale_length: float, n_frets: int) -> list[float]:
    """Distance from the nut to fret wires 1..n (equal temperament)."""
    return [scale_length * (1.0 - 2.0 ** (-n / 12.0)) for n in range(1, n_frets + 1)]


def string_y(p: UkuleleParams, string: int, x: float) -> float:
    """Lateral position of a string at distance x from the nut.

    Strings fan out linearly from the nut spread to the saddle spread.
    """
    t = x / p.L
    spread = p.s(p.nut_spread) + (p.s(p.saddle_spread) - p.s(p.nut_spread)) * t
    return -spread / 2.0 + spread * string / 3.0


def string_z(p: UkuleleParams, x: float) -> float:
    """Height of the string centre line above the fretboard surface."""
    t = x / p.L
    return p.s(p.nut_height) + (p.s(p.saddle_height) - p.s(p.nut_height)) * t


def press_point(p: UkuleleParams, fret: int, string: int) -> tuple[float, float, float]:
    """Where a fingertip should press to sound `fret` on `string` (fret >= 1)."""
    if fret < 1:
        raise ValueError("fret must be >= 1; fret 0 is the open string")
    x = fret_positions(p.L, fret)[-1] - p.s(p.press_offset)
    return (x, string_y(p, string, x), string_z(p, x))


def midi_pitch(p: UkuleleParams, string: int, fret: int) -> int:
    """MIDI pitch of a string stopped at `fret` (0 = open)."""
    return p.open_midi[string] + fret
