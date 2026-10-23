"""Geometry tests for the ukulele generator. Pure Python, no MuJoCo needed.

    python -m unittest discover -s tests -v
"""

import unittest

from strum.ukulele import UkuleleParams, fret_positions, midi_pitch, press_point, string_y, string_z


class FretGeometry(unittest.TestCase):
    def setUp(self):
        self.p = UkuleleParams()

    def test_first_fret_matches_concert_scale(self):
        self.assertAlmostEqual(fret_positions(0.381, 1)[0] * 1000, 21.4, delta=0.1)

    def test_twelfth_fret_is_half_the_scale(self):
        self.assertAlmostEqual(fret_positions(self.p.L, 12)[-1], self.p.L / 2, places=12)

    def test_formula_within_a_tenth_of_a_millimetre(self):
        for n, x in enumerate(fret_positions(self.p.L, 18), start=1):
            self.assertAlmostEqual(x, self.p.L * (1 - 2 ** (-n / 12)), delta=1e-4)

    def test_frets_get_closer_by_one_semitone_ratio(self):
        f = [0.0] + fret_positions(self.p.L, 18)
        gaps = [b - a for a, b in zip(f, f[1:])]
        for g0, g1 in zip(gaps, gaps[1:]):
            self.assertAlmostEqual(g1 / g0, 2 ** (-1 / 12), places=9)

    def test_scale_factor_scales_frets(self):
        big = UkuleleParams(scale_factor=1.15)
        for a, b in zip(fret_positions(self.p.L, 5), fret_positions(big.L, 5)):
            self.assertAlmostEqual(b, 1.15 * a, places=12)


class StringsAndPressPoints(unittest.TestCase):
    def setUp(self):
        self.p = UkuleleParams()

    def test_string_spacing_at_nut(self):
        gap = string_y(self.p, 1, 0.0) - string_y(self.p, 0, 0.0)
        self.assertAlmostEqual(gap * 1000, 29 / 3, places=6)

    def test_strings_are_symmetric_about_the_neck(self):
        for x in (0.0, 0.1, self.p.L):
            self.assertAlmostEqual(string_y(self.p, 0, x), -string_y(self.p, 3, x), places=12)

    def test_strings_clear_the_fret_crowns(self):
        for x in fret_positions(self.p.L, 18):
            self.assertGreater(string_z(self.p, x) - self.p.string_radius, self.p.fret_crown)

    def test_press_point_sits_between_its_frets(self):
        frets = [0.0] + fret_positions(self.p.L, 5)
        for fret in range(1, 6):
            for s in range(4):
                x, y, z = press_point(self.p, fret, s)
                self.assertTrue(frets[fret - 1] < x < frets[fret])
                self.assertAlmostEqual(y, string_y(self.p, s, x), places=12)
                self.assertAlmostEqual(z, string_z(self.p, x), places=12)

    def test_open_string_has_no_press_point(self):
        with self.assertRaises(ValueError):
            press_point(self.p, 0, 0)

    def test_chord_pitches(self):
        # C major, frets G C E A = 0 0 0 3 -> G4 C4 E4 C5.
        frets = (0, 0, 0, 3)
        self.assertEqual([midi_pitch(self.p, s, f) for s, f in enumerate(frets)], [67, 60, 64, 72])


if __name__ == "__main__":
    unittest.main()
