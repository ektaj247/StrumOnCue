"""Idle fingers make room: a version of the planner's hand for the human-like grip.

The planner (strum/planner.py) moves only the pressing fingers and leaves every
other finger in one fixed rest pose. In the human-like grip that rest pose lays
the idle fingers over the fretboard right beside the pressing ones, so a
pressing finger that leans sideways runs into its neighbour.

IdleSpaceHand places the pressing fingers exactly like the planner, then, for
each chord, gives every idle finger the shape (sideways spread J4, knuckle J3,
outer joints J2 = J1) that keeps it clearest of the other fingers, the thumb and
the ukulele along the whole approach: rest -> hover -> press. Among shapes that
leave at least CLEAR_OK of room it picks the one closest to the normal rest pose.

Defined here only; strum/planner.py is not changed.
"""
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strum.planner import FINGERS, Hand  # noqa: E402

# Candidate idle shapes (J4, J3, J2) in radians, clipped to the joint ranges.
SHAPES = list(itertools.product((-.3, 0., .3), (.6, 1.2, 1.5), (.3, .7, 1.3)))
CLEAR_OK = .003      # 3 mm of room counts as enough
REACH = (.25, .5, .75, 1.)    # samples along rest -> hover
PRESS = (.5, 1.)              # samples along hover -> press


def smooth(u):
    return u * u * (3 - 2 * u)


def path(rest, hover, press):
    """Joint states sampled along the approach, with the same easing calibration uses."""
    return ([rest + smooth(u) * (hover - rest) for u in REACH]
            + [hover + smooth(u) * (press - hover) for u in PRESS])


class IdleSpaceHand(Hand):
    def __init__(self, m, p):
        super().__init__(m, p)
        self.chosen = {}
        solid = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
        body_name = lambda g: m.body(int(m.geom_bodyid[g])).name
        self.finger_geoms = {f: [g for g in range(m.ngeom) if solid(g)
                                 and body_name(g).startswith(f'lh_{f.lower()}')
                                 and not body_name(g).startswith(f'lh_{f.lower()}metacarpal')]
                             for f in FINGERS}
        thumb = [g for g in range(m.ngeom) if solid(g) and body_name(g).startswith('lh_th')]
        uke = m.body('ukulele').id
        instrument = [g for g in range(m.ngeom) if solid(g) and int(m.geom_bodyid[g]) == uke]
        self.obstacles = {f: [g for o in FINGERS if o != f for g in self.finger_geoms[o]] + thumb + instrument
                          for f in FINGERS}
        self.joint_ranges = {}
        for f in FINGERS:
            ids = [m.joint(f'lh_{f}{j}').id for j in ('J4', 'J3', 'J2')]
            self.joint_ranges[f] = (m.jnt_range[ids, 0], m.jnt_range[ids, 1])

    def set_shape(self, q, finger, shape):
        low, high = self.joint_ranges[finger]
        j4, j3, j2 = np.clip(shape, low + 1e-6, high - 1e-6)
        q[self.adr(finger, 'J4')], q[self.adr(finger, 'J3')] = j4, j3
        q[self.adr(finger, 'J2')] = q[self.adr(finger, 'J1')] = j2

    def clearance(self, finger, states):
        """Smallest gap (m, capped at 10 mm) between `finger` and its obstacles over the states."""
        m, k = self.m, self.k
        worst = .01
        for q in states:
            k.qpos[:] = q
            mujoco.mj_kinematics(m, k)
            for g in self.finger_geoms[finger]:
                for o in self.obstacles[finger]:
                    worst = min(worst, mujoco.mj_geomDistance(m, k, g, o, .01, None))
        return worst

    def choose(self, assignment, press, hover):
        idle = sorted((f for f in FINGERS if f not in assignment),
                      key=lambda f: min(abs(FINGERS.index(f) - FINGERS.index(a)) for a in assignment))
        press, hover = press.copy(), hover.copy()
        shapes = {}
        for finger in idle:
            rest_shape = np.array([self.rest[self.adr(finger, j)] for j in ('J4', 'J3', 'J2')])
            scored = []
            for shape in SHAPES:
                p, h = press.copy(), hover.copy()
                self.set_shape(p, finger, shape)
                self.set_shape(h, finger, shape)
                room = self.clearance(finger, path(self.rest, h, p))
                scored.append((min(room, CLEAR_OK), -np.abs(np.array(shape) - rest_shape).sum(), room, shape))
            best = max(scored, key=lambda s: (s[0], s[1]))
            shapes[finger] = best[3]
            self.set_shape(press, finger, best[3])
            self.set_shape(hover, finger, best[3])
        return shapes

    def press_pose(self, assignment, depth):
        q, pads, errors = super().press_pose(assignment, depth)
        hover = super().hover_pose(assignment, pads)
        shapes = self.choose(assignment, q, hover)
        self.chosen[frozenset(assignment.items())] = shapes
        for finger, shape in shapes.items():
            self.set_shape(q, finger, shape)
        return q, pads, errors

    def hover_pose(self, assignment, pads, base=None):
        q = super().hover_pose(assignment, pads, base)
        for finger, shape in self.chosen.get(frozenset(assignment.items()), {}).items():
            self.set_shape(q, finger, shape)
        return q
