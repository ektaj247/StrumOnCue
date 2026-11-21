"""Turn the whole two-hand scene upright, as a player holds the ukulele.

After both hands are added, every body in the scene (ukulele, left hand, right
hand and its stage) is moved into one parent body, `player_rig`, which is
turned so that, seen from a camera in front:

  * the fretboard faces the camera,
  * the G string is on top and the A string at the bottom,
  * the neck points to the viewer's right, tilted up NECK_TILT degrees,
  * the ukulele sits HEIGHT metres above the floor.

Gravity is turned by the same rotation, so in the ukulele's own frame the
physics is unchanged; only the floor, lights and camera see a difference.

install() wraps strum.bimanual.add_right_hand (whichever version is installed,
for example swing.py's), in memory only.
"""
import functools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strum.bimanual as bimanual  # noqa: E402

NECK_TILT = 15.
HEIGHT = 1.0

# Ukulele frame -> world: x (nut to saddle) to -X, y (G to A) to -Z, z (out of the fretboard) to -Y.
UPRIGHT = np.array([[-1., 0., 0.],
                    [0., 0., -1.],
                    [0., -1., 0.]])


def rotation(neck_tilt_deg=NECK_TILT):
    t = np.deg2rad(neck_tilt_deg)
    raise_neck = np.array([[np.cos(t), 0., -np.sin(t)],     # turns +X (toward the nut) upward
                           [0., 1., 0.],
                           [np.sin(t), 0., np.cos(t)]])
    return raise_neck @ UPRIGHT


def stand_upright(root, neck_tilt_deg=NECK_TILT, height=HEIGHT):
    rot = rotation(neck_tilt_deg)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(9))
    world = root.find('worldbody')
    rig = ET.Element('body', name='player_rig', pos=f'0 0 {height}', quat=' '.join(f'{v:.9g}' for v in quat))
    for child in list(world):
        if child.tag == 'body':
            world.remove(child)
            rig.append(child)
    world.append(rig)
    gravity = rot @ np.array([0., 0., -9.81])
    root.find('option').set('gravity', ' '.join(f'{v:.9g}' for v in gravity))


def wrapped(original, root, repo, **kwargs):
    result = original(root, repo, **kwargs)
    stand_upright(root)
    return result


def install():
    bimanual.add_right_hand = functools.partial(wrapped, bimanual.add_right_hand)


def front_camera(camera, m, d, distance=.75, offset=(.17, .02, 0.)):
    """Point a free camera at the fretboard from the front (G string on top)."""
    uke = m.body('ukulele').id
    U = d.xmat[uke].reshape(3, 3)
    camera.lookat[:] = d.xpos[uke] + U @ np.array(offset)
    forward = -U[:, 2]
    camera.distance = distance
    camera.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    camera.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))
