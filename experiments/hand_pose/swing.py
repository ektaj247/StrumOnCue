"""Strumming hand that swings like a forearm, curled fingers brushing the strings.

Experiment version of the right hand (strum.bimanual.add_right_hand) and its
controller (strum.strumming.Strummer). In the ukulele's frame (x along the
strings from the nut, y from the G string to the A string, z out of the
fretboard):

  * The forearm lies almost parallel to the ukulele: it points from the "elbow"
    (the forearm body's far end, past the bridge) toward the neck (-x), tilted
    TILT degrees down toward the strings. The palm faces the strings (-z).
  * All four fingers are curled (J3 1.2, J2 = J1 0.7). The strum point is the
    lowest point of the curled fingers; the strum point sits 0.7 mm above the
    string tops, as in the project's strummer.
  * A hinge at the elbow, with its axis out of the soundboard (z), swings the
    hand across the strings: down (G to A) and up (A to G). The existing
    three-axis stage stays for positioning and lifts the hand off the strings
    for the silent return to the top before a new chord's first downstroke.

Strokes per held chord: down, up, down (three, at the project's 70 BPM
onsets). Upstrokes are expected notes here.

install(tilt) swaps both in, in memory only.
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
import strum.strumming as strumming  # noqa: E402
from strum.fretting import compile_root  # noqa: E402
from strum.ukulele import CHORDS, OPEN_MIDI, STRING_NAMES, string_y, string_z  # noqa: E402

TILT = 8.                  # forearm tilt toward the strings, degrees ("almost parallel")
CURL = {'J3': 1.2, 'J2': .7, 'J1': .7}
FINGERS = ('FF', 'MF', 'RF', 'LF')
SWING_KP, SWING_KV = 400., 20.
LIFT = .02                 # silent return: lift 20 mm off the strings
SWEEP = .30                # seconds from y0 to y1, as in the project's strummer
OVERSHOOT = .03            # the swing continues 30 mm past the outer strings' strike zone


def unit(v):
    return v / np.linalg.norm(v)


def frame(forward, normal):
    f = unit(forward)
    n = unit(normal - (normal @ f) * f)
    return np.column_stack([f, n, np.cross(f, n)])


def mesh_points(m, d, g):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        return d.geom_xpos[g][None, :]
    k = m.geom_dataid[g]
    v = np.array(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]], dtype=float)
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def merge_right_hand(root, repo):
    """Unchanged part of add_right_hand: load, rename, merge, stage, collision bits."""
    root.find('option').set('integrator', 'implicitfast')
    source = Path(repo) / 'external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls/right_hand.xml'
    right = ET.parse(source).getroot()
    for mesh in right.findall('asset/mesh'):
        if not mesh.get('name'):
            mesh.set('name', Path(mesh.get('file')).stem)
    refs = {'name', 'class', 'childclass', 'material', 'mesh', 'joint', 'tendon', 'body1', 'body2'}
    for e in right.iter():
        for k in refs:
            if k in e.attrib:
                e.set(k, 'strum_' + e.get(k))
    for b in right.findall('.//body'):
        b.set('gravcomp', '1')
        for i, g in enumerate(b.findall('geom')):
            if not g.get('name'):
                g.set('name', b.get('name') + f'_geom_{i}')
    for actuator in right.findall('actuator/position'):
        actuator.set('kp', '20'); actuator.set('kv', '1'); actuator.set('forcerange', '-10 10')
    for section in ['default', 'asset', 'contact', 'tendon', 'actuator']:
        src = right.find(section)
        if src is not None:
            dest = root.find(section)
            if dest is None:
                dest = ET.SubElement(root, section)
            for child in list(src):
                dest.append(child)
    mount = ET.SubElement(root.find('worldbody'), 'body', name='right_stage', gravcomp='1')
    # The forearm now hinges on the stage, so the stage needs a little mass of its own.
    ET.SubElement(mount, 'inertial', pos='0 0 0', mass='.1', diaginertia='1e-4 1e-4 1e-4')
    for axis, vector in zip('xyz', ['1 0 0', '0 1 0', '0 0 1']):
        ET.SubElement(mount, 'joint', name=f'right_stage_{axis}', type='slide', axis=vector, range='-1 1', damping='1')
        ET.SubElement(root.find('actuator'), 'position', name=f'right_stage_{axis}_servo', joint=f'right_stage_{axis}',
                      kp='8000', kv='200', ctrlrange='-1 1', forcerange='-200 200')
    forearm = right.find('worldbody/body'); mount.append(forearm)
    for geom in forearm.iter('geom'):
        if geom.get('class') == 'strum_plastic_collision':
            geom.set('contype', '4'); geom.set('conaffinity', '1')
    for geom in root.findall(".//body[@name='ukulele']/geom"):
        if geom.get('name', '').startswith('string_'):
            geom.set('contype', '2'); geom.set('conaffinity', '1')
    return forearm


def add_right_hand_swing(root, repo, tilt_deg=TILT, report=None):
    p = strumming_profile()
    forearm = merge_right_hand(root, repo)
    old = root.find('keyframe')
    qleft = np.fromstring(old.find('key').get('qpos'), sep=' ')
    cleft = np.fromstring(old.find('key').get('ctrl'), sep=' ')
    root.remove(old)
    m = compile_root(root, Path(repo) / 'scene.xml'); d = mujoco.MjData(m)
    d.qpos[:len(qleft)] = qleft
    mujoco.mj_forward(m, d)
    bid = m.body('strum_rh_forearm').id
    U, O = d.xmat[m.body('ukulele').id].reshape(3, 3), d.xpos[m.body('ukulele').id].copy()
    R_arm = d.xmat[bid].reshape(3, 3).copy()
    local = lambda v: R_arm.T @ v
    # Forearm direction (elbow to wrist) and palm direction, in the forearm's own frame.
    along = local(d.xpos[m.body('strum_rh_wrist').id] - d.xpos[bid])
    distal = m.body('strum_rh_ffdistal').id
    j3 = m.jnt_qposadr[m.joint('strum_rh_FFJ3').id]
    tip0 = d.xpos[distal].copy()
    d.qpos[j3] += .3; mujoco.mj_kinematics(m, d)
    palm = local(d.xpos[distal] - tip0)
    d.qpos[j3] -= .3
    # Target: forearm toward the neck, tilted down; palm toward the strings.
    t = np.deg2rad(tilt_deg)
    along_target = np.cos(t) * -U[:, 0] - np.sin(t) * U[:, 2]
    rot = frame(along_target, -U[:, 2]) @ frame(along, palm).T
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, rot.reshape(9))
    m.body_quat[bid] = quat
    for f in FINGERS:
        for joint, value in CURL.items():
            d.qpos[m.jnt_qposadr[m.joint(f'strum_rh_{f}{joint}').id]] = value
    mujoco.mj_forward(m, d)
    # Strum point: the lowest point (along the ukulele's z) of the curled fingers.
    lowest = {}
    for f in FINGERS:
        best = None
        for part in ('proximal', 'middle', 'distal'):
            b = m.body(f'strum_rh_{f.lower()}{part}').id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != b or not m.geom_contype[g]:
                    continue
                pts = mesh_points(m, d, g)
                i = int(np.argmin(pts @ U[:, 2]))
                if best is None or pts[i] @ U[:, 2] < best[0]:
                    best = (pts[i] @ U[:, 2], b, pts[i])
        lowest[f] = best
    finger = min(lowest, key=lambda f: lowest[f][0])
    _, tip_body, tip = lowest[finger]
    site_local = d.xmat[tip_body].reshape(3, 3).T @ (tip - d.xpos[tip_body])
    # Neutral pose: strum point over the middle of the strings, 0.7 mm above the string tops.
    xs = .29 * p.scale_length / .38
    yc = (string_y(p, 0, xs) + string_y(p, 3, xs)) / 2
    target = O + U @ np.array([xs, yc, string_z(p, xs) + .0007])
    m.body_pos[bid] += target - tip; mujoco.mj_forward(m, d)
    forearm.set('pos', ' '.join(map(str, m.body_pos[bid]))); forearm.set('quat', ' '.join(map(str, quat)))
    ET.SubElement(forearm.find(f".//body[@name='{m.body(tip_body).name}']"), 'site', name='right_strum_tip',
                  pos=' '.join(map(str, site_local)), size='.0015', rgba='1 .3 .1 1')
    # Elbow hinge: at the forearm body's origin, axis out of the soundboard.
    axis_local = rot.T @ U[:, 2]
    forearm.insert(0, ET.Element('joint', name='right_swing', type='hinge', pos='0 0 0',
                                 axis=' '.join(map(str, axis_local)), range='-1 1', damping='.5', armature='.01'))
    ET.SubElement(root.find('actuator'), 'position', name='right_swing_servo', joint='right_swing',
                  kp=str(SWING_KP), kv=str(SWING_KV), ctrlrange='-1 1', forcerange='-80 80')
    # Recompile with the hinge to build the keyframe.
    m = compile_root(root, Path(repo) / 'scene.xml'); d = mujoco.MjData(m)
    d.qpos[:len(qleft)] = qleft
    for f in FINGERS:
        for joint, value in CURL.items():
            d.qpos[m.jnt_qposadr[m.joint(f'strum_rh_{f}{joint}').id]] = value
    mujoco.mj_forward(m, d)
    ctrl = d.actuator_length.copy(); ctrl[:len(cleft)] = cleft
    key = ET.SubElement(root, 'keyframe')
    ET.SubElement(key, 'key', name='candidate', qpos=' '.join(map(str, d.qpos)), ctrl=' '.join(map(str, ctrl)))

    # Report: each finger's lowest point and the hand's other parts, relative to the strum point.
    mujoco.mj_forward(m, d)
    z0 = d.site('right_strum_tip').xpos @ U[:, 2]
    rows = []
    for f in FINGERS:
        rows.append(f'{f} {(lowest[f][0] - lowest[finger][0]) * 1000:+.1f}')
    rest = np.inf
    for g in range(m.ngeom):
        name = m.body(int(m.geom_bodyid[g])).name
        if not name.startswith('strum_rh_') or not m.geom_contype[g]:
            continue
        if any(name.startswith(f'strum_rh_{f.lower()}') and not name.endswith('knuckle') for f in FINGERS):
            continue
        rest = min(rest, float((mesh_points(m, d, g) @ U[:, 2]).min() - z0))
    anchor = U.T @ (d.xanchor[m.joint('right_swing').id] - O)
    message = (f'Swing hand (tilt {tilt_deg:.0f} deg): strum point on {m.body(tip_body).name}; '
               f'each finger\'s lowest point vs it (mm): {", ".join(rows)}; palm/thumb/forearm stay '
               f'{rest * 1000:.1f} mm above it; elbow hinge at x {anchor[0] * 1000:.0f} mm, '
               f'{np.linalg.norm((U.T @ (d.site("right_strum_tip").xpos - O) - anchor)[:2]) * 1000:.0f} mm from the strum point.')
    print(message, flush=True)
    if report is not None:
        report['message'] = message
    return target


def strumming_profile():
    from strum.ukulele import profile
    return profile('ek')


class SwingStrummer(strumming.Strummer):
    """Down-up-down strokes per held chord, driven by the elbow hinge."""

    def __init__(self, m, d, p, schedule):
        super().__init__(m, d, p, schedule)
        self.hinge = m.joint('right_swing').id
        self.hinge_q = m.joint('right_swing').qposadr[0]
        self.hinge_act = m.actuator('right_swing_servo').id
        # Rebuild strokes with alternating directions: down, up, down in each hold.
        self.strokes, self.expected = [], []
        start = 0.
        for cycle, (name, seconds, src, dst) in schedule:
            if name.startswith('hold_') or name == 'returned_C':
                for beat in range(3):
                    onset = start + 1. + beat * 60 / 70
                    direction = 'D' if beat % 2 == 0 else 'U'
                    self.strokes.append((onset, direction))
                    for i, s in enumerate(STRING_NAMES):
                        y = string_y(p, i, self.x)
                        share = (y - self.y0) / (self.y1 - self.y0)
                        self.expected.append({'cycle': cycle, 'chord': dst, 'string': s,
                                              'pitch': OPEN_MIDI[i] + CHORDS[dst][i], 'direction': direction,
                                              'onset_s': onset + SWEEP * (share if direction == 'D' else 1 - share)})
            start += seconds
        # Geometry of the swing in the ukulele frame, measured at the current hinge angle.
        mujoco.mj_forward(m, d)
        self.speed = (self.y1 - self.y0) / SWEEP
        self.top, self.bottom = self.y0 - OVERSHOOT, self.y1 + OVERSHOOT
        self.q0 = float(d.qpos[self.hinge_q])
        self.anchor = self.local(d.xanchor[self.hinge])
        self.radius_vec = self.local(d.site('right_strum_tip').xpos) - self.anchor
        self.stage_z = m.actuator('right_stage_z_servo').id
        self.stage_home = d.ctrl[self.act].copy()
        # Start at the top, beyond the G string (set before the first step, so no crossing).
        q_top = self.angle_for(self.top)
        d.qpos[self.hinge_q] = q_top
        d.ctrl[self.hinge_act] = q_top
        mujoco.mj_forward(m, d)
        self.previous = self.local(d.site('right_strum_tip').xpos)

    def angle_for(self, y):
        """Hinge angle that puts the strum point at ukulele-frame y (rotation about z)."""
        vx, vy = self.radius_vec[0], self.radius_vec[1]
        r = np.hypot(vx, vy)
        base = np.arctan2(vy, vx)
        target = np.arcsin(np.clip((y - self.anchor[1]) / r, -1, 1))
        # Two solutions; take the one nearest the starting angle.
        sols = [target - base, np.pi - target - base]
        sol = min(sols, key=lambda s: abs(np.angle(np.exp(1j * s))))
        return self.q0 + float(np.angle(np.exp(1j * sol)))

    def plan(self, t):
        """Desired strum-point y and lift at time t."""
        lead = (self.y0 - self.top) / self.speed
        side = 'top'
        for k, (onset, direction) in enumerate(self.strokes):
            begin, end = onset - lead, onset + SWEEP + lead
            a, b = (self.top, self.bottom) if direction == 'D' else (self.bottom, self.top)
            if t < begin:
                needed = 'top' if direction == 'D' else 'bottom'
                if side != needed:
                    # Silent return: lift, swing over, lower, finishing 0.1 s before the stroke.
                    r0, r1 = begin - .9, begin - .1
                    if t >= r0:
                        u = np.clip((t - r0) / (r1 - r0), 0, 1)
                        y_from = self.bottom if side == 'bottom' else self.top
                        y_to = self.top if needed == 'top' else self.bottom
                        lift = LIFT * np.sin(np.pi * u) ** .5
                        s = u * u * (3 - 2 * u)
                        return y_from + s * (y_to - y_from), lift
                return (self.top if side == 'top' else self.bottom), 0.
            if t <= end:
                return a + (b - a) * (t - begin) / (end - begin), 0.
            side = 'bottom' if direction == 'D' else 'top'
        return (self.top if side == 'top' else self.bottom), 0.

    def control(self, d):
        t = float(d.time) + self.m.opt.timestep
        y, lift = self.plan(t)
        y2, _ = self.plan(t + 1e-3)
        q, q2 = self.angle_for(y), self.angle_for(y2)
        d.ctrl[self.hinge_act] = q + (SWING_KV / SWING_KP) * (q2 - q) / 1e-3
        d.ctrl[self.act] = self.stage_home
        d.ctrl[self.stage_z] = self.stage_home[2] + lift

    def summary(self):
        out = super().summary()
        up = lambda events: [e for e in events if e['direction'] == 'U']
        out['unexpected_upstrokes'] = strumming.score(up(self.expected), up(self.actual))['fp']
        out['strokes_per_hold'] = 'down, up, down'
        out['scope'] = 'Experiment: elbow-hinge swing, curled fingers brushing; ' + out['scope']
        return out


def install(tilt_deg=TILT):
    bimanual.add_right_hand = functools.partial(add_right_hand_swing, tilt_deg=tilt_deg)
    strumming.Strummer = SwingStrummer
