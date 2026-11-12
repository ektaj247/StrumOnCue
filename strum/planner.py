"""Fingertip-space planning for the fretting hand.

Chord poses and the waypoints between them come from per-finger inverse
kinematics on the shared instrument geometry. Position commands include the
static torque each actuator must supply at the planned pose, so free-space
waypoints are reached without per-chord control offsets.

Requires the distal coupling added by couple_distal(): without it the J0
tendon only fixes J1 + J2 and the finger shape depends on its history.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .ukulele import STRING_NAMES, fingertip_target, fret_positions

FINGERS = ('FF', 'MF', 'RF', 'LF')
REST = {'J4': 0., 'J3': 1.2, 'J2': .7, 'J1': .7}
HOVER = .010           # pad clearance above the press point while travelling
FRET_MARGIN = .002     # keep pad targets this far inside the fret interval
# Commanded pad depth below the string-top touch point, tried in this order.
DEPTHS = (-.00075, -.0005, -.0010, -.00025, -.00125, -.0015, -.002)
TIMES = {'lift': .5, 'move': 1., 'press': .5, 'hold': 5.}


def couple_distal(root):
    """Add J1 = J2 equality constraints for the four tendon-driven fingers.

    The upstream model drives J1 + J2 with one tendon and leaves J1 - J2
    free. That mode creeps under soft joint friction at about half a degree
    per second, so a fixed command has no fixed pose. The calibration IK has
    always assumed J1 = J2; this makes the simulated hand agree with it.
    """
    equality = root.find('equality')
    if equality is None:
        equality = root.makeelement('equality', {})
        root.insert(list(root).index(root.find('actuator')), equality)
    existing = {e.get('joint1') for e in equality.findall('joint')}
    for finger in FINGERS:
        if f'lh_{finger}J1' not in existing:
            equality.append(equality.makeelement('joint', {
                'name': f'lh_{finger}_distal_coupling',
                'joint1': f'lh_{finger}J1', 'joint2': f'lh_{finger}J2',
                'polycoef': '0 1 0 0 0'}))


def is_coupled(m):
    pairs = set()
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            pairs.add(frozenset((m.joint(int(m.eq_obj1id[i])).name,
                                 m.joint(int(m.eq_obj2id[i])).name)))
    return all(frozenset((f'lh_{f}J1', f'lh_{f}J2')) in pairs for f in FINGERS)


class Hand:
    """Kinematics and static commands for the fixed-mount left hand."""

    def __init__(self, m, p):
        if not is_coupled(m):
            raise ValueError('Planner needs the distal coupling; recalibrate with this version')
        self.m, self.p = m, p
        self.k = mujoco.MjData(m)
        self.rest = np.zeros(m.nq)
        for finger in FINGERS:
            for suffix, value in REST.items():
                self.rest[self.adr(finger, suffix)] = value
        self.k.qpos[:] = self.rest
        mujoco.mj_forward(m, self.k)
        uke = m.body('ukulele').id
        self.origin = self.k.xpos[uke].copy()
        self.rotation = self.k.xmat[uke].reshape(3, 3).copy()
        self.pad = m.site_pos[m.site('ring_surface_reference').id].copy()
        self.edges = [0.] + fret_positions(p.scale_length, p.n_frets)

    def adr(self, finger, suffix):
        return int(self.m.joint(f'lh_{finger}{suffix}').qposadr[0])

    def pad_local(self, data, finger):
        """Pad marker in the instrument frame (metres)."""
        body = self.m.body(f'lh_{finger.lower()}distal').id
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ self.pad
        return self.rotation.T @ (point - self.origin)

    def copy_finger(self, destination, source, finger):
        for suffix in REST:
            destination[self.adr(finger, suffix)] = source[self.adr(finger, suffix)]

    def ik(self, q, finger, string, fret, dz, x=None):
        """Place one finger's pad at a string/fret point raised by dz.

        y and z are matched exactly. With x=None the pad may sit anywhere
        inside the fret interval (any point there frets the same note), with
        a weak pull towards the nominal press point; this is what lets a
        finger at its splay limit still take a note. Returns (q, pad, error_mm).
        """
        m, k = self.m, self.k
        nominal = np.array(fingertip_target(self.p, fret, STRING_NAMES.index(string)))
        nominal[2] += dz
        low, high = self.edges[fret - 1] + FRET_MARGIN, self.edges[fret] - FRET_MARGIN
        joints = [m.joint(f'lh_{finger}{s}').id for s in ('J4', 'J3', 'J2')]
        lower, upper = m.jnt_range[joints, 0].copy(), m.jnt_range[joints, 1].copy()
        distal = m.joint(f'lh_{finger}J1').id
        lower[2], upper[2] = max(lower[2], m.jnt_range[distal, 0]), min(upper[2], m.jnt_range[distal, 1])
        address = [self.adr(finger, s) for s in ('J4', 'J3', 'J2', 'J1')]

        def error(v):
            k.qpos[:] = q
            k.qpos[address[:3]] = v
            k.qpos[address[3]] = v[2]
            mujoco.mj_kinematics(m, k)
            e = (self.pad_local(k, finger) - nominal) * 1000
            if x is not None:
                e[0] = (self.pad_local(k, finger)[0] - x) * 1000
                return e, e
            px = self.pad_local(k, finger)[0]
            outside = (max(0., low - px) + max(0., px - high)) * 1000
            return np.array([outside * np.sign(e[0]) + .05 * e[0], e[1], e[2]]), np.array([outside, e[1], e[2]])

        best = None
        for seed in (np.asarray(q)[address[:3]], [0., .4, .4], [0., 1., .2]):
            start = np.clip(seed, lower + 1e-6, upper - 1e-6)
            fit = least_squares(lambda v: error(v)[0], start, bounds=(lower, upper), max_nfev=200)
            miss = float(np.linalg.norm(error(fit.x)[1]))
            if best is None or miss < best[1] - 1e-9:
                best = (fit.x, miss)
            if miss < 1e-3:
                break
        out = np.array(q, dtype=float)
        out[address[:3]] = best[0]
        out[address[3]] = best[0][2]
        error(best[0])
        return out, self.pad_local(k, finger).copy(), best[1]

    def press_pose(self, assignment, depth):
        """assignment: {finger: (string, fret)}. Unassigned fingers rest."""
        q, pads, errors = self.rest.copy(), {}, {}
        for finger, (string, fret) in assignment.items():
            # Fingers sharing a fret would overlap at the nominal press
            # point, so they spread across the fret interval in finger order.
            sharing = sorted((f for f, (_, n) in assignment.items() if n == fret), key=FINGERS.index)
            x = None
            if len(sharing) > 1:
                low, high = self.edges[fret - 1] + FRET_MARGIN, self.edges[fret] - FRET_MARGIN
                x = low + (high - low) * sharing.index(finger) / (len(sharing) - 1)
            q, pads[finger], errors[finger] = self.ik(q, finger, string, fret, depth, x=x)
        return q, pads, errors

    def hover_pose(self, assignment, pads, base=None):
        """Raise each assigned pad straight up from where its press pose put it."""
        q = self.rest.copy() if base is None else base.copy()
        for finger, (string, fret) in assignment.items():
            q, _, _ = self.ik(q, finger, string, fret, HOVER, x=float(pads[finger][0]))
        return q

    def command(self, q):
        """Position targets that hold q statically (gravity feedforward)."""
        m, k = self.m, self.k
        k.qpos[:] = q
        k.qvel[:] = 0
        mujoco.mj_forward(m, k)
        torque = k.qfrc_bias
        ctrl = np.zeros(m.nu)
        for i in range(m.nu):
            kp = m.actuator_gainprm[i, 0]
            target = int(m.actuator_trnid[i, 0])
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                ctrl[i] = q[m.jnt_qposadr[target]] + torque[m.jnt_dofadr[target]] / kp
            elif m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
                # Coupled joints move together, so the tendon force that
                # balances them is the summed joint torque over the summed
                # moment arms.
                row = np.zeros(m.nv)
                start = m.tendon_adr[target]
                for w in range(start, start + m.tendon_num[target]):
                    row[m.jnt_dofadr[m.wrap_objid[w]]] = m.wrap_prm[w]
                dofs = np.nonzero(row)[0]
                ctrl[i] = k.ten_length[target] + torque[dofs].sum() / row[dofs].sum() / kp
            else:
                raise ValueError('Unsupported actuator transmission')
        limited = m.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], m.actuator_ctrlrange[limited, 0], m.actuator_ctrlrange[limited, 1])
        return ctrl


def plan(hand, controls, fingers, depths, route, cycles, time_scale=1.):
    """Direct chord-to-chord routes: lift, travel at hover height, press.

    Fingers that fret the same note in both chords stay down. Fills
    controls/fingers with the waypoint keys and returns (sequence, schedule)
    in the same form as tasks.plan.
    """
    press, pads, assign = {}, {}, {}
    for chord in dict.fromkeys(route):
        assign[chord] = {f: (s, n) for f, s, n in fingers[chord]}
        press[chord], pads[chord], _ = hand.press_pose(assign[chord], depths[chord])
        command = hand.command(press[chord])
        if not np.allclose(command, controls[chord][:len(command)], rtol=0, atol=1e-9):
            raise ValueError('Calibrated command disagrees with planner: ' + chord)
    sequence = []
    for a, b in zip(route, route[1:]):
        stay = [f for f in assign[a] if assign[b].get(f) == assign[a][f]]
        leave = [f for f in assign[a] if f not in stay]
        arrive = {f: v for f, v in assign[b].items() if f not in stay}
        lifted = hand.hover_pose({f: assign[a][f] for f in leave}, pads[a], base=press[a])
        over = hand.hover_pose(arrive, pads[b], base=hand.rest)
        for finger in stay:
            hand.copy_finger(over, press[a], finger)
        held = [list(x) for x in fingers[a] if x[0] in stay]
        one, two = f'{a}>{b}:lifted', f'{a}>{b}:over'
        controls[one], controls[two] = hand.command(lifted), hand.command(over)
        fingers[one], fingers[two] = held, held
        t = {k: (v if k == 'hold' else v * time_scale) for k, v in TIMES.items()}
        if leave:
            sequence.append((f'lift_{a}', t['lift'], a, one))
        else:
            controls[one] = controls[a].copy()
        sequence.extend([(f'move_{b}', t['move'], one if leave else a, two),
                         (f'press_{b}', t['press'], two, b),
                         (f'hold_{b}', t['hold'], b, b)])
    schedule = [(0, (f'initial_{route[0]}', 3., route[0], route[0]))]
    schedule += [(cycle, phase) for cycle in range(1, cycles + 1) for phase in sequence]
    return sequence, schedule
