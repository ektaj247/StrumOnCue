"""Can the fretting hand reach all four chords from a human-like forearm pose?

Human-like here means: forearm level, palm facing up (toward the G string,
which is on top when the ukulele is held upright), knuckles under the neck so
the fingers curl up and over onto the fretboard.

Kinematics only. No physics, no calibration, no project files changed. For
each candidate forearm pose, IK puts each assigned fingertip on its string and
fret at calibration's first press depth. We report how far each fingertip
misses (mm) and how deep any other part of the hand cuts into the ukulele or
into another finger (mm), and which part that is.

Two IKs are compared:
  planner  the project's IK unchanged (strum/planner.py), which starts from
           three fairly straight finger shapes. Calibration uses this one.
  hook     the same IK, also tried from curled, hook-like finger shapes.
           Defined only in this file; the planner is not modified.
If a pose passes with "hook" but not with "planner", the hand can reach it but
the planner would need those extra starting shapes.

All poses are given in the ukulele's frame:
  x along the strings from the nut, y across them from G to A, z out of the
  fretboard. "Below the neck" is +y (the A side), "behind" is -z.

Yaw is the direction the level forearm points, from elbow to fingers:
  0 deg = toward +z (from behind the neck, like a player's arm),
  90 deg = toward +x (along the neck, toward the body),
  -90 deg = toward -x (toward the headstock).

Run from the repository root:
    .venv/Scripts/python experiments/hand_pose/check_hand_pose.py
    .venv/Scripts/python experiments/hand_pose/check_hand_pose.py --generations 20
    .venv/Scripts/python experiments/hand_pose/check_hand_pose.py --show outputs/hand-pose/<run>

Writes only to outputs/hand-pose/<run>/.
"""
import argparse
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from build_geometry_scene import build  # noqa: E402
from strum.fretting import compile_root  # noqa: E402
from strum.planner import DEPTHS, FINGERS, Hand, couple_distal  # noqa: E402
from strum.ukulele import classify_contact, fret_positions, profile  # noqa: E402

SEEDS = ROOT / 'assets/fretting_seed_v1'
PROFILE = 'ek'
DEPTH = DEPTHS[0]          # the first press depth calibration tries
MISS_OK_MM = 1.0           # every assigned fingertip within 1 mm of its target
PEN_OK_MM = 0.5            # nothing else cuts deeper than 0.5 mm
PEN_WEIGHT = 10.           # 1 mm of penetration costs as much as 10 mm of miss
WRONG_COST = 5.            # each fingertip whose real contact is the wrong note costs 5 mm

# Search box: yaw, pitch and roll in degrees, knuckle position in mm.
NAMES = ('yaw_deg', 'pitch_deg', 'roll_deg', 'knuckle_x_mm', 'knuckle_y_mm', 'knuckle_z_mm')
LOWER = np.array([-180., -15., -25., -20., 27., -120.])
UPPER = np.array([180., 15., 25., 150., 150., 60.])
STEP = np.array([20., 8., 10., 20., 15., 20.])


class HookHand(Hand):
    """The planner's IK, also started from curled finger shapes (J4, J3, J2)."""
    EXTRA = ([0., 1.2, 1.4], [0., 1.5, 1.0], [0., .8, 1.5], [0., 1.5, 1.5], [0., 1.0, 1.0])

    def ik(self, q, finger, string, fret, dz, x=None):
        best = super().ik(q, finger, string, fret, dz, x)
        if best[2] < 1e-3:
            return best
        address = [self.adr(finger, s) for s in ('J4', 'J3', 'J2', 'J1')]
        for shape in self.EXTRA:
            # The planner's IK tries the finger's current joints first, so
            # setting them to a hook shape makes that its first start.
            start = np.array(q, dtype=float)
            start[address[:3]] = shape
            start[address[3]] = shape[2]
            candidate = super().ik(start, finger, string, fret, dz, x)
            if candidate[2] < best[2]:
                best = candidate
            if best[2] < 1e-3:
                break
        return best


def unit(v):
    return v / np.linalg.norm(v)


def frame(forward, normal):
    """Right-handed frame whose first axis is `forward`, second is `normal` made perpendicular."""
    f = unit(forward)
    n = unit(normal - (normal @ f) * f)
    return np.column_stack([f, n, np.cross(f, n)])


def fingerings():
    meta = json.loads((SEEDS / 'summary.json').read_text())
    entry = next(e for e in meta['candidates'] if e['mount'] == 'x_30')
    out = {}
    for label in entry['selected_chords']:
        chord = 'G' if label.startswith('G') else label
        out[chord] = {'label': label, 'fingering': entry['chords'][label]['fingering']}
    return out


def make_base_scene(run):
    """The scene calibration builds for a chord: seed hand, current ukulele, coupled distal joints."""
    base = run / 'base'
    build(ROOT, base / 'geometry', PROFILE, 'C')
    source = SEEDS / 'x_30-C.xml'
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    assets = (source.parent / compiler.get('meshdir', '')).resolve()
    compiler.set('meshdir', os.path.relpath(assets, base))
    world = root.find('worldbody')
    world.remove(world.find("body[@name='ukulele']"))
    world.append(ET.parse(base / 'geometry/scene.xml').getroot().find("worldbody/body[@name='ukulele']"))
    couple_distal(root)
    scene = base / 'scene.xml'
    ET.ElementTree(root).write(scene)
    return scene


class Checker:
    def __init__(self, scene):
        self.root = ET.parse(scene).getroot()
        self.m = compile_root(self.root, scene)
        self.d = mujoco.MjData(self.m)
        p = self.p = profile(PROFILE)
        self.edges = [0.] + fret_positions(p.scale_length, p.n_frets)
        from idle_space import IdleSpaceHand
        self.hands = {'planner': Hand(self.m, p), 'hook': HookHand(self.m, p), 'space': IdleSpaceHand(self.m, p)}
        self.hand = self.hands['planner']
        self.chords = fingerings()
        m = self.m
        self.uke = m.body('ukulele').id
        self.forearm = m.body('lh_forearm').id
        self.original = (m.body_pos[self.forearm].copy(), m.body_quat[self.forearm].copy())
        self.u_pos = self.hand.origin.copy()
        self.u_rot = self.hand.rotation.copy()
        self.distal = {f: m.body(f'lh_{f.lower()}distal').id for f in FINGERS}
        self.finger_of = {}
        for f in FINGERS:
            for part in ('knuckle', 'proximal', 'middle', 'distal'):
                self.finger_of[m.body(f'lh_{f.lower()}{part}').id] = f
        # Same, plus the thumb as 'TH' (used by the path check).
        self.part_of = dict(self.finger_of)
        for b in range(m.nbody):
            if m.body(b).name.startswith('lh_th'):
                self.part_of[b] = 'TH'
        self._measure_hand()

    def _measure_hand(self):
        """Forearm-frame knuckle point, finger direction and palm normal."""
        m, k = self.m, mujoco.MjData(self.m)
        m.body_pos[self.forearm] = 0
        m.body_quat[self.forearm] = [1, 0, 0, 0]
        k.qpos[:] = self.hand.rest
        mujoco.mj_kinematics(m, k)
        wrist = k.xpos[m.body('lh_wrist').id].copy()
        self.knuckle_local = k.xpos[m.body('lh_mfknuckle').id].copy()
        self.forward_local = unit(self.knuckle_local - wrist)
        # Palm direction: start from straight fingers, curl them a little and
        # see where the tips go. (From the rest pose the fingers are already
        # bent, and curling further moves the tips back toward the palm.)
        straight = self.hand.rest.copy()
        for f in FINGERS:
            for joint in ('J3', 'J2', 'J1'):
                straight[self.hand.adr(f, joint)] = 0.
        k.qpos[:] = straight
        mujoco.mj_kinematics(m, k)
        tips_straight = self._tips(k)
        for f in FINGERS:
            k.qpos[self.hand.adr(f, 'J3')] += .3
        mujoco.mj_kinematics(m, k)
        curl = np.mean(self._tips(k) - tips_straight, axis=0)
        self.palm_local = unit(curl - (curl @ self.forward_local) * self.forward_local)
        m.body_pos[self.forearm], m.body_quat[self.forearm] = self.original

    def _tips(self, data):
        return np.array([data.xpos[self.distal[f]] + data.xmat[self.distal[f]].reshape(3, 3) @ self.hand.pad
                         for f in FINGERS])

    def pose(self, x):
        """World position and quaternion of the forearm for one candidate (see NAMES)."""
        yaw, pitch, roll = np.deg2rad(x[:3])
        xu, yu, zu = self.u_rot.T
        level = np.cos(yaw) * zu + np.sin(yaw) * xu
        forward = np.cos(pitch) * level + np.sin(pitch) * (-yu)
        up = unit(-yu - (-yu @ forward) * forward)
        palm = np.cos(roll) * up + np.sin(roll) * np.cross(forward, up)
        rotation = frame(forward, palm) @ frame(self.forward_local, self.palm_local).T
        knuckle = self.u_pos + self.u_rot @ (np.asarray(x[3:6]) / 1000)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.reshape(9))
        return knuckle - rotation @ self.knuckle_local, quat

    def evaluate(self, pos, quat, ik='hook'):
        m, d, hand = self.m, self.d, self.hands[ik]
        m.body_pos[self.forearm], m.body_quat[self.forearm] = pos, quat
        result = {}
        for chord, info in self.chords.items():
            assign = {f: (s, n) for f, s, n in info['fingering']}
            q, pads, errors = hand.press_pose(assign, DEPTH)
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            pen, part = 0., ''
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist >= 0:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if 0 in (b1, b2):
                    continue  # floor
                n1, n2 = m.body(b1).name, m.body(b2).name
                hand1, hand2 = n1.startswith('lh_'), n2.startswith('lh_')
                if hand1 and hand2:
                    if self.finger_of.get(b1) == self.finger_of.get(b2):
                        continue
                else:
                    touching = b1 if hand1 else b2
                    if any(touching == self.distal[f] for f in assign):
                        continue  # a pressing fingertip on the instrument is the point
                if -c.dist > pen:
                    other = m.geom(int(c.geom2 if hand1 else c.geom1)).name or (n2 if hand1 else n1)
                    pen, part = -c.dist, f'{(n1 if hand1 else n2).removeprefix("lh_")}>{other}'
            wrong = self.notes(assign)
            result[chord] = {'q': q, 'miss_mm': max(errors.values()), 'pen_mm': pen * 1000, 'part': part,
                             'wrong': wrong, 'path': self.path_problems(hand, assign, q, pads)}
        return result

    def path_problems(self, hand, assign, press, pads):
        """Check the approach (rest -> hover -> press) the way calibration judges it.

        Samples the joint path with calibration's easing and reports any
        finger-on-finger contact ('clash:MF/RF') and any touch of the
        instrument other than a pressing fingertip on its own note or the
        fretboard inside its own fret ('touch:FF>A1', 'touch:RF>string_E').
        """
        from idle_space import path
        m, d = self.m, self.d
        hover = hand.hover_pose(assign, pads)
        problems = []
        for q in path(hand.rest, hover, press):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist >= 0:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if 0 in (b1, b2):
                    continue
                f1, f2 = self.part_of.get(b1), self.part_of.get(b2)
                if f1 and f2:
                    if f1 != f2:
                        problems.append('clash:' + '/'.join(sorted((f1, f2))))
                    continue
                if self.uke not in (b1, b2) or not (f1 or f2):
                    continue
                body, geom = (b1, int(c.geom2)) if f1 else (b2, int(c.geom1))
                finger, name = f1 or f2, m.geom(geom).name
                point = self.u_rot.T @ (c.pos - self.u_pos)
                if finger in assign and body == self.distal[finger]:
                    string, fret = assign[finger]
                    if name == f'string_{string}' and classify_contact(self.p, point) == (string, fret):
                        continue
                    if name == 'neck' and self.edges[fret - 1] < point[0] < self.edges[fret]:
                        continue
                    label = classify_contact(self.p, point) if name.startswith('string_') else None
                    problems.append(f'touch:{finger}>{label[0]}{label[1]}' if label else f'touch:{finger}>{name}')
                else:
                    problems.append(f'touch:{finger}>{name or "body"}')
        return list(dict.fromkeys(problems))

    def notes(self, assign):
        """Judge each pressing fingertip's real contacts the way calibration does.

        Uses the contacts already computed by mj_forward. A finger is right when
        it touches its own string at a point classify_contact() labels with its
        own string and fret, and touches nothing else except the fretboard
        inside its own fret. Returns labels like 'FF:A2>A3' or 'FF:A2>none'.
        """
        m, d = self.m, self.d
        wrong = []
        for finger, (string, fret) in assign.items():
            body = self.distal[finger]
            target = f'string_{string}'
            right, bad = False, []
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist >= 0:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                if body not in (int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])):
                    continue
                other = g2 if int(m.geom_bodyid[g1]) == body else g1
                if int(m.geom_bodyid[other]) != self.uke:
                    continue  # finger-to-finger contact is scored as penetration
                name = m.geom(other).name
                point = self.u_rot.T @ (c.pos - self.u_pos)
                if name == target:
                    label = classify_contact(self.p, point)
                    if label == (string, fret):
                        right = True
                    else:
                        bad.append(f'{label[0]}{label[1]}' if label else 'off')
                elif name.startswith('string_'):
                    bad.append(name.removeprefix('string_'))
                elif name == 'neck' and self.edges[fret - 1] < point[0] < self.edges[fret]:
                    continue  # fretboard inside its own fret is allowed
                else:
                    bad.append(name or 'body')
            if bad or not right:
                wrong.append(f'{finger}:{string}{fret}>{"/".join(dict.fromkeys(bad)) or "none"}')
        return wrong

    @staticmethod
    def cost(result):
        return sum(r['miss_mm'] + PEN_WEIGHT * r['pen_mm'] + WRONG_COST * (len(r['wrong']) + len(r['path']))
                   for r in result.values())

    @staticmethod
    def passes(result):
        return all(r['miss_mm'] < MISS_OK_MM and r['pen_mm'] < PEN_OK_MM and not r['wrong'] and not r['path']
                   for r in result.values())

    def objective(self, x, fixed_yaw=None):
        if fixed_yaw is not None:
            x = np.concatenate([[fixed_yaw], x])
        return self.cost(self.evaluate(*self.pose(x)))

    def snapshot(self, run, label, x):
        """Save the pose `x` and draw it pressing G and C into <run>/progress/."""
        from render_poses import Painter  # imported here: render_poses imports this file
        folder = run / 'progress'
        folder.mkdir(exist_ok=True)
        if not hasattr(self, 'painter'):
            self.painter = Painter(self)
        pos, quat = self.pose(x)
        result = self.evaluate(pos, quat, ik='hook')
        record = {**{n: float(v) for n, v in zip(NAMES, x)}, 'pass': Checker.passes(result),
                  'chords': {c: {k: v for k, v in r.items() if k != 'q'} for c, r in result.items()}}
        (folder / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
        files = []
        for chord in ('G', 'C'):
            files += self.painter.draw(f'{label}_{chord}', pos, quat, result[chord]['q'], folder=folder)
        page = folder / 'progress.md'
        if not page.exists():
            page.write_text('# Search progress\n\nBest pose so far, newest at the bottom. Hook IK. '
                            'Pictures: G then C, each front and end view (G string on top).\n\n',
                            encoding='utf-8')
        misses = ', '.join(f'{c} {r["miss_mm"]:.1f} mm' for c, r in result.items())
        worst = max(result.values(), key=lambda r: r['pen_mm'])
        wrong = ', '.join(f'{c} {w}' for c, r in result.items() for w in r['wrong']) or 'none'
        text = (f'## {label}\n\n' + ', '.join(f'{n} {v:.0f}' for n, v in zip(NAMES, x)) + '\n\n'
                f'Misses: {misses}. Wrong notes: {wrong}. '
                f'Deepest penetration {worst["pen_mm"]:.1f} mm ({worst["part"] or "none"}). '
                f'{"PASS" if record["pass"] else "fail"}.\n\n' + ' '.join(f'![{f}]({f})' for f in files) + '\n\n')
        with page.open('a', encoding='utf-8') as out:
            out.write(text)

    def global_search(self, generations, popsize, run=None, every=5):
        started = time.time()

        def report(intermediate_result):
            print(f'  generation {report.n:3d}: best cost {intermediate_result.fun:8.2f}'
                  f'   ({time.time() - started:.0f} s)', flush=True)
            if run is not None and report.n % every == 0:
                self.snapshot(run, f'gen_{report.n:03d}', intermediate_result.x)
            report.n += 1
        report.n = 1
        fit = differential_evolution(self.objective, list(zip(LOWER, UPPER)), maxiter=generations,
                                     popsize=popsize, seed=0, polish=False, tol=0, callback=report)
        return fit.x

    def local_search(self, start, evals, fixed_yaw=None):
        lower, upper, step = LOWER, UPPER, STEP
        if fixed_yaw is not None:
            lower, upper, step, start = LOWER[1:], UPPER[1:], STEP[1:], np.asarray(start)[1:]
        simplex = [start] + [start + step * np.eye(len(start))[i] for i in range(len(start))]
        simplex = np.clip(simplex, lower, upper)
        fit = minimize(self.objective, start, args=(fixed_yaw,), method='Nelder-Mead',
                       bounds=list(zip(lower, upper)),
                       options={'maxfev': evals, 'initial_simplex': simplex, 'xatol': .2, 'fatol': .01})
        return fit.x if fixed_yaw is None else np.concatenate([[fixed_yaw], fit.x])


CHORD_ORDER = ('C', 'G', 'Am', 'F')
HEADER = (f'{"pose":>16}  ' + '  '.join(f'{c + " miss/pen":>11}' for c in CHORD_ORDER)
          + '  result  problems (deepest contact; wrong notes as finger:target>actual)')


def row(label, result):
    cells = [f'{label:>16}']
    for chord in CHORD_ORDER:
        r = result[chord]
        cells.append(f'{r["miss_mm"]:6.2f}/{r["pen_mm"]:4.1f}')
    cells.append('PASS  ' if Checker.passes(result) else 'fail  ')
    worst = max(result.values(), key=lambda r: r['pen_mm'])
    problems = [worst['part']] if worst['pen_mm'] > 0 else []
    problems += [f'{c} {w}' for c in CHORD_ORDER for w in result[c]['wrong']]
    problems += [f'{c} path {w}' for c in CHORD_ORDER for w in result[c]['path']]
    cells.append(', '.join(problems) or '-')
    return '  '.join(cells)


def save(run, checker, x, results):
    pos, quat = checker.pose(x)
    best = {**{n: float(v) for n, v in zip(NAMES, x)},
            'forearm_pos': pos.tolist(), 'forearm_quat': quat.tolist(), 'press_depth_m': DEPTH,
            'pass': {ik: Checker.passes(r) for ik, r in results.items()},
            'chords': {ik: {c: {k: (v.tolist() if k == 'q' else v) for k, v in r.items()} for c, r in res.items()}
                       for ik, res in results.items()}}
    (run / 'best.json').write_text(json.dumps(best, indent=2) + '\n')
    # A seed folder in the same form as assets/fretting_seed_v1, with only the mount changed.
    seed = run / 'seed'
    seed.mkdir(exist_ok=True)
    pos_text, quat_text = ' '.join(map(str, pos)), ' '.join(map(str, quat))
    for info in checker.chords.values():
        source = SEEDS / f'x_30-{info["label"]}.xml'
        root = ET.parse(source).getroot()
        compiler = root.find('compiler')
        assets = (source.parent / compiler.get('meshdir', '')).resolve()
        compiler.set('meshdir', os.path.relpath(assets, seed))
        mount = root.find("worldbody/body[@name='lh_forearm']")
        mount.set('pos', pos_text)
        mount.set('quat', quat_text)
        ET.ElementTree(root).write(seed / source.name)
    shutil.copy2(SEEDS / 'summary.json', seed / 'summary.json')


def show(run, ik='hook'):
    """Static view of the saved best pose, cycling through the chords. No physics."""
    from mujoco import viewer
    checker = Checker(run / 'base/scene.xml')
    best = json.loads((run / 'best.json').read_text())
    m, d = checker.m, checker.d
    m.body_pos[checker.forearm] = best['forearm_pos']
    m.body_quat[checker.forearm] = best['forearm_quat']
    with viewer.launch_passive(m, d) as window:
        window.cam.lookat[:] = checker.u_pos + checker.u_rot @ [.05, 0, 0]
        window.cam.distance, window.cam.azimuth, window.cam.elevation = .45, 120, -25
        i = 0
        while window.is_running():
            chord = CHORD_ORDER[i % 4]
            d.qpos[:] = best['chords'][ik][chord]['q']
            mujoco.mj_forward(m, d)
            print('showing', chord, f'({ik} IK)', flush=True)
            end = time.time() + 2.
            while window.is_running() and time.time() < end:
                window.sync()
                time.sleep(.02)
            i += 1


def main():
    global SEEDS  # --seed replaces the default seed folder for the whole run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generations', type=int, default=60, help='global search generations')
    parser.add_argument('--popsize', type=int, default=15, help='global search population per parameter')
    parser.add_argument('--progress-every', type=int, default=5,
                        help='save and draw the best pose so far every N generations')
    parser.add_argument('--local-evals', type=int, default=600, help='evaluations for the final local search')
    parser.add_argument('--sweep-evals', type=int, default=200, help='evaluations per yaw in the comparison')
    parser.add_argument('--view', action='store_true', help='show the best pose when done')
    parser.add_argument('--show', type=Path, help='only show the saved result of an earlier run')
    parser.add_argument('--ik', choices=['hook', 'planner'], default='hook', help='which IK result --show uses')
    parser.add_argument('--seed', type=Path, default=SEEDS,
                        help='seed folder: fingerings and fingertip marker (default: assets/fretting_seed_v1)')
    parser.add_argument('--score', type=Path, help='only score the pose in this best.json, then stop')
    a = parser.parse_args()
    if a.show:
        show(a.show.resolve(), a.ik)
        return
    SEEDS = a.seed.resolve()

    run = ROOT / 'outputs/hand-pose' / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    if a.score:
        checker = Checker(make_base_scene(run))
        best = json.loads(a.score.read_text())
        x = np.array([best[n] for n in NAMES])
        print('Seed:', SEEDS)
        print('Fingertip marker (mm):', np.round(checker.hand.pad * 1000, 2))
        print('Pose:', ', '.join(f'{n} {v:.1f}' for n, v in zip(NAMES, x)), '\n')
        print(HEADER)
        for ik in ('planner', 'hook', 'space'):
            result = checker.evaluate(*checker.pose(x), ik=ik)
            print(row(f'scored/{ik}', result))
        shapes = checker.hands['space'].chosen
        print('\nIdle-finger shapes chosen by "space" (J4, J3, J2 in rad):')
        for chord, info in checker.chords.items():
            key = frozenset({f: (s, n) for f, s, n in info['fingering']}.items())
            print(f'  {chord}: ' + ', '.join(f'{f} {tuple(round(v, 2) for v in s)}' for f, s in shapes[key].items()))
        return
    checker = Checker(make_base_scene(run))
    print('Run folder:', run)
    print('Palm normal in forearm frame:', np.round(checker.palm_local, 3),
          ' finger direction:', np.round(checker.forward_local, 3))

    started = time.time()
    checker.objective(np.array([0., 0., 0., 40., 60., -10.]))
    each = time.time() - started
    total = (a.generations + 1) * a.popsize * len(NAMES) * each
    print(f'One evaluation (4 chords, hook IK) takes about {each:.2f} s;'
          f' the global search needs roughly {total / 60:.0f} min.\n')

    print(HEADER)
    print(row('current/planner', checker.evaluate(*checker.original, ik='planner')))
    print(row('current/hook', checker.evaluate(*checker.original, ik='hook')), flush=True)

    print('\nGlobal search over all six numbers (hook IK):')
    print('Progress pictures:', run / 'progress/progress.md')
    x = checker.global_search(a.generations, a.popsize, run, a.progress_every)
    x = checker.local_search(x, a.local_evals)
    checker.snapshot(run, 'final', x)
    results = {ik: checker.evaluate(*checker.pose(x), ik=ik) for ik in ('planner', 'hook')}
    print()
    print(HEADER)
    print(row('best/planner', results['planner']))
    print(row('best/hook', results['hook']))
    print('  ' + ', '.join(f'{n} {v:.1f}' for n, v in zip(NAMES, x)))
    save(run, checker, x, results)

    print('\nSame pose, other forearm directions (hook IK, local search with yaw fixed):')
    print(HEADER)
    for yaw in range(-180, 180, 30):
        y = checker.local_search(x, a.sweep_evals, fixed_yaw=float(yaw))
        print(row(f'yaw {yaw:+4d}', checker.evaluate(*checker.pose(y))), flush=True)

    print(f'\nFinished in {(time.time() - started) / 60:.1f} min. Saved', run / 'best.json')
    print('Columns: worst fingertip miss in mm / deepest unwanted penetration in mm.')
    print(f'PASS means every miss < {MISS_OK_MM} mm and every penetration < {PEN_OK_MM} mm.')
    if a.view:
        show(run)


if __name__ == '__main__':
    main()
