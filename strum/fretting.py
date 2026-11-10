"""Geometry-aware chord calibration and evaluation. Run with python -m strum.fretting."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .planner import DEPTHS, HOVER, TIMES, Hand, couple_distal
from .ukulele import STRING_NAMES, classify_contact, profile

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def compile_root(root, scene):
    root = ET.fromstring(ET.tostring(root, encoding='unicode'))
    compiler = root.find('compiler')
    # Supply assets through MuJoCo's in-memory filesystem. Saved XML stays
    # portable; MuJoCo no longer needs to open these mesh paths itself.
    assets = {}
    for tag, field in [('mesh', 'meshdir'), ('texture', 'texturedir')]:
        directory = (scene.parent / compiler.get(field, '')).resolve()
        for node in root.findall('asset/' + tag):
            filename = node.get('file')
            if not filename:
                continue
            path = (directory / filename).resolve()
            try:
                contents = path.read_bytes()
            except OSError as error:
                raise OSError(f'Python could not read asset {path}: {error}') from error
            # Preserve names that MuJoCo otherwise derives from filenames.
            if node.get('name') is None:
                node.set('name', Path(filename).stem)
            virtual_name = f'asset_{len(assets):03d}{path.suffix}'
            assets[virtual_name] = contents
            node.set('file', virtual_name)
        compiler.set(field, '')
    return mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding='unicode'), assets=assets)

def load(scene, presentation=None):
    scene = scene.resolve()
    root = ET.parse(scene).getroot()
    if presentation is not None:
        from .presentation import decorate
        decorate(root, presentation, ROOT)
    m = compile_root(root, scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def contact_forces(m, d, p, fingering):
    uke = m.body('ukulele').id
    hand = {i for i in range(m.ngeom)
            if m.body(int(m.geom_bodyid[i])).name.startswith(('lh_', 'strum_rh_'))}
    assignments = {frozenset((m.geom('string_' + s).id,
                   m.geom('lh_' + f.lower() + 'distal_geom_1').id)): (s, n)
                   for f, s, n in fingering}
    forces = {s: 0. for _, s, _ in fingering}
    unwanted = 0.
    pairs = set()
    wrench = np.zeros(6)
    for k, c in enumerate(d.contact):
        pair = frozenset((int(c.geom1), int(c.geom2)))
        if not pair & hand:
            continue
        mujoco.mj_contactForce(m, d, k, wrench)
        force = max(0., float(wrench[0]))
        point = d.xmat[uke].reshape(3, 3).T @ (c.pos - d.xpos[uke])
        assignment = assignments.get(pair)
        if assignment and classify_contact(p, point) == assignment:
            forces[assignment[0]] += force
        else:
            unwanted += force
            if force >= .01:
                pairs.add(tuple(sorted(m.geom(i).name for i in pair)))
    return forces, unwanted, pairs


def hold(scene, p, fingering, seconds, trace=None):
    m, d = load(scene)
    controls = d.ctrl.copy()
    limited = m.actuator_ctrllimited.astype(bool)
    if np.any(controls[limited] < m.actuator_ctrlrange[limited, 0]) or np.any(controls[limited] > m.actuator_ctrlrange[limited, 1]):
        raise ValueError('Control target outside limits')
    q0 = d.qpos.copy()
    rows, pairs = [], set()
    for _ in range(int(np.ceil(seconds / m.opt.timestep))):
        d.ctrl[:] = controls
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            break
        forces, bad, current = contact_forces(m, d, p, fingering)
        pairs |= current
        rows.append({'time_s': float(d.time), 'all_contact': int(all(v >= .01 for v in forces.values())),
                     'unwanted_N': bad, 'max_drift_deg': float(np.rad2deg(np.max(np.abs(d.qpos-q0)))),
                     **{s + '_force_N': v for s, v in forces.items()}})
    measured = [r for r in rows if r['time_s'] >= .5]
    fraction = float(np.mean([r['all_contact'] for r in measured])) if measured else 0.
    bad_fraction = float(np.mean([r['unwanted_N'] >= .01 for r in measured])) if measured else 1.
    warnings = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
    completed = bool(rows and rows[-1]['time_s'] >= seconds - m.opt.timestep/2)
    passed = completed and not warnings and fraction >= .95 and bad_fraction == 0.
    if trace and rows:
        with trace.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return {'pass': passed, 'completed': completed, 'seconds': seconds,
            'hold_fraction': fraction, 'unwanted_fraction': bad_fraction,
            'unwanted_pairs': sorted(pairs), 'warnings': warnings,
            'max_joint_drift_deg': max((r['max_drift_deg'] for r in rows), default=0.),
            'mean_target_forces_N': {s: float(np.mean([r[s+'_force_N'] for r in measured])) if measured else None for _, s, _ in fingering}}


def approach(scene, p, fingering, depth):
    """Reach a chord from the rest pose in one rollout: reach, press, hold.

    Same rules as transitions: no unwanted contact in any phase, and at
    least 95% simultaneous target contact after .5 s of the hold. Returns
    the result plus the settled state and command used for the keyframe.
    """
    m, d = load(scene)
    hand = Hand(m, p)
    assignment = {f: (s, n) for f, s, n in fingering}
    press, pads, errors = hand.press_pose(assignment, depth)
    keys = [hand.command(hand.rest), hand.command(hand.hover_pose(assignment, pads)), hand.command(press)]
    d.qpos[:] = hand.rest
    d.qvel[:] = 0
    d.ctrl[:] = keys[0]
    mujoco.mj_forward(m, d)
    phases, finite = [], True
    for name, seconds, a, b in [('reach', TIMES['move'], 0, 1), ('press', TIMES['press'], 1, 2), ('hold', TIMES['hold'], 2, 2)]:
        steps = int(round(seconds / m.opt.timestep))
        rows, pairs = [], set()
        for step in range(steps):
            u = (step + 1) / steps
            blend = u*u*(3-2*u)
            d.ctrl[:] = (1-blend)*keys[a] + blend*keys[b]
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                finite = False
                break
            forces, bad, current = contact_forces(m, d, p, fingering)
            pairs |= current
            rows.append(((step + 1) * m.opt.timestep, all(v >= .01 for v in forces.values()), bad >= .01, forces))
        measured = [r for r in rows if a != b or r[0] >= .5]
        fraction = float(np.mean([r[1] for r in measured])) if measured else 0.
        bad_fraction = float(np.mean([r[2] for r in measured])) if measured else 1.
        phases.append({'phase': name, 'completed': len(rows) == steps,
                       'pass': len(rows) == steps and bad_fraction == 0. and (a != b or fraction >= .95),
                       'target_contact_fraction': fraction, 'unwanted_fraction': bad_fraction,
                       'unwanted_pairs': sorted(pairs),
                       'mean_target_forces_N': {s: float(np.mean([r[3][s] for r in measured])) if measured else None for _, s, _ in fingering}})
        if not finite:
            break
    warnings = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
    result = {'pass': finite and not warnings and len(phases) == 3 and all(x['pass'] for x in phases),
              'press_depth_m': depth, 'ik_error_mm': errors,
              'pad_x_mm': {f: float(v[0] * 1000) for f, v in pads.items()},
              'warnings': warnings, 'phases': phases}
    return result, d.qpos.copy(), keys[2]


def calibrate(a):
    sys.path.insert(0, str(a.repo / 'scripts'))
    from build_geometry_scene import build
    metadata = json.loads((a.placements / 'summary.json').read_text())
    entry = next(e for e in metadata['candidates'] if e['mount'] == 'x_30')
    p = profile(a.profile)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Fixed-mount chords reached from the rest pose, then initialized holds; no chord transitions, strumming, robustness, or RL.',
              'planner': 'ik', 'distal_coupling': True, 'hover_m': HOVER, 'approach_times_s': TIMES,
              'profile': a.profile, 'parameters': p.to_dict(), 'mujoco_version': mujoco.__version__,
              'force_threshold_N': .01, 'settle_s': .5, 'required_hold_fraction': .95,
              'thresholds_provisional': True, 'chords': {}}
    previous = None
    if a.resume_from:
        previous = json.loads((a.resume_from / 'summary.json').read_text())
        for field in ('profile', 'parameters', 'mujoco_version', 'force_threshold_N', 'settle_s', 'required_hold_fraction', 'planner', 'hover_m', 'approach_times_s'):
            if previous.get(field) != report[field]:
                raise ValueError('Resume settings differ: ' + field)
    mount_reference = None
    for label in entry['selected_chords']:
        chord = 'G' if label.startswith('G') else label
        fingering = entry['chords'][label]['fingering']
        source = a.placements / f'x_30-{label}.xml'
        folder = a.output / chord
        folder.mkdir()
        build(a.repo, folder/'geometry', a.profile, chord)
        root = ET.parse(source).getroot()
        compiler = root.find('compiler')
        assets = (source.parent / compiler.get('meshdir', '')).resolve()
        compiler.set('meshdir', os.path.relpath(assets, folder))
        world = root.find('worldbody')
        mount = world.find("body[@name='lh_forearm']")
        pose = (mount.get('pos'), mount.get('quat'))
        if mount_reference is None: mount_reference = pose
        if pose != mount_reference: raise ValueError('Source chords must share one fixed mount')
        cached = previous['chords'].get(chord) if previous else None
        if cached and cached['accepted'] and cached['verification']['seconds'] >= a.verify_seconds:
            prior_scene = a.resume_from / chord / 'scene.xml'
            if cached['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or cached['fingering'] != fingering:
                raise ValueError('Seed changed for cached chord: ' + chord)
            if cached['scene_sha256'] != hashlib.sha256(prior_scene.read_bytes()).hexdigest():
                raise ValueError('Cached scene changed: ' + chord)
            cached_root = ET.parse(prior_scene).getroot()
            comp = cached_root.find('compiler')
            prior_assets = (prior_scene.parent / comp.get('meshdir', '')).resolve()
            comp.set('meshdir', os.path.relpath(prior_assets, folder))
            copied_scene = folder / 'scene.xml'
            ET.ElementTree(cached_root).write(copied_scene)
            load(copied_scene)
            cached['scene_sha256'] = hashlib.sha256(copied_scene.read_bytes()).hexdigest()
            cached['reused_from'] = str(prior_scene)
            if (prior_scene.parent / 'trace.csv').exists():
                shutil.copy2(prior_scene.parent / 'trace.csv', folder / 'trace.csv')
            save_json(folder / 'summary.json', cached)
            report['chords'][chord] = cached
            save_json(a.output / 'summary.json', report)
            print('REUSED VERIFIED HOLD:', chord, flush=True)
            continue
        world.remove(world.find("body[@name='ukulele']"))
        uke = ET.parse(folder/'geometry/scene.xml').getroot().find("worldbody/body[@name='ukulele']")
        world.append(uke)
        couple_distal(root)
        scene = folder/'scene.xml'
        ET.ElementTree(root).write(scene)
        key = root.find('keyframe/key')
        trials = []
        accepted = None
        best = None
        # One press depth for every finger of the chord; the first passing
        # value in the fixed order is kept. No per-actuator offsets.
        for index, depth in enumerate(DEPTHS):
            reached, q, ctrl = approach(scene, p, fingering, depth)
            key.set('qpos', ' '.join(map(str, q)))
            key.set('qvel', ' '.join('0' for _ in range(len(q))))
            key.set('ctrl', ' '.join(map(str, ctrl)))
            key.set('time', '0')
            ET.ElementTree(root).write(scene)
            result = hold(scene, p, fingering, a.screen_seconds)
            trial = {'trial': index, 'press_depth_m': depth, 'approach': reached, 'screen': result}
            loss = 10*(1-result['hold_fraction']) + result['unwanted_fraction'] + (0 if reached['pass'] else 1)
            if best is None or loss < best[0]: best = (loss, scene.read_bytes(), depth)
            if reached['pass'] and result['pass']:
                trial['verification'] = hold(scene, p, fingering, a.verify_seconds, folder/'trace.csv')
                if trial['verification']['pass']: accepted = index
            trials.append(trial)
            print(chord, json.dumps({'trial': index, 'press_depth_mm': depth*1000, 'approach_pass': reached['pass'],
                                     'screen': result, 'verification': trial.get('verification')}), flush=True)
            if accepted is not None: break
        if accepted is None and best is not None:
            scene.write_bytes(best[1])
        final = trials[-1]['verification'] if accepted is not None else hold(scene, p, fingering, a.verify_seconds, folder/'trace.csv')
        record = {'accepted': bool(accepted is not None and final['pass']), 'fingering': fingering,
                  'press_depth_m': DEPTHS[accepted] if accepted is not None else best[2],
                  'trials': trials, 'verification': final,
                  'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest()}
        save_json(folder/'summary.json', record)
        report['chords'][chord] = record
        save_json(a.output/'summary.json', report)
    report['all_pass'] = all(r['accepted'] for r in report['chords'].values())
    save_json(a.output/'summary.json', report)
    print('ALL FOUR VERIFIED:', report['all_pass'])
    print('Saved', a.output/'summary.json')


def verify(a):
    report = json.loads((a.run / 'summary.json').read_text())
    p = profile(report['profile'])
    if p.to_dict() != report['parameters']:
        raise ValueError('Geometry definition changed since calibration')
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for chord, record in report['chords'].items():
        scene = a.run / chord / 'scene.xml'
        if hashlib.sha256(scene.read_bytes()).hexdigest() != record['scene_sha256']:
            raise ValueError('Scene changed since calibration: ' + chord)
        results[chord] = hold(scene, p, record['fingering'], a.seconds, a.output / (chord + '.csv'))
        print(chord, json.dumps(results[chord]), flush=True)
    save_json(a.output / 'summary.json', {'scope': report['scope'], 'mujoco_version': mujoco.__version__,
              'all_pass': all(r['pass'] for r in results.values()), 'chords': results})


def view(a):
    from mujoco import viewer
    params = None
    if a.polished:
        meta = json.loads((a.scene.parent.parent / 'summary.json').read_text())
        params = profile(meta['profile'])
        if params.to_dict() != meta['parameters']:
            raise ValueError('Geometry parameters changed')
    m, d = load(a.scene, presentation=params)
    controls = d.ctrl.copy()
    with viewer.launch_passive(m, d) as window:
        from .presentation import set_camera
        set_camera(window.cam, a.camera)
        while window.is_running():
            start = time.monotonic()
            for _ in range(10):
                d.ctrl[:] = controls
                mujoco.mj_step(m, d)
            window.sync()
            time.sleep(max(0., 10*m.opt.timestep-(time.monotonic()-start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('calibrate')
    c.add_argument('--repo', type=Path, default=ROOT)
    c.add_argument('--placements', type=Path, required=True)
    c.add_argument('--resume-from', type=Path)
    c.add_argument('--profile', choices=['baseline','ek'], default='ek')
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--screen-seconds', type=float, default=10.)
    c.add_argument('--verify-seconds', type=float, default=60.)
    t = commands.add_parser('verify')
    t.add_argument('--run', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    t.add_argument('--seconds', type=float, default=60.)
    x = commands.add_parser('transition')
    x.add_argument('--run', type=Path, required=True)
    x.add_argument('--output', type=Path, required=True)
    x.add_argument('--cycles', type=int, default=10)
    x.add_argument('--route', nargs='+', choices=['C','Am','F','G'])
    x.add_argument('--view', action='store_true')
    x.add_argument('--polished', action='store_true')
    x.add_argument('--two-hands', action='store_true')
    x.add_argument('--camera', choices=['overview', 'fretting'], default='overview')
    x.add_argument('--approach-proximal', type=float, default=0.)
    x.add_argument('--approach-tendon', type=float, default=2.)
    x.add_argument('--time-scale', type=float, default=1., help='Scale lift/move/press durations (ik planner only)')
    b = commands.add_parser('coverage')
    b.add_argument('--run', type=Path, required=True)
    b.add_argument('--manifest', type=Path, default=ROOT/'manifests/benchmark_v1.json')
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--time-scale', type=float, default=1.)
    v = commands.add_parser('view')
    v.add_argument('--scene', type=Path, required=True)
    v.add_argument('--polished', action='store_true')
    v.add_argument('--camera', choices=['overview', 'fretting'], default='overview')
    a = parser.parse_args()
    if a.command == 'calibrate':
        if not (1 <= a.screen_seconds <= a.verify_seconds): parser.error('Require 1 <= screen <= verify seconds')
        a.repo = a.repo.resolve(); a.placements = a.placements.resolve(); a.output = a.output.resolve()
        calibrate(a)
    elif a.command == 'verify':
        if a.seconds < 1: parser.error('Require seconds >= 1')
        verify(a)
    elif a.command == 'transition':
        if a.cycles < 1: parser.error('cycles must be positive')
        if a.route and (len(a.route)<2 or any(x==y for x,y in zip(a.route,a.route[1:]))): parser.error('Route needs at least two chords and no adjacent repeats')
        if a.route and a.cycles>1 and a.route[0]!=a.route[-1]: parser.error('Repeated routes must end at the starting chord; otherwise use --cycles 1')
        from .transitions import run
        run(a)
    elif a.command == 'coverage':
        from .tasks import coverage
        coverage(a)
    else:
        try: view(a)
        except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
