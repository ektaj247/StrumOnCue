"""Geometry-aware initialized hold calibration. Run with python -m strum.fretting."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .ukulele import STRING_NAMES, classify_contact, profile

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(scene, presentation=None):
    scene = scene.resolve()
    root = ET.parse(scene).getroot()
    if presentation is not None:
        from .presentation import decorate
        decorate(root, presentation, ROOT)
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
    m = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding='unicode'), assets=assets)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def contact_forces(m, d, p, fingering):
    uke = m.body('ukulele').id
    hand = {i for i in range(m.ngeom)
            if m.body(int(m.geom_bodyid[i])).name.startswith('lh_')}
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


def fit_pose(scene, fingering):
    m, d = load(scene)
    q0 = d.qpos.copy()
    # Existing surface proxy is approximate; dynamics is the acceptance test.
    pad = m.site_pos[m.site('ring_surface_reference').id].copy()
    indices, lower, upper = [], [], []
    for finger, _, _ in fingering:
        for suffix in ['J4', 'J3', 'J2']:
            j = m.joint('lh_' + finger + suffix).id
            indices.append(int(m.jnt_qposadr[j]))
            lo, hi = m.jnt_range[j]
            if suffix == 'J2':
                j1 = m.joint('lh_' + finger + 'J1').id
                lo, hi = max(lo, m.jnt_range[j1, 0]), min(hi, m.jnt_range[j1, 1])
            lower.append(lo); upper.append(hi)
    bodies = [m.body('lh_' + f.lower() + 'distal').id for f, _, _ in fingering]
    targets = [d.site(f'target_{s}_{n}').xpos.copy() for _, s, n in fingering]
    def residual(x):
        d.qpos[:] = q0
        d.qpos[indices] = x
        for f, _, _ in fingering:
            d.qpos[m.joint('lh_'+f+'J1').qposadr[0]] = d.qpos[m.joint('lh_'+f+'J2').qposadr[0]]
        mujoco.mj_forward(m, d)
        return np.concatenate([(d.xpos[b] + d.xmat[b].reshape(3,3) @ pad - t)*1000 for b,t in zip(bodies,targets)])
    fit = least_squares(residual, np.clip(q0[indices], lower, upper), bounds=(lower, upper), max_nfev=200)
    errors = np.linalg.norm(residual(fit.x).reshape(-1,3), axis=1)
    return d.qpos.copy(), d.actuator_length.copy(), errors.tolist()


def calibrate(a):
    sys.path.insert(0, str(a.repo / 'scripts'))
    from build_geometry_scene import build
    metadata = json.loads((a.placements / 'summary.json').read_text())
    entry = next(e for e in metadata['candidates'] if e['mount'] == 'x_30')
    p = profile(a.profile)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Initialized fixed-mount holds only; no approach, transitions, strumming, robustness, or RL.',
              'profile': a.profile, 'parameters': p.to_dict(), 'mujoco_version': mujoco.__version__,
              'force_threshold_N': .01, 'settle_s': .5, 'required_hold_fraction': .95,
              'thresholds_provisional': True, 'chords': {}}
    previous = None
    if a.resume_from:
        previous = json.loads((a.resume_from / 'summary.json').read_text())
        for field in ('profile', 'parameters', 'mujoco_version', 'force_threshold_N', 'settle_s', 'required_hold_fraction'):
            if previous[field] != report[field]:
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
        scene = folder/'scene.xml'
        ET.ElementTree(root).write(scene)
        q, controls, errors = fit_pose(scene, fingering)
        key = root.find('keyframe/key')
        key.set('qpos', ' '.join(map(str, q)))
        key.set('qvel', ' '.join('0' for _ in q))
        key.set('time', '0')
        model, _ = load(scene)
        trials = []
        # Small deterministic control search; no hand physics changes.
        grid = list(itertools.product([0., -.04, .04], [0., .04, .08, -.04, .12]))
        grid.sort(key=lambda v: abs(v[0])+abs(v[1]))
        accepted = None
        best = None
        for index, (dp, dt) in enumerate(grid):
            ctrl = controls.copy()
            for finger, _, _ in fingering:
                ctrl[model.actuator('lh_A_'+finger+'J3').id] += dp
                ctrl[model.actuator('lh_A_'+finger+'J0').id] += dt
            limited = model.actuator_ctrllimited.astype(bool)
            if np.any(ctrl[limited] < model.actuator_ctrlrange[limited,0]) or np.any(ctrl[limited] > model.actuator_ctrlrange[limited,1]):
                continue
            key.set('ctrl', ' '.join(map(str, ctrl)))
            ET.ElementTree(root).write(scene)
            result = hold(scene, p, fingering, a.screen_seconds)
            trial = {'trial': index, 'proximal_delta': dp, 'tendon_delta': dt, 'screen': result}
            loss = 10*(1-result['hold_fraction']) + result['unwanted_fraction']
            if best is None or loss < best[0]: best = (loss, scene.read_bytes())
            if result['pass']:
                trial['verification'] = hold(scene, p, fingering, a.verify_seconds, folder/'trace.csv')
                if trial['verification']['pass']: accepted = index
            trials.append(trial)
            print(chord, json.dumps(trial), flush=True)
            if accepted is not None: break
        if accepted is None and best is not None:
            scene.write_bytes(best[1])
        final = trials[-1]['verification'] if accepted is not None else hold(scene, p, fingering, a.verify_seconds, folder/'trace.csv')
        record = {'accepted': bool(accepted is not None and final['pass']), 'fingering': fingering,
                  'proxy_errors_mm': errors, 'trials': trials, 'verification': final,
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
    x.add_argument('--view', action='store_true')
    x.add_argument('--polished', action='store_true')
    x.add_argument('--camera', choices=['overview', 'fretting'], default='overview')
    x.add_argument('--approach-proximal', type=float, default=0.)
    x.add_argument('--approach-tendon', type=float, default=2.)
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
        from .transitions import run
        run(a)
    else:
        try: view(a)
        except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
