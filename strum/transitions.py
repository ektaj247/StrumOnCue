"""Continuous actuator-driven transitions, evaluated against shared geometry."""
from contextlib import nullcontext
import csv
import hashlib
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .fretting import ROOT, load, contact_forces, save_json
from .ukulele import profile


def run(a):
    source = json.loads((a.run / 'summary.json').read_text())
    p = profile(source['profile'])
    if p.to_dict() != source['parameters']:
        raise ValueError('Geometry changed since hold calibration')
    route = getattr(a, 'route', None) or ['C','Am','C']
    models, data, fingers, hashes = {}, {}, {}, {}
    for chord in dict.fromkeys(route):
        record = source['chords'][chord]
        if not record['accepted']:
            raise ValueError('Need accepted initialized holds: ' + chord)
        scene = a.run / chord / 'scene.xml'
        digest = hashlib.sha256(scene.read_bytes()).hexdigest()
        if digest != record['scene_sha256']:
            raise ValueError('Scene changed: ' + chord)
        models[chord], data[chord] = load(scene, presentation=p if a.polished else None)
        fingers[chord] = record['fingering']
        hashes[chord] = digest
    m, d = models[route[0]], data[route[0]]
    for other in list(models.values())[1:]:
        if m.nq != other.nq or m.nu != other.nu:
            raise ValueError('Hand dimensions differ')
        for kind, count in [('joint', m.njnt), ('actuator', m.nu), ('geom', m.ngeom)]:
            if [getattr(m,kind)(i).name for i in range(count)] != [getattr(other,kind)(i).name for i in range(count)]:
                raise ValueError('Different model ordering: ' + kind)
        for field in ['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size',
                      'geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp',
                      'jnt_range','dof_damping','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']:
            np.testing.assert_allclose(getattr(m,field),getattr(other,field),rtol=0,atol=1e-12,err_msg=field)
    controls = {c: data[c].ctrl.copy() for c in models}
    time_scale = getattr(a, 'time_scale', 1.)
    if source.get('planner') == 'ik':
        from .planner import Hand, plan
        depths = {c: source['chords'][c]['press_depth_m'] for c in models}
        sequence,schedule=plan(Hand(m,p),controls,fingers,depths,route,a.cycles,time_scale)
    else:
        from .tasks import plan
        sequence,schedule=plan(m,controls,fingers,route,a.cycles,a.approach_proximal,a.approach_tendon)
    composed = None
    if a.two_hands:
        from .bimanual import compose
        m,d,composed = compose(a.run/route[0]/'scene.xml', ROOT, p if a.polished else None)
        for c, left in list(controls.items()):
            controls[c] = d.ctrl.copy()
            controls[c][:len(left)] = left
    strummer = None
    if a.two_hands:
        from .strumming import Strummer
        strummer = Strummer(m,d,p,schedule)
    a.output.mkdir(parents=True,exist_ok=False)
    if composed is not None:
        ET.ElementTree(composed).write(a.output/'scene.xml',encoding='unicode')
    report = {'scope':'Scripted continuous chord-route transitions on fixed mount; no strumming, RL, generalization, or robustness claim.',
              'two_hands':a.two_hands,'presentation_only':a.polished,'profile':source['profile'],'mujoco_version':mujoco.__version__,'source_scene_sha256':hashes,
              'route':route,'rollout_reset_count':1,'state_overwrites_after_start':0,'cycles_requested':a.cycles,
              'force_threshold_N':.01,'required_hold_fraction':.95,'hold_settle_s':.5,
              'thresholds_provisional':True,'planner':source.get('planner','legacy'),'time_scale':time_scale,'approach_proximal':a.approach_proximal,'approach_tendon':a.approach_tendon,
              'rules':'Moves permit source/destination intended contacts and contact gaps. Holds require >=95% simultaneous contact after .5s; all phases require no unwanted force >=.01N over their evaluated samples.',
              'phases':[]}
    rows = []
    invalid = False
    context = nullcontext(None)
    if a.view:
        from mujoco import viewer
        context = viewer.launch_passive(m,d)
    with context as window:
        if window:
            from .presentation import set_camera
            set_camera(window.cam, a.camera)
        for cycle,(name,seconds,src,dst) in schedule:
            moving = src != dst
            allowed = []
            for assignment in fingers[src] + fingers[dst]:
                if assignment not in allowed: allowed.append(assignment)
            targets = [s for _,s,_ in fingers[dst]]
            steps = int(round(seconds/m.opt.timestep))
            phase_rows, pairs = [], set()
            start = float(d.time)
            for step in range(steps):
                if window and not window.is_running(): break
                wall = time.monotonic()
                u = (step+1)/steps
                blend = u*u*(3-2*u) if moving else 1.
                with (window.lock() if window else nullcontext()):
                    d.ctrl[:] = (1-blend)*controls[src] + blend*controls[dst]
                    if strummer: strummer.control(d)
                    mujoco.mj_step(m,d)
                    mujoco.mj_forward(m,d)
                    if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                        invalid=True; break
                    if strummer: strummer.observe(d)
                    forces,bad,current = contact_forces(m,d,p,allowed)
                    pairs |= current
                    row = {'time_s':float(d.time),'cycle':cycle,'phase':name,
                           'phase_time_s':float(d.time-start),'unwanted_N':bad,
                           'target_contact':int(all(forces.get(s,0.)>=.01 for s in targets)) if targets else None,
                           'target_force_N':sum(forces.get(s,0.) for s in targets)}
                    rows.append(row);phase_rows.append(row)
                if window:
                    window.sync()
                    time.sleep(max(0.,m.opt.timestep-(time.monotonic()-wall)))
            measured = phase_rows if moving else [r for r in phase_rows if r['phase_time_s']>=.5]
            fraction = float(np.mean([r['target_contact'] for r in measured])) if targets and measured else None
            bad_fraction = float(np.mean([r['unwanted_N']>=.01 for r in measured])) if measured else 1.
            completed = len(phase_rows)==steps
            passed = completed and bad_fraction==0. and (moving or not targets or fraction>=.95)
            item = {'cycle':cycle,'phase':name,'completed':completed,'pass':passed,
                    'target_contact_fraction':fraction,'unwanted_fraction':bad_fraction,
                    'max_unwanted_force_N':max((r['unwanted_N'] for r in phase_rows),default=0.),
                    'unwanted_pairs':sorted(pairs)}
            report['phases'].append(item)
            if name.startswith(('initial_', 'hold_')) or not passed:
                print(json.dumps(item),flush=True)
            if not completed or invalid: break
    report['warnings'] = {str(i):int(w.number) for i,w in enumerate(d.warning) if w.number}
    report['nonfinite_state'] = invalid
    report['cycles'] = []
    for cycle in range(1,a.cycles+1):
        phases = [r for r in report['phases'] if r['cycle']==cycle]
        passed = len(phases)==len(sequence) and all(r['pass'] for r in phases)
        report['cycles'].append({'cycle':cycle,'pass':passed,'failed_phases':[r['phase'] for r in phases if not r['pass']]})
    report['cycles_passed'] = sum(c['pass'] for c in report['cycles'])
    report['simulated_seconds'] = float(d.time)
    report['screen_pass'] = len(report['phases'])==len(schedule) and all(r['pass'] for r in report['phases']) and not invalid and not report['warnings']
    if strummer:
        report['strumming'] = strummer.summary()
        report['scope'] = 'Scripted two-hand chord routes on a fixed instrument; no RL, robustness, or generalization claim.'
        report['screen_pass'] = report['screen_pass'] and report['strumming']['notes']['f1']==1. and report['strumming']['unexpected_upstrokes']==0
        save_json(a.output/'expected_events.json',strummer.expected)
        save_json(a.output/'actual_events.json',strummer.actual)
        print('NOTES:',json.dumps(report['strumming']['notes']))
    with (a.output/'trace.csv').open('w',newline='') as file:
        if rows:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(a.output/'summary.json',report)
    print('CYCLES PASSED:',report['cycles_passed'],'/',a.cycles)
    print('TRANSITION SCREEN PASS:',report['screen_pass'])
    print('Saved',a.output/'summary.json')
    return report
