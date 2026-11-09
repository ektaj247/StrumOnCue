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
    models, data, fingers, hashes = {}, {}, {}, {}
    for chord in ['C', 'Am']:
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
    m, d = models['C'], data['C']
    other = models['Am']
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
    composed = None
    if a.two_hands:
        from .bimanual import compose
        m,d,composed = compose(a.run/'C/scene.xml', ROOT, p if a.polished else None)
        for c, left in list(controls.items()):
            controls[c] = d.ctrl.copy()
            controls[c][:len(left)] = left
    controls['clear'] = controls['C'].copy()
    for finger in ['FF','MF','RF','LF']:
        for joint, value in [('J4',0.),('J3',1.2),('J0',1.4)]:
            controls['clear'][m.actuator('lh_A_'+finger+joint).id] = value
    controls['home'] = controls['clear'].copy()
    for finger in ['FF','MF','RF','LF']:
        controls['home'][m.actuator('lh_A_'+finger+'J0').id] = 0.
    controls['pre_Am'] = controls['Am'].copy()
    controls['pre_Am'][m.actuator('lh_A_MFJ0').id] = a.approach_tendon
    controls['pre_Am'][m.actuator('lh_A_MFJ3').id] = a.approach_proximal
    controls['land_Am'] = controls['Am'].copy()
    controls['land_Am'][m.actuator('lh_A_MFJ3').id] = a.approach_proximal
    controls['pre_C'] = controls['C'].copy()
    controls['pre_C'][m.actuator('lh_A_RFJ0').id] = 1.4
    for ctrl in controls.values():
        limited = m.actuator_ctrllimited.astype(bool)
        if np.any(ctrl[limited]<m.actuator_ctrlrange[limited,0]) or np.any(ctrl[limited]>m.actuator_ctrlrange[limited,1]):
            raise ValueError('Control outside actuator limits')
    fingers.update({'clear':[], 'home':[], 'pre_C':[], 'pre_Am':[], 'land_Am':fingers['Am']})
    sequence = [
        ('release_C',1.,'C','clear'),
        ('home_before_Am',3.,'clear','home'),
        ('recurl_before_Am',1.,'home','clear'),
        ('clear_before_Am',1.,'clear','clear'),
        ('align_Am',1.,'clear','pre_Am'),
        ('lower_Am',1.,'pre_Am','land_Am'),
        ('press_Am',1.,'land_Am','Am'),
        ('hold_Am',5.,'Am','Am'),
        ('unpress_Am',1.,'Am','land_Am'),
        ('lift_Am',1.,'land_Am','pre_Am'),
        ('release_Am',1.,'pre_Am','clear'),
        ('home_before_C',3.,'clear','home'),
        ('recurl_before_C',1.,'home','clear'),
        ('clear_before_C',1.,'clear','clear'),
        ('align_C',1.,'clear','pre_C'),
        ('press_C',1.,'pre_C','C'),
        ('returned_C',5.,'C','C'),
    ]
    schedule = [(0,('initial_C',3.,'C','C'))]
    schedule += [(cycle,phase) for cycle in range(1,a.cycles+1) for phase in sequence]
    strummer = None
    if a.two_hands:
        from .strumming import Strummer
        strummer = Strummer(m,d,p,schedule)
    a.output.mkdir(parents=True,exist_ok=False)
    if composed is not None:
        ET.ElementTree(composed).write(a.output/'scene.xml',encoding='unicode')
    report = {'scope':'Scripted continuous C-Am-C transitions on fixed mount; no strumming, RL, generalization, or robustness claim.',
              'two_hands':a.two_hands,'presentation_only':a.polished,'profile':source['profile'],'mujoco_version':mujoco.__version__,'source_scene_sha256':hashes,
              'rollout_reset_count':1,'state_overwrites_after_start':0,'cycles_requested':a.cycles,
              'force_threshold_N':.01,'required_hold_fraction':.95,'hold_settle_s':.5,
              'thresholds_provisional':True,'approach_proximal':a.approach_proximal,'approach_tendon':a.approach_tendon,
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
            if name in ['initial_C','hold_Am','returned_C'] or not passed:
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
        report['scope'] = 'Scripted two-hand C-Am-C on a fixed instrument; no RL, robustness, or generalization claim.'
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
