"""Diagnostic fixed-command C hold; no RL, feedback adaptation, or pose resets."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=Path('outputs/c-contact/scene.xml'))
    p.add_argument('--output', type=Path, default=Path('outputs/c-hold'))
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--settle-seconds', type=float, default=.5)
    p.add_argument('--force-threshold', type=float, default=.01)
    p.add_argument('--view', action='store_true')
    a = p.parse_args()
    if not 0 <= a.settle_seconds < a.seconds or a.force_threshold <= 0:
        p.error('Require 0 <= settle-seconds < seconds and positive force-threshold')
    m = mujoco.MjModel.from_xml_path(str(a.scene.resolve()))
    d = mujoco.MjData(m)
    key = m.key('candidate').id
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    control = d.ctrl.copy(); initial_qpos = d.qpos.copy()
    for i in range(m.nu):
        if m.actuator_ctrllimited[i] and not (m.actuator_ctrlrange[i, 0] <= control[i] <= m.actuator_ctrlrange[i, 1]):
            raise ValueError(f'Saved control outside limits: {m.actuator(i).name}')
    reference = d.site('ring_surface_reference').xpos.copy()
    target = d.site('target_A_3').xpos.copy()
    intended_pair = {m.geom('string_A').id, m.geom('lh_rfdistal_geom_1').id}
    instrument = {i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body('ukulele').id}
    hand = {i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith('lh_')}
    low = .38*(1-2**(-2/12)); high = .38*(1-2**(-3/12))
    wrench = np.zeros(6); rows=[]; unwanted_pairs=set(); invalid=False
    a.output.mkdir(parents=True, exist_ok=True)
    context = nullcontext(None)
    if a.view:
        from mujoco import viewer
        context = viewer.launch_passive(m, d)
    with context as v:
        if v:
            v.cam.lookat[:] = [.06, -.10, .19]; v.cam.distance=.45
            v.cam.azimuth=120; v.cam.elevation=-25
        for step in range(int(np.ceil(a.seconds/m.opt.timestep))):
            start = time.monotonic()
            if v and not v.is_running():
                break
            with (v.lock() if v else nullcontext()):
                d.ctrl[:] = control
                mujoco.mj_step(m, d)
                # Recompute contacts/forces at the updated state for consistent logs.
                mujoco.mj_forward(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    invalid=True; break
                force=0.; bad_force=0.; contact_points=0; self_force=0.
                for k, c in enumerate(d.contact):
                    mujoco.mj_contactForce(m, d, k, wrench)
                    normal=max(0., float(wrench[0])); pair={int(c.geom1),int(c.geom2)}
                    intended=pair==intended_pair and low < c.pos[0] < high
                    if intended:
                        force+=normal
                        if normal > 0: contact_points+=1
                    elif pair & instrument and pair & hand:
                        bad_force+=normal
                        if normal > a.force_threshold:
                            unwanted_pairs.add(tuple(sorted([m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name])))
                    elif pair <= hand:
                        self_force+=normal
                point=d.site('ring_surface_reference').xpos.copy()
                row={'time_s':float(d.time),'intended_force_N':force,
                     'intended_contact_points':contact_points,
                     'unwanted_instrument_force_N':bad_force,'self_contact_force_N':self_force,
                     'tip_drift_mm':float(np.linalg.norm(point-reference)*1000),
                     'tip_target_distance_mm':float(np.linalg.norm(point-target)*1000),
                     'max_joint_drift_deg':float(np.max(np.abs(d.qpos-initial_qpos))*180/np.pi)}
                rows.append(row)
            if v:
                v.sync(); time.sleep(max(0.,m.opt.timestep-(time.monotonic()-start)))
    if not rows: raise RuntimeError('No finite steps recorded')
    with (a.output/'trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    measured=[r for r in rows if r['time_s'] >= a.settle_seconds]
    warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(d.warning) if w.number}
    finished=rows[-1]['time_s'] >= a.seconds-m.opt.timestep/2
    if not measured: raise RuntimeError('Run ended before measurement interval; trace saved')
    contact_rate=float(np.mean([r['intended_force_N'] >= a.force_threshold for r in measured]))
    unwanted_rate=float(np.mean([r['unwanted_instrument_force_N'] >= a.force_threshold for r in measured]))
    forces=[r['intended_force_N'] for r in measured]
    passed=finished and not invalid and not warnings and contact_rate >= .95 and unwanted_rate==0
    summary={'status':'HOLD_SCREEN_PASS' if passed else 'HOLD_SCREEN_FAIL',
             'meaning':'Diagnostic hold from initialized contact; not approach, chord generalization, or RL success',
             'mujoco_version':mujoco.__version__, 'scene':str(a.scene.resolve()),
             'duration_s':rows[-1]['time_s'],'settle_seconds':a.settle_seconds,
             'force_threshold_N':a.force_threshold,'required_contact_fraction':.95,
             'measured_contact_fraction':contact_rate,'unwanted_contact_fraction':unwanted_rate,
             'intended_force_min_mean_max_N':[min(forces),float(np.mean(forces)),max(forces)],
             'max_tip_drift_mm':max(r['tip_drift_mm'] for r in measured),
             'final_joint_drift_deg':rows[-1]['max_joint_drift_deg'],
             'max_self_contact_force_N':max(r['self_contact_force_N'] for r in measured),
             'unwanted_pairs':sorted(unwanted_pairs),'warnings':warnings,
             'nonfinite_state':invalid,'completed':finished,
             'control_targets':{m.actuator(i).name:float(control[i]) for i in range(m.nu)}}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(f'Saved {a.output / "summary.json"} and {a.output / "trace.csv"}')
    print('Screen thresholds are provisional diagnostics, not validated note-force thresholds.')


if __name__=='__main__': main()
