"""Single deterministic release/return test, initialized from C contact."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('outputs/c-contact/scene.xml'))
    p.add_argument('--output',type=Path,default=Path('outputs/c-reacquire-diagnostic'))
    p.add_argument('--release-delta',type=float,default=.35)
    p.add_argument('--force-threshold',type=float,default=.01)
    p.add_argument('--view',action='store_true')
    a=p.parse_args()
    m=mujoco.MjModel.from_xml_path(str(a.scene.resolve()));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('candidate').id);mujoco.mj_forward(m,d)
    nominal=d.ctrl.copy();released=nominal.copy()
    actuator=m.actuator('lh_A_RFJ0').id
    released[actuator]+=a.release_delta
    if a.force_threshold<=0: p.error('force-threshold must be positive')
    for i in range(m.nu):
        if m.actuator_ctrllimited[i] and not m.actuator_ctrlrange[i,0]<=released[i]<=m.actuator_ctrlrange[i,1]:
            p.error(f'Release target outside actuator range: {m.actuator(i).name}')
    base_pos=m.body_pos.copy();base_quat=m.body_quat.copy()
    pair_target={m.geom('string_A').id,m.geom('lh_rfdistal_geom_1').id}
    instrument={i for i in range(m.ngeom) if m.geom_bodyid[i]==m.body('ukulele').id}
    hand={i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith('lh_')}
    low=.38*(1-2**(-2/12));high=.38*(1-2**(-3/12))
    initial_tip=d.site('ring_surface_reference').xpos.copy()
    target=d.site('target_A_3').xpos.copy()
    joint_addresses={name:int(m.jnt_qposadr[m.joint(name).id]) for name in ['lh_RFJ3','lh_RFJ2','lh_RFJ1']}
    tendon_id=m.tendon('lh_RFJ0').id
    rows=[];bad_pairs=set();wrench=np.zeros(6)
    a.output.mkdir(parents=True,exist_ok=True)
    context=nullcontext(None)
    if a.view:
        from mujoco import viewer
        context=viewer.launch_passive(m,d)
    with context as v:
        if v:
            v.cam.lookat[:]=[.06,-.10,.19];v.cam.distance=.45
            v.cam.azimuth=120;v.cam.elevation=-25
        for step in range(int(round(5/m.opt.timestep))):
            tick=time.monotonic()
            if v and not v.is_running():break
            t=step*m.opt.timestep
            # 0-.5: initial hold; .5-1: release ramp; 1-2: release hold;
            # 2-2.5: return ramp; 2.5-5: final hold.
            if t<.5:phase='initial_hold';blend=0.
            elif t<1:phase='release_ramp';blend=(t-.5)/.5
            elif t<2:phase='released';blend=1.
            elif t<2.5:phase='return_ramp';blend=1-(t-2)/.5
            else:phase='final_hold';blend=0.
            # Smooth endpoint velocities without changing commands' bounds.
            blend=blend*blend*(3-2*blend)
            with (v.lock() if v else nullcontext()):
                d.ctrl[:]=nominal+(released-nominal)*blend
                mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    raise RuntimeError('Nonfinite state: stop and inspect dynamics')
                force=0.;unwanted=0.;self_force=0.;target_points=0;neck_force=0.
                for k,c in enumerate(d.contact):
                    mujoco.mj_contactForce(m,d,k,wrench);normal=max(0.,float(wrench[0]))
                    pair={int(c.geom1),int(c.geom2)}
                    if pair==pair_target and low<c.pos[0]<high:
                        force+=normal
                        if normal>0:target_points+=1
                    elif pair & hand and pair & instrument:
                        unwanted+=normal
                        if m.geom('neck').id in pair: neck_force+=normal
                        if normal>=a.force_threshold:
                            bad_pairs.add(tuple(sorted([m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name])))
                    elif pair<=hand:self_force+=normal
                tip=d.site('ring_surface_reference').xpos.copy()
                rows.append({'time_s':float(d.time),'phase':phase,
                             'ring_tendon_target':float(d.ctrl[actuator]),
                             'ring_tendon_actual_rad':float(d.ten_length[tendon_id]),
                             'ring_tendon_actuator_force':float(d.actuator_force[actuator]),
                             **{name+'_rad':float(d.qpos[address]) for name,address in joint_addresses.items()},
                             'reference_x_offset_mm':float((tip[0]-target[0])*1000),
                             'reference_y_offset_mm':float((tip[1]-target[1])*1000),
                             'reference_z_offset_mm':float((tip[2]-target[2])*1000),
                             'neck_force_N':neck_force,
                             'intended_force_N':force,'unwanted_force_N':unwanted,
                             'self_contact_force_N':self_force,
                             'intended_contact_points':target_points,
                             'reference_displacement_mm':float(np.linalg.norm(tip-initial_tip)*1000)})
            if v:
                v.sync();time.sleep(max(0.,m.opt.timestep-(time.monotonic()-tick)))
    if not rows:raise RuntimeError('No steps recorded')
    with (a.output/'trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    release=[r for r in rows if 1.5<=r['time_s']<2.]
    final=[r for r in rows if r['time_s']>=3.]
    if not release or not final:raise RuntimeError('Incomplete rollout: trace saved, do not count as a pass')
    released_fraction=float(np.mean([r['intended_force_N']<a.force_threshold for r in release]))
    held_fraction=float(np.mean([r['intended_force_N']>=a.force_threshold for r in final]))
    unwanted_fraction=float(np.mean([r['unwanted_force_N']>=a.force_threshold for r in rows]))
    warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(d.warning) if w.number}
    base_fixed=bool(np.array_equal(base_pos,m.body_pos) and np.array_equal(base_quat,m.body_quat))
    completed=rows[-1]['time_s']>=5-m.opt.timestep/2
    passed=completed and base_fixed and not warnings and released_fraction>=.95 and held_fraction>=.95 and unwanted_fraction==0
    contact_after_return=next((r['time_s'] for r in rows if r['time_s']>=2 and r['intended_force_N']>=a.force_threshold),None)
    summary={'status':'REACQUIRE_SCREEN_PASS' if passed else 'REACQUIRE_SCREEN_FAIL',
             'mujoco_version':mujoco.__version__,'completed':completed,'duration_s':rows[-1]['time_s'],
             'fixed_forearm':base_fixed,'release_actuator':'lh_A_RFJ0',
             'nominal_target':float(nominal[actuator]),'released_target':float(released[actuator]),
             'force_threshold_N':a.force_threshold,
             'released_fraction_last_0_5s':released_fraction,'final_hold_fraction_after_3s':held_fraction,
             'unwanted_contact_fraction_whole_run':unwanted_fraction,
             'first_contact_after_return_start_s':contact_after_return,
             'final_hold_mean_force_N':float(np.mean([r['intended_force_N'] for r in final])),
             'max_self_contact_force_N':max(r['self_contact_force_N'] for r in rows),
             'unwanted_pairs':sorted(bad_pairs),'warnings':warnings,
             'scope':'Single scripted release/return from initialized C contact. No RL, recovery policy, or multi-chord claim.'}
    summary['clean_release_fraction_last_0_5s']=float(np.mean([
        r['intended_force_N']<a.force_threshold and r['unwanted_force_N']<a.force_threshold
        for r in release]))
    first_neck=next((r for r in rows if r['neck_force_N']>=a.force_threshold),None)
    summary['first_neck_contact']=first_neck
    summary['phase_snapshots']={label:min(rows,key=lambda r:abs(r['time_s']-t))
        for label,t in [('before_release',.48),('released',1.9),('after_return',2.6),('end',5.)]}
    summary['reference_position_is_not_surface_clearance']=True
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(f'Trace: {a.output / "trace.csv"}')


if __name__=='__main__':main()
