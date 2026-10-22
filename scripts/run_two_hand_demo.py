"""Two-hand engineering demo with a scripted position-actuated strumming stage."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from bimanual_scene import add_ukulele_visuals, add_right_hand


def score_events(expected, actual, tolerance=.05):
    """Maximum-cardinality chronological matching within string/pitch groups."""
    groups={(e['string'],e['pitch']) for e in expected+actual}
    matched=[]
    for group in sorted(groups):
        left=sorted([e for e in expected if (e['string'],e['pitch'])==group],key=lambda e:e['onset_s'])
        right=sorted([e for e in actual if (e['string'],e['pitch'])==group],key=lambda e:e['onset_s'])
        i=j=0
        while i<len(left) and j<len(right):
            delta=right[j]['onset_s']-left[i]['onset_s']
            if delta < -tolerance:j+=1
            elif delta > tolerance:i+=1
            else:matched.append(delta);i+=1;j+=1
    tp=len(matched);fp=len(actual)-tp;fn=len(expected)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/len(actual) if actual else 0.,
            'recall':tp/len(expected) if expected else 0.,
            'f1':2*tp/(len(expected)+len(actual)) if expected or actual else 0.,
            'matched_mean_absolute_onset_error_ms':float(np.mean(np.abs(matched))*1000) if matched else None}


def check_evaluator():
    e=[{'string':'A','pitch':72,'onset_s':1.}]
    assert score_events(e,e)['f1']==1
    assert score_events(e,[])['fn']==1
    assert score_events(e,e+e)['fp']==1
    assert score_events(e,[dict(e[0],pitch=69)])['tp']==0
    assert score_events(e,[dict(e[0],onset_s=1.2)])['tp']==0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('outputs/c-contact/scene.xml'))
    p.add_argument('--manifest',type=Path,default=Path('manifests/benchmark_v1.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/two-hand-01'))
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--tendon-release-delta',type=float,default=.30)
    p.add_argument('--view',action='store_true')
    p.add_argument('--video',action='store_true')
    a=p.parse_args()
    if a.cycles<1:p.error('cycles must be positive')
    if a.output.exists():p.error('Use a new output directory; previous results are preserved')
    check_evaluator()
    manifest=json.loads(a.manifest.read_text()); strings=manifest['string_order']
    assert strings==['G','C','E','A']
    open_pitch=dict(zip(strings,manifest['open_string_midi']))
    required=dict(zip(strings,manifest['chord_frets']['C']))
    tree=ET.parse(a.scene);root=tree.getroot()
    compiler=root.find('compiler')
    compiler.set('meshdir',str((a.scene.resolve().parent/compiler.get('meshdir')).resolve()))
    add_ukulele_visuals(root)
    stage_origin=add_right_hand(root,Path.cwd())
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('candidate').id);mujoco.mj_forward(m,d)
    nominal=d.ctrl.copy();released=nominal.copy()
    released[m.actuator('lh_A_RFJ3').id]-=.30
    released[m.actuator('lh_A_RFJ0').id]+=a.tendon_release_delta
    for i in range(m.nu):
        if m.actuator_ctrllimited[i]:
            assert m.actuator_ctrlrange[i,0]<=released[i]<=m.actuator_ctrlrange[i,1]
    base_pos=m.body_pos.copy();base_quat=m.body_quat.copy()
    instrument={i for i in range(m.ngeom) if m.geom_bodyid[i]==m.body('ukulele').id}
    hand={i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith(('lh_','strum_rh_'))}
    string_ids={m.geom('string_'+s).id:s for s in strings}
    ring=m.geom('lh_rfdistal_geom_1').id
    instrument_origin=d.body('ukulele').xpos.copy()
    string_y={s:float(d.geom('string_'+s).xpos[1]) for s in strings}
    string_z=float(d.geom('string_A').xpos[2])
    stage_actuators=[m.actuator(f'right_stage_{axis}_servo').id for axis in 'xyz']
    stage_q=[m.jnt_qposadr[m.joint(f'right_stage_{axis}').id] for axis in 'xyz']
    target=d.site('target_A_3').xpos.copy(); reference=d.site('ring_surface_reference').xpos.copy()
    fret_edges=np.array([.38*(1-2**(-n/12)) for n in range(6)])
    cycle_time=8.; bpm=70.;beat=60/bpm;stroke_duration=.30
    y0=min(string_y.values())-.009;y1=max(string_y.values())+.009
    strokes=[{'cycle':c,'start':c*cycle_time+3.+b*beat,'id':c*4+b} for c in range(a.cycles) for b in range(4)]
    expected=[]
    for stroke in strokes:
        for s in strings:
            expected.append({'cycle':stroke['cycle'],'stroke':stroke['id'],'string':s,
                'pitch':open_pitch[s]+required[s],'onset_s':stroke['start']+stroke_duration*(string_y[s]-y0)/(y1-y0),'direction':'D'})
    rows=[];actual=[];emitted=set();wrench=np.zeros(6);previous=d.site('right_strum_tip').xpos.copy();last_event={s:-1. for s in strings};bad_pairs=set()
    a.output.mkdir(parents=True)
    ET.indent(root,space='  ');ET.ElementTree(root).write(a.output/'scene.xml',encoding='unicode')
    context=nullcontext(None)
    if a.view:
        from mujoco import viewer
        context=viewer.launch_passive(m,d)
    renderer=writer=None
    if a.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=480,width=640)
        writer=imageio.get_writer(str(a.output/'performance.mp4'),fps=25)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.19,-.10,.24];cam.distance=.85;cam.azimuth=120;cam.elevation=-30
    try:
        with context as v:
            if v:
                v.cam.lookat[:]=cam.lookat;v.cam.distance=cam.distance;v.cam.azimuth=cam.azimuth;v.cam.elevation=cam.elevation
            for step in range(int(round(a.cycles*cycle_time/m.opt.timestep))):
                tick=time.monotonic()
                if v and not v.is_running():break
                t0=float(d.time);cycle=min(int((t0+1e-9)/cycle_time),a.cycles-1);phase_t=t0-cycle*cycle_time
                if phase_t<.5:blend=0.;phase='initial_hold'
                elif phase_t<1.:blend=(phase_t-.5)/.5;phase='release_ramp'
                elif phase_t<2.:blend=1.;phase='released'
                elif phase_t<2.5:blend=1-(phase_t-2.)/.5;phase='return_ramp'
                else:blend=0.;phase='playing_hold'
                blend=blend*blend*(3-2*blend)
                # Probe motion is scheduled independently of fret readiness.
                # No readiness gating, skipped strokes, or score time shifting.
                t1=t0+m.opt.timestep
                stroke=next((s for s in strokes if s['start']-1e-9<=t1<=s['start']+stroke_duration+1e-9),None)
                def desired_tip(t):
                    def smooth(x):
                        x=np.clip(x,0,1);return x*x*(3-2*x)
                    for st in strokes:
                        dt=t-st['start']
                        if -.25<=dt<0:
                            return np.array([.29,y0,.18+(string_z+.0007-.18)*smooth((dt+.25)/.25)])
                        if 0<=dt<=stroke_duration:
                            return np.array([.29,y0+(y1-y0)*dt/stroke_duration,string_z+.0007])
                        if stroke_duration<dt<=stroke_duration+.15:
                            return np.array([.29,y1,string_z+.0007+(.18-string_z-.0007)*smooth((dt-stroke_duration)/.15)])
                        if stroke_duration+.15<dt<=stroke_duration+.25:
                            return np.array([.29,y1+(y0-y1)*smooth((dt-stroke_duration-.15)/.10),.18])
                    return np.array([.29,y0,.18])
                probe=desired_tip(t1)
                velocity=(desired_tip(t1+.0001)-desired_tip(t1-.0001))/.0002
                with (v.lock() if v else nullcontext()):
                    d.ctrl[:]=nominal+(released-nominal)*blend
                    d.ctrl[stage_actuators]=d.qpos[stage_q]+(probe-d.site('right_strum_tip').xpos)+.025*velocity
                    mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite simulation')
                    active={s:{} for s in strings};bad=0.;self_force=0.;intended_force=0.
                    for k,c in enumerate(d.contact):
                        mujoco.mj_contactForce(m,d,k,wrench);f=max(0.,float(wrench[0]));pair={int(c.geom1),int(c.geom2)}
                        if pair<=hand:self_force+=f
                        if not(pair & instrument and pair & hand):continue
                        sid=next((gid for gid in pair if gid in string_ids),None)
                        fret=0
                        if sid is not None:
                            x=float(c.pos[0]-instrument_origin[0]);fret=int(np.searchsorted(fret_edges,x,side='right'))
                            if 1<=fret<=5:
                                name=string_ids[sid];active[name][fret]=active[name].get(fret,0.)+f
                        intended=sid==m.geom('string_A').id and ring in pair and fret==3
                        if intended:intended_force+=f
                        else:
                            bad+=f
                            if f>=.01:bad_pairs.add(tuple(sorted([m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name])))
                    frets={s:max([n for n,f in active[s].items() if f>=.01],default=0) for s in strings}
                    ready=all(frets[s]==required[s] for s in strings)
                    # Crossings measured from the physical hand's actual fingertip.
                    # No schedule gating: unexpected low return strokes also score.
                    probe=d.site('right_strum_tip').xpos.copy()
                    dy=probe[1]-previous[1]
                    if abs(dy)/m.opt.timestep>=.05:
                        direction='D' if dy>0 else 'U'
                        for s in strings:
                            crossed=(previous[1]<string_y[s]<=probe[1]) if dy>0 else (probe[1]<=string_y[s]<previous[1])
                            if not crossed or t0-last_event[s]<.03:continue
                            fraction=(string_y[s]-previous[1])/dy
                            z=previous[2]+fraction*(probe[2]-previous[2])
                            x=previous[0]+fraction*(probe[0]-previous[0])
                            if abs(z-string_z)>.002 or not .20<x<.38:continue
                            last_event[s]=t0
                            actual.append({'cycle':cycle,'stroke':stroke['id'] if stroke else -1,'string':s,
                                'pitch':open_pitch[s]+frets[s],'onset_s':t0+fraction*m.opt.timestep,
                                'direction':direction,'detected_fret':frets[s],'chord_ready':ready})
                    rows.append({'time_s':float(d.time),'cycle':cycle,'phase':phase,'phase_time_s':phase_t,
                                 'intended_force_N':intended_force,'unwanted_force_N':bad,'self_contact_force_N':self_force,
                                 'right_tip_x':float(probe[0]),'right_tip_y':float(probe[1]),'right_tip_z':float(probe[2]),'chord_ready':int(ready),'tip_drift_mm':float(np.linalg.norm(d.site('ring_surface_reference').xpos-reference)*1000)})
                previous=probe.copy()
                if writer and step%max(1,round(.04/m.opt.timestep))==0:
                    renderer.update_scene(d,camera=cam);writer.append_data(renderer.render())
                if v:v.sync();time.sleep(max(0.,m.opt.timestep-(time.monotonic()-tick)))
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
    with (a.output/'trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    per_cycle=[]
    for c in range(a.cycles):
        cr=[r for r in rows if r['cycle']==c]
        release=[r for r in cr if 1.5<=r['phase_time_s']<2.]
        hold=[r for r in cr if r['phase_time_s']>=3.]
        per_cycle.append({'cycle':c,'clean_release_fraction':float(np.mean([r['intended_force_N']<.01 and r['unwanted_force_N']<.01 for r in release])) if release else None,
                          'hold_fraction':float(np.mean([r['intended_force_N']>=.01 for r in hold])) if hold else None,
                          'unwanted_fraction':float(np.mean([r['unwanted_force_N']>=.01 for r in cr])) if cr else None,
                          'notes':score_events([e for e in expected if e['cycle']==c],[e for e in actual if e['cycle']==c])})
    completed=bool(rows and rows[-1]['time_s']>=a.cycles*cycle_time-m.opt.timestep/2)
    warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(d.warning) if w.number}
    summary={'scope':'Two Shadow Hands: scripted fretting and a right hand on a 3-axis position-actuated stage. Not a full robot arm or learned controller. Right-hand/string collisions disabled; other collisions retained.',
             'mujoco_version':mujoco.__version__,'tendon_release_delta':a.tendon_release_delta,'cycles_requested':a.cycles,'completed':completed,
             'fretting_mount_fixed':bool(np.array_equal(base_pos,m.body_pos) and np.array_equal(base_quat,m.body_quat)),
             'notes':score_events(expected,actual),'per_cycle':per_cycle,'warnings':warnings,
             'unwanted_pairs':sorted(bad_pairs),'max_self_contact_force_N':max(r['self_contact_force_N'] for r in rows),
             'note_force_threshold_N':.01,'timing_tolerance_s':.05,
             'detector_limitations':'Provisional force threshold; approximate swept point crossing with height tolerance and 30ms debounce. Not acoustics or validated real-string mechanics.',
             'timing_commands_are_scripted':True,'onsets_measured_from_actual_hand':True,
             'right_hand_settings':{'gravity_compensation':True,'integrator':'implicitfast','joint_position_kp':20,'joint_position_kv':1,'joint_force_limit':10,'strum_height_offset_m':.0007,'stage_kp':8000,'stage_kv':200,'stage_force_limit_N':200,'cartesian_velocity_feedforward_s':.025},
             'unexpected_upstroke_events':sum(e['direction']=='U' for e in actual)}
    summary['screen_pass']=completed and not warnings and summary['fretting_mount_fixed'] and summary['unexpected_upstroke_events']==0 and all(
        c['clean_release_fraction'] is not None and c['clean_release_fraction']>=.95 and
        c['hold_fraction']>=.95 and c['unwanted_fraction']==0 and c['notes']['f1']==1 for c in per_cycle)
    for name,value in [('summary',summary),('expected_events',expected),('actual_events',actual)]:
        (a.output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(f'Artifacts: {a.output}')


if __name__=='__main__':main()
