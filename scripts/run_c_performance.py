"""Integrated engineering demo: actuated C fretting + scripted noncolliding probe.
No RL, second Shadow Hand, acoustic simulation, or generalization claim.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


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
    p.add_argument('--output',type=Path,default=Path('outputs/c-performance-01'))
    p.add_argument('--cycles',type=int,default=10)
    p.add_argument('--tendon-release-delta',type=float,default=.15)
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
    body=ET.SubElement(root.find('worldbody'),'body',name='strumming_probe',mocap='true',pos='.29 -.15 .18')
    ET.SubElement(body,'geom',name='probe_visual',type='sphere',size='.003',rgba='1 .1 .8 1',contype='0',conaffinity='0')
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
    hand={i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith('lh_')}
    string_ids={m.geom('string_'+s).id:s for s in strings}
    ring=m.geom('lh_rfdistal_geom_1').id
    instrument_origin=d.body('ukulele').xpos.copy()
    string_y={s:float(d.geom('string_'+s).xpos[1]) for s in strings}
    string_z=float(d.geom('string_A').xpos[2]);probe_id=int(m.body_mocapid[m.body('strumming_probe').id])
    target=d.site('target_A_3').xpos.copy(); reference=d.site('ring_surface_reference').xpos.copy()
    fret_edges=np.array([.38*(1-2**(-n/12)) for n in range(6)])
    cycle_time=8.; bpm=70.;beat=60/bpm;stroke_duration=.12
    y0=min(string_y.values())-.009;y1=max(string_y.values())+.009
    strokes=[{'cycle':c,'start':c*cycle_time+3.+b*beat,'id':c*4+b} for c in range(a.cycles) for b in range(4)]
    expected=[]
    for stroke in strokes:
        for s in strings:
            expected.append({'cycle':stroke['cycle'],'stroke':stroke['id'],'string':s,
                'pitch':open_pitch[s]+required[s],'onset_s':stroke['start']+stroke_duration*(string_y[s]-y0)/(y1-y0),'direction':'D'})
    rows=[];actual=[];emitted=set();wrench=np.zeros(6);previous=np.array([.29,y0,.18]);bad_pairs=set()
    a.output.mkdir(parents=True)
    context=nullcontext(None)
    if a.view:
        from mujoco import viewer
        context=viewer.launch_passive(m,d)
    renderer=writer=None
    if a.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=480,width=640)
        writer=imageio.get_writer(str(a.output/'performance.mp4'),fps=25)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.16,-.10,.20];cam.distance=.65;cam.azimuth=120;cam.elevation=-30
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
                if stroke:
                    alpha=np.clip((t1-stroke['start'])/stroke_duration,0,1)
                    probe=np.array([.29,y0+alpha*(y1-y0),string_z])
                else:
                    soon=next((s for s in strokes if 0<s['start']-t1<.05),None)
                    probe=np.array([.29,y0,string_z if soon else .18])
                with (v.lock() if v else nullcontext()):
                    d.ctrl[:]=nominal+(released-nominal)*blend;d.mocap_pos[probe_id]=probe
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
                    # Swept crossings, speed and height checks, one event per
                    # string per stroke. Pitches come from contacts, not goals.
                    if stroke and probe[1]>previous[1] and (probe[1]-previous[1])/m.opt.timestep>=.05:
                        for s in strings:
                            key=(stroke['id'],s)
                            if key in emitted or not(previous[1]<string_y[s]<=probe[1]):continue
                            fraction=(string_y[s]-previous[1])/(probe[1]-previous[1])
                            z=previous[2]+fraction*(probe[2]-previous[2])
                            if abs(z-string_z)>.002:continue
                            emitted.add(key);actual.append({'cycle':cycle,'stroke':stroke['id'],'string':s,
                                'pitch':open_pitch[s]+frets[s],'onset_s':t0+fraction*m.opt.timestep,
                                'direction':'D','detected_fret':frets[s],'chord_ready':ready})
                    rows.append({'time_s':float(d.time),'cycle':cycle,'phase':phase,'phase_time_s':phase_t,
                                 'intended_force_N':intended_force,'unwanted_force_N':bad,'self_contact_force_N':self_force,
                                 'chord_ready':int(ready),'tip_drift_mm':float(np.linalg.norm(d.site('ring_surface_reference').xpos-reference)*1000)})
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
    summary={'scope':'Scripted C/P1/70 BPM engineering demo with noncolliding mocap strumming probe; no second hand, RL, audio, or holdout evaluation.',
             'mujoco_version':mujoco.__version__,'tendon_release_delta':a.tendon_release_delta,'cycles_requested':a.cycles,'completed':completed,
             'forearm_fixed':bool(np.array_equal(base_pos,m.body_pos) and np.array_equal(base_quat,m.body_quat)),
             'notes':score_events(expected,actual),'per_cycle':per_cycle,'warnings':warnings,
             'unwanted_pairs':sorted(bad_pairs),'max_self_contact_force_N':max(r['self_contact_force_N'] for r in rows),
             'note_force_threshold_N':.01,'timing_tolerance_s':.05,
             'detector_limitations':'Force threshold provisional; finger surface location/pressure not a validated acoustic note model. Detector only exercises scheduled downstrokes.',
             'timing_is_scripted':True}
    summary['screen_pass']=completed and not warnings and summary['forearm_fixed'] and all(
        c['clean_release_fraction'] is not None and c['clean_release_fraction']>=.95 and
        c['hold_fraction']>=.95 and c['unwanted_fraction']==0 and c['notes']['f1']==1 for c in per_cycle)
    for name,value in [('summary',summary),('expected_events',expected),('actual_events',actual)]:
        (a.output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(f'Artifacts: {a.output}')


if __name__=='__main__':main()
