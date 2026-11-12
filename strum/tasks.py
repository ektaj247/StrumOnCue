"""Reusable chord routes and training-only transition coverage."""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .ukulele import CHORDS


def plan(m, controls, fingers, route, cycles, proximal=0., tendon=2.):
    controls['clear'] = controls[route[0]].copy()
    for finger in ['FF','MF','RF','LF']:
        for joint,value in [('J4',0.),('J3',1.2),('J0',1.4)]:
            controls['clear'][m.actuator('lh_A_'+finger+joint).id]=value
    controls['home']=controls['clear'].copy()
    for finger in ['FF','MF','RF','LF']:
        controls['home'][m.actuator('lh_A_'+finger+'J0').id]=0.
    for chord in dict.fromkeys(route):
        controls['pre_'+chord]=controls[chord].copy()
        fingers['pre_'+chord]=[]
        if chord=='C':
            for finger,_,_ in fingers[chord]:
                controls['pre_'+chord][m.actuator('lh_A_'+finger+'J0').id]=1.4
        else:
            controls['land_'+chord]=controls[chord].copy()
            fingers['land_'+chord]=fingers[chord]
            for finger,_,_ in fingers[chord]:
                controls['pre_'+chord][m.actuator('lh_A_'+finger+'J0').id]=tendon if finger=='MF' else 1.4
                if finger=='MF':
                    controls['pre_'+chord][m.actuator('lh_A_'+finger+'J3').id]=proximal
                    controls['land_'+chord][m.actuator('lh_A_'+finger+'J3').id]=proximal
    fingers['clear']=[];fingers['home']=[]
    sequence=[]
    for source,destination in zip(route,route[1:]):
        if source!='C':
            sequence.extend([(f'unpress_{source}',1.,source,'land_'+source),
                             (f'lift_{source}',1.,'land_'+source,'pre_'+source),
                             (f'release_{source}',1.,'pre_'+source,'clear')])
        else:
            sequence.append(('release_C',1.,'C','clear'))
        sequence.extend([(f'home_before_{destination}',3.,'clear','home'),
                         (f'recurl_before_{destination}',1.,'home','clear'),
                         (f'clear_before_{destination}',1.,'clear','clear'),
                         (f'align_{destination}',1.,'clear','pre_'+destination)])
        if destination!='C':
            sequence.extend([(f'lower_{destination}',1.,'pre_'+destination,'land_'+destination),
                             (f'press_{destination}',1.,'land_'+destination,destination)])
        else:
            sequence.append(('press_C',1.,'pre_C','C'))
        sequence.append((f'hold_{destination}',5.,destination,destination))
    for ctrl in controls.values():
        limited=m.actuator_ctrllimited.astype(bool)
        if np.any(ctrl[limited]<m.actuator_ctrlrange[limited,0]) or np.any(ctrl[limited]>m.actuator_ctrlrange[limited,1]):
            raise ValueError('Control outside actuator limits')
    schedule=[(0,(f'initial_{route[0]}',3.,route[0],route[0]))]
    schedule += [(cycle,phase) for cycle in range(1,cycles+1) for phase in sequence]
    return sequence,schedule


def audit(manifest):
    data=json.loads(manifest.read_text())
    assert data['chord_frets']=={k:list(v) for k,v in CHORDS.items()}, 'Manifest chords disagree with geometry module'
    cells=data['cells']
    assert len({c['id'] for c in cells})==len(cells),'Duplicate cell IDs'
    expected={(s,d,p,b) for s in CHORDS for d in CHORDS if s!=d for p in data['patterns'] for b in data['tempos_bpm']}
    actual={(c['source'],c['destination'],c['pattern'],c['bpm']) for c in cells}
    assert actual==expected and len(actual)==len(cells),'Incomplete/duplicated task grid'
    counts={split:sum(c['split']==split for c in cells) for split in ['train','heldout_pair','heldout_transition']}
    assert counts==data['counts'],'Split counts disagree'
    train={(c['source'],c['destination']) for c in cells if c['split']=='train'}
    held={(c['source'],c['destination']) for c in cells if c['split']=='heldout_transition'}
    assert not train & held,'Held-out transition leaked into train'
    train_pairs={(c['destination'],c['pattern']) for c in cells if c['split']=='train'}
    held_pairs={(c['destination'],c['pattern']) for c in cells if c['split']=='heldout_pair'}
    assert not train_pairs & held_pairs,'Held-out chord-pattern pair leaked into train'
    return {'cell_counts':counts,'training_transitions':sorted(train),
            'heldout_transitions':sorted(held),'heldout_chord_pattern_pairs':sorted(held_pairs),
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}


def coverage(a):
    from .transitions import run
    split=audit(a.manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for source,destination in split['training_transitions']:
        name=source+'_to_'+destination
        args=argparse.Namespace(run=a.run,output=a.output/name,cycles=1,view=False,
          polished=False,two_hands=False,camera='overview',approach_proximal=0.,approach_tendon=2.,time_scale=getattr(a,'time_scale',1.),route=[source,destination])
        try:
            with (a.output/(name+'.log')).open('w') as log,contextlib.redirect_stdout(log):
                result=run(args)
            final=result['phases'][-1]
            row={'source':source,'destination':destination,'pass':result['screen_pass'],
                 'destination_hold_fraction':final['target_contact_fraction'],
                 'destination_unwanted_fraction':final['unwanted_fraction'],
                 'failed_phases':';'.join(p['phase'] for p in result['phases'] if not p['pass']),
                 'error':''}
        except Exception as error:
            row={'source':source,'destination':destination,'pass':False,'destination_hold_fraction':None,
                 'destination_unwanted_fraction':None,'failed_phases':'','error':repr(error)}
        results.append(row)
        print(json.dumps(row),flush=True)
        summary={'scope':'Training-transition engineering coverage only; slow scripted left-hand routes, not benchmark scores or held-out evaluation.',
          'manifest':split,'heldout_executed':False,'results':results,
          'passed':sum(r['pass'] for r in results),'attempted':len(results),'complete':len(results)==len(split['training_transitions'])}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'coverage.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print('TRAINING TRANSITIONS PASSED:',summary['passed'],'/',summary['attempted'])
    print('Held-out transitions were not executed.')
