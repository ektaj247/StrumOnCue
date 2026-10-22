"""Bounded development sweep. All candidates and failures are preserved."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('outputs/c-contact/scene.xml'))
    p.add_argument('--output',type=Path,default=Path('outputs/c-coordinated-sweep'))
    a=p.parse_args()
    if a.output.exists():
        p.error('Output directory already exists; choose a new --output to preserve results')
    a.output.mkdir(parents=True)
    test=Path(__file__).with_name('test_c_coordinated.py')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # The original failed baseline, followed by a fixed 5 x 5 development grid.
    candidates=[(0.,.35)]+list(itertools.product([-.3,-.15,0.,.15,.3],[-.3,-.15,.15,.3,.6]))
    rows=[]
    for index,(prox,tendon) in enumerate(candidates):
        run=a.output/f'candidate_{index:02d}';run.mkdir()
        cmd=[sys.executable,str(test),'--scene',str(a.scene.resolve()),'--output',str(run),
             '--proximal-release-delta',str(prox),'--release-delta',str(tendon)]
        with (run/'console.log').open('w') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        row={'candidate':index,'proximal_delta':prox,'tendon_delta':tendon,'returncode':result.returncode}
        summary=run/'summary.json'
        if result.returncode==0 and summary.exists():
            s=json.loads(summary.read_text())
            row.update(status=s['status'],clean_release=s['clean_release_fraction_last_0_5s'],
                       final_hold=s['final_hold_fraction_after_3s'],
                       unwanted=s['unwanted_contact_fraction_whole_run'],
                       final_force_N=s['final_hold_mean_force_N'])
        else:row['status']='EXECUTION_ERROR'
        rows.append(row);print(row,flush=True)
    fields=['candidate','proximal_delta','tendon_delta','returncode','status','clean_release','final_hold','unwanted','final_force_N']
    with (a.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    passed=[r for r in rows if r['status']=='REACQUIRE_SCREEN_PASS' and r['clean_release']>=.95]
    (a.output/'results.json').write_text(json.dumps({'candidates':rows,'passing_candidates':passed},indent=2)+'\n')
    print(f'Passing candidates: {len(passed)}/{len(rows)}. Results: {a.output}')
    print('Development selection only; passing once does not establish repeatability or robustness.')


if __name__=='__main__':main()
