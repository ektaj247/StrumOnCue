"""Write the chord table the RL task trains against.

Reads a calibrated run (python -m strum.fretting calibrate) and stores, per
chord, the settled joint state, the holding command and where each fretting
fingertip ends up. The RL task loads only this file, so training does not
need the calibration outputs or scipy. This module lives outside strum.rl so
it runs where mjlab is not installed.

    python -m strum.export_chords --run outputs/ek-holds-06 \
        --output assets/rl_chords_v1.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .fretting import ROOT, FRETBOARD_RULE, load
from .planner import FINGERS, Hand
from .tasks import audit
from .ukulele import CHORDS, STRING_NAMES, fret_positions, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifests/benchmark_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    source = json.loads((a.run / 'summary.json').read_text())
    if not source.get('all_pass') or source.get('planner') != 'ik':
        raise ValueError('Need a fully accepted run made with the ik planner')
    p = profile(source['profile'])
    if p.to_dict() != source['parameters']:
        raise ValueError('Geometry changed since calibration')
    split = audit(a.manifest)
    table = {'schema': 1, 'profile': source['profile'], 'parameters': source['parameters'],
             'fretboard_rule': source.get('fretboard_rule', FRETBOARD_RULE),
             'string_names': list(STRING_NAMES), 'fingers': list(FINGERS),
             'fret_edges': [0.] + fret_positions(p.scale_length, p.n_frets),
             'transitions': {'train': [list(x) for x in split['training_transitions']],
                             'heldout': [list(x) for x in split['heldout_transitions']]},
             'manifest_sha256': split['manifest_sha256'], 'chords': {}}
    for chord, record in source['chords'].items():
        scene = a.run / chord / 'scene.xml'
        if hashlib.sha256(scene.read_bytes()).hexdigest() != record['scene_sha256']:
            raise ValueError('Scene changed since calibration: ' + chord)
        m, d = load(scene)
        hand = Hand(m, p)
        joints = [m.joint(i).name for i in range(m.njnt)]
        actuators = [m.actuator(i).name for i in range(m.nu)]
        targets = []
        for i in range(m.nu):
            kind = mujoco.mjtObj.mjOBJ_JOINT if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else mujoco.mjtObj.mjOBJ_TENDON
            targets.append(mujoco.mj_id2name(m, kind, int(m.actuator_trnid[i, 0])))
        root = ET.parse(scene).getroot()
        mount = root.find("worldbody/body[@name='lh_forearm']")
        shared = {'joint_names': joints, 'actuator_names': actuators, 'actuator_targets': targets,
                  'mount': {'pos': [float(x) for x in mount.get('pos').split()],
                            'quat': [float(x) for x in mount.get('quat').split()]},
                  'ukulele_pos': [float(x) for x in d.xpos[m.body('ukulele').id]],
                  'pad_offset': [float(x) for x in hand.pad],
                  'rest': {'qpos': hand.rest.tolist(), 'ctrl': hand.command(hand.rest).tolist()}}
        for key, value in shared.items():
            if table.setdefault(key, value) != value:
                raise ValueError('Chord scenes disagree on ' + key)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        assert CHORDS[chord] == tuple(max([n for _, s, n in record['fingering'] if s == name], default=0) for name in STRING_NAMES)
        table['chords'][chord] = {
            'frets': list(CHORDS[chord]), 'fingering': record['fingering'],
            'press_depth_m': record['press_depth_m'],
            'qpos': d.qpos.tolist(), 'ctrl': d.ctrl.tolist(),
            'pads': {f: [float(x) for x in hand.pad_local(d, f)] for f, _, _ in record['fingering']},
            'scene_sha256': record['scene_sha256']}
    a.output.write_text(json.dumps(table, indent=1) + '\n')
    print('Wrote', a.output, 'with chords', list(table['chords']))


if __name__ == '__main__':
    main()
