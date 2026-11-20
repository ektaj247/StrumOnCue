"""Chord table shared by the RL scene, command, rewards and evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'assets/rl_chords_v1.json'
HAND_XML = ROOT / 'external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls/left_hand.xml'

FINGERS = ('FF', 'MF', 'RF', 'LF')
# Policy action layout: splay, proximal and coupled-distal target per finger.
ACTION_JOINTS = tuple(f'lh_{f}{j}' for j in ('J4', 'J3') for f in FINGERS)
ACTION_TENDONS = tuple(f'lh_{f}J0' for f in FINGERS)
MAX_FRET = 3          # highest fret any benchmark chord uses; goal is one-hot over 0..3


def load_table(path=TABLE):
    table = json.loads(Path(path).read_text())
    if table['schema'] != 1 or tuple(table['fingers']) != FINGERS:
        raise ValueError('Unsupported chord table: ' + str(path))
    table['chord_names'] = list(table['chords'])
    return table


def pairs(table, split):
    """Directed (source, target) chord pairs for a split: train, heldout or all."""
    if split == 'all':
        return table['transitions']['train'] + table['transitions']['heldout']
    return table['transitions'][split]
