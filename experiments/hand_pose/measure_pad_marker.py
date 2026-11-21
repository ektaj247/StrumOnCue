"""Move the fingertip marker to where the finger really touches the string.

The planner aims one marker point on each fingertip (the site
`ring_surface_reference`, given in the distal segment's own frame) at the
press point. That marker was measured for the original pose, where fingers
come straight down. In the human-like grip a different part of the fingertip
meets the string.

This script reads a calibrated run, holds each chord for one second, collects
every contact between an assigned fingertip and its own string, expresses each
contact point in that fingertip's own frame, and averages them. It then writes
a copy of the seed folder with the marker moved there. Nothing else changes.

    .venv/Scripts/python experiments/hand_pose/measure_pad_marker.py \
        --run outputs/holds-human --seed outputs/hand-pose/20261119-175825/seed \
        --output outputs/hand-pose/seed-contact-marker
"""
import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strum.fretting import load  # noqa: E402


def contact_points(run):
    """Contact points of assigned fingertips on their own strings, in the fingertip frame (m)."""
    rows = []
    for chord_dir in sorted(p for p in run.iterdir() if (p / 'scene.xml').exists()):
        fingering = json.loads((chord_dir / 'summary.json').read_text())['fingering']
        m, d = load(chord_dir / 'scene.xml')
        while d.time < 1.0:
            mujoco.mj_step(m, d)
        for finger, string, _ in fingering:
            body = m.body(f'lh_{finger.lower()}distal').id
            target = m.geom(f'string_{string}').id
            for k in range(d.ncon):
                c = d.contact[k]
                pair = {int(c.geom1), int(c.geom2)}
                if target not in pair:
                    continue
                if body not in {int(m.geom_bodyid[g]) for g in pair}:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, wrench)
                local = d.xmat[body].reshape(3, 3).T @ (c.pos - d.xpos[body])
                rows.append((chord_dir.name, finger, string, local, float(wrench[0])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='calibrated run in the human-like pose')
    parser.add_argument('--seed', type=Path, required=True, help='seed folder that run was calibrated from')
    parser.add_argument('--output', type=Path, required=True, help='new seed folder to write')
    a = parser.parse_args()

    rows = contact_points(a.run)
    if not rows:
        raise SystemExit('No fingertip-on-own-string contacts found')
    np.set_printoptions(precision=2, suppress=True)
    print('Contact points in the fingertip frame (mm):')
    for chord, finger, string, local, force in rows:
        print(f'  {chord:2s} {finger} on {string}: {local * 1000}  force {force:.3f} N')
    points = np.array([r[3] for r in rows])
    marker = points.mean(axis=0)
    spread = np.linalg.norm(points - marker, axis=1).max()

    old_root = ET.parse(next(a.seed.glob('x_30-*.xml'))).getroot()
    old = np.array(old_root.find(".//site[@name='ring_surface_reference']").get('pos').split(), dtype=float)
    print(f'\nOld marker: {old * 1000} mm')
    print(f'New marker: {marker * 1000} mm   (moved {np.linalg.norm(marker - old) * 1000:.1f} mm;'
          f' contacts lie within {spread * 1000:.1f} mm of it)')

    a.output.mkdir(parents=True, exist_ok=False)
    for source in sorted(a.seed.glob('x_30-*.xml')):
        root = ET.parse(source).getroot()
        compiler = root.find('compiler')
        assets = (source.parent / compiler.get('meshdir', '')).resolve()
        compiler.set('meshdir', str(Path(__import__('os').path.relpath(assets, a.output))).replace('\\', '/'))
        root.find(".//site[@name='ring_surface_reference']").set('pos', ' '.join(map(str, marker)))
        ET.ElementTree(root).write(a.output / source.name)
    shutil.copy2(a.seed / 'summary.json', a.output / 'summary.json')
    (a.output / 'marker.json').write_text(json.dumps({
        'from_run': str(a.run), 'from_seed': str(a.seed), 'old_marker_m': old.tolist(),
        'new_marker_m': marker.tolist(), 'contacts_used': len(rows), 'max_spread_m': float(spread)}, indent=2) + '\n')
    print('Wrote', a.output)


if __name__ == '__main__':
    main()
