"""Create a provisional C-chord approach pose, not a reachable-chord result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path)
    p.add_argument('--clearance-mm', type=float, default=30.)
    p.add_argument('--view', action='store_true')
    a = p.parse_args()
    if a.clearance_mm < 10:
        p.error('Keep at least 10 mm reference clearance during placement inspection.')
    repo = a.repo.resolve()
    out = (a.output or repo / 'outputs/c-approach').resolve()
    subprocess.run([sys.executable, str(repo / 'scripts/build_fretting_scene.py'),
                    '--repo', str(repo), '--chord', 'C', '--output', str(out)], check=True)
    path = out / 'scene.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    # Reference markers only: these approximate distal-end locations and must
    # be calibrated against the collision mesh before contact scoring.
    names = {'index': 'lh_ffdistal', 'middle': 'lh_mfdistal',
             'ring': 'lh_rfdistal', 'little': 'lh_lfdistal'}
    for name, body in names.items():
        b = root.find(f".//body[@name='{body}']")
        ET.SubElement(b, 'site', name=f'{name}_tip_reference', pos='0 0 .026',
                      size='.002', rgba='0 1 .3 1')
    tree.write(path, encoding='unicode')
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # Rotate the fixed forearm so the ring distal's local +Z points downward.
    # Choosing placement using a target is legitimate scene setup, not IK.
    desired = np.diag([1., -1., -1.])
    delta = desired @ d.body('lh_rfdistal').xmat.reshape(3, 3).T
    forearm_id = m.body('lh_forearm').id
    rotation = delta @ d.body('lh_forearm').xmat.reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(9))
    m.body_quat[forearm_id] = quat
    mujoco.mj_forward(m, d)
    target = d.site('target_A_3').xpos.copy()
    goal = target + [0., 0., a.clearance_mm / 1000]
    m.body_pos[forearm_id] += goal - d.site('ring_tip_reference').xpos
    mujoco.mj_forward(m, d)
    body = root.find(".//body[@name='lh_forearm']")
    body.set('pos', ' '.join(map(str, m.body_pos[forearm_id])))
    body.set('quat', ' '.join(map(str, quat)))
    ET.indent(tree, space='  '); tree.write(path, encoding='unicode')
    # Reload the saved artifact, ensuring the pose survives serialization.
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    error = d.site('ring_tip_reference').xpos - goal
    assert np.linalg.norm(error) < 1e-6
    contacts = []
    for c in d.contact:
        contacts.append({'geom1': m.geom(int(c.geom1)).name,
                         'geom2': m.geom(int(c.geom2)).name,
                         'distance_m': float(c.dist)})
    report = {'status': 'provisional_approach_not_contact_or_ik',
              'reference_sites_calibrated': False,
              'clearance_mm': a.clearance_mm,
              'alignment_error_mm': float(np.linalg.norm(error) * 1000),
              'forearm_position_m': m.body_pos[forearm_id].tolist(),
              'forearm_quaternion_wxyz': quat.tolist(),
              'tip_reference_world_m': {n: d.site(f'{n}_tip_reference').xpos.tolist() for n in names},
              'target_world_m': target.tolist(), 'contacts': contacts}
    (out / 'approach.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {path}')
    if a.view:
        from mujoco import viewer
        with viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [.08, -.10, .23]
            v.cam.distance = .6; v.cam.azimuth = 120; v.cam.elevation = -25
            while v.is_running():
                with v.lock():
                    mujoco.mj_forward(m, d)
                v.sync(); time.sleep(.02)


if __name__ == '__main__':
    main()
