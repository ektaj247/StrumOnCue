"""Search a static C contact candidate. This is not IK or actuated success."""
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
    p.add_argument('--view', action='store_true')
    a = p.parse_args(); repo = a.repo.resolve()
    out = (a.output or repo / 'outputs/c-contact').resolve()
    subprocess.run([sys.executable, str(repo / 'scripts/build_fretting_scene.py'),
                    '--repo', str(repo), '--chord', 'C', '--output', str(out)], check=True)
    path = out / 'scene.xml'; tree = ET.parse(path); root = tree.getroot()
    # Name collision geometry for readable contact diagnostics.
    for body in root.findall('.//body'):
        for i, geom in enumerate(body.findall('geom')):
            if not geom.get('name'):
                geom.set('name', f'{body.get("name")}_geom_{i}')
    tree.write(path, encoding='unicode')
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    forearm = m.body('lh_forearm').id; distal = m.body('lh_rfdistal').id
    geom_id = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == distal
                   and (m.geom_contype[i] or m.geom_conaffinity[i]))
    mesh = int(m.geom_dataid[geom_id]); start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
    vertices = np.array(m.mesh_vert[start:start+count], dtype=float)
    def set_joint(name, value):
        j = m.joint(name).id
        assert m.jnt_range[j, 0] <= value <= m.jnt_range[j, 1]
        d.qpos[m.jnt_qposadr[j]] = value
    # A manually chosen bent posture, not an optimized solution.
    for finger in ['FF', 'MF', 'RF', 'LF']:
        angles = (.45, .35, .35) if finger == 'RF' else (1.2, .7, .7)
        for suffix, value in zip(['J3', 'J2', 'J1'], angles):
            set_joint(f'lh_{finger}{suffix}', value)
    mujoco.mj_forward(m, d)
    desired = np.diag([1., -1., -1.])
    rotation = desired @ d.xmat[distal].reshape(3, 3).T @ d.xmat[forearm].reshape(3, 3)
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, rotation.reshape(9))
    m.body_quat[forearm] = quat; mujoco.mj_forward(m, d)
    world = vertices @ d.geom_xmat[geom_id].reshape(3, 3).T + d.geom_xpos[geom_id]
    lowest = world[np.argmin(world[:, 2])]
    marker_local = d.xmat[distal].reshape(3, 3).T @ (lowest-d.xpos[distal])
    target = d.site('target_A_3').xpos.copy()
    m.body_pos[forearm] += target + [0, 0, .001] - lowest
    base = m.body_pos[forearm].copy()
    string = m.geom('string_A').id
    fret_start = .38*(1-2**(-2/12)); fret_end = .38*(1-2**(-3/12))
    history = []; chosen = None
    for descent in np.linspace(0, .002, 41):
        m.body_pos[forearm] = base - [0, 0, descent]
        mujoco.mj_forward(m, d)
        contacts = []
        for c in d.contact:
            if c.dist > 0:
                continue
            pair = {int(c.geom1), int(c.geom2)}
            intended = pair == {geom_id, string} and fret_start < c.pos[0] < fret_end
            contacts.append({'geom1': m.geom(int(c.geom1)).name,
                             'geom2': m.geom(int(c.geom2)).name,
                             'distance_mm': float(c.dist*1000),
                             'position_m': c.pos.tolist(), 'intended': bool(intended)})
        good = bool(contacts) and all(c['intended'] for c in contacts)
        history.append({'descent_mm': float(descent*1000), 'contacts': contacts})
        if good:
            chosen = history[-1]; break
    status = 'STATIC_CONTACT_CANDIDATE' if chosen else 'NO_CLEAN_CANDIDATE'
    # Save the inspected posture as an explicit keyframe, including tendon
    # actuator lengths as provisional targets. Dynamics are not tested here.
    d.ctrl[:] = d.actuator_length
    body = root.find(".//body[@name='lh_forearm']")
    body.set('pos', ' '.join(map(str, m.body_pos[forearm])))
    body.set('quat', ' '.join(map(str, quat)))
    ET.SubElement(root.find(".//body[@name='lh_rfdistal']"), 'site',
                  name='ring_surface_reference', pos=' '.join(map(str, marker_local)),
                  size='.002', rgba='0 1 .2 1')
    keyframe = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframe, 'key', name='candidate',
                  qpos=' '.join(map(str, d.qpos)), ctrl=' '.join(map(str, d.ctrl)))
    ET.indent(tree, space='  '); tree.write(path, encoding='unicode')
    report = {'status': status, 'mujoco_version': mujoco.__version__,
              'dynamic_control_validated': False, 'forearm_fixed_for_all_chords': False,
              'surface_marker_body_local_m': marker_local.tolist(),
              'candidate': chosen, 'scan': history}
    (out/'contact_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{status}: {out / "contact_report.json"}')
    print(json.dumps(chosen or history[-1], indent=2))
    print('Only static geometry tested; no contact-force or sustained-control claim.')
    if a.view:
        from mujoco import viewer
        model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
        with viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [.06, -.10, .19]; v.cam.distance=.45
            v.cam.azimuth=120; v.cam.elevation=-25
            while v.is_running():
                with v.lock(): mujoco.mj_forward(model, data)
                v.sync(); time.sleep(.02)


if __name__ == '__main__':
    main()
