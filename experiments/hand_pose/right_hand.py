"""Strumming hand at a human-like angle (experiment copy of strum.bimanual.add_right_hand).

The project mounts the right hand with the index finger pointing straight down
at the strings. Here, in the ukulele's frame (x along the strings, y from the
G string to the A string, z out of the fretboard):

  * the index finger points toward +y (down toward the floor once the scene is
    turned upright), tilted ANGLE degrees in toward the strings (-z);
  * the back of the hand faces +z (the camera once upright), so the palm faces
    the ukulele.

Everything else is unchanged from add_right_hand: the stage, gains, collision
bits, curled middle/ring/little fingers, the strum-tip site at the index
fingertip point nearest the strings, and its start position.

install(angle) swaps this in for strum.bimanual.add_right_hand, in memory only.
"""
import functools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strum.bimanual as bimanual  # noqa: E402
from strum.fretting import compile_root  # noqa: E402

ANGLE = 40.


def unit(v):
    return v / np.linalg.norm(v)


def frame(forward, normal):
    f = unit(forward)
    n = unit(normal - (normal @ f) * f)
    return np.column_stack([f, n, np.cross(f, n)])


def add_right_hand_angled(root, repo, angle_deg=ANGLE, report=None):
    # --- unchanged from strum.bimanual.add_right_hand ---------------------
    root.find('option').set('integrator', 'implicitfast')
    source = Path(repo) / 'external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls/right_hand.xml'
    right = ET.parse(source).getroot()
    for mesh in right.findall('asset/mesh'):
        if not mesh.get('name'):
            mesh.set('name', Path(mesh.get('file')).stem)
    refs = {'name', 'class', 'childclass', 'material', 'mesh', 'joint', 'tendon', 'body1', 'body2'}
    for e in right.iter():
        for k in refs:
            if k in e.attrib:
                e.set(k, 'strum_' + e.get(k))
    for b in right.findall('.//body'):
        b.set('gravcomp', '1')
        for i, g in enumerate(b.findall('geom')):
            if not g.get('name'):
                g.set('name', b.get('name') + f'_geom_{i}')
    for actuator in right.findall('actuator/position'):
        actuator.set('kp', '20'); actuator.set('kv', '1'); actuator.set('forcerange', '-10 10')
    for section in ['default', 'asset', 'contact', 'tendon', 'actuator']:
        src = right.find(section)
        if src is not None:
            dest = root.find(section)
            if dest is None:
                dest = ET.SubElement(root, section)
            for child in list(src):
                dest.append(child)
    mount = ET.SubElement(root.find('worldbody'), 'body', name='right_stage')
    for axis, vector in zip('xyz', ['1 0 0', '0 1 0', '0 0 1']):
        ET.SubElement(mount, 'joint', name=f'right_stage_{axis}', type='slide', axis=vector, range='-1 1', damping='1')
        ET.SubElement(root.find('actuator'), 'position', name=f'right_stage_{axis}_servo', joint=f'right_stage_{axis}',
                      kp='8000', kv='200', ctrlrange='-1 1', forcerange='-200 200')
    forearm = right.find('worldbody/body'); mount.append(forearm)
    for geom in forearm.iter('geom'):
        if geom.get('class') == 'strum_plastic_collision':
            geom.set('contype', '4'); geom.set('conaffinity', '1')
    for geom in root.findall(".//body[@name='ukulele']/geom"):
        if geom.get('name', '').startswith('string_'):
            geom.set('contype', '2'); geom.set('conaffinity', '1')
    old = root.find('keyframe')
    qleft = np.fromstring(old.find('key').get('qpos'), sep=' ')
    cleft = np.fromstring(old.find('key').get('ctrl'), sep=' ')
    root.remove(old)
    m = compile_root(root, Path(repo) / 'scene.xml'); d = mujoco.MjData(m)
    d.qpos[:len(qleft)] = qleft
    for f in ['MF', 'RF', 'LF']:
        for suffix, value in [('J3', 1.2), ('J2', .7), ('J1', .7)]:
            j = m.joint(f'strum_rh_{f}{suffix}').id; d.qpos[m.jnt_qposadr[j]] = value
    mujoco.mj_forward(m, d)
    distal = m.body('strum_rh_ffdistal').id; bid = m.body('strum_rh_forearm').id

    # --- changed: orientation from finger direction and back of the hand --
    R_arm = d.xmat[bid].reshape(3, 3).copy()
    to_local = lambda v: R_arm.T @ v
    prox = m.body('strum_rh_ffproximal').id
    finger = to_local(d.xpos[distal] - d.xpos[prox])          # straight index finger
    j3 = m.jnt_qposadr[m.joint('strum_rh_FFJ3').id]
    tip0 = d.xpos[distal].copy()
    d.qpos[j3] += .3; mujoco.mj_kinematics(m, d)
    palm = to_local(d.xpos[distal] - tip0)                     # curling moves the tip toward the palm
    d.qpos[j3] -= .3; mujoco.mj_forward(m, d)
    back = -(palm - (palm @ unit(finger)) * unit(finger))
    U = d.xmat[m.body('ukulele').id].reshape(3, 3)
    a = np.deg2rad(angle_deg)
    finger_target = np.cos(a) * U[:, 1] - np.sin(a) * U[:, 2]
    back_target = U[:, 2]
    rot = frame(finger_target, back_target) @ frame(finger, back).T
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, rot.reshape(9))
    m.body_quat[bid] = quat; mujoco.mj_forward(m, d)
    # Strum point: the index fingertip point nearest the strings (lowest along the ukulele's z).
    gid = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == distal and m.geom_contype[i])
    mid = m.geom_dataid[gid]
    v = np.array(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], dtype=float)
    w = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    tip = w[np.argmin(w @ U[:, 2])]

    # --- unchanged: place the tip, write pose, site and keyframe ------------
    local = d.xmat[distal].reshape(3, 3).T @ (tip - d.xpos[distal])
    initial = np.array([.29, -.1425, .18]); m.body_pos[bid] += initial - tip; mujoco.mj_forward(m, d)
    forearm.set('pos', ' '.join(map(str, m.body_pos[bid]))); forearm.set('quat', ' '.join(map(str, quat)))
    ET.SubElement(forearm.find(".//body[@name='strum_rh_ffdistal']"), 'site', name='right_strum_tip',
                  pos=' '.join(map(str, local)), size='.0015', rgba='1 .3 .1 1')
    ctrl = d.actuator_length.copy(); ctrl[:len(cleft)] = cleft
    key = ET.SubElement(root, 'keyframe')
    ET.SubElement(key, 'key', name='candidate', qpos=' '.join(map(str, d.qpos)), ctrl=' '.join(map(str, ctrl)))

    # How far every other right-hand part stays above the strum point (along the ukulele's z).
    low = np.inf
    for g in range(m.ngeom):
        name = m.body(int(m.geom_bodyid[g])).name
        if not name.startswith('strum_rh_') or g == gid or not m.geom_contype[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mm = m.geom_dataid[g]
            vv = np.array(m.mesh_vert[m.mesh_vertadr[mm]:m.mesh_vertadr[mm] + m.mesh_vertnum[mm]], dtype=float)
            ww = vv @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            low = min(low, float((ww @ U[:, 2]).min() - initial @ U[:, 2]))
        else:
            low = min(low, float(d.geom_xpos[g] @ U[:, 2] - m.geom_rbound[g] - initial @ U[:, 2]))
    message = (f'Right hand at {angle_deg:.0f} deg: other parts stay {low * 1000:.1f} mm above the strum point '
               '(negative means they reach lower than the fingertip).')
    print(message, flush=True)
    if report is not None:
        report['clearance_mm'] = low * 1000
    return initial


def install(angle_deg=ANGLE):
    bimanual.add_right_hand = functools.partial(add_right_hand_angled, angle_deg=angle_deg)
