"""Composition of the existing three-axis strumming stage with the left-hand scene."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .fretting import compile_root


def compose(scene, repo, p=None):
    scene = scene.resolve()
    root = ET.parse(scene).getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((scene.parent / compiler.get('meshdir','')).resolve()))
    if p is not None:
        from .presentation import decorate
        decorate(root, p, repo)
    add_right_hand(root, repo)
    m = compile_root(root, scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    return m,d,root


def add_right_hand(root,repo):
    root.find('option').set('integrator','implicitfast')
    source=Path(repo)/'external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls/right_hand.xml'
    right=ET.parse(source).getroot()
    # Unique namespace for shared mesh/material/default names.
    for mesh in right.findall('asset/mesh'):
        if not mesh.get('name'):mesh.set('name',Path(mesh.get('file')).stem)
    refs={'name','class','childclass','material','mesh','joint','tendon','body1','body2'}
    for e in right.iter():
        for k in refs:
            if k in e.attrib:e.set(k,'strum_'+e.get(k))
    for b in right.findall('.//body'):
        b.set('gravcomp','1')
        for i,g in enumerate(b.findall('geom')):
            if not g.get('name'):g.set('name',b.get('name')+f'_geom_{i}')
    for actuator in right.findall('actuator/position'):
        actuator.set('kp','20');actuator.set('kv','1');actuator.set('forcerange','-10 10')
    for section in ['default','asset','contact','tendon','actuator']:
        src=right.find(section)
        if src is not None:
            dest=root.find(section)
            if dest is None:dest=ET.SubElement(root,section)
            for child in list(src):dest.append(child)
    # 3 translational position-actuated stage DOFs, not a full robot arm.
    mount=ET.SubElement(root.find('worldbody'),'body',name='right_stage')
    for axis,vector in zip('xyz',['1 0 0','0 1 0','0 0 1']):
        ET.SubElement(mount,'joint',name=f'right_stage_{axis}',type='slide',axis=vector,range='-1 1',damping='1')
        ET.SubElement(root.find('actuator'),'position',name=f'right_stage_{axis}_servo',joint=f'right_stage_{axis}',kp='8000',kv='200',ctrlrange='-1 1',forcerange='-200 200')
    forearm=right.find('worldbody/body');mount.append(forearm)
    # Keep right-hand collision with neck/left hand, disable right/string pairs.
    for geom in forearm.iter('geom'):
        if geom.get('class')=='strum_plastic_collision':
            geom.set('contype','4');geom.set('conaffinity','1')
    for geom in root.findall(".//body[@name='ukulele']/geom"):
        if geom.get('name','').startswith('string_'):
            geom.set('contype','2');geom.set('conaffinity','1')
    # Remove old key while composing; rebuild with all DOFs and actuators.
    old=root.find('keyframe');qleft=np.fromstring(old.find('key').get('qpos'),sep=' ');cleft=np.fromstring(old.find('key').get('ctrl'),sep=' ')
    root.remove(old)
    m=compile_root(root, Path(repo)/"scene.xml");d=mujoco.MjData(m)
    d.qpos[:len(qleft)]=qleft
    for f in ['MF','RF','LF']:
        for suffix,value in [('J3',1.2),('J2',.7),('J1',.7)]:
            j=m.joint(f'strum_rh_{f}{suffix}').id;d.qpos[m.jnt_qposadr[j]]=value
    mujoco.mj_forward(m,d)
    distal=m.body('strum_rh_ffdistal').id;bid=m.body('strum_rh_forearm').id
    desired=np.diag([-1.,1.,-1.])
    rot=desired @ d.xmat[distal].reshape(3,3).T @ d.xmat[bid].reshape(3,3)
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rot.reshape(9));m.body_quat[bid]=quat;mujoco.mj_forward(m,d)
    gid=next(i for i in range(m.ngeom) if m.geom_bodyid[i]==distal and m.geom_contype[i])
    mid=m.geom_dataid[gid];v=np.array(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]],dtype=float)
    w=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];tip=w[np.argmin(w[:,2])]
    local=d.xmat[distal].reshape(3,3).T@(tip-d.xpos[distal])
    initial=np.array([.29,-.1425,.18]);m.body_pos[bid]+=initial-tip;mujoco.mj_forward(m,d)
    forearm.set('pos',' '.join(map(str,m.body_pos[bid])));forearm.set('quat',' '.join(map(str,quat)))
    ET.SubElement(forearm.find(".//body[@name='strum_rh_ffdistal']"),'site',name='right_strum_tip',pos=' '.join(map(str,local)),size='.0015',rgba='1 .3 .1 1')
    ctrl=d.actuator_length.copy();ctrl[:len(cleft)]=cleft
    key=ET.SubElement(root,'keyframe');ET.SubElement(key,'key',name='candidate',qpos=' '.join(map(str,d.qpos)),ctrl=' '.join(map(str,ctrl)))
    return initial
