"""Visual ukulele upgrade and actuator-driven right-hand mounting stage."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def visual(parent,name,kind,**attrs):
    return ET.SubElement(parent,'geom',name=name,type=kind,contype='0',conaffinity='0',group='2',**attrs)


def add_ukulele_visuals(root):
    uke=root.find(".//body[@name='ukulele']")
    # Preserve original collision geometry, hide only its rendering.
    for name in ['body_proxy','neck']:
        uke.find(f"geom[@name='{name}']").set('rgba','0 0 0 0')
    # Procedural extruded outline with upper/lower bouts and a waist.
    xs=np.array([.185,.195,.215,.235,.255,.275,.300,.325,.350,.375,.395])
    widths=np.array([.012,.036,.052,.048,.036,.046,.065,.073,.065,.042,.008])
    outline=[(float(x),float(y)) for x,y in zip(xs,widths)]+[(float(x),float(-y)) for x,y in zip(xs[::-1],widths[::-1])]
    n=len(outline);vertices=[(x,y,z) for z in [-.055,-.001] for x,y in outline]
    vertices += [(float(xs.mean()),0,-.055),(float(xs.mean()),0,-.001)]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces.extend([(2*n,j,i),(2*n+1,n+i,n+j),(i,j,n+j),(i,n+j,n+i)])
    asset=root.find('asset')
    ET.SubElement(asset,'mesh',name='uke_visual_body_mesh',vertex=' '.join(str(v) for xyz in vertices for v in xyz),face=' '.join(str(v) for face in faces for v in face))
    visual(uke,'uke_visual_body','mesh',mesh='uke_visual_body_mesh',rgba='.64 .34 .12 1')
    # Faux soundhole/rosette are appearance only; no acoustic cavity claim.
    visual(uke,'rosette','cylinder',pos='.275 0 -.0005',size='.024 .0004',rgba='.86 .69 .38 1')
    visual(uke,'soundhole','cylinder',pos='.275 0 .0001',size='.020 .0003',rgba='.025 .017 .012 1')
    visual(uke,'fretboard','box',pos='.10 0 -.002',size='.10 .0175 .002',rgba='.105 .055 .03 1')
    visual(uke,'headstock','box',pos='-.031 0 -.006',size='.030 .024 .007',rgba='.36 .15 .055 1')
    visual(uke,'nut_visual','box',pos='0 0 .001',size='.001 .018 .001',rgba='.90 .85 .68 1')
    visual(uke,'bridge','box',pos='.377 0 -.001',size='.009 .030 .001',rgba='.12 .06 .025 1')
    visual(uke,'saddle','box',pos='.380 0 .001',size='.001 .018 .001',rgba='.9 .85 .68 1')
    for i,(x,y) in enumerate([(-.015,-.017),(-.045,-.017),(-.015,.017),(-.045,.017)]):
        visual(uke,f'tuner_{i}','cylinder',pos=f'{x} {y} .004',size='.003 .006',rgba='.7 .73 .76 1')
        visual(uke,f'tuner_button_{i}','ellipsoid',pos=f'{x} {np.sign(y)*.028} -.004',size='.005 .006 .003',rgba='.85 .85 .8 1')
    for fret in [3,5,7,10]:
        x=.38*(1-2**(-(fret-.5)/12))
        if x<.195:visual(uke,f'inlay_{fret}','cylinder',pos=f'{x} 0 .00015',size='.0018 .00015',rgba='.9 .85 .7 1')
    # Brighter studio lighting improves inspection of black hands.
    ET.SubElement(root.find('worldbody'),'light',pos='.2 -.5 .8',dir='0 .4 -1',diffuse='.8 .8 .8',ambient='.3 .3 .3',castshadow='false')


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
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
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
