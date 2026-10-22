"""Visual ukulele upgrade and actuator-driven right-hand mounting stage."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def visual(parent,name,kind,**attrs):
    return ET.SubElement(parent,'geom',name=name,type=kind,contype='0',conaffinity='0',group='2',mass='0',**attrs)


def add_ukulele_visuals(root):
    """Decorative geometry only; existing instrument collisions are retained."""
    uke=root.find(".//body[@name='ukulele']")
    asset=root.find('asset')
    for name in ['body_proxy','neck']:
        uke.find(f"geom[@name='{name}']").set('rgba','0 0 0 0')
    texture=Path(__file__).resolve().parent/'visual_assets/koa_grain.png'
    ET.SubElement(asset,'texture',name='koa_grain',type='2d',file=str(texture))
    ET.SubElement(asset,'material',name='koa_top',texture='koa_grain',texuniform='false',
                  specular='.18',shininess='.22',rgba='1 1 1 1')
    def bezier(points,n=16):
        p=np.asarray(points,dtype=float);t=np.linspace(0,1,n,endpoint=False)[:,None]
        return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]
    pieces=[[(.182,0),(.182,.028),(.201,.052),(.222,.052)],
            [(.222,.052),(.239,.052),(.244,.034),(.260,.034)],
            [(.260,.034),(.276,.034),(.285,.074),(.326,.074)],
            [(.326,.074),(.365,.074),(.401,.043),(.401,0)]]
    half=np.vstack([bezier(p) for p in pieces]+[np.array([[.401,0]])])
    outline=np.vstack([half,half[-2:0:-1]*[1,-1]])
    def mesh(name,points,bottom,top):
        n=len(points);vertices=[(x,y,z) for z in [bottom,top] for x,y in points]
        # A center within the waist's visibility region makes a fan valid.
        center=(.260,0)
        vertices += [(*center,bottom),(*center,top)]
        faces=[]
        for i in range(n):
            j=(i+1)%n
            faces += [(2*n,j,i),(2*n+1,n+i,n+j),(i,j,n+j),(i,n+j,n+i)]
        uv=[((v[0]-.182)/(.401-.182),(v[1]+.074)/.148) for v in vertices]
        ET.SubElement(asset,'mesh',name=name,
            vertex=' '.join(str(v) for xyz in vertices for v in xyz),
            face=' '.join(str(v) for f in faces for v in f),
            texcoord=' '.join(str(v) for xy in uv for v in xy))
        return name
    visual(uke,'body_sides','mesh',mesh=mesh('smooth_body',outline,-.050,-.0015),rgba='.25 .105 .046 1')
    visual(uke,'soundboard','mesh',mesh=mesh('smooth_top',outline,-.0015,-.0006),material='koa_top')
    for i,(p,q) in enumerate(zip(outline,np.roll(outline,-1,axis=0))):
        visual(uke,f'binding_{i}','capsule',fromto=f'{p[0]} {p[1]} -.0006 {q[0]} {q[1]} -.0006',
               size='.00065',rgba='.86 .74 .51 1')
    for name,r,z,color in [('rosette_outer',.025,.0001,'.24 .12 .05 1'),
                            ('rosette_light',.0235,.0003,'.86 .76 .53 1'),
                            ('rosette_inner',.022,.0005,'.25 .12 .045 1'),
                            ('soundhole',.020,.0007,'.012 .009 .007 1')]:
        visual(uke,name,'cylinder',pos=f'.275 0 {z}',size=f'{r} .00012',rgba=color)
    visual(uke,'fretboard','box',pos='.10 0 -.002',size='.10 .0175 .002',rgba='.075 .042 .027 1')
    visual(uke,'neck_wood','box',pos='.10 0 -.010',size='.10 .017 .007',rgba='.30 .135 .06 1')
    # Rounded, tapered headstock outline (visual only).
    head=[]
    for angle in np.linspace(0,2*np.pi,64,endpoint=False):
        head.append((-.033+.033*np.cos(angle),.022*np.sin(angle)))
    # Separate convex headstock mesh to avoid relying on the body fan center.
    n=len(head);verts=[(x,y,z) for z in [-.014,-.002] for x,y in head]
    faces=[]
    for i in range(1,n-1):faces += [(0,i+1,i),(n,n+i,n+i+1)]
    for i in range(n):
        j=(i+1)%n;faces += [(i,j,n+j),(i,n+j,n+i)]
    ET.SubElement(asset,'mesh',name='rounded_headstock',vertex=' '.join(str(v) for xyz in verts for v in xyz),face=' '.join(str(v) for f in faces for v in f))
    visual(uke,'headstock','mesh',mesh='rounded_headstock',rgba='.31 .14 .065 1')
    visual(uke,'nut_visual','box',pos='0 0 .001',size='.001 .018 .001',rgba='.91 .86 .71 1')
    visual(uke,'bridge','box',pos='.377 0 -.0006',size='.009 .030 .0012',rgba='.105 .055 .028 1')
    visual(uke,'saddle','box',pos='.380 0 .001',size='.001 .018 .001',rgba='.92 .87 .72 1')
    for i,(x,y) in enumerate([(-.018,-.014),(-.045,-.013),(-.018,.014),(-.045,.013)]):
        visual(uke,f'tuner_washer_{i}','cylinder',pos=f'{x} {y} -.0005',size='.004 .0007',rgba='.67 .69 .68 1')
        visual(uke,f'tuner_post_{i}','cylinder',pos=f'{x} {y} .003',size='.002 .004',rgba='.74 .75 .71 1')
        visual(uke,f'tuner_button_{i}','ellipsoid',pos=f'{x} {np.sign(y)*.025} -.006',size='.004 .005 .0025',rgba='.85 .81 .68 1')
    for fret in [3,5,7,10]:
        x=.38*(1-2**(-(fret-.5)/12))
        if x<.195:visual(uke,f'inlay_{fret}','cylinder',pos=f'{x} 0 .00015',size='.0015 .00015',rgba='.84 .83 .72 1')
    world=root.find('worldbody')
    for light in list(world.findall('light')):world.remove(light)
    for pos,direction,diffuse in [('.1 -.4 1','0 .3 -1','.65 .61 .55'),('.5 .3 .8','-.2 -.3 -1','.4 .45 .5')]:
        ET.SubElement(world,'light',pos=pos,dir=direction,diffuse=diffuse,ambient='.20 .20 .20',castshadow='false')
    world.find("geom[@name='ground']").set('rgba','.16 .185 .20 1')


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
