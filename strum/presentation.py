"""Presentation-only geometry and lighting for the parameterized ukulele."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def visual(parent,name,kind,**attrs):
    return ET.SubElement(parent,'geom',name=name,type=kind,contype='0',conaffinity='0',group='2',mass='0',**attrs)


def decorate(root, p, repo):
    """Decorative geometry only; existing instrument collisions are retained."""
    uke=root.find(".//body[@name='ukulele']")
    asset=root.find('asset')
    for name in ['body_proxy','neck']:
        uke.find(f"geom[@name='{name}']").set('rgba','0 0 0 0')
    ET.SubElement(asset, 'texture', name='presentation_sky', type='skybox', builtin='gradient', rgb1='.12 .17 .23', rgb2='.38 .43 .48', width='512', height='3072')
    texture=Path(repo)/'scripts/visual_assets/koa_grain.png'
    if not texture.is_file():
        raise FileNotFoundError(f'Missing existing wood texture: {texture}')
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
    half=np.vstack([bezier(points) for points in pieces]+[np.array([[.401,0]])])
    half[:,0] *= p.scale_length/.38
    outline=np.vstack([half,half[-2:0:-1]*[1,-1]])
    def mesh(name,points,bottom,top):
        n=len(points);vertices=[(x,y,z) for z in [bottom,top] for x,y in points]
        # A center within the waist's visibility region makes a fan valid.
        center=(.260*p.scale_length/.38,0)
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
    for i,(edge_p,q) in enumerate(zip(outline,np.roll(outline,-1,axis=0))):
        visual(uke,f'binding_{i}','capsule',fromto=f'{edge_p[0]} {edge_p[1]} -.0006 {q[0]} {q[1]} -.0006',
               size='.00065',rgba='.86 .74 .51 1')
    for name,r,z,color in [('rosette_outer',.025,.0001,'.24 .12 .05 1'),
                            ('rosette_light',.0235,.0003,'.86 .76 .53 1'),
                            ('rosette_inner',.022,.0005,'.25 .12 .045 1'),
                            ('soundhole',.020,.0007,'.012 .009 .007 1')]:
        visual(uke,name,'cylinder',pos=f'{.275*p.scale_length/.38} 0 {z}',size=f'{r} .00012',rgba=color)
    visual(uke,'fretboard','box',pos=f'{p.neck_length/2} 0 -.002',size=f'{p.neck_length/2} {p.neck_width/2} .002',rgba='.075 .042 .027 1')
    visual(uke,'neck_wood','box',pos=f'{p.neck_length/2} 0 {-p.neck_depth/2}',size=f'{p.neck_length/2} {p.neck_width/2-.0005} {p.neck_depth/2-.0005}',rgba='.30 .135 .06 1')
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
    visual(uke,'nut_visual','box',pos=f'0 0 {(p.nut_height-p.string_radius)/2}',size=f'.001 {p.neck_width/2} {(p.nut_height-p.string_radius)/2}',rgba='.91 .86 .71 1')
    visual(uke,'bridge','box',pos=f'{p.scale_length-.003} 0 .0006',size=f'.009 {p.saddle_spread/2+.008} .0012',rgba='.105 .055 .028 1')
    visual(uke,'saddle','box',pos=f'{p.scale_length} 0 {(p.saddle_height-p.string_radius)/2}',size=f'.001 {p.saddle_spread/2+.002} {(p.saddle_height-p.string_radius)/2}',rgba='.92 .87 .72 1')
    for i,(x,y) in enumerate([(-.018,-.014),(-.045,-.013),(-.018,.014),(-.045,.013)]):
        visual(uke,f'tuner_washer_{i}','cylinder',pos=f'{x} {y} -.0005',size='.004 .0007',rgba='.67 .69 .68 1')
        visual(uke,f'tuner_post_{i}','cylinder',pos=f'{x} {y} .003',size='.002 .004',rgba='.74 .75 .71 1')
        visual(uke,f'tuner_button_{i}','ellipsoid',pos=f'{x} {np.sign(y)*.025} -.006',size='.004 .005 .0025',rgba='.85 .81 .68 1')
    for fret in [3,5,7,10]:
        x=p.scale_length*(1-2**(-(fret-.5)/12))
        if x<p.neck_length:visual(uke,f'inlay_{fret}','cylinder',pos=f'{x} 0 .00015',size='.0015 .00015',rgba='.84 .83 .72 1')
    world=root.find('worldbody')
    for light in list(world.findall('light')):world.remove(light)
    for pos,direction,diffuse in [('.1 -.4 1','0 .3 -1','.65 .61 .55'),('.5 .3 .8','-.2 -.3 -1','.4 .45 .5')]:
        ET.SubElement(world,'light',pos=pos,dir=direction,diffuse=diffuse,ambient='.20 .20 .20',castshadow='false')
    world.find("geom[@name='ground']").set('rgba','.32 .36 .39 1')


    # Hide diagnostic target markers in presentation mode only.
    for site in root.findall('.//site'):
        site.set('rgba', '0 0 0 0')
    visual_settings = root.find('visual')
    if visual_settings is None:
        visual_settings = ET.SubElement(root, 'visual')
    headlight = visual_settings.find('headlight')
    if headlight is None:
        headlight = ET.SubElement(visual_settings, 'headlight')
    headlight.set('ambient', '.3 .3 .3')
    headlight.set('diffuse', '.65 .65 .65')
    headlight.set('specular', '.15 .15 .15')
    for mat in root.findall('asset/material'):
        if mat.get('name') == 'gray':
            mat.set('rgba', '.30 .34 .38 1')
    for name in ['ground']:
        geom = world.find(f"geom[@name='{name}']")
        geom.set('rgba', '.32 .36 .39 1')


def set_camera(camera, preset='overview'):
    if preset == 'fretting':
        camera.lookat[:] = [.06, -.12, .19]
        camera.distance = .48
        camera.azimuth = 125
        camera.elevation = -25
    else:
        camera.lookat[:] = [.16, -.10, .28]
        camera.distance = 1.05
        camera.azimuth = 120
        camera.elevation = -35
