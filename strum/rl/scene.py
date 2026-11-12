"""mjlab entities for the fretting task: fixed-mount left hand and ukulele.

The hand is the upstream left_hand.xml with the same two additions the
scripted calibration uses: the J1 = J2 distal coupling and a pad marker on
each fingertip. The ukulele collision geometry comes from strum.ukulele.
"""
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from ..ukulele import STRING_NAMES, fret_positions, profile, string_y, string_z
from .chords import FINGERS, HAND_XML, load_table


def hand_spec(table):
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    assets = HAND_XML.parent / 'assets'
    spec.assets = {f'{spec.meshdir}/{f.name}': f.read_bytes() for f in assets.iterdir() if f.is_file()}
    # mjlab places a fixed-base entity through its own base body, so the mount
    # pose from the chord table replaces the forearm's pose in the XML.
    spec.body('lh_forearm').pos = [0, 0, 0]
    spec.body('lh_forearm').quat = [1, 0, 0, 0]
    counts = {}
    for geom in spec.geoms:
        if geom.name:
            continue
        kind = 'visual' if geom.contype == 0 and geom.conaffinity == 0 else 'collision'
        key = f'{geom.parent.name}_{kind}'
        index = counts.get(key, 0)
        counts[key] = index + 1
        geom.name = f'{key}_{index}' if index else key
    for finger in FINGERS:
        equality = spec.add_equality()
        equality.name = f'lh_{finger}_distal_coupling'
        equality.type = mujoco.mjtEq.mjEQ_JOINT
        equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
        equality.name1, equality.name2 = f'lh_{finger}J1', f'lh_{finger}J2'
        equality.data[:5] = [0, 1, 0, 0, 0]
        spec.body(f'lh_{finger.lower()}distal').add_site(
            name=f'pad_{finger.lower()}', pos=table['pad_offset'], size=[.002, .002, .002], rgba=[0, 1, .2, 1])
    return spec


def ukulele_spec(table):
    p = profile(table['profile'])
    if p.to_dict() != table['parameters']:
        raise ValueError('Geometry changed since the chord table was exported')
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='ukulele')
    body.add_geom(name='neck', type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[p.neck_length / 2, 0, -p.neck_depth / 2],
                  size=[p.neck_length / 2, p.neck_width / 2, p.neck_depth / 2], rgba=[.36, .19, .09, 1])
    body.add_geom(name='body_proxy', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=[.29, 0, -.035], size=[.105, .075, .035], rgba=[.62, .34, .14, 1])
    for i, name in enumerate(STRING_NAMES):
        body.add_geom(name='string_' + name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, string_y(p, i, 0), string_z(p, 0),
                              p.scale_length, string_y(p, i, p.scale_length), string_z(p, p.scale_length)],
                      size=[p.string_radius, 0, 0], friction=[.7, .01, .001], rgba=[.88, .88, .77, 1])
    for n, x in enumerate([0.] + fret_positions(p.scale_length, p.n_frets)):
        body.add_geom(name=f'fret_visual_{n}', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[x, -p.neck_width / 2, p.fret_radius, x, p.neck_width / 2, p.fret_radius],
                      size=[p.fret_radius, 0, 0], contype=0, conaffinity=0, rgba=[.65, .65, .65, 1])
    return spec


def hand_cfg(table=None):
    table = table or load_table()
    joints = dict(zip(table['joint_names'], table['rest']['qpos']))
    joint_targets = tuple(t for t in table['actuator_targets'] if not t.endswith('J0'))
    tendon_targets = tuple(t for t in table['actuator_targets'] if t.endswith('J0'))
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=tuple(table['mount']['pos']), rot=tuple(table['mount']['quat']),
                                            joint_pos=joints, joint_vel={'.*': 0.}),
        spec_fn=lambda: hand_spec(table),
        articulation=EntityArticulationInfoCfg(actuators=(
            XmlActuatorCfg(target_names_expr=joint_targets),
            XmlActuatorCfg(target_names_expr=tendon_targets, transmission_type=TransmissionType.TENDON))))


def ukulele_cfg(table=None):
    table = table or load_table()
    return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=tuple(table['ukulele_pos'])),
                     spec_fn=lambda: ukulele_spec(table))
