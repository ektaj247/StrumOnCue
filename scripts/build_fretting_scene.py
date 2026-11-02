"""Build a geometry-only ukulele/left-hand scene. No learning or IK yet."""
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco


def numbers(values):
    return ' '.join(f'{x:.8g}' for x in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--chord', choices=['C', 'G', 'Am', 'F'], default='C')
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = (args.output or repo / 'outputs' / 'fretting-scene').resolve()
    hand_dir = repo / 'external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls'
    source = hand_dir / 'left_hand.xml'
    manifest = json.loads((repo / 'manifests/benchmark_v1.json').read_text())
    assert manifest['string_order'] == ['G', 'C', 'E', 'A']
    out.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set('model', 'strum_geometry_v0')
    compiler = root.find('compiler')
    compiler.set('meshdir', os.path.relpath(hand_dir / 'assets', out))
    option = root.find('option')
    option.set('timestep', '0.002')
    world = root.find('worldbody')
    # Keep the upstream hand orientation. This is an inspection pose, not a
    # validated fretting placement. All hand joint positions remain neutral.
    world.find("body[@name='lh_forearm']").set('pos', '0 0 0.22')
    ET.SubElement(world, 'light', pos='0.2 -0.3 1', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='1 1 .05',
                  pos='0 0 0', rgba='.16 .19 .23 1')
    # Geometry prototype: 380 mm scale, 35 mm width, 9 mm string spacing.
    # The 200 mm neck displays five frets; strings continue to the bridge.
    scale = .380
    neck_y = -.12
    surface_z = .12
    radius = .0006
    fret_x = [scale * (1 - 2 ** (-n / 12)) for n in range(6)]
    neck = ET.SubElement(world, 'body', name='ukulele', pos=f'0 {neck_y} {surface_z}')
    ET.SubElement(neck, 'geom', name='neck', type='box', pos='.10 0 -.009',
                  size='.10 .0175 .009', rgba='.36 .19 .09 1')
    ET.SubElement(neck, 'geom', name='body_proxy', type='ellipsoid',
                  pos='.29 0 -.035', size='.105 .075 .035', rgba='.62 .34 .14 1')
    for n, x in enumerate(fret_x):
        ET.SubElement(neck, 'geom', name=f'fret_visual_{n}', type='capsule',
                      fromto=numbers([x, -.0175, .0008, x, .0175, .0008]),
                      size='.0006', rgba='.65 .65 .65 1', contype='0', conaffinity='0')
    targets = {}
    frets = manifest['chord_frets'][args.chord]
    for idx, name in enumerate(manifest['string_order']):
        y = (idx - 1.5) * .009
        ET.SubElement(neck, 'geom', name=f'string_{name}', type='capsule',
                      fromto=numbers([0, y, .002, scale, y, .002]),
                      size=str(radius), rgba='.88 .88 .77 1', friction='.7 .01 .001')
        for n in range(1, 6):
            # Target is halfway inside its fret interval, above string surface.
            p = [(fret_x[n - 1] + fret_x[n]) / 2, y, .002 + radius]
            selected = frets[idx] == n
            ET.SubElement(neck, 'site', name=f'target_{name}_{n}', pos=numbers(p),
                          size='.0025' if selected else '.0008',
                          rgba='1 .2 .1 1' if selected else '.2 .7 .9 .35')
            if selected:
                targets[name] = {'fret': n, 'site': f'target_{name}_{n}'}
    ET.SubElement(root, 'statistic', center='.20 -.06 .17', extent='.6')
    xml = out / 'scene.xml'
    ET.indent(tree, space='  ')
    tree.write(xml, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for value in targets.values():
        value['world_position_m'] = data.site(value['site']).xpos.tolist()
    report = {
        'status': 'geometry_only_not_reachability_validated',
        'chord': args.chord,
        'mujoco_version': mujoco.__version__,
        'scale_length_m': scale,
        'fret_positions_m': fret_x,
        'pressed_targets': targets,
        'open_strings': [s for s, f in zip(manifest['string_order'], frets) if f == 0],
        'nq': model.nq, 'nv': model.nv, 'actuators': model.nu,
        'joints': [model.joint(i).name for i in range(model.njnt)],
        'actuator_names': [model.actuator(i).name for i in range(model.nu)],
        'initial_contacts': data.ncon,
    }
    (out / 'inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    assert len(targets) == sum(f > 0 for f in frets)
    assert all(fret_x[i] < fret_x[i+1] for i in range(5))
    print(f'PASS: scene compiles; {model.nq} positions, {model.nu} actuators')
    print(f'Chord {args.chord}: {len(targets)} pressed targets; open strings {report["open_strings"]}')
    print(f'Scene: {xml}\nReport: {out / "inspection.json"}')
    print('No IK, contact success, dynamic stability, or learned control claimed.')
    if args.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [.20, -.06, .17]
            viewer.cam.distance = .65
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -30
            while viewer.is_running():
                # Deliberately inspect kinematics; no uncommanded physics drop.
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(.02)


if __name__ == '__main__':
    main()
