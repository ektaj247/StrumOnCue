"""Still pictures of fretting-hand poses, to see what check_hand_pose.py is trying.

No physics: the hand is posed and drawn. Writes PNGs and a previews.md page
(open it with VS Code's Markdown preview, Ctrl+Shift+V) to
outputs/hand-pose/previews/. Changes no project files.

    .venv/Scripts/python experiments/hand_pose/render_poses.py
    .venv/Scripts/python experiments/hand_pose/render_poses.py --run outputs/hand-pose/<run>

--run adds the best pose saved by a finished check_hand_pose.py run.
"""
import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_hand_pose import ROOT, Checker, make_base_scene  # noqa: E402

OUT = ROOT / 'outputs/hand-pose/previews'
WIDTH, HEIGHT = 480, 360
# The ukulele lies flat in this scene (strings face up). Each picture is
# turned afterwards (rot = quarter turns) so it looks as if the ukulele were
# held upright with the G string on top, as in the presentation.
VIEWS = {
    # Facing the fretboard; neck points right, body on the left.
    'front': (dict(distance=.55, azimuth=90., elevation=-70.), 2),
    # Looking down the neck from the headstock; fretboard faces left, G on top.
    'end': (dict(distance=.45, azimuth=0., elevation=-3.), 1),
}


def write_png(path, rgb):
    """Minimal PNG writer (8-bit RGB), so no image library is needed."""
    height, width, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    path.write_bytes(png)


class Painter:
    def __init__(self, checker):
        self.c = checker
        self.renderer = mujoco.Renderer(checker.m, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = checker.u_pos + checker.u_rot @ [.06, .03, 0.]

    def draw(self, name, pos, quat, q, folder=OUT):
        c = self.c
        c.m.body_pos[c.forearm], c.m.body_quat[c.forearm] = pos, quat
        c.d.qpos[:] = q
        mujoco.mj_forward(c.m, c.d)
        files = []
        for view, (settings, turns) in VIEWS.items():
            for key, value in settings.items():
                setattr(self.camera, key, value)
            self.renderer.update_scene(c.d, self.camera)
            path = folder / f'{name}_{view}.png'
            write_png(path, np.ascontiguousarray(np.rot90(self.renderer.render(), turns)))
            files.append(path.name)
        return files


def misses(result):
    return ', '.join(f'{ch} {r["miss_mm"]:.1f} mm' for ch, r in result.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, help='a finished check_hand_pose.py run folder to include')
    a = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scene = OUT / 'base/scene.xml'
    if not scene.exists():
        make_base_scene(OUT)
    checker = Checker(scene)
    painter = Painter(checker)
    page = ['# Fretting-hand poses', '',
            'Pictures are turned so the ukulele looks held upright with the G string on top. '
            '**Front**: facing the fretboard, neck pointing right. '
            '**End**: looking down the neck from the headstock, fretboard facing left. '
            'A human-like hand would sit below the neck (bottom of the picture), palm up, fingers '
            'curling up and over onto the fretboard. Misses are how far the worst fingertip is '
            'from its target; pass needs every miss under 1 mm.', '']

    def section(title, text, name, pos, quat, q):
        files = painter.draw(name, pos, quat, q)
        page.extend([f'## {title}', '', text, ''] + [f'![{f}]({f})' for f in files] + [''])
        print('drew', name, flush=True)

    # 1. Today's mount, every chord.
    pos, quat = checker.original
    today = checker.evaluate(pos, quat, ik='planner')
    for chord in ('C', 'G', 'Am', 'F'):
        section(f'Today\'s mount: {chord}', f'What passing looks like. Miss: {today[chord]["miss_mm"]:.2f} mm.',
                f'1_today_{chord}', pos, quat, today[chord]['q'])

    # 2. The kind of pose being tried: level forearm, palm up, under the neck.
    directions = {0: 'from behind the neck, toward the camera side (like a player)',
                  90: 'along the neck, toward the body', 180: 'from the front, toward the back',
                  -90: 'along the neck, toward the headstock'}
    for yaw, words in directions.items():
        x = np.array([yaw, 0., 0., 40., 60., -10.])
        pos, quat = checker.pose(x)
        result = checker.evaluate(pos, quat, ik='hook')
        section(f'Example pose, yaw {yaw:+d}°: forearm points {words}',
                'Knuckles 60 mm below the neck centre, 40 mm from the nut, 10 mm behind the fretboard. '
                'An example of one candidate, not a searched result. Fingers posed for G as close as IK '
                f'gets. Misses: {misses(result)}.',
                f'2_yaw{yaw:+d}_G', pos, quat, result['G']['q'])

    # 3. Best pose of the earlier quick run, if it is there.
    quick = ROOT / 'outputs/hand-pose/20261119-173543/best.json'
    if quick.exists():
        best = json.loads(quick.read_text())
        chords = best['chords']
        section('Best pose of the first quick run (G)',
                f'Yaw {best["yaw_deg"]:+.0f}°. Misses: '
                + ', '.join(f'{ch} {r["miss_mm"]:.1f} mm' for ch, r in chords.items()) + '.',
                '3_quick_best_G', np.array(best['forearm_pos']), np.array(best['forearm_quat']), chords['G']['q'])

    # 4. Best pose of a finished full run.
    if a.run:
        best = json.loads((a.run / 'best.json').read_text())
        pos, quat = np.array(best['forearm_pos']), np.array(best['forearm_quat'])
        for chord in ('C', 'G', 'Am', 'F'):
            r = best['chords']['hook'][chord]
            section(f'Best pose of {a.run.name}: {chord}',
                    f'Yaw {best["yaw_deg"]:+.0f}°, pitch {best["pitch_deg"]:+.0f}°, roll {best["roll_deg"]:+.0f}°. '
                    f'Miss {r["miss_mm"]:.2f} mm, penetration {r["pen_mm"]:.2f} mm ({r["part"] or "none"}).',
                    f'4_best_{chord}', pos, quat, r['q'])

    (OUT / 'previews.md').write_text('\n'.join(page) + '\n', encoding='utf-8')
    print('Open', OUT / 'previews.md', 'with Ctrl+Shift+V in VS Code.')


if __name__ == '__main__':
    main()
