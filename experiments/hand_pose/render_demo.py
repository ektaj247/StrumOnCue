"""Final pictures and video of the human-like scene.

Everything from this experiment, swapped in for the run only (nothing on disk in
strum/ changes): idle fingers make room (idle_space.py), the swinging strumming
hand (swing.py), the upright scene (upright.py) and the red concert look
(look.py). The left hand plays the chords calibrated in outputs/holds-human-4.

    .venv/Scripts/python experiments/hand_pose/render_demo.py stills
    .venv/Scripts/python experiments/hand_pose/render_demo.py video

Writes to outputs/demo/.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import idle_space  # noqa: E402  (puts the repo root on sys.path)
import look  # noqa: E402
import swing  # noqa: E402
import upright  # noqa: E402
from render_poses import write_png  # noqa: E402

ROOT = HERE.parents[1]
HOLDS = ROOT / 'outputs/holds-human-4'
OUT = ROOT / 'outputs/demo'
WIDTH, HEIGHT = 1280, 720
TILT = 8.


def install():
    import strum.fretting as fretting
    import strum.planner as planner
    fretting.Hand = planner.Hand = idle_space.IdleSpaceHand
    swing.install(TILT)
    upright.install()
    look.install()


def renderer_for(m):
    m.vis.global_.offwidth, m.vis.global_.offheight = WIDTH, HEIGHT
    m.site_rgba[:, 3] = 0          # hide marker points (for example the strum point) in pictures
    return mujoco.Renderer(m, HEIGHT, WIDTH)


def stills():
    from strum import bimanual
    from strum.ukulele import profile
    p = profile('ek')
    folder = OUT / 'stills'
    folder.mkdir(parents=True, exist_ok=True)
    # The ukulele alone (hands hidden), to compare with the reference photo.
    m, d, _ = bimanual.compose(HOLDS / 'C' / 'scene.xml', ROOT, p)
    for g in range(m.ngeom):
        if m.body(int(m.geom_bodyid[g])).name.startswith(('lh_', 'strum_rh_')):
            m.geom_matid[g] = -1
            m.geom_rgba[g] = 0
    renderer = renderer_for(m)
    camera = mujoco.MjvCamera()
    upright.front_camera(camera, m, d, distance=.95, offset=(.195, 0., 0.))
    renderer.update_scene(d, camera)
    write_png(folder / 'ukulele_only_front.png', renderer.render())
    print('wrote', (folder / 'ukulele_only_front.png').relative_to(ROOT), flush=True)
    renderer.close()
    shots =[(c, 1.15, 'downstroke') for c in ('C', 'G', 'Am', 'F')] + [('G', 2.01, 'upstroke')]
    for chord, when, label in shots:
        m, d, _ = bimanual.compose(HOLDS / chord / 'scene.xml', ROOT, p)
        strummer = swing.SwingStrummer(m, d, p, [(0, (f'hold_{chord}', 3.0, chord, chord))])
        while d.time < when:
            strummer.control(d)
            mujoco.mj_step(m, d)
            strummer.observe(d)
        renderer = renderer_for(m)
        camera = mujoco.MjvCamera()
        for view, distance in (('front', .8), ('close', .5)):
            upright.front_camera(camera, m, d, distance=distance,
                                 offset=(.17, .02, 0.) if view == 'front' else (.26, .02, 0.))
            renderer.update_scene(d, camera)
            path = folder / f'{chord}_{label}_{view}.png'
            write_png(path, renderer.render())
            print('wrote', path.relative_to(ROOT), flush=True)
        renderer.close()


def video(fps=30):
    """One C -> Am -> C cycle with strumming, recorded from the front camera."""
    import imageio.v2 as imageio
    import strum.fretting as fretting
    import strum.strumming as strumming
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'c_am_c_upright.mp4'
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=8)

    class Recording(swing.SwingStrummer):
        def __init__(self, m, d, p, schedule):
            super().__init__(m, d, p, schedule)
            self.renderer = renderer_for(m)
            self.camera = mujoco.MjvCamera()
            upright.front_camera(self.camera, m, d, distance=.8)
            self.next_frame = 0.
            self.frames = 0

        def observe(self, d):
            super().observe(d)
            if d.time >= self.next_frame:
                self.renderer.update_scene(d, self.camera)
                writer.append_data(self.renderer.render())
                self.next_frame += 1 / fps
                self.frames += 1

    strumming.Strummer = Recording
    run = OUT / 'video-run'
    sys.argv = ['strum.fretting', 'transition', '--run', str(HOLDS), '--cycles', '1', '--two-hands',
                '--polished', '--output', str(run)]
    try:
        fretting.main()
    finally:
        writer.close()
    print('wrote', path.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['stills', 'video'])
    a = parser.parse_args()
    install()
    stills() if a.what == 'stills' else video()


if __name__ == '__main__':
    main()
