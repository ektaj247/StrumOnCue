"""Build a parameterized geometry inspection scene."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strum.ukulele import (
    CHORDS,
    STRING_NAMES,
    fingertip_target,
    fret_positions,
    profile,
    string_y,
    string_z,
)


def nums(values):
    return " ".join(f"{value:.12g}" for value in values)


def build(repo, out, name, chord):
    repo, out = repo.resolve(), out.resolve()
    p = profile(name)

    # Refuse to overwrite an earlier experiment.
    out.mkdir(parents=True, exist_ok=False)

    hand = (
        repo
        / "external/mjlab_hand/src/mjlab_hand/robot/shadow_hand/xmls"
    )

    root = ET.parse(hand / "left_hand.xml").getroot()
    root.set("model", f"strum_geometry_{name}")
    root.find("compiler").set(
        "meshdir",
        os.path.relpath(hand / "assets", out),
    )
    root.find("option").set("timestep", ".002")

    world = root.find("worldbody")
    world.find("body[@name='lh_forearm']").set("pos", "0 0 .22")

    ET.SubElement(
        world, "light", pos=".2 -.3 1", dir="0 0 -1"
    )
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="1 1 .05",
        pos="0 0 0",
        rgba=".16 .19 .23 1",
    )

    uke = ET.SubElement(
        world,
        "body",
        name="ukulele",
        pos="0 -.12 .12",
    )

    ET.SubElement(
        uke,
        "geom",
        name="neck",
        type="box",
        pos=nums([p.neck_length / 2, 0, -p.neck_depth / 2]),
        size=nums([
            p.neck_length / 2,
            p.neck_width / 2,
            p.neck_depth / 2,
        ]),
        rgba=".36 .19 .09 1",
    )

    # Retain the existing body proxy while changing playable geometry.
    # The polished visual shell belongs to the existing demo.
    ET.SubElement(
        uke,
        "geom",
        name="body_proxy",
        type="ellipsoid",
        pos=".29 0 -.035",
        size=".105 .075 .035",
        rgba=".62 .34 .14 1",
    )

    edges = [0.0] + fret_positions(p.scale_length, p.n_frets)

    for n, x in enumerate(edges):
        fret_z = p.fret_top - p.fret_radius
        ET.SubElement(
            uke,
            "geom",
            name=f"fret_visual_{n}",
            type="capsule",
            fromto=nums([
                x, -p.neck_width / 2, fret_z,
                x, p.neck_width / 2, fret_z,
            ]),
            size=str(p.fret_radius),
            rgba=".65 .65 .65 1",
            contype="0",
            conaffinity="0",
        )

    targets = {}

    for i, string in enumerate(STRING_NAMES):
        ET.SubElement(
            uke,
            "geom",
            name=f"string_{string}",
            type="capsule",
            fromto=nums([
                0,
                string_y(p, i, 0),
                string_z(p, 0),
                p.scale_length,
                string_y(p, i, p.scale_length),
                string_z(p, p.scale_length),
            ]),
            size=str(p.string_radius),
            rgba=".88 .88 .77 1",
            friction=".7 .01 .001",
        )

        for n in range(1, p.n_frets + 1):
            selected = CHORDS[chord][i] == n
            site = f"target_{string}_{n}"

            ET.SubElement(
                uke,
                "site",
                name=site,
                pos=nums(fingertip_target(p, n, i)),
                size=".0025" if selected else ".0008",
                rgba="1 .2 .1 1" if selected else ".2 .7 .9 .35",
            )

            if selected:
                targets[string] = {"fret": n, "site": site}

    ET.SubElement(
        root, "statistic", center=".20 -.06 .17", extent=".6"
    )
    ET.indent(root, space="  ")

    scene = out / "scene.xml"
    ET.ElementTree(root).write(scene, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for target in targets.values():
        target["world_position_m"] = (
            data.site(target["site"]).xpos.tolist()
        )

    report = {
        "status": "geometry_only_not_control_validated",
        "profile": name,
        "parameters": p.to_dict(),
        "chord": chord,
        "pressed_targets": targets,
        "nq": model.nq,
        "nu": model.nu,
        "mujoco_version": mujoco.__version__,
        "scope": (
            "Five-fret task; rigid strings; visual-only frets; "
            "original body proxy. No learned control or "
            "realistic string mechanics."
        ),
    }

    (out / "geometry.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(
        f"COMPILED {name}/{chord}: "
        f"{model.nq} positions, {model.nu} actuators — {scene}"
    )
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--profile",
        choices=["baseline", "ek"],
        required=True,
    )
    parser.add_argument("--chord", choices=list(CHORDS), default="C")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    model, data = build(
        args.repo,
        args.output,
        args.profile,
        args.chord,
    )

    if args.view:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as view:
            view.cam.lookat[:] = [0.15, -0.12, 0.16]
            view.cam.distance = 0.7
            view.cam.azimuth = 120
            view.cam.elevation = -30

            # Static inspection: no physics or controller is running.
            while view.is_running():
                view.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()