"""Check geometry math, scene compilation, and baseline equivalence."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strum.ukulele import (
    CHORDS,
    STRING_NAMES,
    classify_contact,
    fingertip_target,
    fret_positions,
    profile,
    string_y,
    string_z,
    swept_crossings,
)
from build_geometry_scene import build


def check_geometry_math():
    for name in ["baseline", "ek"]:
        p = profile(name)

        assert abs(
            fret_positions(p.scale_length, 12)[-1]
            - p.scale_length / 2
        ) < 1e-12

        for i, string in enumerate(STRING_NAMES):
            for fret in range(1, 6):
                point = fingertip_target(p, fret, i)
                assert classify_contact(p, point) == (string, fret)

                above = (point[0], point[1], point[2] + 0.02)
                assert classify_contact(p, above) is None

        x = 0.29
        z = string_z(p, x)

        down = swept_crossings(p, (x, -0.04, z), (x, 0.04, z))
        up = swept_crossings(p, (x, 0.04, z), (x, -0.04, z))

        assert [event["string"] for event in down] == [
            "G", "C", "E", "A"
        ]
        assert [event["string"] for event in up] == [
            "A", "E", "C", "G"
        ]
        assert all(event["direction"] == "D" for event in down)
        assert all(event["direction"] == "U" for event in up)

        assert not swept_crossings(
            p, (x, -0.04, z + 0.02), (x, 0.04, z + 0.02)
        )

        diagonal = swept_crossings(
            p,
            (0.25, -0.04, string_z(p, 0.25)),
            (0.32, 0.04, string_z(p, 0.32)),
        )
        assert len(diagonal) == 4

        for event in diagonal:
            t = event["fraction"]
            i = STRING_NAMES.index(event["string"])
            assert abs(
                -0.04 + 0.08 * t
                - string_y(p, i, 0.25 + 0.07 * t)
            ) < 1e-12

    print(
        "PASS: geometry, target classification, "
        "forward/reverse/diagonal crossings"
    )


def check_baseline(old, new):
    assert (
        old.nq, old.nv, old.nu, old.ngeom, old.nbody
    ) == (
        new.nq, new.nv, new.nu, new.ngeom, new.nbody
    )

    fields = [
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "jnt_range",
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "site_pos",
    ]

    for field in fields:
        np.testing.assert_allclose(
            getattr(new, field),
            getattr(old, field),
            rtol=0,
            atol=2e-8,
            err_msg=field,
        )


def check_compiled_strings(model, name):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p = profile(name)

    for i, string in enumerate(STRING_NAMES):
        geom = model.geom("string_" + string).id

        axis = (
            data.geom_xmat[geom].reshape(3, 3)[:, 2]
            * model.geom_size[geom, 1]
        )

        actual = sorted([
            tuple(data.geom_xpos[geom] - axis),
            tuple(data.geom_xpos[geom] + axis),
        ])

        expected = [
            (
                0,
                string_y(p, i, 0) - 0.12,
                string_z(p, 0) + 0.12,
            ),
            (
                p.scale_length,
                string_y(p, i, p.scale_length) - 0.12,
                string_z(p, p.scale_length) + 0.12,
            ),
        ]

        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-9
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    repo = args.repo.resolve()

    check_geometry_math()

    with tempfile.TemporaryDirectory(
        prefix="strum-geometry-"
    ) as folder:
        folder = Path(folder)
        models = {}

        for name in ["baseline", "ek"]:
            for chord in CHORDS:
                model, _ = build(
                    repo,
                    folder / f"{name}-{chord}",
                    name,
                    chord,
                )
                models[name, chord] = model

        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/build_fretting_scene.py"),
                "--repo", str(repo),
                "--chord", "C",
                "--output", str(folder / "original"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        old = mujoco.MjModel.from_xml_path(
            str(folder / "original/scene.xml")
        )
        check_baseline(old, models["baseline", "C"])

        for name in ["baseline", "ek"]:
            check_compiled_strings(models[name, "C"], name)

    print(
        "PASS: eight scenes compiled; "
        "compiled strings match shared geometry"
    )
    print(
        "PASS: baseline matches existing scene "
        "within 20 nanometres for positions"
    )
    print(
        "NOT TESTED: ek reachability, contact holds, "
        "transitions, or performance"
    )


if __name__ == "__main__":
    main()