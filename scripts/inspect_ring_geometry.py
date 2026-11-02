"""Inspect ring-fingertip collision geometry in the generated approach scene."""

import json
from pathlib import Path

import mujoco
import numpy as np

scene = Path("outputs/c-approach/scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

body_id = model.body("lh_rfdistal").id
body_rotation = data.xmat[body_id].reshape(3, 3)
body_position = data.xpos[body_id]

reference = data.site("ring_tip_reference").xpos.copy()
target = data.site("target_A_3").xpos.copy()

results = []

for geom_id in range(model.ngeom):
    if model.geom_bodyid[geom_id] != body_id:
        continue

    # Exclude visual-only geometry.
    if not (
        model.geom_contype[geom_id]
        or model.geom_conaffinity[geom_id]
    ):
        continue

    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise RuntimeError(
            f"Expected a mesh collision geom; found type "
            f"{model.geom_type[geom_id]}"
        )

    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])

    vertices = np.asarray(
        model.mesh_vert[start:start + count],
        dtype=float,
    )

    # Compiled mesh vertices are in the mesh geom's coordinate frame.
    geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
    geom_position = data.geom_xpos[geom_id]
    world_vertices = vertices @ geom_rotation.T + geom_position

    # Lowest vertex in this specific approach orientation.
    lowest_index = int(np.argmin(world_vertices[:, 2]))
    lowest_world = world_vertices[lowest_index]
    lowest_body = body_rotation.T @ (lowest_world - body_position)

    results.append({
        "geom_id": geom_id,
        "geom_name": model.geom(geom_id).name,
        "vertex_count": count,
        "world_bounds_min_m": world_vertices.min(axis=0).tolist(),
        "world_bounds_max_m": world_vertices.max(axis=0).tolist(),
        "lowest_vertex_world_m": lowest_world.tolist(),
        "lowest_vertex_body_local_m": lowest_body.tolist(),
        "lowest_vertex_height_above_target_mm": float(
            (lowest_world[2] - target[2]) * 1000
        ),
        "lowest_vertex_xy_offset_from_target_mm": (
            (lowest_world[:2] - target[:2]) * 1000
        ).tolist(),
    })

if not results:
    raise RuntimeError("No ring-distal collision mesh found")

report = {
    "status": "collision_geometry_inspection_only",
    "reference_marker_world_m": reference.tolist(),
    "target_world_m": target.tolist(),
    "collision_geometries": results,
}

output = scene.parent / "ring_geometry.json"
output.write_text(json.dumps(report, indent=2) + "\n")

print(json.dumps(report, indent=2))
print(f"Saved {output}")
