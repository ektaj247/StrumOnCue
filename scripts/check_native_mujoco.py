"""Check basic native MuJoCo stepping and ground contact."""

import sys

import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for _ in range(500):
    mujoco.mj_step(model, data)

assert np.isfinite(data.qpos).all()
assert 0.02 < data.qpos[2] < 0.04

print("Python:", sys.version.split()[0])
print("MuJoCo:", mujoco.__version__)
print("NumPy:", np.__version__)
print("Simulation time:", round(data.time, 3))
print("Final sphere height:", round(float(data.qpos[2]), 4))
print("PASS: native MuJoCo stepping and contact.")
