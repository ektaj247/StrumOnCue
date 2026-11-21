"""Run the normal calibration with idle fingers making room.

Calibration (strum/fretting.py) builds its hand with `Hand(m, p)`. For this
run only, that name is pointed at IdleSpaceHand (idle_space.py), in memory.
Nothing on disk changes; strum/planner.py and strum/fretting.py stay as they
are. All arguments are passed to `python -m strum.fretting calibrate`.

    .venv/Scripts/python experiments/hand_pose/calibrate_with_space.py \
        --placements outputs/hand-pose/20261121-021900/seed --output outputs/holds-human-3
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idle_space import IdleSpaceHand  # noqa: E402  (also puts the repo root on sys.path)
import strum.fretting as fretting  # noqa: E402

fretting.Hand = IdleSpaceHand
print('Calibrating with IdleSpaceHand (idle fingers make room); planner files unchanged.', flush=True)
sys.argv = ['strum.fretting', 'calibrate'] + sys.argv[1:]
fretting.main()
