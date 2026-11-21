"""Run any `python -m strum.fretting` command with idle fingers making room.

Points the planner's `Hand` at IdleSpaceHand (idle_space.py), in memory, for
this run only, in both places it is looked up (strum.fretting for calibration,
strum.planner for chord changes). Nothing on disk changes.

    .venv/Scripts/python experiments/hand_pose/fretting_with_space.py coverage \
        --run outputs/holds-human-4 --output outputs/coverage-human-4
    .venv/Scripts/python experiments/hand_pose/fretting_with_space.py transition \
        --run outputs/holds-human-4 --cycles 10 --two-hands --output outputs/two-hand-human-4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from idle_space import IdleSpaceHand  # noqa: E402  (also puts the repo root on sys.path)
import strum.fretting as fretting  # noqa: E402
import strum.planner as planner  # noqa: E402

fretting.Hand = IdleSpaceHand
planner.Hand = IdleSpaceHand
print('Running with IdleSpaceHand (idle fingers make room); planner files unchanged.', flush=True)

# Optional: --right-angle DEG mounts the strumming hand at a human-like angle (right_hand.py).
args = sys.argv[1:]
if '--right-angle' in args:
    i = args.index('--right-angle')
    angle = float(args[i + 1])
    del args[i:i + 2]
    import right_hand
    right_hand.install(angle)
    print(f'Strumming hand mounted at {angle:.0f} deg (right_hand.py); bimanual.py unchanged.', flush=True)

# Optional: --swing TILT uses the swinging strumming hand (swing.py) with the given forearm tilt.
if '--swing' in args:
    i = args.index('--swing')
    tilt = float(args[i + 1])
    del args[i:i + 2]
    import swing
    swing.install(tilt)
    print(f'Swinging strumming hand, forearm tilt {tilt:.0f} deg (swing.py); bimanual.py and strumming.py unchanged.',
          flush=True)

# Optional: --upright turns the whole two-hand scene to a player's pose (upright.py). Applied after
# any strumming-hand option, so it wraps whichever right hand is installed.
if '--upright' in args:
    args.remove('--upright')
    import upright
    upright.install()
    print('Scene turned upright with gravity (upright.py); bimanual.py unchanged.', flush=True)

sys.argv = ['strum.fretting'] + args
fretting.main()
