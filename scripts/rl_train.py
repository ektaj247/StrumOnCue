"""Train the chord-fretting policy. Same options as mjlab's train command.

    python scripts/rl_train.py Strum-Fret-Shadow --env.scene.num-envs 4096
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import strum.rl  # noqa: E402,F401  (registers the tasks)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == '__main__':
    main()
