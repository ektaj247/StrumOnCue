"""Watch a trained chord-fretting policy in the browser viewer.

    python scripts/rl_play.py Strum-Fret-Shadow --checkpoint-file <model_N.pt>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import strum.rl  # noqa: E402,F401  (registers the tasks)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == '__main__':
    main()
