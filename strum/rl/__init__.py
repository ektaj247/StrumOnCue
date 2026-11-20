"""Goal-conditioned chord fretting in mjlab. Importing this registers the tasks.

Strum-Fret-Shadow          training pairs only (10 of the 12 chord changes)
Strum-Fret-Shadow-Heldout  the two chord changes never seen in training
"""
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import fret_ppo_cfg, make_fret_env_cfg

for _task, _split in (('Strum-Fret-Shadow', 'train'), ('Strum-Fret-Shadow-Heldout', 'heldout')):
    register_mjlab_task(task_id=_task, env_cfg=make_fret_env_cfg(split=_split),
                        play_env_cfg=make_fret_env_cfg(play=True, split=_split), rl_cfg=fret_ppo_cfg())
