"""Score a trained policy on every chord change of one split.

Each rollout starts holding the source chord, is given the target chord, and
has --period seconds. A rollout counts as held when the target chord is
correctly fretted for at least 95% of the final second, and as clean when in
addition no unwanted contact occurred at any point. A rollout that terminates
for any reason before the period ends is a failure.

    python scripts/rl_eval.py --checkpoint <model_N.pt> --split train
    python scripts/rl_eval.py --checkpoint <model_N.pt> --split heldout

Run the held-out split only once the training setup is frozen.
"""
import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402

from strum.rl import mdp  # noqa: E402
from strum.rl.chords import load_table, pairs  # noqa: E402
from strum.rl.env_cfg import fret_ppo_cfg, make_fret_env_cfg  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'heldout'], required=True)
    parser.add_argument('--rollouts', type=int, default=256, help='rollouts per chord change')
    parser.add_argument('--period', type=float, default=3., help='seconds allowed per chord change')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True, help='folder for results.csv and summary.json')
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    table = load_table()
    names = table['chord_names']
    todo = pairs(table, a.split)
    cfg = make_fret_env_cfg(split=a.split)
    cfg.seed = a.seed
    cfg.scene.num_envs = len(todo) * a.rollouts
    cfg.commands['chord'].resampling_time_range = (a.period, a.period)
    cfg.episode_length_s = 1e6
    agent = fret_ppo_cfg()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device), clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent), device=a.device)
    runner.load(str(a.checkpoint), load_cfg={'actor': True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    base = env.unwrapped
    term = base.command_manager.get_term('chord')
    index = torch.arange(base.num_envs, device=a.device) // a.rollouts
    source = torch.tensor([names.index(s) for s, _ in todo], device=a.device)[index]
    target = torch.tensor([names.index(t) for _, t in todo], device=a.device)[index]
    term.assign(source, target)
    obs, _ = env.reset()
    assert bool((term.source == source).all() and (term.target == target).all())

    steps = int(round(a.period / base.step_dt)) - 1      # stop before the target would change
    first_ok = torch.full((base.num_envs,), float('nan'), device=a.device)
    ok_steps = torch.zeros(base.num_envs, device=a.device)
    ended = torch.zeros(base.num_envs, dtype=torch.bool, device=a.device)
    with torch.inference_mode():
        for step in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            # A rollout that ended early was reset by the env; whatever it does
            # afterwards belongs to a different rollout.
            ended |= dones.bool()
            ok = mdp.contact_state(base)['chord_ok'] & ~ended
            first_ok = torch.where(ok & first_ok.isnan(), torch.full_like(first_ok, (step + 1) * base.step_dt), first_ok)
            ok_steps += ok
    held, clean = term.period_result()
    held, clean = held & ~ended, clean & ~ended
    # The run stops one step before the target would change, so the judged
    # window is one or two policy steps short of hold_window_s.
    window = int(term.window_steps[~ended].min()) if (~ended).any() else 0
    assert ended.all() or window >= round(term.cfg.hold_window_s / base.step_dt) - 2, 'incomplete hold window'

    rows = []
    for i, (s, t) in enumerate(todo):
        mine = index == i
        rows.append({'source': s, 'target': t, 'rollouts': int(mine.sum()), 'terminated': int(ended[mine].sum()),
                     'held': float(held[mine].float().mean()), 'clean': float(clean[mine].float().mean()),
                     'chord_ok_time_fraction': float((ok_steps[mine] / steps).mean()),
                     'reached': float((~first_ok[mine].isnan()).float().mean()),
                     'median_time_to_chord_s': float(first_ok[mine].nanmedian()) if (~first_ok[mine].isnan()).any() else None})
    with (a.output / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'checkpoint': str(a.checkpoint), 'split': a.split, 'period_s': a.period, 'rollouts_per_pair': a.rollouts,
               'seed': a.seed, 'fretboard_rule': table['fretboard_rule'], 'force_threshold_N': mdp.FORCE_THRESHOLD,
               'required_hold_fraction': term.cfg.required_hold_fraction, 'hold_window_s': term.cfg.hold_window_s, 'hold_window_steps_judged': window,
               'held': float(held.float().mean()), 'clean': float(clean.float().mean()),
               'terminated': int(ended.sum()), 'pairs': rows}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in rows:
        print(f"{row['source']:>2} -> {row['target']:<2}  held {row['held']:.3f}  clean {row['clean']:.3f}  "
              f"time to chord {row['median_time_to_chord_s']}")
    print(f"{a.split.upper()}: held {summary['held']:.3f}, clean {summary['clean']:.3f} over {base.num_envs} rollouts, "
          f"{summary['terminated']} terminated early")
    print('Saved', a.output / 'summary.json')
    env.close()


if __name__ == '__main__':
    main()
