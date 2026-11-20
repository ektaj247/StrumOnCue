"""Checks of the RL environment's contact bookkeeping. Run before training.

    python scripts/rl_selfcheck.py            # GPU if present, else CPU

1. Holding each chord's calibrated command keeps the chord correct.
2. Changing the target inside an env step changes the verdict at once.
3. Envs that restart do not inherit contact history from their last episode.
4. A contact that exists only between policy steps is still counted.
5. More hand contacts than the sensor keeps is reported, not ignored.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from strum.rl import mdp  # noqa: E402
from strum.rl.env_cfg import make_fret_env_cfg  # noqa: E402


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = make_fret_env_cfg()
    cfg.scene.num_envs = 16
    cfg.events['reset_to_chord'].params['noise'] = 0.
    cfg.episode_length_s = 1e6
    cfg.commands['chord'].resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    term = env.command_manager.get_term('chord')
    act = env.action_manager.get_term('fingers')
    chords = term.chords
    every = torch.arange(env.num_envs, device=device) % len(chords.names)
    term.assign(every, every)                     # target = starting chord: a pure hold
    env.reset()
    failures = []

    def check(name, passed, detail=''):
        print(('PASS  ' if passed else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))
        if not passed:
            failures.append(name)

    hold = act.to_action(torch.cat([chords.joint_cmd[every], chords.tendon_cmd[every]], dim=-1))
    ok = torch.zeros(env.num_envs, device=device)
    steps = 100
    for _ in range(steps):
        env.step(hold)
        ok += mdp.contact_state(env)['chord_ok']
    worst = {chords.names[c]: float((ok[every == c] / steps).min()) for c in range(len(chords.names))}
    check('1 calibrated holds stay correct', min(worst.values()) >= .95, f'worst fraction per chord {worst}')

    before = mdp.contact_state(env)['chord_ok'].clone()
    saved = term.target.clone()
    other = (term.target + 1) % len(chords.names)
    term.target[:] = other
    term.version += 1
    after = mdp.contact_state(env)['chord_ok'].clone()
    term.target[:] = saved
    term.version += 1
    check('2 target change inside a step changes the verdict', bool(before.all() and not after.any()),
          f'ok before {int(before.sum())}/{env.num_envs}, after switching target {int(after.sum())}/{env.num_envs}')

    # Dirty every env's history, then restart half of them inside an env step.
    term.period_unwanted[:] = True
    mdp._substeps(env)['unwanted'][:] = 1.
    half = torch.arange(0, env.num_envs, 2, device=device)
    env.episode_length_buf[half] = env.max_episode_length      # time out on the next step
    env.step(hold)
    rest = torch.arange(1, env.num_envs, 2, device=device)
    clean_restart = not term.period_unwanted[half].any() and bool((term._steps[half] == 0).all())
    check('3 restarted envs start with clean history', bool(clean_restart) and bool(term.period_unwanted[rest].all()),
          f'restarted flagged {int(term.period_unwanted[half].sum())}, others still flagged {int(term.period_unwanted[rest].sum())}/{len(rest)}')

    for _ in range(5):
        env.step(hold)
    term.period_unwanted[:] = False
    env.step(hold)
    quiet = not term.period_unwanted.any()
    # Pretend a 0.02 N unwanted contact appeared in one substep and was gone by the end of the step.
    original = mdp._measure

    def one_substep(e):
        out = original(e)
        if (e._sim_step_counter - 1) % e.cfg.decimation == 1:
            out['unwanted'] = out['unwanted'] + .02
        return out

    mdp._measure = one_substep
    env.step(hold)
    mdp._measure = original
    check('4 contact inside one substep is counted', bool(quiet and term.period_unwanted.all()),
          f'flagged {int(term.period_unwanted.sum())}/{env.num_envs}')

    def too_many(e):
        out = original(e)
        out['overflow'] = torch.ones_like(out['overflow'])
        return out

    mdp._measure = too_many
    env.step(hold)
    state = mdp.contact_state(env)
    mdp._measure = original
    check('5 sensor overflow is reported and fails the chord', bool(state['overflow'].all() and not state['chord_ok'].any()))

    env.close()
    print('SELF-CHECK', 'FAILED: ' + ', '.join(failures) if failures else 'PASSED', f'({device})')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
