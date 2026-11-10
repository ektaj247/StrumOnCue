# RL stage 1: goal-conditioned chord fretting

Status 2026-11-09: environment built and checked on CPU. No policy has been trained yet. Reward weights and PPO settings are first choices, not tuned.

## What is being learned

A neural-network policy controls the twelve finger actuators of the fixed-mount left Shadow Hand (splay, proximal and coupled-distal target for each of four fingers). Wrist, thumb and LFJ5 hold their rest command. Fifty times a second the policy reads the hand's joint angles and speeds, its fingertip positions, its previous action, and the target chord, and outputs new position targets.

An episode lasts 9 s. The hand starts holding one chord and is told a target chord. Every 2.5 to 3.5 s the target changes to another chord. The target is given as the fret each string should be pressed at (one-hot over frets 0 to 3 for each of the four strings). The policy is never told which finger goes where or where the fingertips should be; that information only enters through the reward.

The right hand, strum patterns and tempo are not part of this stage.

## The generalization test

There are twelve directed chord changes among C, Am, F and G. Training uses ten. Am→G and F→C never occur in training: the policy has reached G and C from other chords, and has left Am and F for other chords, but has never made those two specific changes. After training, `scripts/rl_eval.py --split heldout` measures them. The split comes from `manifests/benchmark_v1.json` and is copied into the chord table.

## Success rule

Same contact rule as the scripted evaluation: 0.01 N threshold; every assigned fingertip on its string inside its fret; no unwanted hand contact of 0.01 N or more; an assigned fingertip may touch the fretboard inside its own fret. For one second after a target change, contacts belonging to the previous chord are not counted as unwanted.

A chord change is **held** when the target chord is correct for at least 95% of the final second before the next change, and **clean** when in addition no unwanted contact occurred at any time in that period.

Differences from the scripted evaluation, all in the direction of an approximation: contact forces come from mjlab contact sensors (strongest contact per fingertip and string; the eight strongest contacts on the whole hand), and the simulation runs in MuJoCo Warp at a 4 ms step with the implicitfast integrator instead of native MuJoCo at 2 ms.

## Reward

| Term | Weight | Meaning |
|---|---:|---|
| reach | 1.0 | assigned fingertips near where they sit when the chord is fretted (4 cm scale) |
| reach_fine | 1.0 | same, 6 mm scale |
| clear | 0.5 | unused fingertips at least 15 mm above the fretboard |
| contacts | 1.0 | fraction of assigned fingers pressing the right string and fret |
| chord_ok | 3.0 | whole chord correct with no unwanted contact |
| unwanted | −1.0 | unwanted contact force |
| action_rate | −0.01 | change in action between steps |

The critic additionally sees fingertip-to-target errors, contact forces and timers. The actor does not.

## Files

- `assets/rl_chords_v1.json`: chord table (settled joint state, holding command and fingertip positions per chord, mount pose, train and held-out pairs). Regenerate with `python -m strum.export_chords --run <calibrated run> --output assets/rl_chords_v1.json` (runs in the local MuJoCo environment; mjlab is not needed).
- `strum/rl/scene.py`: hand (with distal coupling and pad markers) and ukulele entities.
- `strum/rl/mdp.py`: action, chord command, contact bookkeeping, observations, rewards.
- `strum/rl/env_cfg.py`: environment and PPO configuration. Tasks `Strum-Fret-Shadow` and `Strum-Fret-Shadow-Heldout`.
- `scripts/rl_train.py`, `scripts/rl_play.py`, `scripts/rl_eval.py`, `slurm/fret_train.sbatch`, `slurm/fret_eval.sbatch`.

## Checks run (CPU, mjlab 1.4.0, MuJoCo Warp 3.8.1, 2026-11-09)

- Ukulele collision geometry is identical to the calibrated native scene. After reset, fingertip markers match the chord table to 0.0001 mm.
- Holding each chord's calibrated command through the action interface: chord correct 100% of the time for all four chords.
- Replaying the scripted planner through the action interface on the ten training changes (4 rollouts each): held 100%, clean 100% from exact starts; held 100%, clean 97.5% with the training start noise. The task is solvable through this action space and the in-environment success measure agrees with the scripted one.
- PPO ran four iterations with 64 environments and saved checkpoints; `rl_eval.py` loaded one and scored it 0%, as expected for an untrained policy.
- Random actions for 120 steps: at most 53 constraints in one world (limit 256).

Not checked: learning itself, GPU speed, and video rendering.

## Train on CARC

```bash
# from the project root, after git pull
mkdir -p logs
sbatch --partition=debug --time=00:30:00 --export=ALL,ITERATIONS=20,NUM_ENVS=1024,RUN_NAME=smoke slurm/fret_train.sbatch
sbatch slurm/fret_train.sbatch
sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/strum_fret/<run>/model_2999.pt,SPLIT=train slurm/fret_eval.sbatch
```

Curves to watch in W&B (project `strum-on-cue`): `Metrics/chord/transition_success`, `Metrics/chord/clean_success`, `Metrics/chord/chord_ok`, `Episode_Reward/chord_ok`, `Episode_Reward/unwanted`. Training metrics use the noisy exploring policy, so they sit below what `rl_eval.py` reports for the same checkpoint.

Do not run `SPLIT=heldout` until rewards, observations and training length are frozen. Any change made after looking at held-out numbers invalidates them as a test.
