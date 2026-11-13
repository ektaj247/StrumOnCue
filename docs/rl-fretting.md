# RL stage 1: goal-conditioned chord fretting

Status 2026-11-10: environment built and checked on CPU; contact bookkeeping revised after an external code review (see "Review fixes"). No policy has been trained yet. Reward weights and PPO settings are first choices, not tuned.

The hand is a **modified Shadow Hand model**: the upstream left hand plus a J1 = J2 coupling on each of the four fingers. Results here are for that coupled model, not the unchanged upstream hand.

## What is being learned

A neural-network policy controls the twelve finger actuators of the fixed-mount left Shadow Hand (splay, proximal and coupled-distal target for each of four fingers). Wrist, thumb and LFJ5 hold their rest command. Fifty times a second the policy reads the hand's joint angles and speeds, its fingertip positions, its previous action, and the target chord, and outputs new position targets.

An episode lasts 9 s. The hand starts holding one chord and is told a target chord. Every 2.5 to 3.5 s the target changes to another chord. The target is given as the fret each string should be pressed at (one-hot over frets 0 to 3 for each of the four strings). The policy is never told which finger goes where or where the fingertips should be; that information only enters through the reward.

The right hand, strum patterns and tempo are not part of this stage.

## The generalization test

There are twelve directed chord changes among C, Am, F and G. Training uses ten. Am→G and F→C never occur in training: the policy has reached G and C from other chords, and has left Am and F for other chords, but has never made those two specific changes. After training, `scripts/rl_eval.py --split heldout` measures them. The split comes from `manifests/benchmark_v1.json` and is copied into the chord table.

## Success rule

Same contact rule as the scripted evaluation: 0.01 N threshold; every assigned fingertip on its string inside its fret; no unwanted hand contact of 0.01 N or more. For one second after a target change, contacts belonging to the previous chord are not counted as unwanted.

Fretboard contact (rule adopted 2026-11-09): the fingertip of an assigned finger may touch the fretboard anywhere across the neck's width, as long as the contact lies between the two fret lines of its assigned fret. It is not restricted to a strip beside the assigned string. Touching any string other than the assigned one is still unwanted, so the wider region does not permit muting a neighbour. Results before 2026-11-09 used the stricter rule in which any neck contact was unwanted.

A chord change is **held** when the target chord is correct for at least 95% of the final second before the next change, and **clean** when in addition no unwanted contact occurred at any time in that period.

How contact is measured here, and how that differs from the scripted evaluation:

- Unwanted force is checked at every physics substep (4 ms) and the largest value in the policy step is used, so a contact shorter than one policy step still counts. Whether the chord is correct is sampled once per policy step (20 ms).
- Forces come from mjlab contact sensors: the strongest contact per fingertip and string, and the 16 strongest contacts on the whole hand. Unwanted force is the hand total minus the permitted fingertip contacts. If the hand ever has more than 16 contacts the step is flagged as overflow and counted as not correct; `Metrics/chord/contact_overflow` logs how often. Typical counts are 1 to 3.
- A fingertip with two contact points on its own string has only the stronger one recognised as permitted; the other is counted as unwanted. This errs on the strict side.
- The simulation is MuJoCo Warp at a 4 ms step with the implicitfast integrator, against native MuJoCo at 2 ms for the scripted evaluation.

Numbers reported for a final policy should be confirmed by replaying it in native MuJoCo under `strum.fretting`'s evaluator, which sees every contact.

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

- `assets/rl_chords_v1.json`: chord table (settled joint state, holding command and fingertip positions per chord, mount pose, train and held-out pairs). Regenerate with `python -m strum.rl.export_chords --run <calibrated run> --output assets/rl_chords_v1.json`.
- `strum/rl/scene.py`: hand (with distal coupling and pad markers) and ukulele entities.
- `strum/rl/mdp.py`: action, chord command, contact bookkeeping, observations, rewards.
- `strum/rl/env_cfg.py`: environment and PPO configuration. Tasks `Strum-Fret-Shadow` and `Strum-Fret-Shadow-Heldout`.
- `scripts/rl_selfcheck.py`: five checks of the contact bookkeeping; `slurm/fret_train.sbatch` runs it before training and stops if any fails.
- `scripts/rl_train.py`, `scripts/rl_play.py`, `scripts/rl_eval.py`, `slurm/fret_train.sbatch`, `slurm/fret_eval.sbatch`.

## Checks run (CPU, mjlab 1.4.0, MuJoCo Warp 3.8.1, 2026-11-10)

- Ukulele collision geometry is identical to the calibrated native scene. After reset, fingertip markers match the chord table to 0.0001 mm.
- `scripts/rl_selfcheck.py`: all five checks pass. With the pre-review bookkeeping restored, checks 2 and 3 fail, so they do detect the faults they are meant to.
- Replaying the scripted planner through the action interface on the ten training changes (4 rollouts each, exact starts), with unwanted contact now checked every substep: held 100%, clean 100%. Before the review fixes the same replay with the training start noise gave held 100%, clean 97.5%; that noisy case has not been re-run.
- PPO ran two iterations with 48 environments and saved checkpoints; `rl_eval.py` loaded one and scored it 0%, as expected for an untrained policy, with 0 early terminations.
- Random actions for 120 steps: at most 53 constraints in one world (limit 256).

Not checked: learning itself, GPU speed, and video rendering.

## Review fixes (2026-11-10)

An external review of commit dc99086 found three faults in the evaluation path. All three were real.

1. The contact summary was cached per env step only, so after a reset or target change inside the step it still described the previous target. Rewards were unaffected (they are computed first), but the logged success metrics and one critic observation were. The cache is now also keyed on a command version, and an env that restarts inside a step contributes nothing to its new episode's statistics for that step.
2. The whole-hand sensor kept eight contacts and overflow was silent, and contact was read once per policy step. It now keeps sixteen, overflow is detected and fails the step, and unwanted force is checked every substep.
3. `rl_eval.py` ignored early terminations, so a rollout that ended and was auto-reset could be scored on what happened afterwards. Such rollouts are now failures and their count is reported.

## Run log

### Run 1, "pilot" (2026-11-10, W&B run koxffwpk, checkpoint `2026-11-10_01-57-50_pilot/model_2999.pt`)

4096 environments, 3000 iterations, 1 h 46 min on one A40 (about 46,000 steps/s). Self-check passed on the GPU. No NaN terminations, no sensor overflow.

Training-time metrics at the end (with exploration noise): chord correct 75.5% of the time, chord changes held 72.7%, clean 57.9%. Mean action noise rose from 0.50 to 3.10.

`rl_eval.py --split train` (no exploration noise, 256 rollouts per chord change, CARC job 14035653):

| Change | Held | Clean | Median time to chord |
|---|---:|---:|---:|
| Am→C | 1.000 | 0.863 | 0.08 s |
| Am→F | 1.000 | 0.992 | 0.08 s |
| C→Am | 1.000 | 0.863 | 0.08 s |
| C→F | 0.988 | 0.840 | 0.10 s |
| C→G | 0.000 | 0.000 | never |
| F→Am | 1.000 | 0.988 | 0.02 s |
| F→G | 0.000 | 0.000 | never |
| G→Am | 1.000 | 0.863 | 0.02 s |
| G→C | 1.000 | 0.828 | 0.08 s |
| G→F | 0.992 | 0.621 | 0.06 s |
| All | 0.798 | 0.686 | |

Reading: the eight changes that do not end in G are held essentially every time. Both changes into G fail every time. G is the only three-finger chord and the only one where two fingers share a fret. Leaving G works, but those rollouts start with G already fretted.

The times to chord are a few policy steps. The model's finger actuators can apply up to 1 N·m to links of a few grams, so fingers cross the fretboard in tens of milliseconds. That is a property of the upstream hand model, not evidence about a real hand, and it is why "clean" lags "held": fast moves clip a neighbouring string or finger on the way.

The scripted planner is held and clean on all ten of these changes, so into-G is a learning problem, not a reachability problem.

### Run 2 changes (2026-11-12)

1. **Hold periods.** With probability 0.25 a new target equals the current chord. The episode then starts (or continues) with that chord already fretted, so the policy experiences a correct G and has to keep it, instead of having to discover a three-finger chord from scratch. In run 1 the policy was never asked to hold a chord it started in. Hold periods are scored as `Metrics/chord/hold_success` and are excluded from `transition_success` and `clean_success`. They are not chord changes and do not touch the held-out pairs.
2. **Exploration noise capped.** Standard deviation limited to [0.02, 0.5] and entropy coefficient lowered from 0.003 to 0.001.
3. **Evaluation diagnostics.** For any change held less than 95% of the time, `rl_eval.py` now prints, over the judged window, how often each required finger was correct, each fingertip's distance from where it sits in the calibrated chord, and how often there was unwanted contact.

Two training changes at once, so run 2 cannot say which one mattered.

## Train on CARC

```bash
# from the project root, after git pull
mkdir -p logs
sbatch --partition=debug --time=00:30:00 --export=ALL,ITERATIONS=20,NUM_ENVS=1024,RUN_NAME=smoke slurm/fret_train.sbatch
sbatch slurm/fret_train.sbatch
sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/strum_fret/<run>/model_2999.pt,SPLIT=train slurm/fret_eval.sbatch
```

Curves to watch in W&B (project `strum-on-cue`): `Metrics/chord/transition_success`, `Metrics/chord/clean_success`, `Metrics/chord/chord_ok`, `Episode_Reward/chord_ok`, `Episode_Reward/unwanted`, `Episode_Metrics/substep_unwanted` (fraction of substeps with unwanted contact). Training metrics use the noisy exploring policy, so they sit below what `rl_eval.py` reports for the same checkpoint.

Do not run `SPLIT=heldout` until rewards, observations and training length are frozen. Any change made after looking at held-out numbers invalidates them as a test.
