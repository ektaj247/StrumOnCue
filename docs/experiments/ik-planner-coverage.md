# Distal coupling and fingertip-space planner: holds and training transitions

> **Rule in force for these results (historical).** Everything below was measured on 2026-11-09 under the strict rule in which any neck contact of 0.01 N or more is unwanted. Later the same day the rule was changed to allow an assigned fingertip on the fretboard inside its own fret (see `docs/rl-fretting.md`). The "Open decision" section at the end is therefore closed. The 10/10 training-transition result holds under both rules.
>
> The hand is the upstream Shadow Hand **modified** with a J1 = J2 coupling per finger, not the unchanged upstream model.

Local validation: 2026-11-09, native MuJoCo 3.1.4 (Linux arm64 CPU). The same commands were re-run unchanged on MuJoCo 4.0.0 with identical pass/fail outcomes. Not yet run on CARC or under mujoco_warp.

These are development results for a scripted controller on a fixed mount. They are not benchmark scores, not a robustness estimate, and no learning is involved. The two held-out transitions (Am→G, F→C) were not executed.

## What changed

1. **Distal coupling (hand model).** The upstream hand drives J1 + J2 of each finger with one tendon and leaves J1 − J2 free. Under a fixed command that free mode creeps at about 0.5°/s until a joint limit stops it. In the previous calibrated holds this showed up as 30.6° of joint drift at 60 s and 40.1° at 180 s, with the C-chord contact force falling from 0.51 N to 0.019 N (the pass threshold is 0.01 N). `strum/planner.py: couple_distal` adds a J1 = J2 equality constraint per finger when a scene is calibrated. The calibration IK already assumed J1 = J2. This is a modelling choice and a change from the pinned upstream hand; it is recorded as `distal_coupling` in each summary.
2. **Gravity feedforward.** Commands are the planned joint (or tendon) position plus the static torque the actuator must supply there, divided by its gain. Measured free-space fingertip error at a hover waypoint: about 0.05 mm.
3. **Fingertip-space planner.** Per-finger IK gives a press pose and a hover pose 10 mm above it. A transition is lift → travel at hover height → press. Fingers that fret the same note in both chords stay down. The 3 s homing sequence is gone. The pad may sit anywhere inside its fret interval; two fingers sharing a fret take opposite ends of it (this is what makes G reachable without finger–finger contact).
4. **One press depth.** Pressing is a commanded pad depth below the touch point, the same for every finger of a chord. −0.75 mm was the first value tried and passed for all four chords. The 15-point per-chord actuator-offset grid is removed.

Thresholds are unchanged: 0.01 N force threshold, ≥95% simultaneous target contact after 0.5 s, zero samples of unwanted contact ≥0.01 N in every evaluated phase, neck contact still counted as unwanted.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m strum.fretting calibrate --placements assets/fretting_seed_v1 --profile ek --output outputs/ek-holds-05
python -m strum.fretting coverage --run outputs/ek-holds-05 --output outputs/training-coverage-02
python -m strum.fretting transition --run outputs/ek-holds-05 --route C G F Am C --cycles 10 --output outputs/ek-loop-CGFAmC-10
python -m strum.fretting transition --run outputs/ek-holds-05 --cycles 10 --two-hands --output outputs/ek-two-hand-ik-10
```

Calibration takes about 20 s. Each chord is first reached from the rest pose in one rollout (reach 1 s, press 0.5 s, hold 5 s), then held from the settled state for 10 s and 60 s.

## Results

| Check | Before (ek-holds-03, legacy planner) | Now (ek-holds-05) |
|---|---|---|
| 60 s holds, simultaneous contact | C/Am/F 100%, G 99.38% | all four 100% |
| Max joint drift over 60 s | 30.6° | 0.11° |
| 180 s holds | pass, C force down to 0.019 N | pass, forces unchanged from 60 s (0.05–0.15 N) |
| Chord reached from rest, not initialized in contact | not tested | 4/4 |
| Training transitions, strict rules | 3/10 | 10/10 |
| Time per transition | about 9 s | 2.0 s |
| C→G→F→Am→C, 10 cycles, one rollout | not possible | 10/10 |
| Two-hand C–Am–C, 10 cycles | 240/240 notes | 240/240 notes, 3.77 ms mean onset error |
| Two-hand C→G→F→Am→C, 1 cycle | not possible | 48/48 notes |

Transition speed (`--time-scale` scales lift/move/press): 10/10 at 2.0 s, 1.0 s, 0.6 s and 0.4 s per transition; 7/10 at 0.2 s (Am→F, C→G, F→G lose the destination hold).

Per-transition rows are in `ik-planner-coverage.csv`.

## Limits

- Fixed forearm, fixed instrument, rigid strings, nominal conditions only. No perturbations, no noise, no randomized starts.
- The planner and the shared-fret rule were developed against chord approaches and the training transitions. Passing them is not evidence about held-out transitions.
- Open-loop: commands are fixed waypoints; nothing reacts to measured contact.
- Contact forces are low (0.05–0.15 N). The weakest is the middle finger on G's C string at 0.05 N, five times the threshold.
- Pressing at the nut-side end of a fret interval (index finger in G) counts as the right note here. On a real instrument it would be more likely to buzz.
- Hover height, fret margin, and the depth list are fixed constants, not tuned per chord, but they are still choices.

## Open decision: neck contact

The strict rule fails a rollout if a fretting fingertip touches the fretboard with ≥0.01 N. With rigid strings 2.0–2.6 mm above the fretboard, the passing window in commanded depth is narrow: C passes from −0.25 to −0.75 mm and fails at −1.0 mm (0.05 N on the neck); F fails at −1.5 mm; Am passes down to −2.0 mm. The scripted controller fits inside that window, so nothing was relaxed here. A learned policy would have to find the same sub-millimetre window with 1 N·m/rad actuators, and on a real instrument the fingertip does touch the fretboard. Whether fretboard contact by the assigned finger in its own fret cell should count as a failure is a benchmark-definition choice to make, and write down, before any held-out evaluation.
