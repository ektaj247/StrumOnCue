# StrumOnCue

Final project for CSCI 645 (Fall 2026) by Ekta Jichkar and Jithin Rajan.

We study whether a learned policy for a dexterous hand can reach a specified multi-finger contact configuration by a deadline set from outside, and whether that ability carries over to combinations of target, deadline and coordination pattern that it was not trained on.

Our testbed is two simulated Shadow Hands playing a ukulele in MuJoCo. The left hand frets chords and the right hand strums. Music gives exact ground truth for both contact (which string is pressed at which fret) and timing (when each note must sound).

| Factor in the benchmark | What it stands for |
|---|---|
| Chord | The contact configuration to reach |
| Tempo | The time available to reach it |
| Strum pattern | The timing imposed by the other hand |

## Where we are

- The ukulele model, the note detector and the benchmark are in place.
- A scripted controller plays all four chords with the left hand and strums with the right. It passes every training chord change.
- The RL environment for the left hand is built and the first training runs are under way. The task is described in [docs/rl-fretting.md](docs/rl-fretting.md).
- Strum patterns and tempo are not part of the RL task yet, and the right hand is still scripted.

## The benchmark

We use four chords (C, G, Am, F), three strum patterns and three tempos.

| Pattern | Strokes on an eighth-note grid |
|---|---|
| P1 | `D . D . D . D .` |
| P2 | `D U D U D U D U` |
| P3 | `D . D U . U D U` |

Tempos are 70, 90 and 110 BPM. A task is one chord change played with one pattern at one tempo, which gives 12 × 3 × 3 = 108 tasks. They are listed in [manifests/benchmark_v1.json](manifests/benchmark_v1.json).

We train on 81 tasks and hold out 27:

- 18 tasks cover the chord changes Am→G and F→C, which never appear in training.
- 9 tasks pair pattern P3 with a change into F (from C, G or Am), a pairing that never appears in training.

Every chord, pattern and tempo is seen in training. Only these combinations are withheld.

### Structure of the chord changes

Chord changes differ in how much the hand has to do. With the fingerings in [assets/rl_chords_v1.json](assets/rl_chords_v1.json):

| Change | Split | Fingers whose contact changes | Finger kept in place | Strings whose fret changes |
|---|---|---:|---|---:|
| Am→F, F→Am | Train | 1 | Middle, on G fret 2 | 1 |
| C→Am, Am→C | Train | 2 | None | 2 |
| C→G, C→F, G→C | Train | 3 | None | 3 |
| G→Am, G→F, F→G | Train | 3 | None | 4 |
| F→C | Held out | 3 | None | 3 |
| Am→G | Held out | 3 | None | 4 |

Both held-out changes are of the hardest kind, so a lower held-out score could reflect difficulty rather than novelty. Each held-out change has a training change of identical structure that also ends on the same chord: F→C matches G→C, and Am→G matches F→G. We report held-out results next to these matched changes.

## How notes are detected

We do not simulate string vibration or sound. The strings are rigid, and notes are detected from contact and geometry.

- A string is fretted when a fingertip presses it between two frets with at least 0.01 N.
- A note sounds when the strumming fingertip crosses a string.
- The pitch is the open-string pitch plus the fret being pressed.
- Detected notes are matched to expected notes by string, pitch, direction and time, with a 50 ms tolerance. We report F1.
- A chord change is held when the target chord is correct for at least 95% of the final second before the next change, and clean when, in addition, no unwanted contact occurs in that period.

## Evaluation plan

Planned and not yet implemented:

- **Matched comparison.** Held-out changes are compared with the training changes of the same structure, as described above.
- **Scripted reference.** Every learned result is reported next to the scripted player's score on the same task.
- **Timing and effort.** For each chord change we record the ready margin (how long before the next strum the chord was held), the actuator effort and the fingertip jerk.
- **Predictions first.** Before each held-out evaluation we commit our predictions to [docs/](docs/), so the commit history shows they came first.
- **Advance notice.** At test time we reduce how early the policy learns the next chord, and measure when it stops being ready in time.
- **Instrument pose.** At test time we shift and tilt the ukulele by small amounts. The policy does not observe its position.

## Setup

```bash
git clone --recurse-submodules https://github.com/ektaj247/StrumOnCue.git
cd StrumOnCue
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If you cloned without the submodule, run `git submodule update --init`. The submodule [external/mjlab_hand](external/mjlab_hand) provides the Shadow Hand model.

RL training needs mjlab as well. The cluster scripts expect a Python environment at `external/mjlab_hand/.venv`; see that repository's README for how to create it.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Running the scripted player

Run from the repository root. Results are written to `outputs/`, which is not tracked.

```bash
# Geometry and scene checks
python scripts/check_geometry_integration.py

# Find and verify a hold for each chord
python -m strum.fretting calibrate --placements assets/fretting_seed_v1 --output outputs/holds

# Run the ten training chord changes
python -m strum.fretting coverage --run outputs/holds --output outputs/coverage

# Both hands: C, Am, C with strumming
python -m strum.fretting transition --run outputs/holds --cycles 10 --two-hands --output outputs/two-hand
```

Add `--view` to watch a run and `--polished` for the textured ukulele. On macOS the viewer needs `mjpython` instead of `python`. Each run needs a new output folder.

Results so far (native MuJoCo 3.1.4):

| Check | Result |
|---|---|
| Hold each chord for 60 s | All four hold |
| Ten training chord changes, 2.0 s each | 10 of 10 |
| Same changes at 0.4 s each | 10 of 10 |
| Two hands, C–Am–C, ten cycles | 240 of 240 notes |
| Two hands, C→G→F→Am→C | 48 of 48 notes |

More detail is in [docs/experiments/ik-planner-coverage.md](docs/experiments/ik-planner-coverage.md).

## RL chord fretting

A PPO policy controls the twelve finger actuators of the left hand at 50 Hz. Every few seconds it is given a new target chord, described as the fret to press on each string. It trains on ten chord changes; Am→G and F→C are held out.

On the USC CARC cluster:

```bash
mkdir -p logs
sbatch slurm/fret_train.sbatch
sbatch --export=ALL,CHECKPOINT=<path to model_N.pt>,SPLIT=train slurm/fret_eval.sbatch
```

The task, reward and checks are described in [docs/rl-fretting.md](docs/rl-fretting.md). We only evaluate the held-out split once the training setup is frozen.

## Repository layout

| Path | Contents |
|---|---|
| `strum/ukulele.py` | Ukulele geometry, pitches and contact classification |
| `strum/fretting.py` | Command line for calibrating, verifying and viewing chord holds |
| `strum/planner.py` | Inverse kinematics planner for the fretting hand |
| `strum/transitions.py`, `strum/tasks.py` | Chord changes and routes |
| `strum/strumming.py`, `strum/bimanual.py` | Right-hand strumming, note detection and scoring |
| `strum/presentation.py` | Textures, lighting and cameras |
| `strum/export_chords.py` | Writes the chord table used in training |
| `strum/rl/` | The mjlab task: scene, observations, rewards, PPO settings |
| `scripts/` | Scene builders, checks, RL entry points and early diagnostics |
| `slurm/` | Job scripts for CARC |
| `assets/` | Seed hand placements and the chord table used in training |
| `manifests/` | The benchmark |
| `docs/` | Design notes and experiment write-ups |
| `tests/` | Unit tests |
| `references.md` | Reading list |

## Limitations

- The fretting forearm and the ukulele are fixed in space.
- Strings are rigid, so there is no string bending, plucking physics or audio.
- The right hand does not collide with the strings. Its notes come from geometric crossings.
- The scripted strummer only plays downstrokes at 70 BPM for now.
- The left hand is the upstream Shadow Hand with the two outer joints of each finger coupled, which we added to stop the fingers drifting.
- With four chords there are only two held-out chord changes, so held-out results on this benchmark rest on two cases.
- All results are from simulation without noise or disturbances.
