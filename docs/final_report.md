# Strum on Cue: final report (living draft)

CSCI 645, Fall 2026. Ekta Jichkar and Jithin Rajan.

This file collects everything we will need for the final report and the presentation. It starts from the proposal ([Final_Project_Proposal.pdf](../Final_Project_Proposal.pdf)) and records where the project has gone since, including work beyond the proposal.

How to use it:

- Only numbers that are recorded in the repo go here, each with a link to where it came from.
- Anything not done yet is marked **TODO**. Anything planned but not decided is marked **OPEN**.
- When something changes, add a dated line to the [changelog](#12-changelog) at the bottom.

---

## 0. Status at a glance

Last updated: 2026-11-21. Demo: December 3.

| Part | Status | Source |
|---|---|---|
| Ukulele model with parameters (frets, strings, press points) | Done | [strum/ukulele.py](../strum/ukulele.py) |
| Note detector (fret press, string crossing, F1 scoring) | Done | [strum/strumming.py](../strum/strumming.py) |
| Benchmark: 108 tasks, 81 train, 27 held out | Defined | [manifests/benchmark_v1.json](../manifests/benchmark_v1.json) |
| Scripted left hand (IK planner), all four chords | Done, 10/10 training changes | [ik-planner-coverage.md](experiments/ik-planner-coverage.md) |
| Scripted two-hand playing | Done, 240/240 notes (C–Am–C) | [geometry-ek-two-hand.md](experiments/geometry-ek-two-hand.md) |
| RL environment, left hand, chords only | Built and checked | [rl-fretting.md](rl-fretting.md) |
| RL Run 1 | Done: 80% of changes held, changes into G fail | [rl-fretting.md](rl-fretting.md) (run log on `jith`) |
| RL Run 2 (fix for G) | Code on `jith` only, **results TODO** | commit `5f8b85f` |
| Strum patterns P2, P3 and tempos 90, 110 | **TODO**: scripted strummer plays downstrokes at 70 BPM only | |
| Timing (tempo or deadline) in the RL task | **TODO** | |
| H1: event goals vs. symbolic goals | **TODO** | |
| H2: one policy for both hands | **TODO**, **OPEN** whether to keep it | |
| Held-out evaluation | **TODO**, deliberately not run until setup is frozen | |
| Human-like fretting grip (scripted player) | Done as an experiment: all four chords verified with physics, 10/10 training changes, 240/240 notes with both hands. Not yet adopted into `strum/` | [experiments/hand_pose/README.md](../experiments/hand_pose/README.md) |
| Swinging strumming hand, down and up strokes | Done as an experiment: 240/240 notes, mean onset error 0.36 ms | same |
| Upright scene, red concert-ukulele look | Done as an experiment, visual only; results identical to the flat scene | same |
| Video rendering | Works on Windows: 17 s, 1280x720, 30 fps clip (silent) | `outputs/demo/` (not in Git) |
| Sound from detected notes | **TODO** | |
| Report figures, slides | **TODO**. Demo stills and one clip exist. | |

---

## 1. Abstract

**TODO** once results exist. Draft shape:

> Dexterous robot hands are usually trained one task at a time. We ask whether a learned policy can reach a multi-finger contact configuration (a chord) by a deadline set from outside (the next strum), and whether it can do so for combinations of chord change, rhythm and tempo it never trained on. We build an open MuJoCo ukulele environment with two Shadow Hands, a benchmark of 108 tasks with a compositional held-out split, a scripted reference player, and RL policies for the fretting hand. [Results.] [Main finding.]

---

## 2. Introduction and research question

### 2.1 Motivation

From the proposal: robot music research is mostly piano (RoboPianist and follow-ups), and none of it tests **compositional generalization**, meaning whether a robot can recombine skills it already has, for example playing a familiar chord in a new rhythm. String instruments add a timing constraint the piano does not have: the chord must be pressed *before* the strum. A ukulele has four strings and a few common chords, which makes it easy to hold out specific combinations.

### 2.2 Question and hypotheses as proposed

**RQ:** Can one RL policy controlling both hands play chord, rhythm and tempo combinations it never trained on?

- **H1 (goal format).** Describing the goal as a timeline of upcoming note events generalizes better than describing it with labels (chord, pattern and tempo IDs). Prediction: with event goals, held-out F1 stays within 10 points of training F1; symbolic goals drop more.
- **H2 (two hands vs. one).** One policy for both hands beats an RL fretting hand plus a scripted IK strumming hand. Prediction: the joint policy more often finishes the chord before the strum, scores higher F1, and its advantage grows with tempo.

### 2.3 How the question has developed

The README now frames the core question as: **can a learned policy reach a specified multi-finger contact configuration by a deadline set from outside, and does that carry over to unseen combinations of target, deadline and coordination pattern?**

This is a narrower, more testable version of the proposal's RQ. It treats the right hand as the source of the deadline rather than as something to learn. In terms of the proposal, it is close to the H2 baseline (RL fretting plus a scripted strummer).

**OPEN:** decide with both of us whether the final report keeps H2 as a main result, a stretch goal, or a limitation. See [section 8](#8-beyond-the-proposal).

---

## 3. Related work

**TODO:** write up from [references.md](../references.md) and the proposal's reference list. Points to make:

- **Piano:** RoboPianist (two Shadow Hands in MuJoCo) and follow-ups: multi-song policies (PianoMime), scale (RP1M, Chen et al.), loudness (CANTABILE), diffusion policies (PANDORA), hardware (HandelBot). None test held-out combinations.
- **Guitar:** Luo et al. learn fretting with RL and script plucking with IK; our H2 baseline adapts this. Xu and Wang synchronize separately trained hands but do not test held-out tasks.
- **Drumming:** DexDrummer finds residual RL on a kinematic plan beats RL from scratch. This is relevant to us because we already have a working kinematic plan (the IK planner).
- **Compositional generalization:** Gao et al. (RSS 2024).
- **Tools:** MuJoCo, mjlab (GPU MuJoCo via MuJoCo Warp), mjlab_hand Shadow Hand task code, PPO.

---

## 4. Environment

### 4.1 The ukulele

- Concert scale, 381 mm. Re-entrant tuning G4 C4 E4 A4. Fret positions follow the equal-temperament formula `x_n = L (1 − 2^(−n/12))`, checked to 0.1 mm by unit tests ([tests/test_ukulele.py](../tests/test_ukulele.py)).
- Five frets are modelled for the current task. Strings are rigid capsules from nut to saddle.
- Press points sit just behind each fret ("behind_fret" rule, 3 mm offset).
- Collision bits keep the strumming hand from colliding with the strings; only the fretting hand touches them.
- Presentation-only geometry (body shape, headstock, tuners, inlays, textures) is added by [strum/presentation.py](../strum/presentation.py) and has collisions disabled.

### 4.2 The hands

- Two Shadow Hands from the mjlab_hand submodule ([external/mjlab_hand](../external/mjlab_hand)).
- **Left (fretting) hand:** forearm fixed in space. We added a J1 = J2 equality constraint to each finger. Without it, the free J1 − J2 mode drifted about 0.5°/s under a fixed command (30.6° over 60 s), and the C chord's contact force fell from 0.51 N to 0.019 N. With it, drift over 60 s is 0.11°. Results are for this modified hand, not the unchanged upstream model.
- **Right (strumming) hand:** fingers held curled except the index finger; the hand moves on a three-axis slide stage driven by position actuators. This is a Cartesian stage, not a robot arm.

### 4.3 How notes are detected

We do not simulate string vibration or sound.

- A string is **fretted** when a fingertip presses it between two frets with at least 0.01 N.
- A note **sounds** when the strumming fingertip crosses a string (2 mm height tolerance, 30 ms debounce, at least 0.05 m/s lateral speed).
- **Pitch** = open-string pitch + fret pressed.
- Detected notes are matched to expected notes by string, pitch, direction and time (50 ms tolerance). We report **F1**.
- A chord change is **held** if the target chord is correct for at least 95% of the final second before the next change, and **clean** if, in addition, there is no unwanted contact in that period.
- Fretboard rule (adopted 2026-11-09): an assigned fingertip may touch the fretboard inside its own fret. Results before that date used a stricter rule.

### 4.4 Figures to make

- **TODO** Figure: the scene, overview camera, polished visuals.
- **TODO** Figure: close-up of the fretting hand on each of the four chords.
- **TODO** Figure: diagram of note detection (fret cell, press force, strum crossing).

---

## 5. Benchmark

- Four chords: C, G, Am, F. Twelve directed chord changes.
- Three strum patterns on an eighth-note grid: P1 `D . D . D . D .`, P2 `D U D U D U D U`, P3 `D . D U . U D U`.
- Three tempos: 70, 90, 110 BPM.
- One task = one chord change × one pattern × one tempo, so 12 × 3 × 3 = **108 tasks**.

**Compositional split:** 81 train, 27 held out.

- 18 held-out tasks are the chord changes **Am→G** and **F→C**, which never appear in training.
- 9 held-out tasks pair **P3 with a change into F** (from C, G or Am).
- Every chord, pattern and tempo appears in training. Only these combinations are withheld.

**Matched comparison.** Both held-out changes are of the hardest kind (three fingers move). Each has a training change of identical structure that ends on the same chord: F→C matches G→C, and Am→G matches F→G. Held-out results are reported next to these, so a drop can be attributed to novelty rather than difficulty.

**Known weakness:** with four chords there are only two held-out chord changes.

**TODO** Figure: the 12 × 9 split grid (proposal Fig. 3, updated).

---

## 6. Scripted reference player

The scripted player is the reference every learned result is compared against. It also proves every training change is physically reachable.

**How it works:** per-finger IK gives a press pose and a hover pose 10 mm above it. A chord change is lift, travel at hover height, press. Fingers that keep the same note stay down. Two fingers sharing a fret take opposite ends of it, which is what makes G reachable. Commands include gravity feedforward. One press depth (−0.75 mm) works for all four chords.

### Results (native MuJoCo 3.1.4, CPU)

| Check | Result | Source |
|---|---|---|
| Hold each chord for 60 s | All four, 100% contact | [ik-planner-coverage.md](experiments/ik-planner-coverage.md) |
| Max joint drift over 60 s | 0.11° (was 30.6° before the coupling) | same |
| Chord reached from rest | 4/4 | same |
| Ten training chord changes, 2.0 s each | 10/10 | same |
| Same at 1.0, 0.6, 0.4 s each | 10/10 | same |
| Same at 0.2 s each | 7/10 (Am→F, C→G, F→G fail) | same |
| C→G→F→Am→C, 10 cycles, one rollout | 10/10 | same |
| Two hands, C–Am–C, 10 cycles | 240/240 notes, F1 = 1.0, 3.77 ms mean onset error | [geometry-ek-two-hand.md](experiments/geometry-ek-two-hand.md) |
| Two hands, C→G→F→Am→C, 1 cycle | 48/48 notes | [ik-planner-coverage.md](experiments/ik-planner-coverage.md) |

Limits: open-loop (nothing reacts to measured contact), fixed forearm and instrument, no noise or perturbations, downstrokes at 70 BPM only. The two held-out changes were not run.

**TODO** Run the scripted player on all 81 training tasks once the strummer supports P2, P3, 90 and 110 BPM. This gives the reference column for every results table.

### 6.1 Human-like grip (experiment, 2026-11-19 to 11-21)

The fretting hand above comes down onto the strings from above, palm down. In an experiment kept separate from `strum/`, we moved it to a human-like grip: forearm level and under the neck, palm up, fingers curling up and over onto the fretboard. The strumming hand was rebuilt to lie almost parallel to the ukulele, palm toward the strings, swinging from an elbow pivot so its curled fingers brush the strings down and up. The whole scene was then turned upright for presentation.

| Check (native MuJoCo 3.16, scripted player) | Result |
|---|---|
| All four chords reached from rest and held 60 s | 4/4 |
| Ten training chord changes | 10/10 |
| Two hands, C–Am–C, 10 cycles, down-up-down strokes | 240/240 notes, F1 = 1.0, mean onset error 0.36 ms |
| Same, with the scene turned upright | identical to 13 digits |

What it took, each found by a failed check:

- **A forearm pose found by search.** Yaw −8°, pitch −4.5°, roll +14°, knuckles 29 mm from the nut, 60 mm below and 28 mm in front of the fretboard.
- **The fingertip marker moved 3.0 mm.** The planner aims one point on each fingertip. Measured for the old pose, it made every finger land one fret too high in the new grip.
- **Idle fingers move out of the way.** Otherwise the middle finger, reaching for the far G string at its sideways joint limit, hits the resting ring finger.
- **The standard G fingering** (index C2, middle A2, ring E3) instead of the project's (middle C2, index A2). With the old fingering, the index finger rolled over the fret-2 wire.

Adopting the grip into the project would change the planner (idle fingers) and the chord table (G fingering), so it belongs in the single table freeze (section 8). The full log of 23 steps, including two scripting bugs found by looking at pictures, is in [experiments/hand_pose/README.md](../experiments/hand_pose/README.md).

---

## 7. RL for the fretting hand

### 7.1 Task (stage 1: chords only)

- **Policy:** PPO, controls the 12 finger actuators of the left hand (splay, proximal and coupled distal for four fingers) at 50 Hz. Wrist, thumb and little-finger base hold their rest command.
- **Episode:** 9 s. Starts holding one chord; every 2.5–3.5 s the target changes.
- **Actor observations:** joint angles and speeds, fingertip positions, previous action, target chord as the fret per string (one-hot over frets 0–3 for each of four strings). The policy is never told which finger goes where.
- **Critic** also sees fingertip-to-target errors, contact forces and timers.
- **Simulator:** mjlab with MuJoCo Warp at a 4 ms step (the scripted evaluation uses native MuJoCo at 2 ms).
- Training uses 10 chord changes; Am→G and F→C are held out.

### 7.2 Reward

| Term | Weight | Meaning |
|---|---:|---|
| reach | 1.0 | assigned fingertips near their fretted position (4 cm scale) |
| reach_fine | 1.0 | same, 6 mm scale |
| clear | 0.5 | unused fingertips at least 15 mm above the fretboard |
| contacts | 1.0 | fraction of assigned fingers on the right string and fret |
| chord_ok | 3.0 | whole chord correct, no unwanted contact |
| unwanted | −1.0 | unwanted contact force |
| action_rate | −0.01 | change in action between steps |

### 7.3 Checks before training

- Contact bookkeeping self-check: five checks, all pass; two of them fail when the pre-review bugs are restored, so they detect what they should.
- Replaying the scripted planner through the RL action interface: held 100%, clean 100% on all ten training changes.
- An outside code review found three real faults in the evaluation path; all were fixed before Run 1 (see [rl-fretting.md](rl-fretting.md#review-fixes-2026-11-10)).

### 7.4 Run 1 "pilot" (2026-11-10)

4096 environments, 3000 iterations, 1 h 46 min on one A40 (about 46,000 steps/s). Evaluation without exploration noise, 256 rollouts per change:

| Change | Held | Clean | Median time to chord |
|---|---:|---:|---:|
| Am→C | 1.000 | 0.863 | 0.08 s |
| Am→F | 1.000 | 0.992 | 0.08 s |
| C→Am | 1.000 | 0.863 | 0.08 s |
| C→F | 0.988 | 0.840 | 0.10 s |
| **C→G** | **0.000** | **0.000** | never |
| F→Am | 1.000 | 0.988 | 0.02 s |
| **F→G** | **0.000** | **0.000** | never |
| G→Am | 1.000 | 0.863 | 0.02 s |
| G→C | 1.000 | 0.828 | 0.08 s |
| G→F | 0.992 | 0.621 | 0.06 s |
| **All** | **0.798** | **0.686** | |

**Reading:**

- Every change that does not end in G is held essentially every time. Both changes into G always fail. G is the only three-finger chord and the only one where two fingers share a fret.
- The scripted planner does all ten changes, so into-G is a **learning** problem, not a reachability problem.
- Times to chord are tens of milliseconds because the model's finger actuators are very strong for links of a few grams. This is a property of the simulated hand, not of real hands, and it explains why "clean" lags "held": fast moves clip a neighbouring string or finger.
- Note: F→G is the matched training change for held-out Am→G. Until G works, the matched comparison cannot be made.

### 7.5 Run 2 (set up 2026-11-12)

Changes: (1) hold periods, where with probability 0.25 the new target equals the current chord, so the policy experiences holding a correct G; (2) exploration noise capped to [0.02, 0.5] and entropy coefficient lowered from 0.003 to 0.001; (3) extra evaluation diagnostics per finger. Two changes at once, so Run 2 cannot say which one mattered.

**TODO:** results. The code is on `jith` (commit `5f8b85f`), not yet on `main`.

### 7.6 Figures to make

- **TODO** Training curves from W&B (`transition_success`, `clean_success`, `chord_ok`).
- **TODO** Per-change bar chart: held and clean, RL vs. scripted.
- **TODO** Short clip: RL policy failing into G (Run 1) vs. succeeding (if Run 2 fixes it).

---

## 8. Beyond the proposal

Candidate directions, not yet committed. Each says what it would show and what it costs. **OPEN:** pick which ones to do.

| Idea | What it would show | Cost | Status |
|---|---|---|---|
| **Deadline / tempo in the RL task** | The "on cue" part: is the chord ready before the strum? Measured as ready margin. | Medium | **TODO**, needed for any timing claim |
| **Advance notice test** | At test time, tell the policy the next chord later and later; find when it stops being ready in time. | Small once timing exists | Planned in README |
| **Instrument pose shifts** | Shift and tilt the ukulele slightly at test time (not observed by the policy). Robustness. | Small to medium | Planned in README |
| **Effort and smoothness** | Actuator effort and fingertip jerk per change, RL vs. scripted. | Small | Planned in README |
| **Residual RL on the IK plan** | Following DexDrummer: the policy corrects the scripted planner instead of learning from scratch. Could fix into-G and be a clean comparison. | Medium | Idea |
| **Curriculum for G** | Train on easy changes first, then add G. Alternative fix for into-G. | Small | Idea |
| **H1 goal formats** | Event timeline vs. symbolic IDs (proposal). | Medium, mostly GPU time | **TODO** |
| **H2 joint two-hand policy** | One policy for both hands (proposal). | Large, risky | **OPEN** |
| **Fallback: recovery policy** | From the proposal appendix: learn to retry, delay or skip after fretting failures. | Medium | Only if H1/H2 stall |
| **Human-like fretting grip** | The hand holds the neck as a player does. A more realistic contact problem, and the look the demo needs. | Done in the experiment; adoption needs a planner change and the standard G fingering | **Done (experiment)**, see 6.1 |
| **Swinging strumming hand, down and up strokes** | Upstrokes as real notes, from a forearm swing instead of a sliding stage | Done in the experiment | **Done (experiment)** |
| **Upright scene, concert-ukulele look** | Presentation only. Gravity is turned with the scene, so physics and results are unchanged. | Done in the experiment | **Done (experiment)** |
| **Single table freeze** | Collect every chord-table change (finger speed cap, six more chords, human-like grip, barre chords if they pass) into one table, so the policy is retrained once before the demo, not twice | Planning | **OPEN** |

**Predictions first.** Before any held-out evaluation, we commit our predictions to `docs/`, so the git history shows they came first.

---

## 9. Results

**TODO.** Planned tables:

1. Training vs. held-out, per chord change, next to the matched training change and the scripted reference (held, clean, ready margin, F1).
2. H1: event vs. symbolic goals, training vs. held-out F1.
3. Advance-notice curve: success vs. how early the next chord is announced.
4. Pose-shift robustness.

---

## 10. Limitations

- The fretting forearm and the ukulele are fixed in space. This models a supported-instrument robot, not a human grip.
- Strings are rigid: no bending, plucking physics or audio.
- The right hand does not collide with the strings; its notes come from geometric crossings.
- The project's scripted strummer plays only downstrokes at 70 BPM so far. The experimental swinging hand adds upstrokes, still at 70 BPM.
- In the upright presentation scene, gravity is turned with the instrument so the physics matches calibration. It points into the back of the ukulele, not toward the floor.
- The left hand is the upstream Shadow Hand plus our J1 = J2 coupling.
- Finger actuators are strong for their link masses, so movement times are unrealistically short.
- Only two held-out chord changes.
- RL training (MuJoCo Warp, 4 ms) and scripted evaluation (native MuJoCo, 2 ms) use different simulators and steps. Final RL numbers should be confirmed by replaying in native MuJoCo.
- Simulation only, no noise or disturbances.

---

## 11. Presentation plan

Demo on **December 3**. **TODO:** length and format.

Draft slide list:

1. Title, one-sentence question, demo clip.
2. Why string instruments: the chord must be ready before the strum.
3. Compositional generalization: the 12 × 9 split grid.
4. The environment: ukulele, two hands, note detection.
5. Scripted reference: what it can do (table from section 6).
6. RL task: observations, actions, reward.
7. Run 1: works except G, and why.
8. Fix for G (Run 2 or later).
9. Timing results (ready margin, advance notice).
10. Held-out results vs. matched training changes.
11. Limitations and what we would do next.

Assets to produce: split grid, training curves, results bar charts, sound for the clips. Already made (in `outputs/demo/`, not in Git): stills of each chord mid-strum and the ukulele alone, and a 17 s C–Am–C clip of the upright scene, all from `experiments/hand_pose/render_demo.py`.

---

## 12. Changelog

- 2026-10-21: repository created.
- 2026-10-22: proposal-stage setup; first C-chord hold on CARC (job 13300717); scripted strumming and note scoring.
- 2026-11-02: first merge of `jith` into `main` (PR #1).
- 2026-11-09: parameterized ukulele, J1 = J2 coupling, IK planner, all four chords and ten training changes pass; fretboard contact rule relaxed.
- 2026-11-10: RL environment built; outside review found three evaluation faults, all fixed; Run 1 trained.
- 2026-11-12: Run 2 changes committed on `jith`.
- 2026-11-19: `main` merged into `ek`; this report started. Human-like grip experiment begun.
- 2026-11-21: human-like grip verified with physics (4/4 chords, 10/10 changes, 240/240 notes); swinging strumming hand with upstrokes; upright scene and concert-ukulele look; first video rendered. Committed as `experiments/hand_pose/` (4184ac5).

---

## Appendix A: reproducing the numbers

Run from the repository root. Each run needs a new output folder.

```bash
# Unit tests
python -m unittest discover -s tests -t . -v

# Scripted player
python scripts/check_geometry_integration.py
python -m strum.fretting calibrate --placements assets/fretting_seed_v1 --output outputs/holds
python -m strum.fretting coverage --run outputs/holds --output outputs/coverage
python -m strum.fretting transition --run outputs/holds --cycles 10 --two-hands --output outputs/two-hand

# RL on CARC
sbatch slurm/fret_train.sbatch
sbatch --export=ALL,CHECKPOINT=<path to model_N.pt>,SPLIT=train slurm/fret_eval.sbatch
```

Do not run `SPLIT=heldout` until rewards, observations and training length are frozen.
