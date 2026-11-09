# Two-hand C–Am–C on parameterized ek geometry

The existing three-axis position-actuated right Shadow Hand stage is integrated into the geometry-aware transition runner. The left forearm and instrument remain fixed. This models a supported-instrument robotic task, not a human grip, complete robot arm, or unsupported ukulele.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m strum.fretting transition \
  --run outputs/ek-holds-03 --cycles 10 --two-hands --polished \
  --output outputs/ek-two-hand-10
```

Use `mjpython` and add `--view --camera overview` to watch a one-cycle demonstration. Use a fresh output directory each time.

## Motion and scoring

Each cycle is C → Am → C with a scripted clearance/homing sequence. Three downstrokes are scheduled during each Am and returned-C hold, at 70 BPM within that hold. Chord transitions remain slow; this is not a continuous 70 BPM performance. There is one reset of the running simulation, with no subsequent joint-state teleportation.

String positions and heights come from the same parameterized geometry as the scene. Expected events come from the command schedule. Actual events come from swept physical fingertip motion in the instrument frame, without readiness gating. Fret pitches come from measured left-hand/string contact forces and geometric fret classification. Scoring matches string, pitch, direction and time (50 ms tolerance). Duplicate, missing, wrong-pitch, wrong-direction, late and perfect event cases were checked.

## Simplifications

- Right-hand/string collisions are disabled; fingertip crossings create symbolic note events. This is not string deflection, realistic plucking, or acoustics.
- Other collisions are retained and screened, including contacts involving the right hand.
- Right-hand gravity compensation, a three-axis position stage, and the implicitfast integrator match the existing two-hand implementation.
- Contact threshold 0.01 N, crossing-height tolerance 2 mm, debounce 30 ms, and minimum lateral speed 0.05 m/s are provisional.
- These are nominal scripted engineering checks, not RL, generalization, robustness or hardware-transfer results.

The pose should remain fixed until it demonstrably restricts required transitions or a research objective explicitly requires human-like playing. A new thumb-behind-neck grip would need separate reachability, contact, transition and timing validation.

## Observed native MuJoCo 3.1.4 results

10/10 transition cycles passed over 293 simulated seconds. Symbolic notes: {'tp': 240, 'fp': 0, 'fn': 0, 'f1': 1.0, 'matched_mean_absolute_onset_error_ms': 3.7722886239556073}. No unexpected upstrokes were detected. Companion JSON records the per-phase results and limitations.
