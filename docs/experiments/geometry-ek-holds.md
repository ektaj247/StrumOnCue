# Parameterized ek geometry: initialized fretting holds

Local validation: 2026-11-09, MuJoCo 3.1.4, native Mac CPU.

## Reproduce from repository root

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m strum.fretting calibrate \
  --placements assets/fretting_seed_v1 --profile ek \
  --output outputs/ek-holds-01
```

The geometry builder is `scripts/build_geometry_scene.py`; instrument parameters and geometric classification are in `strum/ukulele.py`. Calibration, fixed-command hold evaluation, independent verification, and live playback share `strum/fretting.py`.

The versioned seed assets retain the original calibrated fixed mount and hand model. They are initialization data, not validated ek poses. The pipeline replaces instrument geometry, fits approximate distal-pad markers to the new targets, and searches a bounded grid of position-control offsets. Each trial resets to initialized contact. Search horizons are 10 seconds; passing candidates undergo a 60-second hold. This verification is part of nominal calibration, not held-out generalization testing.

## Observed results

| Chord | Simultaneous contact fraction after settling | Unwanted contact fraction |
|---|---:|---:|
| C | 1.000000 | 0 |
| Am | 1.000000 | 0 |
| F | 1.000000 | 0 |
| G | 0.993815 | 0 |

All four passed the provisional contact screen. G is not continuously correct throughout the measurement period. Mean forces and control offsets are recorded in the companion JSON. A released-finger negative control failed as expected (zero C-target contact fraction).

Success requires all assigned finger/string contacts within their intended fret regions to exceed 0.01 N for at least 95% of samples after 0.5 seconds settling; total unwanted hand-contact force must stay below 0.01 N during that measurement window. Neck contact is still counted as unwanted. Completed finite rollouts without MuJoCo warnings are required. Thresholds are provisional, not validated musical-force criteria.

## Limits

- Initialized holds only: no approach, transitions, second hand, or learned policy in this milestone.
- Fixed forearm, rigid strings, visual-only frets, original body proxy.
- Approximate pad marker and equal initial distal angles; actual dynamics determines contact-screen acceptance.
- Joint drift reaches roughly 30.6 degrees over 60 seconds. Contact-screen success does not establish mechanical equilibrium or a stable posture indefinitely.
- Playback continues beyond the measured duration until closed; that extra duration is not validated by these results.
- No CARC reproduction, random perturbations, audio, or held-out tests have been completed here.
- The existing two-hand demo still uses its original geometry. Do not feed these scenes into its legacy scoring code.

## Active workflow

```bash
python scripts/check_geometry_integration.py
python -m strum.fretting calibrate --placements assets/fretting_seed_v1 --output outputs/ek-holds-01
python -m strum.fretting verify --run outputs/ek-holds-01 --output outputs/ek-holds-check
mjpython -m strum.fretting view --scene outputs/ek-holds-01/G/scene.xml
```

Use a fresh output directory for every calibration/verification. Retain summaries and CSVs for failed runs too. The next milestone is geometry-aware, single-rollout C–Am–C transitions, followed by four-chord transitions and two-hand integration.
