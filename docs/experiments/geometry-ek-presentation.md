# Geometry-aware presentation and repeated C–Am–C transitions

The `--polished` option adds decorative body meshes, koa texture, binding, headstock, nut, bridge, saddle, inlays, lighting, and two camera presets. Neck width and nut/saddle placement use the selected geometry profile. This remains a fixed instrument with a fixed-mount fretting hand. Decorative additions have zero mass and disabled collisions. The soundhole is decorative; no acoustics is simulated.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m strum.fretting transition \
  --run outputs/ek-holds-03 --cycles 10 --polished \
  --output outputs/ek-polished-10

mjpython -m strum.fretting transition \
  --run outputs/ek-holds-03 --cycles 1 --view --polished \
  --camera overview --output outputs/ek-polished-preview
```

For a close-up, use `--camera fretting`. Omitting `--polished` retains diagnostic targets and the simple body. Both use the same contact evaluation and commands. The wood texture remains at `scripts/visual_assets/koa_grain.png`.

## Checks

On native MuJoCo 3.1.4, the styled three-cycle run passed. All 45,000 contact-presence and force trace samples matched the plain three-cycle run exactly. Checked body masses/inertias, actuator gains/biases, damping, initial states, and original geometry collision masks, friction, solver settings and sizes were preserved. A rendered preview was inspected.

The controller is scripted, uses physical actuator-driven homing, and resets the running simulation only once. A cycle takes 29 simulated seconds, with a 3-second initial hold. Contact gaps are allowed during moves; hold checks discard the first 0.5 seconds. These are provisional diagnostic thresholds, not musical-tempo, robustness, generalization, or learned-control results.

The second hand is not yet part of this new-geometry transition workflow. The existing two-hand demonstration remains a separate baseline.

A subsequent 10-cycle styled run also passed (293 simulated seconds, one running-state reset). All measured Am and returned-C hold windows had full required contact; evaluated unwanted-contact fractions were zero. This is a longer nominal repeat, not a randomized robustness test.
