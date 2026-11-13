# Measured proof of concept

The bundled demo completes two laps without a collision in all nine deterministic
scenarios below, including the technical track excluded from training. These are
results for this simulator and physics configuration, not a Roblox transfer test.

The controller has **74 ternary parameters, 40 nonzero**, and the optimized export
contains **250 AND/OR/XOR/NOT gates**, excluding sensor blocks, wires and actuators.
The gate estimate assumes XOR is a primitive; decomposing it increases the count.

All nine episodes were also replayed using the actual Boolean gate interpreter,
matching the accelerated evaluation's progress, termination step and crash state.
All 216 possible nested-box sensor patterns match mathematical MLP inference.
The 12 unit/integration tests pass, including checkpoint restart, finite box width,
manual UI controls and headless rendering.

| Track | Start fraction | Direction | Offset (px) | Laps | Collision |
|---|---:|---:|---:|---:|---|
| oval | 0 | +1 | 0 | 2.0018 | No |
| oval | 0.25 | +1 | 12 | 2.0012 | No |
| oval | 0.5 | -1 | -12 | 2.0012 | No |
| harbor | 0 | +1 | 0 | 2.0004 | No |
| harbor | 0.25 | +1 | 12 | 2.0015 | No |
| harbor | 0.5 | -1 | -12 | 2.0015 | No |
| technical | 0 | +1 | 0 | 2.0008 | No |
| technical | 0.25 | +1 | 12 | 2.0015 | No |
| technical | 0.5 | -1 | -12 | 2.0014 | No |

## Reproduce

```sh
uv run boatbrain train --warm models/warm_start.json --population 48 --seed 7 --generations 30 --steps 1100 --out runs/reproduce
uv run boatbrain evaluate --model runs/reproduce/best.json --out runs/reproduce/evaluation.json
uv run boatbrain play --model models/demo.json
```

Training optimizes six scenarios: the three listed starts on oval and harbor.
The fixed saved warm start came from an earlier, narrower training experiment;
it is included so the final experiment can be repeated without that preliminary
run. The final run also includes 47 newly randomized genomes at initialization.
The technical track is evaluated only, and is not part of fitness selection.
The 1,100-step training horizon is shorter than the 1,800-step evaluation budget.
Episodes stop after reaching two laps, so small numerical overshoots are expected.

The full machine-readable results are in `models/evaluation.json`, per-generation
training metrics in `models/training_history.csv`, and model hash plus verification
metadata in `models/verification.json`. The live population checkpoint remains in
`runs/robust/checkpoint.json` in this workspace; runs are excluded from version control.

These tests establish a small reactive driving proof of concept. They do not test
sensor noise, asynchronous logic, arbitrary tracks, dynamic obstacles or Roblox
vehicle physics. Match the physical car's turning radius and sensor scale first.
