# Hydra heads (L = 6, 8, 10)

Scaling probes on top of the frozen L2/L4 constitution.

| L | volume | cold CG iters | boundary CG iters |
|---|--------|---------------|-------------------|
| 2 | 16 | ~5 | ~5 |
| 4 | 256 | ~16 | ~16 |
| 6 | 1296 | ~34 | ~34 |
| 8 | 4096 | ~64 | ~64 |
| 10 | 10000 | ~82 | ~82 |

## Seeds (frozen)

- L6 cold 3001 / random 3002 / boundary 3003
- L8 cold 4001 / boundary 4003
- L10 cold 5001 / boundary 5003

Random only at L6 (condition-number probe). L8/L10 stay cold+boundary.

## Policy

- Meta (`golden.meta.json`) is the compliance surface for hydra heads.
- Full tensors stay local (`python scripts/generate_goldens.py`).
- Hydra does **not** change Wilson–Dirac ABI v1.

## Why

First hardware path still targets L2. Hydra answers: how CG cost grows before you pick a datapath.
