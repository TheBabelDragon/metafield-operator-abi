# Golden corpus

**On main**

- `MANIFEST.json` — suite list + frozen seeds
- `L2/*/golden.meta.json` — metrics + CG trajectories
- `L4/*/golden.meta.json` — same
- `test_goldens.py` — meta gates + tensor replay when `.npz` present

**Full tensors**

```bash
export PYTHONPATH="$(pwd)"
python scripts/generate_goldens.py
git add tests/operator/goldens/**/golden.npz
git commit -m "Wilson–Dirac v1 golden tensors"
```

Seeds are law.
