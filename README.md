# metafield-operator-abi

Wilson–Dirac ABI v1 workbench. **Does not collide with local `metafield-work/`.**

## Run (from repo root only)

```bash
git clone https://github.com/TheBabelDragon/metafield-operator-abi.git
cd metafield-operator-abi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# PYTHONPATH must be repo root (the directory that contains scripts/ and metafield/)
export PYTHONPATH="$(pwd)"

python scripts/generate_goldens.py
python -m pytest tests/operator -q
```

If you see `No such file or directory: .../scripts/generate_goldens.py`:

1. You are not in the repo root — `cd` into `metafield-operator-abi` first
2. Or you have a stale clone — `git pull` then confirm:

```bash
ls scripts/generate_goldens.py
ls backends/reference/torch_backend.py
```

## Path order

```text
metafield-operator-abi/          ← cwd and PYTHONPATH=.
├── scripts/generate_goldens.py
├── backends/reference/torch_backend.py
├── metafield/
└── tests/operator/
```

Do **not** run `python scripts/generate_goldens.py` from outside this tree.
