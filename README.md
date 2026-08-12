# metafield-operator-abi

Wilson–Dirac ABI v1 workbench.

## Use this (no pull, no stash, no commit)

```bash
git clone https://github.com/TheBabelDragon/metafield-operator-abi.git
cd metafield-operator-abi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
python -m pytest tests/operator -q
```

Optional full tensors:

```bash
python scripts/generate_goldens.py
python -m pytest tests/operator -q
```

Do **not** `git pull` into an old dirty folder. Clone clean.
