# metafield-operator-abi

**This is the ABI workbench.** Named so it does not collide with an existing local `metafield-work/`.

**MetaField defines the mathematics. Backends implement the operators.**

Wilson–Dirac is the first frozen instruction in the MetaField operator language.

```
MetaField Operator ABI family
│
├── Wilson–Dirac ABI v1     🔒 IMMUTABLE
├── Reduction ABI             later
├── Plaquette ABI             later
└── Gauge-force ABI           later
```

See [`docs/FOUNDATION.md`](docs/FOUNDATION.md).

## Quick start

```bash
git clone https://github.com/TheBabelDragon/metafield-operator-abi.git
cd metafield-operator-abi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/generate_goldens.py
PYTHONPATH=. python -m pytest tests/operator -q
```

## Constitution

`tests/operator/goldens/` is the compliance boundary.

| Check | Gate |
|-------|------|
| `Dψ` | relative error vs oracle |
| γ₅-hermiticity | residual |
| `Q = D†D` hermiticity | residual |
| CG trajectory | residual history |

## Frozen vs experimental

**Frozen:** Wilson math, layouts, γ matrices, seeds, golden requirements, PyTorch oracle.

**Experimental:** `OperatorBackend` surface, DMA, device handles, FPGA/ASIC transport.

## Next (and only next)

```
wilson_dirac(ψ, U) → Dψ on device
L2 cold → random → boundary → L4 → profile
```

Oracle lineage: [TheBabelDragon/metafield](https://github.com/TheBabelDragon/metafield).

Former mistaken name on GitHub: `metafield-work` (kept as redirect target / duplicate until you delete it).
