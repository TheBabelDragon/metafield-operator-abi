"""Constitution tests — meta gates + tensor replay."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backends.reference.torch_backend import TorchReferenceBackend
from metafield.lattice.types import LatticeGeometry, PrecisionPolicy
from metafield.operators.protocol import WilsonParams

GOLDEN_ROOT = Path(__file__).parent / "goldens"
TOL = json.loads((Path(__file__).parent / "tolerances.json").read_text())["complex128"]


def _suite_dirs():
    man = GOLDEN_ROOT / "MANIFEST.json"
    if not man.exists():
        return []
    manifest = json.loads(man.read_text())
    return [
        GOLDEN_ROOT / s["path"]
        for s in manifest["suites"]
        if (GOLDEN_ROOT / s["path"] / "golden.meta.json").exists()
    ]


def _load_array(d: Path, key: str):
    b64p = d / "arrays" / f"{key}.npz.b64"
    if not b64p.exists():
        return None
    try:
        raw = base64.b64decode(b64p.read_text().encode("ascii"))
        z = np.load(__import__("io").BytesIO(raw))
        return z["data"]
    except Exception:
        return None


def _rel_err(got, exp):
    num = torch.linalg.vector_norm((got - exp).reshape(-1))
    den = torch.linalg.vector_norm(exp.reshape(-1)).clamp_min(1e-30)
    return float(num / den)


def _tc(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.complex128)


@pytest.mark.parametrize(
    "d",
    _suite_dirs() or [None],
    ids=lambda p: (f"{p.parent.name}/{p.name}" if p else "none"),
)
def test_meta_gates(d):
    if d is None:
        pytest.skip("no meta")
    meta = json.loads((d / "golden.meta.json").read_text())
    assert meta["abi"] == "wilson_dirac_v1"
    m = meta["metrics"]
    assert m["g5_hermiticity_err"] < TOL["Ddag_identity_abs"]
    if "Q_hermiticity_err" in m:
        assert m["Q_hermiticity_err"] < TOL["Q_hermitian_abs"]
    assert m.get("cg_final_resid") is not None
    # trajectory optional on compact hydra metas
    if "cg_residual_trajectory" in m:
        assert m["cg_residual_trajectory"]
    assert m.get("cg_iters", 0) > 0


@pytest.mark.parametrize(
    "d",
    _suite_dirs() or [None],
    ids=lambda p: (f"{p.parent.name}/{p.name}" if p else "none"),
)
def test_tensor_replay(d):
    if d is None:
        pytest.skip("no suites")
    meta = json.loads((d / "golden.meta.json").read_text())
    # Hydra heads are meta-only on remote
    if meta.get("hydra_head") or meta["L"] > 4:
        pytest.skip("hydra head — meta-only; run generate_goldens.py for tensors")
    p = meta["params"]
    params = WilsonParams(
        mass=p["mass"], wilson_r=p["wilson_r"],
        color_dim=p["color_dim"], spinor_dim=p["spinor_dim"],
    )
    be = TorchReferenceBackend(
        LatticeGeometry(L=meta["L"], n_dims=p["n_dims"]), PrecisionPolicy()
    )
    psi_np = _load_array(d, "inputs__psi")
    dpsi_np = _load_array(d, "outputs__Dpsi")
    if psi_np is None or dpsi_np is None:
        pytest.skip("no psi/Dpsi tensors on remote")
    if meta["kind"] in ("cold", "boundary"):
        U = be.unit_gauge(params)
    else:
        U_np = _load_array(d, "inputs__U")
        if U_np is None:
            pytest.skip("no U tensor")
        U = _tc(U_np)
    psi = _tc(psi_np)
    assert _rel_err(be.wilson_dirac(psi, U, params), _tc(dpsi_np)) < TOL["D_on_noise_rel"]
