"""Constitution tests — meta gates + tensor replay + live seed oracle."""
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


def _make_boundary_psi(be, params):
    psi = torch.zeros(
        be.geometry.shape + (params.spinor_dim, params.color_dim), dtype=be.dtype
    )
    origin = (0,) * be.geometry.n_dims
    psi[origin + (0, 0)] = 1.0 + 0.3j
    psi[origin + (1, 1)] = -0.2 + 0.5j
    return psi


def _make_random_U(be, params, seed: int):
    g = torch.Generator().manual_seed(seed)
    U0 = be.unit_gauge(params)
    shape = U0.shape
    n = params.color_dim
    real = torch.randn(shape, generator=g, dtype=torch.float64)
    imag = torch.randn(shape, generator=g, dtype=torch.float64)
    A = (real + 1j * imag).to(be.dtype) * 0.05
    H = 0.5 * (A + A.conj().transpose(-1, -2))
    tr = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n, dtype=be.dtype)
    H = H - (tr / n)[..., None, None] * eye
    X = 1j * H
    HH = 1j * X
    evals, evecs = torch.linalg.eigh(HH)
    phase = torch.exp(-1j * evals.to(be.dtype))
    Vh = evecs.conj().transpose(-1, -2)
    return (evecs @ (phase[..., :, None] * Vh)) @ U0


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
    if "cg_residual_trajectory" in m:
        assert m["cg_residual_trajectory"]
    assert m.get("cg_iters", 0) > 0


@pytest.mark.parametrize(
    "d",
    _suite_dirs() or [None],
    ids=lambda p: (f"{p.parent.name}/{p.name}" if p else "none"),
)
def test_oracle_compliance(d):
    """Tensor replay when present; otherwise live seed-locked oracle check."""
    if d is None:
        pytest.skip("no suites")
    meta = json.loads((d / "golden.meta.json").read_text())
    p = meta["params"]
    params = WilsonParams(
        mass=p["mass"],
        wilson_r=p["wilson_r"],
        color_dim=p["color_dim"],
        spinor_dim=p["spinor_dim"],
    )
    L = meta["L"]
    kind = meta["kind"]
    seed = meta["seed_base"]
    be = TorchReferenceBackend(
        LatticeGeometry(L=L, n_dims=p["n_dims"]), PrecisionPolicy()
    )

    # Path A — committed tensors
    psi_np = _load_array(d, "inputs__psi")
    dpsi_np = _load_array(d, "outputs__Dpsi")
    if psi_np is not None and dpsi_np is not None:
        if kind in ("cold", "boundary"):
            U = be.unit_gauge(params)
        else:
            U_np = _load_array(d, "inputs__U")
            U = _tc(U_np) if U_np is not None else _make_random_U(be, params, seed + 1)
        psi = _tc(psi_np)
        assert _rel_err(be.wilson_dirac(psi, U, params), _tc(dpsi_np)) < TOL["D_on_noise_rel"]
        return

    # Path B — live seed-locked oracle
    g = torch.Generator().manual_seed(seed)
    if kind == "cold":
        U = be.unit_gauge(params)
        psi = be.random_fermion(params, g)
    elif kind == "boundary":
        U = be.unit_gauge(params)
        psi = _make_boundary_psi(be, params)
    elif kind == "random":
        U = _make_random_U(be, params, seed + 1)
        psi = be.random_fermion(params, g)
    else:
        pytest.fail(f"unknown kind {kind}")

    Dpsi = be.wilson_dirac(psi, U, params)
    Ddag = be.wilson_dirac_dagger(psi, U, params)

    g5psi = torch.einsum("st,...ti->...si", be.g5, psi)
    manual = torch.einsum("st,...ti->...si", be.g5, be.wilson_dirac(g5psi, U, params))
    assert float(be.complex_norm(Ddag - manual)) < TOL["Ddag_identity_abs"]

    # Compact hydra meta rounds norms — allow 1e-3 relative
    if "norm_Dpsi" in meta["metrics"] and meta["metrics"]["norm_Dpsi"]:
        got = float(be.complex_norm(Dpsi))
        exp = float(meta["metrics"]["norm_Dpsi"])
        assert abs(got - exp) / max(exp, 1e-30) < 1e-3

    g2 = torch.Generator().manual_seed(seed + 99)
    phi = be.random_fermion(params, g2)
    Qpsi = be.normal_operator(psi, U, params)
    Qphi = be.normal_operator(phi, U, params)
    q_err = abs(complex(be.complex_dot(phi, Qpsi)) - complex(be.complex_dot(Qphi, psi)))
    assert q_err < TOL["Q_hermitian_abs"]

    if kind == "random" and L >= 6:
        return
    if L >= 10:
        return

    from metafield.algorithms.cg import cg_solve

    g3 = torch.Generator().manual_seed(seed + 123)
    eta = be.random_fermion(params, g3)
    b = be.wilson_dirac_dagger(eta, U, params)

    def matvec(v):
        return be.normal_operator(v, U, params)

    _x, iters, resid = cg_solve(
        matvec, b, dot=be.complex_dot, norm=be.complex_norm,
        tol=1e-8, maxiter=min(200, int(meta["metrics"].get("cg_iters", 80) * 3 + 20)),
    )
    assert resid < 1e-7
    assert iters > 0
