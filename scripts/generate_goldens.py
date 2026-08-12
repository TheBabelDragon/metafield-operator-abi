#!/usr/bin/env python3
"""Regenerate Wilson–Dirac ABI v1 golden corpus from the PyTorch oracle.

  PYTHONPATH=. python scripts/generate_goldens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.reference.torch_backend import TorchReferenceBackend
from metafield.lattice.types import LatticeGeometry, PrecisionPolicy
from metafield.operators.protocol import WilsonParams

GOLDEN_ROOT = ROOT / "tests" / "operator" / "goldens"
PARAMS = WilsonParams(mass=0.1, wilson_r=1.0, color_dim=3, spinor_dim=4)
BETA = 5.5
SEEDS = {
    (2, "cold"): 1001,
    (2, "random"): 1002,
    (2, "boundary"): 1003,
    (4, "cold"): 2001,
    (4, "random"): 2002,
    (4, "boundary"): 2003,
}


def make_cold_U(be, params):
    return be.unit_gauge(params)


def make_random_U(be, params, seed: int):
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


def make_boundary_psi(be, params):
    psi = torch.zeros(
        be.geometry.shape + (params.spinor_dim, params.color_dim), dtype=be.dtype
    )
    origin = (0,) * be.geometry.n_dims
    psi[origin + (0, 0)] = 1.0 + 0.3j
    psi[origin + (1, 1)] = -0.2 + 0.5j
    return psi


def suite_for(L: int, kind: str, seed_base: int) -> None:
    be = TorchReferenceBackend(LatticeGeometry(L=L, n_dims=4), PrecisionPolicy())
    params = PARAMS
    g = torch.Generator().manual_seed(seed_base)

    if kind == "cold":
        U = make_cold_U(be, params)
        psi = be.random_fermion(params, g)
    elif kind == "random":
        U = make_random_U(be, params, seed_base + 1)
        psi = be.random_fermion(params, g)
    elif kind == "boundary":
        U = make_cold_U(be, params)
        psi = make_boundary_psi(be, params)
    else:
        raise ValueError(kind)

    Dpsi = be.wilson_dirac(psi, U, params)
    Ddag = be.wilson_dirac_dagger(psi, U, params)
    Qpsi = be.normal_operator(psi, U, params)

    g5psi = torch.einsum("st,...ti->...si", be.g5, psi)
    Dg5 = be.wilson_dirac(g5psi, U, params)
    manual_dag = torch.einsum("st,...ti->...si", be.g5, Dg5)
    g5_err = float(be.complex_norm(Ddag - manual_dag))

    g2 = torch.Generator().manual_seed(seed_base + 99)
    phi = be.random_fermion(params, g2)
    Qphi = be.normal_operator(phi, U, params)
    q_herm_err = abs(
        complex(be.complex_dot(phi, Qpsi)) - complex(be.complex_dot(Qphi, psi))
    )

    g3 = torch.Generator().manual_seed(seed_base + 123)
    eta = be.random_fermion(params, g3)
    b = be.wilson_dirac_dagger(eta, U, params)
    residuals: list[float] = []

    def matvec(v):
        return be.normal_operator(v, U, params)

    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = float(be.complex_dot(r, r).real)
    b_n = max(float(be.complex_norm(b)), 1e-30)
    maxiter = 80
    final_iters = maxiter
    for it in range(maxiter):
        Ap = matvec(p)
        denom = float(be.complex_dot(p, Ap).real)
        if abs(denom) < 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(be.complex_dot(r, r).real)
        resid = (rs_new**0.5) / b_n
        residuals.append(resid)
        if resid < 1e-10:
            final_iters = it + 1
            break
        p = r + (rs_new / max(rs_old, 1e-30)) * p
        rs_old = rs_new

    S = be.plaquette_action(U, BETA)
    out_dir = GOLDEN_ROOT / f"L{L}" / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    arrays = {
        "inputs__psi": to_np(psi),
        "inputs__U": to_np(U),
        "inputs__eta": to_np(eta),
        "inputs__phi": to_np(phi),
        "outputs__Dpsi": to_np(Dpsi),
        "outputs__Ddag_psi": to_np(Ddag),
        "outputs__Qpsi": to_np(Qpsi),
        "outputs__cg_x": to_np(x),
    }
    np.savez_compressed(out_dir / "golden.npz", **arrays)
    meta = {
        "schema_version": 1,
        "abi": "wilson_dirac_v1",
        "L": L,
        "kind": kind,
        "seed_base": seed_base,
        "params": {
            "mass": params.mass,
            "wilson_r": params.wilson_r,
            "color_dim": params.color_dim,
            "spinor_dim": params.spinor_dim,
            "n_dims": 4,
            "beta": BETA,
        },
        "metrics": {
            "g5_hermiticity_err": g5_err,
            "Q_hermiticity_err": q_herm_err,
            "cg_iters": final_iters,
            "cg_final_resid": residuals[-1] if residuals else None,
            "cg_residual_trajectory": residuals,
            "plaquette_action": float(S.real if hasattr(S, "real") else S),
            "norm_Dpsi": float(be.complex_norm(Dpsi)),
            "norm_psi": float(be.complex_norm(psi)),
        },
        "arrays": sorted(arrays.keys()),
    }
    (out_dir / "golden.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir.relative_to(ROOT)}/golden.npz + golden.meta.json")


def main() -> None:
    for (L, kind), seed in SEEDS.items():
        suite_for(L, kind, seed)
    manifest = {
        "schema_version": 1,
        "abi": "wilson_dirac_v1",
        "format": "npz+meta",
        "description": "Immutable golden corpus.",
        "suites": [
            {
                "path": f"L{L}/{kind}",
                "L": L,
                "kind": kind,
                "seed_base": SEEDS[(L, kind)],
            }
            for L in (2, 4)
            for kind in ("cold", "random", "boundary")
        ],
    }
    (GOLDEN_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("MANIFEST.json ok")


if __name__ == "__main__":
    main()
