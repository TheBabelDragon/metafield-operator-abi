"""Golden-path tests for the frozen Wilson–Dirac ABI (reference backend)."""

from __future__ import annotations

import torch

from backends.reference.torch_backend import TorchReferenceBackend
from metafield.algorithms.cg import cg_solve
from metafield.lattice.types import LatticeGeometry, PrecisionPolicy
from metafield.operators.protocol import WilsonParams


def _backend(L: int = 2) -> TorchReferenceBackend:
    return TorchReferenceBackend(LatticeGeometry(L=L, n_dims=4), PrecisionPolicy())


def test_dirac_identity_unit_gauge():
    be = _backend(2)
    params = WilsonParams(mass=0.1, wilson_r=1.0)
    g = torch.Generator().manual_seed(7)
    psi = be.random_fermion(params, g)
    U = be.unit_gauge(params)
    Dpsi = be.wilson_dirac(psi, U, params)
    assert Dpsi.shape == psi.shape
    assert float(be.complex_norm(Dpsi)) > 0.0


def test_dagger_gamma5_identity():
    be = _backend(2)
    params = WilsonParams()
    g = torch.Generator().manual_seed(11)
    psi = be.random_fermion(params, g)
    U = be.unit_gauge(params)
    dag = be.wilson_dirac_dagger(psi, U, params)
    g5psi = torch.einsum("st,...ti->...si", be.g5, psi)
    Dg5 = be.wilson_dirac(g5psi, U, params)
    manual = torch.einsum("st,...ti->...si", be.g5, Dg5)
    err = float(be.complex_norm(dag - manual))
    assert err < 1e-10


def test_normal_op_hermitian_symmetry():
    be = _backend(2)
    params = WilsonParams()
    g = torch.Generator().manual_seed(13)
    phi = be.random_fermion(params, g)
    psi = be.random_fermion(params, g)
    U = be.unit_gauge(params)
    Qpsi = be.normal_operator(psi, U, params)
    Qphi = be.normal_operator(phi, U, params)
    lhs = be.complex_dot(phi, Qpsi)
    rhs = be.complex_dot(Qphi, psi)
    assert abs(complex(lhs) - complex(rhs)) < 1e-8


def test_cg_converges_on_Q():
    be = _backend(2)
    params = WilsonParams(mass=0.5)
    g = torch.Generator().manual_seed(17)
    eta = be.random_fermion(params, g)
    U = be.unit_gauge(params)
    b = be.wilson_dirac_dagger(eta, U, params)

    def matvec(v):
        return be.normal_operator(v, U, params)

    x, iters, resid = cg_solve(
        matvec, b, dot=be.complex_dot, norm=be.complex_norm, tol=1e-8, maxiter=200
    )
    assert resid < 1e-7
    assert iters < 200


def test_plaquette_unit_gauge_zero_action():
    be = _backend(2)
    U = be.unit_gauge(WilsonParams())
    S = be.plaquette_action(U, beta=5.5)
    assert abs(float(S.real if hasattr(S, "real") else S)) < 1e-10
