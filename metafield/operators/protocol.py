"""Layer 1–2 — mathematical operators + *provisional* backend surface.

Wilson–Dirac ABI v1 is IMMUTABLE (docs/WILSON_DIRAC_ABI.md).
This Protocol is EXPERIMENTAL software glue — method names, handles, and
sync semantics may change as the first FPGA implementation teaches us
what the operator actually wants from hardware.

Physics should depend on the mathematical meaning and the golden corpus,
not on this surface being eternal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from metafield.lattice.types import BoundaryCondition, LatticeGeometry, PrecisionPolicy


@dataclass(frozen=True)
class WilsonParams:
    mass: float = 0.1
    wilson_r: float = 1.0
    color_dim: int = 3
    spinor_dim: int = 4


@runtime_checkable
class OperatorBackend(Protocol):
    """Provisional executor surface. Not a frozen hardware ABI.

    Implement enough to pass tests/operator/goldens/. Prefer matching
    the oracle on wilson_dirac first; everything else is secondary.
    """

    name: str

    def wilson_dirac(self, psi: Any, U: Any, params: WilsonParams) -> Any:
        """Return D_W ψ. Layout: ψ[x, spin, color], U[x, μ, color, color]."""
        ...

    def wilson_dirac_dagger(self, psi: Any, U: Any, params: WilsonParams) -> Any:
        """Return D_W† ψ. Oracle uses γ₅ D γ₅."""
        ...

    def normal_operator(self, psi: Any, U: Any, params: WilsonParams) -> Any:
        """Return (D† D) ψ."""
        ...

    def plaquette_action(self, U: Any, beta: float) -> Any:
        """Scalar Wilson gauge action (not yet a frozen ABI)."""
        ...

    def gauge_force(self, U: Any, beta: float) -> Any:
        """su(N)-valued force (not yet a frozen ABI)."""
        ...

    def complex_dot(self, a: Any, b: Any) -> Any:
        """Hermitian inner product ⟨a,b⟩ = Σ conj(a)·b."""
        ...

    def complex_norm(self, a: Any) -> Any:
        """‖a‖ = √⟨a,a⟩ as a real scalar."""
        ...

    def synchronize(self) -> None:
        """Device barrier (no-op on CPU). Mechanism is experimental."""
        ...


@dataclass
class OperatorContext:
    geometry: LatticeGeometry
    boundary: BoundaryCondition
    precision: PrecisionPolicy
    backend: OperatorBackend
