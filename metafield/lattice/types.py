"""Layer 0 — pure representation. No device assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LatticeGeometry:
    L: int
    n_dims: int = 4

    def __post_init__(self) -> None:
        if self.L < 1:
            raise ValueError("L must be >= 1")
        if self.n_dims < 1:
            raise ValueError("n_dims must be >= 1")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple([self.L] * self.n_dims)

    @property
    def volume(self) -> int:
        return self.L ** self.n_dims


@dataclass(frozen=True)
class BoundaryCondition:
    kind: Literal["periodic", "open", "dirichlet"] = "periodic"


@dataclass(frozen=True)
class PrecisionPolicy:
    """Abstract precision. Backends map these to concrete dtypes."""

    storage: Literal["complex128", "complex64", "float32_pair"] = "complex128"
    compute: Literal["complex128", "complex64", "mixed"] = "complex128"
    accumulation: Literal["complex128", "complex64"] = "complex128"
