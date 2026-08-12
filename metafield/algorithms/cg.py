"""Layer 3 — CG consumes matvec only. No backend imports."""

from __future__ import annotations

from typing import Any, Callable, Tuple


def cg_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    dot: Callable[[Any, Any], Any],
    norm: Callable[[Any], Any],
    x0: Any | None = None,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> Tuple[Any, int, float]:
    """Hermitian CG: matvec implements Q = A†A or other HPD operator.

    `dot` and `norm` are injected so complex reductions match the backend.
    """
    x = b * 0 if x0 is None else x0.clone() if hasattr(x0, "clone") else x0
    r = b - matvec(x)
    p = r.clone() if hasattr(r, "clone") else r
    rs_old = float(dot(r, r).real)
    b_n = max(float(norm(b)), 1e-30)

    for it in range(maxiter):
        Ap = matvec(p)
        denom = float(dot(p, Ap).real)
        if abs(denom) < 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(dot(r, r).real)
        resid = (rs_new ** 0.5) / b_n
        if resid < tol:
            return x, it + 1, resid
        p = r + (rs_new / max(rs_old, 1e-30)) * p
        rs_old = rs_new
    return x, maxiter, (rs_old ** 0.5) / b_n
