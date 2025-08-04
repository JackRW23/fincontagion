"""
algorithms.debtrank
~~~~~~~~~~~~~~~~~~~
DebtRank‑style systemic‑risk metric.

The original DebtRank algorithm (Battiston *et al.*, 2012) tracks how
distress propagates through an interbank leverage matrix until it
converges.  Here we implement a *linear* variant that is lightweight,
equity‑weighted and well‑suited for batch Monte‑Carlo runs inside
:pyfunc:`simulator.engine.run`.

Key API
-------
python def compute(W: Array, equities: Array, beta: float = 0.9) -> float
    Parameters
    ----------
    W
        Weighted, *equity‑normalised* liabilities:
        ``W[i, j] = L_ij / E_i`` with shape ``(n, n)``.
        Borrower indices on columns; lender indices on rows.
    equities
        Initial equity vector (same order as rows of *W*), shape ``(n,)``.
    beta
        Dampening / “forget” factor in (0, 1].  Higher ⇒ stronger
        shock propagation.  Default 0 .9 as in the recipe.

    Returns
    -------
    float
        System‑wide equity depletion ratio in the closed interval [0, 1].

Test focus
----------
*Monotonicity*: scaling *W* uniformly upward must *increase*
the returned value (checked in **tests/test_debtrank.py**).

References
----------
Battiston, S., Puliga, M., Kaushik, R. *et al.*
“DebtRank: Too Central to Fail? Financial Networks, the FED
and Systemic Risk.” *Sci. Rep.* **2**, 541 (2012).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

try:                      # single‑source Array alias
    from datamodel import Array  # type: ignore
except ImportError:       # fallback keeps stub self‑contained for unit tests
    Array = np.ndarray    # type: ignore


__all__: Final[list[str]] = ["compute"]


def _validate_inputs(W: Array, equities: Array) -> None:
    """Lightweight, fail‑fast validation (no heavy imports)."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("`W` must be a square 2‑D array.")
    if equities.ndim != 1 or equities.shape[0] != W.shape[0]:
        raise ValueError("`equities` must be 1‑D and length match `W`.")
    if np.any(equities <= 0.0) or not np.isfinite(equities).all():
        raise ValueError("`equities` must contain positive, finite values.")
    if not np.isfinite(W).all():
        raise ValueError("`W` must contain finite values only.")


def compute(W: Array, equities: Array, beta: float = 0.9) -> float:  # noqa: D401
    """
    Compute the **equity‑weighted DebtRank** of the network.

    The algorithm iterates a *linear* propagation of incremental distress
    originating from *one* bank at a time, then averages results weighted
    by each bank’s share of total equity.

    The propagation rule is::

        Δh(t+1) = beta · W · Δh(t)

    where ``h`` is the cumulative distress level and
    ``Δh(t) = h(t) − h(t−1)``.  Iteration stops when
    ``max(Δh) < 1e‑12`` or 1000 steps (guaranteed to converge for
    beta · ρ(W) < 1, where ρ is the spectral radius).

    Parameters
    ----------
    W, equities, beta
        See module docstring.

    Returns
    -------
    float
        Fraction of system equity ultimately wiped out (clipped to ≤ 1.0).

    Notes
    -----
    * The function is **pure** and has no external side effects.
    * Complexity O(n³) in the worst case (n calls to a sparse‑friendly
      matrix–vector multiply loop).

    """
    _validate_inputs(W, equities)

    if not (0.0 < beta <= 1.0):
        raise ValueError("`beta` must lie in the interval (0, 1].")

    n: int = W.shape[0]
    total_equity: float = float(np.sum(equities))
    if total_equity == 0.0:
        raise ValueError("Total system equity cannot be zero.")

    # Pre‑compute row‑normalised weight matrix for efficient vdot later
    W_mat: Array = np.asarray(W, dtype=float)  # ensure float64 contiguous

    # Helper for a single‑origin DebtRank
    def _single_origin(k: int) -> float:
        """Equity share lost when bank *k* suffers unit distress."""
        h = np.zeros(n, dtype=float)
        delta = np.zeros_like(h)
        h[k] = 1.0
        delta[k] = 1.0

        # Iterate until new distress no longer propagates.
        for _ in range(1000):  # hard safety cap
            delta = beta * (W_mat @ delta)
            if not np.any(delta > 1e-12):
                break
            # cap cumulative distress at 1 (default)
            new_h = np.minimum(1.0, h + delta)
            delta = new_h - h
            h = new_h
        else:  # pragma: no cover
            raise RuntimeError("DebtRank failed to converge in 1000 steps.")

        # Distress of k itself does *not* count toward system amplification.
        incremental = h.copy()
        incremental[k] = 0.0
        depleted_equity = float(np.dot(incremental, equities))
        return depleted_equity / total_equity  # fraction ∈ [0, 1]

    # Equity weights (used to average single‑origin impacts)
    weights = equities / total_equity
    systemic_loss: float = 0.0
    for idx, w in enumerate(weights):
        if w == 0.0:
            continue
        systemic_loss += w * _single_origin(idx)

    # Numerical guard – avoid tiny negatives / overshoots
    return min(max(systemic_loss, 0.0), 1.0)
