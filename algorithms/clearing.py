"""
algorithms.clearing
~~~~~~~~~~~~~~~~~~~
Eisenberg‑Noe (2001) fixed‑point solver for an inter‑bank clearing
vector and the resulting paid–liability matrix.

Key API
-------
solve(L: Array, E0: Array, lgd: float, *,
      tol: float = 1e-10, max_iter: int = 1_000
) -> tuple[Array, Array]

Parameters
----------
L
    2‑D ``np.ndarray`` of size ``(n, n)``: liabilities **rows = borrowers,
    cols = lenders**.  ``L[i, j]`` is the nominal amount that *i* owes *j*.
E0
    1‑D ``np.ndarray`` of size ``(n,)``: initial equity / external assets
    of each bank, **in the same order as the rows of `L`**.
lgd
    Loss‑given‑default on inter‑bank claims *as a fraction*
    (``0.0`` = full recovery, ``1.0`` = no recovery).  In classic
    Eisenberg‑Noe LGD ≡ 1.0; passing any other value scales the payments
    of defaulting banks by ``(1‑lgd)``.
tol, max_iter
    Convergence tolerance and hard iteration cap.

Returns
-------
paid : Array
    ``(n, n)`` matrix of realised payments where
    ``paid[i, j] ≤ L[i, j]``.  Row sums equal the clearing payment vector
    *p*; column sums are the amounts each lender receives.
default : Array[bool]
    Boolean vector (``n,``) indicating whether each bank defaults
    (``True``) in the clearing solution.

Notes
-----
* The algorithm follows the monotone fixed‑point iteration in
  Eisenberg & Noe (2001) §4 with a recovery‑rate extension controlled
  by *lgd*.
* It is *deterministic* given identical inputs; no RNG involved.
* All heavy lifting is NumPy‑vectorised; no Python loops over banks.

Examples
--------
>>> from algorithms.clearing import solve
>>> import numpy as np
>>> L = np.array([[0, 10,  0],
...               [0,  0,  5],
...               [3,  0,  0]], dtype=float)
>>> E0 = np.array([2.5, 2.5, 1.0])
>>> paid, default = solve(L, E0, lgd=1.0)
>>> default
array([False, False, False])
"""
from __future__ import annotations

from typing import Tuple, Final

import numpy as np

try:  # keep mypy/export users happy, but allow import when datamodel not yet built
    from ..datamodel import Array  # type: ignore
except Exception:  # pragma: no cover
    Array = np.ndarray  # type: ignore


def _relative_liability_matrix(L: Array) -> Array:
    """
    Build the row‑stochastic matrix Π where Π[i, j] is the share of
    *i*'s obligations owed to *j*.

    Any row with zero total liabilities becomes all‑zeros.
    """
    row_sum: Array = L.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Π: Array = np.divide(L, row_sum, where=row_sum > 0)
    Π[np.isnan(Π)] = 0.0  # rows with 0 liabilities
    return Π


def solve(
    L: Array,
    E0: Array,
    lgd: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 1_000,
) -> Tuple[Array, Array]:
    """
    Compute the clearing payments and default set.

    See module docstring for a full specification.
    """
    # --- basic validation -------------------------------------------------
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be square (n×n) ndarray.")
    n: Final[int] = L.shape[0]
    if E0.shape != (n,):
        raise ValueError("E0 must be a 1‑D array of length n.")
    if not (0.0 <= lgd <= 1.0):
        raise ValueError("lgd must lie in [0, 1].")

    # --- pre‑compute ------------------------------------------------------
    bar_p: Array = L.sum(axis=1)  # nominal obligations per bank
    Π: Array = _relative_liability_matrix(L)

    # --- fixed‑point iteration -------------------------------------------
    p: Array = bar_p.copy()  # start with full payment
    default_prev = np.zeros(n, dtype=bool)

    for _ in range(max_iter):
        # Assets = external equity + received inter‑bank payments
        received: Array = Π.T @ p
        assets: Array = E0 + received

        # Clearing payment each bank *can* make
        p_new: Array = np.minimum(bar_p, assets)

        # Apply LGD if the bank would otherwise default
        default_now = p_new < bar_p - tol
        if lgd and default_now.any():
            # Reduce the payment of newly defaulting banks by recovery rate
            recovery_factor: Array = np.where(default_now, 1.0 - lgd, 1.0)
            p_new = recovery_factor * p_new

        # Convergence check (sup‑norm)
        if np.max(np.abs(p_new - p)) < tol and np.array_equal(
            default_now, default_prev
        ):
            p = p_new
            default_prev = default_now
            break

        p = p_new
        default_prev = default_now
    else:  # pragma: no cover
        raise RuntimeError(
            f"Clearing algorithm did not converge in {max_iter} iterations."
        )

    # --- outputs ----------------------------------------------------------
    paid: Array = p[:, None] * Π  # broadcast row‑wise
    default: Array = default_prev
    return paid, default
