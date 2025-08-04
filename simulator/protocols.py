"""
simulator.protocols
~~~~~~~~~~~~~~~~~~~
Structural‑typing “plug‑in point” for contagion‑solvers.

Any algorithm that prices interbank liabilities under distress
(Eisenberg–Noe fixed‑point, linear‑programming variants, DebtRank
approximations, etc.) should implement :class:`ContagionSolver`.  
The *simulation engine* can then accept **any** such object without
caring about the concrete class.

A solver must expose one public method::

    solve(L, E0, lgd, **kw) -> (paid, defaulted)

Parameters
----------
L : Array
    Liability matrix, shape (n, n) with *borrower rows / lender cols*.
E0 : Array
    Initial equity vector, length n, **same order as rows of L**.
lgd : float
    Loss‑given‑default in [0, 1].
**kw
    Optional algorithm‑specific parameters.

Returns
-------
paid : Array
    Settlement/payment matrix, shape (n, n).
defaulted : Array
    Boolean (or {0,1} float) vector of length n indicating defaults.
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

# ---------------------------------------------------------------------
# Shared numeric type alias – first try to import the canonical alias
# from the project‑wide datamodel; fall back to numpy.ndarray so the
# module is importable even before datamodel.py exists.
# ---------------------------------------------------------------------
try:
    from datamodel import Array  # single source of truth
except ModuleNotFoundError:  # pragma: no cover
    import numpy as _np

    Array = _np.ndarray  # type: ignore  # fallback for static checkers

__all__ = ["ContagionSolver", "Array"]


@runtime_checkable
class ContagionSolver(Protocol):
    """Protocol that every contagion‑solver must satisfy."""

    # NOTE:  The return annotation uses the older ``Tuple`` so that
    #        the file type‑checks under Python 3.8 if the project
    #        back‑ports typing features; feel free to switch to
    #        ``tuple[Array, Array]`` when Python 3.9+ is guaranteed.
    def solve(
        self,
        L: Array,
        E0: Array,
        lgd: float,
        **kw,
    ) -> Tuple[Array, Array]:
        """Execute the solver and return *(paid, defaulted)*."""
        ...
