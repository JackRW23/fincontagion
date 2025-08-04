"""
simulator.engine
================

Single–scenario driver for the financial‑contagion simulator
(Objective C in the project brief).

The public API is the :func:`run` function, which orchestrates:

1. Conversion of an exposures table into a liability matrix *L*.
2. Application of macro shocks to the banks’ equity vector.
3. Resolution of inter‑bank defaults via Eisenberg‑Noe (or any
   drop‑in solver conforming to :class:`simulator.protocols.ContagionSolver`).
4. **Optional** fire‑sale price dynamics.
5. Calculation of contagion amplification and graph‑level statistics.
6. Packaging of all key outputs into a **plain‐Python dict** convenient
   for downstream tabulation or CSV append.

The function performs *no* I/O and is deterministic given identical
inputs, satisfying the smoke‑test requirement.

Author
------
FinContagion Project (2025) – engine module
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

from algorithms import metrics
from algorithms import macro_shocks
from algorithms import clearing
from algorithms import firesale
from network import to_matrix, to_graph

# --------------------------------------------------------------------------- #
# Optional – type aliases imported only during static type‑checking
# (prevents hard runtime dependency on datamodel if it has not been
# generated yet).
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover
    from datamodel import BankDF, ExposureDF, PortfolioDF  # noqa: F401
else:
    # Fallback runtime aliases so that the module imports even
    # before datamodel.py exists.
    BankDF = pd.DataFrame  # type: ignore
    ExposureDF = pd.DataFrame  # type: ignore
    PortfolioDF = pd.DataFrame  # type: ignore

__all__ = ["run"]

_LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _sort_banks(banks: BankDF) -> pd.DataFrame:
    """
    Return *banks* sorted by ``bank_id`` (ascending), which is the
    canonical ordering used by ``network.to_matrix``.  A copy is returned
    to avoid mutating the caller’s DataFrame.
    """
    if "bank_id" not in banks.columns:
        raise KeyError("'bank_id' column missing from banks DataFrame")
    return banks.sort_values("bank_id", kind="mergesort").reset_index(drop=True)


def _equity_after_macro(
    banks_sorted: BankDF,
    delta_e: np.ndarray,
) -> np.ndarray:
    """
    Combine initial equity with macro shocks ΔE.

    Parameters
    ----------
    banks_sorted
        *bank_id*-sorted view of the banks table.
    delta_e
        One‑dimensional NumPy array of equity shocks, **same order/length**
        as *banks_sorted*.

    Returns
    -------
    ndarray
        Post‑macro equity vector *E_macro*.
    """
    if len(delta_e) != len(banks_sorted):
        raise ValueError(
            "Length mismatch – ΔE has len %d, banks table has %d rows"
            % (len(delta_e), len(banks_sorted))
        )

    if "equity_eur" not in banks_sorted.columns:
        raise KeyError("'equity_eur' column missing from banks DataFrame")

    e0 = banks_sorted["equity_eur"].to_numpy(dtype=float)
    return e0 + delta_e  # ΔE is negative for losses


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run(
    cfg: Dict[str, Any],
    banks: BankDF,
    exposures: ExposureDF,
    portfolios: PortfolioDF,
    scenario_slice: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Execute **one** full contagion scenario.

    Parameters
    ----------
    cfg
        Parsed YAML config as a plain dict.  Expected keys (with defaults):

        * ``lgd`` : *float*, loss‑given‑default applied in clearing (default 1.0)
        * ``firesale_flag`` : *bool*; run fire‑sale sub‑model if *True* (default False)
        * ``fs_max_iter`` : *int*; max iterations for fire‑sale solver (default 10)
        * ``fs_tol`` : *float*; convergence tolerance for fire‑sale solver (default 1e‑5)

    banks
        Pandera‑validated **BankDF** (must contain ``bank_id`` & ``equity_eur``).
    exposures
        **ExposureDF** – interbank EAD data.
    portfolios
        **PortfolioDF** – banks’ asset holdings (for macro shocks & fire‑sale).
    scenario_slice
        Row(s) from ``20_macro_scenarios.csv`` describing the *current* macro
        date & scenario.  Must include a ``scenario_id`` column.

    Returns
    -------
    dict
        Keys required by downstream tests & CSV:

        ================ ================================================
        Key              Description
        ---------------- -----------------------------------------------
        scenario_id      From *scenario_slice* (if present, else *None*)
        loss_macro_only  Σ losses from macro shocks alone  [EUR]
        loss_contagion   Additional losses from clearing    [EUR]
        loss_total       Aggregate macro + contagion loss   [EUR]
        contagion_amp    Amplification ratio (metrics fn)
        defaults_count   Number of defaulted banks
        firesale_lambda  (Optional) final price multiplier from fire‑sale
        *graph stats*    Keys from :func:`metrics.graph_stats`
        ================ ================================================
    """
    # --------------------------------------------------------------------- #
    # 1. Canonical sort & liability matrix
    # --------------------------------------------------------------------- #
    banks_sorted = _sort_banks(banks)
    L = to_matrix(exposures, banks_sorted)

    # --------------------------------------------------------------------- #
    # 2. Macro‑shock equity depletion
    # --------------------------------------------------------------------- #
    delta_e = macro_shocks.apply(portfolios, scenario_slice)
    E_macro = _equity_after_macro(banks_sorted, delta_e)
    macro_loss = -delta_e.sum()  # ΔE is negative for loss

    # Sanity‑check: macro_loss must be non‑negative per design
    if macro_loss < 0:
        _LOGGER.warning("ΔE sums to a positive number – macro shocks yielded gain?")

    # --------------------------------------------------------------------- #
    # 3. Clearing step (Eisenberg‑Noe)
    # --------------------------------------------------------------------- #
    lgd = float(cfg.get("lgd", 1.0))
    paid_mat, default_bool = clearing.solve(L, E_macro, lgd=lgd)

    # Each lender’s contagion loss = unpaid EADs on its asset side
    contagion_loss_vec = (L - paid_mat).sum(axis=1)
    contagion_loss = contagion_loss_vec.sum()

    total_loss = macro_loss + contagion_loss
    contagion_amp = metrics.contagion_amp(total_loss, macro_loss)

    defaults_count = int(np.count_nonzero(default_bool))

    # --------------------------------------------------------------------- #
    # 4. Optional fire‑sale dynamics
    # --------------------------------------------------------------------- #
    firesale_lambda: float | None = None
    if cfg.get("firesale_flag", False):
        fs_series = firesale.iterate(
            portfolios,
            max_iter=int(cfg.get("fs_max_iter", 10)),
            tol=float(cfg.get("fs_tol", 1e-5)),
        )
        firesale_lambda = float(fs_series.iloc[-1])

    # --------------------------------------------------------------------- #
    # 5. Graph‑level metrics
    # --------------------------------------------------------------------- #
    G = to_graph(exposures)
    graph_stats = metrics.graph_stats(G)  # dict[str, float]

    # --------------------------------------------------------------------- #
    # 6. Assemble output row
    # --------------------------------------------------------------------- #
    result: Dict[str, Any] = {
        "scenario_id": (
            scenario_slice["scenario_id"].iloc[0]
            if "scenario_id" in scenario_slice.columns
            else None
        ),
        "loss_macro_only_eur": float(macro_loss),
        "loss_contagion_eur": float(contagion_loss),
        "loss_total_eur": float(total_loss),
        "contagion_amp": float(contagion_amp),
        "defaults_count": defaults_count,
    }

    # Merge graph‑level statistics
    result.update(graph_stats)

    if firesale_lambda is not None:
        # Naming “firesale_lambda” keeps column width minimal in wide CSVs
        result["firesale_lambda"] = firesale_lambda

    # --------------------------------------------------------------------- #
    # Final sanity check (optional but recommended by project guidelines)
    # --------------------------------------------------------------------- #
    assert (
        result["loss_total_eur"]
        - result["loss_macro_only_eur"]
        - result["loss_contagion_eur"]
    ) < 1e-6, "Loss components do not sum to total"

    return result
