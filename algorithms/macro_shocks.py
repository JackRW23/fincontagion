"""
algorithms.macro_shocks
-----------------------

Apply macro‑economic shocks to banks’ portfolios and return the equity impact
ΔE for each bank (negative values = equity loss).

The function relies on three conventions agreed in the project recipes:

* ``PortfolioDF`` (imported from ``datamodel``) is a long‑format DataFrame
  with *at least* the columns
      bank_id, asset_class_id, exposure_eur
  – additional columns are ignored.

* ``scenario_slice`` is a **row slice** of the validated macro‑scenario
  table (schema ``MACRO_SCHEMA`` in ``datamodel``).  It must contain the
  columns
      asset_class_id, shock_pct
  where *shock_pct* is expressed **in decimals** (e.g. -0.05 = –5 %).

* The returned vector is a NumPy ``ndarray`` (alias ``Array`` in
  ``datamodel``) with one entry per bank **ordered by
  ``bank_id.sort_values().unique()``** – identical ordering to the liability
  matrix produced by ``network.to_matrix``.

If every entry in *shock_pct* is (to within ``atol``) zero, a zero vector is
returned without touching the portfolio – this is required for the baseline
unit test.

Author: Fin‑Contagion package
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import pandera as pa

# Project‑wide types / schemas -------------------------------------------------
from datamodel import PortfolioDF, MACRO_SCHEMA

try:  # Array is just an alias for np.ndarray defined in datamodel
    from datamodel import Array  # type: ignore
except ImportError:  # fallback makes standalone type‑checking easier
    Array = np.ndarray  # type: ignore[misc,assignment]

__all__: Sequence[str] = ("apply",)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


def _detect_exposure_column(df: pd.DataFrame) -> str:
    """
    Pick the first recognised exposure column name or raise.

    The fallback list mirrors the recipes (exposure_eur was the canonical
    name, but earlier drafts mentioned value_eur / holding_eur).
    """
    for col in ("exposure_eur", "value_eur", "holding_eur"):
        if col in df.columns:
            return col
    raise KeyError(
        "PortfolioDF must contain one of "
        "'exposure_eur', 'value_eur', or 'holding_eur'."
    )


def _detect_shock_column(df: pd.DataFrame) -> str:
    """Return the name of the shock‑percentage column in *scenario_slice*."""
    for col in ("shock_pct", "pct_change", "pct_chg"):
        if col in df.columns:
            return col
    raise KeyError(
        "scenario_slice must contain a column with the macro shock percentage "
        "(one of 'shock_pct', 'pct_change', or 'pct_chg')."
    )


# -----------------------------------------------------------------------------


def apply(portfolios: PortfolioDF, scenario_slice: pd.DataFrame) -> Array:  # noqa: D401
    """
    Apply a macro‑economic scenario to portfolios and return ΔE per bank.

    Parameters
    ----------
    portfolios
        Long‑format **PortfolioDF** validated elsewhere (schema
        ``PORTFOLIOS_SCHEMA``).  Must contain at least:
            * bank_id
            * asset_class_id
            * exposure_eur (or compatible alias, see below)
    scenario_slice
        A validated **subset** of ``20_macro_scenarios.csv`` rows for one
        (scenario_id, date) pair.  Columns required:
            * asset_class_id
            * shock_pct   (decimal, e.g. -0.02 → -2 %)
        Extra columns are tolerated.

    Returns
    -------
    Array
        NumPy ``float64`` vector ``ΔE`` of length *n_banks*.  Negative entries
        represent equity *losses*; positive entries are equity gains.  The
        ordering matches the ascending ``bank_id`` order used throughout the
        package.

    Notes
    -----
    * If **all** ``shock_pct`` values are (within 1e-12) zero, a zero vector is
      returned early – this satisfies the baseline unit test.
    * Missing shocks for particular asset classes default to **0 %**.
    * Banks holding no shocked asset classes receive a ΔE of 0.0.
    """
    # ------------------------------------------------------------------
    # 1. Validate & prepare inputs
    # ------------------------------------------------------------------
    logger.debug("Validating scenario_slice against MACRO_SCHEMA")
    try:
        scenario_slice = MACRO_SCHEMA.validate(scenario_slice, lazy=True)  # type: ignore
    except pa.errors.SchemaError as exc:
        raise pa.errors.SchemaError(
            "scenario_slice failed MACRO_SCHEMA validation"
        ) from exc

    exposure_col = _detect_exposure_column(portfolios)
    shock_col = _detect_shock_column(scenario_slice)

    # Early‑exit baseline check
    if np.all(np.isclose(scenario_slice[shock_col].values, 0.0, atol=1e-12)):
        n_banks = portfolios["bank_id"].nunique()
        logger.debug("Baseline scenario detected – returning zeros.")
        return np.zeros(n_banks, dtype="float64")  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 2. Build exposure pivot: rows = bank_id, cols = asset_class_id
    # ------------------------------------------------------------------
    pivot = (
        portfolios[["bank_id", "asset_class_id", exposure_col]]
        .pivot_table(
            index="bank_id",
            columns="asset_class_id",
            values=exposure_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=0)  # ensure ascending bank_id order
        .sort_index(axis=1)  # consistent column order (not strictly needed)
        .astype("float64")
    )
    logger.debug("Exposure pivot table built with shape %s", pivot.shape)

    # ------------------------------------------------------------------
    # 3. Align shocks to pivot columns and broadcast
    # ------------------------------------------------------------------
    shocks = (
        scenario_slice[["asset_class_id", shock_col]]
        .drop_duplicates("asset_class_id")
        .set_index("asset_class_id")[shock_col]
        .astype("float64")
    )

    # Reindex to include *all* asset classes present in portfolios
    shocks = shocks.reindex(pivot.columns, fill_value=0.0)

    # Broadcast multiplication: Δ value per (bank, asset_class)
    delta = pivot.values * shocks.values  # type: ignore[arg-type]
    logger.debug("Delta matrix computed.")

    # ------------------------------------------------------------------
    # 4. Aggregate Δ across asset classes ⇒ ΔE per bank
    # ------------------------------------------------------------------
    delta_e = delta.sum(axis=1).astype("float64")  # float64 for consistency

    # ------------------------------------------------------------------
    # 5. Return as Array (np.ndarray alias)
    # ------------------------------------------------------------------
    return delta_e  # type: ignore[return-value]


# -----------------------------------------------------------------------------


if __name__ == "__main__":  # Simple smoke test when run directly
    import sys

    logging.basicConfig(level=logging.INFO)

    msg = (
        "This module is not intended to be executed as a script.\n"
        "Run the unit tests or call algorithms.macro_shocks.apply(...) "
        "from the simulation engine."
    )
    print(msg, file=sys.stderr)
