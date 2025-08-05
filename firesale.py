"""
firesale.py  (root helper wrapper)

Purpose
-------
User‑convenience wrapper around ``algorithms.firesale.iterate`` for quick
exploration and notebook plots.

Key API
-------
run(portfolios_df: PortfolioDF, lambda_col: str = "lambda_price_impact",
    **iterate_kwargs) -> pd.Series

* ``portfolios_df`` – Same Pandera‑validated dataframe passed to the lower‑level
  algorithm.  It **must** contain the column specified by *lambda_col*.
* ``lambda_col``    – Name of the column holding the per‑asset ƛ (price‑impact
  sensitivity).  The default matches the data‑generation recipes.
* ``**iterate_kwargs`` – Forward‑compatibility pass‑through to the underlying
  algorithm (e.g. *max_iter*, *tol_e* …).

Returns
-------
pd.Series
    Price factor λ_t ∈ (0, 1] for every ``asset_class_id`` (index).  A value of
    1 signals no fire‑sale impact.

Behavioural Guarantees (recipe‑driven tests)
--------------------------------------------
* Identity check – If the specified ``lambda_col`` is all zeros, every price
  factor must remain 1.
* Deterministic for a fixed random seed (delegated to the algorithm layer).

Upstream ↔ Downstream
---------------------
* Called by notebooks and quick demos.  
* The full simulator (`simulate.py`) and grid sweeps (`experiments/run_grid.py`)
  route through here for convenience but can call the low‑level algorithm
  directly if needed.
"""
from __future__ import annotations

import inspect
from typing import Any, Final, TYPE_CHECKING, TypeAlias

import pandas as pd

# ---------------------------------------------------------------------------
# Typing setup
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    # During static analysis we import the real schema alias defined in
    # datamodel.py, but this import is NOT executed at run‑time.
    from datamodel import PortfolioDF               # pragma: no cover
else:
    # At run‑time we simply treat a PortfolioDF as a plain DataFrame.
    PortfolioDF: TypeAlias = pd.DataFrame

# Core algorithm import
from algorithms import firesale as _algo_firesale

DEFAULT_LAMBDA_COL: Final[str] = "lambda_price_impact"

__all__ = ["run", "DEFAULT_LAMBDA_COL"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _validate_inputs(df: PortfolioDF, lambda_col: str) -> None:
    """Basic sanity checks before handing off to the heavy algorithm layer."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"`portfolios_df` must be a pandas DataFrame; got {type(df).__name__}"
        )

    if lambda_col not in df.columns:
        raise KeyError(
            f"Column '{lambda_col}' not found in portfolios_df. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("portfolios_df is empty – nothing to iterate on.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run(
    portfolios_df: PortfolioDF,
    lambda_col: str = DEFAULT_LAMBDA_COL,
    **iterate_kwargs: Any,
) -> pd.Series:
    """
    Wrapper around :pyfunc:`algorithms.firesale.iterate`.

    Parameters
    ----------
    portfolios_df
        Pandera‑validated dataframe of current bank holdings. Must include the
        fire‑sale impact column specified by *lambda_col*.
    lambda_col
        Name of the column containing λ (price‑impact) coefficients.
    **iterate_kwargs
        Additional keyword arguments forwarded to the low‑level algorithm
        (e.g. *max_iter*, *tol_e*).

    Returns
    -------
    pd.Series
        Price path indexed by ``asset_class_id``.
    """
    _validate_inputs(portfolios_df, lambda_col)

    # ------------------------------------------------------------------ #
    # Forward‑compatibility shim: older/newer versions of the algorithm
    # may or may not expose `lambda_col` as an explicit parameter.  We
    # inspect its signature at runtime and forward accordingly.
    # ------------------------------------------------------------------ #
    iterate_sig = inspect.signature(_algo_firesale.iterate)
    if "lambda_col" in iterate_sig.parameters:
        price_series: pd.Series = _algo_firesale.iterate(
            portfolios_df, lambda_col=lambda_col, **iterate_kwargs
        )
    else:
        # Fall back – assume the algorithm expects the DataFrame already
        # containing the correctly named column.
        if lambda_col != DEFAULT_LAMBDA_COL and DEFAULT_LAMBDA_COL not in portfolios_df:
            # Preserve the original DF by working on a shallow copy with the
            # expected column name.
            portfolios_df = portfolios_df.copy()
            portfolios_df[DEFAULT_LAMBDA_COL] = portfolios_df[lambda_col]
        price_series = _algo_firesale.iterate(portfolios_df, **iterate_kwargs)

    # Final defensive check for the identity property the recipe requires.
    if (portfolios_df[lambda_col] == 0).all():
        if not (price_series == 1).all():
            raise AssertionError(
                "Identity test failed: λ=0 should yield price factors of 1."
            )

    return price_series
