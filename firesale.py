"""
firesale.py  –  Convenience wrapper around algorithms.firesale.iterate
=====================================================================

* Exposes a single helper `run()` so notebooks / demos can call the
  firesale algorithm with one line.
* Keeps *all* numerical logic in `algorithms/firesale.py`.
* Passes the identity‑property test (λ = 0 → price factor 1).

This file only touches typing; the public behaviour is identical to
the earlier draft.
"""
from __future__ import annotations

import inspect
from typing import Any, Final, TYPE_CHECKING, TypeAlias

import pandas as pd

# ------------------------------------------------------------------ #
# Typing setup
# ------------------------------------------------------------------ #
if TYPE_CHECKING:
    # During static analysis we import the real schema‑aware alias that
    # `datamodel.py` will create.  Nothing is executed at run‑time.
    from datamodel import PortfolioDF        # pragma: no cover
else:
    # When the program actually runs we are happy with a plain DataFrame.
    PortfolioDF: TypeAlias = pd.DataFrame

# ------------------------------------------------------------------ #
# Implementation
# ------------------------------------------------------------------ #
from algorithms import firesale as _algo_firesale

DEFAULT_LAMBDA_COL: Final[str] = "lambda_price_impact"

__all__ = ["run", "DEFAULT_LAMBDA_COL"]


def _validate_inputs(df: PortfolioDF, lambda_col: str) -> None:
    """Lightweight checks before delegation to the core algorithm."""
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


def run(
    portfolios_df: PortfolioDF,
    lambda_col: str = DEFAULT_LAMBDA_COL,
    **iterate_kwargs: Any,
) -> pd.Series:
    """
    Friendly wrapper around :pyfunc:`algorithms.firesale.iterate`.

    Parameters
    ----------
    portfolios_df : PortfolioDF
        Pandera‑validated dataframe of bank holdings.  
        **Must contain** the column named *lambda_col*.
    lambda_col : str, default "lambda_price_impact"
        Column holding the λ (price‑impact) coefficients.
    **iterate_kwargs
        Extra keyword arguments forwarded verbatim to the lower‑level
        ``iterate`` function (e.g. *max_iter*, *tol_e*).

    Returns
    -------
    pd.Series
        Final price factor λ_t (index = ``asset_class_id``).
    """
    _validate_inputs(portfolios_df, lambda_col)

    # Some versions of the algorithm accept lambda_col explicitly,
    # others expect the DF already prepared.  Detect at run‑time.
    iterate_sig = inspect.signature(_algo_firesale.iterate)
    if "lambda_col" in iterate_sig.parameters:
        price_series = _algo_firesale.iterate(
            portfolios_df, lambda_col=lambda_col, **iterate_kwargs
        )
    else:
        if lambda_col != DEFAULT_LAMBDA_COL and DEFAULT_LAMBDA_COL not in portfolios_df:
            portfolios_df = portfolios_df.copy()
            portfolios_df[DEFAULT_LAMBDA_COL] = portfolios_df[lambda_col]
        price_series = _algo_firesale.iterate(portfolios_df, **iterate_kwargs)

    # Identity check required by the recipe: λ = 0 → price factor 1
    if (portfolios_df[lambda_col] == 0).all() and not (price_series == 1).all():
        raise AssertionError(
            "Identity property violated: all‑zero λ should yield all‑ones price factors."
        )

    return price_series
