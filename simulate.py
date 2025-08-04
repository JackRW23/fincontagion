"""
simulate.py
===========

Single public entry‑point for running one contagion‑and‑fire‑sale simulation
scenario.  Designed for both programmatic use

    >>> import fincontagion as fc
    >>> s = fc.simulate("config.yaml")

and CLI use

    $ python -m fincontagion.simulate config.yaml

It pulls together *validated* inputs, delegates the heavy‑lifting to
``simulator.engine.run`` (re‑exported by ``simulator/__init__.py``) and
returns a **pd.Series** whose index exactly matches the header of
``data/processed/20_simulation_runs.csv``.  Numeric columns are coerced to
proper dtypes so that downstream appends / concatenations do not trigger
implicit object conversions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

import io_utils                 # processed‑data loader / validator
import shocks                    # handy macro‑scenario slicer
from simulator import simulate_engine  # re‑exported engine.run


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
_DATA_PROCESSED_DIR = (
    Path(__file__).resolve().parent / "data" / "processed"
)
_TEMPLATE_RUNS_PATH = _DATA_PROCESSED_DIR / "20_simulation_runs.csv"


def _read_cfg(path: str | Path) -> Dict[str, Any]:
    """Parse YAML config into a plain dict (no validation beyond YAML)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def _template_cols() -> List[str]:
    """Column order / names expected for the returned Series."""
    if _TEMPLATE_RUNS_PATH.exists():
        return list(pd.read_csv(_TEMPLATE_RUNS_PATH, nrows=0).columns)
    # Fallback (tests may not ship the CSV); order is then indifferent.
    return []


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert anything that *can* be numeric to numeric dtype."""
    for col in series.index:
        try:
            series[col] = pd.to_numeric(series[col])
        except (ValueError, TypeError):
            # leave non‑numeric (e.g. strings) untouched
            pass
    return series


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run(config_path: str | Path) -> pd.Series:
    """
    Execute **one** simulation scenario.

    Parameters
    ----------
    config_path
        Path to a YAML file whose keys are consumed by the underlying engine.
        At minimum it must contain ``scenario_id`` and ``as_of_date`` (YYYY‑MM‑DD),
        plus any model parameters (α, ρ, κ, LGD, etc.) required by
        :pyfunc:`simulator.engine.run`.

    Returns
    -------
    pandas.Series
        A single row whose index matches the columns of
        ``20_simulation_runs.csv``; suitable for ``df.append`` / ``to_csv``
        by the higher‑level *experiments/run_grid.py* utility.
    """
    # --------------------------------------------------------------------- #
    # 1. Configuration & data ingestion
    # --------------------------------------------------------------------- #
    cfg = _read_cfg(config_path)

    # Typed / schema‑validated DataFrames
    banks = io_utils.load("banks")
    exposures = io_utils.load("exposures")
    portfolios = io_utils.load("portfolios")

    # Macro slice used by the engine (sub‑DF for one scenario & date)
    try:
        scenario_id = cfg["scenario_id"]
        as_of_date = cfg["as_of_date"]
    except KeyError as e:
        raise KeyError(
            f"Missing required key in YAML config: {e.args[0]}"
        ) from None
    scenario_slice = shocks.slice(scenario_id, as_of_date)

    # --------------------------------------------------------------------- #
    # 2. Core simulation – delegated to the deterministic engine
    # --------------------------------------------------------------------- #
    result_dict: Dict[str, Any] = simulate_engine(
        cfg=cfg,
        banks=banks,
        exposures=exposures,
        portfolios=portfolios,
        scenario_slice=scenario_slice,
    )

    # --------------------------------------------------------------------- #
    # 3. Normalise to pd.Series with correct column set & dtypes
    # --------------------------------------------------------------------- #
    if (cols := _template_cols()):
        series = pd.Series(index=cols, dtype="object")
        # Fill what we have, leave the rest NaN
        for k, v in result_dict.items():
            if k not in series.index:
                # Engine might have produced an unexpected extra column
                # – keep it but append at the end for visibility.
                series.loc[k] = v
            else:
                series[k] = v
    else:
        # No template available (e.g. in isolated CI), just use whatever we got
        series = pd.Series(result_dict)

    series = _coerce_numeric(series)
    return series


# --------------------------------------------------------------------------- #
# Command‑line hook (optional, nice for ad‑hoc runs)
# --------------------------------------------------------------------------- #
def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run a single contagion simulation.")
    parser.add_argument(
        "config_path", help="Path to YAML configuration file describing the scenario."
    )
    args = parser.parse_args()
    output = run(args.config_path)
    # Pretty‑print to stdout (tab‑delimited); can be piped to jq/csvcut if needed
    print(output.to_csv(sep="\t", header=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
