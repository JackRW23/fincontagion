"""
simulator package
=================

Public interface
----------------
`simulate_engine`
    Alias to :func:`simulator.engine.run`.  Runs a *single* contagion
    simulation scenario and returns the results dictionary described in the
    project specification.

Example
-------
>>> from simulator import simulate_engine
>>> result_row = simulate_engine(cfg, banks_df, exposures_df,
...                              portfolios_df, scenario_slice_df)

Keeping this alias at package level lets users remain agnostic about the
internal module layout while still benefiting from static type‑checking and
documentation tooling.

Nothing else is re‑exported; internal helpers stay encapsulated.
"""

from __future__ import annotations

# Re‑export the core engine entry‑point under a friendlier name.
from .engine import run as simulate_engine  # noqa: F401

__all__: list[str] = ["simulate_engine"]
