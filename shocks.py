"""
fincontagion.shocks
===================

Utility helpers to *slice* the macro‑scenario table created in the
processed‑data stage.

Example
-------
>>> from fincontagion.shocks import slice
>>> df = slice("adverse2025", "2025-06-30")
>>> df.head()
  scenario_id       date  asset_class_id  price_shock_pct  ...
0  adverse2025 2025-06-30               1            -3.5  ...
1  adverse2025 2025-06-30               2            -8.0  ...
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import pandas as pd
import pandera as pa

from .io_utils import load
from .datamodel import MACRO_SCHEMA
from .errors import SchemaError

# --------------------------------------------------------------------------- #
# Column constants (avoid typos elsewhere)
# --------------------------------------------------------------------------- #
_SCENARIO: Final[str] = "scenario_id"
_DATE: Final[str] = "date"


def slice(scenario_id: str, date: str | datetime) -> pd.DataFrame:
    """
    Return the *macro‑scenario slice* for a given `scenario_id` **and** `date`.

    Parameters
    ----------
    scenario_id
        Identifier as stored in `20_scenario_catalog.csv`
        (e.g. ``"baseline2025"``, ``"adverse2025"``).
    date
        ISO‑8601 date string (``"YYYY-MM-DD"``) *or* a `datetime` object.

    Returns
    -------
    pd.DataFrame
        Sub‑DataFrame of `macro_scenarios` validated by ``MACRO_SCHEMA``.

    Raises
    ------
    KeyError
        If no rows match the provided `scenario_id` and `date`.
    SchemaError
        Propagated from :pyfunc:`fincontagion.io_utils.load` if the source CSV
        fails schema validation.

    Notes
    -----
    The returned DataFrame preserves **original index ordering**; callers should
    `.reset_index(drop=True)` if positional indexing is required.
    """
    macro_df = load("macro_scenarios")  # may raise SchemaError

    # normalise date argument to pandas.Timestamp for comparison
    date_ts = pd.to_datetime(date).normalize()

    mask = (macro_df[_SCENARIO] == scenario_id) & (
        macro_df[_DATE] == date_ts
    )
    sub = macro_df.loc[mask]

    if sub.empty:
        raise KeyError(
            f"No rows found for scenario '{scenario_id}' on date '{date_ts.date()}'."
        )

    # Extra safety: re‑validate slice (cheap, so we keep it)
    try:
        return MACRO_SCHEMA.validate(sub, lazy=True)
    except pa.errors.SchemaError as err:  # pragma: no cover  (should not happen)
        raise SchemaError("Slice failed MACRO_SCHEMA validation") from err
