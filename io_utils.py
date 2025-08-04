"""
fincontagion.io_utils
=====================

Thin I/O façade used by *all* downstream code:

    >>> from fincontagion.io_utils import load
    >>> banks = load("banks")                 # returns schema-validated DataFrame
    >>> exposures = load("exposures", chunksize=100_000)  # large-file support

If the CSV (or its header/dtypes) violates the Pandera schema declared in
:pyfile:`fincontagion.datamodel`, a :class:`fincontagion.errors.SchemaError`
is raised *immediately*, preventing silent propagation of bad data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Final

import pandas as pd
import pandera as pa

from .datamodel import (
    BANKS_SCHEMA,
    EXPOSURES_SCHEMA,
    ASSETS_SCHEMA,
    PORTFOLIOS_SCHEMA,
    MACRO_SCHEMA,
    SCEN_CATALOG_SCHEMA,
)
from .errors import SchemaError

# --------------------------------------------------------------------------- #
# File registry (single source of file names)
# --------------------------------------------------------------------------- #
_DATA_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "processed"
)

_SCHEMAS: dict[
    str, tuple[Path, pa.DataFrameSchema]
] = {
    "banks": (_DATA_DIR / "20_banks.csv", BANKS_SCHEMA),
    "exposures": (_DATA_DIR / "20_exposures.csv", EXPOSURES_SCHEMA),
    "asset_classes": (_DATA_DIR / "20_asset_classes.csv", ASSETS_SCHEMA),
    "portfolios": (_DATA_DIR / "20_portfolios.csv", PORTFOLIOS_SCHEMA),
    "macro_scenarios": (_DATA_DIR / "20_macro_scenarios.csv", MACRO_SCHEMA),
    "scenario_catalog": (_DATA_DIR / "20_scenario_catalog.csv", SCEN_CATALOG_SCHEMA),
}

Kind = Literal[
    "banks",
    "exposures",
    "asset_classes",
    "portfolios",
    "macro_scenarios",
    "scenario_catalog",
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load(kind: Kind, *, chunksize: int | None = None) -> pd.DataFrame:
    """
    Load one of the stage-20 processed artefacts and validate it.

    Parameters
    ----------
    kind
        One of the literals enumerated above.
    chunksize
        If provided, the CSV is streamed in chunks of this many rows,
        concatenated after validation **per chunk**.  Useful for very
        large exposures tables.

    Returns
    -------
    pd.DataFrame
        Guaranteed to satisfy the Pandera schema associated with *kind*.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file is missing.
    SchemaError
        If the file fails Pandera validation (wrong columns, dtypes, etc.).
    """
    try:
        path, schema = _SCHEMAS[kind]
    except KeyError as exc:  # defensive: unsupported kind
        raise KeyError(
            f"Unknown dataset kind '{kind}'. "
            f"Allowed values are {sorted(_SCHEMAS)}."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")

    # ------------------------------------------------------------------ #
    # Chunked or normal load
    # ------------------------------------------------------------------ #
    if chunksize:
        frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            try:
                frames.append(schema.validate(chunk, lazy=True))
            except pa.errors.SchemaErrors as err:
                raise SchemaError(
                    f"{path.name} failed schema validation on a chunk"
                ) from err
        df = pd.concat(frames, ignore_index=True)
    else:
        raw = pd.read_csv(path)
        try:
            df = schema.validate(raw, lazy=True)
        except pa.errors.SchemaErrors as err:
            raise SchemaError(f"{path.name} failed schema validation") from err

    return df
