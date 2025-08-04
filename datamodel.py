"""
fincontagion.datamodel
======================

• Central **type–contract hub** for the FinContagion code‑base.  
• Declares Pandera schemas for every processed CSV you already created.  
• Hosts small enumerations and a NumPy `Array` alias so all downstream
  modules share the *same* typing vocabulary.

Nothing in here performs I/O or heavy computation; the file is imported
for its constants only, keeping circular‑import risk near zero.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Universal type alias
# --------------------------------------------------------------------------- #
import numpy as np

Array = np.ndarray           # <-- tool‑friendly shorthand used everywhere

# --------------------------------------------------------------------------- #
# Enumerations (kept minimal; extend only if new categories appear)
# --------------------------------------------------------------------------- #
from enum import Enum


class Seniority(str, Enum):
    secured = "secured"
    unsecured = "unsecured"
    subordinated = "subordinated"


class LiquidityBucket(str, Enum):
    HQLA = "HQLA"
    Level2 = "Level2"
    Illiquid = "Illiquid"


class BusinessModel(str, Enum):
    retail = "retail"
    universal = "universal"
    investment = "investment"
    cooperative = "cooperative"
    other = "other"

# --------------------------------------------------------------------------- #
# Data‑frame schemas (Pandera) that mirror the seven processed artefacts
# --------------------------------------------------------------------------- #
import pandera as pa
from pandera import Column, DataFrameSchema, Check


BANKS_SCHEMA = DataFrameSchema(
    {
        "bank_id": Column(int, Check.greater_than(0), unique=True),
        "name": Column(str),
        "country_iso2": Column(str, Check.str_length(2, 2)),
        "total_assets_eur": Column(float, Check.greater_than(0)),
        "rwa_eur": Column(float, Check.greater_than(0)),
        "tier1_capital_eur": Column(float, Check.greater_than(0)),
        "equity_eur": Column(float, Check.greater_than(0)),
        "leverage_ratio": Column(float, Check.in_range(0, 1)),
        "business_model": Column(
            pa.Category, checks=Check.isin([m.value for m in BusinessModel])
        ),
    },
    strict=True,
    coerce=True,
)

EXPOSURES_SCHEMA = DataFrameSchema(
    {
        "lender_id": Column(int),
        "borrower_id": Column(int),
        "exposure_ead_eur": Column(float, Check.greater_than(0)),
        "seniority": Column(
            pa.Category, checks=Check.isin([s.value for s in Seniority])
        ),
        "maturity_days": Column(int, Check.greater_than_or_equal_to(0)),
        "int_rate_bps": Column(int),
    },
    strict=True,
    coerce=True,
)

ASSETS_SCHEMA = DataFrameSchema(
    {
        "asset_class_id": Column(int, unique=True),
        "asset_class_name": Column(str),
        "liquidity_bucket": Column(
            pa.Category, checks=Check.isin([b.value for b in LiquidityBucket])
        ),
        "base_volatility_pct": Column(float, Check.greater_than(0)),
    },
    strict=True,
    coerce=True,
)

PORTFOLIOS_SCHEMA = DataFrameSchema(
    {
        "bank_id": Column(int),
        "asset_class_id": Column(int),
        "asset_class_name": Column(str),
        "market_value_eur": Column(float, Check.greater_than_or_equal_to(0)),
        "haircut_pct": Column(float, Check.in_range(0, 100)),
        "lambda_price_impact": Column(float, Check.greater_than(0)),
    },
    strict=True,
    coerce=True,
)

MACRO_SCHEMA = DataFrameSchema(
    {
        "scenario_id": Column(str),
        "date": Column(pa.DateTime),
        "asset_class_id": Column(int),
        "price_shock_pct": Column(float),
        "risk_free_shift_bps": Column(int),
        "pd_shift_bps": Column(int),
    },
    strict=True,
    coerce=True,
)

SCEN_CATALOG_SCHEMA = DataFrameSchema(
    {
        "scenario_id": Column(str, unique=True),
        "label": Column(str),
        "source": Column(str),
        "start_date": Column(pa.DateTime),
        "horizon_days": Column(int),
        "description": Column(str),
    },
    strict=True,
    coerce=True,
)

# --------------------------------------------------------------------------- #
# What this module exports
# --------------------------------------------------------------------------- #
__all__ = [
    "Array",
    "Seniority",
    "LiquidityBucket",
    "BusinessModel",
    "BANKS_SCHEMA",
    "EXPOSURES_SCHEMA",
    "ASSETS_SCHEMA",
    "PORTFOLIOS_SCHEMA",
    "MACRO_SCHEMA",
    "SCEN_CATALOG_SCHEMA",
]
