"""
fincontagion.errors
===================

Centralised custom exceptions for the **FinContagion** package.

Keeping all domain‑specific errors in one place guarantees that:

* every layer of the stack (I/O → network converters → algorithms → simulator)
  can signal failures in a uniform way;
* user code and test suites only need to import **one** base class
  (`FincontagionError`) to catch anything project‑specific;
* adding new specialised errors later (e.g. `CalibrationError`) is
  non‑breaking for downstream users already catching the base class.

None of the classes below add custom behaviour—just semantic clarity.
"""

from __future__ import annotations


class FincontagionError(Exception):
    """Base‑class for all FinContagion‐specific exceptions."""


class SchemaError(FincontagionError):
    """
    Raised by :pymod:`fincontagion.io_utils` when a CSV or YAML file fails
    :pydata:`pandera` schema validation.

    Typical causes
    --------------
    * Missing or miss‑spelled column headers
    * Wrong data‑type (e.g. string where float expected)
    * Values outside allowed category sets
    """


class SimulationError(FincontagionError):
    """
    Raised by algorithms or the simulator when mathematical routines cannot
    complete.

    Examples
    --------
    * Eisenberg–Noe fixed‑point fails to converge within *max_iter*
    * Dimension mismatch between liabilities matrix **L** and equity vector **E0**
    * Firesale loop diverges (prices < 0 or > 1)
    """

__all__ = ["FincontagionError", "SchemaError", "SimulationError"]
