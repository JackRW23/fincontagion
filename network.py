"""
fincontagion.network
====================

I/O‑agnostic converters that turn the **processed exposures table** into
algorithm‑friendly data structures:

* NumPy liability matrix  L[i, j]  =  euros *borrower i* owes *lender j*
* NetworkX `DiGraph`  with edge attribute  'weight'

These utilities sit between the data‑layer (CSV → DataFrame) and the
algorithm‑layer (clearing solver, graph metrics).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import networkx as nx

from .datamodel import Array
from .errors import SchemaError

# --------------------------------------------------------------------------- #
# Column constants (avoid magic strings)
# --------------------------------------------------------------------------- #
_LENDER: Final[str] = "lender_id"
_BORROWER: Final[str] = "borrower_id"
_WEIGHT: Final[str] = "exposure_ead_eur"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def to_matrix(exposures: pd.DataFrame, banks: pd.DataFrame) -> Array:
    """
    Convert exposures into a square **liabilities** matrix.

    Parameters
    ----------
    exposures
        DataFrame validated by  fincontagion.io_utils.load("exposures").
    banks
        DataFrame validated by  fincontagion.io_utils.load("banks").
        Determines the ordering of rows / columns.

    Returns
    -------
    np.ndarray
        Matrix  L  with shape (n, n), where  n = len(banks_sorted)  and
        where  L[i, j]  equals *euros borrower i owes lender j*.

    Raises
    ------
    SchemaError
        If `exposures` contains a bank_id not present in `banks`.
    """
    bank_ids = banks["bank_id"].sort_values().to_numpy()
    n = len(bank_ids)
    id_to_idx = {bid: idx for idx, bid in enumerate(bank_ids)}

    # Sanity‑check: every id appearing in exposures must exist in banks
    unknown_ids = (
        set(exposures[_LENDER]).union(exposures[_BORROWER]) - set(bank_ids)
    )
    if unknown_ids:
        raise SchemaError(
            f"Exposure table references unknown bank_id(s): {sorted(unknown_ids)}"
        )

    L = np.zeros((n, n), dtype=float)

    # Vectorised assembly via grouping
    row_idx = exposures[_BORROWER].map(id_to_idx)
    col_idx = exposures[_LENDER].map(id_to_idx)
    np.add.at(
        L,
        (row_idx.to_numpy(), col_idx.to_numpy()),
        exposures[_WEIGHT].to_numpy(float),
    )
    return L


def to_graph(exposures: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph where each edge carries a 'weight' attribute.

    Parameters
    ----------
    exposures
        The same validated DataFrame as above.

    Returns
    -------
    networkx.DiGraph
        Nodes are **all** unique lender & borrower ids.
        Edges: borrower ➜ lender   (direction of obligation).
    """
    G = nx.DiGraph()
    # Add all nodes so isolated banks are retained
    all_nodes = pd.unique(
        exposures[[_LENDER, _BORROWER]].to_numpy().ravel()
    )
    G.add_nodes_from(all_nodes)

    # Each CSV row becomes one edge; multiple exposures summed automatically
    for row in exposures.itertuples(index=False):
        b = getattr(row, _BORROWER)
        l = getattr(row, _LENDER)
        w = getattr(row, _WEIGHT)
        if G.has_edge(b, l):
            G[b][l]["weight"] += w
        else:
            G.add_edge(b, l, weight=w)

    return G
