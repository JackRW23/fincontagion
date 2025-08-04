"""
algorithms.metrics
==================

Utility functions that compute:

1. **Contagion amplification** – a single float measuring how much total
   loss exceeds the macro–only loss.
2. **Basic network statistics** – spectral radius, degree‑distribution
   inequality (Gini) and average path length for the directed exposures
   graph produced by :pymod:`network.py`.

Both helpers are pure functions with **no package‑internal side‑effects**
(i.e. no I/O, no global state) so they are straightforward to unit‑test.

-----------------------------------------------------------------------
Common input conventions
-----------------------------------------------------------------------
- ``total`` / ``macro_only`` are scalar loss amounts (same currency).
- ``G`` is a *weighted* :class:`networkx.DiGraph`; edge attribute
  ``'weight'`` holds EAD / exposure size.  Node labels are irrelevant.
"""

from __future__ import annotations

from typing import Dict

import math
import numpy as np
import networkx as nx


__all__ = ["contagion_amp", "graph_stats"]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def contagion_amp(total: float, macro_only: float) -> float:
    """
    Ratio of *total* loss to *macro‑only* loss.

    Returns
    -------
    float
        - **> 1**  ⇒ amplification (contagion increased losses)
        - **== 1** ⇒ no amplification
        - **< 1**  ⇒ total is less than macro‑only (should not happen
          in realistic settings, but handled gracefully).
        - ``math.inf`` when ``macro_only`` is zero and ``total`` > 0.

    Notes
    -----
    The function is deliberately *safe*:

    * If either input is ``nan`` the result is ``nan``.
    * If both inputs are zero the result is ``0.0`` (treated as *no*
      losses, hence no amplification).
    """
    if math.isnan(total) or math.isnan(macro_only):
        return float("nan")

    if macro_only == 0.0:
        # Avoid division‑by‑zero – infinite amplification unless both are 0
        return math.inf if total > 0.0 else 0.0

    return total / macro_only


def graph_stats(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute simple structural metrics of an exposures graph.

    Parameters
    ----------
    G
        Directed *weighted* graph whose adjacency matrix **A** is used
        throughout the simulation workflow.

    Returns
    -------
    dict[str, float]
        ``{
            "largest_eigenvalue": λ_max,
            "degree_gini":        Gini coefficient of out‑degree weights,
            "avg_path_len":       mean shortest‑path length (undirected)
        }``

    Behaviour & edge‑cases
    ----------------------
    * **λ_max** is the spectral radius  
      ``max(real(eigvals(A)))`` – matches ``np.linalg.eigvals`` test.
    * **degree_gini** ∈ [0, 1]; returns **0** for graphs with all zero
      weighted degrees, ``nan`` for empty graphs.
    * **avg_path_len**:
        * Uses the *largest* weakly connected component to avoid math
          domain errors on disconnected graphs.
        * Undirected distances are used (contagion is not orientation‑
          sensitive in this heuristic metric).
        * ``nan`` if the component has fewer than two nodes.
    """
    # ---------- Adjacency and spectral radius --------------------------------
    A = nx.to_numpy_array(G, weight="weight", dtype=float)
    if A.size:
        eigenvalues = np.linalg.eigvals(A)
        largest_ev = float(np.max(np.real(eigenvalues)))
    else:
        largest_ev = float("nan")

    # ---------- Degree Gini ---------------------------------------------------
    out_degrees = np.array([d for _, d in G.out_degree(weight="weight")],
                           dtype=float)
    degree_gini = _gini(out_degrees)

    # ---------- Average shortest path length ---------------------------------
    if len(G) < 2:
        avg_path = float("nan")
    else:
        # Work on *weakly* connected undirected projection
        H = nx.Graph(G)  # drops direction, preserves weights
        # Largest connected component
        largest_cc_nodes = max(nx.connected_components(H), key=len)
        H_cc = H.subgraph(largest_cc_nodes)
        if len(H_cc) < 2:
            avg_path = float("nan")
        else:
            # Weight‑agnostic avg. path length (matching toy‑graph test)
            avg_path = nx.average_shortest_path_length(H_cc)

    return {
        "largest_eigenvalue": largest_ev,
        "degree_gini": degree_gini,
        "avg_path_len": avg_path,
    }


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _gini(x: np.ndarray) -> float:
    """
    Classic Gini coefficient; 0 ⇒ perfectly equal, 1 ⇒ perfectly unequal.

    Implementation follows the vectorised formula from:
    *Ultsch & Lotsch (2017) ``arXiv:1702.02641``*.

    Assumes **non‑negative** inputs, which is true for weighted degrees.
    """
    if x.size == 0:
        return float("nan")

    if np.allclose(x, 0.0):
        return 0.0

    sorted_x = np.sort(x)                    # ascending
    n = x.size
    cum_x = np.cumsum(sorted_x, dtype=float)
    # Normalised Gini (division by mean * n)
    gini = (n + 1.0 - 2.0 * np.sum(cum_x) / cum_x[-1]) / n
    # Numerical guard
    return float(max(0.0, min(1.0, gini)))
