"""
experiments.run_grid
====================

Command‑line utility to sweep α, ρ and κ across a user‑specified grid, run the
contagion simulator in parallel, and append the results to
`data/processed/20_simulation_runs.csv`.

Example
-------
python -m experiments.run_grid grid \
    0.1,0.2 0.5,1.0 0.01,0.02 --config-path config.yaml --procs 4
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import typer
import yaml

# Library entry‑point we drive.
from simulate import run as simulate_run  # noqa: WPS433 (external import is intended)

# --------------------------------------------------------------------------- #
# CLI set‑up
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Sweep α, ρ, κ grids and run simulations.")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "20_simulation_runs.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_grid(arg: str) -> List[float]:
    """Parse a comma‑separated list of floats from the command‑line."""
    try:
        vals = [float(x) for x in arg.split(",") if x.strip()]
        if not vals:
            raise ValueError
        return vals
    except ValueError as exc:  # pragma: no cover
        raise typer.BadParameter(
            f"Grid '{arg}' must be a comma‑separated list of numbers."
        ) from exc


def _worker(
    args: Tuple[float, float, float, str, str],
) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """
    Run **one** simulation in a separate process.

    Parameters
    ----------
    args
        (alpha, rho, kappa, base_yaml_text, tmp_dir)

    Returns
    -------
    pd.Series
        Row returned by `simulate.run`.
    tuple
        The (alpha, rho, kappa) triple – handy when we concat later.
    """
    alpha, rho, kappa, base_yaml, tmp_dir = args
    cfg = yaml.safe_load(base_yaml)

    # Accept both flat and nested ("parameters") layouts.
    for key, val in (("alpha", alpha), ("rho", rho), ("kappa", kappa)):
        if key in cfg:
            cfg[key] = val
        else:
            cfg.setdefault("parameters", {})[key] = val

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="cfg_", dir=tmp_dir, delete=False
    ) as fh:
        yaml.safe_dump(cfg, fh)
        cfg_path = fh.name

    result_series = simulate_run(cfg_path)
    return result_series, (alpha, rho, kappa)


# --------------------------------------------------------------------------- #
# CLI command
# --------------------------------------------------------------------------- #
@app.command("grid")
def grid(  # noqa: D401 (imperative mood is fine for CLI verbs)
    α: str = typer.Argument(
        ...,
        metavar="ALPHA_GRID",
        help="Comma‑separated list of α values, e.g. '0.1,0.2'.",
    ),
    ρ: str = typer.Argument(
        ...,
        metavar="RHO_GRID",
        help="Comma‑separated list of ρ values, e.g. '0.5,1'.",
    ),
    κ: str = typer.Argument(
        ...,
        metavar="KAPPA_GRID",
        help="Comma‑separated list of κ values, e.g. '0.01,0.02'.",
    ),
    config_path: Path = typer.Option(
        "config.yaml",
        "--config-path",
        "-c",
        exists=True,
        readable=True,
        help="Base YAML configuration file.",
    ),
    procs: int = typer.Option(
        mp.cpu_count(),
        "--procs",
        "-p",
        min=1,
        help="Number of parallel worker processes.",
    ),
    output_csv: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Destination CSV (defaults to data/processed/20_simulation_runs.csv).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Construct the grid and exit without running simulations.",
    ),
) -> None:
    """
    Sweep the Cartesian product of *α*, *ρ* and *κ*, running the simulator once
    for every grid‑point.
    """
    alpha_vals = _parse_grid(α)
    rho_vals = _parse_grid(ρ)
    kappa_vals = _parse_grid(κ)

    combos: List[Tuple[float, float, float]] = list(
        itertools.product(alpha_vals, rho_vals, kappa_vals)
    )
    typer.echo(f"Parameter grid size: {len(combos)}")

    if dry_run:
        typer.echo("Dry‑run complete — no simulations executed.")
        raise typer.Exit()

    base_yaml_text = config_path.read_text()

    with tempfile.TemporaryDirectory() as tmp_dir:
        payload: Iterable[Tuple[float, float, float, str, str]] = (
            (*combo, base_yaml_text, tmp_dir) for combo in combos
        )

        with mp.Pool(processes=procs) as pool:
            results = pool.map(_worker, payload)

    # Unpack the results – first element is pd.Series, second is (α,ρ,κ)
    series_rows, param_rows = zip(*results)
    df_new = pd.concat(series_rows, axis=1).T.reset_index(drop=True)
    df_new["alpha"] = [p[0] for p in param_rows]
    df_new["rho"] = [p[1] for p in param_rows]
    df_new["kappa"] = [p[2] for p in param_rows]

    # Append (or create) the master CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        df_master = pd.read_csv(output_csv)
        df_master = pd.concat([df_master, df_new], ignore_index=True).drop_duplicates()
    else:
        df_master = df_new

    df_master.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(df_new)} new rows → {output_csv}")


if __name__ == "__main__":  # pragma: no cover
    app()
