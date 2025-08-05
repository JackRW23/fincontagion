# FinContagion – Early Development Snapshot 🛠️

> *Systemic‑risk simulation suite for exploratory research & eventual thesis work*  
> **Status:** **ALPHA / proof‑of‑concept** – APIs and data contracts **WILL change**.

---

## Project Goals

| Objective | Brief description |
|-----------|-------------------|
| A | Load & validate synthetic banking data (Pandera schemas) |
| B | Implement core contagion algorithms (Eisenberg‑Noe clearing, DebtRank, fire‑sale feedback) |
| C | Orchestrate one full scenario (`simulator.engine.run`) |
| D | Wrap experiments / dashboard for interactive slicing |
| E | Enable CLI grid sweeps with multiprocessing |
| F | Ship a thin public API (`simulate.run`, `contagion.py`) for notebooks |

This repo currently satisfies **Objectives A–F for *toy 5‑bank data***.  
Scaling to real or larger data sets is future work (see Roadmap).

---

## Repository Layout (tree)
