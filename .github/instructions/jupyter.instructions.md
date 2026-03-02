---
applyTo: "**/*.ipynb"
---

# Jupyter Notebook Rules

## Structure
- Start every notebook with a **Markdown title cell** describing the notebook's purpose, date, and target symbol/model.
- Group cells into clearly labelled sections with Markdown headers: `## 1. Imports`, `## 2. Config`, `## 3. Data Loading`, etc.
- Place all configuration variables (paths, dates, hyperparameters) in a single **Config cell** near the top so they are easy to find and modify.

## Imports & Dependencies
- All imports go in the first code cell.
- Import project modules using their package paths (e.g. `from dataset_mngr.add_indicators import ...`) rather than adding `sys.path` hacks where possible.

## Reproducibility
- Set the random seed using `RANDOM_KEY = 28` (consistent with the project constant).
- Fix seeds for numpy, sklearn, lgbm, xgboost, and keras at the top of the notebook.

## Outputs
- Clear all outputs before committing a notebook (`Cell > All Outputs > Clear`).
- Exception: notebooks in `notebooks/backtest/` may retain final performance metric outputs for reference.
- Do not commit notebooks with large embedded plot data or model weights.

## Code Quality
- Notebooks are for exploration and reporting, not production logic.
- Refactor any reusable logic (indicator creation, model training helpers) into the appropriate module in `dataset_mngr/` or `backtest/` rather than keeping it only in a notebook.
- Avoid deeply nested cells; break long processing chains into intermediate cells with a brief comment.

## Paths
- Use `pathlib.Path` for all file paths.
- Reference the project root relatively so notebooks work across environments (local, Docker).

## Docker Notebooks
- Notebooks under `dockers/*/mnt/code/` are the Docker versions; keep them in sync with their counterparts in `notebooks/` when logic changes.
