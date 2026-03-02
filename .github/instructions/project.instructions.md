---
applyTo: "**"
---

# Project Rules — stockmarket-ml-toolkit

## Project Overview
This toolkit handles the full ML pipeline for stock market prediction and backtesting:
- **dataset_mngr/** — data fetching, indicator engineering, dataset splitting, ML model management, DB I/O.
- **backtest/** — custom portfolio/position engine, strategy rules, ML-driven strategies, strategy registry.
- **notebooks/** — exploratory analysis, model training, backtest reporting.
- **DB/** — DDL and migration scripts for SQLite / MariaDB.
- **tests/** — unit tests with pytest.

## Architecture Principles
- Module separation: keep data management (`dataset_mngr`) strictly decoupled from backtesting logic (`backtest`).
- Strategy rules (`strategy_rules.py`) must remain pure functions with no side effects; they operate only on DataFrames and settings dicts.
- The `Portfolio` / `Position` classes encapsulate all position state; never manipulate portfolio internals from outside these classes.
- DB access goes through SQLAlchemy ORM (`db_models.py`) or Peewee; raw SQL is reserved for migrations and reporting queries.

## ML & Data Conventions
- The random seed constant `RANDOM_KEY = 28` must be used for all sklearn/xgboost/lgbm random states.
- Feature engineering belongs in `add_indicators.py`; do not scatter indicator logic across notebooks.
- Model artefacts (fitted objects) are managed by `model_mngr.py` and `lgbm_mngr.py`.
- Dataset splits follow the helpers in `split_merge.py`; never implement ad-hoc train/test splits elsewhere.
- Supported ML frameworks: **scikit-learn**, **LightGBM**, **XGBoost**, **Keras (LSTM/Bidirectional)**.

## Backtesting Conventions
- Strategies are registered via `strategy_registry.py`; adding a new strategy means registering it there.
- Entry/exit signal functions follow the pattern in `strategy_rules.py`: accept `(df, models, settings)` and return a `pd.Series` of booleans.
- Commission is expressed as a percentage (float, e.g. `0.001` for 0.1%).
- All logging inside `Portfolio` must use the Python `logging` module, never `print`.

## Database
- Two supported backends: **SQLite** (local dev) and **MariaDB/MySQL** (production, via Docker).
- SQLAlchemy models are defined in `dataset_mngr/db_models.py`; do not duplicate model definitions.
- Migration scripts live in `DB/`.

## Testing
- Tests live in `tests/` and use **pytest**.
- Each public utility function in `dataset_mngr/` should have a corresponding test.
