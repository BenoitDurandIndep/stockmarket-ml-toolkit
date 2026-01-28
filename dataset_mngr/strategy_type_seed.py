"""Insert strategy types into STRATEGY_TYPE table."""
from __future__ import annotations
import sys
from pathlib import Path

from sqlalchemy.orm import sessionmaker

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "dataset_mngr").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if (PROJECT_ROOT / "dataset_mngr").exists():
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dataset_mngr.db_models import StrategyType
from dataset_mngr.sqlite_io import get_connection, insert_object
from backtest.strategy_registry import StrategyTypeSpec


def upsert_strategy_type(session, payload: dict):
    payload_lower = {key.lower(): value for key, value in payload.items()}
    existing = session.query(StrategyType).filter(StrategyType.name == payload_lower["name"]).first()
    if existing is None:
        insert_object(session, StrategyType, payload_lower)
        return
    for key, value in payload_lower.items():
        setattr(existing, key, value)
    session.commit()

# Predefined strategy types to insert for testing
STRATEGY_TYPES = [
    StrategyTypeSpec(
        name="Threshold v1",
        description="Single model threshold entry/exit",
        code_entry="backtest.strategy_rules:entry_signal_threshold",
        code_exit="backtest.strategy_rules:exit_signal_threshold",
        param_entry={"entry_threshold": 4},
        param_exit={"exit_threshold": 0},
    ),
    StrategyTypeSpec(
        name="Dual Threshold v1",
        description="Two-model combined thresholds",
        code_entry="backtest.strategy_rules:entry_signal_dual_threshold",
        code_exit="backtest.strategy_rules:exit_signal_dual_threshold",
        param_entry={"entry_threshold": 4, "entry_threshold_2": 4},
        param_exit={"exit_threshold": 0, "exit_threshold_2": 0},
    ),
]


def main():
    con = get_connection()
    if con is None:
        raise RuntimeError("Database connection failed")
    session = sessionmaker(bind=con)()
    for spec in STRATEGY_TYPES:
        upsert_strategy_type(session, spec.to_db_payload())
    session.close()
    con.close()


if __name__ == "__main__":
    main()
