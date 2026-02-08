"""Insert strategy types into STRATEGY_TYPE table."""
from __future__ import annotations
import sys
from pathlib import Path
import json

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

def load_strategy_types(spec_path: Path) -> list[StrategyTypeSpec]:
    if not spec_path.exists():
        raise FileNotFoundError(f"strategy types file not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("strategy types payload must be a list of objects")

    specs: list[StrategyTypeSpec] = []
    for item in payload:
        specs.append(
            StrategyTypeSpec(
                name=item["name"],
                description=item.get("description", ""),
                code_entry=item["code_entry"],
                code_exit=item["code_exit"],
                param_entry=item.get("param_entry", {}),
                param_exit=item.get("param_exit", {}),
            )
        )
    return specs


def main():
    spec_path = Path(__file__).resolve().parent / "strategy_types.json"
    specs = load_strategy_types(spec_path)

    con = get_connection()
    if con is None:
        raise RuntimeError("Database connection failed")
    session = sessionmaker(bind=con)()
    for spec in specs:
        upsert_strategy_type(session, spec.to_db_payload())
    session.close()
    con.close()


if __name__ == "__main__":
    main()
