"""Utilities to load strategy functions by import path and manage params."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class StrategyTypeSpec:
    name: str
    description: str
    code_entry: str
    code_exit: str
    param_entry: Dict[str, Any]
    param_exit: Dict[str, Any]

    def to_db_payload(self) -> Dict[str, Any]:
        return {
            "NAME": self.name,
            "DESCRIPTION": self.description,
            "CODE_ENTRY": self.code_entry,
            "CODE_EXIT": self.code_exit,
            "PARAM_ENTRY": json.dumps(self.param_entry, ensure_ascii=False),
            "PARAM_EXIT": json.dumps(self.param_exit, ensure_ascii=False),
        }


def resolve_callable(path: str) -> Callable:
    """Resolve 'module.submodule:function' to a callable."""
    module_path, func_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Resolved object is not callable: {path}")
    return func


def load_params(param_json: str) -> Dict[str, Any]:
    return json.loads(param_json) if param_json else {}
