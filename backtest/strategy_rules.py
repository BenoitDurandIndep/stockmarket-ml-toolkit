"""Strategy entry/exit functions for backtesting."""
from __future__ import annotations

from typing import Dict, List
import pandas as pd


def entry_signal_threshold(df: pd.DataFrame, models, settings: Dict) -> pd.Series:
    """Entry when model prediction exceeds threshold.

    Expected in settings:
        - entry_threshold (int|float)

    Uses first model in `models` list.
    """
    threshold = settings.get("entry_threshold", 0)
    return df[models[0].predict_col] >= threshold


def exit_signal_threshold(df: pd.DataFrame, models, settings: Dict) -> pd.Series:
    """Exit when model prediction below threshold.

    Expected in settings:
        - exit_threshold (int|float)

    Uses first model in `models` list.
    """
    threshold = settings.get("exit_threshold", 0)
    return df[models[0].predict_col] <= threshold


def entry_signal_dual_threshold(df: pd.DataFrame, models, settings: Dict) -> pd.Series:
    """Entry when model 0 >= entry_threshold and model 1 >= entry_threshold_2.

    Expected in settings:
        - entry_threshold (int|float)
        - entry_threshold_2 (int|float)
    """
    thr_1 = settings.get("entry_threshold", 0)
    thr_2 = settings.get("entry_threshold_2", 0)
    return (df[models[0].predict_col] >= thr_1) & (df[models[1].predict_col] >= thr_2)


def exit_signal_dual_threshold(df: pd.DataFrame, models, settings: Dict) -> pd.Series:
    """Exit when model 0 <= exit_threshold or model 1 <= exit_threshold_2.

    Expected in settings:
        - exit_threshold (int|float)
        - exit_threshold_2 (int|float)
    """
    thr_1 = settings.get("exit_threshold", 0)
    thr_2 = settings.get("exit_threshold_2", 0)
    return (df[models[0].predict_col] <= thr_1) | (df[models[1].predict_col] <= thr_2)
