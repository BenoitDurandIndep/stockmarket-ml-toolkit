"""Strategy entry/exit functions for backtesting."""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

def _require_models(models: List, count: int) -> None:
    """Validate that enough models are provided.

    Args:
        models (List): List of model objects (must expose `predict_col`/`proba_col`).
        count (int): Minimum number of models required.

    Raises:
        ValueError: If the number of models is less than `count`.
    """
    if len(models) < count:
        raise ValueError(f"Expected at least {count} models, got {len(models)}")


def _thresholds(settings: Dict, base_key: str, count: int, default: float = 0) -> List[float]:
    """Build a list of thresholds from settings.

    Args:
        settings (Dict): Strategy settings dictionary.
        base_key (str): Base key name (e.g., "entry_threshold").
        count (int): Number of thresholds to extract.
        default (float, optional): Default value if a key is missing. Defaults to 0.

    Returns:
        List[float]: Thresholds in order for each model.
    """
    values = []
    for i in range(count):
        key = base_key if i == 0 else f"{base_key}_{i + 1}"
        values.append(settings.get(key, default))
    return values


def _signal_all(df: pd.DataFrame, cols: List[str], thresholds: List[float]) -> pd.Series:
    """Return True where all columns meet/exceed their thresholds.

    Args:
        df (pd.DataFrame): Data with model columns.
        cols (List[str]): Column names to evaluate.
        thresholds (List[float]): Threshold per column.

    Returns:
        pd.Series: Boolean entry signal.
    """
    mask = df[cols[0]] >= thresholds[0]
    for col, thr in zip(cols[1:], thresholds[1:]):
        mask = mask & (df[col] >= thr)
    return mask


def _signal_any_below(df: pd.DataFrame, cols: List[str], thresholds: List[float]) -> pd.Series:
    """Return True where any column is below/equal to its threshold.

    Args:
        df (pd.DataFrame): Data with model columns.
        cols (List[str]): Column names to evaluate.
        thresholds (List[float]): Threshold per column.

    Returns:
        pd.Series: Boolean exit signal.
    """
    mask = df[cols[0]] <= thresholds[0]
    for col, thr in zip(cols[1:], thresholds[1:]):
        mask = mask | (df[col] <= thr)
    return mask


def entry_signal_generic(
    df: pd.DataFrame,
    models: List,
    settings: Dict,
    count: int = 1,
    use_proba: bool = False,
) -> pd.Series:
    """Generic entry signal for 1..4 models with optional proba check.

    Args:
        df (pd.DataFrame): DataFrame containing model score/proba columns.
        models (List): List of model objects (uses models[0..count-1]).
        settings (Dict): Thresholds. For score: `entry_threshold`, `entry_threshold_2`, ...
                 For proba: `entry_proba_threshold`, `entry_proba_threshold_2`, ...
        count (int, optional): Number of models to use (1..4). Defaults to 1.
        use_proba (bool, optional): If True, require both score AND proba to meet thresholds.
                                    Defaults to False.

    Returns:
        pd.Series: Boolean entry signal.
    """
    _require_models(models, count)
    score_thresholds = _thresholds(settings, "entry_threshold", count)
    score_cols = [m.predict_col for m in models[:count]]
    score_mask = _signal_all(df, score_cols, score_thresholds)

    if not use_proba:
        return score_mask

    proba_thresholds = _thresholds(settings, "entry_proba_threshold", count)
    proba_cols = [m.proba_col for m in models[:count]]
    proba_mask = _signal_all(df, proba_cols, proba_thresholds)
    return score_mask & proba_mask


def exit_signal_generic(
    df: pd.DataFrame,
    models: List,
    settings: Dict,
    count: int = 1,
    use_proba: bool = False,
) -> pd.Series:
    """Generic exit signal for 1..4 models with optional proba check.

    Args:
        df (pd.DataFrame): DataFrame containing model score/proba columns.
        models (List): List of model objects (uses models[0..count-1]).
        settings (Dict): Thresholds. For score: `exit_threshold`, `exit_threshold_2`, ...
                         For proba: `exit_proba_threshold`, `exit_proba_threshold_2`, ...
        count (int, optional): Number of models to use (1..4). Defaults to 1.
        use_proba (bool, optional): If True, trigger exit on score OR proba threshold breach.
                                    Defaults to False.

    Returns:
        pd.Series: Boolean exit signal.
    """
    _require_models(models, count)
    score_thresholds = _thresholds(settings, "exit_threshold", count)
    score_cols = [m.predict_col for m in models[:count]]
    score_mask = _signal_any_below(df, score_cols, score_thresholds)

    if not use_proba:
        return score_mask

    proba_thresholds = _thresholds(settings, "exit_proba_threshold", count)
    proba_cols = [m.proba_col for m in models[:count]]
    proba_mask = _signal_any_below(df, proba_cols, proba_thresholds)
    return score_mask | proba_mask

