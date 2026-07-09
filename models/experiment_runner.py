from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.hypothesis_engine import Hypothesis, Condition


@dataclass
class ExperimentResult:
    hypothesis_id: str
    passed_rows: int
    winrate: float
    expectancy: float
    occurrence: int
    wins: int
    losses: int


class ExperimentRunner:
    """Evaluate hypotheses using binary-outcome logic.

    This version avoids repeated pandas filtering inside the hot loop by caching
    NumPy masks per dataframe/condition and caching the next-close series per
    dataframe.
    """

    def __init__(self, close_col: str = "close", asset_col: str = "asset") -> None:
        self.close_col = close_col
        self.asset_col = asset_col
        self._mask_cache: dict[tuple[int, str, str, object], np.ndarray] = {}
        self._next_close_cache: dict[int, np.ndarray] = {}
        self._entry_close_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _normalize_value(value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    def _cache_key(self, df: pd.DataFrame, condition: Condition) -> tuple[int, str, str, object]:
        return (id(df), condition.feature, condition.operator, self._normalize_value(condition.value))

    @staticmethod
    def _mask_for_condition_array(values: np.ndarray, condition: Condition) -> np.ndarray:
        op = condition.operator
        value = condition.value

        if op == ">":
            return values > value
        if op == ">=":
            return values >= value
        if op == "<":
            return values < value
        if op == "<=":
            return values <= value
        if op == "==":
            return values == value
        if op == "!=":
            return values != value
        if op == "between":
            low, high = value
            return (values >= low) & (values <= high)
        raise ValueError(f"Unsupported operator: {op}")

    def _get_condition_mask(self, df: pd.DataFrame, condition: Condition) -> np.ndarray:
        key = self._cache_key(df, condition)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        if condition.feature not in df.columns:
            raise KeyError(f"Missing feature column: {condition.feature}")

        values = pd.to_numeric(df[condition.feature], errors="coerce").to_numpy(dtype=float, copy=False)
        mask = self._mask_for_condition_array(values, condition)
        self._mask_cache[key] = mask
        return mask

    def _get_next_close(self, df: pd.DataFrame) -> np.ndarray:
        key = id(df)
        cached = self._next_close_cache.get(key)
        if cached is not None:
            return cached

        if self.close_col not in df.columns:
            raise ValueError(f"Close column '{self.close_col}' not found in dataframe")

        close = pd.to_numeric(df[self.close_col], errors="coerce")
        if self.asset_col in df.columns:
            next_close = (
                df.assign(_close=close)
                .groupby(self.asset_col, sort=False)["_close"]
                .shift(-1)
                .to_numpy(dtype=float, copy=False)
            )
        else:
            next_close = close.shift(-1).to_numpy(dtype=float, copy=False)

        self._next_close_cache[key] = next_close
        return next_close

    def _get_entry_close(self, df: pd.DataFrame) -> np.ndarray:
        key = id(df)
        cached = self._entry_close_cache.get(key)
        if cached is not None:
            return cached

        if self.close_col not in df.columns:
            raise ValueError(f"Close column '{self.close_col}' not found in dataframe")

        entry = pd.to_numeric(df[self.close_col], errors="coerce").to_numpy(dtype=float, copy=False)
        self._entry_close_cache[key] = entry
        return entry

    def evaluate(self, df: pd.DataFrame, hypothesis: Hypothesis) -> ExperimentResult:
        next_close = self._get_next_close(df)
        entry_close = self._get_entry_close(df)

        mask = np.ones(len(df), dtype=bool)
        for condition in hypothesis.conditions:
            mask &= self._get_condition_mask(df, condition)

        passed_idx = np.flatnonzero(mask)
        if passed_idx.size == 0:
            return ExperimentResult(hypothesis.id, 0, 0.0, 0.0, 0, 0, 0)

        target_next = next_close[passed_idx]
        target_entry = entry_close[passed_idx]
        valid = ~np.isnan(target_next) & ~np.isnan(target_entry)
        if not np.any(valid):
            return ExperimentResult(hypothesis.id, 0, 0.0, 0.0, 0, 0, 0)

        target_next = target_next[valid]
        target_entry = target_entry[valid]
        passed_count = int(target_next.size)

        direction = hypothesis.direction.upper()
        if direction == "SELL":
            diff = target_entry - target_next
        else:
            diff = target_next - target_entry

        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff <= 0))
        winrate = float(wins / passed_count) if passed_count else 0.0
        expectancy = float(np.mean(diff)) if passed_count else 0.0
        return ExperimentResult(hypothesis.id, passed_count, winrate, expectancy, passed_count, wins, losses)
