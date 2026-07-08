from __future__ import annotations

from dataclasses import dataclass

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

    A row is treated as a trade setup. For BUY setups, the outcome is a win
    when the next close is above the entry close. For SELL setups, the outcome
    is a win when the next close is below the entry close.

    The current implementation uses the next candle as the expiry candle.
    """

    def __init__(self, close_col: str = "close", asset_col: str = "asset") -> None:
        self.close_col = close_col
        self.asset_col = asset_col

    @staticmethod
    def _mask_for_condition(df: pd.DataFrame, condition: Condition) -> pd.Series:
        series = df[condition.feature]
        op = condition.operator
        value = condition.value

        if op == ">":
            return series > value
        if op == ">=":
            return series >= value
        if op == "<":
            return series < value
        if op == "<=":
            return series <= value
        if op == "==":
            return series == value
        if op == "!=":
            return series != value
        if op == "between":
            low, high = value
            return series.between(low, high)
        raise ValueError(f"Unsupported operator: {op}")

    def _build_next_close(self, df: pd.DataFrame) -> pd.Series:
        if self.asset_col in df.columns and self.close_col in df.columns:
            return df.groupby(self.asset_col, sort=False)[self.close_col].shift(-1)
        return df[self.close_col].shift(-1)

    def evaluate(self, df: pd.DataFrame, hypothesis: Hypothesis) -> ExperimentResult:
        if self.close_col not in df.columns:
            raise ValueError(f"Close column '{self.close_col}' not found in dataframe")

        mask = pd.Series(True, index=df.index)
        for condition in hypothesis.conditions:
            if condition.feature not in df.columns:
                raise KeyError(f"Missing feature column: {condition.feature}")
            mask &= self._mask_for_condition(df, condition)

        passed = df.loc[mask].copy()
        occurrence = int(len(passed))
        if occurrence == 0:
            return ExperimentResult(hypothesis.id, 0, 0.0, 0.0, 0, 0, 0)

        next_close = self._build_next_close(passed)
        entry_close = passed[self.close_col].astype(float)
        direction = hypothesis.direction.upper()

        if direction == "SELL":
            diff = entry_close - next_close.astype(float)
        else:
            diff = next_close.astype(float) - entry_close

        diff = diff.dropna()
        occurrence = int(len(diff))
        if occurrence == 0:
            return ExperimentResult(hypothesis.id, 0, 0.0, 0.0, 0, 0, 0)

        wins = int((diff > 0).sum())
        losses = int((diff <= 0).sum())
        winrate = float(wins / occurrence)
        expectancy = float(diff.mean())
        return ExperimentResult(hypothesis.id, occurrence, winrate, expectancy, occurrence, wins, losses)
