from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


class ExperimentRunner:
    """Evaluate hypotheses against a feature DataFrame.

    The runner interprets each hypothesis as a conjunction of conditions.
    For now the target is an exploratory proxy:
    - BUY hypothesis scores rows where next return is positive
    - SELL hypothesis scores rows where next return is negative

    This is an initial research runner, not a live trading engine.
    """

    def __init__(self, target_col: str = "return_1") -> None:
        self.target_col = target_col

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

    def evaluate(self, df: pd.DataFrame, hypothesis: Hypothesis) -> ExperimentResult:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe")

        mask = pd.Series(True, index=df.index)
        for condition in hypothesis.conditions:
            if condition.feature not in df.columns:
                raise KeyError(f"Missing feature column: {condition.feature}")
            mask &= self._mask_for_condition(df, condition)

        passed = df.loc[mask].copy()
        occurrence = int(len(passed))
        if occurrence == 0:
            return ExperimentResult(hypothesis.id, 0, 0.0, 0.0, 0)

        target = passed[self.target_col].astype(float)
        if hypothesis.direction.upper() == "SELL":
            target = -target

        wins = (target > 0).sum()
        winrate = float(wins / occurrence)
        expectancy = float(target.mean())
        return ExperimentResult(hypothesis.id, occurrence, winrate, expectancy, occurrence)
