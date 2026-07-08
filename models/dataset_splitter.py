from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DatasetSplitter:
    """Split a feature dataframe by time order, not randomly."""

    def __init__(self, time_col: str = "timestamp") -> None:
        self.time_col = time_col

    def split(self, df: pd.DataFrame, train_ratio: float = 0.70, validation_ratio: float = 0.15) -> DatasetSplit:
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column: {self.time_col}")

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")
        if train_ratio + validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1")

        ordered = df.sort_values(self.time_col).reset_index(drop=True)
        n = len(ordered)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + validation_ratio))

        train = ordered.iloc[:train_end].copy().reset_index(drop=True)
        validation = ordered.iloc[train_end:valid_end].copy().reset_index(drop=True)
        test = ordered.iloc[valid_end:].copy().reset_index(drop=True)

        return DatasetSplit(train=train, validation=validation, test=test)


def split_dataframe(df: pd.DataFrame, train_ratio: float = 0.70, validation_ratio: float = 0.15) -> DatasetSplit:
    return DatasetSplitter().split(df, train_ratio=train_ratio, validation_ratio=validation_ratio)
