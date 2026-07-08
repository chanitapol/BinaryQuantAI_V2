from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import pandas as pd


@dataclass
class RankingResult:
    hypothesis_id: str
    score: float
    winrate: float
    expectancy: float
    confidence: float
    stability: float
    occurrence: int
    gap: float
    status: str


class RankingEngine:
    """Score hypotheses using winrate, expectancy, confidence and stability."""

    def __init__(self, payout: float = 0.80, min_occurrence: int = 1000) -> None:
        self.payout = payout
        self.min_occurrence = min_occurrence

    def expectancy(self, winrate: float) -> float:
        return (winrate * self.payout) - ((1.0 - winrate) * 1.0)

    @staticmethod
    def confidence(winrate: float, occurrence: int) -> float:
        if occurrence <= 0:
            return 0.0
        return max(0.0, min(1.0, (winrate - 0.5) * sqrt(occurrence) * 2.0))

    @staticmethod
    def stability(train_winrate: float, validation_winrate: float, test_winrate: float) -> float:
        mean = (train_winrate + validation_winrate + test_winrate) / 3.0
        variance = (
            (train_winrate - mean) ** 2
            + (validation_winrate - mean) ** 2
            + (test_winrate - mean) ** 2
        ) / 3.0
        return 1.0 / (1.0 + sqrt(variance))

    def score(
        self,
        train_winrate: float,
        validation_winrate: float,
        test_winrate: float,
        occurrence: int,
        gap: float,
    ) -> tuple[float, float, float, float]:
        winrate = validation_winrate
        exp = self.expectancy(winrate)
        conf = self.confidence(winrate, occurrence)
        stab = self.stability(train_winrate, validation_winrate, test_winrate)

        # Weighted score with a penalty for large train/test divergence.
        score = (
            0.35 * winrate
            + 0.30 * max(exp, 0.0)
            + 0.20 * conf
            + 0.15 * stab
            - 0.50 * gap
        )
        return score, exp, conf, stab

    def rank_rows(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Rank a dataframe of experiment outcomes.

        Required columns:
        - hypothesis_id
        - train_winrate
        - validation_winrate
        - test_winrate
        - occurrence
        - gap
        - status
        """
        required = {
            "hypothesis_id",
            "train_winrate",
            "validation_winrate",
            "test_winrate",
            "occurrence",
            "gap",
            "status",
        }
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        rows = results_df.copy()
        scores = []
        expectancies = []
        confidences = []
        stabilities = []

        for _, row in rows.iterrows():
            score, exp, conf, stab = self.score(
                float(row["train_winrate"]),
                float(row["validation_winrate"]),
                float(row["test_winrate"]),
                int(row["occurrence"]),
                float(row["gap"]),
            )
            scores.append(score)
            expectancies.append(exp)
            confidences.append(conf)
            stabilities.append(stab)

        rows["score"] = scores
        rows["expectancy"] = expectancies
        rows["confidence"] = confidences
        rows["stability"] = stabilities
        rows = rows.sort_values(["score", "validation_winrate", "occurrence"], ascending=[False, False, False]).reset_index(drop=True)
        rows["rank"] = range(1, len(rows) + 1)
        return rows
