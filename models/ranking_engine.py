from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

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
    """Score hypotheses using edge, consistency, confidence, and evidence size."""

    def __init__(self, payout: float = 0.80, min_occurrence: int = 1000, exploration_mode: bool = False) -> None:
        self.payout = payout
        self.min_occurrence = min_occurrence
        self.exploration_mode = exploration_mode

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

    def evidence_weight(self, occurrence: int) -> float:
        """Smoothly reward evidence size without letting tiny samples dominate."""
        if occurrence <= 0:
            return 0.0
        target = max(int(self.min_occurrence), 1)
        return max(0.0, min(1.0, sqrt(occurrence / target)))

    @staticmethod
    def status_priority(status: str) -> int:
        normalized = str(status).upper()
        if normalized == "PASS":
            return 2
        if normalized == "WATCH":
            return 1
        return 0

    @staticmethod
    def expectancy_score(expectancy: float) -> float:
        """Convert expectancy into a signed contribution.

        Positive expectancy helps; negative expectancy penalizes proportionally.
        """
        return expectancy

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
        evidence = self.evidence_weight(occurrence)
        exp_score = self.expectancy_score(exp)

        if self.exploration_mode:
            raw_score = (
                0.25 * winrate
                + 0.35 * exp_score
                + 0.15 * conf
                + 0.15 * stab
                + 0.10 * evidence
                - 0.25 * gap
            )
            score = raw_score * (0.35 + 0.65 * evidence)
        else:
            raw_score = (
                0.20 * winrate
                + 0.40 * exp_score
                + 0.15 * conf
                + 0.15 * stab
                + 0.10 * evidence
                - 0.50 * gap
            )
            score = raw_score * (0.25 + 0.75 * evidence)

        return score, exp, conf, stab

    def rank_rows(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Rank experiment outcomes with explicit PASS/WATCH/REJECT priority."""
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
        evidence_weights = []

        for _, row in rows.iterrows():
            occurrence = int(row["occurrence"])
            score, exp, conf, stab = self.score(
                float(row["train_winrate"]),
                float(row["validation_winrate"]),
                float(row["test_winrate"]),
                occurrence,
                float(row["gap"]),
            )
            scores.append(score)
            expectancies.append(exp)
            confidences.append(conf)
            stabilities.append(stab)
            evidence_weights.append(self.evidence_weight(occurrence))

        rows["score"] = scores
        rows["expectancy"] = expectancies
        rows["confidence"] = confidences
        rows["stability"] = stabilities
        rows["evidence_weight"] = evidence_weights
        rows["status_priority"] = rows["status"].map(self.status_priority)

        rows = rows.sort_values(
            ["status_priority", "score", "evidence_weight", "validation_winrate", "occurrence"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        rows["rank"] = range(1, len(rows) + 1)
        return rows