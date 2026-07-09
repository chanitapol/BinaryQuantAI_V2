from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


@dataclass
class KnowledgeQueryResult:
    rows: list[dict[str, Any]]


class KnowledgeQuery:
    """Convenience query layer for research.db."""

    def __init__(self, db_path: str = "research.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def top_rankings(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """
            SELECT *
            FROM rankings
            ORDER BY score DESC, rank ASC
            LIMIT ?
            """,
            (limit,),
        )

    def top_features(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """
            SELECT feature, total, passed, rejected, avg_score, avg_winrate
            FROM feature_statistics
            ORDER BY avg_score DESC, avg_winrate DESC, total DESC
            LIMIT ?
            """,
            (limit,),
        )

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """
            SELECT feature, operator, threshold, total, passed, avg_score, avg_expectancy
            FROM threshold_statistics
            ORDER BY avg_score DESC, avg_expectancy DESC, total DESC
            LIMIT ?
            """,
            (limit,),
        )

    def hypotheses(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """
            SELECT id, signature, direction, conditions, feature_count, created_at
            FROM hypotheses
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def experiments(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """
            SELECT *
            FROM experiments
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def feature_summary(self, feature: str) -> pd.DataFrame:
        return self._read(
            """
            SELECT *
            FROM feature_statistics
            WHERE feature = ?
            """,
            (feature,),
        )

    def threshold_summary(self, feature: str, operator: str) -> pd.DataFrame:
        return self._read(
            """
            SELECT *
            FROM threshold_statistics
            WHERE feature = ? AND operator = ?
            ORDER BY avg_score DESC, total DESC
            """,
            (feature, operator),
        )
