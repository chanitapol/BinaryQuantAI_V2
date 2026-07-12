from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


class KnowledgeQuery:
    """Read research.db within an optional run/generation scope."""

    def __init__(self, db_path: str = "research.db", run_id: int | None = None, generation: int | None = None) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.run_id = run_id
        self.generation = generation

    def close(self) -> None:
        self.conn.close()

    def set_scope(self, run_id: int | None = None, generation: int | None = None) -> None:
        self.run_id = run_id
        self.generation = generation

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def top_rankings(self, limit: int = 20) -> pd.DataFrame:
        sql = """
            SELECT r.run_id, r.generation, r.hypothesis_id, r.rank,
                   e.train_win AS train_winrate,
                   e.validation_win AS validation_winrate,
                   e.test_win AS test_winrate,
                   e.occurrence, e.expectancy, e.confidence,
                   e.stability, e.gap, e.score, e.status
            FROM rankings r
            JOIN experiments e
              ON e.run_id = r.run_id
             AND e.generation = r.generation
             AND e.hypothesis_id = r.hypothesis_id
            WHERE 1=1
        """
        params: list[object] = []
        if self.run_id is not None:
            sql += " AND r.run_id = ?"
            params.append(self.run_id)
        if self.generation is not None:
            sql += " AND r.generation = ?"
            params.append(self.generation)
        sql += " ORDER BY e.score DESC, r.rank ASC LIMIT ?"
        params.append(limit)
        return self._read(sql, tuple(params))

    def top_features(self, limit: int = 20) -> pd.DataFrame:
        from models.statistics_engine import StatisticsEngine
        stats = StatisticsEngine(str(self.db_path), run_id=self.run_id, generation=self.generation)
        try:
            return stats.top_features(limit)
        finally:
            stats.close()

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        from models.statistics_engine import StatisticsEngine
        stats = StatisticsEngine(str(self.db_path), run_id=self.run_id, generation=self.generation)
        try:
            return stats.best_thresholds(limit)
        finally:
            stats.close()

    def hypotheses(self, limit: int = 20) -> pd.DataFrame:
        return self._read(
            """SELECT id, signature, direction, conditions, feature_count, created_at
               FROM hypotheses ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )

    def hypotheses_by_ids(self, ids: list[str]) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in ids)
        return self._read(
            f"""SELECT id, signature, direction, conditions, feature_count, created_at
                FROM hypotheses WHERE id IN ({placeholders})""",
            tuple(ids),
        )

    def experiments(self, limit: int = 20) -> pd.DataFrame:
        sql = "SELECT * FROM experiments WHERE 1=1"
        params: list[object] = []
        if self.run_id is not None:
            sql += " AND run_id = ?"
            params.append(self.run_id)
        if self.generation is not None:
            sql += " AND generation = ?"
            params.append(self.generation)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return self._read(sql, tuple(params))

    def feature_summary(self, feature: str) -> pd.DataFrame:
        df = self.top_features(100000)
        return df[df["feature"] == feature].reset_index(drop=True) if not df.empty else df

    def threshold_summary(self, feature: str, operator: str) -> pd.DataFrame:
        df = self.best_thresholds(100000)
        if df.empty:
            return df
        return df[(df["feature"] == feature) & (df["operator"] == operator)].reset_index(drop=True)
