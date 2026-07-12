from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = ["feature", "total", "passed", "rejected", "avg_score", "avg_winrate"]
THRESHOLD_COLUMNS = ["feature", "operator", "threshold", "total", "passed", "avg_score", "avg_expectancy"]


class StatisticsEngine:
    """Aggregate experiment statistics within an explicit research run scope."""

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

    def _experiments_df(self) -> pd.DataFrame:
        sql = "SELECT * FROM experiments WHERE 1=1"
        params: list[object] = []
        if self.run_id is not None:
            sql += " AND run_id = ?"
            params.append(self.run_id)
        if self.generation is not None:
            sql += " AND generation = ?"
            params.append(self.generation)
        return pd.read_sql_query(sql, self.conn, params=tuple(params))

    def _hypotheses_df(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT id, conditions FROM hypotheses", self.conn)

    @staticmethod
    def _conditions(value: object) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(value or "[]")
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def feature_statistics(self) -> pd.DataFrame:
        ex = self._experiments_df()
        hy = self._hypotheses_df()
        if ex.empty or hy.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        merged = ex.merge(hy, left_on="hypothesis_id", right_on="id", how="left")
        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            features = sorted({
                c.get("feature") for c in self._conditions(row.get("conditions"))
                if c.get("feature") is not None
            })
            for feature in features:
                rows.append({
                    "feature": feature,
                    "status": row.get("status"),
                    "score": row.get("score", 0.0),
                    "winrate": row.get("validation_win", 0.0),
                })
        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        tmp = pd.DataFrame(rows)
        out = (
            tmp.assign(passed=tmp["status"].eq("PASS").astype(int))
            .groupby("feature", as_index=False)
            .agg(
                total=("feature", "size"),
                passed=("passed", "sum"),
                rejected=("passed", lambda s: int((1 - s).sum())),
                avg_score=("score", "mean"),
                avg_winrate=("winrate", "mean"),
            )
        )
        return out.sort_values(["avg_score", "avg_winrate", "total"], ascending=[False, False, False])

    def threshold_statistics(self) -> pd.DataFrame:
        ex = self._experiments_df()
        hy = self._hypotheses_df()
        if ex.empty or hy.empty:
            return pd.DataFrame(columns=THRESHOLD_COLUMNS)

        merged = ex.merge(hy, left_on="hypothesis_id", right_on="id", how="left")
        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            for cond in self._conditions(row.get("conditions")):
                feature = cond.get("feature")
                if feature is None:
                    continue
                rows.append({
                    "feature": feature,
                    "operator": cond.get("operator"),
                    "threshold": json.dumps(cond.get("value"), sort_keys=True),
                    "status": row.get("status"),
                    "score": row.get("score", 0.0),
                    "expectancy": row.get("expectancy", 0.0),
                })
        if not rows:
            return pd.DataFrame(columns=THRESHOLD_COLUMNS)

        tmp = pd.DataFrame(rows)
        out = (
            tmp.assign(passed=tmp["status"].eq("PASS").astype(int))
            .groupby(["feature", "operator", "threshold"], as_index=False)
            .agg(
                total=("feature", "size"),
                passed=("passed", "sum"),
                avg_score=("score", "mean"),
                avg_expectancy=("expectancy", "mean"),
            )
        )
        return out.sort_values(["avg_score", "avg_expectancy", "total"], ascending=[False, False, False])

    def top_features(self, limit: int = 20) -> pd.DataFrame:
        return self.feature_statistics().head(limit)

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        return self.threshold_statistics().head(limit)
