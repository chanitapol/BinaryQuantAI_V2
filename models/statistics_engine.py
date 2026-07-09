from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FeatureStat:
    feature: str
    total: int
    passed: int
    rejected: int
    avg_score: float
    avg_winrate: float


@dataclass
class ThresholdStat:
    feature: str
    operator: str
    threshold: str
    total: int
    passed: int
    avg_score: float
    avg_expectancy: float


class StatisticsEngine:
    """Read research.db and aggregate experiment statistics."""

    def __init__(self, db_path: str = "research.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def _experiments_df(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM experiments", self.conn)

    def _hypotheses_df(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM hypotheses", self.conn)

    @staticmethod
    def _extract_conditions(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                conditions = pd.io.json.loads(row["conditions"])
            except Exception:
                conditions = []
            for cond in conditions:
                value = cond.get("value")
                if isinstance(value, (list, tuple)):
                    value = jsonable(value)
                rows.append(
                    {
                        "hypothesis_id": row["id"],
                        "feature": cond.get("feature"),
                        "operator": cond.get("operator"),
                        "threshold": str(value),
                    }
                )
        return pd.DataFrame(rows)

    def feature_statistics(self) -> pd.DataFrame:
        ex = self._experiments_df()
        hy = self._hypotheses_df()
        if ex.empty or hy.empty:
            return pd.DataFrame(columns=["feature", "total", "passed", "rejected", "avg_score", "avg_winrate"])

        merged = ex.merge(hy[["id", "conditions"]], left_on="hypothesis_id", right_on="id", how="left")
        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            try:
                conditions = pd.io.json.loads(row["conditions"])
            except Exception:
                conditions = []
            features = sorted({c.get("feature") for c in conditions if c.get("feature") is not None})
            if not features:
                continue
            for feature in features:
                rows.append(
                    {
                        "feature": feature,
                        "status": row.get("status"),
                        "score": row.get("score", 0.0),
                        "winrate": row.get("validation_winrate", 0.0),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["feature", "total", "passed", "rejected", "avg_score", "avg_winrate"])

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
            return pd.DataFrame(columns=["feature", "operator", "threshold", "total", "passed", "avg_score", "avg_expectancy"])

        merged = ex.merge(hy[["id", "conditions"]], left_on="hypothesis_id", right_on="id", how="left")
        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            try:
                conditions = pd.io.json.loads(row["conditions"])
            except Exception:
                conditions = []
            for cond in conditions:
                feature = cond.get("feature")
                if feature is None:
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "operator": cond.get("operator"),
                        "threshold": str(cond.get("value")),
                        "status": row.get("status"),
                        "score": row.get("score", 0.0),
                        "expectancy": row.get("expectancy", 0.0),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["feature", "operator", "threshold", "total", "passed", "avg_score", "avg_expectancy"])

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


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value
