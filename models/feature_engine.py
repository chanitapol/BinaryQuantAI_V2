from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from models.feature_factory import FeatureFactory


@dataclass
class FeatureEngineConfig:
    db_path: str | Path = "database/binary_quant.db"
    table_name: str = "candles"
    asset_col: str = "asset"
    timeframe_col: str = "timeframe"
    timestamp_col: str = "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"


class FeatureEngine:
    """Load candles from SQLite, validate schema, and build a feature matrix."""

    REQUIRED_COLUMNS: tuple[str, ...] = ("asset", "timeframe", "timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        self.config = config or FeatureEngineConfig()
        self.conn: sqlite3.Connection | None = None
        self.factory = FeatureFactory()

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.config.db_path))
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def load_candles(self, limit: int | None = None) -> pd.DataFrame:
        conn = self.connect()
        sql = f"SELECT * FROM {self.config.table_name} ORDER BY {self.config.asset_col}, {self.config.timestamp_col}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return pd.read_sql_query(sql, conn)

    def validate_schema(self, df: pd.DataFrame) -> list[str]:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return list(self.REQUIRED_COLUMNS)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_schema(df)
        out = df.copy()
        out = out.sort_values([self.config.asset_col, self.config.timestamp_col]).reset_index(drop=True)

        for col in [
            self.config.timeframe_col,
            self.config.timestamp_col,
            self.config.open_col,
            self.config.high_col,
            self.config.low_col,
            self.config.close_col,
            self.config.volume_col,
        ]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=[self.config.asset_col, self.config.timestamp_col, self.config.open_col, self.config.high_col, self.config.low_col, self.config.close_col])
        return out

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare(df)
        enriched = self.factory.enrich(prepared)
        enriched = enriched.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return enriched

    def run(self, limit: int | None = None) -> pd.DataFrame:
        candles = self.load_candles(limit=limit)
        return self.build_features(candles)


def build_feature_frame(db_path: str | Path = "database/binary_quant.db", limit: int | None = None) -> pd.DataFrame:
    engine = FeatureEngine(FeatureEngineConfig(db_path=db_path))
    try:
        return engine.run(limit=limit)
    finally:
        engine.close()
