from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


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
    """Load candles from SQLite and build a feature matrix from OHLC data."""

    REQUIRED_COLUMNS: tuple[str, ...] = ("asset", "timeframe", "timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        self.config = config or FeatureEngineConfig()
        self.conn: sqlite3.Connection | None = None

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

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_schema(df)

        out = df.copy()
        out = out.sort_values([self.config.asset_col, self.config.timestamp_col]).reset_index(drop=True)

        # Force numeric columns to numeric dtypes
        for col in [self.config.timeframe_col, self.config.timestamp_col, self.config.open_col, self.config.high_col, self.config.low_col, self.config.close_col, self.config.volume_col]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=[self.config.asset_col, self.config.timestamp_col, self.config.open_col, self.config.high_col, self.config.low_col, self.config.close_col])

        o = out[self.config.open_col].astype(float)
        h = out[self.config.high_col].astype(float)
        l = out[self.config.low_col].astype(float)
        c = out[self.config.close_col].astype(float)
        v = out[self.config.volume_col].astype(float)

        out["body"] = c - o
        out["abs_body"] = out["body"].abs()
        out["range"] = h - l
        out["upper_wick"] = h - np.maximum(o, c)
        out["lower_wick"] = np.minimum(o, c) - l
        out["body_ratio"] = self._safe_div(out["abs_body"], out["range"])
        out["upper_wick_ratio"] = self._safe_div(out["upper_wick"], out["range"])
        out["lower_wick_ratio"] = self._safe_div(out["lower_wick"], out["range"])
        out["close_pos_in_range"] = self._safe_div(c - l, out["range"])
        out["bull"] = (c > o).astype(np.int8)
        out["bear"] = (c < o).astype(np.int8)
        out["volume_log"] = np.log1p(v.fillna(0.0))

        group = out.groupby(self.config.asset_col, sort=False)
        out["prev_close"] = group[self.config.close_col].shift(1)
        out["prev_open"] = group[self.config.open_col].shift(1)
        out["prev_body"] = group["body"].shift(1)
        out["prev_range"] = group["range"].shift(1)
        out["return_1"] = self._safe_div(c - out["prev_close"], out["prev_close"])
        out["gap"] = o - out["prev_close"]

        tr = pd.concat(
            [
                out["range"],
                (h - out["prev_close"]).abs(),
                (l - out["prev_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["tr"] = tr
        out["atr_14"] = group["tr"].transform(lambda s: s.rolling(14, min_periods=1).mean())
        out["rolling_range_20"] = group["range"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["rolling_abs_body_20"] = group["abs_body"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["volatility_20"] = group["return_1"].transform(lambda s: s.rolling(20, min_periods=1).std())
        out["momentum_3"] = group[self.config.close_col].transform(lambda s: s.diff(3))
        out["momentum_5"] = group[self.config.close_col].transform(lambda s: s.diff(5))
        out["rolling_high_20"] = group[self.config.high_col].transform(lambda s: s.rolling(20, min_periods=1).max())
        out["rolling_low_20"] = group[self.config.low_col].transform(lambda s: s.rolling(20, min_periods=1).min())
        out["position_20"] = self._safe_div(c - out["rolling_low_20"], out["rolling_high_20"] - out["rolling_low_20"])
        out["wick_imbalance"] = out["upper_wick"] - out["lower_wick"]
        out["body_to_wick"] = self._safe_div(out["abs_body"], out[["upper_wick", "lower_wick"]].sum(axis=1))
        out["range_pct_20"] = self._safe_div(out["range"], out["rolling_range_20"])
        out["body_pct_20"] = self._safe_div(out["abs_body"], out["rolling_abs_body_20"])

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna().reset_index(drop=True)
        return out

    def run(self, limit: int | None = None) -> pd.DataFrame:
        candles = self.load_candles(limit=limit)
        return self.build_features(candles)


def build_feature_frame(db_path: str | Path = "database/binary_quant.db", limit: int | None = None) -> pd.DataFrame:
    engine = FeatureEngine(FeatureEngineConfig(db_path=db_path))
    try:
        return engine.run(limit=limit)
    finally:
        engine.close()
