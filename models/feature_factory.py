from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureFactoryConfig:
    asset_col: str = "asset"
    timeframe_col: str = "timeframe"
    timestamp_col: str = "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"


class FeatureFactory:
    """Generate a larger feature set from a cleaned OHLC dataframe."""

    def __init__(self, config: FeatureFactoryConfig | None = None) -> None:
        self.config = config or FeatureFactoryConfig()

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).rank(pct=True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.config.asset_col,
            self.config.timeframe_col,
            self.config.timestamp_col,
            self.config.open_col,
            self.config.high_col,
            self.config.low_col,
            self.config.close_col,
            self.config.volume_col,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy().sort_values([self.config.asset_col, self.config.timestamp_col]).reset_index(drop=True)
        g = out.groupby(self.config.asset_col, sort=False)

        o = out[self.config.open_col].astype(float)
        h = out[self.config.high_col].astype(float)
        l = out[self.config.low_col].astype(float)
        c = out[self.config.close_col].astype(float)
        v = out[self.config.volume_col].astype(float)

        # Candle anatomy
        out["body"] = c - o
        out["abs_body"] = out["body"].abs()
        out["range"] = h - l
        out["upper_wick"] = h - np.maximum(o, c)
        out["lower_wick"] = np.minimum(o, c) - l
        out["body_ratio"] = self._safe_div(out["abs_body"], out["range"])
        out["upper_wick_ratio"] = self._safe_div(out["upper_wick"], out["range"])
        out["lower_wick_ratio"] = self._safe_div(out["lower_wick"], out["range"])
        out["close_pos_in_range"] = self._safe_div(c - l, out["range"])
        out["open_pos_in_range"] = self._safe_div(o - l, out["range"])
        out["bull"] = (c > o).astype(np.int8)
        out["bear"] = (c < o).astype(np.int8)
        out["volume_log"] = np.log1p(v.fillna(0.0))
        out["range_log"] = np.log1p(out["range"].clip(lower=0.0))
        out["body_to_wick"] = self._safe_div(out["abs_body"], out[["upper_wick", "lower_wick"]].sum(axis=1))
        out["wick_imbalance"] = out["upper_wick"] - out["lower_wick"]
        out["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0))

        # Previous candle relationships
        out["prev_close"] = g[self.config.close_col].shift(1)
        out["prev_open"] = g[self.config.open_col].shift(1)
        out["prev_body"] = g["body"].shift(1)
        out["prev_range"] = g["range"].shift(1)
        out["prev_upper_wick"] = g["upper_wick"].shift(1)
        out["prev_lower_wick"] = g["lower_wick"].shift(1)
        out["gap"] = o - out["prev_close"]
        out["return_1"] = self._safe_div(c - out["prev_close"], out["prev_close"])
        out["return_3"] = g[self.config.close_col].transform(lambda s: s.pct_change(3))
        out["return_5"] = g[self.config.close_col].transform(lambda s: s.pct_change(5))

        # Rolling volatility / trend
        out["rolling_mean_5"] = g[self.config.close_col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        out["rolling_mean_10"] = g[self.config.close_col].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_mean_20"] = g[self.config.close_col].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["rolling_std_10"] = g[self.config.close_col].transform(lambda s: s.rolling(10, min_periods=1).std())
        out["rolling_std_20"] = g[self.config.close_col].transform(lambda s: s.rolling(20, min_periods=1).std())
        out["rolling_range_10"] = g["range"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_range_20"] = g["range"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["rolling_body_10"] = g["abs_body"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_body_20"] = g["abs_body"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["volatility_20"] = self._safe_div(out["rolling_std_20"], out["rolling_mean_20"].abs())
        out["momentum_3"] = g[self.config.close_col].transform(lambda s: s.diff(3))
        out["momentum_5"] = g[self.config.close_col].transform(lambda s: s.diff(5))
        out["trend_5"] = self._safe_div(c - out["rolling_mean_5"], out["rolling_mean_5"])
        out["trend_20"] = self._safe_div(c - out["rolling_mean_20"], out["rolling_mean_20"])
        out["atr_14"] = g["range"].transform(lambda s: s.rolling(14, min_periods=1).mean())

        # Candle regime / structure
        out["higher_high"] = (h > g[self.config.high_col].shift(1)).astype(np.int8)
        out["lower_low"] = (l < g[self.config.low_col].shift(1)).astype(np.int8)
        out["inside_bar"] = ((h <= g[self.config.high_col].shift(1)) & (l >= g[self.config.low_col].shift(1))).astype(np.int8)
        out["expansion_bar"] = (out["range"] > out["rolling_range_20"] * 1.5).astype(np.int8)
        out["small_body"] = (out["body_ratio"] < 0.25).astype(np.int8)
        out["large_body"] = (out["body_ratio"] > 0.75).astype(np.int8)
        out["doji_like"] = (out["body_ratio"] < 0.10).astype(np.int8)
        out["pinbar_bull"] = ((out["lower_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.30) & (out["bull"] == 1)).astype(np.int8)
        out["pinbar_bear"] = ((out["upper_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.30) & (out["bear"] == 1)).astype(np.int8)
        out["engulf_bull"] = ((out["bull"] == 1) & (out["prev_body"] < 0) & (out["body"].abs() > out["prev_body"].abs())).astype(np.int8)
        out["engulf_bear"] = ((out["bear"] == 1) & (out["prev_body"] > 0) & (out["body"].abs() > out["prev_body"].abs())).astype(np.int8)

        # Statistical regimes
        out["zclose_20"] = g[self.config.close_col].transform(lambda s: self._zscore(s, 20))
        out["zrange_20"] = self._zscore(out["range"], 20)
        out["zbody_20"] = self._zscore(out["abs_body"], 20)
        out["prange_20"] = self._percentile_rank(out["range"], 20)
        out["pbody_20"] = self._percentile_rank(out["abs_body"], 20)
        out["pvol_20"] = self._percentile_rank(v.fillna(0.0), 20)

        # Volume context
        out["volume_change_1"] = g[self.config.volume_col].transform(lambda s: s.pct_change(1))
        out["volume_change_5"] = g[self.config.volume_col].transform(lambda s: s.pct_change(5))
        out["volume_ma_10"] = g[self.config.volume_col].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["volume_ratio_10"] = self._safe_div(v, out["volume_ma_10"])

        # Time context
        ts = pd.to_datetime(out[self.config.timestamp_col], unit="s", errors="coerce")
        out["hour"] = ts.dt.hour.fillna(0).astype(int)
        out["minute"] = ts.dt.minute.fillna(0).astype(int)
        out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
        out["is_asia_session"] = out["hour"].between(0, 7).astype(np.int8)
        out["is_london_session"] = out["hour"].between(8, 15).astype(np.int8)
        out["is_newyork_session"] = out["hour"].between(13, 20).astype(np.int8)
        out["is_late_session"] = out["hour"].between(21, 23).astype(np.int8)

        # Cross-feature interactions
        out["body_x_volatility"] = out["body_ratio"] * out["volatility_20"].fillna(0.0)
        out["wick_x_range"] = out["upper_wick_ratio"] * self._safe_div(out["range"], out["rolling_range_20"])
        out["momentum_x_range"] = out["momentum_3"].fillna(0.0) * out["range"]
        out["trend_x_volume"] = out["trend_20"].fillna(0.0) * out["volume_ratio_10"].fillna(0.0)

        # Clean up only rows that are unusable for core candle fields.
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=[
            self.config.asset_col,
            self.config.timeframe_col,
            self.config.timestamp_col,
            self.config.open_col,
            self.config.high_col,
            self.config.low_col,
            self.config.close_col,
            self.config.volume_col,
        ]).reset_index(drop=True)

        # Fill remaining feature NaNs after rolling calculations so rows are preserved.
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].fillna(0)

        return out
