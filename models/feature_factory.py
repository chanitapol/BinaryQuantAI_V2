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
    """Generate scale-aware price-action, indicator and semantic market features."""

    def __init__(self, config: FeatureFactoryConfig | None = None) -> None:
        self.config = config or FeatureFactoryConfig()

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.replace(0, np.nan)

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).rank(pct=True)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)

    @staticmethod
    def _true_range(group: pd.DataFrame, high_col: str, low_col: str, close_col: str) -> pd.Series:
        prev_close = group[close_col].shift(1)
        return pd.concat([
            group[high_col] - group[low_col],
            (group[high_col] - prev_close).abs(),
            (group[low_col] - prev_close).abs(),
        ], axis=1).max(axis=1)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        required = {cfg.asset_col, cfg.timeframe_col, cfg.timestamp_col, cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy().sort_values([cfg.asset_col, cfg.timestamp_col]).reset_index(drop=True)
        for col in (cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col):
            out[col] = pd.to_numeric(out[col], errors="coerce")

        o, h, l, c, v = (out[cfg.open_col], out[cfg.high_col], out[cfg.low_col], out[cfg.close_col], out[cfg.volume_col])
        g = out.groupby(cfg.asset_col, sort=False, group_keys=False)

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
        out["direction"] = np.sign(out["body"]).astype(np.int8)
        out["volume_log"] = np.log1p(v.clip(lower=0).fillna(0))
        out["range_log"] = np.log1p(out["range"].clip(lower=0))
        out["body_to_wick"] = self._safe_div(out["abs_body"], out["upper_wick"] + out["lower_wick"])
        out["wick_imbalance"] = out["upper_wick"] - out["lower_wick"]

        # Previous candle relationships
        g = out.groupby(cfg.asset_col, sort=False, group_keys=False)
        out["prev_close"] = g[cfg.close_col].shift(1)
        out["prev_open"] = g[cfg.open_col].shift(1)
        out["prev_body"] = g["body"].shift(1)
        out["prev_range"] = g["range"].shift(1)
        out["prev_upper_wick"] = g["upper_wick"].shift(1)
        out["prev_lower_wick"] = g["lower_wick"].shift(1)
        out["gap"] = o - out["prev_close"]
        out["return_1"] = self._safe_div(c - out["prev_close"], out["prev_close"])
        out["return_3"] = g[cfg.close_col].transform(lambda s: s.pct_change(3))
        out["return_5"] = g[cfg.close_col].transform(lambda s: s.pct_change(5))

        # Rolling statistics
        out["rolling_mean_5"] = g[cfg.close_col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        out["rolling_mean_10"] = g[cfg.close_col].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_mean_20"] = g[cfg.close_col].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["rolling_std_10"] = g[cfg.close_col].transform(lambda s: s.rolling(10, min_periods=2).std())
        out["rolling_std_20"] = g[cfg.close_col].transform(lambda s: s.rolling(20, min_periods=2).std())
        out["rolling_range_10"] = g["range"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_range_20"] = g["range"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["rolling_body_10"] = g["abs_body"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["rolling_body_20"] = g["abs_body"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        out["volatility_20"] = self._safe_div(out["rolling_std_20"], out["rolling_mean_20"].abs())
        out["momentum_3"] = g[cfg.close_col].transform(lambda s: s.diff(3))
        out["momentum_5"] = g[cfg.close_col].transform(lambda s: s.diff(5))
        out["trend_5"] = self._safe_div(c - out["rolling_mean_5"], out["rolling_mean_5"])
        out["trend_20"] = self._safe_div(c - out["rolling_mean_20"], out["rolling_mean_20"])

        # True ATR (Wilder smoothing), grouped per asset
        tr = g.apply(lambda x: self._true_range(x, cfg.high_col, cfg.low_col, cfg.close_col), include_groups=False).reset_index(level=0, drop=True)
        tr = tr.reindex(out.index)
        out["true_range"] = tr
        out["atr_14"] = out.groupby(cfg.asset_col, sort=False)["true_range"].transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
        out["atr_pct"] = self._safe_div(out["atr_14"], c.abs())

        # EMA / trend structure
        for period in (5, 10, 20, 50):
            out[f"ema_{period}"] = g[cfg.close_col].transform(lambda s, p=period: s.ewm(span=p, adjust=False, min_periods=p).mean())
            out[f"ema_{period}_dist_atr"] = self._safe_div(c - out[f"ema_{period}"], out["atr_14"])
            out[f"ema_{period}_slope_atr"] = self._safe_div(out.groupby(cfg.asset_col, sort=False)[f"ema_{period}"].diff(), out["atr_14"])
        out["ema_bull_alignment"] = ((out["ema_5"] > out["ema_10"]) & (out["ema_10"] > out["ema_20"]) & (out["ema_20"] > out["ema_50"])).astype(np.int8)
        out["ema_bear_alignment"] = ((out["ema_5"] < out["ema_10"]) & (out["ema_10"] < out["ema_20"]) & (out["ema_20"] < out["ema_50"])).astype(np.int8)
        out["ema_5_20_spread_atr"] = self._safe_div(out["ema_5"] - out["ema_20"], out["atr_14"])
        out["ema_10_50_spread_atr"] = self._safe_div(out["ema_10"] - out["ema_50"], out["atr_14"])

        # RSI family and semantic states
        for period in (7, 14, 21):
            out[f"rsi_{period}"] = g[cfg.close_col].transform(lambda s, p=period: self._rsi(s, p))
            out[f"rsi_{period}_slope"] = out.groupby(cfg.asset_col, sort=False)[f"rsi_{period}"].diff()
        prev_rsi14 = out.groupby(cfg.asset_col, sort=False)["rsi_14"].shift(1)
        out["rsi14_oversold"] = (out["rsi_14"] < 30).astype(np.int8)
        out["rsi14_overbought"] = (out["rsi_14"] > 70).astype(np.int8)
        out["rsi14_cross_30_up"] = ((prev_rsi14 <= 30) & (out["rsi_14"] > 30)).astype(np.int8)
        out["rsi14_cross_70_down"] = ((prev_rsi14 >= 70) & (out["rsi_14"] < 70)).astype(np.int8)

        # MACD
        ema12 = g[cfg.close_col].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
        ema26 = g[cfg.close_col].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out.groupby(cfg.asset_col, sort=False)["macd"].transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
        out["macd_hist"] = out["macd"] - out["macd_signal"]
        out["macd_hist_atr"] = self._safe_div(out["macd_hist"], out["atr_14"])
        prev_hist = out.groupby(cfg.asset_col, sort=False)["macd_hist"].shift(1)
        out["macd_bull_cross"] = ((prev_hist <= 0) & (out["macd_hist"] > 0)).astype(np.int8)
        out["macd_bear_cross"] = ((prev_hist >= 0) & (out["macd_hist"] < 0)).astype(np.int8)

        # Bollinger Bands
        bb_mid = out["rolling_mean_20"]
        bb_std = out["rolling_std_20"]
        out["bb_upper"] = bb_mid + 2.0 * bb_std
        out["bb_lower"] = bb_mid - 2.0 * bb_std
        out["bb_percent_b"] = self._safe_div(c - out["bb_lower"], out["bb_upper"] - out["bb_lower"])
        out["bb_bandwidth"] = self._safe_div(out["bb_upper"] - out["bb_lower"], bb_mid.abs())
        out["bb_bandwidth_rank_50"] = out.groupby(cfg.asset_col, sort=False)["bb_bandwidth"].transform(lambda s: self._percentile_rank(s, 50))
        out["bb_squeeze"] = (out["bb_bandwidth_rank_50"] <= 0.20).astype(np.int8)
        out["bb_break_upper"] = (c > out["bb_upper"]).astype(np.int8)
        out["bb_break_lower"] = (c < out["bb_lower"]).astype(np.int8)

        # Stochastic oscillator
        low14 = g[cfg.low_col].transform(lambda s: s.rolling(14, min_periods=14).min())
        high14 = g[cfg.high_col].transform(lambda s: s.rolling(14, min_periods=14).max())
        out["stoch_k"] = 100.0 * self._safe_div(c - low14, high14 - low14)
        out["stoch_d"] = out.groupby(cfg.asset_col, sort=False)["stoch_k"].transform(lambda s: s.rolling(3, min_periods=3).mean())
        prev_k = out.groupby(cfg.asset_col, sort=False)["stoch_k"].shift(1)
        prev_d = out.groupby(cfg.asset_col, sort=False)["stoch_d"].shift(1)
        out["stoch_bull_cross"] = ((prev_k <= prev_d) & (out["stoch_k"] > out["stoch_d"])).astype(np.int8)
        out["stoch_bear_cross"] = ((prev_k >= prev_d) & (out["stoch_k"] < out["stoch_d"])).astype(np.int8)

        # ADX / directional movement
        prev_h = g[cfg.high_col].shift(1)
        prev_l = g[cfg.low_col].shift(1)
        up_move = h - prev_h
        down_move = prev_l - l
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
        plus_sm = plus_dm.groupby(out[cfg.asset_col], sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
        minus_sm = minus_dm.groupby(out[cfg.asset_col], sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
        out["plus_di_14"] = 100.0 * self._safe_div(plus_sm, out["atr_14"])
        out["minus_di_14"] = 100.0 * self._safe_div(minus_sm, out["atr_14"])
        dx = 100.0 * self._safe_div((out["plus_di_14"] - out["minus_di_14"]).abs(), out["plus_di_14"] + out["minus_di_14"])
        out["adx_14"] = dx.groupby(out[cfg.asset_col], sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
        out["adx_trending"] = (out["adx_14"] >= 25).astype(np.int8)
        out["di_bull"] = (out["plus_di_14"] > out["minus_di_14"]).astype(np.int8)
        out["di_bear"] = (out["minus_di_14"] > out["plus_di_14"]).astype(np.int8)

        # ROC
        out["roc_5"] = g[cfg.close_col].transform(lambda s: s.pct_change(5) * 100.0)
        out["roc_10"] = g[cfg.close_col].transform(lambda s: s.pct_change(10) * 100.0)

        # Candle regime / patterns
        prev_high, prev_low = g[cfg.high_col].shift(1), g[cfg.low_col].shift(1)
        out["higher_high"] = (h > prev_high).astype(np.int8)
        out["higher_low"] = (l > prev_low).astype(np.int8)
        out["lower_high"] = (h < prev_high).astype(np.int8)
        out["lower_low"] = (l < prev_low).astype(np.int8)
        out["inside_bar"] = ((h <= prev_high) & (l >= prev_low)).astype(np.int8)
        out["expansion_bar"] = (out["range"] > out["rolling_range_20"] * 1.5).astype(np.int8)
        out["small_body"] = (out["body_ratio"] < 0.25).astype(np.int8)
        out["large_body"] = (out["body_ratio"] > 0.75).astype(np.int8)
        out["doji_like"] = (out["body_ratio"] < 0.10).astype(np.int8)
        out["pinbar_bull"] = ((out["lower_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.30) & (out["bull"] == 1)).astype(np.int8)
        out["pinbar_bear"] = ((out["upper_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.30) & (out["bear"] == 1)).astype(np.int8)
        out["engulf_bull"] = ((out["bull"] == 1) & (out["prev_body"] < 0) & (o <= out["prev_close"]) & (c >= out["prev_open"])).astype(np.int8)
        out["engulf_bear"] = ((out["bear"] == 1) & (out["prev_body"] > 0) & (o >= out["prev_close"]) & (c <= out["prev_open"])).astype(np.int8)

        # Multi-candle market structure and breakout states
        bull_i = out["bull"].astype(int)
        bear_i = out["bear"].astype(int)
        out["bull_streak_3"] = bull_i.groupby(out[cfg.asset_col], sort=False).transform(lambda s: s.rolling(3, min_periods=3).sum()).eq(3).astype(np.int8)
        out["bear_streak_3"] = bear_i.groupby(out[cfg.asset_col], sort=False).transform(lambda s: s.rolling(3, min_periods=3).sum()).eq(3).astype(np.int8)
        out["hh_hl_2"] = ((out["higher_high"] == 1) & (out["higher_low"] == 1)).astype(np.int8)
        out["lh_ll_2"] = ((out["lower_high"] == 1) & (out["lower_low"] == 1)).astype(np.int8)
        prior_high20 = g[cfg.high_col].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
        prior_low20 = g[cfg.low_col].transform(lambda s: s.shift(1).rolling(20, min_periods=5).min())
        out["breakout_up_20"] = (c > prior_high20).astype(np.int8)
        out["breakout_down_20"] = (c < prior_low20).astype(np.int8)
        out["false_break_up_20"] = ((h > prior_high20) & (c <= prior_high20)).astype(np.int8)
        out["false_break_down_20"] = ((l < prior_low20) & (c >= prior_low20)).astype(np.int8)
        out["compression_5"] = (out["rolling_range_10"] < out["rolling_range_20"] * 0.75).astype(np.int8)
        prev_compression = out.groupby(cfg.asset_col, sort=False)["compression_5"].shift(1).fillna(0)
        out["compression_expansion"] = ((prev_compression == 1) & (out["expansion_bar"] == 1)).astype(np.int8)
        out["rejection_bull"] = ((out["lower_wick_ratio"] >= 0.45) & (out["close_pos_in_range"] >= 0.65)).astype(np.int8)
        out["rejection_bear"] = ((out["upper_wick_ratio"] >= 0.45) & (out["close_pos_in_range"] <= 0.35)).astype(np.int8)

        # Statistical regimes - always grouped by asset
        out["zclose_20"] = g[cfg.close_col].transform(lambda s: self._zscore(s, 20))
        out["zrange_20"] = out.groupby(cfg.asset_col, sort=False)["range"].transform(lambda s: self._zscore(s, 20))
        out["zbody_20"] = out.groupby(cfg.asset_col, sort=False)["abs_body"].transform(lambda s: self._zscore(s, 20))
        out["prange_20"] = out.groupby(cfg.asset_col, sort=False)["range"].transform(lambda s: self._percentile_rank(s, 20))
        out["pbody_20"] = out.groupby(cfg.asset_col, sort=False)["abs_body"].transform(lambda s: self._percentile_rank(s, 20))
        out["pvol_20"] = g[cfg.volume_col].transform(lambda s: self._percentile_rank(s.fillna(0), 20))

        # Volume context
        out["volume_change_1"] = g[cfg.volume_col].transform(lambda s: s.pct_change(1))
        out["volume_change_5"] = g[cfg.volume_col].transform(lambda s: s.pct_change(5))
        out["volume_ma_10"] = g[cfg.volume_col].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out["volume_ratio_10"] = self._safe_div(v, out["volume_ma_10"])

        # Time context
        ts = pd.to_datetime(out[cfg.timestamp_col], unit="s", errors="coerce", utc=True)
        out["hour"] = ts.dt.hour.fillna(0).astype(int)
        out["minute"] = ts.dt.minute.fillna(0).astype(int)
        out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
        out["is_asia_session"] = out["hour"].between(0, 7).astype(np.int8)
        out["is_london_session"] = out["hour"].between(8, 15).astype(np.int8)
        out["is_newyork_session"] = out["hour"].between(13, 20).astype(np.int8)
        out["is_late_session"] = out["hour"].between(21, 23).astype(np.int8)

        # Cross-feature interactions
        out["body_x_volatility"] = out["body_ratio"] * out["volatility_20"].fillna(0)
        out["wick_x_range"] = out["upper_wick_ratio"] * self._safe_div(out["range"], out["rolling_range_20"])
        out["momentum_x_range"] = out["momentum_3"].fillna(0) * out["range"]
        out["trend_x_volume"] = out["trend_20"].fillna(0) * out["volume_ratio_10"].fillna(0)
        out["trend_regime_bull"] = ((out["ema_bull_alignment"] == 1) & (out["adx_trending"] == 1) & (out["di_bull"] == 1)).astype(np.int8)
        out["trend_regime_bear"] = ((out["ema_bear_alignment"] == 1) & (out["adx_trending"] == 1) & (out["di_bear"] == 1)).astype(np.int8)
        out["mean_reversion_long"] = ((out["rsi14_oversold"] == 1) & (out["bb_percent_b"] < 0.10)).astype(np.int8)
        out["mean_reversion_short"] = ((out["rsi14_overbought"] == 1) & (out["bb_percent_b"] > 0.90)).astype(np.int8)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=[cfg.asset_col, cfg.timeframe_col, cfg.timestamp_col, cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col]).reset_index(drop=True)
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].fillna(0)
        return out
