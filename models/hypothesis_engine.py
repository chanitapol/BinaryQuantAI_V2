from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import sqlite3
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Condition:
    feature: str
    operator: str
    value: object


@dataclass
class Hypothesis:
    id: str
    direction: str
    conditions: list[Condition]
    signature: str


class HypothesisEngine:
    """Generate directional hypotheses from semantic trading events and setups.

    Search order:
      1. semantic event
      2. two-condition semantic setup
      3. regime + trigger + confirmation
      4. wider multi-condition semantic combinations

    Raw OHLC levels and rolling price-level proxies are deliberately excluded.
    """

    EXCLUDE_COLUMNS = {
        "id", "timestamp", "timeframe", "open", "high", "low", "close",
        "volume", "direction",
    }

    BLOCKED_FEATURES = {
        "open", "high", "low", "close",
        "prev_open", "prev_close", "prev_high", "prev_low",
        "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
        "rolling_std_10", "rolling_std_20",
    }

    BUY_EVENTS = {
        "rsi14_oversold", "rsi14_cross_30_up",
        "ema_bull_alignment", "macd_bull_cross", "stoch_bull_cross",
        "di_bull", "trend_regime_bull", "mean_reversion_long",
        "rejection_bull", "breakout_up_20", "false_break_down_20",
        "hh_hl_2", "bull_streak_3",
    }
    SELL_EVENTS = {
        "rsi14_overbought", "rsi14_cross_70_down",
        "ema_bear_alignment", "macd_bear_cross", "stoch_bear_cross",
        "di_bear", "trend_regime_bear", "mean_reversion_short",
        "rejection_bear", "breakout_down_20", "false_break_up_20",
        "lh_ll_2", "bear_streak_3",
    }
    NEUTRAL_EVENTS = {"bb_squeeze", "adx_trending", "compression_expansion"}

    BUY_REGIMES = {"trend_regime_bull", "ema_bull_alignment", "di_bull", "mean_reversion_long"}
    SELL_REGIMES = {"trend_regime_bear", "ema_bear_alignment", "di_bear", "mean_reversion_short"}

    BUY_TRIGGERS = {
        "rsi14_cross_30_up", "macd_bull_cross", "stoch_bull_cross",
        "breakout_up_20", "false_break_down_20", "rejection_bull",
    }
    SELL_TRIGGERS = {
        "rsi14_cross_70_down", "macd_bear_cross", "stoch_bear_cross",
        "breakout_down_20", "false_break_up_20", "rejection_bear",
    }

    BUY_CONFIRMATIONS = {"adx_trending", "hh_hl_2", "bull_streak_3", "compression_expansion"}
    SELL_CONFIRMATIONS = {"adx_trending", "lh_ll_2", "bear_streak_3", "compression_expansion"}

    CONTINUOUS_PREFIXES = (
        "rsi_", "ema_", "macd", "bb_", "stoch_", "adx_",
        "plus_di_", "minus_di_", "roc_",
    )

    def __init__(
        self,
        max_features: int = 25,
        mutation_steps: int = 2,
        quantiles: tuple[float, ...] = (0.2, 0.3, 0.4, 0.6, 0.7, 0.8),
        research_db: str = "research.db",
        top_feature_limit: int = 18,
        top_threshold_limit: int = 5,
    ) -> None:
        self.max_features = max_features
        self.mutation_steps = mutation_steps
        self.quantiles = quantiles
        self.research_db = research_db
        self.top_feature_limit = top_feature_limit
        self.top_threshold_limit = top_threshold_limit

    @staticmethod
    def _signature(direction: str, conditions: list[Condition]) -> str:
        canonical = sorted((c.feature, c.operator, repr(c.value)) for c in conditions)
        return sha1((direction + repr(canonical)).encode()).hexdigest()

    @staticmethod
    def _event_condition(feature: str) -> Condition:
        return Condition(feature, ">", 0.5)

    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        cols = df.select_dtypes(include=np.number).columns
        return [c for c in cols if c not in self.EXCLUDE_COLUMNS and c not in self.BLOCKED_FEATURES]

    def _available(self, df: pd.DataFrame, names: set[str]) -> list[str]:
        return sorted(f for f in names if f in df.columns and f not in self.BLOCKED_FEATURES)

    def _best_features_from_db(self) -> list[str]:
        path = Path(self.research_db)
        if not path.exists():
            return []
        conn = sqlite3.connect(path)
        try:
            try:
                rows = pd.read_sql_query(
                    """
                    SELECT feature, avg_score, avg_winrate, total
                    FROM feature_statistics
                    ORDER BY avg_score DESC, avg_winrate DESC, total DESC
                    LIMIT ?
                    """,
                    conn,
                    params=(self.top_feature_limit,),
                )
            except Exception:
                return []
        finally:
            conn.close()
        if rows.empty:
            return []
        return [str(x) for x in rows["feature"].tolist() if isinstance(x, str) and x not in self.BLOCKED_FEATURES]

    def feature_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        semantic = self.BUY_EVENTS | self.SELL_EVENTS | self.NEUTRAL_EVENTS
        rows = []
        for feature in self._numeric_features(df):
            rows.append({
                "feature": feature,
                "is_semantic": int(feature in semantic),
                "dtype": str(df[feature].dtype),
                "n_unique": int(df[feature].nunique(dropna=True)),
                "missing": int(df[feature].isna().sum()),
            })
        if not rows:
            return pd.DataFrame(columns=["feature", "is_semantic", "dtype", "n_unique", "missing"])
        return pd.DataFrame(rows).sort_values(
            ["is_semantic", "n_unique", "feature"], ascending=[False, False, True]
        ).reset_index(drop=True)

    def diagnostics(self, df: pd.DataFrame) -> dict[str, object]:
        numeric = self._numeric_features(df)
        semantic_set = self.BUY_EVENTS | self.SELL_EVENTS | self.NEUTRAL_EVENTS
        semantic = [c for c in numeric if c in semantic_set]
        return {
            "total_numeric_features": len(numeric),
            "semantic_features": semantic,
            "semantic_count": len(semantic),
            "prefix_counts": {
                "rsi": sum(c.startswith("rsi_") or c.startswith("rsi14_") for c in numeric),
                "ema": sum(c.startswith("ema_") for c in numeric),
                "macd": sum(c.startswith("macd") for c in numeric),
                "bb": sum(c.startswith("bb_") for c in numeric),
                "stoch": sum(c.startswith("stoch_") for c in numeric),
                "adx": sum(c.startswith("adx_") for c in numeric),
                "regime": sum(c.startswith(("trend_regime_", "mean_reversion_", "breakout_", "false_break_", "compression_", "rejection_")) for c in numeric),
            },
        }

    def _append(self, out: list[Hypothesis], seen: set[str], direction: str, conditions: list[Condition]) -> None:
        features = [c.feature for c in conditions]
        if len(features) != len(set(features)):
            return
        sig = self._signature(direction, conditions)
        if sig in seen:
            return
        seen.add(sig)
        out.append(Hypothesis("", direction, conditions, sig))

    def _build_semantic_events(self, df: pd.DataFrame) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        seen: set[str] = set()
        for feature in self._available(df, self.BUY_EVENTS):
            self._append(out, seen, "BUY", [self._event_condition(feature)])
        for feature in self._available(df, self.SELL_EVENTS):
            self._append(out, seen, "SELL", [self._event_condition(feature)])
        return out

    def _build_semantic_pairs(self, df: pd.DataFrame) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        seen: set[str] = set()

        buy_regimes = self._available(df, self.BUY_REGIMES)
        buy_triggers = self._available(df, self.BUY_TRIGGERS)
        sell_regimes = self._available(df, self.SELL_REGIMES)
        sell_triggers = self._available(df, self.SELL_TRIGGERS)
        neutral = self._available(df, self.NEUTRAL_EVENTS)

        for regime in buy_regimes:
            for trigger in buy_triggers:
                self._append(out, seen, "BUY", [self._event_condition(regime), self._event_condition(trigger)])
        for trigger in buy_triggers:
            for confirm in neutral:
                self._append(out, seen, "BUY", [self._event_condition(trigger), self._event_condition(confirm)])

        for regime in sell_regimes:
            for trigger in sell_triggers:
                self._append(out, seen, "SELL", [self._event_condition(regime), self._event_condition(trigger)])
        for trigger in sell_triggers:
            for confirm in neutral:
                self._append(out, seen, "SELL", [self._event_condition(trigger), self._event_condition(confirm)])
        return out

    def _build_regime_trigger_confirmation(self, df: pd.DataFrame) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        seen: set[str] = set()

        for regime in self._available(df, self.BUY_REGIMES):
            for trigger in self._available(df, self.BUY_TRIGGERS):
                for confirmation in self._available(df, self.BUY_CONFIRMATIONS):
                    self._append(out, seen, "BUY", [
                        self._event_condition(regime),
                        self._event_condition(trigger),
                        self._event_condition(confirmation),
                    ])

        for regime in self._available(df, self.SELL_REGIMES):
            for trigger in self._available(df, self.SELL_TRIGGERS):
                for confirmation in self._available(df, self.SELL_CONFIRMATIONS):
                    self._append(out, seen, "SELL", [
                        self._event_condition(regime),
                        self._event_condition(trigger),
                        self._event_condition(confirmation),
                    ])
        return out

    def _build_wider_combinations(self, df: pd.DataFrame) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        seen: set[str] = set()
        semantic_candidates = [
            *self._available(df, self.BUY_EVENTS | self.SELL_EVENTS | self.NEUTRAL_EVENTS),
        ]
        if len(semantic_candidates) < 4:
            return out

        # Build wider 3-4 condition hypotheses around the strongest semantic zones.
        for combo in combinations(semantic_candidates[: min(len(semantic_candidates), 14)], 3):
            combo_set = set(combo)
            direction = "BUY" if len(combo_set & self.BUY_EVENTS) >= len(combo_set & self.SELL_EVENTS) else "SELL"
            self._append(out, seen, direction, [self._event_condition(c) for c in combo])
        for combo in combinations(semantic_candidates[: min(len(semantic_candidates), 12)], 4):
            combo_set = set(combo)
            direction = "BUY" if len(combo_set & self.BUY_EVENTS) >= len(combo_set & self.SELL_EVENTS) else "SELL"
            self._append(out, seen, direction, [self._event_condition(c) for c in combo])
        return out

    def _continuous_fallback(self, df: pd.DataFrame) -> list[Hypothesis]:
        """Fallback only when semantic event columns are absent.

        This preserves pipeline compatibility while avoiding raw-price proxies.
        Direction is inferred from indicator semantics instead of testing every
        condition as both BUY and SELL.
        """
        out: list[Hypothesis] = []
        seen: set[str] = set()
        candidates = [
            c for c in self._numeric_features(df)
            if c.startswith(self.CONTINUOUS_PREFIXES) and df[c].nunique(dropna=True) >= 10
        ][: self.max_features]

        for feature in candidates:
            s = pd.to_numeric(df[feature], errors="coerce").dropna()
            if s.empty:
                continue
            q30 = float(s.quantile(0.30))
            q70 = float(s.quantile(0.70))
            name = feature.lower()

            if "rsi" in name or "stoch" in name:
                self._append(out, seen, "BUY", [Condition(feature, "<", q30)])
                self._append(out, seen, "SELL", [Condition(feature, ">", q70)])
            elif any(token in name for token in ("slope", "spread", "hist", "roc", "plus_di")):
                self._append(out, seen, "BUY", [Condition(feature, ">", q70)])
                self._append(out, seen, "SELL", [Condition(feature, "<", q30)])
            elif "minus_di" in name:
                self._append(out, seen, "SELL", [Condition(feature, ">", q70)])
                self._append(out, seen, "BUY", [Condition(feature, "<", q30)])
        return out

    def generate_from_dataframe(self, df: pd.DataFrame, max_features: int = 3) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        hypotheses.extend(self._build_semantic_events(df))
        if max_features >= 2:
            hypotheses.extend(self._build_semantic_pairs(df))
        if max_features >= 3:
            hypotheses.extend(self._build_regime_trigger_confirmation(df))
        if max_features >= 4:
            hypotheses.extend(self._build_wider_combinations(df))

        if not hypotheses:
            hypotheses.extend(self._continuous_fallback(df))

        unique: list[Hypothesis] = []
        seen: set[str] = set()
        for h in hypotheses:
            if h.signature in seen:
                continue
            seen.add(h.signature)
            unique.append(h)

        for i, h in enumerate(unique, start=1):
            h.id = f"H{i:06d}"
        return unique

    def generate(self, feature_rules=None, max_features=3):
        if feature_rules:
            hid = 1
            for name, rule in feature_rules.items():
                direction = "AUTO"
                if name in self.BUY_EVENTS:
                    direction = "BUY"
                elif name in self.SELL_EVENTS:
                    direction = "SELL"
                conditions = [Condition(name, rule[0], rule[1])]
                sig = self._signature(direction, conditions)
                yield Hypothesis(id=f"H{hid:06d}", direction=direction, conditions=conditions, signature=sig)
                hid += 1
            return
        raise ValueError("feature_rules required for generate()")