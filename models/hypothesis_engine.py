from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from itertools import combinations
import json
import sqlite3
from pathlib import Path

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
    """Knowledge-guided hypothesis generator."""

    EXCLUDE_COLUMNS = {
        "id",
        "timestamp",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "direction",
    }

    BLOCKED_FEATURES = {
        "open",
        "high",
        "low",
        "close",
        "prev_open",
        "prev_close",
        "prev_high",
        "prev_low",
        "rolling_mean_5",
        "rolling_mean_10",
        "rolling_mean_20",
        "rolling_std_10",
        "rolling_std_20",
    }

    SEMANTIC_PRIORITY_PREFIXES = (
        "rsi_",
        "ema_",
        "macd",
        "bb_",
        "stoch_",
        "adx_",
        "plus_di_",
        "minus_di_",
        "roc_",
        "trend_regime_",
        "mean_reversion_",
        "breakout_",
        "false_break_",
        "compression_",
        "rejection_",
        "hh_",
        "lh_",
        "bull_streak_",
        "bear_streak_",
    )

    SEMANTIC_FEATURE_ALLOWLIST = {
        "rsi14_oversold",
        "rsi14_overbought",
        "rsi14_cross_30_up",
        "rsi14_cross_70_down",
        "ema_bull_alignment",
        "ema_bear_alignment",
        "ema_5_20_spread_atr",
        "ema_10_50_spread_atr",
        "macd_bull_cross",
        "macd_bear_cross",
        "bb_squeeze",
        "bb_break_upper",
        "bb_break_lower",
        "stoch_bull_cross",
        "stoch_bear_cross",
        "adx_trending",
        "di_bull",
        "di_bear",
        "trend_regime_bull",
        "trend_regime_bear",
        "mean_reversion_long",
        "mean_reversion_short",
        "compression_expansion",
        "rejection_bull",
        "rejection_bear",
        "breakout_up_20",
        "breakout_down_20",
        "false_break_up_20",
        "false_break_down_20",
        "hh_hl_2",
        "lh_ll_2",
        "bull_streak_3",
        "bear_streak_3",
    }

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
        txt = direction + str([(c.feature, c.operator, c.value) for c in conditions])
        return sha1(txt.encode()).hexdigest()

    @staticmethod
    def _safe_json_loads(value: object) -> object:
        try:
            return json.loads(value or "[]")
        except Exception:
            return []

    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        cols = df.select_dtypes(include=np.number).columns
        return [c for c in cols if c not in self.EXCLUDE_COLUMNS and c not in self.BLOCKED_FEATURES]

    def _semantic_features(self, df: pd.DataFrame) -> list[str]:
        cols = self._numeric_features(df)
        semantic = [c for c in cols if c in self.SEMANTIC_FEATURE_ALLOWLIST or c.startswith(self.SEMANTIC_PRIORITY_PREFIXES)]
        if semantic:
            return semantic
        return [c for c in cols if c not in self.EXCLUDE_COLUMNS and c not in self.BLOCKED_FEATURES]

    def _best_features_from_db(self) -> list[str]:
        path = Path(self.research_db)
        if not path.exists():
            return []

        conn = sqlite3.connect(path)
        try:
            try:
                df = pd.read_sql_query(
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

        if df.empty:
            return []
        return [str(x) for x in df["feature"].tolist() if isinstance(x, str) and x]

    def _best_thresholds_from_db(self) -> dict[str, list[tuple[str, object]]]:
        path = Path(self.research_db)
        if not path.exists():
            return {}

        conn = sqlite3.connect(path)
        try:
            try:
                df = pd.read_sql_query(
                    """
                    SELECT feature, operator, threshold, avg_score, avg_expectancy, total
                    FROM threshold_statistics
                    ORDER BY avg_score DESC, avg_expectancy DESC, total DESC
                    LIMIT ?
                    """,
                    conn,
                    params=(max(50, self.top_feature_limit * self.top_threshold_limit),),
                )
            except Exception:
                return {}
        finally:
            conn.close()

        out: dict[str, list[tuple[str, object]]] = {}
        if df.empty:
            return out

        for _, row in df.iterrows():
            feature = row.get("feature")
            operator = row.get("operator")
            threshold = row.get("threshold")
            if feature is None or operator is None:
                continue
            feat = str(feature)
            if feat in self.BLOCKED_FEATURES:
                continue
            out.setdefault(feat, []).append((str(operator), self._parse_threshold(threshold)))
        return out

    @staticmethod
    def _parse_threshold(value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    loaded = json.loads(text)
                    if isinstance(loaded, list) and len(loaded) == 2:
                        return tuple(loaded)
                except Exception:
                    pass
            try:
                return float(text)
            except Exception:
                return value
        return value

    def _thresholds_from_series(self, series: pd.Series) -> list[tuple[str, object]]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return []

        q20 = float(s.quantile(0.20))
        q30 = float(s.quantile(0.30))
        q40 = float(s.quantile(0.40))
        q60 = float(s.quantile(0.60))
        q70 = float(s.quantile(0.70))
        q80 = float(s.quantile(0.80))

        return [
            (">", q60),
            (">", q70),
            (">", q80),
            ("<", q40),
            ("<", q30),
            ("<", q20),
            ("between", (q40, q60)),
        ]

    def _thresholds_for_feature(
        self,
        feature: str,
        df: pd.DataFrame,
        db_thresholds: dict[str, list[tuple[str, object]]],
    ) -> list[tuple[str, object]]:
        learned = db_thresholds.get(feature, [])[: self.top_threshold_limit]
        if learned:
            return learned
        return self._thresholds_from_series(df[feature])

    def _mutate_numeric(self, value: object) -> list[object]:
        if not isinstance(value, (int, float, np.number)):
            return [value]
        base = float(value)
        delta = abs(base) * 0.05
        if delta == 0:
            delta = 0.001
        out = [base + (k * delta) for k in range(-self.mutation_steps, self.mutation_steps + 1)]
        return sorted(set(out))

    def _mutation_candidates(self, operator: str, value: object) -> list[object]:
        if operator == "between" and isinstance(value, tuple) and len(value) == 2:
            low, high = value
            low_vals = self._mutate_numeric(low)
            high_vals = self._mutate_numeric(high)
            candidates: list[object] = []
            for lo in low_vals:
                for hi in high_vals:
                    if lo < hi:
                        candidates.append((lo, hi))
            return candidates[: max(5, self.mutation_steps * 4)]
        return self._mutate_numeric(value)[: max(5, self.mutation_steps * 4)]

    def _best_features_for_df(self, df: pd.DataFrame) -> list[str]:
        cols = self._numeric_features(df)
        if not cols:
            return []

        semantic = [c for c in self._semantic_features(df) if c in cols]
        learned = [f for f in self._best_features_from_db() if f in cols and f not in self.BLOCKED_FEATURES]
        ordered: list[str] = []
        for bucket in (semantic, learned, cols):
            for c in bucket:
                if c not in ordered and c not in self.EXCLUDE_COLUMNS and c not in self.BLOCKED_FEATURES:
                    ordered.append(c)
        return ordered[: self.max_features]

    def feature_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        numeric = self._numeric_features(df)
        semantic = set(self._semantic_features(df))
        for c in numeric:
            rows.append({
                "feature": c,
                "is_semantic": int(c in semantic),
                "dtype": str(df[c].dtype),
                "n_unique": int(df[c].nunique(dropna=True)),
                "missing": int(df[c].isna().sum()),
            })
        if not rows:
            return pd.DataFrame(columns=["feature", "is_semantic", "dtype", "n_unique", "missing"])
        return pd.DataFrame(rows).sort_values(["is_semantic", "n_unique", "feature"], ascending=[False, False, True]).reset_index(drop=True)

    def diagnostics(self, df: pd.DataFrame) -> dict[str, object]:
        numeric = self._numeric_features(df)
        semantic = [c for c in self._semantic_features(df) if c in numeric]
        indicator_prefix_counts = {
            "rsi": sum(c.startswith("rsi_") for c in numeric),
            "ema": sum(c.startswith("ema_") for c in numeric),
            "macd": sum(c.startswith("macd") for c in numeric),
            "bb": sum(c.startswith("bb_") for c in numeric),
            "stoch": sum(c.startswith("stoch_") for c in numeric),
            "adx": sum(c.startswith("adx_") for c in numeric),
            "regime": sum(c.startswith(("trend_regime_", "mean_reversion_", "breakout_", "false_break_", "compression_", "rejection_")) for c in numeric),
        }
        return {
            "total_numeric_features": len(numeric),
            "semantic_features": semantic,
            "semantic_count": len(semantic),
            "prefix_counts": indicator_prefix_counts,
        }

    def _build_single_feature_hypotheses(
        self,
        features: list[str],
        df: pd.DataFrame,
        db_thresholds: dict[str, list[tuple[str, object]]],
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        seen: set[str] = set()
        hid = 1

        for feature in features:
            if feature not in df.columns:
                continue
            if feature in self.BLOCKED_FEATURES:
                continue
            if df[feature].nunique(dropna=True) < 10:
                continue

            thresholds = self._thresholds_for_feature(feature, df, db_thresholds)
            if not thresholds:
                continue

            for op, value in thresholds:
                for new_value in self._mutation_candidates(op, value):
                    for direction in ("BUY", "SELL"):
                        cond = [Condition(feature, op, new_value)]
                        sig = self._signature(direction, cond)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        hypotheses.append(Hypothesis(id=f"H{hid:06d}", direction=direction, conditions=cond, signature=sig))
                        hid += 1

        return hypotheses

    def _build_pair_hypotheses(
        self,
        features: list[str],
        df: pd.DataFrame,
        db_thresholds: dict[str, list[tuple[str, object]]],
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        seen: set[str] = set()
        hid = 1
        top = features[: min(20, len(features))]
        if len(top) < 2:
            return []

        for f1, f2 in combinations(top, 2):
            if f1 in self.BLOCKED_FEATURES or f2 in self.BLOCKED_FEATURES:
                continue
            if f1 not in df.columns or f2 not in df.columns:
                continue
            if df[f1].nunique(dropna=True) < 10 or df[f2].nunique(dropna=True) < 10:
                continue

            t1 = self._thresholds_for_feature(f1, df, db_thresholds)
            t2 = self._thresholds_for_feature(f2, df, db_thresholds)
            if not t1 or not t2:
                continue

            op1, v1 = t1[0]
            op2, v2 = t2[0]
            mut1 = self._mutation_candidates(op1, v1)
            mut2 = self._mutation_candidates(op2, v2)

            for nv1 in mut1[:4]:
                for nv2 in mut2[:4]:
                    for direction in ("BUY", "SELL"):
                        cond = [Condition(f1, op1, nv1), Condition(f2, op2, nv2)]
                        sig = self._signature(direction, cond)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        hypotheses.append(Hypothesis(id=f"H{hid:06d}", direction=direction, conditions=cond, signature=sig))
                        hid += 1

        return hypotheses

    def generate_from_dataframe(self, df: pd.DataFrame, max_features: int = 2) -> list[Hypothesis]:
        features = self._best_features_for_df(df)
        if not features:
            return []

        db_thresholds = self._best_thresholds_from_db()
        hypotheses = self._build_single_feature_hypotheses(features, df, db_thresholds)
        if max_features >= 2:
            hypotheses.extend(self._build_pair_hypotheses(features, df, db_thresholds))

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
                yield Hypothesis(id=f"H{hid:06d}", direction="AUTO", conditions=[Condition(name, rule[0], rule[1])], signature=name)
                hid += 1