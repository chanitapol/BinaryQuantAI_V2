from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from models.knowledge_query import KnowledgeQuery
from models.hypothesis_engine import Hypothesis, Condition


@dataclass
class EvolutionCandidate:
    parent_id: str
    child_id: str
    feature: str
    operator: str
    value: object
    reason: str


class EvolutionEngine:
    """Generate next-generation hypotheses from research.db statistics."""

    MIN_PARENT_OCCURRENCE = 50
    MIN_PARENT_EXP = -0.05

    def __init__(self, db_path: str = "research.db") -> None:
        self.db_path = Path(db_path)
        self.query = KnowledgeQuery(db_path)

    def close(self) -> None:
        self.query.close()

    @staticmethod
    def _normalize_threshold(value: object) -> object:
        if isinstance(value, str):
            try:
                return float(value)
            except Exception:
                return value
        return value

    @staticmethod
    def _safe_json_loads(value: object) -> list[dict[str, Any]]:
        try:
            return json.loads(value or "[]")
        except Exception:
            return []

    @staticmethod
    def _to_float(value: object) -> object:
        try:
            return float(value)
        except Exception:
            return value

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float))

    def _merge_same_feature(self, left: Condition, right: Condition) -> Condition:
        feature = left.feature
        op1, op2 = left.operator, right.operator
        v1, v2 = self._to_float(left.value), self._to_float(right.value)

        if op1 == op2:
            if op1 == ">" and self._is_number(v1) and self._is_number(v2):
                return Condition(feature, ">", max(float(v1), float(v2)))
            if op1 == "<" and self._is_number(v1) and self._is_number(v2):
                return Condition(feature, "<", min(float(v1), float(v2)))
            if op1 == "between":
                a1, b1 = v1 if isinstance(v1, tuple) and len(v1) == 2 else (None, None)
                a2, b2 = v2 if isinstance(v2, tuple) and len(v2) == 2 else (None, None)
                if self._is_number(a1) and self._is_number(b1) and self._is_number(a2) and self._is_number(b2):
                    low = max(float(a1), float(a2))
                    high = min(float(b1), float(b2))
                    if low < high:
                        return Condition(feature, "between", (low, high))
            return right

        if op1 == "between":
            low, high = v1 if isinstance(v1, tuple) and len(v1) == 2 else (None, None)
            if op2 == ">" and self._is_number(low) and self._is_number(v2):
                low = max(float(low), float(v2))
            elif op2 == "<" and self._is_number(high) and self._is_number(v2):
                high = min(float(high), float(v2))
            if self._is_number(low) and self._is_number(high) and float(low) < float(high):
                return Condition(feature, "between", (float(low), float(high)))
            return left

        if op2 == "between":
            low, high = v2 if isinstance(v2, tuple) and len(v2) == 2 else (None, None)
            if op1 == ">" and self._is_number(low) and self._is_number(v1):
                low = max(float(low), float(v1))
            elif op1 == "<" and self._is_number(high) and self._is_number(v1):
                high = min(float(high), float(v1))
            if self._is_number(low) and self._is_number(high) and float(low) < float(high):
                return Condition(feature, "between", (float(low), float(high)))
            return right

        if {op1, op2} == {">", "<"} and self._is_number(v1) and self._is_number(v2):
            low = min(float(v1), float(v2))
            high = max(float(v1), float(v2))
            if low < high:
                return Condition(feature, "between", (low, high))

        return right

    def _simplify_conditions(self, conditions: list[Condition]) -> list[Condition]:
        by_feature: dict[str, Condition] = {}
        for cond in conditions:
            if not cond.feature or not cond.operator:
                continue
            existing = by_feature.get(cond.feature)
            by_feature[cond.feature] = cond if existing is None else self._merge_same_feature(existing, cond)
        return [by_feature[k] for k in sorted(by_feature)]

    def _mutate_condition(self, cond: Condition) -> Condition:
        value = self._to_float(cond.value)
        if cond.operator == ">" and self._is_number(value):
            v = float(value)
            return Condition(cond.feature, ">", v * 1.03 if v != 0 else 0.001)
        if cond.operator == "<" and self._is_number(value):
            v = float(value)
            return Condition(cond.feature, "<", v * 0.97 if v != 0 else -0.001)
        if cond.operator == "between" and isinstance(value, tuple) and len(value) == 2:
            low, high = value
            if self._is_number(low) and self._is_number(high):
                low_f, high_f = float(low), float(high)
                width = high_f - low_f
                if width > 0:
                    pad = width * 0.10
                    new_low, new_high = low_f + pad, high_f - pad
                    if new_low < new_high:
                        return Condition(cond.feature, "between", (new_low, new_high))
        return Condition(cond.feature, cond.operator, value)

    def _add_new_feature(self, conditions: list[Condition], candidate_features: list[str]) -> list[Condition]:
        existing = {c.feature for c in conditions}
        for feat in candidate_features:
            if feat in existing:
                continue
            # add one new feature from the best thresholds as a structural mutation
            thresholds = self.best_thresholds(10)
            row = thresholds[thresholds["feature"] == feat].head(1)
            if row.empty:
                continue
            r = row.iloc[0]
            op = str(r.get("operator", ">"))
            value = self._normalize_threshold(r.get("threshold"))
            conditions.append(Condition(feat, op, value))
            break
        return self._simplify_conditions(conditions)

    def _make_child_conditions(self, conditions: list[Condition], candidate_features: list[str]) -> list[Condition]:
        simplified = self._simplify_conditions(conditions)
        mutated = [self._mutate_condition(c) for c in simplified[:2]]
        child = self._simplify_conditions(mutated)
        if len(child) < 2:
            child = self._add_new_feature(child, candidate_features)
        return child

    def best_features(self, limit: int = 10) -> pd.DataFrame:
        return self.query.top_features(limit)

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        return self.query.best_thresholds(limit)

    def _rank_bias(self, row: pd.Series) -> float:
        score = float(row.get("score", 0.0))
        expectancy = float(row.get("expectancy", 0.0))
        winrate = float(
            row.get(
                "validation_winrate",
                row.get(
                    "test_winrate",
                    row.get("winrate", row.get("train_winrate", 0.0)),
                ),
            )
        )
        occurrence = int(row.get("occurrence", 0))
        stability = float(row.get("stability", 0.0))
        occ_term = min(1.0, sqrt(max(occurrence, 0)) / 100.0) if occurrence > 0 else 0.0
        return 0.40 * expectancy + 0.20 * winrate + 0.15 * stability + 0.15 * occ_term + 0.10 * score

    def seed_candidates(self, limit: int = 20) -> list[EvolutionCandidate]:
        thresholds = self.best_thresholds(limit)
        candidates: list[EvolutionCandidate] = []
        for i, row in thresholds.iterrows():
            feature = row.get("feature")
            operator = row.get("operator")
            value = self._normalize_threshold(row.get("threshold"))
            if feature is None or operator is None:
                continue
            candidates.append(
                EvolutionCandidate(
                    parent_id="",
                    child_id=f"EV{i:06d}",
                    feature=str(feature),
                    operator=str(operator),
                    value=value,
                    reason="best_threshold",
                )
            )
        return candidates

    def _best_parent_ids(self, top_n: int = 20) -> list[str]:
        rankings = self.query.top_rankings(max(top_n * 2, top_n))
        if rankings.empty:
            return []
        scores: list[tuple[str, float]] = []
        for _, row in rankings.iterrows():
            occ = int(row.get("occurrence", 0))
            exp = float(row.get("expectancy", 0.0))
            if occ < self.MIN_PARENT_OCCURRENCE or exp < self.MIN_PARENT_EXP:
                continue
            hid = str(row.get("hypothesis_id"))
            scores.append((hid, self._rank_bias(row)))
        scores.sort(key=lambda x: x[1], reverse=True)
        parent_ids: list[str] = []
        seen: set[str] = set()
        for hid, _ in scores:
            if hid in seen:
                continue
            seen.add(hid)
            parent_ids.append(hid)
            if len(parent_ids) >= top_n:
                break
        return parent_ids

    def _best_hypothesis_rows(self, top_n: int = 20) -> pd.DataFrame:
        rankings = self.query.top_rankings(max(top_n * 2, top_n))
        if rankings.empty:
            return rankings
        rankings = rankings.copy()
        rankings["evolution_bias"] = rankings.apply(self._rank_bias, axis=1)
        sort_cols = ["evolution_bias"]
        for col in ("score", "validation_winrate", "test_winrate", "winrate", "occurrence", "expectancy"):
            if col in rankings.columns:
                sort_cols.append(col)
        rankings = rankings.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
        return rankings.head(top_n)

    def evolve_from_rankings(self, top_n: int = 20) -> list[dict]:
        rankings = self._best_hypothesis_rows(top_n)
        if rankings.empty:
            return []

        parent_ids = rankings["hypothesis_id"].astype(str).tolist()
        hypotheses = self.query.hypotheses_by_ids(parent_ids)
        hyp_map = {str(row["id"]): row for _, row in hypotheses.iterrows()} if not hypotheses.empty else {}

        thresholds = self.best_thresholds(50)
        candidate_features = [str(x) for x in thresholds["feature"].dropna().astype(str).tolist()] if not thresholds.empty else []

        proposals: list[dict] = []
        for _, r in rankings.iterrows():
            hid = str(r["hypothesis_id"])
            hrow = hyp_map.get(hid)
            if hrow is None:
                continue

            parsed = self._safe_json_loads(hrow.get("conditions"))
            conditions = [
                Condition(
                    feature=cond.get("feature"),
                    operator=cond.get("operator"),
                    value=cond.get("value"),
                )
                for cond in parsed
                if cond.get("feature") is not None and cond.get("operator") is not None
            ]
            if not conditions:
                continue

            child_conditions = self._make_child_conditions(conditions, candidate_features)
            if not child_conditions:
                continue

            proposals.append(
                {
                    "parent_id": hid,
                    "direction": hrow.get("direction", "AUTO"),
                    "conditions": [
                        {"feature": c.feature, "operator": c.operator, "value": c.value} for c in child_conditions
                    ],
                    "source_score": float(r.get("score", 0.0)),
                    "evolution_bias": float(r.get("evolution_bias", 0.0)),
                    "reason": "top_rank_mutation",
                }
            )

        parent_ids = self._best_parent_ids(top_n)
        if len(parent_ids) >= 2 and not hypotheses.empty:
            parent_rows = {str(row["id"]): row for _, row in hypotheses.iterrows()}
            for i in range(0, len(parent_ids) - 1, 2):
                p1 = parent_rows.get(parent_ids[i])
                p2 = parent_rows.get(parent_ids[i + 1])
                if p1 is None or p2 is None:
                    continue

                conds1 = self._simplify_conditions([
                    Condition(c.get("feature"), c.get("operator"), c.get("value"))
                    for c in self._safe_json_loads(p1.get("conditions"))
                    if c.get("feature") is not None and c.get("operator") is not None
                ])
                conds2 = self._simplify_conditions([
                    Condition(c.get("feature"), c.get("operator"), c.get("value"))
                    for c in self._safe_json_loads(p2.get("conditions"))
                    if c.get("feature") is not None and c.get("operator") is not None
                ])
                if not conds1 or not conds2:
                    continue

                crossover = self._simplify_conditions([conds1[0], conds2[0]])
                if not crossover:
                    continue

                # Structural mutation: if the child is too small, inject one new feature from the threshold table.
                if len(crossover) < 2 and candidate_features:
                    crossover = self._add_new_feature(crossover, candidate_features)

                proposals.append(
                    {
                        "parent_id": f"{parent_ids[i]}|{parent_ids[i + 1]}",
                        "direction": p1.get("direction", "AUTO"),
                        "conditions": [
                            {"feature": c.feature, "operator": c.operator, "value": c.value} for c in crossover
                        ],
                        "source_score": float(p1.get("score", 0.0)) + float(p2.get("score", 0.0)),
                        "evolution_bias": float(p1.get("evolution_bias", 0.0)) + float(p2.get("evolution_bias", 0.0)),
                        "reason": "top_rank_crossover",
                    }
                )

        return proposals

    def proposals_as_hypotheses(self, top_n: int = 20) -> list[Hypothesis]:
        proposals = self.evolve_from_rankings(top_n)
        out: list[Hypothesis] = []
        seen: set[str] = set()
        idx = 1

        for p in proposals:
            conds = [
                Condition(**c)
                for c in p["conditions"]
                if c.get("feature") is not None and c.get("operator") is not None
            ]
            conds = self._simplify_conditions(conds)
            if not conds:
                continue
            sig = f"{p['direction']}|{[(c.feature, c.operator, c.value) for c in conds]}"
            if sig in seen:
                continue
            seen.add(sig)
            out.append(
                Hypothesis(
                    id=f"EVO{idx:06d}",
                    direction=p["direction"],
                    conditions=conds,
                    signature=sig,
                )
            )
            idx += 1

        return out