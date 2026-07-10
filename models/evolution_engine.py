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
    """Generate next-generation hypotheses from research.db statistics.

    V2 strategy:
    - Prefer high-value parent hypotheses from rankings.
    - Use best feature/threshold statistics as mutation seeds.
    - Apply threshold mutation and simple crossover on top parents.
    - Keep generation lightweight and deterministic enough for research runs.
    """

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

    def best_features(self, limit: int = 10) -> pd.DataFrame:
        return self.query.top_features(limit)

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        return self.query.best_thresholds(limit)

    def _rank_bias(self, row: pd.Series) -> float:
        """Bias score toward expectancy while preserving useful winners.

        This is separate from ranking_engine.score() so Evolution can prioritize
        promising parents even when the score distribution is negative.
        """
        score = float(row.get("score", 0.0))
        expectancy = float(row.get("expectancy", 0.0))
        winrate = float(row.get("validation_winrate", row.get("winrate", 0.0)))
        occurrence = int(row.get("occurrence", 0))
        stability = float(row.get("stability", 0.0))

        occ_term = min(1.0, sqrt(max(occurrence, 0)) / 100.0) if occurrence > 0 else 0.0
        return (
            0.40 * expectancy
            + 0.20 * winrate
            + 0.15 * stability
            + 0.15 * occ_term
            + 0.10 * score
        )

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
            hid = str(row.get("hypothesis_id"))
            scores.append((hid, self._rank_bias(row)))

        scores.sort(key=lambda x: x[1], reverse=True)
        seen: set[str] = set()
        parent_ids: list[str] = []
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
        rankings = rankings.sort_values(
            ["evolution_bias", "score", "validation_winrate", "occurrence"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        return rankings.head(top_n)

    def evolve_from_rankings(self, top_n: int = 20) -> list[dict]:
        """Build next-generation proposals from top ranked hypotheses.

        Returns proposal dicts that can be converted into Hypothesis objects.
        """
        rankings = self._best_hypothesis_rows(top_n)
        if rankings.empty:
            return []

        hypotheses = self.query.hypotheses(max(top_n * 2, top_n))
        hyp_map = {str(row["id"]): row for _, row in hypotheses.iterrows()} if not hypotheses.empty else {}

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

            # mutation: keep the strongest few conditions and nudge numeric thresholds.
            mutated_conditions: list[dict[str, Any]] = []
            for c in conditions[:2]:
                value = self._to_float(c.value)
                if isinstance(value, (int, float)):
                    delta = abs(float(value)) * 0.03
                    if delta == 0:
                        delta = 0.001
                    for offset in (-2, -1, 0, 1, 2):
                        mutated_conditions.append(
                            {
                                "feature": c.feature,
                                "operator": c.operator,
                                "value": float(value) + (offset * delta),
                            }
                        )
                else:
                    mutated_conditions.append(
                        {
                            "feature": c.feature,
                            "operator": c.operator,
                            "value": value,
                        }
                    )

            # crossover: combine first two conditions if possible.
            crossover_conditions: list[dict[str, Any]] = []
            if len(conditions) >= 2:
                a, b = conditions[0], conditions[1]
                crossover_conditions = [
                    {
                        "feature": a.feature,
                        "operator": a.operator,
                        "value": self._to_float(a.value),
                    },
                    {
                        "feature": b.feature,
                        "operator": b.operator,
                        "value": self._to_float(b.value),
                    },
                ]

            proposals.append(
                {
                    "parent_id": hid,
                    "direction": hrow.get("direction", "AUTO"),
                    "conditions": mutated_conditions[:6] if mutated_conditions else crossover_conditions,
                    "source_score": float(r.get("score", 0.0)),
                    "evolution_bias": float(r.get("evolution_bias", 0.0)),
                    "reason": "top_rank_mutation",
                }
            )

        # Add a few direct crossover proposals from the best parents.
        parent_ids = self._best_parent_ids(top_n)
        if len(parent_ids) >= 2:
            parent_rows = {str(row["id"]): row for _, row in hypotheses.iterrows()} if not hypotheses.empty else {}
            for i in range(0, len(parent_ids) - 1, 2):
                p1 = parent_rows.get(parent_ids[i])
                p2 = parent_rows.get(parent_ids[i + 1])
                if p1 is None or p2 is None:
                    continue

                conds1 = [
                    Condition(
                        feature=c.get("feature"),
                        operator=c.get("operator"),
                        value=c.get("value"),
                    )
                    for c in self._safe_json_loads(p1.get("conditions"))
                    if c.get("feature") is not None and c.get("operator") is not None
                ]
                conds2 = [
                    Condition(
                        feature=c.get("feature"),
                        operator=c.get("operator"),
                        value=c.get("value"),
                    )
                    for c in self._safe_json_loads(p2.get("conditions"))
                    if c.get("feature") is not None and c.get("operator") is not None
                ]
                if not conds1 or not conds2:
                    continue

                crossover = []
                crossover.extend(
                    {
                        "feature": c.feature,
                        "operator": c.operator,
                        "value": self._to_float(c.value),
                    }
                    for c in conds1[:1]
                )
                crossover.extend(
                    {
                        "feature": c.feature,
                        "operator": c.operator,
                        "value": self._to_float(c.value),
                    }
                    for c in conds2[:1]
                )

                proposals.append(
                    {
                        "parent_id": f"{parent_ids[i]}|{parent_ids[i + 1]}",
                        "direction": p1.get("direction", "AUTO"),
                        "conditions": crossover,
                        "source_score": float(p1.get("score", 0.0)) + float(p2.get("score", 0.0)),
                        "evolution_bias": float(p1.get("evolution_bias", 0.0)) + float(p2.get("evolution_bias", 0.0)),
                        "reason": "top_rank_crossover",
                    }
                )

        return proposals

    def proposals_as_hypotheses(self, top_n: int = 20) -> list[Hypothesis]:
        proposals = self.evolve_from_rankings(top_n)
        out: list[Hypothesis] = []
        idx = 1
        seen: set[str] = set()

        for p in proposals:
            conds = [Condition(**c) for c in p["conditions"] if c.get("feature") is not None and c.get("operator") is not None]
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