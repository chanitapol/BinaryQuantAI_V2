from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

    def __init__(self, db_path: str = "research.db") -> None:
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

    def best_features(self, limit: int = 10) -> pd.DataFrame:
        return self.query.top_features(limit)

    def best_thresholds(self, limit: int = 20) -> pd.DataFrame:
        return self.query.best_thresholds(limit)

    def seed_candidates(self, limit: int = 20) -> list[EvolutionCandidate]:
        thresholds = self.best_thresholds(limit)
        candidates: list[EvolutionCandidate] = []
        for i, row in thresholds.iterrows():
            feature = row.get("feature")
            operator = row.get("operator")
            value = self._normalize_threshold(row.get("threshold"))
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

    def evolve_from_rankings(self, top_n: int = 20) -> list[dict]:
        """Build a lightweight next-generation proposal list from top rankings.

        This does not mutate the database. It returns proposal dicts that can be
        converted into Hypothesis objects later.
        """
        rankings = self.query.top_rankings(top_n)
        if rankings.empty:
            return []

        hypotheses = self.query.hypotheses(top_n)
        hyp_map = {row["id"]: row for _, row in hypotheses.iterrows()} if not hypotheses.empty else {}

        proposals: list[dict] = []
        for _, r in rankings.iterrows():
            hid = r["hypothesis_id"]
            hrow = hyp_map.get(hid)
            if hrow is None:
                continue

            conditions = []
            try:
                parsed = pd.io.json.loads(hrow["conditions"])
            except Exception:
                parsed = []

            for cond in parsed:
                conditions.append(
                    Condition(
                        feature=cond.get("feature"),
                        operator=cond.get("operator"),
                        value=cond.get("value"),
                    )
                )

            if not conditions:
                continue

            # Simple mutation: keep strongest conditions and propose a tighter threshold.
            child_conditions = []
            for c in conditions[:2]:
                value = c.value
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except Exception:
                        pass
                child_conditions.append(
                    {
                        "feature": c.feature,
                        "operator": c.operator,
                        "value": value,
                    }
                )

            proposals.append(
                {
                    "parent_id": hid,
                    "direction": hrow.get("direction", "AUTO"),
                    "conditions": child_conditions,
                    "source_score": float(r.get("score", 0.0)),
                    "reason": "top_rank_mutation",
                }
            )

        return proposals

    def proposals_as_hypotheses(self, top_n: int = 20) -> list[Hypothesis]:
        proposals = self.evolve_from_rankings(top_n)
        out: list[Hypothesis] = []
        idx = 1
        for p in proposals:
            conds = [Condition(**c) for c in p["conditions"]]
            sig = f"{p['direction']}|{[(c.feature, c.operator, c.value) for c in conds]}"
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
