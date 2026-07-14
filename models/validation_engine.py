from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class ValidationResult:
    passed: bool
    status: str
    reason: str
    confidence: float
    train_winrate: float
    validation_winrate: float
    test_winrate: float
    occurrence: int
    gap: float


class ValidationEngine:
    """Validate hypotheses with separate exploration, watch, and production gates."""

    def __init__(
        self,
        min_winrate: float = 0.53,
        min_occurrence: int = 1000,
        max_gap: float = 0.03,
        exploration_min_winrate: float = 0.56,
        exploration_min_occurrence: int = 50,
        exploration_max_gap: float = 0.10,
        watch_min_winrate: float = 0.52,
        watch_min_occurrence: int = 10,
        watch_max_gap: float = 0.20,
    ) -> None:
        self.min_winrate = min_winrate
        self.min_occurrence = min_occurrence
        self.max_gap = max_gap
        self.exploration_min_winrate = exploration_min_winrate
        self.exploration_min_occurrence = exploration_min_occurrence
        self.exploration_max_gap = exploration_max_gap
        self.watch_min_winrate = watch_min_winrate
        self.watch_min_occurrence = watch_min_occurrence
        self.watch_max_gap = watch_max_gap

    def confidence(self, winrate: float, occurrence: int) -> float:
        if occurrence <= 0:
            return 0.0
        return max(0.0, min(1.0, (winrate - 0.5) * sqrt(occurrence) * 2))

    def _watch_result(
        self,
        train_winrate: float,
        validation_winrate: float,
        test_winrate: float,
        occurrence: int,
        gap: float,
        reason: str,
    ) -> ValidationResult:
        return ValidationResult(False, "WATCH", reason, self.confidence(validation_winrate, occurrence), train_winrate, validation_winrate, test_winrate, occurrence, gap)

    def evaluate(
        self,
        train_winrate: float,
        validation_winrate: float,
        test_winrate: float,
        occurrence: int,
        mode: str = "production",
    ) -> ValidationResult:
        c = self.confidence(validation_winrate, occurrence)
        gap = abs(train_winrate - test_winrate)
        train_test_ok = abs(train_winrate - test_winrate) <= self.max_gap
        validation_edge = validation_winrate >= self.min_winrate
        exploration_edge = validation_winrate >= self.exploration_min_winrate
        watch_edge = validation_winrate >= self.watch_min_winrate

        if mode == "exploration":
            if occurrence < self.exploration_min_occurrence:
                if occurrence >= self.watch_min_occurrence and watch_edge and gap <= self.watch_max_gap:
                    return self._watch_result(train_winrate, validation_winrate, test_winrate, occurrence, gap, "watch_exploration")
                return ValidationResult(False, "REJECT", "insufficient_occurrence_exploration", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

            if validation_winrate < self.exploration_min_winrate:
                if occurrence >= self.watch_min_occurrence and watch_edge and gap <= self.watch_max_gap:
                    return self._watch_result(train_winrate, validation_winrate, test_winrate, occurrence, gap, "watch_exploration")
                return ValidationResult(False, "REJECT", "validation_winrate_below_threshold_exploration", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

            if gap > self.exploration_max_gap:
                if occurrence >= self.watch_min_occurrence and watch_edge and gap <= self.watch_max_gap:
                    return self._watch_result(train_winrate, validation_winrate, test_winrate, occurrence, gap, "watch_exploration")
                return ValidationResult(False, "REJECT", "train_test_gap_too_large_exploration", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

            if not train_test_ok:
                return ValidationResult(False, "REJECT", "train_test_gap_too_large_exploration", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

            # Exploration PASS requires the edge to be visible on train, validation and test.
            if train_winrate < 0.50 or test_winrate < self.exploration_min_winrate:
                if occurrence >= self.watch_min_occurrence and watch_edge and gap <= self.watch_max_gap:
                    return self._watch_result(train_winrate, validation_winrate, test_winrate, occurrence, gap, "watch_exploration")
                return ValidationResult(False, "REJECT", "train_or_test_below_exploration_gate", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

            return ValidationResult(True, "PASS", "accepted_exploration", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        if occurrence < self.min_occurrence:
            if occurrence >= self.watch_min_occurrence and watch_edge and gap <= self.watch_max_gap:
                return self._watch_result(train_winrate, validation_winrate, test_winrate, occurrence, gap, "watch_production")
            return ValidationResult(False, "REJECT", "insufficient_occurrence", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        if train_winrate < self.min_winrate:
            return ValidationResult(False, "REJECT", "train_winrate_below_threshold", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        if validation_winrate < self.min_winrate:
            return ValidationResult(False, "REJECT", "validation_winrate_below_threshold", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        if test_winrate < self.min_winrate:
            return ValidationResult(False, "REJECT", "test_winrate_below_threshold", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        if gap > self.max_gap:
            return ValidationResult(False, "REJECT", "train_test_gap_too_large", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)

        return ValidationResult(True, "PASS", "accepted", c, train_winrate, validation_winrate, test_winrate, occurrence, gap)