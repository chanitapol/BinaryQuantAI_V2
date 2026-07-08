from dataclasses import dataclass
from math import sqrt


@dataclass
class ValidationResult:
    passed: bool
    reason: str
    confidence: float
    train_winrate: float
    validation_winrate: float
    test_winrate: float
    occurrence: int
    gap: float


class ValidationEngine:
    def __init__(
        self,
        min_winrate: float = 0.53,
        min_occurrence: int = 1000,
        max_gap: float = 0.03,
    ) -> None:
        self.min_winrate = min_winrate
        self.min_occurrence = min_occurrence
        self.max_gap = max_gap

    def confidence(self, winrate: float, occurrence: int) -> float:
        if occurrence <= 0:
            return 0.0
        return max(0.0, min(1.0, (winrate - 0.5) * sqrt(occurrence) * 2))

    def evaluate(
        self,
        train_winrate: float,
        validation_winrate: float,
        test_winrate: float,
        occurrence: int,
    ) -> ValidationResult:
        best_ref = validation_winrate
        c = self.confidence(best_ref, occurrence)
        gap = abs(train_winrate - test_winrate)

        if occurrence < self.min_occurrence:
            return ValidationResult(
                False,
                "insufficient_occurrence",
                c,
                train_winrate,
                validation_winrate,
                test_winrate,
                occurrence,
                gap,
            )

        if train_winrate < self.min_winrate:
            return ValidationResult(
                False,
                "train_winrate_below_threshold",
                c,
                train_winrate,
                validation_winrate,
                test_winrate,
                occurrence,
                gap,
            )

        if validation_winrate < self.min_winrate:
            return ValidationResult(
                False,
                "validation_winrate_below_threshold",
                c,
                train_winrate,
                validation_winrate,
                test_winrate,
                occurrence,
                gap,
            )

        if test_winrate < self.min_winrate:
            return ValidationResult(
                False,
                "test_winrate_below_threshold",
                c,
                train_winrate,
                validation_winrate,
                test_winrate,
                occurrence,
                gap,
            )

        if gap > self.max_gap:
            return ValidationResult(
                False,
                "train_test_gap_too_large",
                c,
                train_winrate,
                validation_winrate,
                test_winrate,
                occurrence,
                gap,
            )

        return ValidationResult(
            True,
            "accepted",
            c,
            train_winrate,
            validation_winrate,
            test_winrate,
            occurrence,
            gap,
        )
