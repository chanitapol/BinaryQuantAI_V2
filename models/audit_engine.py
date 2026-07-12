from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AuditResult:
    passed: bool
    issues: list[str]
    metrics: dict[str, Any]


class AuditEngine:
    """Run lightweight integrity checks before hypothesis discovery."""

    def audit(self, df: pd.DataFrame, split) -> AuditResult:
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        if df is None or df.empty:
            return AuditResult(False, ["dataset is empty"], metrics)

        metrics["rows"] = int(len(df))
        metrics["features"] = int(len(df.columns))
        metrics["train_rows"] = int(len(split.train))
        metrics["validation_rows"] = int(len(split.validation))
        metrics["test_rows"] = int(len(split.test))

        # Basic integrity
        null_count = int(df.isna().sum().sum())
        inf_count = int((~pd.DataFrame(df).replace([float("inf"), float("-inf")], pd.NA).isna()).sum().sum())
        metrics["null_count"] = null_count
        metrics["inf_count"] = inf_count
        if null_count > 0:
            issues.append(f"dataset contains {null_count} null values")
        if inf_count > 0:
            issues.append(f"dataset contains {inf_count} inf values")

        # Split overlap by index
        train_idx = set(split.train.index)
        val_idx = set(split.validation.index)
        test_idx = set(split.test.index)
        overlap_tv = len(train_idx & val_idx)
        overlap_tt = len(train_idx & test_idx)
        overlap_vt = len(val_idx & test_idx)
        metrics["overlap_train_validation"] = overlap_tv
        metrics["overlap_train_test"] = overlap_tt
        metrics["overlap_validation_test"] = overlap_vt
        if overlap_tv or overlap_tt or overlap_vt:
            issues.append("split index overlap detected")

        # Chronological order, if a timestamp-like column exists
        time_col = self._find_time_column(df)
        if time_col is not None:
            metrics["time_column"] = time_col
            ts = pd.to_datetime(df[time_col], errors="coerce")
            if ts.isna().any():
                issues.append(f"time column '{time_col}' contains unparseable timestamps")
            if not ts.is_monotonic_increasing:
                issues.append(f"time column '{time_col}' is not monotonic increasing")

        # Duplicate rows
        dup_rows = int(df.duplicated().sum())
        metrics["duplicate_rows"] = dup_rows
        if dup_rows > 0:
            issues.append(f"dataset contains {dup_rows} duplicate rows")

        # Split label / target balance proxy
        for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            part_metrics = self._label_balance_metrics(part)
            metrics[f"{name}_label_balance"] = part_metrics
            if part_metrics.get("distinct_labels", 0) < 2:
                issues.append(f"{name} split has insufficient label diversity")

        passed = len(issues) == 0
        return AuditResult(passed=passed, issues=issues, metrics=metrics)

    @staticmethod
    def _find_time_column(df: pd.DataFrame) -> str | None:
        candidates = ["time", "timestamp", "datetime", "date", "candle_time"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _label_balance_metrics(part: pd.DataFrame) -> dict[str, Any]:
        for col in ("label", "target", "y", "direction", "signal"):
            if col in part.columns:
                counts = part[col].value_counts(dropna=False)
                return {
                    "label_column": col,
                    "distinct_labels": int(counts.shape[0]),
                    "top_label_count": int(counts.iloc[0]) if not counts.empty else 0,
                }
        return {"label_column": None, "distinct_labels": 0, "top_label_count": 0}
