"""Severity classification for SentinelScan.

Maps risk scores to severity levels for threat prioritization.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd


logger = logging.getLogger(__name__)


class ThreatClassifier:
    """Map risk scores to severity levels."""

    severity_column: Final[str] = "severity"

    def classify_severity(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Classify severity based on risk score.

        Args:
            dataframe: DataFrame with risk_score column.

        Returns:
            DataFrame with severity column added.
        """
        logger.info("Classifying severity")

        result = dataframe.copy()

        if "risk_score" not in result.columns:
            result[self.severity_column] = "LOW"
            return result

        risk_scores = pd.to_numeric(result["risk_score"], errors="coerce")
        result[self.severity_column] = risk_scores.apply(self._map_severity)

        logger.info("Severity classification complete")
        return result

    @staticmethod
    def _map_severity(score: float | None) -> str:
        """Map numeric score to severity label."""
        if score is None or pd.isna(score):
            return "LOW"

        try:
            score = float(score)
        except (TypeError, ValueError):
            return "LOW"

        score = min(max(score, 0.0), 100.0)

        if score <= 25:
            return "LOW"
        if score <= 50:
            return "MEDIUM"
        if score <= 75:
            return "HIGH"
        return "CRITICAL"
