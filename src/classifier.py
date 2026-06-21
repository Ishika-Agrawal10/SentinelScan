"""Severity classification utilities for SentinelScan.

This module maps normalized risk scores to human-readable severity levels for
downstream reporting and analyst workflows.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ThreatClassifier:
	"""Classify traffic risk scores into severity levels."""

	required_columns: Final[tuple[str, ...]] = ("srcip", "risk_score")

	severity_column: Final[str] = "severity"

	low_upper_bound: Final[float] = 25.0
	medium_upper_bound: Final[float] = 50.0
	high_upper_bound: Final[float] = 75.0

	def classify_severity(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Classify each row into a severity bucket based on risk score.

		Args:
			dataframe: DataFrame containing `risk_score` values.

		Returns:
			A copy of the input DataFrame with a `severity` column appended.
		"""

		logger.info("Classifying severity for %d records", len(dataframe))

		result = dataframe.copy()

		missing_columns = [
			column for column in self.required_columns if column not in result.columns
		]

		if missing_columns:
			logger.warning(
				"Missing required columns for severity classification: %s",
				", ".join(missing_columns),
			)

		if "risk_score" not in result.columns:
			result[self.severity_column] = "LOW"
			logger.warning(
				"risk_score column missing; defaulting all rows to LOW severity"
			)
			return result

		risk_scores = pd.to_numeric(result["risk_score"], errors="coerce")
		severity_values = risk_scores.apply(self._map_severity)

		result[self.severity_column] = severity_values

		logger.info("Severity classification completed for %d records", len(result))
		return result

	def _map_severity(self, risk_score: float | int | np.floating | np.integer | None) -> str:
		"""Map a numeric risk score to a severity label."""

		if risk_score is None or pd.isna(risk_score):
			return "LOW"

		try:
			score = float(risk_score)
		except (TypeError, ValueError):
			logger.debug("Invalid risk score encountered: %r", risk_score)
			return "LOW"

		score = min(max(score, 0.0), 100.0)

		if score <= self.low_upper_bound:
			return "LOW"

		if score <= self.medium_upper_bound:
			return "MEDIUM"

		if score <= self.high_upper_bound:
			return "HIGH"

		return "CRITICAL"
