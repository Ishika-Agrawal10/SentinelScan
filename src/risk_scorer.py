"""Risk scoring utilities for SentinelScan threat analysis.

This module converts feature-engineered traffic summaries into a normalized
risk score between 0 and 100 using a weighted scoring model.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RiskScorer:
	"""Compute risk scores for network traffic feature rows."""

	destination_weight: Final[float] = 0.4
	port_weight: Final[float] = 0.4
	connection_weight: Final[float] = 0.2

	required_columns: Final[tuple[str, ...]] = (
		"srcip",
		"total_connections",
		"unique_destinations",
		"unique_ports",
	)

	score_column: Final[str] = "risk_score"

	def calculate_risk_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Calculate a 0-100 risk score for each traffic entity.

		Args:
			dataframe: Feature-engineered network traffic data.

		Returns:
			A copy of the DataFrame with a `risk_score` column appended.
		"""

		logger.info("Calculating risk scores for %d records", len(dataframe))

		result = dataframe.copy()

		score_inputs = {
			"unique_destinations": self._safe_numeric_series(result, "unique_destinations"),
			"unique_ports": self._safe_numeric_series(result, "unique_ports"),
			"total_connections": self._safe_numeric_series(result, "total_connections"),
		}

		normalized_destinations = self._normalize_series(score_inputs["unique_destinations"])
		normalized_ports = self._normalize_series(score_inputs["unique_ports"])
		normalized_connections = self._normalize_series(score_inputs["total_connections"])

		weighted_score = (
			normalized_destinations * self.destination_weight
			+ normalized_ports * self.port_weight
			+ normalized_connections * self.connection_weight
		)

		result[self.score_column] = np.round(weighted_score * 100.0, 2)

		logger.info("Risk scoring completed for %d records", len(result))
		return result

	def _safe_numeric_series(self, dataframe: pd.DataFrame, column_name: str) -> pd.Series:
		"""Return a numeric series for scoring, defaulting to zeros if missing."""

		if column_name not in dataframe.columns:
			logger.warning("Missing column %s; defaulting to zeros for risk scoring", column_name)
			return pd.Series(0.0, index=dataframe.index, dtype=float)

		numeric_series = pd.to_numeric(dataframe[column_name], errors="coerce").fillna(0.0)
		return numeric_series.astype(float)

	@staticmethod
	def _normalize_series(series: pd.Series) -> pd.Series:
		"""Normalize a numeric series to the 0-1 range using min-max scaling."""

		if series.empty:
			return series.astype(float)

		minimum = float(series.min())
		maximum = float(series.max())

		if np.isclose(minimum, maximum):
			return pd.Series(0.0, index=series.index, dtype=float)

		return (series - minimum) / (maximum - minimum)
