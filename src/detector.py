"""Threat detection engine for SentinelScan network traffic analysis.

This module converts feature-engineered traffic summaries into high-level
security classifications for suspicious port scanning behavior.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd


logger = logging.getLogger(__name__)


class ThreatDetector:
	"""Detect suspicious port-scanning activity in feature-engineered traffic data."""

	required_columns = [
		"srcip",
		"total_connections",
		"unique_destinations",
		"unique_ports",
	]

	status_column = "detection_status"
	reason_column = "detection_reason"

	def detect_suspicious_activity(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Annotate feature rows with threat detection outcomes.

		Args:
			dataframe: Feature-engineered network traffic data.

		Returns:
			A copy of the input DataFrame with detection status and reason columns
			appended.
		"""

		logger.info("Starting threat detection for %d records", len(dataframe))

		result = dataframe.copy()

		missing_columns = [
			column for column in self.required_columns if column not in result.columns
		]

		if missing_columns:
			logger.warning(
				"Missing required feature columns: %s",
				", ".join(missing_columns),
			)
			result[self.status_column] = "NORMAL"
			result[self.reason_column] = (
				"Missing required feature columns; unable to reliably evaluate "
				"suspicious scanning indicators."
			)
			return result

		statuses = []
		reasons = []

		for _, row in result.iterrows():
			status, reason = self._classify_row(row)
			statuses.append(status)
			reasons.append(reason)

		result[self.status_column] = statuses
		result[self.reason_column] = reasons

		logger.info("Threat detection completed for %d records", len(result))
		return result

	def _classify_row(self, row: pd.Series) -> Tuple[str, str]:
		"""Classify a single aggregated traffic row."""

		unique_destinations = self._safe_int(row.get("unique_destinations", 0))
		unique_ports = self._safe_int(row.get("unique_ports", 0))

		if unique_destinations > 30 and unique_ports > 20:
			return (
				"SUSPICIOUS",
				"High destination diversity and high port diversity consistent with port scanning behavior.",
			)

		if unique_destinations > 15 or unique_ports > 10:
			return (
				"MONITOR",
				"Elevated network exploration activity detected.",
			)

		return (
			"NORMAL",
			"No suspicious scanning indicators observed.",
		)

	@staticmethod
	def _safe_int(value: object) -> int:
		"""Convert a value to an integer without raising on bad input."""

		if pd.isna(value):
			return 0

		try:
			return int(value)
		except (TypeError, ValueError):
			logger.debug("Non-numeric value encountered during threat detection: %r", value)
			return 0
