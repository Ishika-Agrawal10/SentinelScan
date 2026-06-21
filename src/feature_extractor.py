"""Feature extraction utilities for SentinelScan network traffic analysis.

This module converts packet-level traffic records into per-source summary
features used by downstream threat detection, risk scoring, and classification.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd


logger = logging.getLogger(__name__)


class FeatureExtractor:
	"""Aggregate network traffic rows into source-level features."""

	required_columns: Final[tuple[str, ...]] = (
		"srcip",
		"dstip",
		"sport",
		"dsport",
		"proto",
	)

	def extract_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Transform raw traffic rows into aggregated source features.

		Args:
			dataframe: Parsed network traffic records.

		Returns:
			A DataFrame with one row per source IP containing the fields required
			by the downstream detection pipeline.

		Raises:
			ValueError: If any required input columns are missing.
		"""

		logger.info("Extracting features from %d raw records", len(dataframe))

		missing_columns = [
			column for column in self.required_columns if column not in dataframe.columns
		]

		if missing_columns:
			message = "Missing required columns for feature extraction: " + ", ".join(
				missing_columns
			)
			logger.error(message)
			raise ValueError(message)

		normalized = dataframe.copy()
		normalized["srcip"] = normalized["srcip"].astype(str)
		normalized["dstip"] = normalized["dstip"].astype(str)
		normalized["dsport"] = pd.to_numeric(normalized["dsport"], errors="coerce")

		features = (
			normalized.groupby("srcip", dropna=False)
			.agg(
				total_connections=("dstip", "size"),
				unique_destinations=("dstip", "nunique"),
				unique_ports=("dsport", "nunique"),
			)
			.reset_index()
		)

		features["unique_ports"] = features["unique_ports"].fillna(0).astype(int)
		features["unique_destinations"] = features["unique_destinations"].fillna(0).astype(int)
		features["total_connections"] = features["total_connections"].fillna(0).astype(int)

		logger.info("Feature extraction produced %d aggregated rows", len(features))
		return features
