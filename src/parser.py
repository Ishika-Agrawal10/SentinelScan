"""Utilities for loading and validating network traffic CSV files.

This module provides a focused parser for SentinelScan's network traffic
ingestion pipeline. It loads CSV data into a pandas DataFrame, validates the
required schema, and raises explicit exceptions when input is invalid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


logger = logging.getLogger(__name__)


class NetworkTrafficParser:
	"""Parse and validate SentinelScan network traffic CSV files."""

	required_columns: List[str] = ["srcip", "dstip", "sport", "dsport", "proto"]

	def load_csv(self, file_path: str | Path) -> pd.DataFrame:
		"""Load a CSV file and validate its required columns.

		Args:
			file_path: Path to the CSV file containing network traffic records.

		Returns:
			A pandas DataFrame containing the validated data.

		Raises:
			FileNotFoundError: If the file does not exist.
			EmptyDataError: If the CSV file is empty.
			ParserError: If the CSV cannot be parsed.
			ValueError: If required columns are missing.
		"""

		csv_path = Path(file_path)
		logger.info("Loading network traffic CSV from %s", csv_path)

		if not csv_path.exists():
			logger.error("CSV file not found: %s", csv_path)
			raise FileNotFoundError(f"File not found: {csv_path}")

		try:
			dataframe = pd.read_csv(csv_path)
		except FileNotFoundError:
			logger.exception("File disappeared while being read: %s", csv_path)
			raise
		except EmptyDataError:
			logger.exception("CSV file is empty: %s", csv_path)
			raise
		except ParserError:
			logger.exception("Failed to parse CSV file: %s", csv_path)
			raise

		return self.validate_columns(dataframe)

	def validate_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Validate that the DataFrame contains the required schema.

		Args:
			dataframe: DataFrame loaded from a network traffic CSV file.

		Returns:
			The same DataFrame if validation succeeds.

		Raises:
			ValueError: If any required columns are missing.
		"""

		missing_columns = [
			column for column in self.required_columns if column not in dataframe.columns
		]

		if missing_columns:
			logger.error("Missing required columns: %s", ", ".join(missing_columns))
			raise ValueError(
				"Missing required columns: " + ", ".join(missing_columns)
			)

		logger.info("CSV validation successful; required columns are present")
		return dataframe
