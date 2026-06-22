"""Network traffic CSV parser for SentinelScan.

This module provides robust CSV loading and validation for SentinelScan's
network traffic ingestion pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


logger = logging.getLogger(__name__)


class NetworkTrafficParser:
    """Load and validate network traffic CSV files."""

    required_columns: Final[tuple[str, ...]] = (
        "srcip",
        "dstip",
        "sport",
        "dsport",
        "proto",
    )

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Load and validate a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Validated DataFrame.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If validation fails.
        """
        csv_path = Path(file_path)
        logger.info("Loading CSV from: %s", csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        try:
            dataframe = pd.read_csv(csv_path)
            if dataframe.empty:
                raise ValueError("CSV file is empty")
        except EmptyDataError as exc:
            raise ValueError("CSV file is empty") from exc
        except ParserError as exc:
            raise ValueError(f"Failed to parse CSV: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Error reading CSV: {exc}") from exc

        return self.validate_columns(dataframe)

    def validate_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns exist.

        Args:
            dataframe: DataFrame to validate.

        Returns:
            The same DataFrame if valid.

        Raises:
            ValueError: If required columns missing.
        """
        missing = [col for col in self.required_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        logger.info("CSV validation successful for %d records", len(dataframe))
        return dataframe
