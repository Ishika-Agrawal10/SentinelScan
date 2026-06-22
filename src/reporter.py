"""Report generation for SentinelScan.

Exports analysis results to CSV format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate analysis reports."""

    def save_report(self, dataframe: pd.DataFrame, output_path: str | Path) -> Path:
        """Save analysis results to CSV.

        Args:
            dataframe: Analysis results.
            output_path: Output file path.

        Returns:
            Path to saved report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(output_path, index=False)
        logger.info("Report saved to: %s", output_path)

        return output_path
