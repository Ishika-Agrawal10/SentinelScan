"""Threat detection engine for SentinelScan.

Detects suspicious port-scanning behavior based on traffic feature analysis.
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


class ThreatDetector:
    """Classify traffic as suspicious, monitor, or normal."""

    required_columns: Final[tuple[str, ...]] = (
        "srcip",
        "total_connections",
        "unique_destinations",
        "unique_ports",
    )

    status_column: Final[str] = "detection_status"
    reason_column: Final[str] = "detection_reason"

    def detect_suspicious_activity(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Classify threat status for each source IP.

        Args:
            dataframe: Aggregated traffic features.

        Returns:
            DataFrame with detection_status and detection_reason columns.
        """
        logger.info("Detecting threats for %d records", len(dataframe))

        result = dataframe.copy()
        statuses = []
        reasons = []

        for _, row in result.iterrows():
            status, reason = self._classify_row(row)
            statuses.append(status)
            reasons.append(reason)

        result[self.status_column] = statuses
        result[self.reason_column] = reasons

        logger.info("Threat detection complete")
        return result

    def _classify_row(self, row: pd.Series) -> Tuple[str, str]:
        """Classify a single source IP."""
        dest = self._safe_int(row.get("unique_destinations", 0))
        ports = self._safe_int(row.get("unique_ports", 0))

        if dest > 30 and ports > 20:
            return "SUSPICIOUS", "High destination and port diversity detected (port scanning behavior)"

        if dest > 15 or ports > 10:
            return "MONITOR", "Elevated destination or port diversity detected"

        return "NORMAL", "Normal traffic pattern"

    @staticmethod
    def _safe_int(value: object) -> int:
        """Safely convert value to int."""
        if pd.isna(value):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
