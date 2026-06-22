"""Attack pattern classification for SentinelScan."""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)


class AttackClassifier:
    """Classify source IP behavior into attack patterns."""

    attack_type_column: Final[str] = "attack_type"

    def classify(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Assign attack type labels based on source-level indicators."""
        logger.info("Classifying attack patterns for %d sources", len(dataframe))

        result = dataframe.copy()
        attack_types = []

        for _, row in result.iterrows():
            attack_types.append(self._classify_row(row))

        result[self.attack_type_column] = attack_types
        logger.info("Attack classification complete")
        return result

    def _classify_row(self, row: pd.Series) -> str:
        unique_destinations = self._safe_int(row.get("unique_destinations", 0))
        unique_ports = self._safe_int(row.get("unique_ports", 0))
        total_connections = self._safe_int(row.get("total_connections", 0))

        if unique_destinations <= 1 and unique_ports > 15:
            return "Vertical Scan"

        if unique_destinations > 10 and unique_ports <= 5:
            return "Horizontal Scan"

        if unique_ports >= 5 and self._has_service_enumeration(row):
            return "Service Enumeration"

        if unique_destinations > 20 and unique_ports > 20 and total_connections > 50:
            return "Aggressive Reconnaissance"

        return "Normal Traffic"

    @staticmethod
    def _has_service_enumeration(row: pd.Series) -> bool:
        common_ports = {21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3389}
        ports = row.get("unique_ports", 0)
        if pd.api.types.is_numeric_dtype(type(ports)):
            return int(ports) >= 5
        return False

    @staticmethod
    def _safe_int(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
