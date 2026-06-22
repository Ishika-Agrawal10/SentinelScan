"""Risk scoring for SentinelScan.

Calculates 0-100 normalized risk scores using weighted feature analysis.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RiskScorer:
    """Calculate risk scores from traffic features."""

    dest_weight: Final[float] = 0.4
    port_weight: Final[float] = 0.4
    conn_weight: Final[float] = 0.2

    score_column: Final[str] = "risk_score"

    def calculate_risk_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate 0-100 risk score for each source.

        Args:
            dataframe: Traffic features.

        Returns:
            DataFrame with risk_score column added.
        """
        logger.info("Calculating risk scores")

        result = dataframe.copy()

        dest = self._get_numeric(result, "unique_destinations")
        ports = self._get_numeric(result, "unique_ports")
        conns = self._get_numeric(result, "total_connections")

        norm_dest = self._normalize(dest)
        norm_ports = self._normalize(ports)
        norm_conns = self._normalize(conns)

        weighted = (
            norm_dest * self.dest_weight
            + norm_ports * self.port_weight
            + norm_conns * self.conn_weight
        )

        result[self.score_column] = np.round(weighted * 100.0, 2)

        logger.info("Risk scoring complete")
        return result

    @staticmethod
    def _get_numeric(df: pd.DataFrame, col: str) -> pd.Series:
        """Get numeric series, default to 0 if missing."""
        if col not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Min-max normalize to 0-1."""
        if series.empty:
            return pd.Series(0.0, index=series.index, dtype=float)
        min_val = float(series.min())
        max_val = float(series.max())
        if np.isclose(min_val, max_val):
            return pd.Series(0.0, index=series.index, dtype=float)
        return (series - min_val) / (max_val - min_val)
