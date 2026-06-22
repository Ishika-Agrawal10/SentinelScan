"""Feature extraction for SentinelScan.

Aggregates per-packet traffic data into per-source IP summary statistics.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract source-level features from packet-level traffic data."""

    required_columns: Final[tuple[str, ...]] = (
        "srcip",
        "dstip",
        "sport",
        "dsport",
        "proto",
    )

    def extract_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Aggregate traffic into per-source features.

        Args:
            dataframe: Raw network traffic records.

        Returns:
            Aggregated features per source IP.

        Raises:
            ValueError: If required columns missing.
        """
        logger.info("Extracting features from %d records", len(dataframe))

        missing = [col for col in self.required_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        df = dataframe.copy()
        df["srcip"] = df["srcip"].astype(str)
        df["dstip"] = df["dstip"].astype(str)
        df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce").fillna(0)

        features = (
            df.groupby("srcip", dropna=False)
            .agg(
                total_connections=("dstip", "size"),
                unique_destinations=("dstip", "nunique"),
                unique_ports=("dsport", "nunique"),
                most_common_dstport=("dsport", lambda values: int(values.mode().iloc[0]) if not values.mode().empty else 0),
                unique_protocols=("proto", "nunique"),
                common_protocol=("proto", lambda values: str(values.mode().iloc[0]) if not values.mode().empty else "UNKNOWN"),
            )
            .reset_index()
        )

        features["unique_ports"] = features["unique_ports"].fillna(0).astype(int)
        features["unique_destinations"] = features["unique_destinations"].fillna(0).astype(int)
        features["total_connections"] = features["total_connections"].fillna(0).astype(int)
        features["most_common_dstport"] = features["most_common_dstport"].fillna(0).astype(int)
        features["unique_protocols"] = features["unique_protocols"].fillna(0).astype(int)
        features["common_protocol"] = features["common_protocol"].fillna("UNKNOWN").astype(str)

        logger.info("Produced %d aggregated features", len(features))
        return features
