"""Isolation Forest anomaly detection for SentinelScan."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class MLDetector:
    """Unsupervised anomaly detector for network threat analysis."""

    ml_prediction_column: Final[str] = "ml_prediction"
    ml_result_column: Final[str] = "ml_result"
    anomaly_score_column: Final[str] = "anomaly_score"
    confidence_column: Final[str] = "confidence_percentage"
    final_assessment_column: Final[str] = "final_threat_assessment"

    feature_columns: Final[tuple[str, ...]] = (
        "total_connections",
        "unique_destinations",
        "unique_ports",
        "risk_score",
    )

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Analyze aggregated traffic using Isolation Forest."""
        logger.info("Running ML anomaly detection for %d sources", len(dataframe))

        result = dataframe.copy()
        features = self._prepare_features(result)

        if len(features) < 5 or features.nunique().sum() < 10:
            logger.warning("Insufficient data for Isolation Forest, defaulting ML results to NORMAL")
            result[self.ml_prediction_column] = "NORMAL"
            result[self.ml_result_column] = "NORMAL"
            result[self.anomaly_score_column] = 0.0
            result[self.confidence_column] = 100.0
            result[self.final_assessment_column] = self._combine_assessment(result)
            return result

        try:
            model = IsolationForest(
                contamination="auto",
                n_estimators=150,
                max_samples="auto",
                random_state=42,
                behaviour="deprecated",
            )
        except TypeError:
            model = IsolationForest(
                contamination="auto",
                n_estimators=150,
                max_samples="auto",
                random_state=42,
            )

        model.fit(features)
        raw_scores = pd.Series(
            -model.decision_function(features),
            index=result.index,
        )

        anomaly_score = self._normalize_series(raw_scores)
        is_anomaly = model.predict(features) == -1

        result[self.anomaly_score_column] = anomaly_score
        result[self.ml_prediction_column] = np.where(is_anomaly, "SUSPICIOUS", "NORMAL")
        result[self.ml_result_column] = np.where(is_anomaly, "ANOMALY", "NORMAL")
        result[self.confidence_column] = self._compute_confidence(anomaly_score, is_anomaly)
        result[self.final_assessment_column] = self._combine_assessment(result)

        logger.info("ML anomaly detection complete")
        return result

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        features = dataframe.reindex(columns=self.feature_columns).copy()
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        return features.fillna(0.0).astype(float)

    @staticmethod
    def _normalize_series(series: pd.Series | np.ndarray) -> pd.Series:
        series = pd.Series(series)
        if len(series) == 0:
            return pd.Series([], dtype=float)
        min_val = float(series.min())
        max_val = float(series.max())
        if np.isclose(min_val, max_val):
            return pd.Series(0.0, index=series.index, dtype=float)
        normalized = (series - min_val) / (max_val - min_val)
        return np.round(normalized * 100.0, 2)

    def _compute_confidence(self, anomaly_score: pd.Series, is_anomaly: np.ndarray) -> pd.Series:
        confidence = np.where(
            is_anomaly,
            60.0 + np.minimum(40.0, anomaly_score * 0.4),
            80.0 + np.minimum(20.0, 100.0 - anomaly_score * 0.4),
        )
        return np.round(confidence, 2)

    def _combine_assessment(self, result: pd.DataFrame) -> pd.Series:
        rule_status = result.get("detection_status", pd.Series(dtype=str))
        ml_result = result.get(self.ml_result_column, pd.Series(dtype=str))

        assessments: list[str] = []
        for status, ml in zip(rule_status.fillna("NORMAL"), ml_result.fillna("NORMAL")):
            if status == "SUSPICIOUS" and ml == "ANOMALY":
                assessments.append("HIGH RISK PORT SCANNING")
            elif status == "SUSPICIOUS" and ml == "NORMAL":
                assessments.append("RULE-BASED SUSPICIOUS ACTIVITY")
            elif status != "SUSPICIOUS" and ml == "ANOMALY":
                assessments.append("ANOMALOUS NETWORK BEHAVIOR")
            else:
                assessments.append("NORMAL TRAFFIC")

        return pd.Series(assessments, index=result.index)
