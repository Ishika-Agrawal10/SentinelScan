"""Command-line application pipeline for SentinelScan.

This module wires together the backend processing stages for loading traffic
data, extracting features, detecting threats, scoring risk, and classifying
severity.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.classifier import ThreatClassifier
from src.detector import ThreatDetector
from src.feature_extractor import FeatureExtractor
from src.parser import NetworkTrafficParser
from src.risk_scorer import RiskScorer


logger = logging.getLogger(__name__)


def configure_logging() -> None:
	"""Configure application logging for console output."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the SentinelScan pipeline."""

	parser = argparse.ArgumentParser(
		description="Run the SentinelScan backend processing pipeline."
	)
	parser.add_argument(
		"input_csv",
		help="Path to the raw network traffic CSV file.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run the SentinelScan backend processing pipeline."""

	configure_logging()
	arguments = parse_args()

	input_path = Path(arguments.input_csv)
	output_path = Path("reports") / "final_analysis.csv"

	parser = NetworkTrafficParser()
	extractor = FeatureExtractor()
	detector = ThreatDetector()
	scorer = RiskScorer()
	classifier = ThreatClassifier()

	try:
		print("[1/5] Loading data...")
		raw_data = parser.load_csv(input_path)

		print("[2/5] Extracting features...")
		features = extractor.extract_features(raw_data)

		print("[3/5] Detecting threats...")
		detections = detector.detect_suspicious_activity(features)

		print("[4/5] Calculating risk scores...")
		scored = scorer.calculate_risk_score(detections)

		print("[5/5] Classifying severity...")
		final_results = classifier.classify_severity(scored)

		output_path.parent.mkdir(parents=True, exist_ok=True)
		final_results.to_csv(output_path, index=False)

		print("\nFinal Results (first 10 rows):")
		print(final_results.head(10).to_string(index=False))
		print(f"\nSaved final output to: {output_path}")

	except FileNotFoundError as exc:
		logger.exception("Input file could not be found")
		raise SystemExit(f"Error: {exc}") from exc
	except ValueError as exc:
		logger.exception("Validation failed during pipeline execution")
		raise SystemExit(f"Error: {exc}") from exc
	except Exception as exc:  # pragma: no cover - defensive top-level guard
		logger.exception("Unexpected error while running SentinelScan")
		raise SystemExit(f"Unexpected error: {exc}") from exc


if __name__ == "__main__":
	main()
