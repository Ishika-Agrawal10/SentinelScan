"""SentinelScan backend processing pipeline.

This module orchestrates the threat detection workflow from raw CSV data
through feature extraction, threat detection, risk scoring, and severity
classification.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path for proper module resolution
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.attack_classifier import AttackClassifier
from src.classifier import ThreatClassifier
from src.detector import ThreatDetector
from src.feature_extractor import FeatureExtractor
from src.ml_detector import MLDetector
from src.parser import NetworkTrafficParser
from src.reporter import ReportGenerator
from src.risk_scorer import RiskScorer
from src.threat_intelligence import ThreatIntelligence


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SentinelScan threat detection pipeline"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/raw/sample_traffic.csv",
        help="Path to network traffic CSV (default: data/raw/sample_traffic.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="reports/final_analysis.csv",
        help="Output report path (default: reports/final_analysis.csv)",
    )
    return parser.parse_args()


def process_pipeline(input_path: Path, output_path: Path) -> bool:
    """Run the full threat detection pipeline.

    Args:
        input_path: Path to input CSV file.
        output_path: Path to output report file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        print("\n" + "=" * 60)
        print("SentinelScan Threat Detection Pipeline")
        print("=" * 60 + "\n")

        # Stage 1: Parse
        print("[1/6] Loading and validating CSV...")
        parser = NetworkTrafficParser()
        raw_data = parser.load_csv(input_path)
        print(f"      ✓ Loaded {len(raw_data)} records\n")

        # Stage 2: Extract Features
        print("[2/6] Extracting traffic features...")
        extractor = FeatureExtractor()
        features = extractor.extract_features(raw_data)
        print(f"      ✓ Extracted features for {len(features)} source IPs\n")

        # Stage 3: Detect Threats
        print("[3/8] Detecting suspicious activity...")
        detector = ThreatDetector()
        detections = detector.detect_suspicious_activity(features)
        suspicious_count = len(detections[detections["detection_status"] == "SUSPICIOUS"])
        print(f"      ✓ Identified {suspicious_count} suspicious sources\n")

        # Stage 4: Score Risk
        print("[4/8] Calculating risk scores...")
        scorer = RiskScorer()
        scored = scorer.calculate_risk_score(detections)
        avg_risk = scored["risk_score"].mean()
        print(f"      ✓ Average risk score: {avg_risk:.2f}\n")

        # Stage 5: Classify Severity
        print("[5/8] Classifying severity levels...")
        classifier = ThreatClassifier()
        classified = classifier.classify_severity(scored)
        critical_count = len(classified[classified["severity"] == "CRITICAL"])
        print(f"      ✓ Identified {critical_count} critical threats\n")

        # Stage 6: ML Anomaly Detection
        print("[6/8] Running ML anomaly detection...")
        ml_detector = MLDetector()
        ml_result = ml_detector.analyze(classified)
        print(f"      ✓ ML anomaly detection complete\n")

        # Stage 7: Attack Classification
        print("[7/8] Classifying attack patterns...")
        attack_classifier = AttackClassifier()
        attack_result = attack_classifier.classify(ml_result)
        print("      ✓ Attack classification complete\n")

        # Stage 8: Threat Intelligence
        print("[8/8] Generating threat intelligence and executive report...")
        intelligence = ThreatIntelligence()
        intelligence_result = intelligence.analyze(attack_result)
        reporter = ReportGenerator()
        reporter.save_report(intelligence_result, output_path)
        intelligence.export_executive_text(intelligence_result, Path(output_path).parent / "sentinelscan_executive_summary.txt")
        print(f"      ✓ Report saved to {output_path}\n")

        # Summary
        print("=" * 60)
        print("Analysis Results:")
        print(f"  Total Sources:      {len(intelligence_result)}")
        print(f"  SUSPICIOUS:         {len(intelligence_result[intelligence_result['detection_status'] == 'SUSPICIOUS'])}")
        print(f"  MONITOR:            {len(intelligence_result[intelligence_result['detection_status'] == 'MONITOR'])}")
        print(f"  NORMAL:             {len(intelligence_result[intelligence_result['detection_status'] == 'NORMAL'])}")
        print(f"  ANOMALY:            {len(intelligence_result[intelligence_result['ml_result'] == 'ANOMALY'])}")
        print(f"  CRITICAL:           {len(intelligence_result[intelligence_result['severity'] == 'CRITICAL'])}")
        print(f"  HIGH:               {len(intelligence_result[intelligence_result['severity'] == 'HIGH'])}")
        print(f"  MEDIUM:             {len(intelligence_result[intelligence_result['severity'] == 'MEDIUM'])}")
        print(f"  LOW:                {len(intelligence_result[intelligence_result['severity'] == 'LOW'])}")
        print(f"  Avg Risk Score:     {avg_risk:.2f}")
        print("=" * 60 + "\n")

        # Display top threats
        print("Top 5 Highest Risk Sources:")
        top_5 = intelligence_result.nlargest(5, "risk_score")[
            ["srcip", "total_connections", "unique_destinations", "unique_ports", "risk_score", "severity", "attack_type", "ml_result"]
        ]
        print(top_5.to_string(index=False))
        print()

        return True

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        print(f"\n✗ Error: {exc}\n")
        return False
    except ValueError as exc:
        logger.error("Validation failed: %s", exc)
        print(f"\n✗ Error: {exc}\n")
        return False
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error")
        print(f"\n✗ Unexpected error: {exc}\n")
        return False


def main() -> None:
    """Main entry point."""
    configure_logging()
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output)

    success = process_pipeline(input_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
	main()
