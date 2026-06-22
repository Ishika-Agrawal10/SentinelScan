"""Threat intelligence engine for SentinelScan."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)


class ThreatIntelligence:
    """Generate analyst-friendly threat intelligence metrics."""

    analyst_findings_column: Final[str] = "intelligence_findings"

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add threat intelligence insights to the analysis results."""
        logger.info("Generating threat intelligence for %d sources", len(dataframe))

        result = dataframe.copy()
        result = self._add_summary_metrics(result)
        findings = self._generate_findings(result)
        result[self.analyst_findings_column] = "\n".join(findings)

        logger.info("Threat intelligence analysis complete")
        return result

    def _add_summary_metrics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        summary = self._build_summary(dataframe)

        for key, value in summary.items():
            dataframe[key] = value

        return dataframe

    def _build_summary(self, dataframe: pd.DataFrame) -> dict[str, object]:
        most_active = self._most_active_source(dataframe)
        highest_risk = self._highest_risk_host(dataframe)
        most_targeted_port = self._most_targeted_port(dataframe)
        most_used_protocol = self._most_used_protocol(dataframe)
        network_level = self._compute_network_security_level(dataframe)
        threat_density = self._compute_threat_density(dataframe)

        return {
            "most_active_source_ip": most_active,
            "highest_risk_host": highest_risk,
            "most_targeted_port": most_targeted_port,
            "most_used_protocol": most_used_protocol,
            "network_threat_level": network_level,
            "threat_density": threat_density,
        }

    def _most_active_source(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "N/A"
        best = dataframe.nlargest(1, "total_connections").iloc[0]
        return str(best.get("srcip", "N/A"))

    def _highest_risk_host(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "N/A"
        best = dataframe.nlargest(1, "risk_score").iloc[0]
        return str(best.get("srcip", "N/A"))

    def _most_targeted_port(self, dataframe: pd.DataFrame) -> int | str:
        if dataframe.empty or "most_common_dstport" not in dataframe.columns:
            return "N/A"
        grouped = dataframe["most_common_dstport"].dropna().astype(int)
        return int(grouped.mode().iloc[0]) if not grouped.mode().empty else "N/A"

    def _most_used_protocol(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "N/A"
        if "common_protocol" in dataframe.columns:
            protocol_mode = dataframe["common_protocol"].dropna()
            if not protocol_mode.empty:
                return str(protocol_mode.mode().iloc[0])
        if "proto" in dataframe.columns:
            protocol_mode = dataframe["proto"].dropna()
            if not protocol_mode.empty:
                return str(protocol_mode.mode().iloc[0])
        return "N/A"

    def _compute_network_security_level(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "Unknown"
        threat_density = self._compute_threat_density(dataframe)
        if threat_density > 60:
            return "High Risk"
        if threat_density > 30:
            return "Elevated Risk"
        if threat_density > 10:
            return "Moderate Risk"
        return "Normal"

    def _compute_threat_density(self, dataframe: pd.DataFrame) -> float:
        if dataframe.empty:
            return 0.0
        suspicious_count = len(dataframe[dataframe["detection_status"] == "SUSPICIOUS"])
        return round(suspicious_count / len(dataframe) * 100.0, 2)

    def _generate_findings(self, dataframe: pd.DataFrame) -> list[str]:
        findings: list[str] = []
        active = self._most_active_source(dataframe)
        density = self._compute_threat_density(dataframe)
        high_risk = len(dataframe[dataframe["severity"] == "CRITICAL"])
        anomaly_pct = len(dataframe[dataframe["ml_prediction"] == "SUSPICIOUS"]) / len(dataframe) * 100 if len(dataframe) > 0 else 0

        if active != "N/A":
            findings.append(
                f"Source IP {active} is the most active host and should be prioritized for investigation."
            )

        if density > 25:
            findings.append(
                "A high proportion of suspicious hosts indicates active reconnaissance or scanning campaigns in the network."
            )
        elif density > 10:
            findings.append(
                "Threat density is elevated, suggesting targeted scanning behavior that warrants review."
            )

        if high_risk > 0:
            findings.append(
                f"{high_risk} critical threats were identified, requiring immediate SOC attention."
            )

        if anomaly_pct > 15:
            findings.append(
                "Anomaly detection flagged a significant portion of traffic as suspicious, indicating unusual host activity."
            )

        findings.append(
            "Review firewall policies, IDS tuning, and host-level controls for the highest risk sources."
        )

        return findings

    def export_executive_text(self, dataframe: pd.DataFrame, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self._build_summary(dataframe)

        lines = [
            "SentinelScan Executive Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "\nEXECUTIVE SUMMARY",
            f"Total Source IPs: {len(dataframe)}",
            f"Most Active Source IP: {summary['most_active_source_ip']}",
            f"Highest Risk Host: {summary['highest_risk_host']}",
            f"Most Targeted Port: {summary['most_targeted_port']}",
            f"Most Used Protocol: {summary['most_used_protocol']}",
            f"Network Threat Level: {summary['network_threat_level']}",
            f"Threat Density: {summary['threat_density']}%",
            "\nSECURITY RECOMMENDATIONS",
            "- Block or isolate suspicious hosts",
            "- Review firewall policies and access controls",
            "- Monitor highest risk hosts closely",
            "- Tune IDS/IPS rules for reconnaissance behavior",
        ]

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Executive report saved to: %s", output_path)
        return output_path
