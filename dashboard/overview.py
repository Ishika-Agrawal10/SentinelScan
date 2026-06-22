"""Streamlit dashboard for SentinelScan threat detection and analysis.

This application provides a modern Security Operations Center (SOC) interface
for visualizing network traffic threat analysis, risk scoring, and incident
classification.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.classifier import ThreatClassifier
from src.detector import ThreatDetector
from src.feature_extractor import FeatureExtractor
from src.parser import NetworkTrafficParser
from src.risk_scorer import RiskScorer


logger = logging.getLogger(__name__)


def configure_page() -> None:
    """Configure Streamlit page settings and custom CSS."""

    st.set_page_config(
        page_title="SentinelScan - Port Scan Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    custom_css = """
    <style>
    /* Modern dark theme with cybersecurity aesthetic */
    :root {
        --primary-color: #00d9ff;
        --secondary-color: #ff006e;
        --dark-bg: #0a0e27;
        --card-bg: #1a1f3a;
        --text-primary: #e0e0e0;
        --text-secondary: #a0a0a0;
        --border-color: #00d9ff33;
    }

    body {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }

    .metric-card {
        background: var(--card-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);
        transform: translateY(-2px);
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-value {
        color: var(--primary-color);
        font-size: 32px;
        font-weight: bold;
        margin-top: 8px;
    }

    .section-header {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .severity-critical {
        color: #ff0055;
        font-weight: bold;
    }

    .severity-high {
        color: #ff6b35;
        font-weight: bold;
    }

    .severity-medium {
        color: #ffd60a;
        font-weight: bold;
    }

    .severity-low {
        color: #06d6a0;
        font-weight: bold;
    }

    .detection-suspicious {
        color: #ff006e;
        font-weight: bold;
    }

    .detection-monitor {
        color: #ff9500;
        font-weight: bold;
    }

    .detection-normal {
        color: #06d6a0;
        font-weight: bold;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg);
    }

    /* Improve spacing */
    .stMetric {
        background-color: transparent;
    }

    /* Table styling */
    [data-testid="dataframe"] {
        font-size: 13px;
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: var(--dark-bg);
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 16px rgba(255, 0, 110, 0.4);
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: var(--secondary-color);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }

    /* Tabs styling */
    [data-testid="stTabs"] [role="tablist"] button {
        color: var(--text-secondary);
        border-radius: 8px;
    }

    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 3px solid var(--primary-color);
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def render_sidebar() -> pd.DataFrame | None:
    """Render the sidebar with upload, project info, and about sections."""

    with st.sidebar:
        st.markdown("### 🛡️ SentinelScan")
        st.markdown(
            "**Port Scan Detection Engine**  \n"
            "Advanced threat detection for data center network traffic."
        )

        st.divider()

        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader(
            "Select a network traffic CSV file",
            type="csv",
            help="Upload raw network traffic logs for analysis",
        )

        if uploaded_file is not None:
            try:
                dataframe = pd.read_csv(uploaded_file)
                st.success(f"✓ Loaded {len(dataframe)} records")
                return dataframe
            except Exception as exc:
                st.error(f"Error loading file: {exc}")
                return None

        st.divider()

        st.markdown("### ℹ️ About Project")
        st.markdown(
            """
            **SentinelScan** is a cybersecurity threat detection system that:
            - Analyzes network traffic patterns
            - Identifies suspicious port-scanning behavior
            - Calculates normalized risk scores (0-100)
            - Classifies threats by severity level
            - Enables rapid incident response

            **Key Metrics:**
            - Destination diversity detection
            - Port diversity analysis
            - Connection volume tracking
            - Risk scoring and severity classification
            """
        )

        st.markdown("---")
        st.markdown("*Built with Streamlit + Pandas + Plotly*")

    return None


def process_pipeline(raw_dataframe: pd.DataFrame) -> pd.DataFrame | None:
    """Run the full backend pipeline on uploaded data."""

    try:
        parser = NetworkTrafficParser()
        extractor = FeatureExtractor()
        detector = ThreatDetector()
        scorer = RiskScorer()
        classifier = ThreatClassifier()

        validated_data = parser.validate_columns(raw_dataframe)
        features = extractor.extract_features(validated_data)
        detections = detector.detect_suspicious_activity(features)
        scored = scorer.calculate_risk_score(detections)
        final_results = classifier.classify_severity(scored)

        return final_results

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return None


def render_kpi_cards(dataframe: pd.DataFrame) -> None:
    """Render KPI metric cards at the top of the dashboard."""

    col1, col2, col3, col4 = st.columns(4)

    total_ips = len(dataframe)
    suspicious_ips = len(dataframe[dataframe["detection_status"] == "SUSPICIOUS"]) if "detection_status" in dataframe.columns else 0
    critical_threats = len(dataframe[dataframe["severity"] == "CRITICAL"]) if "severity" in dataframe.columns else 0
    avg_risk = dataframe["risk_score"].mean() if "risk_score" in dataframe.columns and not dataframe["risk_score"].dropna().empty else 0.0

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Source IPs</div>
                <div class="metric-value">{total_ips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Suspicious IPs</div>
                <div class="metric-value" style="color: #ff006e;">{suspicious_ips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Critical Threats</div>
                <div class="metric-value" style="color: #ff0055;">{critical_threats}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Risk Score</div>
                <div class="metric-value">{avg_risk:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_visualizations(dataframe: pd.DataFrame) -> None:
    """Render threat analysis visualizations."""

    if dataframe.empty:
        st.warning("No data available for visualizations.")
        return

    st.markdown("<h2 class='section-header'>📊 Threat Analysis Visualizations</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Severity Distribution")
        if "severity" not in dataframe.columns or dataframe["severity"].dropna().empty:
            st.info("Severity distribution is not available.")
        else:
            severity_counts = dataframe["severity"].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color_discrete_map={
                    "LOW": "#06d6a0",
                    "MEDIUM": "#ffd60a",
                    "HIGH": "#ff6b35",
                    "CRITICAL": "#ff0055",
                },
                title="Threats by Severity Level",
            )
            fig_severity.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1f172a",
                plot_bgcolor="#1f172a",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            st.plotly_chart(fig_severity, use_container_width=True)

    with col2:
        st.markdown("#### Detection Status")
        if "detection_status" not in dataframe.columns or dataframe["detection_status"].dropna().empty:
            st.info("Detection status distribution is not available.")
        else:
            status_counts = dataframe["detection_status"].value_counts()
            fig_status = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                color=status_counts.index,
                color_discrete_map={
                    "NORMAL": "#06d6a0",
                    "MONITOR": "#ff9500",
                    "SUSPICIOUS": "#ff006e",
                },
                title="Threats by Detection Status",
                labels={"x": "Detection Status", "y": "Count"},
            )
            fig_status.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1f172a",
                plot_bgcolor="#1f172a",
                font=dict(color="#e0e0e0"),
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_status, use_container_width=True)

    st.markdown("#### Risk Score Distribution")
    if "risk_score" not in dataframe.columns or dataframe["risk_score"].dropna().empty:
        st.info("Risk score distribution is not available.")
    else:
        fig_histogram = px.histogram(
            dataframe,
            x="risk_score",
            nbins=30,
            title="Risk Score Distribution Across All Sources",
            labels={"risk_score": "Risk Score", "count": "Number of Sources"},
            color_discrete_sequence=["#00d9ff"],
        )
        fig_histogram.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1f172a",
            plot_bgcolor="#1f172a",
            font=dict(color="#e0e0e0"),
            height=400,
        )
        st.plotly_chart(fig_histogram, use_container_width=True)


def render_data_tables(dataframe: pd.DataFrame) -> None:
    """Render detailed data tables."""

    st.markdown("<h2 class='section-header'>📋 Detailed Analysis</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔴 Top 10 High-Risk IPs", "📊 Complete Analysis"])

    with tab1:
        if "risk_score" not in dataframe.columns:
            st.info("Top risk IPs cannot be computed without risk_score.")
        else:
            available_top_columns = [col for col in ["srcip", "total_connections", "unique_destinations", "unique_ports", "risk_score", "severity"] if col in dataframe.columns]
            top_10 = dataframe.nlargest(10, "risk_score")[available_top_columns].reset_index(drop=True)
            top_10.index = top_10.index + 1
            st.dataframe(
                top_10,
                use_container_width=True,
                height=400,
            )

    with tab2:
        display_columns = [
            "srcip",
            "total_connections",
            "unique_destinations",
            "unique_ports",
            "detection_status",
            "risk_score",
            "severity",
        ]
        available_columns = [col for col in display_columns if col in dataframe.columns]
        if available_columns:
            st.dataframe(
                dataframe[available_columns].reset_index(drop=True),
                use_container_width=True,
                height=500,
            )
        else:
            st.warning("No displayable columns are available for the current dataset.")


def render_download_section(dataframe: pd.DataFrame) -> None:
    """Render report download section."""

    st.markdown("<h2 class='section-header'>💾 Export Results</h2>", unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Analysis CSV",
        data=csv_data,
        file_name="sentinelscan_analysis.csv",
        mime="text/csv",
        help="Download the complete analysis results as CSV",
    )

    report_path = Path("reports") / "final_analysis.csv"
    if report_path.exists():
        with open(report_path, "rb") as file:
            st.download_button(
                label="📥 Download Latest Report",
                data=file,
                file_name="final_analysis.csv",
                mime="text/csv",
                help="Download the latest analysis report",
            )


def main() -> None:
    """Main Streamlit application."""

    configure_page()

    st.markdown("# 🛡️ SentinelScan - Port Scan Detection Dashboard")
    st.markdown(
        "Advanced threat detection and analysis for data center network traffic. "
        "Identify suspicious port-scanning behavior with AI-driven risk scoring and severity classification."
    )

    st.divider()

    uploaded_dataframe = render_sidebar()

    if uploaded_dataframe is not None:
        st.info("🔄 Processing network traffic data through the detection pipeline...")
        results = process_pipeline(uploaded_dataframe)

        if results is not None:
            st.success("✓ Analysis complete!")

            render_kpi_cards(results)

            st.divider()

            render_visualizations(results)

            st.divider()

            render_data_tables(results)

            st.divider()

            render_download_section(results)
    else:
        st.info(
            "👈 **Upload a CSV file** in the sidebar to begin network traffic analysis. "
            "The file should contain columns: srcip, dstip, sport, dsport, proto"
        )

        with st.expander("📋 Expected CSV Format"):
            st.markdown(
                """
                Your CSV file should include these columns:
                - `srcip`: Source IP address
                - `dstip`: Destination IP address
                - `sport`: Source port number
                - `dsport`: Destination port number
                - `proto`: Protocol (TCP, UDP, etc.)

                **Example:**
                | srcip | dstip | sport | dsport | proto |
                |-------|-------|-------|--------|-------|
                | 192.168.1.10 | 10.0.0.1 | 54321 | 22 | TCP |
                | 192.168.1.10 | 10.0.0.2 | 54322 | 22 | TCP |
                """
            )


if __name__ == "__main__":
    main()
