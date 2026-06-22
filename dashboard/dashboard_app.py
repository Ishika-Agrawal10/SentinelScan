"""SentinelScan Professional Cybersecurity Analytics Dashboard.

Enterprise-grade Security Operations Center (SOC) interface for network
threat detection, risk visualization, and executive reporting.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).parent.parent
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


def _safe_series(df: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float if isinstance(default, (int, float)) else object)
    series = df[column]
    if isinstance(default, (int, float)):
        return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)
    return series.fillna(default)


def _safe_scalar(df: pd.DataFrame, column: str, default=None):
    if column not in df.columns or df[column].empty:
        return default
    value = df[column].iloc[0]
    return default if pd.isna(value) else value


def _safe_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=int)
    return df[column].value_counts()


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def configure_page() -> None:
    """Configure Streamlit page with professional enterprise styling."""
    st.set_page_config(
        page_title="SentinelScan Cybersecurity Analytics",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown('''
    <style>
    :root {
        --bg-color: #0f172a;
        --surface: #111827;
        --surface-soft: #1f2937;
        --primary: #3b82f6;
        --secondary: #60a5fa;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --border: #334155;
    }

    html, body, .main, .block-container {
        background-color: var(--bg-color);
        color: var(--text);
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    .metric-box, .insight-card, .recommendation-card, .stSidebar, .stButton > button, .stDownloadButton > button {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    .metric-box {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.24);
    }

    .section-header {
        color: var(--text);
        font-size: 22px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 3px solid var(--primary);
    }

    .metric-label {
        font-size: 12px;
        color: var(--muted);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        color: var(--primary);
        font-weight: 700;
        margin-top: 6px;
    }

    .insight-card {
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .insight-title {
        color: var(--secondary);
        font-weight: 700;
        margin-bottom: 8px;
    }

    .insight-text {
        color: var(--text);
        font-size: 14px;
        line-height: 1.6;
    }

    .recommendation-card {
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .recommendation-title {
        color: var(--warning);
        font-weight: 700;
        margin-bottom: 8px;
    }

    .recommendation-text {
        color: var(--text);
        font-size: 14px;
        line-height: 1.6;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        background-color: var(--primary) !important;
        color: var(--text) !important;
        border: 1px solid transparent !important;
        font-weight: 700;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        opacity: 0.95;
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface-soft) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stTabs"] [role="tablist"] button {
        color: var(--text) !important;
    }

    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        border-bottom: 3px solid var(--primary) !important;
    }

    .stMarkdown p, .stMarkdown div {
        color: var(--text) !important;
    }

    [data-testid="dataframe"] {
        font-size: 13px;
    }
    </style>
    ''', unsafe_allow_html=True)


def render_header() -> None:
    """Render main dashboard header."""
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown("# SentinelScan Security Analytics")
        st.markdown("**Professional Cybersecurity Analytics Platform**")
    with col2:
        st.markdown(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.divider()


def render_sidebar() -> pd.DataFrame | None:
    """Render sidebar with upload and project info."""
    with st.sidebar:
        st.markdown("## SentinelScan")
        st.markdown("Advanced threat detection for enterprise networks")
        st.divider()

        st.markdown("### Upload Traffic Data")
        uploaded_file = st.file_uploader(
            "Select CSV file",
            type="csv",
            help="Upload network traffic logs (srcip, dstip, sport, dsport, proto)",
        )

        if uploaded_file is not None:
            try:
                return pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None

        st.divider()
        st.markdown("### About Platform")
        st.markdown("""
        **SentinelScan** provides:
        - Real-time threat detection
        - Network traffic analysis
        - Automated risk assessment
        - Executive reporting
        - Incident investigation
        """)

        st.divider()
        st.markdown("*Enterprise Security Analytics*")

    return None


def process_pipeline(df: pd.DataFrame) -> pd.DataFrame | None:
    """Execute threat detection pipeline."""
    try:
        parser = NetworkTrafficParser()
        extractor = FeatureExtractor()
        detector = ThreatDetector()
        scorer = RiskScorer()
        classifier = ThreatClassifier()
        ml_detector = MLDetector()
        attack_classifier = AttackClassifier()
        intelligence = ThreatIntelligence()

        validated = parser.validate_columns(df)
        features = extractor.extract_features(validated)
        detections = detector.detect_suspicious_activity(features)
        scored = scorer.calculate_risk_score(detections)
        classified = classifier.classify_severity(scored)
        ml_result = ml_detector.analyze(classified)
        attack_result = attack_classifier.classify(ml_result)
        final_results = intelligence.analyze(attack_result)

        return final_results
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return None


def render_executive_overview(df: pd.DataFrame) -> None:
    """Render executive overview with key metrics."""
    st.markdown('<h2 class="section-header">Executive Overview</h2>', unsafe_allow_html=True)

    total_records = int(_safe_series(df, "total_connections", 0.0).sum())
    total_ips = len(df)
    suspicious = int(_safe_series(df, "detection_status", 0.0).eq("SUSPICIOUS").sum()) if "detection_status" in df.columns else 0
    critical = int(_safe_series(df, "severity", 0.0).eq("CRITICAL").sum()) if "severity" in df.columns else 0
    highest_risk = float(_safe_series(df, "risk_score", 0.0).max())
    avg_risk = float(_safe_series(df, "risk_score", 0.0).mean())
    threat_density = float(_safe_scalar(df, "threat_density", (suspicious / total_ips * 100 if total_ips else 0.0)))
    network_security_level = str(_safe_scalar(df, "network_threat_level", "Normal"))
    anomaly_count = int(_safe_series(df, "ml_result", 0.0).eq("ANOMALY").sum()) if "ml_result" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col5, col6 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Traffic Records</div>
            <div class="metric-value">{total_records}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Source IPs</div>
            <div class="metric-value">{total_ips}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Suspicious Hosts</div>
            <div class="metric-value" style="color: #ef4444;">{suspicious}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Critical Threats</div>
            <div class="metric-value" style="color: #f97316;">{critical}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Network Security Level</div>
            <div class="metric-value">{network_security_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Threat Density</div>
            <div class="metric-value">{threat_density:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    col7, col8 = st.columns(2)

    with col7:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Average Risk Score</div>
            <div class="metric-value">{avg_risk:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">ML Anomalies</div>
            <div class="metric-value" style="color: #60a5fa;">{anomaly_count}</div>
        </div>
        """, unsafe_allow_html=True)


def render_threat_intelligence(df: pd.DataFrame) -> None:
    """Render threat intelligence center with detailed metrics."""
    if df.empty:
        st.warning("No data available for threat intelligence.")
        return

    st.markdown('<h2 class="section-header">Threat Intelligence Center</h2>', unsafe_allow_html=True)

    most_active = str(_safe_scalar(df, "most_active_source_ip", "N/A"))
    highest_risk = str(_safe_scalar(df, "highest_risk_host", "N/A"))
    targeted_port = str(_safe_scalar(df, "most_targeted_port", "N/A"))
    common_proto = str(_safe_scalar(df, "most_used_protocol", "N/A"))
    attack_trend = _safe_value_counts(df, "attack_type").reset_index()
    if not attack_trend.empty:
        attack_trend.columns = ["attack_type", "count"]
        common_attack_type = attack_trend.iloc[0, 0]
    else:
        common_attack_type = "N/A"
    network_level = str(_safe_scalar(df, "network_threat_level", "Normal"))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Most Active Source</div>
            <div class="metric-value">{most_active}</div>
            <div class="metric-label">Connections Leader</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Highest Risk Host</div>
            <div class="metric-value">{highest_risk}</div>
            <div class="metric-label">Top risk source identified</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Network Threat Level</div>
            <div class="metric-value">{network_level}</div>
            <div class="metric-label">Operational risk posture</div>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Most Targeted Port</div>
            <div class="metric-value">{targeted_port}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Most Common Protocol</div>
            <div class="metric-value">{common_proto}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Most Common Attack Type</div>
            <div class="metric-value">{common_attack_type}</div>
        </div>
        """, unsafe_allow_html=True)

    if not attack_trend.empty and attack_trend.shape[1] >= 2:
        x_column = attack_trend.columns[0]
        y_column = attack_trend.columns[1]
        fig = px.bar(
            attack_trend,
            x=x_column,
            y=y_column,
            labels={x_column: "Attack Type", y_column: "Count"},
            title="Attack Type Trend",
            color=x_column,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig.update_layout(template="plotly_dark", height=380, showlegend=False, plot_bgcolor="#121826", paper_bgcolor="#121826")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Attack type trend data is not available.")


def render_security_gauge(df: pd.DataFrame) -> None:
    """Render security risk gauge visualization."""
    if df.empty:
        st.warning("No data available for security risk assessment.")
        return

    st.markdown('<h2 class="section-header">Security Risk Assessment</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        avg_risk = float(_safe_series(df, "risk_score", 0.0).mean())
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_risk,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Network Risk Score"},
            delta={"reference": 50, "suffix": "vs target"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0052a3"},
                "steps": [
                    {"range": [0, 25], "color": "#dcfce7"},
                    {"range": [25, 50], "color": "#fef3c7"},
                    {"range": [50, 75], "color": "#fed7aa"},
                    {"range": [75, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        ))
        fig_gauge.update_layout(template="plotly_white", height=350, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        low = int(_safe_series(df, "severity", 0.0).eq("LOW").sum()) if "severity" in df.columns else 0
        medium = int(_safe_series(df, "severity", 0.0).eq("MEDIUM").sum()) if "severity" in df.columns else 0
        high = int(_safe_series(df, "severity", 0.0).eq("HIGH").sum()) if "severity" in df.columns else 0
        critical = int(_safe_series(df, "severity", 0.0).eq("CRITICAL").sum()) if "severity" in df.columns else 0

        st.markdown("**Severity Breakdown:**")
        st.markdown(f"""
        <div class="metric-box" style="text-align: left; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>LOW Risk</span>
                <span style="font-weight: 600; color: var(--primary);">{low}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>MEDIUM Risk</span>
                <span style="font-weight: 600; color: var(--warning);">{medium}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>HIGH Risk</span>
                <span style="font-weight: 600; color: #f97316;">{high}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>CRITICAL Risk</span>
                <span style="font-weight: 600; color: var(--danger);">{critical}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_visualizations(df: pd.DataFrame) -> None:
    """Render comprehensive threat visualizations."""
    if df.empty:
        st.warning("No data available for threat visualizations.")
        return

    st.markdown('<h2 class="section-header">Threat Visualizations</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Severity",
        "Detection Status",
        "Risk Distribution",
        "Attack Types",
        "ML Anomaly",
        "Protocol Analysis",
    ])

    with tab1:
        severity_counts = _safe_value_counts(df, "severity")
        if severity_counts.empty:
            st.info("Severity distribution is not available.")
        else:
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color_discrete_map={
                    "CRITICAL": "#dc2626",
                    "HIGH": "#f97316",
                    "MEDIUM": "#eab308",
                    "LOW": "#16a34a",
                },
                title="Threat Severity Distribution",
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        status_counts = _safe_value_counts(df, "detection_status")
        if status_counts.empty:
            st.info("Detection status distribution is not available.")
        else:
            fig = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                color=status_counts.index,
                color_discrete_map={
                    "SUSPICIOUS": "#ef4444",
                    "MONITOR": "#f59e0b",
                    "NORMAL": "#22c55e",
                },
                title="Detection Status Distribution",
                labels={"x": "Status", "y": "Count"},
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "risk_score" not in df.columns or df["risk_score"].dropna().empty:
            st.info("Risk score distribution is not available.")
        else:
            fig = px.histogram(
                df,
                x="risk_score",
                nbins=30,
                title="Risk Score Distribution",
                labels={"risk_score": "Risk Score", "count": "Frequency"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        attack_counts = _safe_value_counts(df, "attack_type")
        if attack_counts.empty:
            st.info("Attack type distribution is not available.")
        else:
            fig = px.bar(
                x=attack_counts.index,
                y=attack_counts.values,
                title="Attack Type Distribution",
                labels={"x": "Attack Type", "y": "Count"},
                color=attack_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        anomaly_counts = _safe_value_counts(df, "ml_result")
        if anomaly_counts.empty:
            st.info("ML anomaly metrics are not available.")
        else:
            fig = px.pie(
                values=anomaly_counts.values,
                names=anomaly_counts.index,
                title="ML Anomaly Detection Results",
                color_discrete_map={"ANOMALY": "#ef4444", "NORMAL": "#22c55e"},
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a")
            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        protocol_counts = _safe_value_counts(df, "common_protocol")
        if protocol_counts.empty:
            st.info("Protocol analytics are not available.")
        else:
            fig = px.bar(
                x=protocol_counts.index,
                y=protocol_counts.values,
                title="Protocol Usage Distribution",
                labels={"x": "Protocol", "y": "Count"},
                color=protocol_counts.index,
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_layout(template="plotly_dark", height=450, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_threat_explorer(df: pd.DataFrame) -> None:
    """Render interactive threat explorer with search and filters."""
    if df.empty:
        st.warning("No data available for threat explorer.")
        return

    st.markdown('<h2 class="section-header">Threat Explorer</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        search_ip = st.text_input("Search Source IP", placeholder="e.g., 192.168.1.10")
    with col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"],
        )
    with col3:
        status_filter = st.multiselect(
            "Filter by Detection Status",
            options=["NORMAL", "MONITOR", "SUSPICIOUS"],
            default=["SUSPICIOUS"],
        )
    with col4:
        attack_filter = st.multiselect(
            "Filter by Attack Type",
            options=sorted(df["attack_type"].unique()) if "attack_type" in df.columns else [],
            default=[],
        )

    prediction_filter = st.selectbox(
        "ML Prediction",
        options=["ALL", "NORMAL", "SUSPICIOUS"],
        index=0,
    )

    filtered_df = df.copy()
    if search_ip and "srcip" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["srcip"].str.contains(search_ip, case=False, na=False)]
    if severity_filter and "severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]
    if status_filter and "detection_status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["detection_status"].isin(status_filter)]
    if attack_filter and "attack_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["attack_type"].isin(attack_filter)]
    if prediction_filter != "ALL" and "ml_prediction" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ml_prediction"] == prediction_filter]

    st.markdown(f"**Found {len(filtered_df)} matching records**")

    if len(filtered_df) > 0:
        display_cols = [
            "srcip",
            "total_connections",
            "unique_destinations",
            "unique_ports",
            "detection_status",
            "ml_prediction",
            "attack_type",
            "risk_score",
            "severity",
            "final_threat_assessment",
        ]
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        if available_cols:
            st.dataframe(
                filtered_df[available_cols].reset_index(drop=True),
                use_container_width=True,
                height=420,
            )
        else:
            st.warning("No displayable columns are available for the current dataset.")
    else:
        st.info("No threats match the current filters.")


def generate_ai_insights(df: pd.DataFrame) -> list[str]:
    """Generate AI-powered security insights."""
    if df.empty:
        return ["No data available for AI security insights."]

    insights = []

    avg_dest = df["unique_destinations"].mean() if "unique_destinations" in df.columns else 0
    if avg_dest > 30:
        insights.append("High destination diversity detected across network traffic - potential reconnaissance activity.")
    elif avg_dest > 15:
        insights.append("Moderate destination diversity observed - possible network exploration.")

    avg_ports = df["unique_ports"].mean() if "unique_ports" in df.columns else 0
    if avg_ports > 20:
        insights.append("Significant port diversity detected - strong indicators of systematic port scanning.")

    anomaly_pct = (len(df[df["ml_prediction"] == "SUSPICIOUS"]) / len(df) * 100) if len(df) > 0 and "ml_prediction" in df.columns else 0
    if anomaly_pct > 10:
        insights.append(f"{anomaly_pct:.1f}% of sources were flagged by ML anomaly detection - investigate unusual behavior.")

    suspicious_pct = (len(df[df["detection_status"] == "SUSPICIOUS"]) / len(df) * 100) if len(df) > 0 and "detection_status" in df.columns else 0
    if suspicious_pct > 20:
        insights.append(f"{suspicious_pct:.1f}% of hosts exhibit suspicious behavior - elevated threat level.")
    elif suspicious_pct > 5:
        insights.append(f"{suspicious_pct:.1f}% suspicious hosts detected - recommend investigation.")

    critical_count = len(df[df["severity"] == "CRITICAL"]) if "severity" in df.columns else 0
    if critical_count > 5:
        insights.append(f"{critical_count} critical threats identified - immediate action required.")
    elif critical_count > 0:
        insights.append(f"{critical_count} critical threat(s) detected - prioritize investigation.")

    if "attack_type" in df.columns and len(df[df["attack_type"] != "Normal Traffic"]) > 0:
        insights.append("Attack pattern classification identifies multiple non-normal traffic types; prioritize host investigation.")

    avg_connections = df["total_connections"].mean() if "total_connections" in df.columns else 0
    if avg_connections > 50:
        insights.append("High connection volume detected - potential DDoS or lateral movement.")

    return insights if insights else ["Network traffic appears normal with no significant anomalies detected."]


def render_ai_insights(df: pd.DataFrame) -> None:
    """Render AI security insights."""
    st.markdown('<h2 class="section-header">AI Security Insights</h2>', unsafe_allow_html=True)

    insights = generate_ai_insights(df)

    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Security Alert</div>
            <div class="insight-text">{insight}</div>
        </div>
        """, unsafe_allow_html=True)


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    """Generate actionable security recommendations."""
    if df.empty:
        return ["No data available to generate recommendations."]

    recommendations = []

    suspicious_count = int(_safe_series(df, "detection_status", "").eq("SUSPICIOUS").sum())
    if suspicious_count > 0:
        recommendations.append("Isolate and investigate all hosts marked as SUSPICIOUS for potential compromise.")

    critical_count = len(df[df["severity"] == "CRITICAL"]) if "severity" in df.columns else 0
    if critical_count > 0 and "risk_score" in df.columns:
        top_critical = df[df["severity"] == "CRITICAL"].nlargest(3, "risk_score")
        ips = ", ".join(top_critical["srcip"].tolist()) if "srcip" in top_critical.columns else "N/A"
        recommendations.append(f"Implement immediate network isolation for critical hosts: {ips}")
    elif critical_count > 0:
        recommendations.append("Implement immediate network isolation for critical hosts.")

    avg_risk = df["risk_score"].mean() if "risk_score" in df.columns else 0
    if avg_risk > 50:
        recommendations.append("Deploy additional monitoring and threat detection capabilities for high-risk sources.")

    port_scanners = len(df[df["unique_ports"] > 20]) if "unique_ports" in df.columns else 0
    if port_scanners > 0:
        recommendations.append(f"Enable port scanning detection and rate-limiting for {port_scanners} identified scanner(s).")

    recommendations.append("Review firewall rules and implement stricter egress filtering policies.")
    recommendations.append("Conduct threat hunt on identified suspicious sources to determine root cause.")

    return recommendations


def render_recommendations(df: pd.DataFrame) -> None:
    """Render security recommendations."""
    st.markdown('<h2 class="section-header">Security Recommendations</h2>', unsafe_allow_html=True)

    recommendations = generate_recommendations(df)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="recommendation-title">Recommendation #{i}</div>
            <div class="recommendation-text">{rec}</div>
        </div>
        """, unsafe_allow_html=True)


def render_executive_report(df: pd.DataFrame) -> io.StringIO:
    """Generate executive summary report."""
    report = io.StringIO()
    report.write("SentinelScan Executive Summary Report\n")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write("=" * 60 + "\n\n")

    report.write("EXECUTIVE SUMMARY\n")
    report.write("-" * 60 + "\n")
    total = len(df)
    suspicious = int(_safe_series(df, "detection_status", "").eq("SUSPICIOUS").sum()) if "detection_status" in df.columns else 0
    critical = int(_safe_series(df, "severity", "").eq("CRITICAL").sum()) if "severity" in df.columns else 0
    avg_risk = float(_safe_series(df, "risk_score", 0.0).mean())
    network_level = str(_safe_scalar(df, "network_threat_level", "Normal"))
    threat_density = float(_safe_scalar(df, "threat_density", (suspicious / total * 100 if total else 0.0)))
    top_attack_type = "N/A"
    if "attack_type" in df.columns:
        attack_mode = df["attack_type"].mode()
        top_attack_type = str(attack_mode.iloc[0]) if not attack_mode.empty else "N/A"
    most_targeted_port = str(_safe_scalar(df, "most_targeted_port", "N/A"))
    most_used_protocol = str(_safe_scalar(df, "most_used_protocol", "N/A"))

    report.write(f"Total Source IPs Analyzed: {total}\n")
    report.write(f"Suspicious Hosts Detected: {suspicious} ({suspicious/total*100:.1f}%)\n")
    report.write(f"Critical Threats: {critical}\n")
    report.write(f"Average Risk Score: {avg_risk:.1f}\n")
    report.write(f"Network Security Level: {network_level}\n")
    report.write(f"Threat Density: {threat_density:.1f}%\n")
    report.write(f"Top Attack Type: {top_attack_type}\n")
    report.write(f"Most Targeted Port: {most_targeted_port}\n")
    report.write(f"Most Used Protocol: {most_used_protocol}\n\n")

    report.write("THREAT DISTRIBUTION\n")
    report.write("-" * 60 + "\n")
    severity_counts = _safe_value_counts(df, "severity")
    if severity_counts.empty:
        report.write("Severity distribution is not available.\n")
    else:
        for severity, count in severity_counts.items():
            report.write(f"{severity}: {count}\n")
    report.write("\n")

    report.write("TOP 10 HIGHEST RISK SOURCES\n")
    report.write("-" * 60 + "\n")
    if "risk_score" in df.columns:
        available_columns = [col for col in ["srcip", "risk_score", "severity", "attack_type", "final_threat_assessment"] if col in df.columns]
        if available_columns:
            top_10 = df.nlargest(10, "risk_score")[available_columns]
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                srcip = row["srcip"] if "srcip" in row.index else "N/A"
                risk = float(row["risk_score"]) if "risk_score" in row.index else 0.0
                severity = row["severity"] if "severity" in row.index else "N/A"
                attack = row["attack_type"] if "attack_type" in row.index else "N/A"
                assessment = row["final_threat_assessment"] if "final_threat_assessment" in row.index else "N/A"
                report.write(
                    f"{idx}. {srcip} - Score: {risk:.1f} ({severity}) | Attack: {attack} | Assessment: {assessment}\n"
                )
        else:
            report.write("No top risk source columns are available.\n")
    else:
        report.write("Risk score information is not available for top source summary.\n")

    report.seek(0)
    return report


def render_executive_panel(df: pd.DataFrame) -> None:
    """Render executive report panel."""
    if df.empty:
        st.warning("No data available for executive report panel.")
        return

    st.markdown('<h2 class="section-header">Executive Report Panel</h2>', unsafe_allow_html=True)

    report = render_executive_report(df)
    report_text = report.getvalue()

    with st.expander("View Executive Summary", expanded=False):
        st.text(report_text)


def render_download_center(df: pd.DataFrame) -> None:
    """Render download center with multiple report formats."""
    if df.empty:
        st.warning("No data available for download.")
        return

    st.markdown('<h2 class="section-header">Download Center</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Full Analysis Report (CSV)",
            data=csv_data,
            file_name="sentinelscan_full_analysis.csv",
            mime="text/csv",
        )

    with col2:
        exec_report = render_executive_report(df)
        st.download_button(
            label="Executive Summary (TXT)",
            data=exec_report.getvalue(),
            file_name="sentinelscan_executive_summary.txt",
            mime="text/plain",
        )


def main() -> None:
    """Main application entry point."""
    configure_page()
    render_header()

    uploaded_df = render_sidebar()

    if uploaded_df is not None:
        with st.spinner("Processing threat detection pipeline..."):
            results = process_pipeline(uploaded_df)

        if results is not None:
            st.success("Analysis complete!")

            render_executive_overview(results)
            st.divider()

            render_threat_intelligence(results)
            st.divider()

            render_security_gauge(results)
            st.divider()

            render_visualizations(results)
            st.divider()

            render_threat_explorer(results)
            st.divider()

            render_ai_insights(results)
            st.divider()

            render_recommendations(results)
            st.divider()

            render_executive_panel(results)
            st.divider()

            render_download_center(results)
    else:
        st.info(
            "Upload a CSV file in the sidebar to begin threat analysis. "
            "Expected columns: srcip, dstip, sport, dsport, proto"
        )
        with st.expander("Expected CSV Format"):
            st.markdown("""
            | srcip | dstip | sport | dsport | proto |
            |-------|-------|-------|--------|-------|
            | 192.168.1.10 | 10.0.0.1 | 54321 | 22 | TCP |
            | 192.168.1.10 | 10.0.0.2 | 54322 | 22 | TCP |
            """)


if __name__ == "__main__":
    main()
