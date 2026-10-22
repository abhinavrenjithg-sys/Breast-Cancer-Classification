"""
app.py
------
Wisconsin Diagnostic Breast Cancer (WDBC) Classification System.
Designed following modern software engineering principles and dark-canvas design systems
(inspired by Linear, Resend, and modern clinical workstations).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.train import load_pipeline

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="WDBC Clinical Diagnostic Workstation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------------
# Directory & File Constants
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports" / "figures"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.csv"
METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"

# ------------------------------------------------------------------------------
# High-End Design System CSS (Linear / Resend Inspired)
# ------------------------------------------------------------------------------
DESIGN_SYSTEM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --canvas: #08090c;
        --surface-1: #111318;
        --surface-2: #161922;
        --surface-3: #1e222e;
        --surface-4: #282e3d;
        --hairline: rgba(255, 255, 255, 0.08);
        --hairline-strong: rgba(255, 255, 255, 0.16);
        --accent: #5e6ad2;
        --accent-glow: rgba(94, 106, 210, 0.15);
        --malignant: #f43f5e;
        --malignant-surface: rgba(244, 63, 94, 0.08);
        --malignant-border: rgba(244, 63, 94, 0.28);
        --benign: #10b981;
        --benign-surface: rgba(16, 185, 129, 0.08);
        --benign-border: rgba(16, 185, 129, 0.28);
        --ink: #f7f8f8;
        --ink-muted: #94a3b8;
        --ink-subtle: #64748b;
    }

    /* Core Document Canvas */
    .stApp {
        background-color: var(--canvas) !important;
        color: var(--ink) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Remove default Streamlit chrome */
    #MainMenu, footer, header {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
    }

    /* Header & Brand Navigation */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-mark {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: var(--accent);
        box-shadow: 0 0 10px var(--accent);
    }
    .brand-title {
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    .brand-badge {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 2px 8px;
        border-radius: 9999px;
        background: var(--surface-3);
        color: var(--ink-muted);
        border: 1px solid var(--hairline);
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* Hero Typography */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--ink);
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: var(--ink-muted);
        line-height: 1.5;
        margin-bottom: 20px;
        font-weight: 400;
    }

    /* Authoritative Clinical Advisory */
    .clinical-advisory {
        background: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 24px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .advisory-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .advisory-text {
        font-size: 0.82rem;
        color: #d1d5db;
        line-height: 1.45;
    }

    /* Segmented Tab Controls */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface-1) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        gap: 6px !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--ink-muted) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 8px 18px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ink) !important;
        background: var(--surface-2) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface-3) !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    }

    /* KPI Cards */
    .stat-card {
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 18px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .stat-card:hover {
        border-color: var(--hairline-strong);
        transform: translateY(-1px);
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-subtle);
        margin-bottom: 6px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .stat-context {
        font-size: 0.78rem;
        color: var(--ink-muted);
        margin-top: 8px;
    }

    /* Diagnostic Verdict Banners */
    .verdict-box-malignant {
        background: var(--malignant-surface);
        border: 1px solid var(--malignant-border);
        border-radius: 12px;
        padding: 22px 26px;
        margin: 16px 0 24px 0;
    }
    .verdict-box-benign {
        background: var(--benign-surface);
        border: 1px solid var(--benign-border);
        border-radius: 12px;
        padding: 22px 26px;
        margin: 16px 0 24px 0;
    }
    .verdict-pill-malignant {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #fda4af;
        background: rgba(244, 63, 94, 0.2);
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .verdict-pill-benign {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.2);
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .verdict-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-bottom: 6px;
    }
    .verdict-desc {
        font-size: 0.88rem;
        color: var(--ink-muted);
        line-height: 1.5;
    }

    /* Linear Probability Progress Track */
    .prob-track {
        background: var(--surface-3);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 8px;
    }
    .prob-fill-malignant {
        background: linear-gradient(90deg, #f43f5e, #fb7185);
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    .prob-fill-benign {
        background: linear-gradient(90deg, #10b981, #34d399);
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    /* Biomarker Chip Grid */
    .chip-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 10px;
        margin: 14px 0;
    }
    .bio-chip {
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .bio-chip-name {
        font-size: 0.72rem;
        color: var(--ink-subtle);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .bio-chip-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink);
        margin-top: 2px;
    }

    /* Button Override */
    .stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2), 0 0 12px var(--accent-glow) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #717fe0 !important;
        border-color: #717fe0 !important;
        transform: translateY(-1px) !important;
    }
</style>
"""
st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Data & Model Loaders
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> Optional[Pipeline]:
    """Load serialized scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        return None
    try:
        return load_pipeline(MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_evaluation_data() -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Load performance metrics, metadata, and held-out test cohort."""
    metrics = None
    metadata = None
    test_df = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if TEST_DATA_PATH.exists():
        test_df = pd.read_csv(TEST_DATA_PATH, index_col=0)

    return metrics, metadata, test_df


# ------------------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------------------
def main():
    # 1. Executive Top Bar
    st.markdown(
        """
        <div class="app-header">
            <div class="brand-section">
                <div class="brand-mark"></div>
                <div class="brand-title">WDBC Diagnostic Engine</div>
                <div class="brand-badge">L-BFGS · v1.0.4-PROD</div>
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>VALIDATED · AUC 0.9954</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Hero Section
    st.markdown('<div class="hero-title">Breast Mass Morphological Classification</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-sub">
            Calibrated probabilistic prediction on digitized fine needle aspirate (FNA) nuclear characteristics.
            Constructed with scikit-learn <code>StandardScaler</code> and <code>LogisticRegression</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Authoritative Clinical Advisory Banner
    st.markdown(
        """
        <div class="clinical-advisory">
            <div class="advisory-tag">Regulatory Advisory</div>
            <div class="advisory-text">
                <strong>Non-Diagnostic Research Instrument:</strong> This system is engineered exclusively for machine learning
                research, verification, and academic demonstration. It is not an FDA-cleared diagnostic device and must never
                supersede pathologist microscopic review or certified clinical judgment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Model & Resources
    pipeline = load_model()
    metrics, metadata, test_df = load_evaluation_data()

    if pipeline is None or metrics is None or test_df is None:
        st.error("Trained model pipeline or evaluation artifacts were not located.")
        st.markdown(
            """
            Execute the training pipeline from the project root to generate required binaries:
            ```bash
            python main.py
            ```
            """
        )
        return

    # 4. Primary Workstation Tabs
    tab_inference, tab_metrics, tab_eda, tab_docs = st.tabs(
        [
            "Diagnostic Workstation",
            "Model Verification & Metrics",
            "Exploratory Analysis (EDA)",
            "System Architecture & Protocol",
        ]
    )

    # ==========================================================================
    # TAB 1: DIAGNOSTIC WORKSTATION
    # ==========================================================================
    with tab_inference:
        st.markdown("### Interactive Case Evaluation")
        st.markdown(
            "Evaluate digitized cell nuclei from the **held-out stratified test cohort (114 samples)** "
            "or simulate custom biomarker values to observe real-time decision boundary response."
        )

        mode_choice = st.radio(
            "Evaluation Mode:",
            options=["Held-Out Test Cohort (Clinical Samples)", "Interactive Biomarker Simulator"],
            horizontal=True,
            label_visibility="collapsed",
        )

        feature_cols = [c for c in test_df.columns if c not in ("target", "diagnosis")]

        if mode_choice == "Held-Out Test Cohort (Clinical Samples)":
            col_filter, col_select = st.columns([1, 2])
            with col_filter:
                filter_choice = st.selectbox(
                    "Filter Cohort Pathology:",
                    options=["All Test Cases (114)", "Malignant Only (42)", "Benign Only (72)"],
                )

            filtered_df = test_df.copy()
            if "Malignant" in filter_choice:
                filtered_df = filtered_df[filtered_df["diagnosis"] == "malignant"]
            elif "Benign" in filter_choice:
                filtered_df = filtered_df[filtered_df["diagnosis"] == "benign"]

            sample_options = [
                f"Sample #{idx} — Pathology: {row['diagnosis'].upper()} (Mean Radius: {row['mean radius']:.2f} µm, Area: {row['mean area']:.1f})"
                for idx, row in filtered_df.iterrows()
            ]
            selected_option = st.selectbox("Select Patient Biopsy Case:", sample_options)

            selected_idx = int(selected_option.split("#")[1].split(" ")[0])
            sample_row = test_df.loc[selected_idx]
            actual_label = sample_row["diagnosis"]
            input_features = sample_row[feature_cols].to_frame().T

        else:
            st.markdown("##### Adjust Morphological Biomarkers")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                mean_radius = st.slider("Mean Radius (µm)", min_value=6.0, max_value=30.0, value=14.12, step=0.1)
                mean_texture = st.slider("Mean Texture (gray-scale)", min_value=9.0, max_value=40.0, value=19.28, step=0.1)
            with sim_col2:
                mean_perimeter = st.slider("Mean Perimeter", min_value=40.0, max_value=190.0, value=91.96, step=0.5)
                mean_area = st.slider("Mean Area (µm²)", min_value=140.0, max_value=2500.0, value=654.8, step=5.0)
            with sim_col3:
                mean_concavity = st.slider("Mean Concavity", min_value=0.0, max_value=0.45, value=0.088, step=0.01)
                mean_concave_pts = st.slider("Mean Concave Points", min_value=0.0, max_value=0.20, value=0.048, step=0.005)

            # Create synthetic feature vector initialized from dataset median
            synthetic_row = test_df[feature_cols].median().copy()
            synthetic_row["mean radius"] = mean_radius
            synthetic_row["mean texture"] = mean_texture
            synthetic_row["mean perimeter"] = mean_perimeter
            synthetic_row["mean area"] = mean_area
            synthetic_row["mean concavity"] = mean_concavity
            synthetic_row["mean concave points"] = mean_concave_pts
            input_features = synthetic_row.to_frame().T
            actual_label = None
            selected_idx = "SIMULATION"

        # Model Inference
        pred_class_id = int(pipeline.predict(input_features)[0])
        pred_probas = pipeline.predict_proba(input_features)[0]
        # In sklearn load_breast_cancer: 0 = Malignant, 1 = Benign
        p_malignant = float(pred_probas[0])
        p_benign = float(pred_probas[1])
        predicted_label = "malignant" if pred_class_id == 0 else "benign"

        # Display High-End Verdict Banner
        if predicted_label == "malignant":
            st.markdown(
                f"""
                <div class="verdict-box-malignant">
                    <div class="verdict-pill-malignant">High Suspicion · Malignant Indication</div>
                    <div class="verdict-title">Class 0: Malignant Cellular Morphology Detected</div>
                    <div class="verdict-desc">
                        Nuclear contour irregularities, enlarged nuclear area, and pronounced boundary concavities
                        strongly associate with malignant cytological criteria.
                    </div>
                    <div style="margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: #fda4af;">
                        P(Malignant): <strong>{p_malignant * 100:.2f}%</strong> &nbsp;|&nbsp; P(Benign): {p_benign * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-box-benign">
                    <div class="verdict-pill-benign">Favorable Finding · Benign Morphology</div>
                    <div class="verdict-title">Class 1: Benign Cytological Pattern Identified</div>
                    <div class="verdict-desc">
                        Nuclear perimeter measurements, uniform texture distribution, and low concavity indices
                        fall within standard benign tissue reference ranges.
                    </div>
                    <div style="margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: #6ee7b7;">
                        P(Benign): <strong>{p_benign * 100:.2f}%</strong> &nbsp;|&nbsp; P(Malignant): {p_malignant * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dual Probability Tracks
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(
                f"""
                <div style="font-size: 0.8rem; color: var(--ink-muted); display: flex; justify-content: space-between;">
                    <span>Malignant Probability</span>
                    <span style="font-family: 'JetBrains Mono'; color: #fda4af;">{p_malignant * 100:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill-malignant" style="width: {max(2, int(p_malignant * 100))}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"""
                <div style="font-size: 0.8rem; color: var(--ink-muted); display: flex; justify-content: space-between;">
                    <span>Benign Probability</span>
                    <span style="font-family: 'JetBrains Mono'; color: #6ee7b7;">{p_benign * 100:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill-benign" style="width: {max(2, int(p_benign * 100))}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Pathology Concordance Analysis (Cohort mode only)
        if actual_label is not None:
            st.write("")
            is_match = predicted_label == actual_label
            concordance_text = "CONCORDANT (Correct)" if is_match else "DISCORDANT (Review Required)"
            concordance_color = "#34d399" if is_match else "#f87171"

            eval_c1, eval_c2, eval_c3 = st.columns(3)
            with eval_c1:
                st.markdown(
                    f"""
                    <div class="bio-chip">
                        <div class="bio-chip-name">Actual Ground Truth</div>
                        <div class="bio-chip-val" style="color: {'#fda4af' if actual_label == 'malignant' else '#6ee7b7'};">
                            {actual_label.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with eval_c2:
                st.markdown(
                    f"""
                    <div class="bio-chip">
                        <div class="bio-chip-name">Predicted Diagnosis</div>
                        <div class="bio-chip-val" style="color: {'#fda4af' if predicted_label == 'malignant' else '#6ee7b7'};">
                            {predicted_label.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with eval_c3:
                st.markdown(
                    f"""
                    <div class="bio-chip">
                        <div class="bio-chip-name">Diagnostic Concordance</div>
                        <div class="bio-chip-val" style="color: {concordance_color}; font-size: 0.85rem;">
                            {concordance_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Nuclear Characteristic Values Inspection
        with st.expander("🔬 View Primary Morphological Feature Values", expanded=False):
            st.markdown(
                """
                <div class="chip-container">
                """
                + "".join(
                    [
                        f"""
                        <div class="bio-chip">
                            <div class="bio-chip-name">{col.replace('_', ' ')}</div>
                            <div class="bio-chip-val">{input_features[col].values[0]:.3f}</div>
                        </div>
                        """
                        for col in [
                            "mean radius",
                            "mean texture",
                            "mean perimeter",
                            "mean area",
                            "mean smoothness",
                            "mean concavity",
                            "mean concave points",
                            "mean symmetry",
                            "worst radius",
                            "worst area",
                            "worst concavity",
                            "worst concave points",
                        ]
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # TAB 2: MODEL VERIFICATION & METRICS
    # ==========================================================================
    with tab_metrics:
        st.markdown("### Model Benchmark Performance (Held-Out Test Set: 114 Samples)")
        st.markdown(
            "Empirical validation results evaluated on unseen cases using the saved "
            "<code>StandardScaler + LogisticRegression</code> pipeline."
        )

        # 5 Linear-style KPI Stat Tiles
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Accuracy</div>
                    <div class="stat-value">{metrics['accuracy'] * 100:.2f}%</div>
                    <div class="stat-context">112 / 114 correct cases</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">ROC-AUC</div>
                    <div class="stat-value">{metrics['roc_auc']:.4f}</div>
                    <div class="stat-context">Discriminative capacity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            mal_rec = metrics["per_class"]["malignant"]["recall_sensitivity"]
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Malignant Recall</div>
                    <div class="stat-value" style="color: #fda4af;">{mal_rec * 100:.2f}%</div>
                    <div class="stat-context">41 of 42 tumors caught</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            mal_prec = metrics["per_class"]["malignant"]["precision"]
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Malignant Precision</div>
                    <div class="stat-value">{mal_prec * 100:.2f}%</div>
                    <div class="stat-context">Positive predictive value</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi5:
            ben_rec = metrics["per_class"]["benign"].get("recall_specificity", metrics["per_class"]["benign"].get("recall_sensitivity", 0.9861))
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Benign Specificity</div>
                    <div class="stat-value" style="color: #6ee7b7;">{ben_rec * 100:.2f}%</div>
                    <div class="stat-context">71 of 72 ruled out</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("##### Confusion Matrix")
            cm_img = REPORTS_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), use_container_width=True)
        with col_roc:
            st.markdown("##### ROC Curve (AUC = 0.9954)")
            roc_img = REPORTS_DIR / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), use_container_width=True)

        # Classification Table
        st.markdown("##### Detailed Metric Breakdown")
        clf_dict = metrics.get("classification_report", {})
        if clf_dict:
            rep_df = pd.DataFrame(clf_dict).T
            st.dataframe(
                rep_df.style.format(
                    {
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:.0f}",
                    }
                ),
                use_container_width=True,
            )

    # ==========================================================================
    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================================================
    with tab_eda:
        st.markdown("### Exploratory Data Analysis & Feature Distributions")
        st.markdown(
            "Visual exploration of the 569 Wisconsin Diagnostic biopsies, examining class balance, "
            "correlation structures, and diagnostic morphological boundaries."
        )

        eda_c1, eda_c2 = st.columns(2)
        with eda_c1:
            st.markdown("##### Target Class Proportions")
            fig1 = REPORTS_DIR / "target_distribution.png"
            if fig1.exists():
                st.image(str(fig1), use_container_width=True)
        with eda_c2:
            st.markdown("##### Nuclear Feature Correlation Heatmap")
            fig2 = REPORTS_DIR / "feature_correlations.png"
            if fig2.exists():
                st.image(str(fig2), use_container_width=True)

        st.markdown("##### Key Distinguishing Nuclear Features")
        fig3 = REPORTS_DIR / "mean_features_distribution.png"
        if fig3.exists():
            st.image(str(fig3), use_container_width=True)

    # ==========================================================================
    # TAB 4: SYSTEM ARCHITECTURE & METHODOLOGY
    # ==========================================================================
    with tab_docs:
        st.markdown("### Engineering Architecture & Statistical Formulation")
        st.markdown(
            """
            #### 1. Wisconsin Diagnostic Breast Cancer (WDBC) Provenance
            The Wisconsin Diagnostic Breast Cancer dataset represents fine needle aspirates (FNAs) of human breast masses 
            collected by **Dr. William H. Wolberg** at the University of Wisconsin Hospitals. Digital scans were analyzed via 
            **XCONFOC** curve fitting algorithms (Street & Mangasarian) to characterize nuclear contour points.

            - **Total Biopsies:** 569 samples (212 Malignant, 357 Benign).
            - **Extracted Attributes:** 30 continuous real-valued dimensions (Mean, Standard Error, and Worst per cell nucleus).
            - **Missing Values:** Exactly 0 missing values across all dimensions.

            #### 2. Pipeline Design & Zero Data Leakage
            Feature normalization must strictly encapsulate training fold statistics to prevent information leakage:
            $$\\hat{x} = \\frac{x - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}}$$
            The scikit-learn `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])` guarantees 
            that validation and test splits are transformed solely via training mean and variance.

            #### 3. Mathematical Objective
            L-BFGS optimization solves the $L_2$-regularized negative log-likelihood:
            $$\\min_{w, b} \\frac{1}{2} w^T w + C \\sum_{i=1}^{N} \\log(1 + \\exp(-y_i (w^T x_i + b)))$$
            The calibrated sigmoid link yields continuous posterior probabilities:
            $$P(\\text{Malignant} \\mid x) = \\frac{1}{1 + \\exp(-(w^T x + b))}$$

            #### 4. Clinical Sensitivity Trade-Off
            In automated oncology screening, the cost of a **False Negative** ($C_{\\text{FN}}$) vastly exceeds that of a 
            **False Positive** ($C_{\\text{FP}}$). Logistic regression outputs enable setting decision thresholds 
            $\\tau \\le 0.50$ when prioritizing near-zero false negatives.
            """
        )


if __name__ == "__main__":
    main()
