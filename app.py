"""
app.py
------
Streamlit Web Application for Wisconsin Diagnostic Breast Cancer Classification.
Provides interactive model inference on held-out test data, evaluation dashboard,
and clinical metric interpretation.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.train import load_pipeline

# ------------------------------------------------------------------------------
# Application Configuration & Constants
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports" / "figures"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.csv"
METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"

# Custom Styling for polished aesthetic
CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #475569;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Prediction Result Cards */
    .result-card-benign {
        background-color: #ECFDF5;
        border: 2px solid #059669;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .result-card-malignant {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .result-title-benign {
        color: #065F46;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-title-malignant {
        color: #991B1B;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    /* Disclaimer Banner */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 24px;
        color: #92400E;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Resource Loaders (Avoids retraining / reloading on every interaction)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_pipeline() -> Optional[Pipeline]:
    """Load the trained scikit-learn pipeline from disk."""
    if not MODEL_PATH.exists():
        return None
    try:
        return load_pipeline(MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_metrics_and_metadata() -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Load evaluation metrics, dataset metadata, and held-out test data."""
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
# Main Application Flow
# ------------------------------------------------------------------------------
def main():
    # Header & Title
    st.title("🎗️ Breast Cancer Diagnostic Classification")
    st.markdown(
        "**Wisconsin Diagnostic Breast Cancer (WDBC)** machine learning system with **StandardScaler + Logistic Regression**."
    )

    # Prominent Medical Disclaimer (Mandatory Requirement)
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ IMPORTANT MEDICAL & EDUCATIONAL DISCLAIMER:</strong><br>
            This application is created strictly for <strong>educational demonstration and machine learning research purposes</strong>.
            It is <strong>NOT</strong> an FDA-approved diagnostic medical device, nor should it ever be utilized as a substitute
            for professional medical judgment, clinical evaluation, or treatment decisions.
            Always seek advice from a certified oncologist or licensed healthcare provider for any diagnostic inquiries.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check Artifacts Availability
    pipeline = get_model_pipeline()
    metrics, metadata, test_df = get_metrics_and_metadata()

    if pipeline is None or metrics is None or test_df is None:
        st.error("⚠️ **Trained Model Artifacts Not Found**")
        st.markdown(
            """
            The trained model pipeline and evaluation artifacts have not been generated yet.
            
            To train the model and generate required artifacts, execute this command in your terminal from the project root:
            ```bash
            python main.py
            ```
            After running the command, reload this page to access full features.
            """
        )
        st.info("Artifacts expected: `artifacts/model_pipeline.joblib`, `artifacts/metrics.json`, `artifacts/test_data.csv`.")
        return

    # Sidebar Navigation & Context
    with st.sidebar:
        st.header("🔬 Project Information")
        st.markdown(
            f"""
            - **Dataset:** Wisconsin Diagnostic (WDBC)
            - **Total Samples:** 569 instances
            - **Features:** 30 continuous cell nuclei attributes
            - **Split:** 80% Train ({metadata.get('n_train_samples', 455)}) / 20% Test ({metadata.get('n_test_samples', 114)})
            - **Model:** `StandardScaler` + `LogisticRegression`
            - **Solver:** L-BFGS (max_iter=2000)
            """
        )
        st.markdown("---")
        st.subheader("⚡ Quick Commands")
        st.code("python main.py\npytest\nstreamlit run app.py", language="bash")
        st.markdown("---")
        st.caption("Built with Scikit-Learn, Streamlit, Pandas & Seaborn.")

    # Main Tabs
    tab_inference, tab_metrics, tab_eda, tab_dataset = st.tabs(
        [
            "🔬 Test Sample Inference",
            "📈 Performance & Metrics",
            "📊 Data Exploration (EDA)",
            "ℹ️ Dataset & Methodology",
        ]
    )

    # ==========================================================================
    # TAB 1: INTERACTIVE TEST SAMPLE INFERENCE
    # ==========================================================================
    with tab_inference:
        st.subheader("Live Inference on Held-Out Test Samples")
        st.markdown(
            "Select an unseen sample from the **stratified 20% test partition** (114 samples) to test the trained pipeline."
        )

        # Filter option for user convenience
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            filter_choice = st.selectbox(
                "Filter test cohort by actual diagnosis:",
                options=["All Test Samples", "Malignant Only", "Benign Only"],
            )

        filtered_df = test_df.copy()
        if filter_choice == "Malignant Only":
            filtered_df = filtered_df[filtered_df["diagnosis"] == "malignant"]
        elif filter_choice == "Benign Only":
            filtered_df = filtered_df[filtered_df["diagnosis"] == "benign"]

        # Dropdown to choose sample
        sample_options = [
            f"Sample #{idx} (Actual Diagnosis: {row['diagnosis'].title()})"
            for idx, row in filtered_df.iterrows()
        ]
        selected_option = st.selectbox("Select patient sample from test set:", sample_options)

        # Extract selected row
        selected_idx = int(selected_option.split("#")[1].split(" ")[0])
        sample_row = test_df.loc[selected_idx]
        actual_label = sample_row["diagnosis"]
        feature_cols = [c for c in test_df.columns if c not in ("target", "diagnosis")]
        sample_features = sample_row[feature_cols].to_frame().T

        # Expandable inspection of the sample's 30 features
        with st.expander("🔍 View Raw Morphological Feature Values for this Sample"):
            col_m1, col_m2, col_m3 = st.columns(3)
            mean_cols = [c for c in feature_cols if "mean" in c]
            se_cols = [c for c in feature_cols if "error" in c]
            worst_cols = [c for c in feature_cols if "worst" in c]

            with col_m1:
                st.markdown("**Mean Attributes:**")
                st.dataframe(sample_features[mean_cols].T.rename(columns={selected_idx: "Value"}), height=300)
            with col_m2:
                st.markdown("**Standard Error Attributes:**")
                st.dataframe(sample_features[se_cols].T.rename(columns={selected_idx: "Value"}), height=300)
            with col_m3:
                st.markdown("**Worst (Largest) Attributes:**")
                st.dataframe(sample_features[worst_cols].T.rename(columns={selected_idx: "Value"}), height=300)

        # Inference Action
        st.write("")
        predict_button = st.button("🚀 Run Diagnostic Model on Selected Sample", type="primary", use_container_width=True)

        if predict_button:
            # Model prediction (using exact saved pipeline)
            pred_class_id = int(pipeline.predict(sample_features)[0])
            pred_probas = pipeline.predict_proba(sample_features)[0]

            # In scikit-learn load_breast_cancer: 0 = Malignant, 1 = Benign
            proba_malignant = float(pred_probas[0])
            proba_benign = float(pred_probas[1])
            predicted_diagnosis = "malignant" if pred_class_id == 0 else "benign"

            # Visual Result Card
            if predicted_diagnosis == "malignant":
                st.markdown(
                    f"""
                    <div class="result-card-malignant">
                        <div class="result-title-malignant">⚠️ PREDICTED DIAGNOSIS: MALIGNANT (Class 0)</div>
                        <p style="color: #7F1D1D; font-size: 1.05rem; margin-bottom: 12px;">
                            The classifier detected cellular morphology patterns highly indicative of <strong>malignancy</strong>.
                        </p>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #991B1B;">
                            Confidence (P[Malignant]): {proba_malignant * 100:.2f}% | P[Benign]: {proba_benign * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-benign">
                        <div class="result-title-benign">✅ PREDICTED DIAGNOSIS: BENIGN (Class 1)</div>
                        <p style="color: #064E3B; font-size: 1.05rem; margin-bottom: 12px;">
                            The classifier identified cellular morphology consistent with <strong>benign tissue</strong>.
                        </p>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #065F46;">
                            Confidence (P[Benign]): {proba_benign * 100:.2f}% | P[Malignant]: {proba_malignant * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probability Bars
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(f"**Malignant Probability:** `{proba_malignant * 100:.2f}%`")
                st.progress(proba_malignant)
            with pcol2:
                st.markdown(f"**Benign Probability:** `{proba_benign * 100:.2f}%`")
                st.progress(proba_benign)

            # Ground Truth Comparison & Clinical Explanation
            st.markdown("---")
            st.subheader("💡 Clinical Interpretation & Diagnostic Context")
            is_match = predicted_diagnosis == actual_label
            status_badge = "✅ MATCH (Correct Prediction)" if is_match else "❌ MISMATCH"

            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.markdown(
                    f"""
                    - **Sample Index:** #{selected_idx}
                    - **Actual Pathology Diagnosis:** `{actual_label.upper()}`
                    - **Model Predicted Diagnosis:** `{predicted_diagnosis.upper()}`
                    - **Validation Status:** **{status_badge}**
                    """
                )
            with col_eval2:
                if actual_label == "malignant" and predicted_diagnosis == "malignant":
                    st.success("True Positive: Malignant case caught successfully by model.")
                elif actual_label == "benign" and predicted_diagnosis == "benign":
                    st.success("True Negative: Benign tissue correctly ruled out.")
                elif actual_label == "malignant" and predicted_diagnosis == "benign":
                    st.error("False Negative: Life-critical missed malignant case.")
                else:
                    st.warning("False Positive: Benign sample flagged for further biopsy.")

            st.info(
                """
                **Why Sensitivity (Recall) is Vital in Oncology:**
                In breast cancer detection, a **False Negative** (diagnosing a malignant tumor as benign) can delay life-saving medical interventions.
                Conversely, a **False Positive** triggers secondary confirmatory biopsies. Logistic Regression with calibrated probability output allows
                clinical teams to set decision thresholds prioritizing high sensitivity for malignant detection.
                """
            )

    # ==========================================================================
    # TAB 2: MODEL PERFORMANCE & METRICS
    # ==========================================================================
    with tab_metrics:
        st.subheader("Evaluation Metrics (Held-Out Test Set: 114 Samples)")

        # KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value">{metrics['accuracy'] * 100:.2f}%</div>
                    <div class="metric-sub">Overall Test Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">ROC-AUC</div>
                    <div class="metric-value">{metrics['roc_auc']:.4f}</div>
                    <div class="metric-sub">Separation Ability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            mal_rec = metrics["per_class"]["malignant"]["recall_sensitivity"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Malignant Recall</div>
                    <div class="metric-value">{mal_rec * 100:.2f}%</div>
                    <div class="metric-sub">Sensitivity (Critical)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            mal_prec = metrics["per_class"]["malignant"]["precision"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Malignant Precision</div>
                    <div class="metric-value">{mal_prec * 100:.2f}%</div>
                    <div class="metric-sub">Biopsy Confirmation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi5:
            ben_f1 = metrics["per_class"]["benign"]["f1_score"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Benign F1-Score</div>
                    <div class="metric-value">{ben_f1:.4f}</div>
                    <div class="metric-sub">Harmonic Mean</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        # Visualizations (Confusion Matrix & ROC Curve)
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("#### Confusion Matrix")
            cm_img_path = REPORTS_DIR / "confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), use_container_width=True)
            else:
                st.warning("Confusion matrix chart not found. Run `python main.py`.")

        with col_viz2:
            st.markdown("#### ROC Curve")
            roc_img_path = REPORTS_DIR / "roc_curve.png"
            if roc_img_path.exists():
                st.image(str(roc_img_path), use_container_width=True)
            else:
                st.warning("ROC curve chart not found. Run `python main.py`.")

        # Detailed Classification Report
        st.markdown("#### Detailed Classification Report")
        clf_dict = metrics.get("classification_report", {})
        if clf_dict:
            report_df = pd.DataFrame(clf_dict).T
            st.dataframe(
                report_df.style.format(
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
    # TAB 3: DATA EXPLORATION (EDA)
    # ==========================================================================
    with tab_eda:
        st.subheader("Exploratory Data Analysis Figures")
        st.markdown(
            "Visual exploration of the 569 Wisconsin Diagnostic samples generated during pipeline execution."
        )

        eda1, eda2 = st.columns(2)
        with eda1:
            st.markdown("##### 1. Target Class Distribution")
            dist_fig = REPORTS_DIR / "target_distribution.png"
            if dist_fig.exists():
                st.image(str(dist_fig), use_container_width=True)
            else:
                st.warning("Target distribution figure not found.")

        with eda2:
            st.markdown("##### 2. Feature Correlation Heatmap")
            corr_fig = REPORTS_DIR / "feature_correlations.png"
            if corr_fig.exists():
                st.image(str(corr_fig), use_container_width=True)
            else:
                st.warning("Correlation figure not found.")

        st.markdown("##### 3. Key Distinguishing Features Distribution")
        dist_multi_fig = REPORTS_DIR / "mean_features_distribution.png"
        if dist_multi_fig.exists():
            st.image(str(dist_multi_fig), use_container_width=True)
        else:
            st.warning("Feature distribution figure not found.")

    # ==========================================================================
    # TAB 4: DATASET & METHODOLOGY
    # ==========================================================================
    with tab_dataset:
        st.subheader("Dataset Description & Engineering Architecture")
        st.markdown(
            """
            ### Wisconsin Diagnostic Breast Cancer (WDBC)
            - **Origin:** University of Wisconsin Hospitals, Madison (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian).
            - **Samples:** 569 patient samples.
            - **Target:** Binary diagnosis — **Malignant** (212 samples, 37.3%) vs **Benign** (357 samples, 62.7%).
            - **Input Features:** 30 continuous real-valued features computed from fine needle aspirate (FNA) digital images of breast masses.
            
            #### 10 Core Morphological Features Measured:
            1. **Radius:** Mean of distances from center to points on the perimeter.
            2. **Texture:** Standard deviation of gray-scale values.
            3. **Perimeter:** Total boundary perimeter of the cell nucleus.
            4. **Area:** Nuclear area.
            5. **Smoothness:** Local variation in radius lengths.
            6. **Compactness:** $(perimeter^2 / area - 1.0)$.
            7. **Concavity:** Severity of concave portions of the contour.
            8. **Concave Points:** Number of concave portions of the contour.
            9. **Symmetry:** Nuclear symmetry.
            10. **Fractal Dimension:** Approximation of coastline $(dimension - 1)$.
            
            *(For each cell nucleus, Mean, Standard Error, and 'Worst' [largest] value are computed, yielding 30 total dimensions).*

            ### Modeling Methodology
            1. **Stratified 80/20 Splitting:** Preserves the underlying clinical class ratio in both training (455 samples) and testing (114 samples).
            2. **Pipeline Standardization:** `StandardScaler` standardizes each feature to zero mean and unit variance. Crucially, the scaler is fit **only on training data** within a scikit-learn Pipeline to prevent data leakage.
            3. **Logistic Regression Classifier:** Fitted with `max_iter=2000` to guarantee numerical convergence on standardized inputs.
            4. **Probabilistic Calibration:** Outputs continuous probability distributions allowing clinical sensitivity tuning.
            """
        )


if __name__ == "__main__":
    main()
