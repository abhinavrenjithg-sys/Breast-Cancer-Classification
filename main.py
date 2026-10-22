"""
main.py
-------
Main execution script for Breast Cancer Classification.
Orchestrates data loading, exploratory figure generation, training,
evaluation, metric reporting, and artifact persistence.
"""

import json
from pathlib import Path
import pandas as pd
from src.data import (
    generate_eda_figures,
    inspect_data,
    load_data,
    split_data,
    TARGET_NAMES,
)
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from src.train import save_pipeline, train_model

# Project directories
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports" / "figures"


def run_pipeline() -> None:
    """Execute the end-to-end Machine Learning training and evaluation pipeline."""
    print("=" * 70)
    print("  BREAST CANCER CLASSIFICATION: MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/5] Loading Wisconsin Diagnostic Breast Cancer dataset...")
    X, y, target_names, feature_names, full_df = load_data()
    print(f"      Loaded {len(full_df)} instances with {len(feature_names)} features.")

    # 2. Inspect Data
    print("\n[2/5] Inspecting dataset integrity & characteristics...")
    inspection = inspect_data(full_df)
    print(f"      - Shape: {inspection['n_samples']} samples, {inspection['n_features']} features")
    print(f"      - Missing values: {inspection['total_missing_values']}")
    print(f"      - Duplicate records: {inspection['duplicate_rows']}")
    print(f"      - Class counts: {inspection['target_counts_named']}")
    print(f"      - Proportions: {inspection['target_proportions']}")

    # Generate and save EDA figures
    print("      Generating exploratory charts under reports/figures/...")
    eda_figures = generate_eda_figures(full_df, REPORTS_DIR)
    for fig_path in eda_figures:
        print(f"      -> Created {fig_path.relative_to(BASE_DIR)}")

    # 3. Stratified Train / Test Split (80 / 20)
    print("\n[3/5] Splitting dataset into stratified train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f"      - Training set: {X_train.shape[0]} samples")
    print(f"      - Testing set:  {X_test.shape[0]} samples")

    # 4. Model Training (Pipeline: StandardScaler + LogisticRegression)
    print("\n[4/5] Training Pipeline (StandardScaler + LogisticRegression)...")
    print("      Preprocessing fitted strictly on training data to avoid data leakage.")
    pipeline = train_model(X_train, y_train, random_state=42, max_iter=2000)
    print("      Pipeline training completed successfully.")

    # 5. Model Evaluation
    print("\n[5/5] Evaluating model on held-out test set...")
    metrics = evaluate_model(pipeline, X_test, y_test, target_names=target_names)

    print("\n" + "-" * 70)
    print("                    EVALUATION METRICS SUMMARY")
    print("-" * 70)
    print(f"  Overall Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print(f"  ROC-AUC Score:         {metrics['roc_auc']:.4f}")
    print(f"  Macro F1-Score:        {metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1-Score:     {metrics['weighted_avg']['f1_score']:.4f}")
    print("\n  Per-Class Performance:")
    for cls_name, cls_stats in metrics["per_class"].items():
        print(f"    * {cls_name.upper():<10} | Precision: {cls_stats['precision']:.4f} | "
              f"Recall: {cls_stats.get('recall_sensitivity', cls_stats.get('recall_specificity')):.4f} | "
              f"F1: {cls_stats['f1_score']:.4f} | Support: {cls_stats['support']}")

    print("\n  Classification Report:")
    print(metrics["classification_report_str"])

    print("  Confusion Matrix [rows: actual (malignant, benign), cols: predicted]:")
    cm_display = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Malignant", "Actual Benign"],
        columns=["Pred Malignant", "Pred Benign"],
    )
    print(cm_display.to_string())

    # Generate evaluation visualizations
    cm_fig_path = plot_confusion_matrix(
        metrics["confusion_matrix"], REPORTS_DIR / "confusion_matrix.png", target_names=target_names
    )
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_fig_path = plot_roc_curve(y_test, y_test_proba, REPORTS_DIR / "roc_curve.png")
    print(f"\n  Saved evaluation visuals:")
    print(f"  -> Confusion Matrix: {cm_fig_path.relative_to(BASE_DIR)}")
    print(f"  -> ROC Curve:        {roc_fig_path.relative_to(BASE_DIR)}")

    # 6. Save Artifacts
    print("\nSaving artifacts under artifacts/ directory...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Pipeline
    model_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    save_pipeline(pipeline, model_path)
    print(f"  -> Pipeline:       {model_path.relative_to(BASE_DIR)}")

    # Save Metrics JSON
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"  -> Metrics JSON:   {metrics_path.relative_to(BASE_DIR)}")

    # Save Held-out Test Data for the Streamlit App
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df["diagnosis"] = y_test.map({0: "malignant", 1: "benign"})
    test_data_path = ARTIFACTS_DIR / "test_data.csv"
    test_df.to_csv(test_data_path, index=True)
    print(f"  -> Test Dataset:   {test_data_path.relative_to(BASE_DIR)}")

    # Save Feature Names & Target Names Metadata
    metadata = {
        "feature_names": feature_names,
        "target_names": target_names,
        "n_samples": len(full_df),
        "n_features": len(feature_names),
        "n_test_samples": len(test_df),
        "n_train_samples": len(X_train),
    }
    metadata_path = ARTIFACTS_DIR / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Metadata JSON:  {metadata_path.relative_to(BASE_DIR)}")

    print("\n" + "=" * 70)
    print("  Pipeline execution complete! Ready for Streamlit app & pytest.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
