"""
src/evaluate.py
---------------
Model evaluation, clinical metrics calculation, and visualization module
for Breast Cancer Classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

TARGET_NAMES = ["malignant", "benign"]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to properly serialize NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: List[str] = TARGET_NAMES,
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics across both classes, paying special
    clinical attention to the malignant category (label 0).

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Held-out test features (DataFrame).
        y_test: True test labels (Series).
        target_names: Names for labels [0, 1] (['malignant', 'benign']).

    Returns:
        metrics (Dict[str, Any]): Comprehensive dictionary of evaluation results.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Class 1 is 'benign' and Class 0 is 'malignant'
    # For standard binary ROC-AUC in sklearn where 1 is positive:
    proba_benign = y_proba[:, 1]
    proba_malignant = y_proba[:, 0]

    # Overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, proba_benign))

    # Macro & weighted averages
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_mac = float(f1_score(y_test, y_pred, average="macro"))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted"))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted"))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted"))

    # Per-class detailed breakdown
    # Label 0: Malignant, Label 1: Benign
    prec_per_class = precision_score(y_test, y_pred, average=None)
    rec_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    clf_report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    clf_report_str = classification_report(
        y_test, y_pred, target_names=target_names
    )

    metrics = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "macro_avg": {
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1_score": round(f1_mac, 4),
        },
        "weighted_avg": {
            "precision": round(prec_weighted, 4),
            "recall": round(rec_weighted, 4),
            "f1_score": round(f1_wt, 4),
        },
        "per_class": {
            "malignant": {
                "label_id": 0,
                "precision": round(float(prec_per_class[0]), 4),
                "recall_sensitivity": round(float(rec_per_class[0]), 4),
                "f1_score": round(float(f1_per_class[0]), 4),
                "support": int((y_test == 0).sum()),
            },
            "benign": {
                "label_id": 1,
                "precision": round(float(prec_per_class[1]), 4),
                "recall_specificity": round(float(rec_per_class[1]), 4),
                "f1_score": round(float(f1_per_class[1]), 4),
                "support": int((y_test == 1).sum()),
            },
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report_dict,
        "classification_report_str": clf_report_str,
    }
    return metrics


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    output_path: Path,
    target_names: List[str] = TARGET_NAMES,
) -> Path:
    """
    Generate and save a publication-quality confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array or nested list.
        output_path: Path to save figure.
        target_names: Class labels (['malignant', 'benign']).

    Returns:
        output_path (Path): Path to saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_array = np.array(cm)

    # Compute percentages across actual rows
    row_sums = cm_array.sum(axis=1, keepdims=True)
    out_pct = np.zeros_like(cm_array, dtype=float)
    cm_pct = np.divide(cm_array.astype(float), row_sums.astype(float), out=out_pct, where=row_sums != 0) * 100

    labels = [
        [f"{val}\n({pct:.1f}%)" for val, pct in zip(row_val, row_pct)]
        for row_val, row_pct in zip(cm_array, cm_pct)
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#FAFAFA")

    sns.heatmap(
        cm_array,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[t.title() for t in target_names],
        yticklabels=[t.title() for t in target_names],
        ax=ax,
        linewidths=1.2,
        linecolor="#E2E8F0",
        annot_kws={"size": 12, "fontweight": "bold"},
    )

    ax.set_title("Confusion Matrix (Held-out Test Set)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Actual Diagnosis", fontsize=11, fontweight="semibold", labelpad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Generate and save a clear Receiver Operating Characteristic (ROC) curve.

    Args:
        y_test: Ground truth labels (0 or 1).
        y_proba: Predicted probabilities for the positive class (benign, label 1).
        output_path: Path to save figure.

    Returns:
        output_path (Path): Path to saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#FAFAFA")

    ax.plot(
        fpr,
        tpr,
        color="#2563EB",
        lw=2.5,
        label=f"Logistic Regression ROC (AUC = {auc_val:.4f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563EB")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """
    Save metrics dictionary to a formatted JSON file.

    Args:
        metrics: Dictionary of evaluated metrics.
        output_path: Filepath for the JSON file.

    Returns:
        resolved_path (Path): Output filepath.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    return path
