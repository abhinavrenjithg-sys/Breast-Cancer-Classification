"""
src/data.py
-----------
Module responsible for loading, inspecting, splitting, and exploring
the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Standard target category mapping in sklearn's load_breast_cancer:
# Index 0 -> 'malignant', Index 1 -> 'benign'
TARGET_NAMES = ["malignant", "benign"]
LABEL_MAP = {0: "malignant", 1: "benign"}
COLOR_PALETTE = {"malignant": "#DC2626", "benign": "#059669"}


def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.DataFrame]:
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.

    The dataset contains 569 instances and 30 continuous numerical features
    extracted from digitized fine-needle aspirate (FNA) images of breast masses.
    Target categories are:
      - 0: Malignant (212 samples, ~37.3%)
      - 1: Benign (357 samples, ~62.7%)

    Returns:
        X (pd.DataFrame): 569 x 30 feature matrix.
        y (pd.Series): Binary target labels (0 = malignant, 1 = benign).
        target_names (List[str]): List of target class names ['malignant', 'benign'].
        feature_names (List[str]): List of 30 feature column names.
        full_df (pd.DataFrame): Combined DataFrame containing features, target,
                                and human-readable diagnosis column.
    """
    raw_data = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = raw_data.data
    y: pd.Series = raw_data.target
    target_names: List[str] = list(raw_data.target_names)
    feature_names: List[str] = list(raw_data.feature_names)

    # Combine into a single DataFrame for exploratory analysis
    full_df = X.copy()
    full_df["target"] = y
    full_df["diagnosis"] = y.map(LABEL_MAP)

    return X, y, target_names, feature_names, full_df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural and data-quality inspection on the dataset.

    Inspects shape, feature count, missing values, duplicates, and target distribution.
    No observations are silently modified or dropped.

    Args:
        df: The combined DataFrame with features and target.

    Returns:
        summary (Dict[str, Any]): Dictionary containing comprehensive inspection stats.
    """
    n_samples, n_cols = df.shape
    feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]
    n_features = len(feature_cols)

    missing_by_col = df[feature_cols].isnull().sum().to_dict()
    total_missing = sum(missing_by_col.values())
    n_duplicates = int(df.duplicated(subset=feature_cols).sum())

    target_counts = df["target"].value_counts().to_dict()
    diagnosis_counts = df["diagnosis"].value_counts().to_dict()
    diagnosis_proportions = (
        df["diagnosis"].value_counts(normalize=True).round(4).to_dict()
    )

    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_names": feature_cols,
        "total_missing_values": total_missing,
        "duplicate_rows": n_duplicates,
        "target_counts_raw": target_counts,
        "target_counts_named": diagnosis_counts,
        "target_proportions": diagnosis_proportions,
    }
    return summary


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets with stratification.

    Stratification ensures that the proportion of malignant vs. benign samples
    is preserved across both training (80%) and testing (20%) sets.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset allocated to the test split (default 0.20).
        random_state: Seed for reproducibility (default 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def generate_eda_figures(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Generate and save exploratory data analysis figures.

    Produces:
      1. target_distribution.png: Class frequency and percentage distribution.
      2. feature_correlations.png: Correlation heatmap of primary mean cell features.
      3. mean_features_distribution.png: Boxplots highlighting key distinguishing features.

    Args:
        df: Combined DataFrame including features and diagnosis column.
        output_dir: Directory where figures will be stored (reports/figures/).

    Returns:
        saved_paths (List[Path]): List of generated figure filepaths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. Target Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # Bar chart
    counts = df["diagnosis"].value_counts()
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=[COLOR_PALETTE["benign"], COLOR_PALETTE["malignant"]],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Diagnosis Class Counts", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Diagnosis", fontsize=11)
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="semibold")

    # Donut chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=[COLOR_PALETTE["benign"], COLOR_PALETTE["malignant"]],
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontweight="bold"),
    )
    axes[1].set_title("Class Ratio", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    target_dist_path = output_dir / "target_distribution.png"
    fig.savefig(target_dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(target_dist_path)

    # 2. Correlation Heatmap for Mean Features
    # The first 10 columns in the dataset correspond to "mean" measurements
    mean_cols = [c for c in df.columns if "mean" in c]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#FAFAFA")
    corr = df[mean_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap: Mean Cell Nuclei Features", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    corr_path = output_dir / "feature_correlations.png"
    fig.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(corr_path)

    # 3. Boxplots of Key Distinguishing Features
    key_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean concavity",
        "mean concave points",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor("#FAFAFA")
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            sns.boxplot(
                x="diagnosis",
                y=feature,
                data=df,
                hue="diagnosis",
                palette=[COLOR_PALETTE["benign"], COLOR_PALETTE["malignant"]],
                legend=False,
                ax=axes[idx],
                width=0.5,
            )
            axes[idx].set_title(feature.title(), fontsize=11, fontweight="bold")
            axes[idx].set_xlabel("")
            axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Distributions of Key Cell Nuclei Features by Diagnosis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    dist_path = output_dir / "mean_features_distribution.png"
    fig.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(dist_path)

    return saved_paths
