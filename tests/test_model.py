"""
tests/test_model.py
-------------------
Automated unit and integration tests for the Breast Cancer Classification project.
Tests dataset loading, pipeline construction, training, probability outputs,
metrics validity, and model persistence.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data import (
    LABEL_MAP,
    TARGET_NAMES,
    inspect_data,
    load_data,
    split_data,
)
from src.evaluate import evaluate_model
from src.train import (
    build_pipeline,
    load_pipeline,
    save_pipeline,
    train_model,
)


@pytest.fixture(scope="module")
def dataset():
    """Fixture providing loaded Wisconsin breast cancer dataset."""
    X, y, target_names, feature_names, full_df = load_data()
    return {
        "X": X,
        "y": y,
        "target_names": target_names,
        "feature_names": feature_names,
        "full_df": full_df,
    }


@pytest.fixture(scope="module")
def data_splits(dataset):
    """Fixture providing stratified train and test splits."""
    X_train, X_test, y_train, y_test = split_data(
        dataset["X"], dataset["y"], test_size=0.20, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture(scope="module")
def trained_pipeline(data_splits):
    """Fixture providing a fitted pipeline on training data."""
    return train_model(
        data_splits["X_train"],
        data_splits["y_train"],
        random_state=42,
        max_iter=2000,
    )


# ==============================================================================
# 1. Dataset Loading and Integrity Tests
# ==============================================================================

def test_load_data_shape(dataset):
    """Verify Wisconsin dataset has exactly 569 samples and 30 numerical features."""
    X = dataset["X"]
    y = dataset["y"]
    assert X.shape == (569, 30), f"Expected shape (569, 30), got {X.shape}"
    assert len(y) == 569, f"Expected 569 target labels, got {len(y)}"


def test_target_categories(dataset):
    """Verify target classes are 'malignant' and 'benign' with correct values."""
    assert dataset["target_names"] == ["malignant", "benign"]
    unique_labels = sorted(dataset["y"].unique().tolist())
    assert unique_labels == [0, 1]
    # Check exact counts documented in sklearn
    counts = dataset["y"].value_counts().to_dict()
    assert counts[0] == 212, f"Expected 212 malignant samples, got {counts[0]}"
    assert counts[1] == 357, f"Expected 357 benign samples, got {counts[1]}"


def test_no_missing_values(dataset):
    """Ensure raw dataset has zero missing values."""
    inspection = inspect_data(dataset["full_df"])
    assert inspection["total_missing_values"] == 0
    assert inspection["duplicate_rows"] == 0


def test_stratified_split(data_splits, dataset):
    """Verify train/test split retains the 80/20 proportion and stratification."""
    X_train = data_splits["X_train"]
    X_test = data_splits["X_test"]
    y_train = data_splits["y_train"]
    y_test = data_splits["y_test"]

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert len(X_train) + len(X_test) == 569

    # Check stratification ratio (benign proportion should be ~62.7% in both)
    train_benign_ratio = (y_train == 1).mean()
    test_benign_ratio = (y_test == 1).mean()
    overall_benign_ratio = (dataset["y"] == 1).mean()

    assert abs(train_benign_ratio - overall_benign_ratio) < 0.02
    assert abs(test_benign_ratio - overall_benign_ratio) < 0.02


# ==============================================================================
# 2. Pipeline and Training Tests
# ==============================================================================

def test_pipeline_structure():
    """Verify pipeline contains StandardScaler followed by LogisticRegression."""
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    step_names = [step[0] for step in pipeline.steps]
    assert step_names == ["scaler", "classifier"]
    assert pipeline.named_steps["classifier"].max_iter == 2000


def test_pipeline_training(trained_pipeline, data_splits):
    """Verify pipeline fits without errors and is capable of inference."""
    X_test = data_splits["X_test"]
    preds = trained_pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_prediction_output_shapes(trained_pipeline, data_splits):
    """Ensure predictions and predict_proba return expected dimensions."""
    X_test = data_splits["X_test"]
    y_pred = trained_pipeline.predict(X_test)
    y_proba = trained_pipeline.predict_proba(X_test)

    assert y_pred.shape == (114,)
    assert y_proba.shape == (114, 2)


def test_probability_predictions_validity(trained_pipeline, data_splits):
    """Verify predicted probabilities are within [0, 1] and sum to 1.0."""
    X_test = data_splits["X_test"]
    y_proba = trained_pipeline.predict_proba(X_test)

    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)
    # Each row must sum to 1.0 within floating point precision
    row_sums = y_proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


# ==============================================================================
# 3. Model Performance Tests
# ==============================================================================

def test_model_performance_thresholds(trained_pipeline, data_splits):
    """
    Ensure the trained model satisfies baseline clinical and diagnostic thresholds:
    - Overall accuracy >= 95%
    - ROC-AUC >= 0.95
    - Malignant Recall (Sensitivity) >= 90% (crucial to avoid false negatives in cancer screening)
    """
    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]

    metrics = evaluate_model(trained_pipeline, X_test, y_test, target_names=TARGET_NAMES)

    assert metrics["accuracy"] >= 0.95, f"Accuracy {metrics['accuracy']} is below 0.95"
    assert metrics["roc_auc"] >= 0.95, f"ROC-AUC {metrics['roc_auc']} is below 0.95"

    malignant_recall = metrics["per_class"]["malignant"]["recall_sensitivity"]
    assert malignant_recall >= 0.90, f"Malignant recall {malignant_recall} is below 0.90"

    malignant_precision = metrics["per_class"]["malignant"]["precision"]
    assert malignant_precision >= 0.90, f"Malignant precision {malignant_precision} is below 0.90"


# ==============================================================================
# 4. Artifact Persistence Tests
# ==============================================================================

def test_model_persistence(trained_pipeline, data_splits):
    """Verify model pipeline can be saved, reloaded, and produces identical predictions."""
    X_test = data_splits["X_test"]
    original_preds = trained_pipeline.predict(X_test)
    original_proba = trained_pipeline.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_model_path = Path(tmp_dir) / "test_pipeline.joblib"
        save_pipeline(trained_pipeline, temp_model_path)
        assert temp_model_path.is_file()

        reloaded_pipeline = load_pipeline(temp_model_path)
        reloaded_preds = reloaded_pipeline.predict(X_test)
        reloaded_proba = reloaded_pipeline.predict_proba(X_test)

        np.testing.assert_array_equal(original_preds, reloaded_preds)
        np.testing.assert_allclose(original_proba, reloaded_proba, rtol=1e-5)


def test_load_pipeline_missing_file():
    """Verify load_pipeline raises FileNotFoundError with descriptive message for missing file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_pipeline("non_existent_model_file.joblib")
    assert "Model artifact not found" in str(exc_info.value)
