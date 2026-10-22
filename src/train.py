"""
src/train.py
------------
Pipeline creation, training, and persistence module for Breast Cancer Classification.
"""

from pathlib import Path
from typing import Union
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(
    random_state: int = 42,
    max_iter: int = 2000,
    C: float = 1.0,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with feature standardization and logistic regression.

    Standardization (StandardScaler) is strictly contained within the pipeline,
    ensuring parameters (mean and standard deviation) are learned solely from
    training data during `fit`, preventing data leakage into test sets.

    Args:
        random_state: Random seed for reproducible solver initialization.
        max_iter: Maximum solver iterations to guarantee full convergence.
        C: Inverse regularization strength (default 1.0).

    Returns:
        pipeline (Pipeline): Configured, unfitted scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=max_iter,
                    C=C,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    max_iter: int = 2000,
    C: float = 1.0,
) -> Pipeline:
    """
    Build and fit the classification pipeline on training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0 = malignant, 1 = benign).
        random_state: Random seed.
        max_iter: Maximum iterations.
        C: Inverse regularization strength.

    Returns:
        fitted_pipeline (Pipeline): Fitted scikit-learn pipeline.
    """
    pipeline = build_pipeline(random_state=random_state, max_iter=max_iter, C=C)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(pipeline: Pipeline, filepath: Union[str, Path]) -> Path:
    """
    Serialize and save a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        filepath: Target output path.

    Returns:
        resolved_path (Path): Path to the saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(filepath: Union[str, Path]) -> Pipeline:
    """
    Load a serialized scikit-learn pipeline from disk.

    Args:
        filepath: Path to the serialized pipeline artifact (.joblib).

    Returns:
        pipeline (Pipeline): Loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at: {path.resolve()}. "
            "Please run 'python main.py' first to train and save the model pipeline."
        )
    return joblib.load(path)
