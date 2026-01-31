from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def benchmark_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    problem_type: str = "classification",
) -> Dict[str, Any]:
    """Benchmark a model for classification or regression.

    Returns a dict of metrics and timing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    t2 = time.time()
    preds = model.predict(X_test)
    t3 = time.time()

    metrics: Dict[str, Any] = {
        "training_time": t1 - t0,
        "testing_time": t3 - t2,
    }

    if problem_type == "classification":
        metrics.update(
            {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0),
                "classification_report": classification_report(
                    y_test, preds, output_dict=True, zero_division=0
                ),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            }
        )
    elif problem_type == "regression":
        metrics.update(
            {
                "mean_squared_error": mean_squared_error(y_test, preds),
                "mean_absolute_error": mean_absolute_error(y_test, preds),
                "r2_score": r2_score(y_test, preds),
                "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_test, preds))),
            }
        )
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")

    return metrics


def extract_feature_importance(model: Any, feature_names: Optional[list[str]] = None) -> Dict[str, float]:
    """Extract normalized feature importance from a fitted model.

    Supports tree models with feature_importances_ and linear models with coef_.
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float)).flatten()
    else:
        raise AttributeError("Model does not expose feature_importances_ or coef_.")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    if len(feature_names) != len(importances):
        raise ValueError("feature_names length must match number of importances")

    total = float(importances.sum())
    if total == 0:
        # Avoid divide-by-zero; return zeros.
        return {k: 0.0 for k in feature_names}

    return {k: float(v / total) for k, v in zip(feature_names, importances)}
