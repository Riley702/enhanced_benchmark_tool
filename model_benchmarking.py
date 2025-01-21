from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix
)
import time
import numpy as np

def benchmark_model(model, X, y, test_size=0.2, random_state=42, problem_type='classification'):
    """
    Benchmarks a machine learning model for classification or regression.

    Args:
        model: Scikit-learn compatible model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.
        problem_type (str): 'classification' or 'regression'.

    Returns:
        dict: Benchmark metrics and training/testing times.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Record training time
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    # Record testing time
    start_test = time.time()
    predictions = model.predict(X_test)
    end_test = time.time()

    # Initialize metrics
    metrics = {
        "training_time": end_train - start_train,
        "testing_time": end_test - start_test,
    }

    if problem_type == 'classification':
        # Classification metrics
        metrics.update({
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average='weighted'),
            "recall": recall_score(y_test, predictions, average='weighted'),
            "f1_score": f1_score(y_test, predictions, average='weighted'),
            "classification_report": classification_report(y_test, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        })
    elif problem_type == 'regression':
        # Regression metrics
        metrics.update({
            "mean_squared_error": mean_squared_error(y_test, predictions),
            "mean_absolute_error": mean_absolute_error(y_test, predictions),
            "r2_score": r2_score(y_test, predictions),
            "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, predictions)),
        })
    else:
        raise ValueError("Invalid problem_type. Use 'classification' or 'regression'.")

    return metrics

def extract_feature_importance(model, feature_names=None):
    """
    Extracts and returns feature importance from a fitted model, if available.

    Args:
        model: A trained scikit-learn compatible model.
        feature_names (list or None): List of feature names. If None, feature indices are used.

    Returns:
        dict: A dictionary mapping feature names to their importance scores.
    """
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        raise AttributeError("The model does not support feature importance extraction.")

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(model.coef_))] if hasattr(model, "coef_") else None

    if hasattr(model, "feature_importances_"):
        # Tree-based models (e.g., RandomForest, DecisionTree)
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models (e.g., LogisticRegression, LinearRegression)
        importances = np.abs(model.coef_).flatten()

    if feature_names:
        importance_dict = dict(zip(feature_names, importances))
    else:
        importance_dict = {f"Feature {i}": importance for i, importance in enumerate(importances)}

    # Normalize importance values for better interpretability
    total_importance = sum(importance_dict.values())
    normalized_importance = {k: v / total_importance for k, v in importance_dict.items()}

    return normalized_importance
