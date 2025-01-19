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
