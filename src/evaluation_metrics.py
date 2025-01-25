"""
Evaluation Metrics Module
=========================

This module provides utilities for evaluating and comparing machine learning models.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import time


def calculate_metrics_comparison(models, X, y, test_size=0.2, random_state=42, problem_type='classification'):
    """
    Compares multiple machine learning models on a given dataset.

    Args:
        models (dict): A dictionary where keys are model names and values are model objects.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.
        problem_type (str): 'classification' or 'regression'.

    Returns:
        dict: A dictionary with model names as keys and their metrics as values.
    """
    results = {}

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for model_name, model in models.items():
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()

        start_test = time.time()
        predictions = model.predict(X_test)
        end_test = time.time()

        metrics = {
            "training_time": end_train - start_train,
            "testing_time": end_test - start_test,
        }

        if problem_type == 'classification':
            metrics.update({
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average='weighted'),
                "recall": recall_score(y_test, predictions, average='weighted'),
                "f1_score": f1_score(y_test, predictions, average='weighted'),
            })
        elif problem_type == 'regression':
            metrics.update({
                "mean_squared_error": mean_squared_error(y_test, predictions),
                "mean_absolute_error": mean_absolute_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions),
                "root_mean_squared_error": mean_squared_error(y_test, predictions, squared=False),
            })
        else:
            raise ValueError("Invalid problem_type. Use 'classification' or 'regression'.")

        results[model_name] = metrics

    return results
