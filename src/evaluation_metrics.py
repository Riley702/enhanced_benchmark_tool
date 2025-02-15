"""
Evaluation Metrics Module
=========================

This module provides utilities for evaluating and comparing machine learning models.

    - Confusion matrices for classification models
    - Automated classification reports
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import time
import pandas as pd


def calculate_metrics_comparison(models, X, y, test_size=0.2, random_state=42, problem_type='classification'):
    """
    Compares multiple machine learning models on a given dataset.

    Args:
        models (dict): A dictionary where keys are model names and values are model objects.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        random_state (int): Seed for reproducibility.

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
                "precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
                "recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, predictions, average='weighted', zero_division=0),
            })

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_test)[:, 1]
                metrics["log_loss"] = log_loss(y_test, probabilities)
                metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
        elif problem_type == 'regression':
            metrics.update({
                "mean_squared_error": mean_squared_error(y_test, predictions),
                "mean_absolute_error": mean_absolute_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions),
                "root_mean_squared_error": mean_squared_error(y_test, predictions, squared=False),
                "adjusted_r2": compute_adjusted_r2(y_test, predictions, X_train.shape[1]),
            })
        else:
            raise ValueError("Invalid problem_type. Use 'classification' or 'regression'.")

        results[model_name] = metrics

    return results


def compute_classification_confusion_matrices(models, X, y, test_size=0.2, random_state=42):
    """
    Computes confusion matrices for multiple classification models.

    Args:
        models (dict): Dictionary of classification models.
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
        test_size (float): Proportion of dataset to allocate for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: Confusion matrices for each model.
    """
    confusion_matrices = {}

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm.tolist()  # Convert to list for JSON serialization

    return confusion_matrices


def compute_regression_errors(models, X, y, test_size=0.2, random_state=42):
    """
    Computes additional error metrics for multiple regression models.

    Args:
        models (dict): Dictionary of regression models.
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target values.
        test_size (float): Proportion of dataset to allocate for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Table of regression error metrics per model.
    """
    errors = []

    # Split data
