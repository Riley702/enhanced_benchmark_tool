import pandas as pd
import numpy as np
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

# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    # Example data
    X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
    y = [0, 1, 0, 1]

    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Benchmark the model
    metrics = benchmark_model(model, X, y, problem_type='classification')
    print("Benchmark Metrics:", metrics)

    # Extract feature importance
    feature_importance = extract_feature_importance(model, feature_names=X.columns)
    print("Feature Importance:", feature_importance)

def evaluate_model_with_thresholds(model, X, y, thresholds=None, test_size=0.2, random_state=42):
    """
    Evaluates a classification model with varying probability thresholds.

    Args:
        model: A scikit-learn compatible classification model with `predict_proba` method.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        thresholds (list or None): List of thresholds to evaluate. If None, defaults to [0.1, 0.2, ..., 0.9].
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each threshold.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The model must have a `predict_proba` method for threshold evaluation.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]  # Use probabilities for the positive class

    if thresholds is None:
        thresholds = [i / 10 for i in range(1, 10)]  # Default thresholds: 0.1 to 0.9

    # Evaluate the model at each threshold
    results = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_model_with_roc_auc(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a classification model using ROC AUC score and plots the ROC curve.

    Args:
        model: A scikit-learn compatible classification model with `predict_proba` method.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the ROC AUC score and other relevant data.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The model must have a `predict_proba` method for ROC AUC evaluation.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_scores = model.predict_proba(X_test)[:, 1]  # Use probabilities for the positive class

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, y_scores)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()

    return {"roc_auc_score": roc_auc}


def evaluate_model_with_classification_report(model, X, y, test_size=0.2, random_state=42):
    """
    Generates a classification report with precision, recall, and F1-score.

    Args:
        model: A scikit-learn compatible classification model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A detailed classification report.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    return report




def evaluate_model_with_log_loss(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a classification model using log loss.

    Args:
        model: A scikit-learn compatible classification model with `predict_proba` method.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the log loss value.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The model must have a `predict_proba` method for log loss evaluation.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)

    # Compute log loss
    log_loss_value = log_loss(y_test, y_probs)

    return {"log_loss": log_loss_value}


def evaluate_model_with_median_absolute_error(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a regression model using the median absolute error.

    Args:
        model: A scikit-learn compatible regression model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the median absolute error.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute median absolute error
    median_abs_error = median_absolute_error(y_test, y_pred)

    return {"median_absolute_error": median_abs_error}





def evaluate_model_with_log_loss(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a classification model using log loss.

    Args:
        model: A scikit-learn compatible classification model with `predict_proba` method.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the log loss value.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The model must have a `predict_proba` method for log loss evaluation.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)

    # Compute log loss
    log_loss_value = log_loss(y_test, y_probs)

    return {"log_loss": log_loss_value}


def evaluate_model_with_median_absolute_error(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a regression model using the median absolute error.

    Args:
        model: A scikit-learn compatible regression model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the median absolute error.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute median absolute error
    median_abs_error = median_absolute_error(y_test, y_pred)

    return {"median_absolute_error": median_abs_error}


def evaluate_model_with_cohen_kappa(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a classification model using Cohen's Kappa score.

    Args:
        model: A scikit-learn compatible classification model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the Cohen's Kappa score.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute Cohen's Kappa score
    kappa_score = cohen_kappa_score(y_test, y_pred)

    return {"cohen_kappa_score": kappa_score}


def evaluate_model_with_mean_squared_log_error(model, X, y, test_size=0.2, random_state=42):
    """
    Evaluates a regression model using Mean Squared Log Error (MSLE).

    Args:
        model: A scikit-learn compatible regression model.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary containing the MSLE value.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Ensure predictions and actual values are positive for MSLE calculation
    y_pred = np.maximum(y_pred, 0)
    y_test = np.maximum(y_test, 0)

    # Compute Mean Squared Log Error
    msle_value = mean_squared_log_error(y_test, y_pred)

    return {"mean_squared_log_error": msle_value}

