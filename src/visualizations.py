import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve



def plot_metrics(metrics):
    """
    Plots benchmark metrics as a bar chart.

    Args:
        metrics (dict): Dictionary of benchmark metrics.
    """
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(names, values, color="skyblue")
    plt.title("Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance):
    """
    Plots feature importance as a horizontal bar chart.

    Args:
        feature_importance (dict): Dictionary of feature names and their importance scores.
    """
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color="steelblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plots a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): Class labels for the confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()


def plot_threshold_metrics(threshold_df):
    """
    Plots accuracy, precision, recall, and F1-score against different thresholds.

    Args:
        threshold_df (pd.DataFrame): DataFrame containing threshold evaluation metrics.
    """
    plt.figure(figsize=(10, 6))
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plt.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Model Performance Across Different Thresholds")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plots residuals (errors) of a regression model.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color="purple", alpha=0.5)
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(alpha=0.5)
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plots actual vs predicted values for regression.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color="darkblue", alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle="--", color="red")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.grid(alpha=0.5)
    plt.show()


def plot_class_distribution(y):
    """
    Plots the distribution of class labels in a dataset.

    Args:
        y (array-like): Target labels.
    """
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique.astype(str), counts, color="salmon")
    plt.xlabel("Class Labels")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_correlation_matrix(df):
    """
    Plots a heatmap of feature correlations.

    Args:
        df (pd.DataFrame): DataFrame containing numerical features.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()
    
def plot_roc_curve(y_true, y_scores):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()


def plot_precision_recall_curve(y_true, y_scores):
    """
    Plots the Precision-Recall curve.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted probabilities for the positive class.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.5)
    plt.show()

