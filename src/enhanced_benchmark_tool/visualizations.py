from __future__ import annotations

from typing import Optional

import matplotlib

# Safe default for non-interactive environments (CI/tests)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve


def plot_metrics(metrics: dict, show: bool = False, save_path: Optional[str] = None) -> None:
    """Plot metric dict as a bar chart."""
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(names, values, color="skyblue")
    plt.title("Model Performance Metrics")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_feature_importance(feature_importance: dict, show: bool = False, save_path: Optional[str] = None) -> None:
    """Plot feature importance as horizontal bar chart."""
    if not feature_importance:
        return
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color="steelblue")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=None, show: bool = False, save_path: Optional[str] = None) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", xticks_rotation=45, ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_correlation_matrix(df, show: bool = False, save_path: Optional[str] = None) -> None:
    """Plot correlation heatmap for numeric columns."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_learning_curve(
    model,
    X,
    y,
    cv: int = 5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring: str = "accuracy",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Plot a learning curve for a model."""
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_mean, label="Training", marker="o")
    plt.plot(sizes, val_mean, label="Validation", marker="o")
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring)
    plt.title("Learning Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
