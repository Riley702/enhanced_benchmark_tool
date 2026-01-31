import numpy as np
from sklearn.metrics import confusion_matrix

from enhanced_benchmark_tool.visualizations import (
    plot_metrics,
    plot_feature_importance,
    plot_confusion_matrix,
)


def test_plot_metrics():
    metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.93, "f1_score": 0.94}
    plot_metrics(metrics, show=False)


def test_plot_feature_importance():
    feature_importance = {"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.4}
    plot_feature_importance(feature_importance, show=False)


def test_plot_confusion_matrix():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    plot_confusion_matrix(y_true, y_pred, labels=["Class 0", "Class 1"], show=False)
