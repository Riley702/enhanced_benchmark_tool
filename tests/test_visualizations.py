from enhanced_benchmark_tool.visualizations import plot_metrics, plot_feature_importance, plot_confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix

def test_plot_metrics():
    """
    Tests the plot_metrics function by providing sample metric values.
    """
    metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.93, "f1_score": 0.94}
    plot_metrics(metrics)
    print("test_plot_metrics passed.")


def test_plot_feature_importance():
    """
    Tests the plot_feature_importance function by providing sample feature importances.
    """
    feature_importance = {
        "Feature A": 0.3,
        "Feature B": 0.2,
        "Feature C": 0.1,
        "Feature D": 0.4,
    }
    plot_feature_importance(feature_importance)
    print("test_plot_feature_importance passed.")


def test_plot_confusion_matrix():
    """
    Tests the plot_confusion_matrix function using a sample confusion matrix.
    """
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, labels=["Class 0", "Class 1"])
    print("test_plot_confusion_matrix passed.")
