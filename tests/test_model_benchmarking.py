from sklearn.tree import DecisionTreeClassifier
from enhanced_benchmark_tool.model_benchmarking import benchmark_model
import pandas as pd

def test_benchmark_model():
    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "target": [0, 1, 0, 1]})
    X = data[["A", "B"]]
    y = data["target"]

    model = DecisionTreeClassifier()
    metrics = benchmark_model(model, X, y)

    assert "accuracy" in metrics
    assert "training_time" in metrics
    print("test_benchmark_model passed.")

def test_benchmark_model_imbalanced():
    """
    Test the benchmark_model function with an imbalanced dataset.
    """
    # Create an imbalanced dataset
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [5, 6, 7, 8, 9, 10],
        "target": [0, 0, 0, 0, 1, 1],  # Imbalance: more 0s than 1s
    })
    X = data[["A", "B"]]
    y = data["target"]

    # Initialize a DecisionTreeClassifier
    model = DecisionTreeClassifier()

    # Benchmark the model
    metrics = benchmark_model(model, X, y)

    # Assertions
    assert "accuracy" in metrics, "Accuracy metric is missing in the output."
    assert "training_time" in metrics, "Training time metric is missing in the output."
    assert metrics["accuracy"] > 0, "Accuracy should be greater than zero for a valid benchmark."

    print("test_benchmark_model_imbalanced passed.")
