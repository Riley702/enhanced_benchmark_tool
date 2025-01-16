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
