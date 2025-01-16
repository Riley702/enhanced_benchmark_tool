from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def benchmark_model(model, X, y, test_size=0.2, random_state=42):
    """
    Benchmarks a machine learning model.

    Args:
        model: Scikit-learn compatible model.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Benchmark metrics and training/testing times.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    predictions = model.predict(X_test)
    end_test = time.time()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1_score": f1_score(y_test, predictions, average='weighted'),
        "training_time": end_train - start_train,
        "testing_time": end_test - start_test,
    }
    return metrics
