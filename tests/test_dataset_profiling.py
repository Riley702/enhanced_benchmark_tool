from enhanced_benchmark_tool.dataset_profiling import profile_dataset
from enhanced_benchmark_tool.model_benchmarking import extract_feature_importance
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Example data
X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
y = [0, 1, 0, 1]

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Extract feature importance
feature_importance = extract_feature_importance(model, feature_names=X.columns)
print("Feature Importance:", feature_importance)


def test_profile_dataset():
    """
    Test profile_dataset function with a sample CSV file.
    """
    sample_data = {"A": [1, 2, 3], "B": [4, None, 6]}
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv("sample.csv", index=False)

    profile = profile_dataset("sample.csv")

    assert profile["shape"] == (3, 2)
    assert profile["null_counts"]["B"] == 1
    print("test_profile_dataset passed.")


def test_extract_feature_importance():
    """
    Tests extract_feature_importance function to ensure correct feature extraction and normalization.
    """
    # Define a simple dataset
    X_test = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
    y_test = [0, 1, 0, 1]

    # Train a RandomForest model
    test_model = RandomForestClassifier(random_state=42)
    test_model.fit(X_test, y_test)

    # Extract feature importance
    feature_importance_test = extract_feature_importance(test_model, feature_names=X_test.columns)

    # Assertions to ensure proper extraction
    assert isinstance(feature_importance_test, dict), "Feature importance should be a dictionary."
    assert set(feature_importance_test.keys()) == set(X_test.columns), "Feature names should match dataset columns."
    assert all(0 <= v <= 1 for v in feature_importance_test.values()), "Importance values should be normalized between 0 and 1."

    print("test_extract_feature_importance passed.")


# Run the tests
test_profile_dataset()
test_extract_feature_importance()
