from enhanced_benchmark_tool.dataset_profiling import profile_dataset
from sklearn.ensemble import RandomForestClassifier

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
    sample_data = {"A": [1, 2, 3], "B": [4, None, 6]}
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv("sample.csv", index=False)

    profile = profile_dataset("sample.csv")

    assert profile["shape"] == (3, 2)
    assert profile["null_counts"]["B"] == 1
    print("test_profile_dataset passed.")
