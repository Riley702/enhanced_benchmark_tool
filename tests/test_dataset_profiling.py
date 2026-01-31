import pandas as pd

from enhanced_benchmark_tool import profile_dataset, extract_feature_importance
from sklearn.ensemble import RandomForestClassifier


def test_profile_dataset(tmp_path):
    sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, None, 6]})
    p = tmp_path / "sample.csv"
    sample_df.to_csv(p, index=False)

    profile = profile_dataset(str(p))

    assert profile["shape"] == (3, 2)
    assert profile["null_counts"]["B"] == 1


def test_extract_feature_importance():
    X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
    y = [0, 1, 0, 1]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    fi = extract_feature_importance(model, feature_names=list(X.columns))

    assert isinstance(fi, dict)
    assert set(fi.keys()) == set(X.columns)
    assert all(0.0 <= v <= 1.0 for v in fi.values())
