from enhanced_benchmark_tool.dataset_profiling import profile_dataset

def test_profile_dataset():
    sample_data = {"A": [1, 2, 3], "B": [4, None, 6]}
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv("sample.csv", index=False)

    profile = profile_dataset("sample.csv")

    assert profile["shape"] == (3, 2)
    assert profile["null_counts"]["B"] == 1
    print("test_profile_dataset passed.")
