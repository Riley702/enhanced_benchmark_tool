import pandas as pd
import utils


def profile_dataset(file_path):
    """
    Generates a detailed statistical profile of the dataset.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        dict: Statistical summary including column info and null counts.
    """
    data = pd.read_csv(file_path)
    profile = {
        "shape": data.shape,
        "columns": list(data.columns),
        "null_counts": data.isnull().sum().to_dict(),
        "data_types": data.dtypes.to_dict(),
        "summary": data.describe(include='all').to_dict()
    }
    return profile
