import pandas as pd
import utils

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
        "summary": data.describe(include="all").to_dict(),
    }
    return profile


def detect_missing_values(df):
    """
    Identifies missing values in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Table showing count and percentage of missing values for each column.
    """
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    missing_df = pd.DataFrame(
        {"missing_count": missing_values, "missing_percent": missing_percent}
    )
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        by="missing_percent", ascending=False
    )

    return missing_df


def detect_outliers(df, threshold=1.5):
    """
    Identifies outliers in numerical columns using the IQR method.

    Args:
        df (pd.DataFrame): Input dataset.
        threshold (float): Threshold for determining outliers (default 1.5).

    Returns:
        dict: Number of outliers detected per numerical column.
    """
    outlier_counts = {}
    for column in df.select_dtypes(include=["number"]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)

        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_counts[column] = outlier_count

    return outlier_counts


def check_class_balance(df, target_column):
    """
    Checks the balance of classes in a categorical target variable.

    Args:
        df (pd.DataFrame): Input dataset.
        target_column (str): Column name of the target variable.

    Returns:
        pd.DataFrame: Table showing class distribution and percentage.
    """
    class_counts = df[target_column].value_counts()
    class_percent = (class_counts / len(df)) * 100

    balance_df = pd.DataFrame(
        {"class_count": class_counts, "class_percent": class_percent}
    )

    return balance_df


def identify_correlations(df, threshold=0.8):
    """
    Identifies highly correlated features in a dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        threshold (float): Correlation threshold to flag variables (default 0.8).

    Returns:
        list: Pairs of highly correlated features.
    """
    corr_matrix = df.corr()
    correlated_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlated_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

    return correlated_pairs



