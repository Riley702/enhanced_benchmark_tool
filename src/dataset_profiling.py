import pandas as pd
import numpy as np
import utils


def profile_dataset(file_path):
    """
    Generates a detailed statistical profile of the dataset.

    Args:
        file_path (str): Path to the CSV file containing dataset.

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


def generate_summary_statistics(df):
    """
    Computes additional summary statistics for numerical features.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary statistics including median, variance, skewness, and kurtosis.
    """
    stats_df = pd.DataFrame()
    numeric_cols = df.select_dtypes(include=["number"])

    stats_df["median"] = numeric_cols.median()
    stats_df["variance"] = numeric_cols.var()
    stats_df["skewness"] = numeric_cols.skew()
    stats_df["kurtosis"] = numeric_cols.kurtosis()

    return stats_df


def detect_duplicate_rows(df):
    """
    Identifies duplicate rows in a dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Duplicate rows in the dataset.
    """
    duplicates = df[df.duplicated()]
    return duplicates


def detect_constant_columns(df):
    """
    Identifies columns with constant values.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        list: List of constant columns.
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    return constant_cols


def compute_feature_cardinality(df, categorical_features):
    """
    Computes the number of unique values for categorical features.

    Args:
        df (pd.DataFrame): Input dataset.
        categorical_features (list): List of categorical feature column names.

    Returns:
        pd.DataFrame: Cardinality of each categorical feature.
    """
    cardinality = {col: df[col].nunique() for col in categorical_features}
    return pd.DataFrame(list(cardinality.items()), columns=["Feature", "Unique Values"])


def detect_anomalous_values(df, numerical_features, threshold=3.0):
    """
    Identifies anomalous values based on standard deviations from the mean.

    Args:
        df (pd.DataFrame): Input dataset.
        numerical_features (list): List of numerical feature column names.
        threshold (float): Number of standard deviations to consider as anomalous.

    Returns:
        dict: Anomalous values for each numerical column.
    """
    anomalies = {}

    for feature in numerical_features:
        mean = df[feature].mean()
        std_dev = df[feature].std()
        lower_bound = mean - (threshold * std_dev)
        upper_bound = mean + (threshold * std_dev)

        anomaly_indices = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index.tolist()
        anomalies[feature] = anomaly_indices

    return anomalies

def analyze_column_types(df):
    """
    Analyzes and summarizes the types of columns in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Counts of numeric, categorical, datetime, and boolean columns.
    """
    types_summary = {
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
        "datetime_columns": len(df.select_dtypes(include=["datetime"]).columns),
        "boolean_columns": len(df.select_dtypes(include=["bool"]).columns),
    }
    return types_summary

def find_high_missing_columns(df, threshold=0.5):
    """
    Identifies columns with missing value percentage higher than the given threshold.

    Args:
        df (pd.DataFrame): Input dataset.
        threshold (float): Proportion threshold (e.g., 0.5 for 50%).

    Returns:
        list: Columns exceeding the missing value threshold.
    """
    missing_ratios = df.isnull().mean()
    high_missing = missing_ratios[missing_ratios > threshold].index.tolist()
    return high_missing

def summarize_top_n_categories(df, column, n=5):
    """
    Summarizes the top N most frequent categories in a categorical column.

    Args:
        df (pd.DataFrame): Input dataset.
        column (str): Categorical column name.
        n (int): Number of top categories to return.

    Returns:
        pd.DataFrame: Top N categories and their frequency and percentage.
    """
    value_counts = df[column].value_counts()
    top_n = value_counts.head(n)
    top_n_percent = (top_n / len(df)) * 100

    return pd.DataFrame({
        "Category": top_n.index,
        "Frequency": top_n.values,
        "Percentage": top_n_percent.values.round(2)
    })


