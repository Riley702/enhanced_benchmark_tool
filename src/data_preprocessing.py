"""
Data Preprocessing Module
=========================

This module provides utilities for preprocessing datasets, including handling missing values,
scaling numerical features, encoding categorical variables, detecting outliers, normalizing features,
and reducing categorical feature cardinality.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(df, numerical_features, categorical_features, missing_strategy="mean", scale=True, encode=True):
    """
    Preprocesses the dataset by handling missing values, scaling numerical features,
    and encoding categorical features.

    Args:
        df (pd.DataFrame): The dataset to preprocess.
        numerical_features (list): List of numerical feature column names.
        categorical_features (list): List of categorical feature column names.
        missing_strategy (str): Strategy for imputing missing values ('mean', 'median', or 'most_frequent').
        scale (bool): Whether to scale numerical features.
        encode (bool): Whether to encode categorical features.

    Returns:
        pd.DataFrame: The preprocessed dataset.
        ColumnTransformer: The fitted transformer for future use.
    """
    transformers = []

    # Handle numerical features
    if numerical_features:
        num_transformer = []
        if missing_strategy:
            num_transformer.append(("imputer", SimpleImputer(strategy=missing_strategy)))
        if scale:
            num_transformer.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_transformer), numerical_features))

    # Handle categorical features
    if categorical_features:
        cat_transformer = []
        if missing_strategy:
            cat_transformer.append(("imputer", SimpleImputer(strategy="most_frequent")))
        if encode:
            cat_transformer.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
        transformers.append(("cat", Pipeline(cat_transformer), categorical_features))

    # Create a column transformer
    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    # Apply transformations
    processed_array = preprocessor.fit_transform(df)
    processed_df = pd.DataFrame(processed_array, columns=preprocessor.get_feature_names_out())

    return processed_df, preprocessor


def detect_outliers_iqr(df, numerical_features, threshold=1.5):
    """
    Identifies outliers in numerical features using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): Dataset containing numerical features.
        numerical_features (list): List of numerical feature column names.
        threshold (float): Multiplier for the IQR to determine outliers (default 1.5).

    Returns:
        dict: Dictionary where keys are feature names and values are lists of outlier indices.
    """
    outliers = {}

    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outlier_indices = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index.tolist()
        outliers[feature] = outlier_indices

    return outliers


def normalize_numerical_features(df, numerical_features):
    """
    Normalizes numerical features using Min-Max Scaling.

    Args:
        df (pd.DataFrame): Dataset containing numerical features.
        numerical_features (list): List of numerical feature column names.

    Returns:
        pd.DataFrame: The dataset with normalized numerical features.
    """
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


def reduce_cardinality(df, categorical_features, threshold=0.05, new_category="Other"):
    """
    Reduces the cardinality of categorical features by grouping rare categories into a single category.

    Args:
        df (pd.DataFrame): The dataset containing categorical features.
        categorical_features (list): List of categorical feature column names.
        threshold (float): Minimum proportion required for a category to remain separate (default 0.05).
        new_category (str): The name of the new category for rare values.

    Returns:
        pd.DataFrame: The dataset with reduced categorical cardinality.
    """
    for feature in categorical_features:
        category_counts = df[feature].value_counts(normalize=True)
        rare_categories = category_counts[category_counts < threshold].index
        df[feature] = df[feature].replace(rare_categories, new_category)

    return df
