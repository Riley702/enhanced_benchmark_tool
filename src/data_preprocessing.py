"""
Data Preprocessing Module
=========================

This module provides utilities for preprocessing datasets, including handling missing values,
scaling numerical features, and encoding categorical variables.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
