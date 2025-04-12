"""
Utility Functions for Property Price Prediction

This module contains helper functions for:
- Data preprocessing
- Feature engineering
- Data conversion between measurement units
"""
import joblib
import numpy as np
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_df(input_dict: dict) -> pd.DataFrame:
    """
    Create pandas DataFrame from input dictionary

    Args:
        input_dict (dict): Input data as dictionary

    Returns:
        pd.DataFrame: DataFrame with single row containing input data
    """
    return pd.DataFrame(input_dict, index=[0])


def convert_marla_to_kanal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert area measurement from Marla to Kanal (1 Kanal = 20 Marla)

    Args:
        df (pd.DataFrame): Input DataFrame with 'area' column

    Returns:
        pd.DataFrame: DataFrame with converted area values
    """
    df = df.copy()
    df['area'] = df['area'] / 20
    return df


def convert_kanal_to_marla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert area measurement from Kanal to Marla

    Args:
        df (pd.DataFrame): Input DataFrame with 'area' column

    Returns:
        pd.DataFrame: DataFrame with converted area values
    """
    df = df.copy()
    df['area'] = df['area'] * 20
    return df

def feature_engineering(X):
    """
    Perform feature engineering on input DataFrame

    Args:
        X (pd.DataFrame): Raw input features

    Returns:
        pd.DataFrame: Engineered features for model prediction

    Processing Steps:
        1. Log-transform area
        2. Add location tiers
        3. Calculate area-per-room ratio
    """
    X = X.copy()
    location_tiers_path = os.path.join(BASE_DIR, 'artifacts', 'models', 'location_tiers.pkl')
    location_tiers = joblib.load(location_tiers_path)
    X['area'] = X.area.apply(np.log1p)
    X["location_tier"] = X["location"].map(location_tiers).fillna(4)
    X['area_room_ratio'] = X['area'] / (X['bedrooms'] + X['baths'])

    return X


def log_transform(y: np.ndarray) -> np.ndarray:
    """
    Apply natural logarithm transformation with smoothing (log1p)

    Args:
        y (np.ndarray): Array of numerical values to transform

    Returns:
        np.ndarray: Transformed values using log(1 + y)

    Example:
        >>> log_transform(np.array([0, 1, 10]))
        array([0.        , 0.69314718, 2.39789527])
    """
    return np.log1p(y)


def exp_transform(y: np.ndarray) -> np.ndarray:
    """
    Apply exponential transformation to reverse log1p (expm1)

    Args:
        y (np.ndarray): Array of log-transformed values

    Returns:
        np.ndarray: Original-scale values using exp(y) - 1

    Example:
        >>> exp_transform(np.array([0, 1, 2]))
        array([ 0.        ,  1.71828183,  6.3890561 ])
    """
    return np.expm1(y)


def make_recommendation_df(input_df: pd.DataFrame, price: float) -> pd.DataFrame:
    """
    Insert price column into dataframe after property_type column

    Args:
        input_df (pd.DataFrame): Original property dataframe
        price (float): Predicted price to insert

    Returns:
        pd.DataFrame: Modified dataframe with price column

    Example:
        >>> df = pd.DataFrame({'property_type': ['house'], 'area': [200]})
        >>> make_recommendation_df(df, 5000000)
           property_type   price  area
        0         house  5000000   200
    """
    df = input_df.copy()
    insert_at = df.columns.get_loc('property_type') + 1
    df.insert(insert_at, 'price', [price])
    return df
