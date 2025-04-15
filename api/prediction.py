"""
Property Price Prediction and Recommendation Module

This module provides functionality for:
- Predicting property prices based on input features
- Recommending similar properties using nearest neighbors approach
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from api.utils import (
    create_df,
    convert_marla_to_kanal,
    convert_kanal_to_marla,
    feature_engineering,
    log_transform,
    exp_transform
)
sys.modules['__main__'].log_transform = log_transform
sys.modules['__main__'].exp_transform = exp_transform

def get_price_prediction(input_dict:dict, purpose:str) -> float:
    """
        Predict property price based on input features and purpose (rent/sale)

        Args:
            input_dict (dict): Dictionary containing property features
            purpose (str): Type of prediction - 'rent' or 'sale'

        Returns:
            float: Predicted price rounded to nearest whole number

        Raises:
            ValueError: If purpose is not 'rent' or 'sale'
        """
    purpose = purpose.lower()
    if purpose not in ('rent', 'sale'):
        raise ValueError("Purpose must be either 'rent' or 'sale'")

    input_df = create_df(input_dict)
    input_df = convert_marla_to_kanal(input_df)
    X = feature_engineering(input_df)

    model_path= os.path.join(BASE_DIR, 'artifacts', 'models', f'{purpose}_predictor_pipeline.pkl')
    model = joblib.load(model_path)
    price = model.predict(X)[0]
    multiplier = 100000 if purpose == 'rent' else 10000000
    return round(price * multiplier)


def recommend_properties(input_df, purpose:str) -> pd.DataFrame:
    """
        Recommend similar properties based on input features

        Args:
            input_df (pd.DataFrame): Input property features
            purpose (str): Type of recommendation - 'rent' or 'sale'

        Returns:
            pd.DataFrame: DataFrame with recommended properties

        Raises:
            ValueError: If purpose is not 'rent' or 'sale'
        """
    purpose = purpose.lower()
    if purpose not in ('rent', 'sale'):
        raise ValueError("Purpose must be either 'rent' or 'sale'")

    assets_path = os.path.join(BASE_DIR, 'artifacts', 'models', f'{purpose}_recommender.pkl')
    assets = joblib.load(assets_path)
    pipeline = assets['pipeline']
    original_data = assets['original_data']
    feature_names = assets['feature_names']

    try:
        # Process input and find neighbors
        processed_input = pipeline['preprocessor'].transform(input_df)
        distances, indices = pipeline['nn'].kneighbors(processed_input)
        # Cluster results for diverse recommendations
        nbrs_idx = indices[0]
        neighbors_df = original_data.iloc[nbrs_idx]
        nbrs_feat = original_data.iloc[nbrs_idx][feature_names]
        proc_nbrs = pipeline['preprocessor'].transform(nbrs_feat)


        n_clusters = min(3, proc_nbrs.shape[0])
        if n_clusters < 3:
            return neighbors_df
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(proc_nbrs)
        centroids = km.cluster_centers_
        labels = km.labels_
        picks = []
        for c in range(n_clusters):
            in_cluster = np.where(labels == c)[0]
            if in_cluster.size > 0:
                dists = np.linalg.norm(proc_nbrs[in_cluster] - centroids[c], axis=1)
                picks.append(in_cluster[np.argmin(dists)])
        res = neighbors_df.iloc[picks].copy()
        return convert_kanal_to_marla(res)

    except Exception as e:
        print(f"Recommendation failed: {e}")
        return original_data.sample(3)


