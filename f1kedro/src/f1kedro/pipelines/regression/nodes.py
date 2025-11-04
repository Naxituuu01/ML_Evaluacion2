# src/f1kedro/pipelines/regression/nodes.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def prepare_regression_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Select or create features for regression and return features DataFrame."""
    df = merged.copy()
    # target
    df['target_points'] = df['points'].astype(float).fillna(0)
    # simple features — puedes mejorar más adelante
    df['is_pole'] = (df['grid'] == 1).astype(int)
    df['race_year'] = df['year'].astype(float)
    # features selection (ajusta según EDA)
    features = df[['raceId','driverId','target_points','grid','age_at_race','is_pole','race_year']].copy()
    # fillna
    features['age_at_race'] = features['age_at_race'].fillna(features['age_at_race'].median())
    return features

def train_regression_models(features: pd.DataFrame):
    """
    Entrena dos modelos de regresión y devuelve:
    - metrics_df (DataFrame con métricas)
    - models_dict (diccionario con modelos para guardar)
    """
    # prepare X,y
    X = features[['grid','age_at_race','is_pole','race_year']].fillna(0)
    y = features['target_points'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    results = []

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results.append({
    'model': 'LinearRegression',
    'r2': float(r2_score(y_test, y_pred)),
    'rmse': float(mean_squared_error(y_test, y_pred) ** 0.5),
    'mae': float(mean_absolute_error(y_test, y_pred))
    })
    models['LinearRegression'] = lr

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append({
        'model': 'RandomForestRegressor',
        'r2': float(r2_score(y_test, y_pred)),
        'rmse': float(mean_squared_error(y_test, y_pred) ** 0.5),
        'mae': float(mean_absolute_error(y_test, y_pred))
    })
    models['RandomForestRegressor'] = rf

    metrics_df = pd.DataFrame(results)
    return metrics_df, models
