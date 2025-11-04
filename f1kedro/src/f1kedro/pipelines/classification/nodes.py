# src/f1kedro/pipelines/classification/nodes.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def prepare_classification_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Prepara features y target para clasificación."""
    df = merged.copy()
    df['target_binary_points'] = (df['points'].astype(float).fillna(0) > 0).astype(int)
    df['is_pole'] = (df['grid'] == 1).astype(int)
    df['race_year'] = df['year'].astype(float)
    df['age_at_race'] = df['age_at_race'].fillna(df['age_at_race'].median())
    features = df[['raceId','driverId','target_binary_points','grid','age_at_race','is_pole','race_year']].copy()
    return features

def train_classification_models(features: pd.DataFrame):
    """Entrena 4 modelos de clasificación y devuelve metrics_df y models_dict"""
    X = features[['grid','age_at_race','is_pole','race_year']].fillna(0)
    y = features['target_binary_points'].astype(int)
    # stratify para mantener proporciones
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}
    results = []

    # Logistic Regression
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    y_pred = log.predict(X_test)
    results.append({
        'model': 'LogisticRegression',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    })
    models['LogisticRegression'] = log

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append({
        'model': 'RandomForestClassifier',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    })
    models['RandomForestClassifier'] = rf

    # SVC
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    results.append({
        'model': 'SVC',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    })
    models['SVC'] = svc

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    results.append({
        'model': 'KNN',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    })
    models['KNN'] = knn

    metrics_df = pd.DataFrame(results)
    return metrics_df, models
