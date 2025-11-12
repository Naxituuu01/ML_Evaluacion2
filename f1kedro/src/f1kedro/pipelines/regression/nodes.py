# src/f1kedro/pipelines/regression/nodes.py
import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

RANDOM_STATE = 42


# =====================================================
# =============== PREPARACIÓN DE DATOS =================
# =====================================================
def prepare_regression_data(df: pd.DataFrame):
    df = df.copy()

    df['target_points'] = df['points'].astype(float).fillna(0)
    df['is_pole'] = (df['grid'] == 1).astype(int)
    df['race_year'] = df['year'].astype(float)
    df['age_at_race'] = df['age_at_race'].fillna(df['age_at_race'].median())

    X = df[['grid', 'age_at_race', 'is_pole', 'race_year']].fillna(0)
    y = df['target_points'].astype(float)

    print(f"[prepare_regression_data] Shape: {X.shape}")
    print(f"[prepare_regression_data] Target mean: {y.mean():.3f}")

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# =====================================================
# =============== ENTRENAMIENTO DE MODELOS =============
# =====================================================
# =====================================================
# =============== ENTRENAMIENTO DE MODELOS =============
# =====================================================
def train_regression_models(data):
    X_train, X_test, y_train, y_test = data

    print(f"[DEBUG] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[DEBUG] Target mean: {y_train.mean():.3f}")

    # Preprocesamiento numérico
    numeric_features = X_train.columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])

    # Modelos a entrenar
    estimators = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=RANDOM_STATE),
        "lasso": Lasso(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
    }

    if LGBMRegressor is not None:
        estimators["lightgbm"] = LGBMRegressor(objective="regression", random_state=RANDOM_STATE)

    if XGBRegressor is not None:
        estimators["xgboost"] = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE)

    # Espacios de búsqueda
    param_distributions = {
        "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
        "lasso": {"model__alpha": [0.01, 0.1, 1.0]},
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None]
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1]
        },
        "svr": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    }

    results = []
    model_dir = Path("data/06_models/regression")
    model_dir.mkdir(parents=True, exist_ok=True)

    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    for name, estimator in estimators.items():
        print(f"\n🔧 Entrenando modelo: {name}")

        try:
            pipe = Pipeline([
                ("pre", preprocessor),
                ("model", estimator)
            ])

            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_distributions.get(name, {}),
                n_iter=5,
                cv=3,
                scoring="r2",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )

            rs.fit(X_train, y_train)
            best = rs.best_estimator_

            y_pred = best.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

            results.append({
                "Model": name,
                "R2_Score": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "CV_R2_Mean": float(cv_scores.mean()),
                "CV_R2_Std": float(cv_scores.std())
            })

            joblib.dump(best, model_dir / f"{name}.pkl")

        except Exception as e:
            print(f"❌ Error entrenando {name}: {str(e)}")

    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False)

    # ================== GUARDAR RESULTADOS ==================
    output_path = Path("data/07_model_output/regression/regression_results.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    print(f"\n✅ Resultados guardados en: {output_path}")

    # ================== GUARDAR MÉTRICAS PARA DVC ==================
    metrics_output = Path("data/08_reporting/regression_metrics_flat.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        row["Model"]: {
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "R2_Score": row["R2_Score"],
            "CV_R2_Mean": row["CV_R2_Mean"]
        }
        for _, row in results_df.iterrows()
    }

    rounded_metrics = {
        model: {
            k: round(v, 2) if isinstance(v, (float, int)) else v
            for k, v in sorted(metrics.items())
        }
        for model, metrics in sorted(metrics_dict.items())
    }

    with open(metrics_output, "w") as f:
        json.dump(rounded_metrics, f, indent=4, sort_keys=True)

    print(f"\n✅ Métricas planas guardadas en: {metrics_output}")

    return results_df