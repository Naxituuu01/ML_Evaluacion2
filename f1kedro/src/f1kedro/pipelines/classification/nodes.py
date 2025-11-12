import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

RANDOM_STATE = 42


# =====================================================
# =============== PREPARACIÓN DE DATOS =================
# =====================================================
def prepare_classification_data(df: pd.DataFrame):
    df = df.copy()

    df['target_binary_points'] = (df['points'].astype(float).fillna(0) > 0).astype(int)
    df['is_pole'] = (df['grid'] == 1).astype(int)
    df['race_year'] = df['year'].astype(float)
    df['age_at_race'] = df['age_at_race'].fillna(df['age_at_race'].median())

    X = df[['grid', 'age_at_race', 'is_pole', 'race_year']].fillna(0)
    y = df['target_binary_points'].astype(int)

    print(f"[prepare_classification_data] Shape: {X.shape}")
    print(f"[prepare_classification_data] Positive class ratio: {y.mean():.3f}")

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


# =====================================================
# =============== ENTRENAMIENTO DE MODELOS =============
# =====================================================
def train_classification_models(data):
    X_train, X_test, y_train, y_test = data

    print(f"[DEBUG] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[DEBUG] Positive class ratio: {y_train.mean():.3f}")

    # Balanceo de clases
    if y_train.mean() < 0.1:
        print("[INFO] Usando SMOTEENN (clases muy desbalanceadas)")
        sampling_method = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE, k_neighbors=3),
            random_state=RANDOM_STATE
        )
    else:
        sampling_method = SMOTE(random_state=RANDOM_STATE)

    # Preprocesamiento
    numeric_features = X_train.columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])

    # Modelos a entrenar
    estimators = {
        "random_forest": RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-3, random_state=RANDOM_STATE)
    }

    if LGBMClassifier is not None:
        estimators["lightgbm"] = LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1
        )

    if XGBClassifier is not None:
        estimators["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)

    # Espacios de búsqueda
    param_distributions = {
        "random_forest": {"model__n_estimators": [200, 300], "model__max_depth": [5, 10, None]},
        "logistic": {"model__C": [0.01, 0.1, 1]},
        "gradient_boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
        "svm": {"model__alpha": [1e-4, 1e-3]},
        "knn": {"model__n_neighbors": [3, 5, 7]}
    }

    results = []
    model_dir = Path("data/06_models/classification")
    model_dir.mkdir(parents=True, exist_ok=True)

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    for name, estimator in estimators.items():
        print(f"\n🔧 Entrenando modelo: {name}")

        try:
            pipe = ImbPipeline([
                ("pre", preprocessor),
                ("smote", sampling_method),
                ("model", estimator)
            ])

            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_distributions.get(name, {}),
                n_iter=5,
                cv=3,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )

            rs.fit(X_train, y_train)
            best = rs.best_estimator_

            y_pred_proba = best.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold = thresholds[np.argmax(f1_scores[:-1])]
            y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

            acc = accuracy_score(y_test, y_pred_optimized)
            f1 = f1_score(y_test, y_pred_optimized)
            prec = precision_score(y_test, y_pred_optimized, zero_division=0)
            rec = recall_score(y_test, y_pred_optimized)
            auc = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

            results.append({
                "Model": name,
                "Accuracy": float(acc),
                "F1_Score": float(f1),
                "Precision": float(prec),
                "Recall": float(rec),
                "AUC_ROC": float(auc),
                "Best_Threshold": float(best_threshold),
                "CV_AUC_Mean": float(cv_scores.mean()),
                "CV_AUC_Std": float(cv_scores.std())
            })

            joblib.dump(best, model_dir / f"{name}.pkl")
            print(f"[{name}] ✅ AUC={auc:.4f}, F1={f1:.4f}")

        except Exception as e:
            print(f"❌ Error entrenando {name}: {str(e)}")

    # === Guardar resultados como .parquet ===
    results_df = pd.DataFrame(results).sort_values("AUC_ROC", ascending=False)
    parquet_output = Path("data/07_model_output/classification/classification_results.parquet")
    parquet_output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(parquet_output, index=False)
    print(f"\n💾 Resultados guardados en: {parquet_output}")

    # === Guardar métricas planas para DVC ===
    metrics_output = Path("data/08_reporting/classification_metrics_flat.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        row["Model"]: {
            "Accuracy": row["Accuracy"],
            "F1_Score": row["F1_Score"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "AUC_ROC": row["AUC_ROC"]
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

    print(f"\n✅ Métricas guardadas en: {metrics_output}")
    return results_df