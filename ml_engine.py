import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score,
    mean_squared_error, silhouette_score
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


def preprocess(df: pd.DataFrame, target: str = None):
    """Clean and encode dataframe."""
    df = df.copy()

    # Drop high-cardinality or ID-like string columns
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == object:
            if df[col].nunique() > 50:
                df.drop(columns=[col], inplace=True)
            else:
                le = LabelEncoder()
                df[col] = df[col].fillna("missing")
                df[col] = le.fit_transform(df[col].astype(str))

    # Fill numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def run_regression(df: pd.DataFrame, target: str):
    df = preprocess(df, target)

    if target not in df.columns:
        return {"error": f"Target column '{target}' not found after preprocessing."}

    X = df.drop(columns=[target])
    y = df[target]

    if len(X) < 20:
        return {"error": "Not enough rows for regression (need at least 20)."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    all_results = []
    best_r2 = -np.inf
    best_model_name = None
    best_fi = None

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            all_results.append({
                "Model": name,
                "R² Score": round(r2, 4),
                "RMSE": round(rmse, 4)
            })
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                if hasattr(model, "feature_importances_"):
                    best_fi = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, "coef_"):
                    best_fi = dict(zip(X.columns, np.abs(model.coef_)))
        except Exception as e:
            all_results.append({"Model": name, "R² Score": "Error", "RMSE": str(e)})

    result = {
        "best_model": {
            "name": best_model_name,
            "primary_metric_name": "R² Score",
            "primary_metric_value": best_r2
        },
        "all_models": all_results,
        "problem_type": "regression"
    }
    if best_fi:
        result["feature_importance"] = best_fi

    return result


def run_classification(df: pd.DataFrame, target: str):
    df = preprocess(df, target)

    if target not in df.columns:
        return {"error": f"Target column '{target}' not found after preprocessing."}

    # Encode target if needed
    if df[target].dtype == object:
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    X = df.drop(columns=[target])
    y = df[target]

    if len(X) < 20:
        return {"error": "Not enough rows for classification (need at least 20)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }

    all_results = []
    best_f1 = -np.inf
    best_model_name = None
    best_fi = None
    avg = "binary" if y.nunique() == 2 else "weighted"

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average=avg, zero_division=0)
            all_results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1 Score": round(f1, 4)
            })
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                if hasattr(model, "feature_importances_"):
                    best_fi = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    if coef.ndim > 1:
                        coef = np.abs(coef).mean(axis=0)
                    best_fi = dict(zip(X.columns, np.abs(coef)))
        except Exception as e:
            all_results.append({"Model": name, "Accuracy": "Error", "F1 Score": str(e)})

    result = {
        "best_model": {
            "name": best_model_name,
            "primary_metric_name": "F1 Score",
            "primary_metric_value": best_f1
        },
        "all_models": all_results,
        "problem_type": "classification"
    }
    if best_fi:
        result["feature_importance"] = best_fi

    return result


def run_clustering(df: pd.DataFrame):
    df = preprocess(df)
    X = df.select_dtypes(include=[np.number]).dropna()

    if X.shape[0] < 10:
        return {"error": "Not enough numeric rows for clustering (need at least 10)."}
    if X.shape[1] < 1:
        return {"error": "No numeric columns found for clustering."}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try KMeans with k=2..6 and pick best silhouette
    best_score = -np.inf
    best_k = 3
    for k in range(2, min(7, len(X))):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            pass

    # Final KMeans
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km_best.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)

    all_results = [
        {"Model": f"KMeans (k={best_k})", "Silhouette Score": round(km_sil, 4), "Clusters": best_k},
        {"Model": f"Agglomerative (k={best_k})", "Silhouette Score": round(agg_sil, 4), "Clusters": best_k},
    ]

    if km_sil >= agg_sil:
        best_name = f"KMeans (k={best_k})"
        best_sil = km_sil
        best_labels = km_labels
    else:
        best_name = f"Agglomerative (k={best_k})"
        best_sil = agg_sil
        best_labels = agg_labels

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    result = {
        "best_model": {
            "name": best_name,
            "primary_metric_name": "Silhouette Score",
            "primary_metric_value": best_sil
        },
        "all_models": all_results,
        "problem_type": "clustering",
        "cluster_plot_data": {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "labels": best_labels.tolist()
        },
        "n_clusters": best_k
    }

    return result


def run_ml_pipeline(df: pd.DataFrame, target: str, problem_type: str):
    """Main entry point."""
    try:
        if problem_type == "regression":
            if not target:
                return {"error": "Please select a target column for regression."}
            return run_regression(df, target)
        elif problem_type == "classification":
            if not target:
                return {"error": "Please select a target column for classification."}
            return run_classification(df, target)
        elif problem_type == "clustering":
            return run_clustering(df)
        else:
            return {"error": f"Unknown problem type: {problem_type}"}
    except Exception as e:
        return {"error": str(e)}
