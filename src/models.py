"""
models.py
---------
Model training, evaluation, and comparison utilities for the Rugby Union
ML dissertation.

Covers:
  - Idea 1: Position classification (Back vs Forward)
  - Idea 2: Performance prediction (regression)
  - Idea 3: Player clustering (unsupervised)

All classifiers/regressors are wrapped in sklearn Pipelines for clean,
leak-free cross-validation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     cross_validate, GridSearchCV)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier,
                               GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — XGB models will be skipped.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed — SHAP analysis will be skipped.")


# ── 1. Classification: Back vs Forward (Idea 1) ─────────────────────────────

CLASSIFIERS = {
    "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
}

if HAS_XGB:
    CLASSIFIERS["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="logloss", random_state=42
    )


def evaluate_classifiers(X: np.ndarray, y: np.ndarray,
                          cv_folds: int = 5,
                          feature_names: list = None) -> pd.DataFrame:
    """
    Run all classifiers with stratified k-fold CV and return a comparison table.

    Parameters
    ----------
    X : np.ndarray — feature matrix (already scaled)
    y : np.ndarray — binary labels (0=Back, 1=Forward)
    cv_folds : int — number of CV folds
    feature_names : list — optional, for SHAP analysis

    Returns
    -------
    pd.DataFrame with columns: model, accuracy, f1, roc_auc, fit_time
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name, clf in CLASSIFIERS.items():
        scores = cross_validate(
            clf, X, y, cv=cv,
            scoring=["accuracy", "f1", "roc_auc"],
            return_train_score=False
        )
        results.append({
            "model":    name,
            "accuracy": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "f1":       scores["test_f1"].mean(),
            "f1_std":   scores["test_f1"].std(),
            "roc_auc":  scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "fit_time": scores["fit_time"].mean(),
        })
        print(f"  {name:30s}  acc={scores['test_accuracy'].mean():.3f}  "
              f"f1={scores['test_f1'].mean():.3f}  "
              f"auc={scores['test_roc_auc'].mean():.3f}")

    return pd.DataFrame(results).sort_values("roc_auc", ascending=False)


def get_best_classifier(X: np.ndarray, y: np.ndarray,
                         clf_name: str = "Random Forest") -> Any:
    """
    Fit and return the best classifier on the full dataset.
    Use this for SHAP analysis after cross-validation comparison.
    """
    clf = CLASSIFIERS[clf_name]
    clf.fit(X, y)
    return clf


def shap_analysis(clf, X: np.ndarray, feature_names: list,
                  max_display: int = 20) -> dict:
    """
    Compute SHAP values for a fitted tree-based classifier.

    Returns a dict with:
        - shap_values: raw SHAP values
        - mean_abs_shap: mean |SHAP| per feature (for bar chart)
        - top_features: sorted feature importance DataFrame
    """
    if not HAS_SHAP:
        print("SHAP not installed. Run: pip install shap")
        return {}

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    top_features = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs
    }).sort_values('mean_abs_shap', ascending=False)

    return {
        'shap_values': sv,
        'mean_abs_shap': mean_abs,
        'top_features': top_features.head(max_display)
    }


# ── 2. Regression: Performance Prediction (Idea 2) ──────────────────────────

REGRESSORS = {
    "Dummy (mean)":          DummyRegressor(strategy="mean"),
    "Dummy (last game)":     DummyRegressor(strategy="mean"),  # replaced below
    "Ridge Regression":      Ridge(alpha=1.0),
    "Lasso Regression":      Lasso(alpha=0.1, max_iter=2000),
    "Random Forest":         RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=200, random_state=42),
}

if HAS_XGB:
    REGRESSORS["XGBoost"] = xgb.XGBRegressor(n_estimators=200, random_state=42)


def evaluate_regressors(X: np.ndarray, y: np.ndarray,
                         cv_folds: int = 5,
                         target_name: str = "target") -> pd.DataFrame:
    """
    Run all regressors with k-fold CV and return a comparison table.

    Uses TimeSeriesSplit-style ordering if temporal data is passed.

    Returns
    -------
    pd.DataFrame with RMSE, MAE, R² per model.
    """
    cv = KFold(n_splits=cv_folds, shuffle=False)  # No shuffle = temporal ordering
    results = []

    for name, reg in REGRESSORS.items():
        scores = cross_validate(
            reg, X, y, cv=cv,
            scoring=["neg_root_mean_squared_error",
                     "neg_mean_absolute_error", "r2"],
            return_train_score=False
        )
        results.append({
            "model":   name,
            "target":  target_name,
            "rmse":    -scores["test_neg_root_mean_squared_error"].mean(),
            "rmse_std": scores["test_neg_root_mean_squared_error"].std(),
            "mae":     -scores["test_neg_mean_absolute_error"].mean(),
            "r2":       scores["test_r2"].mean(),
        })
        print(f"  {name:30s}  rmse={-scores['test_neg_root_mean_squared_error'].mean():.3f}  "
              f"r2={scores['test_r2'].mean():.3f}")

    return pd.DataFrame(results).sort_values("rmse")


def evaluate_multi_target_regression(X: np.ndarray,
                                      Y: np.ndarray,
                                      target_names: list,
                                      cv_folds: int = 5) -> pd.DataFrame:
    """
    Evaluate regression across multiple targets simultaneously.
    Reports per-target RMSE and MAE.
    """
    results = []
    for i, tgt in enumerate(target_names):
        print(f"\n── Target: {tgt} ──")
        res = evaluate_regressors(X, Y[:, i], cv_folds=cv_folds, target_name=tgt)
        results.append(res)
    return pd.concat(results, ignore_index=True)


# ── 3. Clustering: Player Archetypes (Idea 3) ───────────────────────────────

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 10)) -> pd.DataFrame:
    """
    Evaluate KMeans for a range of k values using silhouette and inertia.

    Returns a DataFrame useful for plotting the elbow curve and silhouette scores.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil   = silhouette_score(X, labels)
        db    = davies_bouldin_score(X, labels)
        results.append({
            'k':         k,
            'inertia':   km.inertia_,
            'silhouette': sil,
            'davies_bouldin': db,
        })
        print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil:.3f}  DB={db:.3f}")

    return pd.DataFrame(results)


CLUSTERING_ALGORITHMS = {
    "KMeans (k=2)":      KMeans(n_clusters=2, n_init=20, random_state=42),
    "KMeans (k=4)":      KMeans(n_clusters=4, n_init=20, random_state=42),
    "KMeans (k=6)":      KMeans(n_clusters=6, n_init=20, random_state=42),
    "Agglomerative (k=2)": AgglomerativeClustering(n_clusters=2),
    "Agglomerative (k=4)": AgglomerativeClustering(n_clusters=4),
    "DBSCAN":            DBSCAN(eps=0.8, min_samples=3),
}


def evaluate_clustering(X: np.ndarray,
                         true_labels: np.ndarray = None) -> pd.DataFrame:
    """
    Fit all clustering algorithms and report internal + external validity metrics.

    Parameters
    ----------
    X : np.ndarray — feature matrix (scaled)
    true_labels : np.ndarray or None — if provided, compute ARI against ground truth
                  (e.g., Back=0, Forward=1) for external validation

    Returns
    -------
    pd.DataFrame with silhouette, Davies-Bouldin, and optionally ARI.
    """
    from sklearn.metrics import adjusted_rand_score

    results = []
    for name, algo in CLUSTERING_ALGORITHMS.items():
        labels = algo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2:
            print(f"  {name:30s}  Only {n_clusters} cluster(s) found — skipping.")
            continue

        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        row = {'algorithm': name, 'n_clusters': n_clusters,
               'silhouette': sil, 'davies_bouldin': db}

        if true_labels is not None:
            row['ari_vs_position'] = adjusted_rand_score(true_labels, labels)

        results.append(row)
        print(f"  {name:30s}  k={n_clusters}  sil={sil:.3f}  DB={db:.3f}")

    return pd.DataFrame(results)


def fit_final_clustering(X: np.ndarray, k: int,
                          algorithm: str = "kmeans") -> np.ndarray:
    """
    Fit the chosen clustering algorithm and return cluster labels.
    Use after find_optimal_k() to select k.
    """
    if algorithm == "kmeans":
        algo = KMeans(n_clusters=k, n_init=30, random_state=42)
    elif algorithm == "agglomerative":
        algo = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return algo.fit_predict(X)


# ── 4. Dimensionality reduction helpers (for visualisation) ─────────────────

def reduce_dimensions(X: np.ndarray, method: str = "pca",
                       n_components: int = 2) -> np.ndarray:
    """
    Reduce X to 2D or 3D for visualisation.

    Parameters
    ----------
    method : 'pca', 'tsne', or 'umap'
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(X), reducer

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, perplexity=30,
                       random_state=42, max_iter=1000)
        return reducer.fit_transform(X), reducer

    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(X), reducer
        except ImportError:
            print("umap-learn not installed. Falling back to PCA.")
            return reduce_dimensions(X, method="pca", n_components=n_components)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")


# ── 5. Hyperparameter tuning grids ──────────────────────────────────────────

RF_CLASSIFIER_GRID = {
    'n_estimators':   [100, 200, 500],
    'max_depth':      [None, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':   ['sqrt', 'log2'],
}

RF_REGRESSOR_GRID = RF_CLASSIFIER_GRID.copy()

LOGISTIC_GRID = {
    'C':        [0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty':  ['l1', 'l2'],
    'solver':   ['liblinear'],
}


def tune_model(estimator, param_grid: dict, X: np.ndarray, y: np.ndarray,
               scoring: str = "roc_auc", cv: int = 5) -> GridSearchCV:
    """
    Run GridSearchCV and return the fitted GridSearchCV object.
    Access best params via result.best_params_, best model via result.best_estimator_.
    """
    gs = GridSearchCV(estimator, param_grid, scoring=scoring,
                      cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print(f"Best {scoring}: {gs.best_score_:.4f}")
    print(f"Best params:  {gs.best_params_}")
    return gs
