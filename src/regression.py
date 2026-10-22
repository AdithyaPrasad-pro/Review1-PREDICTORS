"""
Group 12 Machine Learning Capstone
Track 1: Diamond Price Prediction (Regression)

Algorithms required by rubric:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Polynomial Regression (Degree 2)
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor
10. K-Nearest Neighbors Regressor
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.preprocessing import (
    engineer_diamond_features,
    DiamondIQRBoundsClipper,
    build_diamond_preprocessor,
    DIAMOND_CLIP_COLS,
    DIAMOND_NUMERIC_FEATURES,
    DIAMOND_CATEGORICAL_FEATURES
)

RANDOM_STATE = 42

# Official benchmark results verified from notebooks/regression.ipynb
VERIFIED_REGRESSION_BENCHMARKS = [
    {"Rank": 1, "Model": "Random Forest Regressor", "R2": 0.9778, "RMSE": 595.84, "MAE": 308.91, "Status": "Selected Best"},
    {"Rank": 2, "Model": "Polynomial Regression (Degree 2)", "R2": 0.9725, "RMSE": 662.58, "MAE": 392.57, "Status": "Top 2 Runner-up"},
    {"Rank": 3, "Model": "KNN Regressor", "R2": 0.9651, "RMSE": 746.78, "MAE": 376.49, "Status": "Evaluated"},
    {"Rank": 4, "Model": "Decision Tree Regressor", "R2": 0.9640, "RMSE": 758.25, "MAE": 403.71, "Status": "Evaluated"},
    {"Rank": 5, "Model": "Gradient Boosting Regressor", "R2": 0.9482, "RMSE": 909.35, "MAE": 509.13, "Status": "Evaluated"},
    {"Rank": 6, "Model": "Linear Regression", "R2": 0.9280, "RMSE": 1072.26, "MAE": 708.10, "Status": "Baseline"},
    {"Rank": 7, "Model": "Ridge Regression", "R2": 0.9280, "RMSE": 1072.28, "MAE": 708.02, "Status": "Evaluated"},
    {"Rank": 8, "Model": "Lasso Regression", "R2": 0.9280, "RMSE": 1072.29, "MAE": 707.72, "Status": "Evaluated"},
    {"Rank": 9, "Model": "Support Vector Regressor", "R2": 0.9116, "RMSE": 1188.16, "MAE": 629.12, "Status": "Evaluated"},
    {"Rank": 10, "Model": "ElasticNet Regression", "R2": 0.8961, "RMSE": 1288.10, "MAE": 861.20, "Status": "Evaluated"},
]

VERIFIED_REGRESSION_CV = [
    {"Model": "Random Forest Regressor", "Mean CV R2": 0.9774, "Std CV R2": 0.0004, "Folds": 5, "Selection": "Best Model"},
    {"Model": "Polynomial Regression (Degree 2)", "Mean CV R2": 0.9707, "Std CV R2": 0.0025, "Folds": 5, "Selection": "Second Best"},
]

VERIFIED_REGRESSION_TUNING = [
    {
        "Model": "Random Forest Regressor",
        "Tuned Parameters": "max_depth=None, min_samples_leaf=1, n_estimators=200",
        "Baseline Test R2": 0.9778,
        "Tuned Test R2": 0.9805,
        "Test Improvement": "+0.0027",
        "Baseline 3-Fold CV R2": 0.9769,
        "Tuned 3-Fold CV R2": 0.9798,
        "CV Improvement": "+0.0029"
    },
    {
        "Model": "Ridge Regression",
        "Tuned Parameters": "alpha=10 (vs baseline alpha=1.0)",
        "Baseline Test R2": 0.928030,
        "Tuned Test R2": 0.927986,
        "Test Improvement": "-0.000045",
        "Baseline 3-Fold CV R2": 0.927211,
        "Tuned 3-Fold CV R2": 0.927213,
        "CV Improvement": "+0.000002"
    }
]


def get_all_10_regression_models():
    """Returns the 10 required regression algorithms configured per notebook specifications."""
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Polynomial Regression (Degree 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=1.0))
        ]),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE),
        "Support Vector Regressor": SVR(C=10.0, kernel="rbf"),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE)
    }


def prepare_diamond_splits(df: pd.DataFrame, target_col="Price(in US dollars)"):
    """
    Cleans duplicates, creates Volume, fits IQR bounds on train set, and prepares stratified splits.
    """
    df_clean = df.drop_duplicates().copy()
    df_clean = engineer_diamond_features(df_clean)

    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = pd.to_numeric(df_clean[target_col], errors="coerce")
    valid_idx = y.notna()
    df_clean = df_clean.loc[valid_idx]
    y = y.loc[valid_idx]
    X = df_clean.drop(columns=[target_col])

    # Stratify by price decile bins as done in notebook
    price_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=price_bins
    )

    # Apply IQR clipping bounds fitted strictly on training data
    clipper = DiamondIQRBoundsClipper(columns=DIAMOND_CLIP_COLS)
    clipper.fit(X_train)
    X_train_clipped = clipper.transform(X_train)
    X_test_clipped = clipper.transform(X_test)

    # Preprocessor
    num_cols = [c for c in DIAMOND_NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in DIAMOND_CATEGORICAL_FEATURES if c in X.columns]
    preprocessor = build_diamond_preprocessor(num_cols, cat_cols)

    return X_train_clipped, X_test_clipped, y_train, y_test, preprocessor, clipper, num_cols, cat_cols


def train_and_evaluate_all_regression(df: pd.DataFrame, target_col="Price(in US dollars)", progress_callback=None):
    """
    Trains all 10 regression algorithms on the provided dataset and computes R2, RMSE, MAE.
    """
    X_train, X_test, y_train, y_test, preprocessor, clipper, num_cols, cat_cols = prepare_diamond_splits(df, target_col)
    
    # Fit preprocessor on train set
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    models = get_all_10_regression_models()
    results = []
    fitted_pipelines = {}
    test_preds = {}

    total_models = len(models)
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback((idx + 0.1) / total_models, f"Fitting {name}...")

        # Build full pipeline from raw input (after feature engineering and clipping)
        full_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        full_pipe.fit(X_train, y_train)
        pred = full_pipe.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))

        results.append({
            "Model": name,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2)
        })
        fitted_pipelines[name] = full_pipe
        test_preds[name] = pred

        if progress_callback:
            progress_callback((idx + 1.0) / total_models, f"Completed {name}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    return results_df, fitted_pipelines, X_test, y_test, test_preds, clipper


def run_regression_cross_val(fitted_pipelines, X_train, y_train, top_models):
    """
    Computes 5-fold cross-validated R2 for the top 2 models on the training set.
    """
    cv_rows = []
    for name in top_models:
        if name in fitted_pipelines:
            pipe = fitted_pipelines[name]
            scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            cv_rows.append({
                "Model": name,
                "Mean CV R2": round(float(scores.mean()), 4),
                "Std CV R2": round(float(scores.std()), 4),
                "Folds": 5
            })
    return pd.DataFrame(cv_rows)


def plot_actual_vs_predicted(y_actual, y_pred, model_name="Random Forest Regressor"):
    """Generates an Actual vs. Predicted scatter plot with equality reference line."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_actual, y_pred, alpha=0.3, s=15, color="#1e40af", edgecolors="none")
    min_val = min(float(np.min(y_actual)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_actual)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#dc2626", linewidth=1.8, label="Ideal Prediction (y = x)")
    ax.set_title(f"Actual vs. Predicted Price — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Price ($ USD)", fontsize=11)
    ax.set_ylabel("Predicted Price ($ USD)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_residuals(y_actual, y_pred, model_name="Random Forest Regressor"):
    """Generates a Residual Plot against Predicted Prices."""
    residuals = y_actual - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=15, color="#0284c7", edgecolors="none")
    ax.axhline(0, linestyle="--", color="#dc2626", linewidth=1.8, label="Zero Residual Line")
    ax.set_title(f"Residual Plot — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Price ($ USD)", fontsize=11)
    ax.set_ylabel("Residual (Actual - Predicted) ($ USD)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_feature_importance_rf(rf_model, feature_names, top_n=12):
    """Plots Random Forest feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    
    colors = sns.color_palette("viridis", len(top_features))
    sns.barplot(x=top_importances, y=top_features, palette=colors, ax=ax)
    ax.set_title("Random Forest — Top Feature Importances", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Gini Importance Score", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig
