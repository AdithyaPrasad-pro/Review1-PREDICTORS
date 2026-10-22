"""
Group 12 Machine Learning Capstone
Model Training and Serialization Script

Trains and exports the selected best models, preprocessors, and benchmark data
to `models/` directory using joblib and json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.preprocessing import (
    engineer_diamond_features,
    DiamondIQRBoundsClipper,
    build_diamond_preprocessor,
    DIAMOND_CLIP_COLS,
    DIAMOND_NUMERIC_FEATURES,
    DIAMOND_CATEGORICAL_FEATURES,
    engineer_magic_features,
    build_magic_preprocessor,
    prepare_har_data
)

from src.regression import (
    VERIFIED_REGRESSION_BENCHMARKS,
    VERIFIED_REGRESSION_CV,
    VERIFIED_REGRESSION_TUNING
)

from src.classification import (
    VERIFIED_CLASSIFICATION_BENCHMARKS,
    VERIFIED_CLASSIFICATION_CV,
    VERIFIED_CLASSIFICATION_TUNING
)

from src.clustering import (
    VERIFIED_CLUSTERING_BENCHMARKS,
    VERIFIED_LINKAGE_COMPARISON,
    compute_elbow_curve,
    compute_pca_2d
)

RANDOM_STATE = 42
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def export_diamond_artifacts():
    print("Exporting Track 1: Diamond Price artifacts...")
    df = pd.read_csv("data/Diamond Price Prediction.csv").drop_duplicates()
    df = engineer_diamond_features(df)
    target = "Price(in US dollars)"
    y = df[target]
    X = df.drop(columns=[target])

    price_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=price_bins
    )

    clipper = DiamondIQRBoundsClipper(DIAMOND_CLIP_COLS)
    clipper.fit(X_train)
    X_train_c = clipper.transform(X_train)

    num_cols = [c for c in DIAMOND_NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in DIAMOND_CATEGORICAL_FEATURES if c in X.columns]
    preprocessor = build_diamond_preprocessor(num_cols, cat_cols)
    X_train_p = preprocessor.fit_transform(X_train_c)

    # Train best Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_p, y_train)

    # Save artifacts
    joblib.dump(rf, os.path.join(MODELS_DIR, "diamond_rf_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "diamond_preprocessor.joblib"))
    joblib.dump(clipper, os.path.join(MODELS_DIR, "diamond_clipper.joblib"))

    # Feature names
    cat_enc = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names = num_cols + list(cat_enc.get_feature_names_out(cat_cols))
    with open(os.path.join(MODELS_DIR, "diamond_feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save verified benchmarks
    diamond_meta = {
        "benchmarks": VERIFIED_REGRESSION_BENCHMARKS,
        "cross_val": VERIFIED_REGRESSION_CV,
        "tuning": VERIFIED_REGRESSION_TUNING,
        "best_model_name": "Random Forest Regressor",
        "sample_row": X_test.iloc[0].to_dict()
    }
    with open(os.path.join(MODELS_DIR, "diamond_benchmarks.json"), "w") as f:
        json.dump(diamond_meta, f, indent=2)

    print("Diamond artifacts exported successfully.")


def export_magic_artifacts():
    print("Exporting Track 2: MAGIC Telescope artifacts...")
    df = pd.read_csv("data/telescope_data.csv").drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates()
    df = engineer_magic_features(df)
    target = "class"
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_magic_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)

    # Train best Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_p, y_train)

    joblib.dump(rf, os.path.join(MODELS_DIR, "magic_rf_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "magic_preprocessor.joblib"))

    magic_meta = {
        "benchmarks": VERIFIED_CLASSIFICATION_BENCHMARKS,
        "cross_val": VERIFIED_CLASSIFICATION_CV,
        "tuning": VERIFIED_CLASSIFICATION_TUNING,
        "best_model_name": "Random Forest",
        "classes": list(rf.classes_),
        "sample_row": X_test.iloc[0].to_dict()
    }
    with open(os.path.join(MODELS_DIR, "magic_benchmarks.json"), "w") as f:
        json.dump(magic_meta, f, indent=2)

    print("MAGIC artifacts exported successfully.")


def export_har_artifacts():
    print("Exporting Track 3: HAR Clustering artifacts...")
    df = pd.read_csv("data/test.csv")
    X_scaled, feature_names, activity_labels, subject_labels, scaler = prepare_har_data(df)

    # Fit baseline KMeans
    km = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    joblib.dump(km, os.path.join(MODELS_DIR, "har_kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "har_scaler.joblib"))

    # Precalculate PCA 2D coordinates
    pca_coords, pca_var, pca_obj = compute_pca_2d(X_scaled)
    joblib.dump(pca_obj, os.path.join(MODELS_DIR, "har_pca_model.joblib"))

    # Precalculate Elbow inertias
    k_range, inertias = compute_elbow_curve(X_scaled, k_min=2, k_max=10)

    # Precalculate t-SNE coordinates for fast display
    rng = np.random.RandomState(RANDOM_STATE)
    tsne_sample_size = min(1200, len(X_scaled))
    tsne_idx = rng.choice(len(X_scaled), size=tsne_sample_size, replace=False)
    pca_50 = PCA(n_components=min(50, X_scaled.shape[1]), random_state=RANDOM_STATE).fit_transform(X_scaled[tsne_idx])
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
    tsne_coords = tsne.fit_transform(pca_50)

    # Activity interpretation crosstab
    act_crosstab = {}
    if activity_labels is not None:
        ct = pd.crosstab(
            pd.Series(km_labels, name="Cluster"),
            activity_labels.rename("True Activity"),
            normalize="index"
        ).round(3)
        act_crosstab = ct.to_dict(orient="index")

    har_meta = {
        "benchmarks": VERIFIED_CLUSTERING_BENCHMARKS,
        "linkage_comparison": VERIFIED_LINKAGE_COMPARISON,
        "k_range": k_range,
        "inertias": inertias,
        "pca_variance_explained": round(pca_var, 4),
        "activity_crosstab": act_crosstab,
        "tsne_coords": tsne_coords.tolist(),
        "tsne_labels": km_labels[tsne_idx].tolist()
    }
    with open(os.path.join(MODELS_DIR, "har_benchmarks.json"), "w") as f:
        json.dump(har_meta, f, indent=2)

    print("HAR artifacts exported successfully.")


if __name__ == "__main__":
    export_diamond_artifacts()
    export_magic_artifacts()
    export_har_artifacts()
    print("ALL ARTIFACTS EXPORTED TO models/")
