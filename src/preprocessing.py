"""
Group 12 Machine Learning Capstone
Preprocessing & Feature Engineering Module

Faithfully preserves the preprocessing steps, outlier handling, and feature engineering
from the capstone notebooks:
- Track 1 (Diamond Price): Volume engineering, IQR clipping, Median Imputation, StandardScaler, OneHotEncoder.
- Track 2 (MAGIC Telescope): fConcDiff engineering, IQRClipper, Median Imputation, StandardScaler.
- Track 3 (HAR Clustering): Metadata exclusion (Activity & Subject), NaN cleaning, StandardScaler.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# -----------------------------------------------------------------------------
# TRACK 1: Diamond Price Regression Preprocessing
# -----------------------------------------------------------------------------

DIAMOND_CLIP_COLS = [
    "Carat(Weight of Daimond)", "Depth", "Table",
    "X(length)", "Y(width)", "Z(Depth)", "Volume"
]

DIAMOND_NUMERIC_FEATURES = [
    "Carat(Weight of Daimond)", "Depth", "Table",
    "X(length)", "Y(width)", "Z(Depth)", "Volume"
]

DIAMOND_CATEGORICAL_FEATURES = [
    "Cut(Quality)", "Color", "Clarity"
]


class DiamondIQRBoundsClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric features using IQR bounds fitted strictly on the training set.
    """
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns or DIAMOND_CLIP_COLS
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.bounds_ = {}
        for col in self.columns:
            if col in X_df.columns and pd.api.types.is_numeric_dtype(X_df[col]):
                q1 = X_df[col].quantile(0.25)
                q3 = X_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.factor * iqr
                upper = q3 + self.factor * iqr
                self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower, upper)
        return X_df


def engineer_diamond_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the Volume feature: Volume = X(length) * Y(width) * Z(Depth)
    as implemented in the regression notebook.
    """
    df = df.copy()
    x_col = next((c for c in df.columns if "x" in c.lower() and "length" in c.lower()), "X(length)")
    y_col = next((c for c in df.columns if "y" in c.lower() and "width" in c.lower()), "Y(width)")
    z_col = next((c for c in df.columns if "z" in c.lower() and "depth" in c.lower()), "Z(Depth)")
    
    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
        df["Volume"] = df[x_col] * df[y_col] * df[z_col]
    return df


def build_diamond_preprocessor(numeric_cols, categorical_cols):
    """
    Builds the ColumnTransformer for the Diamond dataset.
    """
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder)
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# -----------------------------------------------------------------------------
# TRACK 2: MAGIC Gamma Telescope Classification Preprocessing
# -----------------------------------------------------------------------------

MAGIC_RAW_FEATURES = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"
]


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Exact IQRClipper from classification_review2_humanized (5).ipynb:
    Fits IQR bounds on the training set and clips outliers.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_ = (q3 + self.factor * iqr).to_numpy()
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return np.clip(X_arr, self.lower_, self.upper_)


def engineer_magic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the fConcDiff feature: fConcDiff = fConc - fConc1
    as implemented in the classification notebook.
    """
    df = df.copy()
    if "fConc" in df.columns and "fConc1" in df.columns:
        df["fConcDiff"] = df["fConc"] - df["fConc1"]
    return df


def build_magic_preprocessor():
    """
    Builds the classification pipeline: IQRClipper -> SimpleImputer -> StandardScaler.
    """
    return Pipeline([
        ("clipper", IQRClipper(factor=1.5)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


# -----------------------------------------------------------------------------
# TRACK 3: Human Activity Recognition (HAR) Clustering Preprocessing
# -----------------------------------------------------------------------------

def prepare_har_data(df: pd.DataFrame):
    """
    Prepares HAR sensor features by excluding metadata (Activity and subject),
    handling infinities, imputing missing values, and fitting a StandardScaler.
    Returns: X_scaled (DataFrame or ndarray), feature_names, activity_labels, subject_labels
    """
    df = df.copy()
    activity_col = next((c for c in df.columns if c.lower() == "activity"), None)
    subject_col = next((c for c in df.columns if c.lower() == "subject"), None)

    activity_labels = df[activity_col].copy() if activity_col else None
    subject_labels = df[subject_col].copy() if subject_col else None

    metadata_cols = [c for c in [activity_col, subject_col] if c is not None]
    X_raw = df.drop(columns=metadata_cols, errors="ignore")
    X_raw = X_raw.select_dtypes(include=np.number)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, X_raw.columns.tolist(), activity_labels, subject_labels, scaler
