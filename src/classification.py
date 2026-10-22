"""
Group 12 Machine Learning Capstone
Track 2: MAGIC Gamma Telescope (Classification)

Algorithms required by rubric:
1. Logistic Regression
2. K-Nearest Neighbors
3. Naive Bayes (Gaussian)
4. Decision Tree
5. Support Vector Machine (SVC)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Bagging Classifier
10. MLP Classifier
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay
)

from src.preprocessing import (
    engineer_magic_features,
    build_magic_preprocessor,
    MAGIC_RAW_FEATURES
)

RANDOM_STATE = 42

# Official benchmark results verified from notebooks/classification.ipynb
VERIFIED_CLASSIFICATION_BENCHMARKS = [
    {"Rank": 1, "Model": "Random Forest", "Accuracy": 0.8877, "Precision": 0.8874, "Recall": 0.8877, "F1 Score": 0.8861, "ROC-AUC": 0.9400, "Status": "Selected Best (CV & Test)"},
    {"Rank": 2, "Model": "MLP (Neural Network)", "Accuracy": 0.8846, "Precision": 0.8845, "Recall": 0.8846, "F1 Score": 0.8826, "ROC-AUC": 0.9386, "Status": "Evaluated"},
    {"Rank": 3, "Model": "SVM (RBF Kernel)", "Accuracy": 0.8796, "Precision": 0.8825, "Recall": 0.8796, "F1 Score": 0.8761, "ROC-AUC": 0.9256, "Status": "Evaluated"},
    {"Rank": 4, "Model": "Bagging (Decision Tree)", "Accuracy": 0.8775, "Precision": 0.8773, "Recall": 0.8775, "F1 Score": 0.8754, "ROC-AUC": 0.9359, "Status": "Evaluated"},
    {"Rank": 5, "Model": "Gradient Boosting", "Accuracy": 0.8767, "Precision": 0.8781, "Recall": 0.8767, "F1 Score": 0.8736, "ROC-AUC": 0.9276, "Status": "Evaluated"},
    {"Rank": 6, "Model": "AdaBoost", "Accuracy": 0.8470, "Precision": 0.8493, "Recall": 0.8470, "F1 Score": 0.8415, "ROC-AUC": 0.8839, "Status": "Evaluated"},
    {"Rank": 7, "Model": "KNN", "Accuracy": 0.8425, "Precision": 0.8464, "Recall": 0.8425, "F1 Score": 0.8359, "ROC-AUC": 0.8832, "Status": "Evaluated"},
    {"Rank": 8, "Model": "Decision Tree", "Accuracy": 0.8207, "Precision": 0.8208, "Recall": 0.8207, "F1 Score": 0.8208, "ROC-AUC": 0.8038, "Status": "Evaluated"},
    {"Rank": 9, "Model": "Logistic Regression", "Accuracy": 0.8049, "Precision": 0.8026, "Recall": 0.8049, "F1 Score": 0.7988, "ROC-AUC": 0.8462, "Status": "Baseline"},
    {"Rank": 10, "Model": "Naive Bayes (Gaussian)", "Accuracy": 0.7237, "Precision": 0.7142, "Recall": 0.7237, "F1 Score": 0.7134, "ROC-AUC": 0.7577, "Status": "Evaluated"},
]

VERIFIED_CLASSIFICATION_CV = [
    {"Model": "Random Forest", "Mean 5-fold CV Weighted F1": 0.8745, "Selection": "Highest CV Score (Selected Model)"},
    {"Model": "Bagging (Decision Tree)", "Mean 5-fold CV Weighted F1": 0.8739, "Selection": "Second Best CV"},
    {"Model": "MLP (Neural Network)", "Mean 5-fold CV Weighted F1": 0.8682, "Selection": "Third Best CV"},
    {"Model": "Gradient Boosting", "Mean 5-fold CV Weighted F1": 0.8646, "Selection": "Fourth Best CV"},
    {"Model": "SVM (RBF Kernel)", "Mean 5-fold CV Weighted F1": 0.8598, "Selection": "Top 5 CV"}
]

VERIFIED_CLASSIFICATION_TUNING = [
    {
        "Model": "Random Forest Classifier",
        "Best Hyperparameters": "{'max_depth': 20, 'min_samples_leaf': 1, 'n_estimators': 200}",
        "CV Weighted F1 (Best)": 0.8749,
        "Baseline Test Accuracy": 0.8877,
        "Tuned Test Accuracy": 0.8883,
        "Baseline Test Weighted F1": 0.8861,
        "Tuned Test Weighted F1": 0.8864,
        "Documented Improvement": "+0.0006 Accuracy, +0.0003 Weighted F1"
    }
]


def get_all_10_classification_models():
    """Returns the 10 required classification models with hyperparameter configurations."""
    try:
        bagging_estimator = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
    except TypeError:
        bagging_estimator = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "Bagging (Decision Tree)": bagging_estimator,
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB()
    }


def prepare_magic_splits(df: pd.DataFrame, target_col="class"):
    """
    Cleans index column, creates fConcDiff feature, and creates stratified train/test split.
    """
    df_clean = df.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates().copy()
    df_clean = engineer_magic_features(df_clean)

    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in telescope dataset.")

    df_clean = df_clean.dropna(subset=[target_col])
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_magic_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate_all_classification(df: pd.DataFrame, target_col="class", progress_callback=None):
    """
    Trains all 10 classifiers on the telescope dataset and computes
    Accuracy, Precision, Recall, Weighted F1, and ROC-AUC.
    """
    X_train, X_test, y_train, y_test, preprocessor = prepare_magic_splits(df, target_col)
    
    models = get_all_10_classification_models()
    results = []
    fitted_pipelines = {}
    test_preds = {}
    test_probas = {}

    # Identify positive class ('g' for gamma ray)
    pos_class = "g" if "g" in y_test.unique() else sorted(y_test.unique())[0]

    total = len(models)
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback((idx + 0.1) / total, f"Fitting {name}...")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

        # ROC-AUC calculation
        auc = np.nan
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)
            test_probas[name] = probs
            try:
                classes = list(pipe.classes_)
                pos_idx = classes.index(pos_class)
                y_bin = (y_test == pos_class).astype(int)
                auc = roc_auc_score(y_bin, probs[:, pos_idx])
            except Exception:
                auc = np.nan
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_test)
            y_bin = (y_test == pos_class).astype(int)
            auc = roc_auc_score(y_bin, scores)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4) if np.isfinite(auc) else "N/A"
        })
        fitted_pipelines[name] = pipe
        test_preds[name] = pred

        if progress_callback:
            progress_callback((idx + 1.0) / total, f"Finished {name}")

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    return results_df, fitted_pipelines, X_test, y_test, test_preds, test_probas


def plot_confusion_matrix_styled(y_actual, y_pred, model_name="Random Forest"):
    """Renders a styled Confusion Matrix for the classification track."""
    classes = sorted(np.unique(y_actual))
    cm = confusion_matrix(y_actual, y_pred, labels=classes)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Gamma (g)", "Hadron (h)"])
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    return fig


def plot_roc_curve_single(pipe, X_test, y_test, model_name="Random Forest", pos_class="g"):
    """Plots the ROC curve for a single model with AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    y_bin = (y_test == pos_class).astype(int)

    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_test)
        classes = list(pipe.classes_)
        pos_idx = classes.index(pos_class)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, pos_idx])
        auc = roc_auc_score(y_bin, probs[:, pos_idx])
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc = roc_auc_score(y_bin, scores)
    else:
        return None

    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_title(f"ROC Curve — {model_name} (Class 'g' vs. 'h')", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    return fig


def plot_roc_curves_multi(all_models, X_test, y_test, pos_class="g"):
    """Plots combined ROC curves for all models in the comparison table."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y_bin = (y_test == pos_class).astype(int)
    palette = sns.color_palette("tab10", len(all_models))

    for idx, (name, pipe) in enumerate(all_models.items()):
        try:
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(X_test)
                pos_idx = list(pipe.classes_).index(pos_class)
                fpr, tpr, _ = roc_curve(y_bin, probs[:, pos_idx])
                auc = roc_auc_score(y_bin, probs[:, pos_idx])
                ax.plot(fpr, tpr, label=f"{name} (AUC: {auc:.3f})", color=palette[idx % len(palette)], linewidth=1.8)
            elif hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_bin, scores)
                auc = roc_auc_score(y_bin, scores)
                ax.plot(fpr, tpr, label=f"{name} (AUC: {auc:.3f})", color=palette[idx % len(palette)], linewidth=1.8)
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.500)")
    ax.set_title("Comparative ROC Curves — All 10 Classifiers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    return fig
