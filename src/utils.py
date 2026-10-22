"""
Group 12 Machine Learning Capstone
Utility & Styling Module

Provides cohesive CSS styling following the reference Streamlit layout:
- Pale gray sidebar
- Clean card sections with light gray borders
- Styled metrics and badges
- Safe dataset upload & validation
"""

import streamlit as st
import pandas as pd
import numpy as np


TEAM_INFO = {
    "group": "Group 12",
    "course": "23CSE301 — Machine Learning Capstone",
    "academic_year": "2026–27",
    "members": [
        {"name": "Adithya Prasad"},
        {"name": "Rahul Reddy"},
        {"name": "Pushkar Mallella"}
    ]
}

TRACK_DESCRIPTIONS = {
    "Home": {
        "title": "Group 12 — Machine Learning Capstone Dashboard",
        "badge": "🏁 Executive Summary",
        "description": "Comprehensive comparative evaluation across 3 machine learning tracks: Regression (Diamond Prices), Classification (MAGIC Telescope), and Clustering (Smartphone Human Activity Recognition)."
    },
    "Track 1": {
        "title": "Track 1: Diamond Price Prediction (Regression)",
        "badge": "💎 Regression Track",
        "task": "Estimate diamond price ($ USD) from quality and geometric attributes",
        "dataset": "Diamond Price Prediction (53,940 rows, 10 attributes)",
        "models": "10 Regression Algorithms (Linear, Ridge, Lasso, ElasticNet, Poly, Tree, RF, GBM, SVR, KNN)",
        "primary_metric": "R² (Coefficient of Determination), RMSE, MAE",
        "description": "Evaluate 10 regression algorithms on identical test splits, tune top models (Random Forest, Ridge), verify with 5-fold cross-validation, and perform interactive price estimation."
    },
    "Track 2": {
        "title": "Track 2: MAGIC Gamma Telescope (Classification)",
        "badge": "🔭 Classification Track",
        "task": "Differentiate high-energy Gamma rays ('g') from Hadronic background ('h')",
        "dataset": "MAGIC Gamma Telescope (19,020 events, 10 continuous shower features)",
        "models": "10 Classifiers (Logistic, KNN, Naive Bayes, Tree, SVM, RF, AdaBoost, GBM, Bagging, MLP)",
        "primary_metric": "Accuracy, Precision, Recall, Weighted F1, ROC-AUC",
        "description": "Consolidated 10-classifier comparison, 5-fold cross-validation, confusion matrices, multi-model ROC curves, and real-time particle shower classification."
    },
    "Track 3": {
        "title": "Track 3: Human Activity Recognition (Clustering)",
        "badge": "📱 Clustering Track",
        "task": "Discover intrinsic groupings in 561-dimensional smartphone sensor features",
        "dataset": "Human Activity Recognition with Smartphones (2,947 observations, 561 sensor features)",
        "models": "K-Means Partitional & Agglomerative Hierarchical (Ward, Complete, Average)",
        "primary_metric": "Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index",
        "description": "Strictly unsupervised clustering, K-Means elbow analysis, hierarchical dendrogram, 2D PCA & t-SNE projections, and post-hoc activity interpretation crosstab."
    }
}


def load_custom_css():
    """Injects responsive, modern light CSS matching the capstone reference UI."""
    css = """
    <style>
        /* Base typography & light background */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1e293b;
        }

        /* Sidebar: Pale gray styling with subtle divider */
        [data-testid="stSidebar"] {
            background-color: #f1f4f9 !important;
            border-right: 1px solid #e2e8f0;
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: #cbd5e1;
        }

        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Bordered section container matching reference screenshot */
        .bordered-section {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }

        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 0.5rem;
        }

        /* Metric cards */
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .stat-val {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1e40af;
            margin-top: 0.2rem;
        }
        .stat-lbl {
            font-size: 0.82rem;
            font-weight: 500;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Badges */
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #e0f2fe;
            color: #0369a1;
            margin-bottom: 0.5rem;
        }

        .badge-success {
            background-color: #dcfce7;
            color: #15803d;
        }

        .badge-warning {
            background-color: #fef3c7;
            color: #b45309;
        }

        /* Prediction banner */
        .result-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
            border: 2px solid #86efac;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            text-align: center;
        }

        .result-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #166534;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.25rem 0;
        }

        .result-subtitle {
            font-size: 0.85rem;
            color: #475569;
        }

        /* Table enhancements */
        .stDataFrame {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Streamlit button full-width styling */
        div.stButton > button:first-child {
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar():
    """Renders the standard sidebar navigation, project track summary, and team information."""
    with st.sidebar:
        st.markdown("## 🎯 Group 12 Capstone")
        st.caption("Machine Learning Capstone Project • 23CSE301")
        
        # Navigation
        track_choice = st.radio(
            "Navigation Menu",
            [
                "🏠 Home",
                "💎 Track 1: Diamond Price (Regression)",
                "🔭 Track 2: MAGIC Telescope (Classification)",
                "📱 Track 3: Human Activity (Clustering)"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.divider()

        # Dynamic Track Summary
        st.markdown("### 📌 Track Summary")
        if "Home" in track_choice:
            st.markdown(
                """
                - **Regression:** Diamond Price ($ USD)
                - **Classification:** MAGIC Gamma vs. Hadron
                - **Clustering:** Smartphone Sensor Activities
                - **Scope:** 10 Regressors, 10 Classifiers, 2 Clustering methods
                """
            )
        elif "Diamond" in track_choice:
            info = TRACK_DESCRIPTIONS["Track 1"]
            st.markdown(f"**Task:** {info['task']}")
            st.markdown(f"**Dataset:** {info['dataset']}")
            st.markdown(f"**Metrics:** {info['primary_metric']}")
            st.markdown("**Top Model:** Random Forest (R²: 0.9778)")
        elif "MAGIC" in track_choice:
            info = TRACK_DESCRIPTIONS["Track 2"]
            st.markdown(f"**Task:** {info['task']}")
            st.markdown(f"**Dataset:** {info['dataset']}")
            st.markdown(f"**Metrics:** {info['primary_metric']}")
            st.markdown("**Top Model:** Random Forest (F1: 0.8861)")
        elif "Human" in track_choice:
            info = TRACK_DESCRIPTIONS["Track 3"]
            st.markdown(f"**Task:** {info['task']}")
            st.markdown(f"**Dataset:** {info['dataset']}")
            st.markdown(f"**Metrics:** {info['primary_metric']}")
            st.markdown("**Unsupervised:** Ground-truth labels used strictly for post-hoc interpretation")

        st.divider()

        # Team Information
        st.markdown("### 👥 Project Team")
        st.markdown(f"**{TEAM_INFO['group']}**")
        for member in TEAM_INFO["members"]:
            st.markdown(f"- **{member['name']}**  \n  <span style='font-size:0.8rem;color:#64748b;'>{member['role']}</span>", unsafe_allow_html=True)
        
        st.markdown(f"<div style='margin-top:0.5rem;font-size:0.82rem;color:#64748b;'>{TEAM_INFO['course']}<br>Academic Year {TEAM_INFO['academic_year']}</div>", unsafe_allow_html=True)

        st.divider()
        st.caption("Group 12 ML Capstone • Deployed on Streamlit Cloud")

    return track_choice


def render_track_header(track_key):
    """Renders consistent track header with badge, title, description, and divider."""
    info = TRACK_DESCRIPTIONS[track_key]
    st.markdown(f"<span class='badge-pill'>{info['badge']}</span>", unsafe_allow_html=True)
    st.title(info["title"])
    st.markdown(f"<div style='font-size:1.05rem;color:#475569;margin-bottom:0.75rem;'>{info['description']}</div>", unsafe_allow_html=True)
    st.divider()


def validate_uploaded_csv(uploaded_file, required_columns=None, track_name=""):
    """
    Validates uploaded CSV file for formatting, readability, and required columns.
    Returns (DataFrame, error_message)
    """
    if uploaded_file is None:
        return None, None
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return None, "The uploaded CSV file is empty."
        
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                return df, f"Warning: Expected columns {missing} were not found in uploaded file. Please confirm column mapping."
        
        return df, None
    except Exception as e:
        return None, f"Could not parse uploaded CSV: {str(e)}"
