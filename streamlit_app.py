"""
Machine Learning Capstone — Model Deployment Dashboard
Interactive Multi-Track Deployment Application

Tracks Deployed:
- Track 1: Diamond Price Prediction (Regression) — Random Forest Regressor
- Track 2: MAGIC Gamma Telescope (Classification) — Random Forest Classifier
- Track 3: Human Activity Recognition (Clustering) — K-Means (k=6)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src module is discoverable
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    engineer_diamond_features,
    DiamondIQRBoundsClipper,
    engineer_magic_features,
    IQRClipper
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Capstone Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Matches reference design)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Section container matching reference image */
    .bordered-section {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Output result cards */
    .prediction-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .prediction-card-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #fcd34d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .prediction-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #166534;
        margin-bottom: 0.25rem;
    }

    .prediction-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.25rem 0;
    }

    .prediction-meta {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 500;
    }

    /* Sidebar info box */
    .sidebar-track-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        background-color: #ede9fe;
        color: #5b21b6;
    }

    .badge-green {
        background-color: #dcfce7;
        color: #15803d;
    }

    /* Action button */
    div.stButton > button:first-child {
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
        transition: all 0.15s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #4338ca;
        color: white;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Artifact & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_diamond_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "diamond_rf_model.joblib"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "diamond_preprocessor.joblib"))
        clipper = joblib.load(os.path.join(BASE_DIR, "models", "diamond_clipper.joblib"))
        with open(os.path.join(BASE_DIR, "models", "diamond_benchmarks.json"), "r") as f:
            benchmarks = json.load(f)
        return model, preprocessor, clipper, benchmarks
    except Exception as e:
        st.error(f"Error loading Diamond model artifacts: {e}")
        return None, None, None, None

@st.cache_resource
def load_magic_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "magic_rf_model.joblib"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "magic_preprocessor.joblib"))
        with open(os.path.join(BASE_DIR, "models", "magic_benchmarks.json"), "r") as f:
            benchmarks = json.load(f)
        return model, preprocessor, benchmarks
    except Exception as e:
        st.error(f"Error loading MAGIC model artifacts: {e}")
        return None, None, None

@st.cache_resource
def load_har_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "har_kmeans_model.joblib"))
        scaler = joblib.load(os.path.join(BASE_DIR, "models", "har_scaler.joblib"))
        pca = joblib.load(os.path.join(BASE_DIR, "models", "har_pca_model.joblib"))
        with open(os.path.join(BASE_DIR, "models", "har_benchmarks.json"), "r") as f:
            benchmarks = json.load(f)
        return model, scaler, pca, benchmarks
    except Exception as e:
        st.error(f"Error loading HAR model artifacts: {e}")
        return None, None, None, None

@st.cache_data
def load_har_test_data():
    csv_path = os.path.join(BASE_DIR, "data", "test.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# -----------------------------------------------------------------------------
# Sidebar Navigation (Faithful to reference image layout)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Capstone Problem Tracks")
    st.write("Select Machine Learning Track:")
    
    selected_track = st.radio(
        "Select Machine Learning Track:",
        [
            "Track 1: Diamond Price (Regression)",
            "Track 2: MAGIC Gamma Telescope (Classification)",
            "Track 3: Human Activity Recognition (Clustering)"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📌 Project Track Summary")

    if "Track 1" in selected_track:
        st.markdown(
            """
            **Regression Track**
            - **Task:** Diamond Market Price ($ USD)
            - **Dataset:** Diamond Price Prediction (`Diamond Price Prediction.csv`, 53,940 rows)
            - **Model:** Random Forest Regressor *(Selected Best Model)*
            - **Metrics:** $R^2 = 0.9778$, $\\text{RMSE} = \\$595.84$, $\\text{MAE} = \\$308.91$
            """
        )
    elif "Track 2" in selected_track:
        st.markdown(
            """
            **Classification Track**
            - **Task:** Particle Shower Classification (`g` vs `h`)
            - **Dataset:** MAGIC Gamma Telescope (`telescope_data.csv`, 19,020 rows)
            - **Model:** Random Forest Classifier *(Selected Best Model)*
            - **Metrics:** $88.77\\%\\text{ Accuracy}$, $0.8861\\text{ F1-Score}$, $0.9400\\text{ ROC-AUC}$
            """
        )
    else:
        st.markdown(
            """
            **Clustering Track**
            - **Task:** Smartphone Activity Grouping (6 Physical Clusters)
            - **Dataset:** Human Activity Recognition (`test.csv`, 2,947 rows $\\times$ 561 features)
            - **Model:** K-Means Clustering ($k=6$) *(Selected Algorithm)*
            - **Metrics:** $\\text{Silhouette} = 0.1030$, $\\text{Calinski-Harabasz} = 744.81$, $\\text{Davies-Bouldin} = 2.6595$
            """
        )

    st.markdown("---")
    st.markdown("### 👥 Team")
    st.markdown(
        """
        - **Adithya Prasad**
        - **Rahul Reddy**
        - **Pushkar Mallella**
        
        Capstone Project • 23CSE301
        """
    )

    st.markdown("---")
    st.caption(
        "Academic Disclaimer: This application was developed as an academic deliverable for the Machine Learning Capstone. All benchmarks and models reflect the submitted capstone deliverables."
    )


# =============================================================================
# TRACK 1: REGRESSION (Diamond Price Prediction)
# =============================================================================
if "Track 1" in selected_track:
    st.markdown("## 💎 Diamond Price Prediction")
    st.markdown(
        "Predict the market price of diamonds in US Dollars based on cut quality, carat weight, color, clarity, and physical geometric dimensions using the selected best model (**Random Forest Regressor**)."
    )

    rf_model, preprocessor, clipper, benchmarks = load_diamond_artifacts()

    if rf_model is None:
        st.error("Model artifacts not available. Please ensure models are exported properly.")
        st.stop()

    st.markdown("### Enter Diamond Specifications & Dimensions")

    # Quick presets
    preset_choice = st.selectbox(
        "⚡ Choose a Quick Preset (or customize below):",
        [
            "Custom Configuration",
            "Preset 1: 0.50 Carat Everyday Round (Ideal Cut, Color G, Clarity VS2)",
            "Preset 2: 1.00 Carat Solitaire Engagement (Ideal Cut, Color E, Clarity VVS2)",
            "Preset 3: 2.05 Carat Luxury Statement (Premium Cut, Color D, Clarity IF)",
            "Preset 4: 0.30 Carat Budget Diamond (Good Cut, Color J, Clarity SI1)"
        ]
    )

    # Preset values
    if preset_choice.startswith("Preset 1"):
        d_carat, d_cut, d_color, d_clarity = 0.50, "Ideal", "G", "VS2"
        d_x, d_y, d_z, d_depth, d_table = 5.12, 5.15, 3.16, 61.6, 56.0
    elif preset_choice.startswith("Preset 2"):
        d_carat, d_cut, d_color, d_clarity = 1.00, "Ideal", "E", "VVS2"
        d_x, d_y, d_z, d_depth, d_table = 6.42, 6.45, 3.96, 61.5, 57.0
    elif preset_choice.startswith("Preset 3"):
        d_carat, d_cut, d_color, d_clarity = 2.05, "Premium", "D", "IF"
        d_x, d_y, d_z, d_depth, d_table = 8.18, 8.24, 5.02, 61.2, 59.0
    elif preset_choice.startswith("Preset 4"):
        d_carat, d_cut, d_color, d_clarity = 0.30, "Good", "J", "SI1"
        d_x, d_y, d_z, d_depth, d_table = 4.30, 4.33, 2.70, 63.4, 58.0
    else:
        d_carat, d_cut, d_color, d_clarity = 0.70, "Ideal", "G", "SI1"
        d_x, d_y, d_z, d_depth, d_table = 5.71, 5.73, 3.52, 61.5, 57.0

    # Section 1: Quality Profile
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>👤</span> Diamond Quality Profile</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        carat = st.number_input("Carat (Weight)", min_value=0.10, max_value=5.00, value=float(d_carat), step=0.01, format="%.2f")
    with col2:
        cut_opts = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
        cut = st.selectbox("Cut (Quality)", cut_opts, index=cut_opts.index(d_cut) if d_cut in cut_opts else 0)
    with col3:
        color_opts = ["D", "E", "F", "G", "H", "I", "J"]
        color = st.selectbox("Color", color_opts, index=color_opts.index(d_color) if d_color in color_opts else 3)
    with col4:
        clarity_opts = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
        clarity = st.selectbox("Clarity", clarity_opts, index=clarity_opts.index(d_clarity) if d_clarity in clarity_opts else 5)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Physical Dimensions & Proportions
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>📐</span> Geometric Dimensions & Proportions</div>', unsafe_allow_html=True)
    dcol1, dcol2, dcol3, dcol4, dcol5 = st.columns(5)
    with dcol1:
        x_len = st.number_input("Length X (mm)", min_value=1.0, max_value=15.0, value=float(d_x), step=0.05, format="%.2f")
    with dcol2:
        y_wid = st.number_input("Width Y (mm)", min_value=1.0, max_value=15.0, value=float(d_y), step=0.05, format="%.2f")
    with dcol3:
        z_dep = st.number_input("Depth Z (mm)", min_value=0.5, max_value=10.0, value=float(d_z), step=0.05, format="%.2f")
    with dcol4:
        depth_pct = st.number_input("Total Depth (%)", min_value=40.0, max_value=85.0, value=float(d_depth), step=0.1, format="%.1f")
    with dcol5:
        table_pct = st.number_input("Table Width (%)", min_value=40.0, max_value=95.0, value=float(d_table), step=0.1, format="%.1f")

    # Calculated volume
    calc_volume = x_len * y_wid * z_dep
    st.caption(f"ℹ️ Automatically calculated diamond volume: **{calc_volume:.2f} mm³** ($X \\times Y \\times Z$)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict Button
    predict_btn = st.button("🔮 Predict Diamond Price", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "Carat(Weight of Daimond)": carat,
            "Cut(Quality)": cut,
            "Color": color,
            "Clarity": clarity,
            "Depth": depth_pct,
            "Table": table_pct,
            "X(length)": x_len,
            "Y(width)": y_wid,
            "Z(Depth)": z_dep
        }])

        # Feature engineering & transformations
        df_eng = engineer_diamond_features(input_data)
        df_clipped = clipper.transform(df_eng)
        df_proc = preprocessor.transform(df_clipped)

        # Random Forest prediction
        predicted_price = float(rf_model.predict(df_proc)[0])
        predicted_price = max(300.0, predicted_price)  # sanity lower bound
        price_per_carat = predicted_price / carat

        # Display result card
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Selected Model: Random Forest Regressor (R² = 0.9778)</div>
                <div class="prediction-value">${predicted_price:,.2f} USD</div>
                <div class="prediction-meta">
                    Estimated 95% Confidence Bounds: <strong>${max(0, predicted_price - 308.91):,.2f} — ${predicted_price + 308.91:,.2f}</strong> (± MAE of $308.91)
                    <br>Price per Carat: <strong>${price_per_carat:,.2f} / ct</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Relevant visualizations for the selected model
        st.markdown("### 📊 Model Insights & Feature Diagnostics")
        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            st.markdown("#### Random Forest Top Feature Importances")
            importances = rf_model.feature_importances_
            with open(os.path.join(BASE_DIR, "models", "diamond_feature_names.json"), "r") as f:
                feature_names = json.load(f)

            fi_df = pd.DataFrame({
                "Feature": feature_names[:len(importances)],
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.barplot(data=fi_df, x="Importance", y="Feature", color="#4f46e5", ax=ax)
            ax.set_title("Relative Feature Importance (Selected Model)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Gini Importance Score")
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with vcol2:
            st.markdown("#### Regression Benchmark Verification")
            st.write(
                """
                **Official Capstone Test Results:**
                - **Coefficient of Determination ($R^2$):** `0.9778` (97.8% variance explained)
                - **Root Mean Squared Error (RMSE):** `$595.84`
                - **Mean Absolute Error (MAE):** `$308.91`
                - **5-Fold Cross-Validation $R^2$:** `0.9774 ± 0.0004`
                
                The Random Forest Regressor captured strong non-linear interactions between carat weight, physical volume, and cut angles, significantly outperforming linear models ($R^2=0.9280$).
                """
            )
            # Comparison table against runner-up
            st.dataframe(
                pd.DataFrame([
                    {"Model": "Random Forest (Selected)", "R²": 0.9778, "RMSE ($)": 595.84, "MAE ($)": 308.91, "CV R²": 0.9774},
                    {"Model": "Polynomial (Degree 2)", "R²": 0.9725, "RMSE ($)": 662.58, "MAE ($)": 392.57, "CV R²": 0.9707},
                    {"Model": "Linear Baseline", "R²": 0.9280, "RMSE ($)": 1072.26, "MAE ($)": 708.10, "CV R²": 0.9272}
                ]),
                hide_index=True,
                use_container_width=True
            )


# =============================================================================
# TRACK 2: CLASSIFICATION (MAGIC Gamma Telescope)
# =============================================================================
elif "Track 2" in selected_track:
    st.markdown("## 🔭 MAGIC Gamma Telescope Particle Classification")
    st.markdown(
        "Differentiate high-energy primary **Gamma (γ) ray signals** from background **Hadronic (h) cosmic cascades** using atmospheric Cherenkov radiation shower parameters with the selected best model (**Random Forest Classifier**)."
    )

    rf_cls, preprocessor, benchmarks = load_magic_artifacts()

    if rf_cls is None:
        st.error("Classification model artifacts not available.")
        st.stop()

    st.markdown("### Enter Cherenkov Shower Parameters")

    preset_choice = st.selectbox(
        "⚡ Choose a Quick Preset (or customize below):",
        [
            "Custom Event Parameters",
            "Preset 1: Typical Primary Gamma Ray Signal (Compact ellipse, small alpha)",
            "Preset 2: Typical Hadronic Cosmic Background (Diffuse shower, high alpha)",
            "Preset 3: Ambiguous Core Shower Event"
        ]
    )

    if preset_choice.startswith("Preset 1"):
        p_len, p_wid, p_size = 28.5, 15.8, 2.65
        p_conc, p_conc1 = 0.38, 0.22
        p_asym, p_m3l, p_m3t = -8.5, 12.0, 4.5
        p_alpha, p_dist = 11.0, 160.0
    elif preset_choice.startswith("Preset 2"):
        p_len, p_wid, p_size = 85.0, 48.0, 3.75
        p_conc, p_conc1 = 0.18, 0.09
        p_asym, p_m3l, p_m3t = -55.0, -60.0, -35.0
        p_alpha, p_dist = 65.0, 260.0
    elif preset_choice.startswith("Preset 3"):
        p_len, p_wid, p_size = 45.0, 28.0, 3.10
        p_conc, p_conc1 = 0.28, 0.15
        p_asym, p_m3l, p_m3t = -20.0, -15.0, 10.0
        p_alpha, p_dist = 35.0, 210.0
    else:
        p_len, p_wid, p_size = 30.5, 18.2, 2.85
        p_conc, p_conc1 = 0.35, 0.22
        p_asym, p_m3l, p_m3t = -12.5, 10.2, -5.3
        p_alpha, p_dist = 15.0, 180.0

    # Section 1: Shower Ellipse Geometry
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>📐</span> Shower Ellipse Geometry</div>', unsafe_allow_html=True)
    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        f_length = st.number_input("fLength (Major Axis, mm)", min_value=2.0, max_value=350.0, value=float(p_len), step=1.0, format="%.2f")
    with gcol2:
        f_width = st.number_input("fWidth (Minor Axis, mm)", min_value=1.0, max_value=250.0, value=float(p_wid), step=1.0, format="%.2f")
    with gcol3:
        f_size = st.number_input("fSize (10-log Photon Energy)", min_value=1.0, max_value=7.0, value=float(p_size), step=0.05, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Light Concentration
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>💡</span> Cherenkov Light Concentration</div>', unsafe_allow_html=True)
    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        f_conc = st.number_input("fConc (2 Highest Pixels Ratio)", min_value=0.0, max_value=1.0, value=float(p_conc), step=0.01, format="%.3f")
    with ccol2:
        f_conc1 = st.number_input("fConc1 (Highest Pixel Ratio)", min_value=0.0, max_value=1.0, value=float(p_conc1), step=0.01, format="%.3f")
    with ccol3:
        f_conc_diff = f_conc - f_conc1
        st.metric("fConcDiff (Engineered)", f"{f_conc_diff:.3f}", help="fConc - fConc1: Concentration gradient")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Orientation & Higher Moments
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>🧭</span> Orientation & Higher Moments</div>', unsafe_allow_html=True)
    ocol1, ocol2, ocol3, ocol4, ocol5 = st.columns(5)
    with ocol1:
        f_alpha = st.number_input("fAlpha (Angle, deg)", min_value=0.0, max_value=90.0, value=float(p_alpha), step=1.0, format="%.1f")
    with ocol2:
        f_dist = st.number_input("fDist (Distance, mm)", min_value=0.0, max_value=500.0, value=float(p_dist), step=5.0, format="%.1f")
    with ocol3:
        f_asym = st.number_input("fAsym (Asymmetry, mm)", min_value=-400.0, max_value=400.0, value=float(p_asym), step=5.0, format="%.1f")
    with ocol4:
        f_m3l = st.number_input("fM3Long (3rd Moment, mm)", min_value=-250.0, max_value=250.0, value=float(p_m3l), step=5.0, format="%.1f")
    with ocol5:
        f_m3t = st.number_input("fM3Trans (Transverse, mm)", min_value=-200.0, max_value=200.0, value=float(p_m3t), step=5.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    # Classification Button
    classify_btn = st.button("⚡ Classify Particle Shower", use_container_width=True)

    if classify_btn:
        input_data = pd.DataFrame([{
            "fLength": f_length,
            "fWidth": f_width,
            "fSize": f_size,
            "fConc": f_conc,
            "fConc1": f_conc1,
            "fAsym": f_asym,
            "fM3Long": f_m3l,
            "fM3Trans": f_m3t,
            "fAlpha": f_alpha,
            "fDist": f_dist
        }])

        df_eng = engineer_magic_features(input_data)
        df_proc = preprocessor.transform(df_eng)

        pred_class = rf_cls.predict(df_proc)[0]
        pred_proba = rf_cls.predict_proba(df_proc)[0]

        gamma_idx = list(rf_cls.classes_).index("g")
        hadron_idx = list(rf_cls.classes_).index("h")
        gamma_prob = pred_proba[gamma_idx]
        hadron_prob = pred_proba[hadron_idx]

        is_gamma = (pred_class == "g")
        card_class = "prediction-card" if is_gamma else "prediction-card-warning"
        title_text = "Primary Gamma Ray Signal (γ)" if is_gamma else "Hadronic Cosmic Ray Cascade (h)"
        sub_text = "Compact electromagnetic Cherenkov shower detected" if is_gamma else "Diffuse hadronic background shower detected"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="prediction-label">Selected Model: Random Forest Classifier (Accuracy = 88.77%, ROC-AUC = 0.9400)</div>
                <div class="prediction-value">{title_text}</div>
                <div class="prediction-meta">
                    {sub_text}
                    <br>Confidence: <strong>{max(gamma_prob, hadron_prob)*100:.1f}%</strong> | Predicted Class: <code>{pred_class}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visualizations
        st.markdown("### 📊 Classification Probability & Diagnostics")
        pcol1, pcol2 = st.columns([1, 1])

        with pcol1:
            st.markdown("#### Class Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(["Hadron (h)", "Gamma (g)"], [hadron_prob, gamma_prob], color=["#f59e0b", "#10b981"], height=0.45)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Predicted Probability")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with pcol2:
            st.markdown("#### Official Capstone Benchmark Metrics")
            st.write(
                """
                - **Accuracy:** `88.77%`
                - **Weighted Precision:** `88.74%`
                - **Weighted Recall:** `88.77%`
                - **Weighted F1-Score:** `0.8861` (Highest among all 10 classifiers)
                - **ROC-AUC (vs Hadron):** `0.9400`
                - **5-Fold Cross-Validation F1:** `0.8745 ± 0.003`
                """
            )
            st.dataframe(
                pd.DataFrame([
                    {"Model": "Random Forest (Selected)", "Accuracy": "88.77%", "F1": 0.8861, "ROC-AUC": 0.9400},
                    {"Model": "MLP Neural Network", "Accuracy": "88.46%", "F1": 0.8826, "ROC-AUC": 0.9386},
                    {"Model": "Support Vector Machine", "Accuracy": "87.96%", "F1": 0.8761, "ROC-AUC": 0.9256},
                    {"Model": "Logistic Regression Baseline", "Accuracy": "80.49%", "F1": 0.7988, "ROC-AUC": 0.8462}
                ]),
                hide_index=True,
                use_container_width=True
            )


# =============================================================================
# TRACK 3: CLUSTERING (Human Activity Recognition with Smartphones)
# =============================================================================
else:
    st.markdown("## 📱 Human Activity Recognition Clustering")
    st.markdown(
        "Discover intrinsic natural groupings in 561-dimensional smartphone sensor features using the selected clustering algorithm (**K-Means with k=6 clusters**). Visualized on held-out test data in 2D PCA space."
    )

    km_model, scaler, pca_model, har_benchmarks = load_har_artifacts()
    test_df = load_har_test_data()

    if km_model is None or test_df is None:
        st.error("Clustering model or test dataset not available.")
        st.stop()

    st.markdown("### Select & Tweak Smartphone Sensor Features")

    feature_cols = [c for c in test_df.columns if c not in ["Activity", "subject"]]

    # Activity Presets
    preset_choice = st.selectbox(
        "⚡ Choose a Real Observation Preset (or select custom index):",
        [
            "Observation: LAYING (Static Recumbent)",
            "Observation: SITTING (Static Inactive)",
            "Observation: STANDING (Static Upright)",
            "Observation: WALKING (Dynamic Flat Motion)",
            "Observation: WALKING UPSTAIRS (High Cadence Ascending)",
            "Observation: WALKING DOWNSTAIRS (High Vertical Impact)",
            "Custom Observation Index"
        ]
    )

    if "LAYING" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "LAYING"].index[0]
    elif "SITTING" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "SITTING"].index[0]
    elif "STANDING" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "STANDING"].index[0]
    elif "WALKING UPSTAIRS" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "WALKING_UPSTAIRS"].index[0]
    elif "WALKING DOWNSTAIRS" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "WALKING_DOWNSTAIRS"].index[0]
    elif "WALKING" in preset_choice:
        row_idx = test_df[test_df["Activity"] == "WALKING"].index[0]
    else:
        row_idx = st.number_input("Observation Row Index (0 to 2946)", min_value=0, max_value=len(test_df)-1, value=100)

    # Base feature values from selected row
    active_row = test_df.iloc[row_idx].copy()
    actual_activity = active_row.get("Activity", "Unknown")

    st.info(f"Loaded test observation **#{row_idx}** (Actual Ground Truth Activity: **{actual_activity}**). You can tweak key sensor features below and observe how the cluster assignment shifts in real time.")

    # Section 1: Dynamic Body Acceleration
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>🏃</span> Dynamic Body Acceleration & Jerk Attributes</div>', unsafe_allow_html=True)
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        acc_mag_mean = st.slider(
            "tBodyAccMag-mean() [Overall Motion Energy]",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tBodyAccMag-mean()", -0.9)),
            step=0.02
        )
    with acol2:
        acc_mag_std = st.slider(
            "tBodyAccMag-std() [Motion Variance]",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tBodyAccMag-std()", -0.9)),
            step=0.02
        )
    with acol3:
        acc_jerk_mean = st.slider(
            "tBodyAccJerkMag-mean() [Sudden Impact/Jerk]",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tBodyAccJerkMag-mean()", -0.9)),
            step=0.02
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Rotational Gyroscope & Gravity Orientation
    st.markdown('<div class="bordered-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>🔄</span> Gyroscope & Gravity Orientation Attributes</div>', unsafe_allow_html=True)
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        gyro_mag_mean = st.slider(
            "tBodyGyroMag-mean()",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tBodyGyroMag-mean()", -0.9)),
            step=0.02
        )
    with gcol2:
        grav_x = st.slider(
            "tGravityAcc-mean()-X",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tGravityAcc-mean()-X", 0.0)),
            step=0.02
        )
    with gcol3:
        grav_y = st.slider(
            "tGravityAcc-mean()-Y",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("tGravityAcc-mean()-Y", 0.0)),
            step=0.02
        )
    with gcol4:
        angle_grav = st.slider(
            "angle(X,gravityMean)",
            min_value=-1.0, max_value=1.0,
            value=float(active_row.get("angle(X,gravityMean)", 0.0)),
            step=0.02
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Assign Cluster Button
    cluster_btn = st.button("📍 Assign Activity Cluster & Visualize", use_container_width=True)

    if cluster_btn or True:  # Default display
        # Construct updated sample
        sample_series = active_row[feature_cols].copy()
        if "tBodyAccMag-mean()" in sample_series: sample_series["tBodyAccMag-mean()"] = acc_mag_mean
        if "tBodyAccMag-std()" in sample_series: sample_series["tBodyAccMag-std()"] = acc_mag_std
        if "tBodyAccJerkMag-mean()" in sample_series: sample_series["tBodyAccJerkMag-mean()"] = acc_jerk_mean
        if "tBodyGyroMag-mean()" in sample_series: sample_series["tBodyGyroMag-mean()"] = gyro_mag_mean
        if "tGravityAcc-mean()-X()" in sample_series: sample_series["tGravityAcc-mean()-X()"] = grav_x
        if "tGravityAcc-mean()-Y()" in sample_series: sample_series["tGravityAcc-mean()-Y()"] = grav_y
        if "angle(X,gravityMean)" in sample_series: sample_series["angle(X,gravityMean)"] = angle_grav

        # Scale and predict cluster
        sample_df = pd.DataFrame([sample_series], columns=feature_cols)
        sample_scaled = scaler.transform(sample_df)
        predicted_cluster = int(km_model.predict(sample_scaled)[0])
        pca_coords = pca_model.transform(sample_scaled)[0]

        # Cluster metadata from notebook crosstab
        crosstab = har_benchmarks.get("activity_crosstab", {}).get(str(predicted_cluster), {})
        dominant_act = max(crosstab, key=crosstab.get) if crosstab else "Sensor Cluster"
        dominant_pct = crosstab.get(dominant_act, 0.0) * 100

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Selected Model: K-Means Clustering (k=6, Silhouette = 0.1030)</div>
                <div class="prediction-value">Assigned to Cluster {predicted_cluster}</div>
                <div class="prediction-meta">
                    Dominant Ground Truth Activity: <strong>{dominant_act}</strong> ({dominant_pct:.1f}% purity in test crosstab)
                    <br>2D PCA Coordinates: <code>PC1 = {pca_coords[0]:.2f}, PC2 = {pca_coords[1]:.2f}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 2D PCA Cluster Map & Activity Profile")
        ccol1, ccol2 = st.columns([1.2, 0.8])

        with ccol1:
            st.markdown("#### PCA 2D Cluster Space (PC1 vs PC2)")
            # Precompute or use sample for PCA visualization
            sample_size = min(800, len(test_df))
            sub_df = test_df.sample(sample_size, random_state=42)
            sub_scaled = scaler.transform(sub_df[feature_cols])
            sub_clusters = km_model.predict(sub_scaled)
            sub_pca = pca_model.transform(sub_scaled)

            fig, ax = plt.subplots(figsize=(7, 4.8))
            palette = sns.color_palette("tab10", 6)
            for k in range(6):
                mask = (sub_clusters == k)
                ax.scatter(
                    sub_pca[mask, 0], sub_pca[mask, 1],
                    label=f"Cluster {k}",
                    color=palette[k],
                    alpha=0.45,
                    s=22
                )

            # Highlight user sample
            ax.scatter(
                pca_coords[0], pca_coords[1],
                color="#dc2626",
                s=240,
                marker="*",
                edgecolor="white",
                linewidth=1.8,
                label="Your Active Sample ⭐",
                zorder=10
            )

            ax.set_title("Smartphone HAR Sensor Clusters in PCA Space", fontsize=11, fontweight="bold")
            ax.set_xlabel("Principal Component 1 (Motion Intensity)")
            ax.set_ylabel("Principal Component 2 (Postural Angle)")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with ccol2:
            st.markdown(f"#### Cluster {predicted_cluster} Activity Composition")
            if crosstab:
                act_df = pd.DataFrame([
                    {"Activity": k, "Proportion": v * 100}
                    for k, v in crosstab.items()
                ]).sort_values("Proportion", ascending=False)

                fig, ax = plt.subplots(figsize=(5, 3.8))
                sns.barplot(data=act_df, x="Proportion", y="Activity", palette="Blues_r", ax=ax)
                ax.set_xlabel("Percentage in Held-Out Test Split (%)")
                ax.set_ylabel("")
                ax.set_xlim(0, 100)
                for p in ax.patches:
                    val = p.get_width()
                    if val > 1:
                        ax.annotate(f"{val:.1f}%", (val + 1, p.get_y() + p.get_height()/2), va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.write(
                """
                **Official Capstone Evaluation:**
                - **Silhouette Score:** `0.1030` *(Favors K-Means over Ward 0.0970)*
                - **Calinski-Harabasz Index:** `744.81` *(Higher is better)*
                - **Davies-Bouldin Index:** `2.6595`
                """
            )
