# Machine Learning Capstone Project (23CSE301)
## Interactive Multi-Track Streamlit Deployment Dashboard

This repository hosts the **Machine Learning Capstone Project** for **Group 12**, featuring interactive web deployment of the final selected best models across all three academic tracks.

---

## 🎯 Capstone Problem Tracks & Deployed Models

| Track | Problem & Dataset | Selected Best Model | Verified Test Metrics |
| :--- | :--- | :--- | :--- |
| **Track 1: Regression** | **Diamond Price Prediction**<br>*(53,940 rows, 10 attributes)* | **Random Forest Regressor** | **R²: 0.9778**<br>RMSE: \$595.84<br>MAE: \$308.91<br>*(5-Fold CV R²: 0.9774)* |
| **Track 2: Classification** | **MAGIC Gamma Telescope**<br>*(19,020 events, 10 continuous shower features)* | **Random Forest Classifier** | **Accuracy: 88.77%**<br>Weighted F1: 0.8861<br>Precision: 88.74%<br>ROC-AUC: 0.9400 |
| **Track 3: Clustering** | **Human Activity Recognition (HAR)**<br>*(2,947 observations × 561 sensor features)* | **K-Means Clustering ($k=6$)** | **Silhouette Score: 0.1030**<br>Calinski-Harabasz: 744.81<br>Davies-Bouldin: 2.6595 |

---

## 🚀 Key Features of the Deployment App

- **Single Best Model Per Track**: Strictly focuses on inference, interactive feature tweaking, and diagnostics for the top model selected in each notebook without re-training all 10 baseline algorithms.
- **Interactive Inputs**: Numerical steppers, sliders, dropdown selects, and **Quick-Fill Presets** for one-click testing of realistic profiles.
- **Real-Time Visualizations**:
  - **Regression**: Gini Feature Importance chart and confidence interval bounds ($\pm \$308.91$ MAE).
  - **Classification**: Cherenkov shower class probability distribution bar (`P(Gamma)` vs `P(Hadron)`).
  - **Clustering**: 2D PCA Cluster Map with held-out test data and the user's active sample marked with a red star ($\star$), plus ground-truth activity composition bar charts.
- **Streamlit Community Cloud Ready**: Configured with cached model loaders (`@st.cache_resource`), compressed model weights, and clean light UI theme.

---

## 📁 Repository Structure

```text
├── streamlit_app.py          # Main Streamlit web application entry point
├── requirements.txt          # Python dependencies for deployment
├── .streamlit/
│   └── config.toml           # Theme configuration (light modern palette)
├── models/                   # Serialized preprocessors, clippers & compressed models
│   ├── diamond_rf_model.joblib
│   ├── diamond_preprocessor.joblib
│   ├── diamond_clipper.joblib
│   ├── magic_rf_model.joblib
│   ├── magic_preprocessor.joblib
│   ├── har_kmeans_model.joblib
│   ├── har_pca_model.joblib
│   └── har_scaler.joblib
├── notebooks/                # Unmodified original academic notebooks
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── src/                      # Feature engineering, preprocessing & inference logic
│   ├── preprocessing.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   └── utils.py
└── data/                     # Evaluation datasets
```

---

## 💻 Running the App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdithyaPrasad-pro/Review1-PREDICTORS.git
   cd Review1-PREDICTORS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit dashboard:**
   ```bash
   python -m streamlit run streamlit_app.py
   ```

4. Open `http://localhost:8501` in your web browser.

---

## 👥 Project Team

- **Adithya Prasad**
- **Rahul Reddy**
- **Pushkar Mallella**

Machine Learning Capstone • Course Code: **23CSE301**
