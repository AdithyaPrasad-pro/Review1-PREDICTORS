"""
Group 12 Machine Learning Capstone
Track 3: Human Activity Recognition with Smartphones (Clustering)

Algorithms required by rubric:
1. K-Means Clustering (Elbow curve for k=2..10)
2. Agglomerative Hierarchical Clustering (Dendrogram, Linkage comparison: Ward, Complete, Average)

Evaluation metrics:
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Index (higher is better)

Visualizations:
- PCA 2D cluster projection
- t-SNE 2D cluster projection
- Elbow curve (Inertia vs. k)
- Dendrogram (linkage="ward")
- Cluster vs. Activity post-hoc interpretation matrix
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.preprocessing import prepare_har_data

RANDOM_STATE = 42

# Official benchmark results verified from notebooks/clustering.ipynb
VERIFIED_CLUSTERING_BENCHMARKS = [
    {
        "Algorithm": "K-Means (k=6)",
        "Clusters": 6,
        "Silhouette Score": 0.1030,
        "Davies-Bouldin Index": 2.3789,
        "Calinski-Harabasz Index": 724.81,
        "Status": "Baseline Partitional Model"
    },
    {
        "Algorithm": "Agglomerative (Ward)",
        "Clusters": 6,
        "Silhouette Score": 0.0970,
        "Davies-Bouldin Index": 2.5856,
        "Calinski-Harabasz Index": 705.17,
        "Status": "Primary Hierarchical Model"
    },
    {
        "Algorithm": "Agglomerative (Complete)",
        "Clusters": 6,
        "Silhouette Score": 0.3375,
        "Davies-Bouldin Index": 1.6284,
        "Calinski-Harabasz Index": 593.30,
        "Status": "Alternative Linkage"
    },
    {
        "Algorithm": "Agglomerative (Average)",
        "Clusters": 6,
        "Silhouette Score": 0.2954,
        "Davies-Bouldin Index": 1.0610,
        "Calinski-Harabasz Index": 312.44,
        "Status": "Alternative Linkage"
    }
]

VERIFIED_LINKAGE_COMPARISON = [
    {"Linkage": "ward", "Silhouette Score": 0.0970, "Davies-Bouldin Index": 2.5856, "Calinski-Harabasz Index": 705.17},
    {"Linkage": "complete", "Silhouette Score": 0.3375, "Davies-Bouldin Index": 1.6284, "Calinski-Harabasz Index": 593.30},
    {"Linkage": "average", "Silhouette Score": 0.2954, "Davies-Bouldin Index": 1.0610, "Calinski-Harabasz Index": 312.44}
]


def compute_clustering_metrics(X, labels):
    """Computes Silhouette, Davies-Bouldin, and Calinski-Harabasz internal validation metrics."""
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(X):
        return np.nan, np.nan, np.nan
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    return sil, db, ch


def run_kmeans_clustering(X_scaled, k=6):
    """Fits KMeans and returns model and labels."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil, db, ch = compute_clustering_metrics(X_scaled, labels)
    return km, labels, {"Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch}


def run_hierarchical_clustering(X_scaled, k=6, linkage_method="ward"):
    """Fits AgglomerativeClustering with specified linkage and returns labels & metrics."""
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = agg.fit_predict(X_scaled)
    sil, db, ch = compute_clustering_metrics(X_scaled, labels)
    return agg, labels, {"Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch}


def compute_elbow_curve(X_scaled, k_min=2, k_max=10):
    """Calculates KMeans inertias for k in range [k_min, k_max]."""
    k_range = list(range(k_min, k_max + 1))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
    return k_range, inertias


def plot_elbow_curve(k_range, inertias, chosen_k=6):
    """Generates K-Means elbow plot with marker on chosen k."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_range, inertias, marker="o", color="#2563eb", linewidth=2.2, markersize=7)
    
    # Highlight chosen k
    if chosen_k in k_range:
        idx = k_range.index(chosen_k)
        ax.plot(chosen_k, inertias[idx], marker="o", color="#dc2626", markersize=11, label=f"Chosen k = {chosen_k}")
        ax.axvline(chosen_k, linestyle="--", color="#dc2626", alpha=0.6)

    ax.set_title("K-Means Elbow Curve (Inertia vs. Number of Clusters)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_dendrogram(X_scaled, sample_size=500, linkage_method="ward"):
    """Generates a hierarchical dendrogram on a reproducible sample."""
    n_samples = min(sample_size, len(X_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_scaled), size=n_samples, replace=False)
    sample_data = X_scaled[sample_idx] if isinstance(X_scaled, np.ndarray) else X_scaled.iloc[sample_idx]

    Z = linkage(sample_data, method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True, ax=ax, color_threshold=None)
    ax.set_title(f"Agglomerative Dendrogram — {linkage_method.capitalize()} Linkage (Sample: {n_samples})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Merged Clusters / Observation Groups", fontsize=11)
    ax.set_ylabel("Linkage Distance Threshold", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def compute_pca_2d(X_scaled):
    """Computes 2-component PCA projection and returns coordinates + explained variance."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_exp = float(pca.explained_variance_ratio_.sum())
    return coords, var_exp, pca


def plot_pca_clusters(coords, labels, title="K-Means Cluster Projection in 2D PCA Space"):
    """Plots 2D PCA cluster visualization."""
    fig, ax = plt.subplots(figsize=(7, 5))
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("tab10", n_clusters)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=[palette[label % len(palette)] for label in labels],
        s=14, alpha=0.75, edgecolors="none"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Custom legend for clusters
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i % len(palette)], markersize=8, label=f"Cluster {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, title="Clusters", loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    return fig


def compute_and_plot_tsne(X_scaled, labels, max_samples=1200, perplexity=30):
    """Computes and plots 2D t-SNE visualization."""
    n_samples = min(max_samples, len(X_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=n_samples, replace=False)
    
    sample_data = X_scaled[idx] if isinstance(X_scaled, np.ndarray) else X_scaled.iloc[idx]
    sample_labels = labels[idx]

    # Pre-reduce with PCA to 50 dims for numerical stability
    pca_dim = min(50, sample_data.shape[1], n_samples - 1)
    pca_data = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit_transform(sample_data)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, n_iter=800)
    tsne_coords = tsne.fit_transform(pca_data)

    fig, ax = plt.subplots(figsize=(7, 5))
    n_clusters = len(np.unique(sample_labels))
    palette = sns.color_palette("tab10", n_clusters)

    ax.scatter(
        tsne_coords[:, 0], tsne_coords[:, 1],
        c=[palette[l % len(palette)] for l in sample_labels],
        s=14, alpha=0.8, edgecolors="none"
    )
    ax.set_title("t-SNE 2D Cluster Space (Perplexity = 30)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i % len(palette)], markersize=8, label=f"Cluster {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, title="Clusters", loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    return fig


def compute_activity_interpretation_crosstab(cluster_labels, activity_labels, cluster_name="Cluster"):
    """
    Computes row-normalized crosstab between fitted clusters and true activities.
    Labels are used exclusively post-fit for interpretation.
    """
    crosstab = pd.crosstab(
        pd.Series(cluster_labels, name=f"{cluster_name} ID"),
        activity_labels.rename("True Activity"),
        normalize="index"
    )
    return crosstab.round(3)
