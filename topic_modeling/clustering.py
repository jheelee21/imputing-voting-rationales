import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import WordNetLemmatizer

import hdbscan
import matplotlib

from bertopic import BERTopic

STOP_WORDS = set(_nltk_stopwords.words("english"))


def cluster_kmeans(X: np.ndarray, k: int = 6, random_state: int = 42) -> np.ndarray:
    print(f"\n[Cluster] K-Means  k={k}")
    km = KMeans(n_clusters=k, n_init=15, max_iter=500, random_state=random_state)
    labels = km.fit_predict(X)
    print(f"  Cluster sizes: {dict(sorted(Counter(labels).items()))}")
    return labels


def sweep_kmeans(X: np.ndarray, k_range: range = range(3, 12)) -> int:
    """Try multiple k values, return k with best silhouette score."""
    print(f"\n[Sweep] K-Means k ∈ [{k_range.start}, {k_range.stop - 1}]")
    best_k, best_score = k_range.start, -1
    rows = []
    sample = min(5000, X.shape[0])
    idx = np.random.choice(X.shape[0], sample, replace=False)
    X_sample = X[idx]

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        lbl = km.fit_predict(X_sample)
        if len(set(lbl)) < 2:
            continue
        sil = silhouette_score(X_sample, lbl, sample_size=min(2000, sample))
        db = davies_bouldin_score(X_sample, lbl)
        ch = calinski_harabasz_score(X_sample, lbl)
        rows.append(
            {"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
        )
        print(f"  k={k:2d}  sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")
        if sil > best_score:
            best_score, best_k = sil, k

    print(f"  → Best k = {best_k}  (silhouette = {best_score:.4f})")
    return best_k, pd.DataFrame(rows)


def cluster_hdbscan(
    X: np.ndarray, min_cluster_size: int = 40, min_samples: int = 10
) -> np.ndarray:
    print(f"\n[Cluster] HDBSCAN  min_cluster_size={min_cluster_size}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    noise = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(
        f"  Clusters found: {n_clusters}  |  Noise points: {noise:,} ({100 * noise / len(labels):.1f}%)"
    )
    print(f"  Cluster sizes: {dict(sorted(Counter(labels[labels >= 0]).items()))}")
    return labels


def cluster_bertopic(texts_raw: pd.Series, nr_topics: int = 8) -> tuple:
    print(f"\n[Cluster] BERTopic  nr_topics={nr_topics}")
    topic_model = BERTopic(
        nr_topics=nr_topics, calculate_probabilities=False, verbose=True
    )
    topics, _ = topic_model.fit_transform(texts_raw.fillna("").tolist())
    info = topic_model.get_topic_info()
    print(info.head(nr_topics + 2).to_string(index=False))
    return np.array(topics), topic_model
