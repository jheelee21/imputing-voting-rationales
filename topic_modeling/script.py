import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from embedding import (
    preprocess_series,
    build_tfidf,
    build_embeddings,
    reduce_svd,
    reduce_umap,
    reduce_umap_2d,
)
from clustering import (
    cluster_kmeans,
    sweep_kmeans,
    cluster_hdbscan,
    cluster_bertopic,
)

STOP_WORDS = set(_nltk_stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute standard clustering metrics (ignores noise label -1)."""
    mask = labels >= 0
    X_valid = X[mask]
    lbl_valid = labels[mask]

    if len(set(lbl_valid)) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

    sample = min(10_000, len(lbl_valid))
    metrics = {}
    try:
        metrics["silhouette"] = round(
            silhouette_score(X_valid, lbl_valid, sample_size=sample, random_state=42), 4
        )
    except Exception:
        metrics["silhouette"] = None

    try:
        metrics["davies_bouldin"] = round(davies_bouldin_score(X_valid, lbl_valid), 4)
    except Exception:
        metrics["davies_bouldin"] = None

    try:
        metrics["calinski_harabasz"] = round(
            calinski_harabasz_score(X_valid, lbl_valid), 2
        )
    except Exception:
        metrics["calinski_harabasz"] = None

    return metrics


def interpret_clusters(
    labels: np.ndarray,
    texts_clean: pd.Series,
    texts_raw: pd.Series,
    vectorizer: TfidfVectorizer,
    top_n_terms: int = 12,
    top_n_examples: int = 3,
) -> pd.DataFrame:
    """
    For each cluster: top TF-IDF terms + representative text examples.
    Returns a summary DataFrame.
    """
    print("\n" + "=" * 70)
    print("CLUSTER INTERPRETATION")
    print("=" * 70)

    terms = np.array(vectorizer.get_feature_names_out())
    rows = []

    for c in sorted(set(labels)):
        mask = labels == c
        tag = "NOISE" if c == -1 else f"Cluster {c}"
        n = mask.sum()

        # TF-IDF mean across cluster docs
        X_c = vectorizer.transform(texts_clean[mask].fillna(""))
        mean = X_c.mean(axis=0).A1
        top_i = mean.argsort()[::-1][:top_n_terms]
        top_terms_list = terms[top_i].tolist()
        top_weights = mean[top_i].tolist()

        # Example texts (pick highest-TF-IDF-score docs in cluster)
        doc_scores = X_c.sum(axis=1).A1
        example_idx = np.where(mask)[0][doc_scores.argsort()[::-1][:top_n_examples]]
        examples = [
            str(texts_raw.iloc[i])[:180].replace("\n", " ") for i in example_idx
        ]

        rows.append(
            {
                "cluster": c,
                "label": tag,
                "n_docs": int(n),
                "pct": round(100 * n / len(labels), 1),
                "top_terms": ", ".join(top_terms_list),
                "top_weights": [round(w, 4) for w in top_weights],
                "examples": examples,
            }
        )

        print(f"\n{'─' * 70}")
        print(f"  {tag}  |  n={n:,}  ({100 * n / len(labels):.1f}%)")
        print(f"  Top terms : {', '.join(top_terms_list)}")
        for j, ex in enumerate(examples, 1):
            print(f"  Example {j}: {ex[:160]} …")

    print("=" * 70)
    return pd.DataFrame(rows)


def cross_tab_keyword_groups(
    df_sub: pd.DataFrame, labels: np.ndarray, keyword_col: str = "rationale_group"
) -> pd.DataFrame:
    """
    If the keyword classifier output is present, cross-tab cluster vs keyword group.
    Helps validate / compare the two approaches.
    """
    tmp = df_sub[[keyword_col]].copy()
    tmp["cluster"] = labels
    ct = pd.crosstab(
        tmp["cluster"], tmp[keyword_col], margins=True, margins_name="TOTAL"
    )
    print(f"\n[Cross-tab] cluster × {keyword_col}")
    print(ct.to_string())
    return ct


def plot_umap_scatter(
    umap_2d: np.ndarray, labels: np.ndarray, title: str, output_path: str
):
    unique_labels = sorted(set(labels))
    palette = cm.get_cmap("tab20", max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        color = "lightgrey" if c == -1 else palette(i)
        lbl = "Noise" if c == -1 else f"Cluster {c}"
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            s=4,
            alpha=0.5,
            color=color,
            label=lbl,
            rasterized=True,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=8, loc="upper right", bbox_to_anchor=(1.18, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_cluster_sizes(labels: np.ndarray, title: str, output_path: str):
    counts = Counter(labels)
    keys = sorted(counts.keys())
    vals = [counts[k] for k in keys]
    xlabels = ["Noise" if k == -1 else f"C{k}" for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.8), 5))
    bars = ax.bar(xlabels, vals, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of documents")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def read_data(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".dta":
        return pd.read_stata(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".pkl":
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def write_data(df: pd.DataFrame, path: str):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext == ".dta":
        df.to_stata(path, write_index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def run(args):
    out_dir = Path(args.output).parent
    stem = Path(args.output).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 1 – LOAD DATA")
    print(f"{'=' * 70}")
    df = read_data(args.input)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    df_sub = df

    # ── Preprocess ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 3 – PREPROCESS TEXT")
    print(f"{'=' * 70}")
    texts_raw = df_sub[args.rat_col]
    texts_clean = preprocess_series(texts_raw, lemmatize=True)
    df_sub["_text_clean"] = texts_clean

    # ── Vectorise ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"STEP 4 – VECTORISE  ({args.vectorizer})")
    print(f"{'=' * 70}")

    vectorizer = (
        None  # always build TF-IDF for interpretation even if clustering on embeds
    )

    if args.vectorizer == "tfidf" or args.method != "bertopic":
        X_tfidf, vectorizer = build_tfidf(texts_clean, max_features=args.max_features)

    if args.vectorizer == "embeddings":
        X_embed = build_embeddings(texts_raw, model_name=args.embed_model)

    # ── Reduce ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 5 – REDUCE DIMENSIONS")
    print(f"{'=' * 70}")

    if args.vectorizer == "tfidf":
        X_red = reduce_svd(X_tfidf, n_components=args.svd_components)
    else:
        X_red = reduce_umap(X_embed, n_components=args.umap_components)

    # ── Cluster ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"STEP 6 – CLUSTER  ({args.method})")
    print(f"{'=' * 70}")

    sweep_df = None
    topic_model = None

    if args.method == "kmeans":
        if args.sweep_k:
            best_k, sweep_df = sweep_kmeans(
                X_red, k_range=range(args.k_min, args.k_max + 1)
            )
            labels = cluster_kmeans(X_red, k=best_k)
        else:
            labels = cluster_kmeans(X_red, k=args.k)

    elif args.method == "hdbscan":
        labels = cluster_hdbscan(
            X_red,
            min_cluster_size=args.hdbscan_min_size,
            min_samples=args.hdbscan_min_samples,
        )

    elif args.method == "bertopic":
        labels, topic_model = cluster_bertopic(texts_raw, nr_topics=args.k)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 7 – EVALUATE")
    print(f"{'=' * 70}")
    metrics = evaluate_clusters(X_red, labels)
    print(f"  Silhouette        : {metrics['silhouette']}")
    print(f"  Davies-Bouldin    : {metrics['davies_bouldin']}  (lower = better)")
    print(f"  Calinski-Harabasz : {metrics['calinski_harabasz']}  (higher = better)")

    # ── Interpret ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 8 – INTERPRET CLUSTERS")
    print(f"{'=' * 70}")

    if vectorizer is not None and args.method != "bertopic":
        summary_df = interpret_clusters(
            labels,
            texts_clean,
            texts_raw,
            vectorizer,
            top_n_terms=args.top_terms,
        )
    else:
        summary_df = pd.DataFrame()

    # ── Cross-tab vs keyword groups ───────────────────────────────────────────
    ct_df = None
    if "rationale_group" in df_sub.columns:
        ct_df = cross_tab_keyword_groups(df_sub, labels)

    # ── Visualise ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 9 – VISUALISE")
    print(f"{'=' * 70}")

    # 2D UMAP scatter
    if args.vectorizer == "tfidf":
        print("  Building 2D UMAP from SVD output …")
        umap_2d = reduce_umap_2d(X_red)
    else:
        umap_2d = reduce_umap_2d(X_embed)

    scatter_path = str(out_dir / f"{stem}_umap_scatter.png")
    bar_path = str(out_dir / f"{stem}_cluster_sizes.png")

    plot_umap_scatter(
        umap_2d,
        labels,
        title=f"Rationale Clusters ({args.method}, n={len(labels):,})",
        output_path=scatter_path,
    )
    plot_cluster_sizes(
        labels, title=f"Cluster Sizes — {args.method}", output_path=bar_path
    )

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 10 – SAVE OUTPUTS")
    print(f"{'=' * 70}")

    # Drop internal clean-text column from df_sub (not needed in output)
    if "_text_clean" in df.columns:
        df = df.drop(columns=["_text_clean"])

    write_data(df, args.output)
    print(f"  Enriched dataset  : {args.output}")

    # Summary CSV
    if not summary_df.empty:
        summary_path = str(out_dir / f"{stem}_cluster_summary.csv")
        summary_df.drop(columns=["top_weights", "examples"], errors="ignore").to_csv(
            summary_path, index=False
        )
        print(f"  Cluster summary   : {summary_path}")

    # Sweep CSV
    if sweep_df is not None:
        sweep_path = str(out_dir / f"{stem}_sweep_k.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"  K sweep results   : {sweep_path}")

    # Cross-tab CSV
    if ct_df is not None:
        ct_path = str(out_dir / f"{stem}_crosstab_keyword.csv")
        ct_df.to_csv(ct_path)
        print(f"  Cross-tab         : {ct_path}")

    # Metrics
    metrics_path = str(out_dir / f"{stem}_metrics.csv")
    pd.DataFrame(
        [
            {
                **metrics,
                "method": args.method,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            }
        ]
    ).to_csv(metrics_path, index=False)
    print(f"  Metrics           : {metrics_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text clustering / topic modelling for voting rationales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input",
        default="embeddings/rationale_text.csv",
        help="Input data file (csv/dta/parquet/xlsx)",
    )
    parser.add_argument(
        "--output",
        default="embeddings/embedded_rationale_text.csv",
        help="Output enriched file",
    )
    parser.add_argument(
        "--rat_col", default="rationale", help="Rationale text column name"
    )

    # Method
    parser.add_argument(
        "--method",
        default="kmeans",
        choices=["kmeans", "hdbscan", "bertopic"],
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--vectorizer",
        default="tfidf",
        choices=["tfidf", "embeddings"],
        help="Text vectorisation method",
    )
    parser.add_argument(
        "--embed_model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (used with --vectorizer embeddings)",
    )

    # K-Means options
    parser.add_argument(
        "--k", type=int, default=6, help="Number of clusters (K-Means / BERTopic)"
    )
    parser.add_argument(
        "--sweep_k", action="store_true", help="Auto-select k via silhouette sweep"
    )
    parser.add_argument("--k_min", type=int, default=3, help="Minimum k for sweep")
    parser.add_argument("--k_max", type=int, default=12, help="Maximum k for sweep")

    # HDBSCAN options
    parser.add_argument(
        "--hdbscan_min_size", type=int, default=40, help="HDBSCAN min_cluster_size"
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=10, help="HDBSCAN min_samples"
    )

    # Vectorisation options
    parser.add_argument(
        "--max_features", type=int, default=8000, help="TF-IDF max vocabulary size"
    )
    parser.add_argument(
        "--svd_components",
        type=int,
        default=100,
        help="SVD n_components (TF-IDF reduction)",
    )
    parser.add_argument(
        "--umap_components",
        type=int,
        default=15,
        help="UMAP n_components (embedding reduction)",
    )

    # Output options
    parser.add_argument(
        "--top_terms", type=int, default=12, help="Top TF-IDF terms shown per cluster"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
