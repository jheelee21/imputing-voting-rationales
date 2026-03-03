import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import WordNetLemmatizer

import umap
from sentence_transformers import SentenceTransformer


STOP_WORDS = set(_nltk_stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────


_HTML_RE = re.compile(r"<[^>]+>")
_UPPER_RE = re.compile(r"\b[A-Z]{2,}\b")  # drop all-caps abbreviations
_SPECIAL_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def preprocess(text: str, lemmatize: bool = True) -> str:
    """Lowercase, strip noise, remove stop-words, optionally lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _HTML_RE.sub(" ", text)
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(texts: pd.Series, lemmatize: bool = True) -> pd.Series:
    total = len(texts)
    processed = []
    for i, t in enumerate(texts):
        processed.append(preprocess(t, lemmatize))
        if (i + 1) % 5000 == 0:
            print(f"  Preprocessed {i + 1:,}/{total:,} texts …")
    return pd.Series(processed, index=texts.index)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VECTORISATION
# ─────────────────────────────────────────────────────────────────────────────


def build_tfidf(
    texts_clean: pd.Series, max_features: int = 8000, ngram_range: tuple = (1, 2)
) -> tuple:
    """
    Returns (X_sparse, vectorizer).
    X_sparse : scipy sparse matrix (n_docs × max_features)
    """
    print(f"\n[Vectorize] TF-IDF  max_features={max_features}  ngrams={ngram_range}")
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts_clean.fillna(""))
    print(f"  Matrix: {X.shape[0]:,} docs × {X.shape[1]:,} terms")
    return X, vec


def build_embeddings(
    texts_raw: pd.Series, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256
) -> np.ndarray:
    """
    Returns dense numpy array (n_docs × 384).
    Uses the raw (not cleaned) text – SBERT handles its own tokenisation.
    """
    print(f"\n[Vectorize] Sentence embeddings  model={model_name}")
    model = SentenceTransformer(model_name)
    texts = texts_raw.fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────


def reduce_svd(X_sparse, n_components: int = 100) -> np.ndarray:
    """LSA / TruncatedSVD on TF-IDF matrix."""
    n_comp = min(n_components, X_sparse.shape[1] - 1, X_sparse.shape[0] - 1)
    print(f"\n[Reduce] TruncatedSVD  n_components={n_comp}")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X_sparse)
    var_explained = svd.explained_variance_ratio_.sum()
    print(f"  Variance explained: {var_explained:.2%}")
    return normalize(X_red)  # L2-normalise for cosine-equivalent k-means


def reduce_umap(
    X_dense: np.ndarray,
    n_components: int = 15,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
) -> np.ndarray:
    """UMAP for dense embeddings (also used for 2-D visualisation)."""
    print(f"\n[Reduce] UMAP  n_components={n_components}  n_neighbors={n_neighbors}")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    X_red = reducer.fit_transform(X_dense)
    print(f"  Output shape: {X_red.shape}")
    return X_red


def reduce_umap_2d(X_dense: np.ndarray) -> np.ndarray:
    """Separate 2-D UMAP projection purely for plotting."""
    print("\n[Reduce] UMAP 2D (for visualisation only)")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    return reducer_2d.fit_transform(X_dense)
