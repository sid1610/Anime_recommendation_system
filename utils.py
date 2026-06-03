import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from urllib.parse import quote_plus
import requests
import streamlit as st

DATA_CSV = Path("anime_recommendation_data.csv")
COSINE_PKL = Path("anime_cosine_similarity.pkl")
INDICES_PKL = Path("anime_indices.pkl")
VECTORIZER_PKL = Path("anime_tfidf_vectorizer.pkl")


def load_dataset(path: Path = DATA_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["name"]).reset_index(drop=True)
    return df


def build_or_load_models(df: pd.DataFrame, force_rebuild: bool = False):
    """
    Load or build the text vectorizer and indices. We avoid precomputing
    or loading the full cosine similarity matrix (large) to prevent OOM
    on small deployment instances. Recommendations compute similarity
    on-demand using the vectorizer.
    Returns: (cosine_sim_or_none, indices, vectorizer)
    """
    # Prefer loading vectorizer + indices only
    if not force_rebuild and INDICES_PKL.exists() and VECTORIZER_PKL.exists():
        indices = joblib.load(INDICES_PKL)
        vectorizer = joblib.load(VECTORIZER_PKL)
        return None, indices, vectorizer

    # Build vectorizer and indices from corpus
    corpus = df['genre'].fillna("")
    vectorizer = TfidfVectorizer(stop_words='english')
    _ = vectorizer.fit_transform(corpus)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    joblib.dump(indices, INDICES_PKL)
    joblib.dump(vectorizer, VECTORIZER_PKL)

    return None, indices, vectorizer


@st.cache_data
def _compute_tfidf_matrix(df, _vectorizer):
    # `_vectorizer` is intentionally underscore-prefixed so Streamlit's
    # cache machinery does not attempt to hash the estimator object.
    corpus = df['genre'].fillna("")
    return _vectorizer.transform(corpus)


def get_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices, vectorizer, topn=10):
    """
    Returns top-N recommendations for a title. If a precomputed
    cosine_sim matrix is provided (backwards compatibility), use it;
    otherwise compute similarity on-demand using the vectorizer.
    """
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    if cosine_sim is not None:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: topn+1]
        anime_indices = [i[0] for i in sim_scores]
        return df.iloc[anime_indices]

    # On-demand compute similarities
    tfidf = _compute_tfidf_matrix(df, vectorizer)
    vec = tfidf[idx]
    sims = linear_kernel(vec, tfidf).flatten()
    ranked_idx = sims.argsort()[::-1]
    # remove the item itself
    ranked_idx = ranked_idx[ranked_idx != idx]
    top_idx = ranked_idx[:topn]
    return df.iloc[top_idx]


@st.cache_data
def fetch_poster(title: str):
    try:
        q = quote_plus(title)
        url = f"https://api.jikan.moe/v4/anime?q={q}&limit=1"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json().get('data')
            if data:
                img = data[0].get('images', {}).get('jpg', {}).get('image_url')
                return img
    except Exception:
        return None
    return None
