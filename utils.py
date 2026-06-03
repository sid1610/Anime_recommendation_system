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


def build_or_load_models(df: pd.DataFrame):
    if COSINE_PKL.exists() and INDICES_PKL.exists() and VECTORIZER_PKL.exists():
        cosine_sim = joblib.load(COSINE_PKL)
        indices = joblib.load(INDICES_PKL)
        vectorizer = joblib.load(VECTORIZER_PKL)
        return cosine_sim, indices, vectorizer

    corpus = df['genre'].fillna("")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)
    cosine_sim = linear_kernel(tfidf, tfidf)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    joblib.dump(cosine_sim, COSINE_PKL)
    joblib.dump(indices, INDICES_PKL)
    joblib.dump(vectorizer, VECTORIZER_PKL)

    return cosine_sim, indices, vectorizer


def get_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices, topn=10):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: topn+1]
    anime_indices = [i[0] for i in sim_scores]
    return df.iloc[anime_indices]


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
