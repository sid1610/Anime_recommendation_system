import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from urllib.parse import quote_plus


DATA_CSV = Path("anime_recommendation_data.csv")
COSINE_PKL = Path("anime_cosine_similarity.pkl")
INDICES_PKL = Path("anime_indices.pkl")
VECTORIZER_PKL = Path("anime_tfidf_vectorizer.pkl")


@st.cache_data
def load_dataset():
    if not DATA_CSV.exists():
        st.error(f"Dataset not found at {DATA_CSV}. Please place the CSV in the app folder.")
        return None
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["name"])  
    return df


def build_or_load_models(df):
    if COSINE_PKL.exists() and INDICES_PKL.exists() and VECTORIZER_PKL.exists():
        cosine_sim = joblib.load(COSINE_PKL)
        indices = joblib.load(INDICES_PKL)
        vectorizer = joblib.load(VECTORIZER_PKL)
        return cosine_sim, indices, vectorizer

    # Build vectorizer on genres (or name+genre) if pickles not present
    corpus = df['genre'].fillna("")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)
    cosine_sim = linear_kernel(tfidf, tfidf)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    joblib.dump(cosine_sim, COSINE_PKL)
    joblib.dump(indices, INDICES_PKL)
    joblib.dump(vectorizer, VECTORIZER_PKL)

    return cosine_sim, indices, vectorizer


@st.cache_data
def fetch_poster(title):
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


def get_recommendations(title, df, cosine_sim, indices, topn=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: topn+1]
    anime_indices = [i[0] for i in sim_scores]
    return df.iloc[anime_indices]


def main():
    st.set_page_config(page_title="Anime Recommendation System", layout="wide")

    st.markdown(
        """
        <style>
        .hero{padding:20px;border-radius:8px;background:linear-gradient(90deg,#0f172a,#0b1220);color:white}
        .card{background:#0b1220;color:#e6eef8;padding:8px;border-radius:8px;text-align:center}
        .card img{width:100%;height:260px;object-fit:cover;border-radius:6px}
        .genres{font-size:0.85em;color:#9fb2d0}
        .placeholder{width:100%;height:260px;background:#22324a;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#9fb2d0}
        </style>
        <div class="hero">
          <h1>Anime Recommendation System</h1>
          <p>Find anime similar to what you love — powered by TF-IDF and cosine similarity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    if df is None:
        return

    cosine_sim, indices, vectorizer = build_or_load_models(df)

    # Controls
    with st.sidebar:
        st.header("Search")
        name_list = df['name'].tolist()
        choice = st.selectbox("Choose anime", options=name_list, index=0)
        n = st.slider("Number of recommendations", 4, 20, 8)
        st.markdown("---")
        st.markdown("Tip: If a specific title isn't found, try selecting a similar popular title from the list.")

    # Show selection
    st.subheader(f"Results for '{choice}'")
    recs = get_recommendations(choice, df, cosine_sim, indices, topn=n)

    cols = st.columns(4)
    for i, (_, row) in enumerate(recs.iterrows()):
        col = cols[i % 4]
        with col:
            poster = fetch_poster(row['name'])
            if poster:
                st.image(poster, caption=row['name'], use_column_width=True)
            else:
                st.write(f"**{row['name']}**")
                st.write(row.get('genre', ''))


if __name__ == '__main__':
    main()
