# Anime Recommendation System

This repository contains a Streamlit app that provides anime recommendations using TF-IDF on genres and cosine similarity.

Files added:

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme settings

How to run locally:

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Place `anime_recommendation_data.csv` and your model pickles (`anime_cosine_similarity.pkl`, `anime_indices.pkl`, `anime_tfidf_vectorizer.pkl`) in the repository root. If pickles are missing, the app will build them on first run.

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Deploy to Streamlit Cloud:

1. Create a GitHub repository (or use the provided one).
2. Push this folder to the repo.
3. On Streamlit Cloud, create a new app and connect it to the repository and branch. Ensure `requirements.txt` is present.

Suggested git commands to push:

```bash
git init
git add .
git commit -m "Add Streamlit anime recommendation app"
git remote add origin https://github.com/sid1610/Anime_recommendation_system.git
git push -u origin main
```
