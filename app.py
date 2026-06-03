import streamlit as st
from pathlib import Path
from utils import load_dataset, build_or_load_models, get_recommendations, fetch_poster
import logging
import traceback

logging.basicConfig(level=logging.INFO)


ASSETS = Path("assets")
PLACEHOLDER = ASSETS / "placeholder.svg"


def main():
    try:
        st.markdown(
        """
        <style>
        .hero{padding:24px;border-radius:12px;background:linear-gradient(90deg,#071033,#092240);color:white}
        .card{background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));color:#e6eef8;padding:8px;border-radius:10px;text-align:left}
        .card img{width:100%;height:300px;object-fit:cover;border-radius:6px}
        .meta{padding-top:6px}
        .title{font-size:1.05rem;margin:0}
        .genres{font-size:0.85em;color:#9fb2d0;margin:0}
        .tag{display:inline-block;background:#083358;color:#cfe9ff;padding:4px 8px;border-radius:6px;margin-right:6px;margin-top:6px;font-size:0.75rem}
        </style>
        <div class="hero">
          <h1>Anime Recommendation System</h1>
          <p>Find anime similar to what you love — powered by TF-IDF and cosine similarity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

        try:
            df = load_dataset()
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure `anime_recommendation_data.csv` is in the repo root.")
            return

        cosine_sim, indices, _ = build_or_load_models(df)

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
                    if PLACEHOLDER.exists():
                        st.image(str(PLACEHOLDER), caption=row['name'], use_column_width=True)
                    else:
                        st.write(f"**{row['name']}**")
                st.markdown(f"<p class='genres'>{row.get('genre','')}</p>", unsafe_allow_html=True)
    except Exception as e:
        logging.exception("Unhandled exception in app")
        st.error("An unexpected error occurred while loading the app.")
        st.exception(e)
        st.text(traceback.format_exc())


if __name__ == '__main__':
    main()

