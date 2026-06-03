import pandas as pd
from utils import build_or_load_models, get_recommendations


def test_build_and_recommend():
    data = {
        'anime_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'genre': ['Action, Drama', 'Comedy, School', 'Action, Comedy']
    }
    df = pd.DataFrame(data)
    cosine_sim, indices, vect = build_or_load_models(df, force_rebuild=True)
    assert len(indices) == 3
    recs = get_recommendations('A', df, cosine_sim, indices, vect, topn=2)
    assert not recs.empty
