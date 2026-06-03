import pandas as pd
from utils import build_or_load_models, get_recommendations


def test_build_and_recommend():
    data = {
        'anime_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'genre': ['Action, Drama', 'Comedy, School', 'Action, Comedy']
    }
    df = pd.DataFrame(data)
    cosine_sim, indices, vect = build_or_load_models(df)
    assert cosine_sim.shape[0] == 3
    recs = get_recommendations('A', df, cosine_sim, indices, topn=2)
    assert not recs.empty
