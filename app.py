"""Streamlit demo: pick a user and algorithm, see movie recommendations."""

import streamlit as st

from src.data import dataset_ready, load_movielens, movie_titles
from src.models import ALGORITHM_LABELS, build_recommenders

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("Movie Recommender")
st.caption("MovieLens · LensKit · cognitive science ML project")

if not dataset_ready():
    st.error("Dataset not found.")
    st.markdown(
        """
Download MovieLens Small from the project folder:

```bash
bash scripts/download_data.sh
```

Then restart this app.
"""
    )
    st.stop()

ratings, movies = load_movielens()
user_ids = sorted(ratings["user"].unique())

with st.sidebar:
    st.header("Settings")
    algorithm = st.selectbox(
        "Algorithm",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda k: ALGORITHM_LABELS[k],
    )
    user_id = st.selectbox("User", user_ids, index=0)
    n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10)
    st.divider()
    st.markdown(
        "**About**  \nCompares random, popularity-based, and personalized "
        "collaborative filtering on the MovieLens dataset."
    )


@st.cache_resource(show_spinner="Training recommender…")
def get_recommendations(algorithm: str, user_id: int, n_recs: int):
    from lenskit import batch

    ratings_df, movies_df = load_movielens()
    model = build_recommenders()[algorithm]
    model.fit(ratings_df)
    recs = batch.recommend(model, [user_id], n_recs, n_jobs=1)
    return recs, movie_titles(movies_df), ratings_df, movies_df


recs, titles, ratings, movies = get_recommendations(algorithm, user_id, n_recs)

st.subheader(f"Top {n_recs} for user {user_id}")
st.write(f"Algorithm: **{ALGORITHM_LABELS[algorithm]}**")

if recs.empty:
    st.warning("No recommendations returned for this user.")
else:
    rows = []
    for rank, row in enumerate(recs.itertuples(index=False), start=1):
        item_id = row.item
        rows.append({"#": rank, "Movie": titles.get(item_id, f"Item {item_id}")})
    st.dataframe(rows, use_container_width=True, hide_index=True)

with st.expander("User rating history (sample)"):
    history = ratings[ratings["user"] == user_id].merge(movies, on="item")
    history = history.sort_values("rating", ascending=False).head(15)
    st.dataframe(
        history[["title", "rating", "genres"]],
        use_container_width=True,
        hide_index=True,
    )
