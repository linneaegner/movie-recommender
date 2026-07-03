"""Streamlit demo: compare recommender algorithms side by side."""

import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lenskit import batch

from src.blend import blend_recommendations
from src.data import dataset_ready, genre_profile, load_movielens, movie_titles
from src.metrics import item_catalog_coverage, novelty
from src.models import ALGORITHM_LABELS, build_recommenders
from src.personas import user_personas

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("Movie Recommender")
st.caption(
    "Compare how random, popularity, and personalized algorithms trade off "
    "familiarity, relevance, and discovery."
)

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
personas = user_personas(ratings, movies)
user_ids = sorted(ratings["user"].unique())

with st.sidebar:
    st.header("Settings")
    user_id = st.selectbox(
        "User",
        user_ids,
        index=0,
        format_func=lambda uid: f"User {uid} · {personas[uid]}",
    )
    n_recs = st.slider("Recommendations per list", min_value=5, max_value=15, value=8)
    st.divider()
    st.markdown(
        "**How to read this**  \n"
        "Each column shows a different strategy for the same user. "
        "Look for overlap (safe hits) vs. surprises (niche picks)."
    )
    with st.expander("Why does coverage look tiny?"):
        st.markdown(
            f"""
**Catalog coverage** is `unique recommended movies ÷ full catalog`.

MovieLens Small has **{len(movies):,} movies**. A list of {n_recs} picks can
cover at most **{100 * n_recs / len(movies):.2f}%** of the catalog — so
values like `0.001` are normal for one user.

Coverage matters when comparing algorithms **across many users** in
`run_evaluation.py` (100 users × 20 recs per round). There, random
explores widely while most-popular keeps hitting the same blockbusters.
"""
        )


@st.cache_resource(show_spinner="Training all recommenders…")
def load_fitted_models():
    ratings_df, movies_df = load_movielens()
    recommenders = build_recommenders()
    for model in recommenders.values():
        model.fit(ratings_df)
    return recommenders, ratings_df, movies_df


@st.cache_data(show_spinner="Generating recommendations…")
def recommend_all(user_id: int, n_recs: int) -> dict[str, pd.DataFrame]:
    recommenders, _, _ = load_fitted_models()
    return {
        name: batch.recommend(model, [user_id], n_recs, n_jobs=1)
        for name, model in recommenders.items()
    }


def recommendation_rows(recs: pd.DataFrame, titles: dict[int, str]) -> list[dict]:
    rows = []
    for rank, row in enumerate(recs.itertuples(index=False), start=1):
        item_id = row.item
        rows.append({"#": rank, "Movie": titles.get(item_id, f"Item {item_id}")})
    return rows


def plot_genre_comparison(
    history_profile: pd.Series,
    rec_profiles: dict[str, pd.Series],
    top_n: int = 8,
) -> plt.Figure:
    genres = history_profile.head(top_n).index.tolist()
    for profile in rec_profiles.values():
        for genre in profile.head(top_n).index:
            if genre not in genres:
                genres.append(genre)
    genres = genres[:top_n]

    x = range(len(genres))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        [pos - 2 * width for pos in x],
        [history_profile.get(genre, 0) for genre in genres],
        width=width,
        label="Your history",
    )

    for index, (name, profile) in enumerate(rec_profiles.items()):
        offset = (index - 1) * width
        ax.bar(
            [pos + offset for pos in x],
            [profile.get(genre, 0) for genre in genres],
            width=width,
            label=ALGORITHM_LABELS[name],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(genres, rotation=30, ha="right")
    ax.set_ylabel("Share of titles")
    ax.set_title("Genre mix: your history vs. each algorithm")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def render_metric_caption(recs: pd.DataFrame) -> None:
    st.caption(
        f"Novelty {novelty(recs, ratings):.2f} · "
        f"{recs['item'].nunique()} unique picks · "
        f"Catalog reach {item_catalog_coverage(recs, movies):.4f}"
    )


recommendations = recommend_all(user_id, n_recs)
titles = movie_titles(movies)
user_history = ratings[ratings["user"] == user_id]
history_items = user_history["item"]
history_profile = genre_profile(history_items, movies)

compare_tab, blend_tab, blind_tab = st.tabs(
    ["Compare algorithms", "Filter bubble", "Blind taste test"]
)

with compare_tab:
    st.subheader(f"Side-by-side recommendations for user {user_id}")
    st.write(f"Persona: **{personas[user_id]}**")

    columns = st.columns(len(ALGORITHM_LABELS))
    for column, (name, label) in zip(columns, ALGORITHM_LABELS.items()):
        recs = recommendations[name]
        with column:
            st.markdown(f"**{label}**")
            if recs.empty:
                st.warning("No recommendations.")
                continue

            render_metric_caption(recs)
            st.dataframe(
                recommendation_rows(recs, titles),
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.subheader("Genre mix")

    rec_profiles = {
        name: genre_profile(recs["item"], movies)
        for name, recs in recommendations.items()
    }

    if history_profile.empty:
        st.info("This user has no genre data to compare.")
    else:
        st.pyplot(
            plot_genre_comparison(history_profile, rec_profiles),
            clear_figure=True,
        )
        st.caption(
            "Bars show each genre's share within a list. Personalized algorithms "
            "often mirror your history; popularity-based picks skew toward mainstream genres."
        )

with blend_tab:
    st.subheader("Filter bubble slider")
    st.write(
        "Blend **Personal KNN** (tailored) with **Most popular** (familiar). "
        "Slide toward discovery or toward safe hits."
    )

    personal_pct = st.slider(
        "Personalization",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        format="%d%%",
        help="0% = all blockbusters, 100% = fully personalized.",
    )
    personal_weight = personal_pct / 100

    blended = blend_recommendations(
        recommendations["personal_knn"],
        recommendations["most_popular"],
        personal_weight=personal_weight,
        n=n_recs,
    )

    blend_cols = st.columns(3)
    lists = [
        ("Most popular", recommendations["most_popular"]),
        (f"Blend ({personal_weight:.0%} personal)", blended),
        ("Personal KNN", recommendations["personal_knn"]),
    ]
    for column, (label, recs) in zip(blend_cols, lists):
        with column:
            st.markdown(f"**{label}**")
            render_metric_caption(recs)
            st.dataframe(
                recommendation_rows(recs, titles),
                hide_index=True,
                width="stretch",
            )

with blind_tab:
    st.subheader("Which list would you click?")
    st.write("Two anonymous lists — pick the one you'd rather watch from.")

    if "blind_algorithms" not in st.session_state:
        st.session_state.blind_algorithms = random.sample(
            list(ALGORITHM_LABELS.keys()), 2
        )
        st.session_state.blind_revealed = False

    left_name, right_name = st.session_state.blind_algorithms
    left_recs = recommendations[left_name]
    right_recs = recommendations[right_name]

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**List A**")
        st.dataframe(
            recommendation_rows(left_recs, titles),
            hide_index=True,
            width="stretch",
        )
    with right_col:
        st.markdown("**List B**")
        st.dataframe(
            recommendation_rows(right_recs, titles),
            hide_index=True,
            width="stretch",
        )

    pick_left, pick_right, new_round = st.columns(3)
    if pick_left.button("I'd pick List A", width="stretch"):
        st.session_state.blind_revealed = True
        st.session_state.blind_choice = left_name
    if pick_right.button("I'd pick List B", width="stretch"):
        st.session_state.blind_revealed = True
        st.session_state.blind_choice = right_name
    if new_round.button("New round", width="stretch"):
        st.session_state.blind_algorithms = random.sample(
            list(ALGORITHM_LABELS.keys()), 2
        )
        st.session_state.blind_revealed = False
        st.session_state.pop("blind_choice", None)
        st.rerun()

    if st.session_state.get("blind_revealed"):
        choice = st.session_state["blind_choice"]
        other = right_name if choice == left_name else left_name
        st.success(
            f"You picked **{ALGORITHM_LABELS[choice]}**. "
            f"The other list was **{ALGORITHM_LABELS[other]}**."
        )

with st.expander("User rating history (sample)"):
    history = user_history.merge(movies, on="item").sort_values(
        "rating", ascending=False
    )
    st.dataframe(
        history[["title", "rating", "genres"]].head(15),
        hide_index=True,
        width="stretch",
    )
