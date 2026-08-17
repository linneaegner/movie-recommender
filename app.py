"""Streamlit demo: compare recommender algorithms side by side."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lenskit import batch

from src.algorithm_guide import ALGORITHM_PROFILES
from src.blend import blend_recommendations
from src.data import dataset_ready, genre_profile, load_movielens, movie_titles
from src.metrics import genre_match_rate, novelty
from src.models import ALGORITHM_LABELS, build_recommenders
from src.personas import user_personas

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

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
    match = genre_match_rate(recs, history_items, movies)
    st.caption(
        f"Novelty {novelty(recs, ratings):.2f} · "
        f"Genre match {match:.0%} (overlap with this user's top genres)"
    )


@st.cache_data
def load_offline_summaries() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    ranking_path = OUTPUT_DIR / "ranking_summary.csv"
    diversity_path = OUTPUT_DIR / "metrics_summary.csv"
    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else None
    diversity = pd.read_csv(diversity_path) if diversity_path.exists() else None
    return ranking, diversity


def offline_metric(summary: pd.DataFrame | None, metric: str, algorithm: str) -> str:
    if summary is None:
        return "—"
    row = summary[(summary["metric"] == metric) & (summary["algorithm"] == algorithm)]
    if row.empty:
        return "—"
    value = row.iloc[0]
    return f"{value['mean']:.3f}"


recommendations = recommend_all(user_id, n_recs)
titles = movie_titles(movies)
user_history = ratings[ratings["user"] == user_id]
history_items = user_history["item"]
history_profile = genre_profile(history_items, movies)

compare_tab, blend_tab, guide_tab = st.tabs(
    ["Compare algorithms", "Filter bubble", "Algorithm guide"]
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

with guide_tab:
    st.subheader("Are the recommendations good?")
    st.write(
        "Two layers of evaluation: **offline quality** (do picks match held-out "
        "ratings across all users?) and **per-user taste fit** (genre match in "
        "the Compare tab). Neither alone tells the full story."
    )

    ranking_summary, diversity_summary = load_offline_summaries()
    if ranking_summary is None or diversity_summary is None:
        st.info(
            "Run `python run_all_evaluations.py` to generate offline score tables."
        )
    else:
        scorecard = []
        for name in ALGORITHM_PROFILES:
            scorecard.append(
                {
                    "Algorithm": ALGORITHM_PROFILES[name].label,
                    "NDCG@10": offline_metric(ranking_summary, "ndcg", name),
                    "Hit rate": offline_metric(ranking_summary, "hit", name),
                    "Novelty": offline_metric(diversity_summary, "novelty", name),
                    "Coverage": offline_metric(
                        diversity_summary, "item_coverage", name
                    ),
                }
            )
        st.dataframe(pd.DataFrame(scorecard), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Pros & cons by approach")

    for profile in ALGORITHM_PROFILES.values():
        with st.expander(profile.label, expanded=False):
            st.markdown(f"**Idea:** {profile.idea}")
            st.markdown(f"**Best for:** {profile.best_for}")
            st.markdown(f"**Offline quality:** {profile.offline_quality}")
            left, right = st.columns(2)
            with left:
                st.markdown("**Pros**")
                for pro in profile.pros:
                    st.markdown(f"- {pro}")
            with right:
                st.markdown("**Cons**")
                for con in profile.cons:
                    st.markdown(f"- {con}")

with st.expander("User rating history (sample)"):
    history = user_history.merge(movies, on="item").sort_values(
        "rating", ascending=False
    )
    st.dataframe(
        history[["title", "rating", "genres"]].head(15),
        hide_index=True,
        width="stretch",
    )
