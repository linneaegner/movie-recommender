import numpy as np
import pandas as pd


def temporal_diversity(recommendations: pd.DataFrame, movies: pd.DataFrame) -> float:
    """Spread of release years among recommended items."""
    enriched = movies.copy()
    enriched["year"] = enriched["title"].str.extract(r"\((\d{4})\)")
    enriched["year"] = pd.to_numeric(enriched["year"])
    movie_years = enriched.set_index("item")["year"]
    rec_years = recommendations["item"].map(movie_years)
    return float(rec_years.std())


def item_catalog_coverage(recommendations: pd.DataFrame, movies: pd.DataFrame) -> float:
    """Share of the catalog that appears in recommendations at least once."""
    unique_items = recommendations["item"].nunique()
    return unique_items / len(movies)


def catalog_coverage_summary(
    recommendations: pd.DataFrame, movies: pd.DataFrame
) -> str:
    """Human-readable catalog coverage for UI and reports."""
    unique_items = recommendations["item"].nunique()
    total = len(movies)
    pct = 100 * unique_items / total
    return f"{unique_items:,} of {total:,} movies ({pct:.2f}%)"


def novelty(recommendations: pd.DataFrame, ratings: pd.DataFrame) -> float:
    """Average self-information of recommended items (higher = less popular)."""
    item_popularity = ratings.groupby("item").size()
    total = len(ratings)
    scores = recommendations["item"].map(
        lambda item: -np.log2(item_popularity.get(item, 1) / total)
    )
    return float(scores.mean())


def genre_match_rate(
    recommendations: pd.DataFrame,
    user_items: pd.Series,
    movies: pd.DataFrame,
    *,
    top_user_genres: int = 5,
) -> float:
    """
    Share of recommended items that match at least one of the user's top genres.
    Per-user signal for whether a list aligns with taste (0–1).
    """
    from src.data import genre_profile, split_genres

    user_genres = set(genre_profile(user_items, movies).head(top_user_genres).index)
    if not user_genres:
        return 0.0

    movie_genres = movies.set_index("item")["genres"]
    matches = 0
    for item in recommendations["item"]:
        genres = set(split_genres(movie_genres.get(item, "")))
        if user_genres & genres:
            matches += 1

    return matches / len(recommendations) if len(recommendations) else 0.0


def mean_interactions(recommendations: pd.DataFrame, ratings: pd.DataFrame) -> float:
    """Average historical interaction count for recommended items."""
    item_interactions = ratings.groupby("item").size()
    return float(recommendations["item"].map(item_interactions).mean())


def simulate_user_interactions(
    recommendations: pd.DataFrame, interaction_prob: float = 0.2
) -> pd.DataFrame:
    """Simulate users rating a subset of recommended items (for iterative evaluation)."""
    new_ratings = []
    timestamp = pd.Timestamp.now().timestamp()
    for _, row in recommendations.iterrows():
        if np.random.random() < interaction_prob:
            rating = np.random.choice([3.5, 4.0, 4.5, 5.0], p=[0.2, 0.3, 0.3, 0.2])
            new_ratings.append(
                {
                    "user": row["user"],
                    "item": row["item"],
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    return pd.DataFrame(new_ratings)
