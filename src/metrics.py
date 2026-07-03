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


def novelty(recommendations: pd.DataFrame, ratings: pd.DataFrame) -> float:
    """Average self-information of recommended items (higher = less popular)."""
    item_popularity = ratings.groupby("item").size()
    total = len(ratings)
    scores = recommendations["item"].map(
        lambda item: -np.log2(item_popularity.get(item, 1) / total)
    )
    return float(scores.mean())


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
