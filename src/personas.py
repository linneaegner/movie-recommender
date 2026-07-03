import numpy as np
import pandas as pd

from src.data import genre_profile

PERSONA_LABELS = {
    "heavy_rater": "Heavy rater",
    "blockbuster_fan": "Blockbuster fan",
    "niche_explorer": "Niche explorer",
    "genre_specialist": "Genre specialist",
    "balanced": "Balanced viewer",
}


def _history_novelty(items: pd.Series, item_popularity: pd.Series, total: int) -> float:
    scores = items.map(lambda item: -np.log2(item_popularity.get(item, 1) / total))
    return float(scores.mean())


def user_personas(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[int, str]:
    """Assign each user a readable viewing persona from rating behavior."""
    item_popularity = ratings.groupby("item").size()
    total = len(ratings)
    rows: list[dict] = []

    for user, group in ratings.groupby("user"):
        items = group["item"]
        profile = genre_profile(items, movies)
        rows.append(
            {
                "user": user,
                "n_ratings": len(group),
                "novelty": _history_novelty(items, item_popularity, total),
                "avg_popularity": float(items.map(item_popularity).mean()),
                "top_genre_share": float(profile.iloc[0]) if not profile.empty else 0.0,
                "genre_count": len(profile),
            }
        )

    stats = pd.DataFrame(rows)
    heavy_cutoff = stats["n_ratings"].quantile(0.75)
    pop_cutoff = stats["avg_popularity"].quantile(0.75)
    novelty_cutoff = stats["novelty"].quantile(0.75)

    personas: dict[int, str] = {}
    for row in stats.itertuples(index=False):
        if row.n_ratings >= heavy_cutoff:
            persona = "heavy_rater"
        elif row.novelty >= novelty_cutoff:
            persona = "niche_explorer"
        elif row.avg_popularity >= pop_cutoff:
            persona = "blockbuster_fan"
        elif row.top_genre_share >= 0.45 and row.genre_count <= 6:
            persona = "genre_specialist"
        else:
            persona = "balanced"
        personas[int(row.user)] = PERSONA_LABELS[persona]

    return personas
