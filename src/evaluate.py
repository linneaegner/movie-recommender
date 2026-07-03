import numpy as np
import pandas as pd
from lenskit import batch

from src.metrics import (
    item_catalog_coverage,
    novelty,
    simulate_user_interactions,
    temporal_diversity,
)
from src.models import ALGORITHM_LABELS, build_recommenders

METRICS = ("novelty", "item_coverage", "temporal_diversity")


def run_evaluation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    n_iterations: int = 10,
    n_users: int = 100,
    n_items: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train recommenders over multiple iterations with simulated feedback.
    Returns per-iteration metrics and mean summary per algorithm.
    """
    rng = np.random.default_rng(seed)
    recommenders = build_recommenders()
    working_ratings = ratings.copy()
    history: list[dict[str, float]] = []

    for iteration in range(n_iterations):
        for model in recommenders.values():
            model.fit(working_ratings)

        selected_users = rng.choice(
            working_ratings["user"].unique(), size=n_users, replace=False
        )

        recommendations = {
            name: batch.recommend(recommenders[name], selected_users, n_items)
            for name in recommenders
        }

        iteration_metrics: dict[str, float] = {}
        for name, recs in recommendations.items():
            iteration_metrics[f"novelty_{name}"] = novelty(recs, working_ratings)
            iteration_metrics[f"item_coverage_{name}"] = item_catalog_coverage(
                recs, movies
            )
            iteration_metrics[f"temporal_diversity_{name}"] = temporal_diversity(
                recs, movies
            )

        history.append(iteration_metrics)

        for recs in recommendations.values():
            new_ratings = simulate_user_interactions(recs)
            if not new_ratings.empty:
                working_ratings = pd.concat(
                    [working_ratings, new_ratings], ignore_index=True
                )

    metrics_df = pd.DataFrame(history)
    summary_rows = []

    for metric in METRICS:
        for name in recommenders:
            column = f"{metric}_{name}"
            summary_rows.append(
                {
                    "metric": metric,
                    "algorithm": name,
                    "label": ALGORITHM_LABELS[name],
                    "mean": metrics_df[column].mean(),
                    "std": metrics_df[column].std(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    return metrics_df, summary


def best_algorithm(summary: pd.DataFrame, metric: str) -> str:
    """Return algorithm key with highest mean for a metric."""
    subset = summary[summary["metric"] == metric]
    row = subset.loc[subset["mean"].idxmax()]
    return str(row["algorithm"])
