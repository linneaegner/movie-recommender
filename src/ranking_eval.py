"""Hold-out ranking evaluation with standard top-N metrics."""

from __future__ import annotations

import pandas as pd
from lenskit import batch, crossfold as xf, topn, util
from lenskit.algorithms import Recommender

from src.models import ALGORITHM_LABELS, build_recommenders

RANKING_METRICS = ("ndcg", "recall", "hit")
DEFAULT_K = 10
DEFAULT_FOLDS = 5


def _test_with_users(ratings: pd.DataFrame, test_indices: pd.Index) -> pd.DataFrame:
    """Restore user IDs dropped by LensKit's crossfold test frames."""
    return ratings.loc[test_indices, ["user", "item", "rating"]].reset_index(drop=True)


def run_ranking_evaluation(
    ratings: pd.DataFrame,
    *,
    k: int = DEFAULT_K,
    folds: int = DEFAULT_FOLDS,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate algorithms with user-based hold-out and standard ranking metrics.

    Uses 5-fold user cross-validation: for each fold, train on 80% of each user's
    ratings, recommend top-K items, and score against the held-out 20%.

    Returns per-user/per-fold results and a mean summary per algorithm.
    """
    data = ratings[["user", "item", "rating"]]
    recommenders = build_recommenders()
    all_recs: list[pd.DataFrame] = []
    truth_parts: list[pd.DataFrame] = []

    for fold, pair in enumerate(
        xf.partition_users(
            data,
            partitions=folds,
            method=xf.SampleFrac(test_frac, rng_spec=seed),
        )
    ):
        train = pair.train
        test = _test_with_users(data, pair.test.index)
        truth_parts.append(test)
        users = test["user"].unique()

        for name, algo in recommenders.items():
            model = Recommender.adapt(util.clone(algo))
            model.fit(train)
            recs = batch.recommend(model, users, k)
            recs["algorithm"] = name
            recs["fold"] = fold
            all_recs.append(recs)

    recommendations = pd.concat(all_recs, ignore_index=True)
    truth = pd.concat(truth_parts, ignore_index=True)

    analysis = topn.RecListAnalysis()
    analysis.add_metric(topn.ndcg)
    analysis.add_metric(topn.recall)
    analysis.add_metric(topn.hit)
    per_list = analysis.compute(recommendations, truth)

    summary_rows = []
    for metric in RANKING_METRICS:
        grouped = per_list.groupby("algorithm")[metric]
        for name in recommenders:
            if name not in grouped.groups:
                continue
            values = grouped.get_group(name)
            summary_rows.append(
                {
                    "metric": metric,
                    "algorithm": name,
                    "label": ALGORITHM_LABELS[name],
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    return per_list, summary


def best_ranking_algorithm(summary: pd.DataFrame, metric: str) -> str:
    """Return algorithm key with highest mean for a ranking metric."""
    subset = summary[summary["metric"] == metric]
    row = subset.loc[subset["mean"].idxmax()]
    return str(row["algorithm"])
