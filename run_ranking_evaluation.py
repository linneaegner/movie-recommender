#!/usr/bin/env python3
"""Run hold-out ranking evaluation (NDCG, recall, hit rate)."""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import dataset_ready, load_movielens
from src.models import ALGORITHM_LABELS
from src.ranking_eval import (
    DEFAULT_K,
    RANKING_METRICS,
    best_ranking_algorithm,
    run_ranking_evaluation,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    if not dataset_ready():
        raise SystemExit(
            "MovieLens data not found. Run: bash scripts/download_data.sh"
        )

    logging.getLogger("numba").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"lenskit\.metrics\.topn")

    print("Hold-out ranking evaluation (~3 minutes)…", flush=True)

    ratings, _movies = load_movielens()
    per_list, summary = run_ranking_evaluation(ratings)

    OUTPUT_DIR.mkdir(exist_ok=True)
    per_list.to_csv(OUTPUT_DIR / "ranking_by_user.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "ranking_summary.csv", index=False)

    print(f"\nHold-out ranking metrics (@{DEFAULT_K}, 5-fold user CV):\n")
    for metric in RANKING_METRICS:
        title = {
            "ndcg": "NDCG@K",
            "recall": "Recall@K",
            "hit": "Hit rate",
        }[metric]
        print(title)
        subset = summary.loc[summary["metric"] == metric].sort_values(
            by="mean", ascending=False
        )
        for _, row in subset.iterrows():
            print(f"  {row['label']:28} {row['mean']:.3f} (±{row['std']:.3f})")
        print(
            f"  → Best: {ALGORITHM_LABELS[best_ranking_algorithm(summary, metric)]}\n"
        )

    _save_bar_chart(summary)
    print(f"Saved CSVs and ranking_metrics.png to {OUTPUT_DIR}/")


def _save_bar_chart(summary: pd.DataFrame) -> None:
    algorithms = list(ALGORITHM_LABELS.keys())
    x = range(len(algorithms))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))

    for index, metric in enumerate(RANKING_METRICS):
        offset = (index - 1) * width
        means = [
            summary.loc[
                (summary["metric"] == metric) & (summary["algorithm"] == name),
                "mean",
            ].iloc[0]
            for name in algorithms
        ]
        ax.bar(
            [pos + offset for pos in x],
            means,
            width=width,
            label={"ndcg": "NDCG@10", "recall": "Recall@10", "hit": "Hit rate"}[metric],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([ALGORITHM_LABELS[name] for name in algorithms], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Hold-out ranking metrics (5-fold user CV)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ranking_metrics.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
