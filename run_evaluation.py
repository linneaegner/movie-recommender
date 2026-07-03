#!/usr/bin/env python3
"""Run offline evaluation and print algorithm comparison."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data import dataset_ready, load_movielens
from src.evaluate import METRICS, best_algorithm, run_evaluation
from src.models import ALGORITHM_LABELS, build_recommenders

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    if not dataset_ready():
        raise SystemExit(
            "MovieLens data not found. Run: bash scripts/download_data.sh"
        )

    ratings, movies = load_movielens()
    metrics_df, summary = run_evaluation(ratings, movies)

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_by_iteration.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    print("\nMean metrics over all iterations:\n")
    for metric in METRICS:
        print(metric.replace("_", " ").title())
        subset = summary[summary["metric"] == metric].sort_values("mean", ascending=False)
        for _, row in subset.iterrows():
            print(f"  {row['label']:28} {row['mean']:.3f} (±{row['std']:.3f})")
        print(f"  → Best: {ALGORITHM_LABELS[best_algorithm(summary, metric)]}\n")

    for metric in METRICS:
        plt.figure(figsize=(8, 4))
        plt.title(f"{metric.replace('_', ' ').title()} over iterations")
        for name in build_recommenders():
            column = f"{metric}_{name}"
            plt.plot(
                metrics_df.index,
                metrics_df[column],
                marker="o",
                label=ALGORITHM_LABELS[name],
            )
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{metric}.png", dpi=120)
        plt.close()

    print(f"Saved charts and CSVs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
