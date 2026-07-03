#!/usr/bin/env python3
"""Run all offline evaluations and write a combined scorecard."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from src.data import dataset_ready, load_movielens
from src.evaluate import run_evaluation
from src.models import ALGORITHM_LABELS
from src.ranking_eval import run_ranking_evaluation

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _quiet_lenskit_noise() -> None:
    """Hide harmless Numba/LensKit startup noise during long batch runs."""
    logging.getLogger("numba").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"lenskit\.metrics\.topn")


def _progress(current: int, total: int) -> None:
    print(f"  iteration {current}/{total}", flush=True)


def _metric_row(summary, metric: str, algorithm: str) -> str:
    row = summary[
        (summary["metric"] == metric) & (summary["algorithm"] == algorithm)
    ]
    if row.empty:
        return "—"
    value = row.iloc[0]
    return f"{value['mean']:.3f} (±{value['std']:.3f})"


def main() -> None:
    if not dataset_ready():
        raise SystemExit(
            "MovieLens data not found. Run: bash scripts/download_data.sh"
        )

    _quiet_lenskit_noise()
    print(
        "Full evaluation takes about 5–8 minutes. "
        "Numba threading messages on first run are normal.\n",
        flush=True,
    )

    ratings, movies = load_movielens()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Running diversity / simulation evaluation…", flush=True)
    _diversity_df, diversity_summary = run_evaluation(
        ratings, movies, on_progress=_progress
    )
    diversity_summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print("  done.", flush=True)

    print("Running hold-out ranking evaluation…", flush=True)
    _ranking_df, ranking_summary = run_ranking_evaluation(ratings)
    ranking_summary.to_csv(OUTPUT_DIR / "ranking_summary.csv", index=False)
    print("  done.", flush=True)

    lines = [
        "# Algorithm scorecard",
        "",
        "Combined offline results from `run_all_evaluations.py`.",
        "",
        "## Recommendation quality (hold-out test)",
        "",
        "Do the top-10 picks match movies the user actually rated highly?",
        "",
        "| Algorithm | NDCG@10 | Recall@10 | Hit rate |",
        "|-----------|---------|-----------|----------|",
    ]

    for name in ALGORITHM_LABELS:
        lines.append(
            f"| {ALGORITHM_LABELS[name]} "
            f"| {_metric_row(ranking_summary, 'ndcg', name)} "
            f"| {_metric_row(ranking_summary, 'recall', name)} "
            f"| {_metric_row(ranking_summary, 'hit', name)} |"
        )

    lines.extend(
        [
            "",
            "## Diversity & discovery (simulation loop)",
            "",
            "| Algorithm | Novelty | Catalog coverage | Temporal diversity |",
            "|-----------|---------|------------------|--------------------|",
        ]
    )

    for name in ALGORITHM_LABELS:
        lines.append(
            f"| {ALGORITHM_LABELS[name]} "
            f"| {_metric_row(diversity_summary, 'novelty', name)} "
            f"| {_metric_row(diversity_summary, 'item_coverage', name)} "
            f"| {_metric_row(diversity_summary, 'temporal_diversity', name)} |"
        )

    lines.extend(
        [
            "",
            "## Pros & cons at a glance",
            "",
        ]
    )

    from src.algorithm_guide import ALGORITHM_PROFILES

    for profile in ALGORITHM_PROFILES.values():
        lines.append(f"### {profile.label}")
        lines.append(f"**Best for:** {profile.best_for}")
        lines.append(f"**Offline quality:** {profile.offline_quality}")
        lines.append("")
        lines.append("**Pros**")
        for pro in profile.pros:
            lines.append(f"- {pro}")
        lines.append("")
        lines.append("**Cons**")
        for con in profile.cons:
            lines.append(f"- {con}")
        lines.append("")

    path = OUTPUT_DIR / "algorithm_scorecard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}", flush=True)


if __name__ == "__main__":
    main()
