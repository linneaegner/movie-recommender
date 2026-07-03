"""Pros, cons, and use-case guidance for each recommender."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmProfile:
    key: str
    label: str
    idea: str
    best_for: str
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    offline_quality: str


ALGORITHM_PROFILES: dict[str, AlgorithmProfile] = {
    "random": AlgorithmProfile(
        key="random",
        label="Random",
        idea="Picks unseen movies at random.",
        best_for="Research baseline and sanity-checking other algorithms.",
        pros=(
            "Explores the widest catalog by design.",
            "High novelty — surfaces obscure titles.",
            "Simple, fast, no training required.",
        ),
        cons=(
            "No personalization — ignores user taste entirely.",
            "Very low hit rate and NDCG in offline tests.",
            "Poor user experience for real recommendations.",
        ),
        offline_quality="Low relevance. Only useful as a lower bound.",
    ),
    "most_popular": AlgorithmProfile(
        key="most_popular",
        label="Most popular",
        idea="Recommends globally most-rated movies.",
        best_for='Homepage rows like "Trending now" or cold-start users.',
        pros=(
            "Best offline NDCG, recall, and hit rate in this project.",
            "Safe, familiar picks users often recognize.",
            "Works with zero user history (cold start).",
            "Fast and easy to explain to stakeholders.",
        ),
        cons=(
            "Same blockbusters for everyone — low personalization.",
            "Lowest novelty and temporal diversity.",
            "Narrows catalog exposure; filter-bubble of mainstream hits.",
        ),
        offline_quality="Strong hold-out accuracy, weak on diversity.",
    ),
    "personal_topn": AlgorithmProfile(
        key="personal_topn",
        label="Personal Top-N (bias)",
        idea="Predicts ratings from user + item bias, ranks unseen items.",
        best_for="Lightweight personalization when you want interpretable scores.",
        pros=(
            "Highest novelty among personalized methods here.",
            "Adapts to each user's average rating level.",
            "Computationally cheaper than KNN.",
        ),
        cons=(
            "Ignores item similarity and neighbor taste — coarse personalization.",
            "Very low catalog coverage in batch evaluation.",
            "Near-zero NDCG@10 on hold-out (recommends unseen, not held-out items).",
        ),
        offline_quality="Niche and novel lists, but weak standard ranking scores.",
    ),
    "personal_knn": AlgorithmProfile(
        key="personal_knn",
        label="Personal KNN",
        idea='Collaborative filtering: "users like you liked…"',
        best_for="Returning users with enough rating history.",
        pros=(
            "Best temporal diversity — spans many release eras.",
            "Lists feel tailored; strong genre match for many users.",
            "Balances personalization with discovery better than popularity.",
            "Classic, interview-friendly CF approach.",
        ),
        cons=(
            "Slow to train on first run (Numba compilation).",
            "Struggles with cold-start and sparse users.",
            "Low naive offline NDCG — optimizes unseen picks, not held-out reranks.",
            "Can reinforce taste bubble without a popularity blend.",
        ),
        offline_quality="Best all-round personalized option; pair with ranking metrics.",
    ),
}


def get_profile(algorithm: str) -> AlgorithmProfile:
    return ALGORITHM_PROFILES[algorithm]
