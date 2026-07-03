from lenskit.algorithms import Recommender, basic

ALGORITHM_LABELS = {
    "random": "Random",
    "most_popular": "Most popular",
    "personal_topn": "Personal Top-N (bias)",
    "personal_knn": "Personal KNN",
}


def build_recommenders() -> dict:
    """Create the four algorithms compared in the course project."""
    return {
        "random": basic.Random(),
        "most_popular": basic.MostPopular(),
        "personal_topn": Recommender.adapt(basic.TopN(basic.Bias())),
        "personal_knn": Recommender.adapt(basic.KNN(basic.Bias())),
    }
