from lenskit.algorithms import Recommender, basic
from lenskit.algorithms.user_knn import UserUser

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
        "most_popular": basic.Popular(),
        "personal_topn": Recommender.adapt(basic.Bias()),
        "personal_knn": Recommender.adapt(UserUser(20)),
    }
