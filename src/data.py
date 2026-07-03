from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "ml-latest-small"


def dataset_ready() -> bool:
    return (DATA_DIR / "ratings.csv").exists() and (DATA_DIR / "movies.csv").exists()


def load_movielens(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens small dataset in LensKit column format."""
    root = data_dir or DATA_DIR
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")

    ratings = ratings.rename(
        columns={
            "userId": "user",
            "movieId": "item",
            "rating": "rating",
            "timestamp": "timestamp",
        }
    )
    movies = movies.rename(columns={"movieId": "item"})
    return ratings, movies


def movie_titles(movies: pd.DataFrame) -> dict[int, str]:
    return movies.set_index("item")["title"].to_dict()
