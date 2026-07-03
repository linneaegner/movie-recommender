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


def split_genres(genre_string: str) -> list[str]:
    """Split MovieLens pipe-delimited genre string into individual genres."""
    if not genre_string or genre_string == "(no genres listed)":
        return []
    return genre_string.split("|")


def genre_profile(item_ids: pd.Series | list[int], movies: pd.DataFrame) -> pd.Series:
    """Return normalized genre proportions for a set of items."""
    subset = movies[movies["item"].isin(item_ids)]
    genres: list[str] = []
    for genre_string in subset["genres"]:
        genres.extend(split_genres(genre_string))

    if not genres:
        return pd.Series(dtype=float)

    counts = pd.Series(genres).value_counts()
    return (counts / counts.sum()).sort_values(ascending=False)
