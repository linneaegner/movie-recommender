# Movie Recommender

Compare four classic recommendation algorithms on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset using [LensKit](https://lenskit.org/). The project evaluates **novelty**, **catalog coverage**, and **temporal diversity**, and includes an interactive [Streamlit](https://streamlit.io/) demo to explore how each strategy balances familiarity, relevance, and discovery.

## Algorithms

| Key | Method | Idea |
|-----|--------|------|
| `random` | Random | Baseline — picks items at random |
| `most_popular` | Popularity | Recommends what everyone watches |
| `personal_topn` | Biased Top-N | Personal ratings + item popularity |
| `personal_knn` | User KNN | “Users like you liked…” (collaborative filtering) |

## Quick start

**Requirements:** Python 3.11 (recommended). Python 3.14 is not yet supported by LensKit’s dependencies.

```bash
git clone https://github.com/linneaegner/movie-recommender.git
cd movie-recommender
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

### Interactive demo

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

The demo has three tabs:

- **Compare algorithms** — side-by-side lists, novelty metrics, and genre charts for the same user
- **Filter bubble** — blend personalized KNN with popular hits using a slider
- **Blind taste test** — pick a list, then reveal which algorithm produced it

Users are labeled with viewing personas (e.g. *Heavy rater*, *Niche explorer*) derived from their rating history.

### Offline evaluation

```bash
python run_evaluation.py
```

Writes `outputs/metrics_summary.csv`, per-iteration logs, and charts. See **[REPORT.md](REPORT.md)** for measured results and interpretation.

## Project structure

```
├── app.py                 # Streamlit UI
├── run_evaluation.py      # Batch evaluation + charts
├── src/
│   ├── data.py            # Load MovieLens, genre helpers
│   ├── models.py          # Algorithm definitions
│   ├── metrics.py         # Novelty, coverage, temporal diversity
│   ├── evaluate.py        # Multi-iteration experiment loop
│   ├── personas.py        # User viewing personas
│   └── blend.py           # Popular / personalized list blending
├── scripts/download_data.sh
├── REPORT.md              # Evaluation findings
└── requirements.txt
```

## Dataset

This repo uses **MovieLens Small** (~100k ratings, ~9k movies). The dataset is not included in git — download it after cloning:

```bash
bash scripts/download_data.sh
```

Data © [GroupLens Research](https://grouplens.org/). Use is subject to the [MovieLens license](https://files.grouplens.org/papers/ml-latest-small-README.html).

## Author

Linnéa Egnér

## License

See repository license. MovieLens dataset terms apply to the downloaded data.
