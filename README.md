# Movie Recommender

A small **recommendation-system** project built for a cognitive science ML course. It compares four classic algorithms on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset using [LensKit](https://lenskit.org/), evaluates them with **novelty** and **catalog coverage**, and includes an interactive **Streamlit** demo.

Good portfolio piece for **data, ML, or UX research** roles when you can explain *why* different algorithms behave differently.

## Algorithms

| Key | Method | Idea |
|-----|--------|------|
| `random` | Random | Baseline — picks items at random |
| `most_popular` | Popularity | Recommends what everyone watches |
| `personal_topn` | Biased Top-N | Personal ratings + item popularity |
| `personal_knn` | User KNN | “Users like you liked…” (collaborative filtering) |

## Quick start

```bash
cd movie-recommender   # or your local path to this folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

### Interactive demo

```bash
streamlit run app.py
```

Pick a user and algorithm → see recommended movies and a sample of their rating history.

The demo has three tabs:

- **Compare algorithms** — side-by-side lists + genre chart
- **Filter bubble** — blend personalized KNN with popular hits
- **Blind taste test** — pick a list, reveal the algorithm

### Offline evaluation

```bash
python run_evaluation.py
```

Writes `outputs/metrics_summary.csv`, iteration logs, and charts. See **[REPORT.md](REPORT.md)** for how to interpret results.

## Project structure

```
├── app.py                 # Streamlit UI
├── run_evaluation.py      # Batch evaluation + charts
├── src/
│   ├── data.py            # Load MovieLens
│   ├── models.py          # Algorithm definitions
│   ├── metrics.py         # Novelty, coverage, simulation
│   └── evaluate.py        # Multi-iteration experiment loop
├── scripts/download_data.sh
├── REPORT.md              # Findings & interpretation
└── requirements.txt
```

## Dataset

MovieLens Small (~100k ratings). **Not committed to git** — run `scripts/download_data.sh` after clone.

## Author

Linnéa Egnér — cognitive science / ML coursework, extended for portfolio use.

## License

Course project — dataset © GroupLens Research (MovieLens license applies).
