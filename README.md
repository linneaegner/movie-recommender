# Movie Recommender

Compare four classic recommendation algorithms on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset using [LensKit](https://lenskit.org/). The project evaluates **ranking accuracy** (NDCG, recall@K) on hold-out data plus **novelty**, **catalog coverage**, and **temporal diversity**, and includes an interactive [Streamlit](https://streamlit.io/) demo.

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
- **Algorithm guide** — are recommendations good? pros & cons per approach

Users are labeled with viewing personas (e.g. *Heavy rater*, *Niche explorer*) derived from their rating history.

### Offline evaluation

**Ranking metrics** (standard hold-out evaluation):

```bash
python run_ranking_evaluation.py
```

5-fold user cross-validation, 20% held-out ratings per user, top-10 recommendations. Writes `outputs/ranking_summary.csv` and `ranking_metrics.png`.

**Diversity & simulation metrics**:

```bash
python run_evaluation.py
```

**All evaluations + scorecard** (ranking + diversity + pros/cons):

```bash
python run_all_evaluations.py
```

Writes CSVs, charts, and `outputs/algorithm_scorecard.md`. See **[REPORT.md](REPORT.md)** for measured results and interpretation.

## Project structure

```
├── app.py                       # Streamlit UI
├── run_evaluation.py            # Diversity / simulation evaluation
├── run_ranking_evaluation.py    # Hold-out NDCG / recall / hit rate
├── run_all_evaluations.py       # Full scorecard + algorithm_scorecard.md
├── src/
│   ├── data.py                  # Load MovieLens, genre helpers
│   ├── models.py                # Algorithm definitions
│   ├── metrics.py               # Novelty, coverage, genre match
│   ├── evaluate.py              # Multi-iteration experiment loop
│   ├── ranking_eval.py          # 5-fold ranking evaluation
│   ├── algorithm_guide.py       # Pros, cons, use-case profiles
│   ├── personas.py              # User viewing personas
│   └── blend.py                 # Popular / personalized list blending
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
