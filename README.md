# Movie Recommender

Compare four classic recommendation algorithms on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset using [LensKit](https://lenskit.org/). The project evaluates **ranking accuracy** (NDCG, recall@K) on hold-out data plus **novelty**, **catalog coverage**, and **temporal diversity**, and includes an interactive [Streamlit](https://streamlit.io/) demo.

## Purpose

This started as ML coursework (University of Gothenburg): a provided Google Colab using LensKit and MovieLens. I extended it with more recommenders, compared how the lists differed on accuracy vs discovery, and rebuilt it as a public GitHub project with a Streamlit demo so others can try the algorithms side by side.

The goal is not one “best” list — it is to **see trade-offs**: familiarity vs novelty, hit rate vs catalog coverage, and how personalization changes what you get.

## Algorithms & how they differ

| Key | Method | Idea | Strong at | Weak at |
|-----|--------|------|-----------|---------|
| `random` | Random | Baseline — picks items at random | Catalog coverage / exploration | Relevance |
| `most_popular` | Popularity | Recommends what everyone watches | Hit rate / safe picks for many users | Filter bubble, low novelty |
| `personal_topn` | Biased Top-N | Personal ratings + item popularity | Novelty / niche-leaning picks | Narrow catalog coverage |
| `personal_knn` | User KNN | “Users like you liked…” (collaborative filtering) | Personalization + temporal diversity | Needs enough neighbor data; heavier |

Numbers and interpretation: **[REPORT.md](REPORT.md)**.

## Quick start

**Requirements:** Python **3.11** (recommended). Python 3.14 is not supported by LensKit’s dependencies yet. Dataset files are **not** in git — download them after cloning.

```bash
git clone https://github.com/linneaegner/movie-recommender.git
cd movie-recommender
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

**Windows without bash:** use [Git Bash](https://git-scm.com/downloads) or WSL for `scripts/download_data.sh`, or download [MovieLens Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) manually, unzip, and place the `ml-latest-small/` folder in the project root (it must contain `ratings.csv`).

### Interactive demo

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

The demo has four tabs:

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

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python3.11: command not found` | Install Python 3.11 (e.g. from [python.org](https://www.python.org/downloads/) or Homebrew `brew install python@3.11`). Do not use 3.14 for this project. |
| `pip install` fails on LensKit / binary wheels | Recreate the venv with **3.11** (`python3.11 -m venv .venv`) and retry. |
| Streamlit: “Dataset not found” | Run `bash scripts/download_data.sh` from the project root, then restart the app. |
| `bash: scripts/download_data.sh: No such file` | You are not in the repo root, or on Windows without bash — see Quick start (Windows). |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

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
├── REPORT.md                    # Evaluation findings
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

Code in this repository is available for learning and portfolio use unless otherwise noted. MovieLens dataset terms apply to the downloaded data.
