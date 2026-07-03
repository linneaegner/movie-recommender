# Evaluation report — Movie Recommender

## Problem

Recommendation systems must balance **accuracy** with qualities like **diversity** and **novelty**. Recommending only blockbusters is safe but boring; recommending only obscure titles explores the catalog but may miss user taste.

This project compares four algorithms on MovieLens and measures:

| Metric | What it means | Higher is… |
|--------|----------------|------------|
| **Novelty** | How *unpopular* recommended items are (self-information) | More surprising / niche picks |
| **Item coverage** | Share of the full catalog that gets recommended (batch scale) | Broader exploration |
| **Temporal diversity** | Spread of release years among recommended items | More era variety |

### Why coverage looks tiny in the Streamlit app

Catalog coverage is `unique recommended movies ÷ 9,742 movies`. One user receiving 8 picks can cover at most **~0.08%** of the catalog, so values like `0.001` are expected. The metric is meaningful when many users are evaluated together — see results below.

## Setup

- **Data:** MovieLens Small (600 users, ~9k movies, 100k ratings)
- **Loop:** 10 iterations × 100 random users × 20 recommendations each
- **Feedback simulation:** 20% of recommendations become new synthetic ratings (feeds the next iteration)

Run locally to reproduce numbers:

```bash
bash scripts/download_data.sh
python run_evaluation.py
```

Results are written to `outputs/metrics_summary.csv` and charted in `outputs/`.

## Results

Mean ± std over 10 iterations (100 users × 20 recs each):

| Algorithm | Novelty ↑ | Catalog coverage ↑ | Temporal diversity ↑ |
|-----------|-----------|--------------------|----------------------|
| Personal Top-N (bias) | **16.38 ± 0.60** | 0.002 ± 0.000 | 14.80 ± 4.60 |
| Personal KNN | 15.23 ± 0.49 | 0.043 ± 0.014 | **20.97 ± 1.58** |
| Random | 14.73 ± 0.09 | **0.186 ± 0.001** | 18.58 ± 0.57 |
| Most popular | 8.87 ± 0.05 | 0.013 ± 0.002 | 7.49 ± 0.12 |

## Interpretation

### Novelty

- **Most popular** scores lowest — it keeps recommending the same widely rated hits.
- **Personal Top-N** and **Personal KNN** score highest — personalization surfaces less mainstream items aligned with user taste.
- **Random** is high but not useful on its own — novelty without relevance.

### Catalog coverage

- **Random** reaches the widest slice of the catalog (~19% of all movies touched across batches).
- **Most popular** has the lowest coverage — a small set of blockbusters dominates.
- **Personal KNN** sits in the middle; **Personal Top-N** barely expands coverage despite high novelty.

### Temporal diversity

- **Personal KNN** recommends titles spread across the widest range of release years.
- **Most popular** concentrates on well-known films from a narrower era band.

### Summary

| Goal | Strongest option in this evaluation |
|------|-------------------------------------|
| Familiar, broadly appealing picks (e.g. a “Trending” row) | **Most popular** |
| Personalized lists for returning users | **Personal KNN** or **Personal Top-N** |
| Unbiased baseline | **Random** |
| Balance of personalization and discovery | **Personal KNN** |

There is no single winner. **Most popular** optimizes familiarity; **KNN** trades some popularity for personalization, novelty, and era variety. In a production setting, the right choice depends on the product goal and should be validated with user behavior metrics.

## Streamlit demo

`app.py` complements the offline evaluation with qualitative exploration:

- **Compare algorithms** — four columns for the same user, with novelty metrics and genre charts
- **Filter bubble** — slider blending Personal KNN with Most popular
- **Blind taste test** — pick a list, then reveal which algorithm produced it
- **User personas** — e.g. Heavy rater, Blockbuster fan, Niche explorer

Useful for asking whether a recommendation list makes sense given a user's rating history.

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems.*
- Ekstrand, M. D., Konstan, J. A., & Terveen, L. [LensKit documentation](https://lenskit.org/)
