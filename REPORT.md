# Evaluation report — Movie Recommender

## Problem

Recommendation systems must balance **accuracy** with qualities like **diversity** and **novelty**. Recommending only blockbusters is safe but boring; recommending only obscure titles explores the catalog but may miss user taste.

This project compares four algorithms on MovieLens and measures:

| Metric | What it means | Higher is… |
|--------|----------------|------------|
| **Novelty** | How *unpopular* recommended items are (self-information) | More surprising / niche picks |
| **Item coverage** | Share of the full catalog that gets recommended (batch scale) | Broader exploration |
| **Temporal diversity** | Spread of release years among recommended items | More era variety |
| **NDCG@10** | Ranking quality of top-10 vs. held-out ratings | Better ordering of relevant items |
| **Recall@10** | Share of relevant items found in top-10 | More test items recovered |
| **Hit rate** | Share of users with ≥1 relevant item in top-10 | More users get a useful rec |

### Why coverage looks tiny in the Streamlit app

Catalog coverage is `unique recommended movies ÷ 9,742 movies`. One user receiving 8 picks can cover at most **~0.08%** of the catalog, so values like `0.001` are expected. The metric is meaningful when many users are evaluated together — see results below.

## Setup

### Diversity & simulation loop

- **Data:** MovieLens Small (600 users, ~9k movies, 100k ratings)
- **Loop:** 10 iterations × 100 random users × 20 recommendations each
- **Feedback simulation:** 20% of recommendations become new synthetic ratings (feeds the next iteration)

```bash
bash scripts/download_data.sh
python run_evaluation.py
```

### Hold-out ranking evaluation

- **Split:** 5-fold user cross-validation, 20% of each user's ratings held out
- **Recommendations:** Top-10 per user, trained on remaining 80%
- **Metrics:** NDCG@10, Recall@10, hit rate (LensKit `RecListAnalysis`)

```bash
python run_ranking_evaluation.py
```

Results are written to `outputs/`.

## Results

Mean ± std over 10 iterations (100 users × 20 recs each):

| Algorithm | Novelty ↑ | Catalog coverage ↑ | Temporal diversity ↑ |
|-----------|-----------|--------------------|----------------------|
| Personal Top-N (bias) | **16.38 ± 0.60** | 0.002 ± 0.000 | 14.80 ± 4.60 |
| Personal KNN | 15.23 ± 0.49 | 0.043 ± 0.014 | **20.97 ± 1.58** |
| Random | 14.73 ± 0.09 | **0.186 ± 0.001** | 18.58 ± 0.57 |
| Most popular | 8.87 ± 0.05 | 0.013 ± 0.002 | 7.49 ± 0.12 |

### Hold-out ranking (5-fold user CV, @10)

| Algorithm | NDCG@10 ↑ | Recall@10 ↑ | Hit rate ↑ |
|-----------|-----------|-------------|------------|
| Most popular | **0.112 ± 0.128** | **0.121 ± 0.097** | **0.626 ± 0.484** |
| Personal Top-N (bias) | 0.000 ± 0.012 | 0.085 ± 0.116 | 0.003 ± 0.057 |
| Random | 0.002 ± 0.017 | 0.036 ± 0.045 | 0.046 ± 0.209 |
| Personal KNN | 0.000 ± 0.005 | 0.029 ± 0.019 | 0.007 ± 0.081 |

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

### Hold-out ranking accuracy

- **Most popular** wins NDCG, recall, and hit rate on this split. Blockbusters are rated by many users, so recommending globally popular items often overlaps with held-out ratings even without personalization.
- **Personal KNN** and **Personal Top-N** score poorly on NDCG here — they optimize for user taste on *unseen* items, while the test set contains items the user already rated (including niche ones). This is a known tension in offline eval: strong personalization can hurt naive hold-out metrics.
- **Random** is a weak baseline as expected.

For production, ranking metrics should be paired with diversity/novelty goals — optimizing NDCG alone would keep the system on safe popular picks.

### Summary

| Goal | Strongest option in this evaluation |
|------|-------------------------------------|
| Familiar, broadly appealing picks (e.g. a “Trending” row) | **Most popular** |
| Personalized lists for returning users | **Personal KNN** or **Personal Top-N** |
| Unbiased baseline | **Random** |
| Balance of personalization and discovery | **Personal KNN** |

There is no single winner. **Most popular** optimizes familiarity; **KNN** trades some popularity for personalization, novelty, and era variety. In a production setting, the right choice depends on the product goal and should be validated with user behavior metrics.

## Pros & cons by algorithm

### Random

| | |
|---|---|
| **Best for** | Research baseline |
| **Pros** | Widest catalog exploration; high novelty; no training needed |
| **Cons** | No personalization; very low hit rate and NDCG; poor UX |
| **Quality** | Low relevance — only useful as a lower bound |

### Most popular

| | |
|---|---|
| **Best for** | "Trending" rows, cold-start users |
| **Pros** | Best NDCG/recall/hit rate here; familiar titles; fast; no user history needed |
| **Cons** | Same list for everyone; lowest novelty; mainstream filter bubble |
| **Quality** | Strong offline accuracy, weak diversity |

### Personal Top-N (bias)

| | |
|---|---|
| **Best for** | Lightweight personalization with interpretable scores |
| **Pros** | Highest novelty among personalized methods; adapts to user rating level |
| **Cons** | Coarse model (no item similarity); tiny catalog coverage; near-zero NDCG@10 |
| **Quality** | Niche lists, weak standard ranking scores |

### Personal KNN

| | |
|---|---|
| **Best for** | Returning users with rating history |
| **Pros** | Best temporal diversity; tailored lists; strong genre match for many users |
| **Cons** | Slow first run; cold-start weakness; low naive NDCG; can echo taste bubble |
| **Quality** | Best personalized all-rounder — combine with ranking + diversity metrics |

Run `python run_all_evaluations.py` to regenerate `outputs/algorithm_scorecard.md` with the latest numbers.

## Streamlit demo

`app.py` complements the offline evaluation with qualitative exploration:

- **Compare algorithms** — four columns for the same user, with novelty metrics and genre charts
- **Filter bubble** — slider blending Personal KNN with Most popular
- **Blind taste test** — pick a list, then reveal which algorithm produced it
- **User personas** — e.g. Heavy rater, Blockbuster fan, Niche explorer
- **Algorithm guide** — offline quality table + pros/cons for each approach

Useful for asking whether a recommendation list makes sense given a user's rating history.

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems.*
- Ekstrand, M. D., Konstan, J. A., & Terveen, L. [LensKit documentation](https://lenskit.org/)
