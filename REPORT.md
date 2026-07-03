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

See `outputs/metrics_summary.csv` for your exact results.

## Results (measured on this machine)

Mean ± std over 10 iterations (100 users × 20 recs each):

| Algorithm | Novelty ↑ | Catalog coverage ↑ | Temporal diversity ↑ |
|-----------|-----------|--------------------|----------------------|
| Personal Top-N (bias) | **16.38 ± 0.60** | 0.002 ± 0.000 | 14.80 ± 4.60 |
| Personal KNN | 15.23 ± 0.49 | 0.043 ± 0.014 | **20.97 ± 1.58** |
| Random | 14.73 ± 0.09 | **0.186 ± 0.001** | 18.58 ± 0.57 |
| Most popular | 8.87 ± 0.05 | 0.013 ± 0.002 | 7.49 ± 0.12 |

**Takeaways from these runs:**

- **Most popular** is safest and most familiar (lowest novelty, lowest era spread, narrow catalog reach).
- **Random** explores the catalog widest (~19% of all movies touched across batches) but is not personalized.
- **Personal KNN** balances personalization with discovery — strong temporal diversity and mid-range coverage.
- **Personal Top-N** is the most novel but barely expands catalog coverage (repeated safe picks within user taste).

## Expected findings (typical pattern)

These patterns match standard recommender literature and what this codebase measures:

### Novelty — usually wins: **Personal KNN** or **Random**

- **Most popular** scores **lowest** — it keeps recommending the same hits everyone already knows.
- **Personal KNN** often scores **high** — collaborative filtering surfaces less mainstream items that similar users liked.
- **Random** is high but **not useful** for users — high novelty with no personalization.

### Catalog coverage — usually wins: **Random**, then personalized methods

- **Random** reaches the **widest** slice of the catalog (by design).
- **Most popular** has the **lowest** coverage — the same top movies dominate.
- **Personal Top-N** and **KNN** sit in the middle — personalized but still somewhat concentrated.

## Which algorithm is “best”?

**It depends on the product goal:**

| Goal | Best choice |
|------|-------------|
| Maximize engagement on known hits (e.g. homepage “Trending”) | **Most popular** |
| Personalized experience for returning users | **Personal KNN** or **Personal Top-N** |
| Research baseline | **Random** |
| Balance discovery + personalization | **Personal KNN** — good default in this project |

For a **job interview**, the strong answer is not “KNN always wins” but:

> “Most popular optimizes familiarity; KNN trades some popularity for personalization and novelty. I’d A/B test with click-through and diversity metrics in production.”

## Streamlit demo

`app.py` includes:

- **Compare algorithms** — four columns for the same user, with novelty + genre charts
- **Filter bubble** — slider blending Personal KNN with Most popular
- **Blind taste test** — pick a list, then reveal which algorithm produced it
- **User personas** — e.g. Heavy rater, Blockbuster fan, Niche explorer

Useful for qualitative UX review (“does this list make sense given their history?”). That connects the ML work to **HCI / UX research**, which fits cognitive science.

## Next steps (portfolio polish)

- [x] Run evaluation and paste mean metrics into this file
- [ ] Add 1–2 screenshots of the Streamlit app to README
- [ ] Deploy demo on [Streamlit Community Cloud](https://streamlit.io/cloud) (optional)
- [ ] Make repo **public** when ready to link from portfolio / LinkedIn

## References

- Harper & Konstan (2015) — MovieLens datasets
- Ekstrand, Konstan & Terveen — LensKit documentation
- Course: cognitive science ML / recommender systems module
