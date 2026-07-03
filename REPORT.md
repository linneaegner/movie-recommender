# Evaluation report — Movie Recommender

## Problem

Recommendation systems must balance **accuracy** with qualities like **diversity** and **novelty**. Recommending only blockbusters is safe but boring; recommending only obscure titles explores the catalog but may miss user taste.

This project compares four algorithms on MovieLens and measures:

| Metric | What it means | Higher is… |
|--------|----------------|------------|
| **Novelty** | How *unpopular* recommended items are (self-information) | More surprising / niche picks |
| **Item coverage** | Share of the full catalog that gets recommended | Broader exploration |

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

`app.py` lets you inspect **one user at a time** — useful for qualitative UX review (“does this list make sense given their history?”). That connects the ML work to **HCI / UX research**, which fits cognitive science.

## Next steps (portfolio polish)

- [ ] Run evaluation and paste your mean metrics into this file
- [ ] Add 1–2 screenshots of the Streamlit app to README
- [ ] Deploy demo on [Streamlit Community Cloud](https://streamlit.io/cloud) (optional)
- [ ] Make repo **public** when ready to link from portfolio / LinkedIn

## References

- Harper & Konstan (2015) — MovieLens datasets
- Ekstrand, Konstan & Terveen — LensKit documentation
- Course: cognitive science ML / recommender systems module
