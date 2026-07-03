import pandas as pd


def blend_recommendations(
    personal_recs: pd.DataFrame,
    popular_recs: pd.DataFrame,
    *,
    personal_weight: float,
    n: int,
) -> pd.DataFrame:
    """
    Mix personalized and popular lists according to a slider weight.

    personal_weight=1.0 → all personalized, 0.0 → all popular.
    """
    n_personal = round(personal_weight * n)
    picked: list[int] = []
    seen: set[int] = set()

    for item in personal_recs["item"]:
        if item in seen:
            continue
        if len(picked) >= n_personal:
            break
        picked.append(int(item))
        seen.add(int(item))

    for item in popular_recs["item"]:
        if item in seen:
            continue
        if len(picked) >= n:
            break
        picked.append(int(item))
        seen.add(int(item))

    for source in (personal_recs["item"], popular_recs["item"]):
        for item in source:
            if len(picked) >= n:
                break
            item = int(item)
            if item not in seen:
                picked.append(item)
                seen.add(item)

    user_id = int(personal_recs["user"].iloc[0])
    return pd.DataFrame(
        {
            "item": picked[:n],
            "user": user_id,
            "rank": range(1, len(picked[:n]) + 1),
        }
    )
