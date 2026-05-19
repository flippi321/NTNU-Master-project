"""Feature-selection helpers shared by ex3/train_reduced.py and the EX3
Bayesian feature search.

Both callers need the same SHAP-driven selection logic: rank features by
mean_abs_shap, optionally cap at top_k, then greedily drop features whose
absolute SHAP-correlation with an already-kept feature meets the threshold.
Keeping the helpers here (rather than in train_reduced.py) avoids dragging
torch/argparse/DataLoader into the search module's import graph.
"""

import re

import pandas as pd


# Same mapping ex2/shap_analyse.py uses to collapse one-hot dummies to a
# single "logical" feature name.
ONE_HOT_GROUPS = {
    "Healt@NT3BLQ1":   "Health",
    "SmoStat@NT3BLQ1": "Smoking Status",
    "Educ@NT2BLQ1":    "Education",
}


def _friendly_name(raw: str) -> str:
    """Strip @NT3BL* cohort suffix and replace underscores with spaces."""
    name = re.sub(r"@NT\d+BL[A-Z]\d*", "", raw)
    return name.replace("_", " ").strip()


def select_features(
    imp_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    imp_thr: float,
    corr_thr: float,
    top_k: int | None,
) -> tuple[list[str], dict]:
    """Greedy redundancy elimination.

    Returns (kept_features_in_importance_order, dropped_for_correlation_map).
    """
    ranked = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    if top_k:
        ranked = ranked.head(top_k)
    else:
        ranked = ranked[ranked["mean_abs_shap"] >= imp_thr]

    candidates = ranked["feature"].tolist()
    kept: list[str] = []
    dropped_for_corr: dict[str, list[str]] = {}

    for f in candidates:
        too_correlated_with = None
        for g in kept:
            try:
                r = corr_df.at[f, g]
            except KeyError:
                continue
            if pd.notna(r) and abs(r) >= corr_thr:
                too_correlated_with = g
                break
        if too_correlated_with is None:
            kept.append(f)
        else:
            dropped_for_corr.setdefault(too_correlated_with, []).append(f)

    return kept, dropped_for_corr


def map_aggregated_to_raw_indices(
    kept_aggregated: list[str],
    raw_feature_names: list[str],
) -> tuple[list[int], list[str]]:
    """Map each kept aggregated feature back to its raw column index/indices.

    - For one-hot groups, expand to ALL dummy columns sharing the prefix.
    - Otherwise, find the single raw column whose _friendly_name matches.
    Indices are returned sorted (preserves raw-column order, so the
    conditioning vector layout is deterministic and human-readable).
    """
    one_hot_label_to_prefix = {label: pref for pref, label in ONE_HOT_GROUPS.items()}
    name_to_raw_idx: dict[str, int] = {n: i for i, n in enumerate(raw_feature_names)}

    selected: set[int] = set()
    for agg in kept_aggregated:
        if agg in one_hot_label_to_prefix:
            prefix = one_hot_label_to_prefix[agg] + "_"
            for raw_name, idx in name_to_raw_idx.items():
                if raw_name.startswith(prefix):
                    selected.add(idx)
        else:
            matches = [idx for raw, idx in name_to_raw_idx.items()
                       if _friendly_name(raw) == agg]
            if not matches:
                raise ValueError(
                    f"Could not map aggregated feature {agg!r} to any raw column. "
                    f"Available friendly names: "
                    f"{sorted({_friendly_name(n) for n in raw_feature_names})}"
                )
            selected.update(matches)

    indices = sorted(selected)
    chosen_names = [raw_feature_names[i] for i in indices]
    return indices, chosen_names
