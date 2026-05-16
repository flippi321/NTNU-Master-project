# Analyse GradientSHAP results produced by shap_compute.py.
#
# Loads raw per-column SHAP values from ex2/results/<model_id>/, aggregates
# one-hot-encoded categorical dummies back into single features, ranks features
# by mean |SHAP|, and produces visualisations.  No GPU required.
#
# Examples (run from project root):
#   # One model:
#   python ex2/shap_analyse.py --model_id film_simple_res_trial_012
#
#   # All models with SHAP data:
#   python ex2/shap_analyse.py --all
#
#   # Custom output directory:
#   python ex2/shap_analyse.py --model_id meta_trial_018 \
#       --out_dir ex2/results/meta_trial_018/analysis_v2

import argparse
import json
import os
import re
import sys

# Make project root importable when this script is invoked as
#   python ex2/shap_analyse.py …
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

RESULTS_ROOT = os.path.join("ex2", "results")

# Maps each raw CSV-column prefix (the categorical feature name in
# health_data_utils.CATEGORICAL_COLS) → human-readable display label.
ONE_HOT_GROUPS = {
    "Healt@NT3BLQ1":   "Health",
    "SmoStat@NT3BLQ1": "Smoking Status",
    "Educ@NT2BLQ1":    "Education",
}


# ─── Data loading + aggregation ───────────────────────────────────────────────

def discover_models(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isdir(d)
                and os.path.isfile(os.path.join(d, "shap_values.csv"))
                and os.path.isfile(os.path.join(d, "meta.json"))):
            out.append(name)
    return out


def load_results(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = os.path.join(RESULTS_ROOT, model_id)
    shap_df = pd.read_csv(os.path.join(base, "shap_values.csv"))
    feat_df = pd.read_csv(os.path.join(base, "feature_values.csv"))
    with open(os.path.join(base, "meta.json")) as f:
        meta = json.load(f)
    return shap_df, feat_df, meta


def _friendly_name(raw: str) -> str:
    """Strip cohort suffix and underscores for plot readability."""
    # Drop @NT3BLQ1, @NT3BLM, @NT3BLI, @NT2BLQ1, @NT3BLQ2, etc.
    name = re.sub(r"@NT\d+BL[A-Z]\d*", "", raw)
    return name.replace("_", " ").strip()


def aggregate_one_hot(values_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum signed values across one-hot dummies for each categorical group.
    Input columns are raw feature names (incl. `patient_id`).
    Output columns: `patient_id` + one column per logical feature.
    """
    out = pd.DataFrame({"patient_id": values_df["patient_id"]})
    consumed = {"patient_id"}

    # Aggregate each one-hot group
    for prefix, label in ONE_HOT_GROUPS.items():
        dummies = [c for c in values_df.columns if c.startswith(prefix + "_")]
        if not dummies:
            continue
        out[label] = values_df[dummies].sum(axis=1)
        consumed.update(dummies)

    # Pass-through the remaining numerical/binary features with cleaner names
    for col in values_df.columns:
        if col in consumed:
            continue
        out[_friendly_name(col)] = values_df[col]

    return out


def build_correlation(shap_agg: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of per-subject SHAP values across features.
    Square DataFrame indexed/columned by aggregated feature name, ordered by
    mean |SHAP| so it visually lines up with feature_importance.csv. Features
    with zero SHAP variance produce NaN rows/cols — left as NaN so the heatmap
    leaves them blank instead of showing misleading zeros.
    """
    features = imp["feature"].tolist()
    return shap_agg[features].corr(method="pearson")


def build_importance(shap_agg: pd.DataFrame) -> pd.DataFrame:
    """One row per aggregated feature with importance stats."""
    cols = [c for c in shap_agg.columns if c != "patient_id"]
    rows = []
    for c in cols:
        v = shap_agg[c].values
        rows.append({
            "feature":         c,
            "mean_abs_shap":   float(np.mean(np.abs(v))),
            "mean_signed_shap":float(np.mean(v)),
            "std_signed_shap": float(np.std(v)),
            "max_abs_shap":    float(np.max(np.abs(v))),
        })
    return (pd.DataFrame(rows)
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_feature_importance(imp: pd.DataFrame, model_id: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.32)))
    bars = ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1],
                   color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|  (impact on prediction loss)")
    ax.set_title(f"Feature importance — {model_id}")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, imp["mean_abs_shap"][::-1]):
        ax.text(v, bar.get_y() + bar.get_height() / 2,
                f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_shap_summary(shap_agg: pd.DataFrame,
                      feat_agg: pd.DataFrame,
                      imp: pd.DataFrame,
                      model_id: str,
                      out_path: str) -> None:
    """SHAP "beeswarm"-style: y=feature (sorted by importance, top = strongest),
    x=signed SHAP value, point colour = rank-normalised feature value."""
    features = imp["feature"].tolist()
    n_feat   = len(features)

    fig, ax = plt.subplots(figsize=(9, max(4, n_feat * 0.32)))
    cmap = plt.get_cmap("coolwarm")

    for i, feat in enumerate(features):
        y_pos = n_feat - 1 - i  # top of axis = most important
        shap_vals = shap_agg[feat].values
        feat_vals = feat_agg[feat].values

        # rank-normalise feature values into [0,1] for colour
        if np.ptp(feat_vals) > 0:
            ranks = pd.Series(feat_vals).rank(pct=True).values
        else:
            ranks = np.full_like(feat_vals, 0.5, dtype=float)
        colors = cmap(ranks)

        jitter = np.random.uniform(-0.18, 0.18, size=len(shap_vals))
        ax.scatter(shap_vals, np.full_like(shap_vals, y_pos) + jitter,
                   c=colors, s=18, alpha=0.7, edgecolors="none")

    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(list(reversed(features)), fontsize=9)
    ax.set_xlabel("SHAP value  (impact on prediction loss; lower = better)")
    ax.set_title(f"SHAP summary — {model_id}")
    ax.grid(axis="x", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
    cb.set_label("Feature value  (low → high)", fontsize=9)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["low", "mid", "high"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_correlation(corr: pd.DataFrame, model_id: str, out_path: str) -> None:
    """Heatmap of pairwise Pearson correlation between feature SHAP values."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))
    sns.heatmap(corr, ax=ax,
                cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.3,
                annot=(n <= 30), fmt=".2f", annot_kws={"size": 7},
                cbar=False)
    ax.set_title(f"SHAP value correlation — {model_id}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_shap_distribution(shap_agg: pd.DataFrame, imp: pd.DataFrame,
                           model_id: str, out_path: str) -> None:
    features = imp["feature"].tolist()
    data = [shap_agg[f].values for f in features]

    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.32)))
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#55A868")
        patch.set_alpha(0.7)
    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(range(1, len(features) + 1))
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.set_title(f"SHAP value distribution — {model_id}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_one(model_id: str, out_dir: str | None) -> None:
    print(f"\n=== {model_id} ===")
    shap_df, feat_df, meta = load_results(model_id)
    print(f"  Loaded {len(shap_df)} subjects × {len(shap_df.columns) - 1} raw features"
          f"   (family={meta.get('model_family')}, "
          f"n_background={meta.get('n_background')})")

    shap_agg = aggregate_one_hot(shap_df)
    feat_agg = aggregate_one_hot(feat_df)
    imp      = build_importance(shap_agg)
    print(f"  Aggregated → {len(imp)} features")
    print(f"  Top 5: {imp['feature'].head(5).tolist()}")

    if out_dir is None:
        out_dir = os.path.join(RESULTS_ROOT, model_id, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    imp_csv = os.path.join(out_dir, "feature_importance.csv")
    imp.to_csv(imp_csv, index=False)
    print(f"  Saved: {imp_csv}")

    plot_feature_importance(imp, model_id,
                            os.path.join(out_dir, "feature_importance.png"))
    plot_shap_summary(shap_agg, feat_agg, imp, model_id,
                      os.path.join(out_dir, "shap_summary.png"))
    plot_shap_distribution(shap_agg, imp, model_id,
                           os.path.join(out_dir, "shap_distribution.png"))

    corr = build_correlation(shap_agg, imp)
    corr_csv = os.path.join(out_dir, "shap_correlation.csv")
    corr.to_csv(corr_csv)
    print(f"  Saved: {corr_csv}")
    plot_correlation(corr, model_id,
                     os.path.join(out_dir, "shap_correlation.png"))


def main():
    parser = argparse.ArgumentParser(description="Analyse stored SHAP results")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_id", help="Model ID (directory name in ex2/results/)")
    group.add_argument("--all",      action="store_true",
                       help="Analyse every model in ex2/results/ with shap_values.csv")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: ex2/results/<model_id>/analysis/)")
    args = parser.parse_args()

    if args.all:
        ids = discover_models(RESULTS_ROOT)
        if not ids:
            sys.exit("No SHAP results found in ex2/results/.")
        print(f"Found {len(ids)} model(s): {ids}")
        for mid in ids:
            run_one(mid, args.out_dir)  # if --out_dir is set with --all, all overwrite each other
    else:
        run_one(args.model_id, args.out_dir)


if __name__ == "__main__":
    main()
