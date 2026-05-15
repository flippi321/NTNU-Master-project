# Analyse stored evaluation results from out/<model_id>/ directories.
# Loads dice_results.csv and meta.json for each model and produces
# comparison graphs and a summary table — no GPU required.
# Handles Dice + Volumetric Similarity (segmentation-level) AND SSIM + L1
# (image-level) metrics, gracefully skipping any that are missing for a model.
#
# Examples (run from project root):
#   # Specific models:
#   python model_analysis/analyse_results.py \
#       --models film_simple_res_trial_012 std_trial_036 meta_trial_018
#
#   # Auto-discover everything in out/ that has already been evaluated:
#   python model_analysis/analyse_results.py --all
#
#   # Custom output directory:
#   python model_analysis/analyse_results.py --all --out_dir out/my_analysis

import argparse
import json
import os
import re
import sys

# Make project root importable when this script is invoked as
#   python model_analysis/analyse_results.py …
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_ROOT = "out"

DICE_COLS = ["dice_GM", "dice_WM", "dice_CSF"]
VS_COLS   = ["vs_GM",   "vs_WM",   "vs_CSF"]
TISSUES   = ["GM", "WM", "CSF"]

FAMILY_PALETTE = {
    "std":             "#4C72B0",
    "film_simple":     "#DD8452",
    "film_complex":    "#55A868",
    "film_simple_res": "#C44E52",
    "film_complex_res":"#8172B2",
    "meta":            "#937860",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def discover_models(out_root: str) -> list[str]:
    model_ids = []
    for name in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, name)
        if (os.path.isdir(d)
                and os.path.isfile(os.path.join(d, "dice_results.csv"))
                and os.path.isfile(os.path.join(d, "meta.json"))):
            model_ids.append(name)
    return model_ids


def load_model_data(model_id: str, out_root: str) -> tuple[pd.DataFrame, dict]:
    csv_path  = os.path.join(out_root, model_id, "dice_results.csv")
    meta_path = os.path.join(out_root, model_id, "meta.json")
    df   = pd.read_csv(csv_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


_RES_RE = re.compile(r"_(res|residual)(?=_|$)")


def _strip_res(name: str) -> str:
    """Remove '_res' or '_residual' as a suffix or before another underscore."""
    return _RES_RE.sub("", name)


def build_master_df(model_ids: list[str], out_root: str) -> pd.DataFrame:
    parts = []
    for model_id in model_ids:
        try:
            df, meta = load_model_data(model_id, out_root)
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping {model_id}")
            continue
        df["model_id"]     = model_id
        df["model_family"] = meta.get("model_family", "unknown")
        df["val_loss"]     = meta.get("val_loss")
        parts.append(df)
    if not parts:
        raise RuntimeError("No valid model results found.")
    master = pd.concat(parts, ignore_index=True)

    # display_name: family with "_res"/"_residual" stripped. If two distinct models
    # collapse to the same stripped name, fall back to model_id so labels stay unique.
    unique = master.drop_duplicates("model_id")
    stripped_per_id = {row.model_id: _strip_res(row.model_family) for row in unique.itertuples()}
    stripped_counts = pd.Series(list(stripped_per_id.values())).value_counts()

    def _display(row):
        stripped = stripped_per_id[row["model_id"]]
        return stripped if stripped_counts.get(stripped, 0) == 1 else row["model_id"]

    master["display_name"] = master.apply(_display, axis=1)
    return master


def build_summary_df(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, grp in master.groupby("model_id", sort=False):
        row = {
            "model_id":     model_id,
            "display_name": grp["display_name"].iloc[0],
            "model_family": grp["model_family"].iloc[0],
            "val_loss":     grp["val_loss"].iloc[0],
            "n_patients":   len(grp),
        }
        for tissue in TISSUES:
            for prefix in ("dice", "vs"):
                col = f"{prefix}_{tissue}"
                if col in grp.columns:
                    row[f"{col}_mean"] = grp[col].mean()
                    row[f"{col}_std"]  = grp[col].std()
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"]  = np.nan
        for col, mkey in (("ssim_loss", "ssim"), ("l1_loss", "l1")):
            if col in grp.columns:
                row[f"{mkey}_mean"] = grp[col].mean()
                row[f"{mkey}_std"]  = grp[col].std()
            else:
                row[f"{mkey}_mean"] = np.nan
                row[f"{mkey}_std"]  = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "dice_GM_mean" if df["dice_GM_mean"].notna().any() else "ssim_mean"
    ascending = sort_col != "dice_GM_mean"
    return df.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)


# ─── Plots ────────────────────────────────────────────────────────────────────

def _model_colors(model_ids: list[str], master: pd.DataFrame) -> list[str]:
    fam_map = master.drop_duplicates("model_id").set_index("model_id")["model_family"].to_dict()
    return [FAMILY_PALETTE.get(fam_map.get(mid, ""), "#888888") for mid in model_ids]


def _zoom_ylim(ax, values, pad_frac: float = 0.1, hard_cap_high: float | None = None) -> None:
    """Auto-zoom y-axis to the actual data range with a margin on each side.
    `values` may contain NaNs; lower bound is never pushed below 0 for [0,1] metrics
    unless data is negative.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return
    lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 1e-6)
    pad  = span * pad_frac
    new_lo = max(0.0, lo - pad) if lo >= 0 else lo - pad
    new_hi = hi + pad
    if hard_cap_high is not None:
        new_hi = min(new_hi, hard_cap_high)
    ax.set_ylim(new_lo, new_hi)


def plot_bar_chart(summary: pd.DataFrame, metric: str, out_path: str) -> None:
    """Grouped bar chart: x=model, bars=tissue type, y=mean score."""
    labels = summary["display_name"].tolist()
    x      = np.arange(len(labels))
    width  = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    all_values = []
    for i, tissue in enumerate(TISSUES):
        means = summary[f"{metric}_{tissue}_mean"].values
        stds  = summary[f"{metric}_{tissue}_std"].fillna(0).values
        ax.bar(x + i * width, means, width, label=tissue,
               color=colors[i], yerr=stds, capsize=3, alpha=0.85)
        all_values.extend((means - stds).tolist())
        all_values.extend((means + stds).tolist())

    label = "Dice Score" if metric == "dice" else "Volumetric Similarity"
    ax.set_ylabel(label)
    ax.set_title(f"{label} by Model")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.legend()
    _zoom_ylim(ax, all_values, pad_frac=0.15, hard_cap_high=1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_boxplot(master: pd.DataFrame, metric: str, out_path: str) -> None:
    """3-panel boxplot: one subplot per tissue, x=model, y=per-patient score."""
    model_ids = master["model_id"].unique().tolist()
    name_map  = master.drop_duplicates("model_id").set_index("model_id")["display_name"].to_dict()
    labels    = [name_map[m] for m in model_ids]
    colors    = _model_colors(model_ids, master)

    # Per-tissue y-axes — CSF's wider variance otherwise crushes GM/WM detail.
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(model_ids) * 1.5), 5))
    label = "Dice" if metric == "dice" else "VS"

    for ax, tissue in zip(axes, TISSUES):
        col  = f"{metric}_{tissue}"
        data = [master.loc[master["model_id"] == mid, col].dropna().values
                for mid in model_ids]
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black"})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"{tissue}")
        ax.set_xticks(range(1, len(model_ids) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        _zoom_ylim(ax, master[col].dropna().values, pad_frac=0.1, hard_cap_high=1.0)

    fig.suptitle(f"{label} Score Distribution by Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_violin(master: pd.DataFrame, metric: str, out_path: str) -> None:
    """3-panel violin plot: one subplot per tissue."""
    model_ids = master["model_id"].unique().tolist()
    name_map  = master.drop_duplicates("model_id").set_index("model_id")["display_name"].to_dict()
    labels    = [name_map[m] for m in model_ids]
    label     = "Dice" if metric == "dice" else "VS"

    # Per-tissue y-axes — CSF's wider variance otherwise crushes GM/WM detail.
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(model_ids) * 1.5), 5))
    for ax, tissue in zip(axes, TISSUES):
        col = f"{metric}_{tissue}"
        df_melt = master[["display_name", col]].dropna()
        sns.violinplot(data=df_melt, x="display_name", y=col, ax=ax,
                       order=labels, hue="display_name", palette="muted",
                       legend=False, cut=0, inner="box")
        ax.set_title(tissue)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        _zoom_ylim(ax, df_melt[col].values, pad_frac=0.1, hard_cap_high=1.0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
            tick.set_fontsize(9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{label} Score Violin Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_val_loss_vs_dice(summary: pd.DataFrame, out_path: str) -> None:
    """Scatter: x=val_loss, y=mean dice, one point per model, coloured by family."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers   = {"GM": "o", "WM": "s", "CSF": "^"}
    all_yvals = []

    for tissue, marker in markers.items():
        for _, row in summary.iterrows():
            color = FAMILY_PALETTE.get(row["model_family"], "#888")
            y     = row[f"dice_{tissue}_mean"]
            ax.scatter(row["val_loss"], y, color=color, marker=marker, s=60, alpha=0.85)
            if pd.notna(y):
                all_yvals.append(float(y))

    for fam, color in FAMILY_PALETTE.items():
        if fam in summary["model_family"].values:
            ax.scatter([], [], color=color, label=fam, s=60)
    for tissue, marker in markers.items():
        ax.scatter([], [], color="grey", marker=marker, label=tissue, s=60)

    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Validation Loss vs Dice Score")
    ax.legend(fontsize=8, ncol=2)
    _zoom_ylim(ax, all_yvals, pad_frac=0.15, hard_cap_high=1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(summary: pd.DataFrame, out_path: str) -> None:
    """Two-panel heatmap: higher-is-better (Dice/VS) and lower-is-better (SSIM/L1).
    Colour ranges auto-zoom to actual data so small differences stay readable.
    """
    higher = {f"Dice {t}": f"dice_{t}_mean" for t in TISSUES}
    higher.update({f"VS {t}": f"vs_{t}_mean" for t in TISSUES})
    lower  = {"SSIM loss": "ssim_mean", "L1 loss": "l1_mean"}

    indexed = summary.set_index("display_name")
    df_h = indexed[[c for c in higher.values() if c in indexed.columns]]
    df_h.columns = [k for k, c in higher.items() if c in indexed.columns]
    df_l = indexed[[c for c in lower.values() if c in indexed.columns]]
    df_l.columns = [k for k, c in lower.items() if c in indexed.columns]

    show_lower = not df_l.dropna(how="all").empty
    ncols = 2 if show_lower else 1
    width_ratios = [df_h.shape[1] or 1, df_l.shape[1] or 1] if show_lower else [1]
    fig, axes = plt.subplots(1, ncols,
                             figsize=(3 + 1.2 * (df_h.shape[1] + (df_l.shape[1] if show_lower else 0)),
                                      max(4, len(summary) * 0.5)),
                             gridspec_kw={"width_ratios": width_ratios})
    if ncols == 1:
        axes = [axes]

    # Auto-zoomed colour range for Dice/VS: 5% padding around actual min/max
    h_vals = df_h.values[~np.isnan(df_h.values)]
    if h_vals.size:
        h_lo, h_hi = float(h_vals.min()), float(h_vals.max())
        pad = max((h_hi - h_lo) * 0.05, 1e-3)
        vmin_h, vmax_h = max(0.0, h_lo - pad), min(1.0, h_hi + pad)
    else:
        vmin_h, vmax_h = 0.0, 1.0

    sns.heatmap(df_h, ax=axes[0], annot=True, fmt=".3f", cmap="YlGn",
                vmin=vmin_h, vmax=vmax_h, linewidths=0.4,
                cbar_kws={"label": "Score (higher = better)"})
    axes[0].set_title("Dice + Volumetric Similarity")
    axes[0].set_ylabel("")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=9)

    if show_lower:
        l_vals = df_l.values[~np.isnan(df_l.values)]
        if l_vals.size:
            l_lo, l_hi = float(l_vals.min()), float(l_vals.max())
            pad = max((l_hi - l_lo) * 0.05, 1e-4)
            vmin_l, vmax_l = max(0.0, l_lo - pad), l_hi + pad
        else:
            vmin_l, vmax_l = 0.0, 0.1
        sns.heatmap(df_l, ax=axes[1], annot=True, fmt=".4f", cmap="YlOrRd",
                    vmin=vmin_l, vmax=vmax_l, linewidths=0.4,
                    cbar_kws={"label": "Loss (lower = better)"})
        axes[1].set_title("SSIM + L1 Loss")
        axes[1].set_ylabel("")
        axes[1].set_yticklabels([], fontsize=9)

    fig.suptitle("Per-Model Metric Heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_loss_bar(summary: pd.DataFrame, metric: str, out_path: str) -> None:
    """Bar chart of mean SSIM or L1 loss per model (lower is better)."""
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    if mean_col not in summary.columns or summary[mean_col].isna().all():
        print(f"  [skip] no {metric.upper()} data")
        return

    sub    = summary.dropna(subset=[mean_col]).sort_values(mean_col)
    labels = sub["display_name"].tolist()
    means  = sub[mean_col].values
    stds   = sub[std_col].fillna(0).values
    colors = [FAMILY_PALETTE.get(f, "#888") for f in sub["model_family"]]

    label = "SSIM Loss" if metric == "ssim" else "L1 Loss"
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5))
    ax.bar(range(len(labels)), means, yerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_ylabel(f"{label} (lower = better)")
    ax.set_title(f"{label} by Model")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    _zoom_ylim(ax, np.concatenate([means - stds, means + stds]), pad_frac=0.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_loss_boxplot(master: pd.DataFrame, metric: str, out_path: str) -> None:
    """Per-patient SSIM or L1 loss distribution per model."""
    col = f"{metric}_loss"
    if col not in master.columns or master[col].dropna().empty:
        print(f"  [skip] no per-patient {metric.upper()} data")
        return

    model_ids = master["model_id"].unique().tolist()
    name_map  = master.drop_duplicates("model_id").set_index("model_id")["display_name"].to_dict()
    labels    = [name_map[m] for m in model_ids]
    colors    = _model_colors(model_ids, master)
    label     = "SSIM Loss" if metric == "ssim" else "L1 Loss"

    data = [master.loc[master["model_id"] == mid, col].dropna().values for mid in model_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel(f"{label} (lower = better)")
    ax.set_title(f"{label} Distribution by Model")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    _zoom_ylim(ax, master[col].dropna().values, pad_frac=0.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_val_loss_vs_metric(summary: pd.DataFrame, metric_col: str,
                            ylabel: str, out_path: str) -> None:
    """Scatter: val_loss vs a single mean metric (lower or higher is better)."""
    if metric_col not in summary.columns or summary[metric_col].isna().all():
        print(f"  [skip] no {metric_col} data")
        return

    sub = summary.dropna(subset=["val_loss", metric_col])
    if sub.empty:
        print(f"  [skip] no overlap between val_loss and {metric_col}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sub.iterrows():
        color = FAMILY_PALETTE.get(row["model_family"], "#888")
        ax.scatter(row["val_loss"], row[metric_col], color=color, s=60, alpha=0.85)
    for fam, color in FAMILY_PALETTE.items():
        if fam in sub["model_family"].values:
            ax.scatter([], [], color=color, label=fam, s=60)
    ax.set_xlabel("Validation Loss")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Validation Loss vs {ylabel}")
    ax.legend(fontsize=8)
    _zoom_ylim(ax, sub[metric_col].values, pad_frac=0.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(summary: pd.DataFrame) -> None:
    def fmt(m, s):
        if pd.isna(m):
            return "    N/A   "
        return f"{m:.3f}±{s:.3f}" if pd.notna(s) else f"{m:.3f}"

    name_width = max(20, summary["display_name"].str.len().max() + 1)
    total_w    = name_width + 8 + 8 + 11 * 8 + 16
    print("\n" + "=" * total_w)
    print(
        f"{'Model':<{name_width}} {'VLoss':>6}  "
        f"{'Dice GM':>11}  {'Dice WM':>11}  {'Dice CSF':>11}  "
        f"{'VS GM':>11}  {'VS WM':>11}  {'VS CSF':>11}  "
        f"{'SSIM':>11}  {'L1':>11}"
    )
    print("-" * total_w)
    for _, row in summary.iterrows():
        vl = f"{row['val_loss']:.4f}" if pd.notna(row["val_loss"]) else "  N/A"
        print(
            f"{row['display_name']:<{name_width}} {vl:>6}  "
            f"{fmt(row['dice_GM_mean'], row['dice_GM_std']):>11}  "
            f"{fmt(row['dice_WM_mean'], row['dice_WM_std']):>11}  "
            f"{fmt(row['dice_CSF_mean'], row['dice_CSF_std']):>11}  "
            f"{fmt(row['vs_GM_mean'], row['vs_GM_std']):>11}  "
            f"{fmt(row['vs_WM_mean'], row['vs_WM_std']):>11}  "
            f"{fmt(row['vs_CSF_mean'], row['vs_CSF_std']):>11}  "
            f"{fmt(row['ssim_mean'], row['ssim_std']):>11}  "
            f"{fmt(row['l1_mean'], row['l1_std']):>11}"
        )
    print("=" * total_w)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse stored evaluation results from out/")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="Model IDs (directory names in out/)")
    group.add_argument("--all",    action="store_true", help="Discover all evaluated models in out/")
    parser.add_argument("--out_dir", default="out/analysis", help="Output directory for figures (default: out/analysis)")
    args = parser.parse_args()

    if args.all:
        model_ids = discover_models(OUT_ROOT)
        if not model_ids:
            print("No evaluated models found in out/ (need dice_results.csv + meta.json).")
            return
        print(f"Found {len(model_ids)} evaluated models: {model_ids}")
    else:
        model_ids = args.models

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading results for {len(model_ids)} model(s)...")
    master  = build_master_df(model_ids, OUT_ROOT)
    summary = build_summary_df(master)

    print_summary(summary)

    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\nGenerating figures...")
    plot_bar_chart(summary, "dice", os.path.join(args.out_dir, "dice_bar_chart.png"))
    plot_bar_chart(summary, "vs",   os.path.join(args.out_dir, "vs_bar_chart.png"))
    plot_boxplot(master, "dice",    os.path.join(args.out_dir, "dice_boxplot.png"))
    plot_boxplot(master, "vs",      os.path.join(args.out_dir, "vs_boxplot.png"))
    plot_violin(master, "dice",     os.path.join(args.out_dir, "dice_violin.png"))
    plot_violin(master, "vs",       os.path.join(args.out_dir, "vs_violin.png"))

    plot_loss_bar(summary, "ssim",   os.path.join(args.out_dir, "ssim_bar_chart.png"))
    plot_loss_bar(summary, "l1",     os.path.join(args.out_dir, "l1_bar_chart.png"))
    plot_loss_boxplot(master, "ssim",os.path.join(args.out_dir, "ssim_boxplot.png"))
    plot_loss_boxplot(master, "l1",  os.path.join(args.out_dir, "l1_boxplot.png"))

    if summary["val_loss"].notna().any():
        plot_val_loss_vs_dice(summary,  os.path.join(args.out_dir, "val_loss_vs_dice.png"))
        plot_val_loss_vs_metric(summary, "ssim_mean", "Mean SSIM Loss",
                                os.path.join(args.out_dir, "val_loss_vs_ssim.png"))
        plot_val_loss_vs_metric(summary, "l1_mean",   "Mean L1 Loss",
                                os.path.join(args.out_dir, "val_loss_vs_l1.png"))

    plot_heatmap(summary,           os.path.join(args.out_dir, "per_tissue_heatmap.png"))

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
