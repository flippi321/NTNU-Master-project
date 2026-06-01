# Train a feature-conditioned 3D model on a SHAP-selected SUBSET of metadata
# features.  The script reads the SHAP outputs from a previous ex2/ run and
# drops features below an importance threshold and/or that are too correlated
# with a more important feature, then trains a fresh model with the surviving
# subset.  Mirrors EX1's training pipeline but takes hyperparameters from the
# CLI (no Bayesian search).
#
# Examples (run from project root):
#
#   # 1. Dry run — print selection without training:
#   python ex3/train_reduced.py \
#       --source_model film_simple_res_trial_012 \
#       --model_type   film_simple_res \
#       --importance_threshold 0.001 \
#       --correlation_threshold 0.7 \
#       --mlp_hidden 131 --lr 5e-5 --weight_decay 1e-5 \
#       --dry_run
#
#   # 2. Real training:
#   python ex3/train_reduced.py \
#       --source_model film_simple_res_trial_012 \
#       --model_type   film_simple_res \
#       --device       cuda:0 \
#       --importance_threshold 0.001 \
#       --correlation_threshold 0.7 \
#       --epochs       5000 \
#       --lr           5e-5 \
#       --weight_decay 1e-5 \
#       --beta1 0.89 --beta2 0.98 \
#       --scheduler    cosine \
#       --mlp_hidden   131
#
#   # 3. Meta model with top-10 features only:
#   python ex3/train_reduced.py \
#       --source_model meta_trial_018 \
#       --model_type   meta \
#       --device       cuda:0 \
#       --top_k 10 \
#       --correlation_threshold 0.8 \
#       --n_meta_tokens 4 --num_heads 2 \
#       --lr 5e-5 --weight_decay 1e-5 --epochs 5000

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Make project root importable when this script is invoked as
#   python ex3/train_reduced.py …
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ex3.feature_selection import (
    map_aggregated_to_raw_indices, select_features,
)
from utils.metadata.combined_metadata_utils import (
    CombinedMetadataUtils, SubsetCombinedMetadataUtil,
)
from utils.model_utils.grid_search import _build_model, _build_scheduler
from utils.model_utils.loss_functions import ssim_l1_loss
from utils.model_utils.train_eval import fit_feature_based_3D
from utils.mri.data_loader import DataLoader

# ─── Constants ────────────────────────────────────────────────────────────────

CROP_AXES   = ((16, 10, 0), (17, 11, 17))
SPLITS_PATH = "data/metadata/splits.json"
SEED        = 69

_FILM_TYPES = {
    "film_simple":      ("simple",  False),
    "film_complex":     ("complex", False),
    "film_simple_res":  ("simple",  True),
    "film_complex_res": ("complex", True),
}

MODEL_TYPE_CHOICES = ["std", "meta"] + list(_FILM_TYPES.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────


def build_entry(model_type: str, mlp_hidden: int | None,
                n_meta_tokens: int | None, num_heads: int | None) -> dict:
    if model_type == "meta":
        if n_meta_tokens is None or num_heads is None:
            raise SystemExit("[ERROR] --n_meta_tokens and --num_heads required for meta.")
        return {
            "model_type":   "meta",
            "n_meta_tokens": int(n_meta_tokens),
            "num_heads":    int(num_heads),
        }
    if model_type in _FILM_TYPES:
        if mlp_hidden is None:
            raise SystemExit("[ERROR] --mlp_hidden is required for film_* model types.")
        gen_type, residual = _FILM_TYPES[model_type]
        return {
            "model_type":          "film",
            "film_generator_type": gen_type,
            "mlp_hidden":          int(mlp_hidden),
            "residual":            bool(residual),
        }
    raise SystemExit(f"[ERROR] Unsupported model_type: {model_type!r}")


def load_splits() -> tuple[list[int], list[int]]:
    with open(SPLITS_PATH) as f:
        s = json.load(f)
    return s["train"], s["val"]


def build_run_name(args, n_features_used: int) -> str:
    if args.name:
        return args.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{args.source_model}__{args.model_type}__{n_features_used}feat__{ts}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a model on a SHAP-selected metadata subset.")
    # Source / selection
    parser.add_argument("--source_model",          required=True, help="SHAP run id under ex2/results/<id>/analysis/")
    parser.add_argument("--importance_threshold",  type=float, default=0.0,
                        help="Drop features with mean_abs_shap below this (default 0 = keep all)")
    parser.add_argument("--top_k",                 type=int, default=None,
                        help="Alternative to --importance_threshold: keep only the top K features.")
    parser.add_argument("--correlation_threshold", type=float, default=1.0,
                        help="When |corr| >= this between two surviving features, drop the less-important one "
                             "(default 1.0 = disabled).")
    # Architecture
    parser.add_argument("--model_type",  required=True, choices=MODEL_TYPE_CHOICES)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--mlp_hidden",    type=int, default=None, help="FiLM only")
    parser.add_argument("--n_meta_tokens", type=int, default=None, help="meta only")
    parser.add_argument("--num_heads",     type=int, default=None, help="meta only")
    # Training
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--epochs",        type=int, default=5000)
    parser.add_argument("--lr",            type=float, required=True)
    parser.add_argument("--weight_decay",  type=float, required=True)
    parser.add_argument("--beta1",         type=float, default=0.9)
    parser.add_argument("--beta2",         type=float, default=0.999)
    parser.add_argument("--scheduler",     choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--checkpoint_every", type=int, default=250)
    # Output
    parser.add_argument("--name",          default=None, help="Output run name (default: auto)")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print the selected feature set and exit before training.")
    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────
    if args.model_type == "std":
        sys.exit("[ERROR] std models are unconditional — feature selection is meaningless.")
    if args.top_k is not None and args.importance_threshold > 0:
        sys.exit("[ERROR] Use either --top_k OR --importance_threshold, not both.")
    if args.top_k is not None and args.top_k < 1:
        sys.exit("[ERROR] --top_k must be >= 1.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Read SHAP analysis ─────────────────────────────────────────────────
    shap_dir = os.path.join("ex2", "results", args.source_model, "analysis")
    imp_csv  = os.path.join(shap_dir, "feature_importance.csv")
    corr_csv = os.path.join(shap_dir, "shap_correlation.csv")
    if not (os.path.isfile(imp_csv) and os.path.isfile(corr_csv)):
        sys.exit(f"[ERROR] Missing SHAP outputs at {shap_dir}. "
                 f"Run ex2/shap_analyse.py --model_id {args.source_model} first.")
    imp_df  = pd.read_csv(imp_csv)
    corr_df = pd.read_csv(corr_csv, index_col=0)

    kept_aggregated, dropped_for_corr = select_features(
        imp_df, corr_df,
        imp_thr=args.importance_threshold,
        corr_thr=args.correlation_threshold,
        top_k=args.top_k,
    )

    # ── Map to raw column indices ──────────────────────────────────────────
    full_loader = CombinedMetadataUtils()
    raw_feature_names = full_loader.feature_names
    raw_indices, chosen_raw_names = map_aggregated_to_raw_indices(
        kept_aggregated, raw_feature_names
    )

    print(f"\n=== Feature selection ===")
    print(f"  Source SHAP run        : {args.source_model}")
    print(f"  Importance threshold   : {args.importance_threshold}")
    print(f"  Top-k                  : {args.top_k}")
    print(f"  Correlation threshold  : {args.correlation_threshold}")
    print(f"  Kept aggregated features ({len(kept_aggregated)}): {kept_aggregated}")
    if dropped_for_corr:
        print(f"  Dropped by correlation:")
        for keeper, dropped in dropped_for_corr.items():
            print(f"    {keeper}  → dropped {dropped}")
    print(f"  → {len(raw_indices)} raw columns (of {full_loader.n_features}): {chosen_raw_names}")

    if args.dry_run:
        print("\n[dry_run] exiting before training.")
        return

    # ── Wrap loader ────────────────────────────────────────────────────────
    loader = SubsetCombinedMetadataUtil(base=full_loader, indices=raw_indices)
    assert loader.n_features == len(raw_indices)

    # ── Splits → pairs ─────────────────────────────────────────────────────
    train_ids, val_ids = load_splits()
    dl = DataLoader(root_path="data/")
    available = set(dl.all_candidates)
    train_pairs = [dl.get_pair_path_from_id(str(i)) for i in train_ids if str(i) in available]
    val_pairs   = [dl.get_pair_path_from_id(str(i)) for i in val_ids   if str(i) in available]
    print(f"\n=== Data ===")
    print(f"  Train pairs : {len(train_pairs)}")
    print(f"  Val pairs   : {len(val_pairs)}")

    # ── Model + optim ──────────────────────────────────────────────────────
    device = torch.device(args.device)
    entry  = build_entry(args.model_type, args.mlp_hidden,
                         args.n_meta_tokens, args.num_heads)
    model = _build_model(entry, base_channels=args.base_channels,
                         cond_dim=loader.n_features, device=device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay,
                            betas=(args.beta1, args.beta2))
    scheduler = _build_scheduler(optimizer, args.scheduler, args.epochs)
    print(f"\n=== Model ===")
    print(f"  Type        : {args.model_type}")
    print(f"  cond_dim    : {loader.n_features}")
    print(f"  base        : {args.base_channels}")
    print(f"  optimizer   : AdamW(lr={args.lr}, wd={args.weight_decay}, betas=({args.beta1},{args.beta2}))")
    print(f"  scheduler   : {args.scheduler}")
    print(f"  epochs      : {args.epochs}")

    # ── Train ──────────────────────────────────────────────────────────────
    run_name = build_run_name(args, n_features_used=loader.n_features)
    out_dir  = os.path.join("out", "ex3", run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Training → out/ex3/{run_name}/ ===")
    _, loss_history, _, best_model, best_val_loss = fit_feature_based_3D(
        model=model,
        device=device,
        training_pairs=train_pairs,
        validation_pairs=val_pairs,
        epochs=args.epochs,
        loss_func=ssim_l1_loss,
        metadata_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        crop_axes=CROP_AXES,
        checkpoint_every=args.checkpoint_every,
    )

    # ── Persist ────────────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "model.pth")
    torch.save(best_model.state_dict(), model_path)

    pd.DataFrame({"epoch": range(len(loss_history)),
                  "train_loss": loss_history}).to_csv(
        os.path.join(out_dir, "loss_history.csv"), index=False)

    selection = {
        "source_model":             args.source_model,
        "importance_threshold":     args.importance_threshold,
        "top_k":                    args.top_k,
        "correlation_threshold":    args.correlation_threshold,
        "kept_aggregated":          kept_aggregated,
        "dropped_for_correlation":  dropped_for_corr,
        "raw_indices":              raw_indices,
        "raw_feature_names":        chosen_raw_names,
        "n_features_original":      full_loader.n_features,
        "n_features_used":          loader.n_features,
    }
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump(selection, f, indent=2)

    meta = {
        "run_name":         run_name,
        "source_model":     args.source_model,
        "model_type":       args.model_type,
        "hyperparams": {
            "base_channels":  args.base_channels,
            "mlp_hidden":     args.mlp_hidden,
            "n_meta_tokens":  args.n_meta_tokens,
            "num_heads":      args.num_heads,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "beta1":          args.beta1,
            "beta2":          args.beta2,
            "scheduler":      args.scheduler,
        },
        "n_features_used":  loader.n_features,
        "epochs":           args.epochs,
        "best_val_loss":    float(best_val_loss),
        "trained_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. best_val_loss={best_val_loss:.4f}")
    print(f"  Saved: {model_path}")
    print(f"  Saved: {os.path.join(out_dir, 'selection.json')}")
    print(f"  Saved: {os.path.join(out_dir, 'meta.json')}")
    print(f"  Saved: {os.path.join(out_dir, 'loss_history.csv')}")


if __name__ == "__main__":
    main()
