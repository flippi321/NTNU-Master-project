"""Run unified Bayesian search for EX3.5 — hyperparameters + features.

Combines the search spaces of EX1 (training hyperparameters, scheduler fixed
to cosine) and EX3 (top_k, correlation_threshold) into one Optuna study, on
the film_simple_res architecture. SHAP outputs from EX2 are loaded once and
reused per trial — no SHAP recomputation.

Usage:
    python EX3.5-train-best-model.py                     # 26 trials, 5000 epochs
    python EX3.5-train-best-model.py --n_trials 5
    python EX3.5-train-best-model.py --stop_after_first
    python EX3.5-train-best-model.py --dry_run           # sample one trial, print, exit
    python EX3.5-train-best-model.py --setup             # env/data check only
    python EX3.5-train-best-model.py --device cuda:1
"""

import argparse
import json
import os
import sys

import torch

from ex3_5.bayes_search_hparams_features import (
    init_yaml_ex3_5, run_bayes_hparams_features,
)
from utils.mri.data_loader import DataLoader

YAML_FILE         = "out/ex3.5/bayes_search/bayes_search.yaml"
DB_PATH           = "out/ex3.5/bayes_search/bayes_search.db"
OUT_DIR           = "out/ex3.5/bayes_search"
DATA_ROOT         = "data"
METADATA_ROOT     = "data/metadata"
SPLITS_PATH       = "data/metadata/splits.json"
SHAP_SOURCE_MODEL = "film_simple_res_trial_012"
SHAP_ANALYSIS_DIR = os.path.join("ex2", "results", SHAP_SOURCE_MODEL, "analysis")


def setup_check():
    print("=== Setup check ===")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: CUDA not available — training will run on CPU (slow).")

    for split in ("HUNT3", "HUNT4"):
        path = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(path):
            print(f"  ERROR: {path} not found.")
            sys.exit(1)
        print(f"  {path}: {len(os.listdir(path))} subjects")

    if not os.path.isfile(SPLITS_PATH):
        print(f"  ERROR: {SPLITS_PATH} not found.")
        sys.exit(1)

    imp_csv  = os.path.join(SHAP_ANALYSIS_DIR, "feature_importance.csv")
    corr_csv = os.path.join(SHAP_ANALYSIS_DIR, "shap_correlation.csv")
    for required in (imp_csv, corr_csv):
        if not os.path.isfile(required):
            print(f"  ERROR: {required} not found (EX2 SHAP analysis required).")
            sys.exit(1)

    print(f"  Splits:       {SPLITS_PATH}")
    print(f"  SHAP source:  {SHAP_ANALYSIS_DIR}")
    print(f"  EX3.5 YAML:   {YAML_FILE}")
    print(f"  EX3.5 DB:     {DB_PATH}")
    print(f"  Output dir:   {OUT_DIR}")
    print("=== Setup OK ===\n")


def load_split_pairs(data_loader: DataLoader) -> tuple[list, list]:
    """Load train/val pairs from data/metadata/splits.json.

    Using splits.json (rather than re-running StratifiedSplitter) guarantees
    byte-identical pairs to the EX2 SHAP run and to EX3.
    """
    with open(SPLITS_PATH) as f:
        s = json.load(f)
    train_ids, val_ids = s["train"], s["val"]
    available = set(data_loader.all_candidates)
    train_pairs = [data_loader.get_pair_path_from_id(str(i)) for i in train_ids if str(i) in available]
    val_pairs   = [data_loader.get_pair_path_from_id(str(i)) for i in val_ids   if str(i) in available]
    return train_pairs, val_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Run unified Bayesian hyperparameter + feature search for EX3.5."
    )
    parser.add_argument("--n_trials", type=int, default=26,
                        help="Total new trials this session (default: 26)")
    parser.add_argument("--stop_after_first", action="store_true",
                        help="Run exactly one trial then exit")
    parser.add_argument("--setup", action="store_true",
                        help="Run environment/data checks and exit without training")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load SHAP, print a sampled trial, exit")
    parser.add_argument("--source_shap_model", type=str, default=SHAP_SOURCE_MODEL,
                        help="SHAP run id under ex2/results/<id>/analysis/")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Epochs per trial (written to YAML on first run; ignored thereafter)")
    parser.add_argument("--base_channels", type=int, default=16,
                        help="Base channel width (written to YAML on first run)")
    parser.add_argument("--restart_threshold", type=float, default=None,
                        help="Val-loss threshold at restart_check_epoch; restart with fresh weights if not met")
    parser.add_argument("--restart_check_epoch", type=int, default=500,
                        help="Epoch at which to check the restart threshold (default: 500)")
    parser.add_argument("--max_total_runs", type=int, default=3,
                        help="Maximum restart attempts per trial (default: 3)")
    parser.add_argument("--checkpoint_every", type=int, default=250,
                        help="How often (in epochs) to run validation (default: 250)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g. 'cuda:0'). Auto-detected if unset.")
    args = parser.parse_args()

    if args.setup:
        setup_check()
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Load splits & pairs ────────────────────────────────────────────────
    data_loader = DataLoader(root_path=DATA_ROOT)
    train_pairs, val_pairs = load_split_pairs(data_loader)
    print(f"\nTrain pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    if args.dry_run:
        import optuna
        import pandas as pd

        from ex3.feature_selection import (
            map_aggregated_to_raw_indices, select_features,
        )
        from ex3_5.bayes_search_hparams_features import _suggest_ex3_5
        from utils.metadata.combined_metadata_utils import CombinedMetadataUtils

        shap_dir = os.path.join("ex2", "results", args.source_shap_model, "analysis")
        imp_df  = pd.read_csv(os.path.join(shap_dir, "feature_importance.csv"))
        corr_df = pd.read_csv(os.path.join(shap_dir, "shap_correlation.csv"), index_col=0)
        print(f"\n=== SHAP source ===")
        print(f"  source_shap_model       : {args.source_shap_model}")
        print(f"  aggregated features     : {len(imp_df)}")
        print(f"  correlation matrix size : {corr_df.shape}")

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        sampled = _suggest_ex3_5(trial, top_k_max=int(len(imp_df)))
        kept, dropped = select_features(
            imp_df, corr_df, imp_thr=0.0,
            corr_thr=sampled["correlation_threshold"], top_k=sampled["top_k"],
        )
        meta_loader = CombinedMetadataUtils(
            csv_path=os.path.join(METADATA_ROOT, "metadata_combined.csv")
        )
        raw_indices, chosen = map_aggregated_to_raw_indices(kept, meta_loader.feature_names)
        print(f"\n=== Sampled trial ===")
        print(f"  learning_rate         : {sampled['learning_rate']:.3e}")
        print(f"  weight_decay          : {sampled['weight_decay']:.3e}")
        print(f"  beta1                 : {sampled['beta1']:.4f}")
        print(f"  beta2                 : {sampled['beta2']:.4f}")
        print(f"  mlp_hidden            : {sampled['mlp_hidden']}")
        print(f"  lr_scheduler          : {sampled['lr_scheduler']}  (fixed)")
        print(f"  top_k                 : {sampled['top_k']}")
        print(f"  correlation_threshold : {sampled['correlation_threshold']:.4f}")
        print(f"  kept aggregated ({len(kept)}): {kept}")
        if dropped:
            print(f"  dropped by correlation:")
            for keeper, drop_list in dropped.items():
                print(f"    {keeper} → {drop_list}")
        print(f"  → raw indices ({len(raw_indices)}): {chosen}")
        print("\n[dry_run] exiting before training.")
        return

    # ── Init YAML (no-op if it exists) ─────────────────────────────────────
    import pandas as pd
    top_k_max = int(len(pd.read_csv(os.path.join(SHAP_ANALYSIS_DIR, "feature_importance.csv"))))
    init_yaml_ex3_5(
        YAML_FILE,
        epochs=args.epochs,
        base_channels=args.base_channels,
        source_shap_model=args.source_shap_model,
        top_k_max=top_k_max,
        restart_threshold=args.restart_threshold,
        restart_check_epoch=args.restart_check_epoch,
        max_total_runs=args.max_total_runs,
    )

    # ── Run search ─────────────────────────────────────────────────────────
    run_bayes_hparams_features(
        yaml_file=YAML_FILE,
        db_path=DB_PATH,
        device=device,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        source_shap_model=args.source_shap_model,
        n_trials=args.n_trials,
        checkpoint_every=args.checkpoint_every,
        metadata_root=METADATA_ROOT,
        out_dir=OUT_DIR,
        stop_after_first=args.stop_after_first,
    )


if __name__ == "__main__":
    main()
