"""Run Bayesian feature-selection search for EX3.

Searches over (top_k, correlation_threshold) for film_simple_res while
freezing training hyperparameters to the EX1 Bayesian champion. The SQLite
database and YAML audit log persist between sessions — stop and resume any
time.

Usage:
    python run_bayes_ex3.py                    # 30 trials with 5000-epoch default
    python run_bayes_ex3.py --n_trials 5
    python run_bayes_ex3.py --stop_after_first
    python run_bayes_ex3.py --dry_run          # load champion + SHAP, print sample
    python run_bayes_ex3.py --setup            # env/data check only
    python run_bayes_ex3.py --device cuda:1
"""

import argparse
import json
import os
import sys

import torch

from ex3.bayes_search_features import init_yaml_ex3, run_bayes_features
from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.model_utils.bayes_search import get_bayes_champion_entries
from utils.mri.data_loader import DataLoader

YAML_FILE      = "out/ex3/bayes_search/bayes_search.yaml"
DB_PATH        = "out/ex3/bayes_search/bayes_search.db"
OUT_DIR        = "out/ex3/bayes_search"
DATA_ROOT      = "data"
METADATA_ROOT  = "data/metadata"
SPLITS_PATH    = "data/metadata/splits.json"
EX1_YAML       = "out/bayes_search/bayes_search.yaml"

CHAMPION_KEY   = "film_simple_res"


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
    if not os.path.isfile(EX1_YAML):
        print(f"  ERROR: {EX1_YAML} not found (EX1 must be run first).")
        sys.exit(1)

    print(f"  Splits:     {SPLITS_PATH}")
    print(f"  EX1 YAML:   {EX1_YAML}")
    print(f"  EX3 YAML:   {YAML_FILE}")
    print(f"  EX3 DB:     {DB_PATH}")
    print(f"  Output dir: {OUT_DIR}")
    print("=== Setup OK ===\n")


def load_champion_hyperparams() -> tuple[str, dict]:
    """Fetch the EX1 film_simple_res champion entry and unpack the frozen HPs."""
    if not os.path.isfile(EX1_YAML):
        sys.exit(f"[ERROR] EX1 YAML not found at {EX1_YAML}. Run run_bayes.py first.")
    champions = get_bayes_champion_entries(EX1_YAML)
    if CHAMPION_KEY not in champions:
        sys.exit(f"[ERROR] No completed {CHAMPION_KEY} champion in {EX1_YAML}")
    entry = champions[CHAMPION_KEY]
    fixed = {
        "model_type":          "film",
        "film_generator_type": entry["film_generator_type"],
        "residual":            bool(entry["residual"]),
        "mlp_hidden":          int(entry["mlp_hidden"]),
        "learning_rate":       float(entry["learning_rate"]),
        "weight_decay":        float(entry["weight_decay"]),
        "beta1":               float(entry["beta1"]),
        "beta2":               float(entry["beta2"]),
        "lr_scheduler":        entry["lr_scheduler"],
    }
    return entry["id"], fixed


def load_split_pairs(data_loader: DataLoader) -> tuple[list, list]:
    """Load train/val pairs from data/metadata/splits.json (matches train_reduced.py).

    Using splits.json (rather than re-running StratifiedSplitter) guarantees
    byte-identical pairs to the EX2 SHAP run and to train_reduced.py.
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
        description="Run Bayesian feature-selection search for EX3."
    )
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Total new trials this session (default: 30)")
    parser.add_argument("--stop_after_first", action="store_true",
                        help="Run exactly one trial then exit")
    parser.add_argument("--setup", action="store_true",
                        help="Run environment/data checks and exit without training")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load champion HPs + SHAP CSVs, print a sampled trial, exit")
    parser.add_argument("--source_shap_model", type=str, default="film_simple_res_trial_012",
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

    # ── Fetch frozen HPs from EX1 champion ─────────────────────────────────
    champion_id, fixed = load_champion_hyperparams()
    print(f"\n=== Frozen hyperparameters (from EX1 champion {champion_id}) ===")
    for k, v in fixed.items():
        print(f"  {k:20s} = {v}")

    # ── Load splits & pairs ────────────────────────────────────────────────
    data_loader = DataLoader(root_path=DATA_ROOT)
    train_pairs, val_pairs = load_split_pairs(data_loader)
    print(f"\nTrain pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    if args.dry_run:
        # Build a one-off Optuna sample so the user can see the suggest space.
        import optuna
        import pandas as pd

        from ex3.bayes_search_features import _suggest_feature_params
        from ex3.feature_selection import (
            map_aggregated_to_raw_indices, select_features,
        )

        shap_dir = os.path.join("ex2", "results", args.source_shap_model, "analysis")
        imp_df  = pd.read_csv(os.path.join(shap_dir, "feature_importance.csv"))
        corr_df = pd.read_csv(os.path.join(shap_dir, "shap_correlation.csv"), index_col=0)
        print(f"\n=== SHAP source ===")
        print(f"  source_shap_model       : {args.source_shap_model}")
        print(f"  aggregated features     : {len(imp_df)}")
        print(f"  correlation matrix size : {corr_df.shape}")

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        fp = _suggest_feature_params(trial, top_k_max=int(len(imp_df)))
        kept, dropped = select_features(
            imp_df, corr_df, imp_thr=0.0,
            corr_thr=fp["correlation_threshold"], top_k=fp["top_k"],
        )
        meta_loader = CombinedMetadataUtils(
            csv_path=os.path.join(METADATA_ROOT, "metadata_combined.csv")
        )
        raw_indices, chosen = map_aggregated_to_raw_indices(kept, meta_loader.feature_names)
        print(f"\n=== Sampled trial ===")
        print(f"  top_k                 : {fp['top_k']}")
        print(f"  correlation_threshold : {fp['correlation_threshold']:.4f}")
        print(f"  kept aggregated ({len(kept)}): {kept}")
        if dropped:
            print(f"  dropped by correlation:")
            for keeper, drop_list in dropped.items():
                print(f"    {keeper} → {drop_list}")
        print(f"  → raw indices ({len(raw_indices)}): {chosen}")
        print("\n[dry_run] exiting before training.")
        return

    # ── Init YAML (no-op if it exists) ─────────────────────────────────────
    init_yaml_ex3(
        YAML_FILE,
        epochs=args.epochs,
        base_channels=args.base_channels,
        source_shap_model=args.source_shap_model,
        champion_id=champion_id,
        fixed_hyperparams=fixed,
        restart_threshold=args.restart_threshold,
        restart_check_epoch=args.restart_check_epoch,
        max_total_runs=args.max_total_runs,
    )

    # ── Run search ─────────────────────────────────────────────────────────
    run_bayes_features(
        yaml_file=YAML_FILE,
        db_path=DB_PATH,
        device=device,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        source_shap_model=args.source_shap_model,
        fixed_hyperparams=fixed,
        n_trials=args.n_trials,
        checkpoint_every=args.checkpoint_every,
        metadata_root=METADATA_ROOT,
        out_dir=OUT_DIR,
        stop_after_first=args.stop_after_first,
    )


if __name__ == "__main__":
    main()
