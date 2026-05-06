"""
Run Bayesian hyperparameter optimization using Optuna TPE.

Each session runs up to --n_trials trials. The SQLite database and YAML
audit log persist between sessions, so you can stop and resume at any time.

Usage:
    python run_bayes.py                                      # all types, 20 trials
    python run_bayes.py --model_filter std                   # only std models
    python run_bayes.py --model_filter film_simple           # only film_simple
    python run_bayes.py --model_filter film_complex          # only film_complex
    python run_bayes.py --model_filter film_simple_res       # film_simple with residual
    python run_bayes.py --model_filter film_complex_res      # film_complex with residual
    python run_bayes.py --model_filter meta                  # only meta UNet
    python run_bayes.py --n_trials 5                         # 5 new trials this session
    python run_bayes.py --stop_after_first                   # 1 trial then exit
    python run_bayes.py --setup                              # env/data check only
"""

import argparse
import os
import sys

import pandas as pd
import torch

from utils.model_utils.bayes_search import init_yaml, run_bayes
from utils.mri.data_loader import DataLoader
from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.metadata.stratified_splitter import StratifiedSplitter

YAML_FILE     = "out/bayes_search/bayes_search.yaml"
DB_PATH       = "out/bayes_search/bayes_search.db"
DATA_ROOT     = "data"
METADATA_ROOT = "data/metadata"
OUT_DIR       = "out/bayes_search"


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

    print(f"  YAML file: {YAML_FILE}")
    print(f"  DB file:   {DB_PATH}")
    print(f"  Output dir: {OUT_DIR}")
    print("=== Setup OK ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian hyperparameter optimization."
    )
    parser.add_argument(
        "--model_filter", type=str, default=None,
        choices=["std", "film_simple", "film_complex",
                 "film_simple_res", "film_complex_res", "meta"],
        help="Restrict to one model type (default: all, round-robin)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Total new trials to run this session across active model types (default: 20)",
    )
    parser.add_argument(
        "--stop_after_first", action="store_true",
        help="Run exactly one trial then exit",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run environment/data checks and exit without training",
    )
    parser.add_argument(
        "--epochs", type=int, default=4000,
        help="Epochs per trial (written to YAML on first run, ignored if YAML already exists)",
    )
    parser.add_argument(
        "--base_channels", type=int, default=16,
        help="Base channel width for all models (written to YAML on first run)",
    )
    parser.add_argument(
        "--restart_threshold", type=float, default=None,
        help="Val-loss threshold at restart_check_epoch; restart with fresh weights if not met",
    )
    parser.add_argument(
        "--restart_check_epoch", type=int, default=250,
        help="Epoch at which to check the restart threshold (default: 250)",
    )
    parser.add_argument(
        "--max_total_runs", type=int, default=3,
        help="Maximum restart attempts per trial (default: 3)",
    )
    args = parser.parse_args()

    if args.setup:
        setup_check()
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    init_yaml(
        YAML_FILE,
        epochs=args.epochs,
        base_channels=args.base_channels,
        restart_threshold=args.restart_threshold,
        restart_check_epoch=args.restart_check_epoch,
        max_total_runs=args.max_total_runs,
    )

    data_loader   = DataLoader(root_path=DATA_ROOT)
    available_ids = set(data_loader.all_candidates)
    meta_loader   = CombinedMetadataUtils()
    df            = pd.read_csv(meta_loader.csv_path)
    splitter      = StratifiedSplitter()
    train_ids, val_ids, _ = splitter.split(df, train_split=0.70, val_split=0.15, seed=69)
    train_pairs = [data_loader.get_pair_path_from_id(i) for i in train_ids if i in available_ids]
    val_pairs   = [data_loader.get_pair_path_from_id(i) for i in val_ids   if i in available_ids]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}\n")

    run_bayes(
        yaml_file=YAML_FILE,
        db_path=DB_PATH,
        device=device,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        n_trials=args.n_trials,
        metadata_root=METADATA_ROOT,
        out_dir=OUT_DIR,
        model_filter=args.model_filter,
        stop_after_first=args.stop_after_first,
    )


if __name__ == "__main__":
    main()
