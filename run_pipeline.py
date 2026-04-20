"""
Run the full grid-search pipeline from the terminal.

Usage:
    python run_pipeline.py                                  # run all untrained entries
    python run_pipeline.py --model_filter std               # only std models
    python run_pipeline.py --model_filter film_simple       # only film simple models
    python run_pipeline.py --stop_after_first               # train one then stop
    python run_pipeline.py --model_filter film --stop_after_first
    python run_pipeline.py --setup                          # check env/data, then exit
"""

import argparse
import os
import sys

import torch

from utils.data_loader import DataLoader
from utils.grid_search import refresh_grid, run_pipeline
from utils.metadata_loader import HealthDataLoader

GRID_FILE     = "data/metadata/grid_search.yaml"
DATA_ROOT     = "data"
METADATA_ROOT = "data/metadata"
OUT_DIR       = "out/grid_search"


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

    print(f"  Grid file: {GRID_FILE}")
    print(f"  Output dir: {OUT_DIR}")
    print("=== Setup OK ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run the grid-search training pipeline.")
    parser.add_argument(
        "--model_filter", type=str, default=None,
        choices=["std", "film", "film_simple", "film_complex"],
        help="Restrict training to one model type (default: all)",
    )
    parser.add_argument(
        "--stop_after_first", action="store_true",
        help="Train the first untrained entry then exit",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run environment/data checks and exit without training",
    )
    args = parser.parse_args()

    if args.setup:
        setup_check()
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure grid YAML exists (safe — preserves completed entries)
    refresh_grid(GRID_FILE)

    # Stratified split
    data_loader   = DataLoader(root_path=DATA_ROOT)
    available_ids = set(data_loader.all_candidates)
    split_loader  = HealthDataLoader(root_path=METADATA_ROOT, external_features=True)
    train_ids, val_ids, _ = split_loader.split(train_split=0.70, val_split=0.15, seed=69)
    train_pairs = [data_loader.get_pair_path_from_id(i) for i in train_ids if i in available_ids]
    val_pairs   = [data_loader.get_pair_path_from_id(i) for i in val_ids   if i in available_ids]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}\n")

    run_pipeline(
        grid_file=GRID_FILE,
        device=device,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        metadata_root=METADATA_ROOT,
        out_dir=OUT_DIR,
        model_filter=args.model_filter,
        stop_after_first=args.stop_after_first,
    )


if __name__ == "__main__":
    main()
