#!/usr/bin/env python3
"""
EX1 — Compare U-Nets: terminal runner.

Mirrors sections 1 (data setup) and 2 (Bayesian hyperparameter search) of
``EX1-compare-unets.ipynb``. All variables are exposed as CLI flags. Either
phase can be skipped independently.

Examples
--------
Full run with defaults::

    python EX1-compare-unets.py

Re-run only Bayesian search (data already prepared)::

    python EX1-compare-unets.py --skip-data-setup --n-trials 25

Force-regenerate every cached CSV / split file::

    python EX1-compare-unets.py --overwrite-data --skip-bayes
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import torch

from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.metadata.fastsurfer_utils import FastSurferLoader, generate_all_csvs
from utils.metadata.health_data_utils import HealthDataLoader
from utils.metadata.stratified_splitter import StratifiedSplitter
from utils.model_utils.bayes_search import init_yaml, run_bayes
from utils.mri.data_loader import DataLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = p.add_argument_group("paths")
    paths.add_argument("--root",            default="data/")
    paths.add_argument("--metadata-root",   default="data/metadata")
    paths.add_argument("--processed-dir",   default="data/metadata/processed")
    paths.add_argument("--fastsurfer-root", default="data/metadata/hdd/sMRI_generated")
    paths.add_argument("--subjects-dir",    default="data/metadata/hdd/full",
                       help="FreeSurfer/FastSurfer subject directory passed to generate_all_csvs.")
    paths.add_argument("--splits-path",     default="data/metadata/splits.json")
    paths.add_argument("--bayes-yaml",      default="out/bayes_search/bayes_search.yaml")
    paths.add_argument("--bayes-db",        default="out/bayes_search/bayes_search.db")
    paths.add_argument("--bayes-out-dir",   default="out/bayes_search")

    flow = p.add_argument_group("flow control")
    flow.add_argument("--overwrite-data",  action="store_true",
                      help="Re-run every cached step in the data pipeline.")
    flow.add_argument("--skip-data-setup", action="store_true",
                      help="Skip section 1 entirely; load splits from --splits-path.")
    flow.add_argument("--skip-bayes",      action="store_true",
                      help="Skip section 2 (Bayesian search).")
    flow.add_argument("--device", default=None,
                      help="Torch device string (e.g. 'cuda:0', 'cpu'). Auto-detected if unset.")

    split = p.add_argument_group("dataset split")
    split.add_argument("--seed",        type=int,   default=69)
    split.add_argument("--train-split", type=float, default=0.70)
    split.add_argument("--val-split",   type=float, default=0.15)

    bayes = p.add_argument_group("bayesian search")
    bayes.add_argument("--epochs",              type=int,   default=5000)
    bayes.add_argument("--base-channels",       type=int,   default=16)
    bayes.add_argument("--restart-threshold",   type=float, default=0.2)
    bayes.add_argument("--restart-check-epoch", type=int,   default=500)
    bayes.add_argument("--max-total-runs",      type=int,   default=3)
    bayes.add_argument("--n-trials",            type=int,   default=10)
    bayes.add_argument("--model-filter",        type=str,   default=None,
                       choices=["std", "film_simple", "film_complex",
                                "film_simple_res", "film_complex_res", "meta"],
                       help="Restrict Bayes to one model type. Default: round-robin across all.")

    return p.parse_args(argv)


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_data_setup(args: argparse.Namespace) -> tuple[list, list, list]:
    """Sav → normalize → fastsurfer → combine → split. Returns (train, val, test)."""
    print("\n=== 1. Data setup ===")
    health   = HealthDataLoader(root=args.metadata_root)
    fs       = FastSurferLoader(root=args.fastsurfer_root)
    comb     = CombinedMetadataUtils()
    splitter = StratifiedSplitter()

    health.sav_to_csv(overwrite=args.overwrite_data)

    cleaned_data_path = health.generate_cleaned(overwrite=args.overwrite_data)

    health_data_path = health.generate_normalized(
        overwrite=args.overwrite_data,
        health_data_path=cleaned_data_path,
        out_dir=args.processed_dir,
    )

    generate_all_csvs(
        overwrite=args.overwrite_data,
        subjects_dir=args.subjects_dir,
        output_dir=args.fastsurfer_root,
    )

    fs_data_path = fs.load_and_normalize(
        overwrite=args.overwrite_data,
        in_dir=args.fastsurfer_root,
        out_dir=args.processed_dir,
    )

    comb.get_overlap(
        health_data_path=health_data_path,
        fastsurfer_data_path=fs_data_path,
    )

    metadata_path = comb.combine(
        overwrite=args.overwrite_data,
        health_data_path=health_data_path,
        fastsurfer_data_path=fs_data_path,
        output_path=args.metadata_root,
    )
    metadata_df = pd.read_csv(metadata_path)

    train, val, test = splitter.split(
        metadata_df,
        id_col="hunt_id",
        save_path=args.splits_path,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def load_splits(splits_path: str) -> tuple[list, list, list]:
    with open(splits_path) as f:
        splits = json.load(f)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"Loaded splits from {splits_path}: Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def run_bayes_step(
    args: argparse.Namespace,
    train: list,
    val: list,
    test: list,
    device: torch.device,
) -> None:
    print("\n=== 2. Bayesian search ===")
    init_yaml(
        yaml_file=args.bayes_yaml,
        epochs=args.epochs,
        base_channels=args.base_channels,
        restart_threshold=args.restart_threshold,
        restart_check_epoch=args.restart_check_epoch,
        max_total_runs=args.max_total_runs,
    )

    data_loader = DataLoader(root_path=args.root)
    train_pairs = [data_loader.get_pair_path_from_id(hid) for hid in train]
    val_pairs   = [data_loader.get_pair_path_from_id(hid) for hid in val]
    test_pairs  = [data_loader.get_pair_path_from_id(hid) for hid in test]
    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}  Test pairs: {len(test_pairs)}")

    run_bayes(
        yaml_file=args.bayes_yaml,
        db_path=args.bayes_db,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        n_trials=args.n_trials,
        metadata_root=args.metadata_root,
        out_dir=args.bayes_out_dir,
        device=device,
        model_filter=args.model_filter,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.skip_data_setup:
        train, val, test = load_splits(args.splits_path)
    else:
        train, val, test = run_data_setup(args)

    if not args.skip_bayes:
        run_bayes_step(args, train, val, test, device)
    else:
        print("\nSkipping Bayesian search (--skip-bayes).")

    return 0

if __name__ == "__main__":
    sys.exit(main())