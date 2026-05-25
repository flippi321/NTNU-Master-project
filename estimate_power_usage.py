#!/usr/bin/env python3
"""
estimate_power_usage.py — EX1 Power Usage Estimator

Runs one Bayesian optimization trial for a given model type for N epochs
(default 1000), wrapped in a CodeCarbon tracker. Produces an overview of:

  - Training time  (s/epoch → extrapolated to the full EX1 run)
  - Average power draw (W)  — GPU + CPU via CodeCarbon
  - Energy consumption (kWh) — measured + extrapolated to full run
  - CO₂-equivalent emissions (g)
  - Peak memory usage (GPU VRAM + RAM)

Assumes the data-setup phase of EX1-compare-unets.py has already been run
and the splits JSON exists at --splits-path.

Examples
--------
Single model::

    python estimate_power_usage.py --model std --splits-path data/metadata/splits.json

Three models in parallel (each on the same GPU; adjust --device for multi-GPU)::

    python estimate_power_usage.py --model std         --device cuda:0 &
    python estimate_power_usage.py --model film_simple --device cuda:0 &
    python estimate_power_usage.py --model meta        --device cuda:0 &
    wait

Each model writes to its own sub-directory under --out-dir so there are no
file collisions between parallel runs.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch
from codecarbon import EmissionsTracker

from utils.model_utils.bayes_search import init_yaml, run_bayes
from utils.mri.data_loader import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FULL_RUN_EPOCHS_DEFAULT = 5000   # EX1 default; used for extrapolation


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        required=True,
        choices=["std", "film_simple", "film_complex",
                 "film_simple_res", "film_complex_res", "meta"],
        help="Model family to benchmark.",
    )

    paths = p.add_argument_group("paths")
    paths.add_argument("--splits-path",   default="data/metadata/splits.json",
                       help="Pre-built splits JSON from EX1 data setup.")
    paths.add_argument("--root",          default="data/",
                       help="Root directory for MRI data pairs.")
    paths.add_argument("--metadata-root", default="data/metadata",
                       help="Directory containing metadata_combined.csv.")
    paths.add_argument("--out-dir",       default="out/power_estimation",
                       help="Parent output directory. A sub-dir per model is created.")

    train = p.add_argument_group("training")
    train.add_argument("--epochs",           type=int,   default=1000,
                       help="Number of epochs for the measurement run.")
    train.add_argument("--full-run-epochs",  type=int,   default=FULL_RUN_EPOCHS_DEFAULT,
                       help="Target epoch count for extrapolation (EX1 full run).")
    train.add_argument("--base-channels",    type=int,   default=16)
    train.add_argument("--checkpoint-every", type=int,   default=250,
                       help="Validation / checkpoint interval (epochs).")
    train.add_argument("--restart-threshold",   type=float, default=0.2)
    train.add_argument("--restart-check-epoch", type=int,   default=250)
    train.add_argument("--max-total-runs",       type=int,   default=3)

    p.add_argument("--device", default=None,
                   help="Torch device string, e.g. 'cuda:0', 'cpu'. Auto-detected if unset.")

    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirrored from EX1-compare-unets.py)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_splits(splits_path: str) -> tuple[list, list, list]:
    with open(splits_path) as f:
        splits = json.load(f)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"Loaded splits from {splits_path}: "
          f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xh Ym Zs'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────

def run_power_measurement(
    args: argparse.Namespace,
    train: list,
    val: list,
    test: list,
    device: torch.device,
) -> dict:
    """Run one Bayesian trial under CodeCarbon; return metrics dict."""

    # Per-model output dir so parallel runs don't collide
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    yaml_file = os.path.join(out_dir, "power_est.yaml")
    db_path   = os.path.join(out_dir, "power_est.db")

    init_yaml(
        yaml_file=yaml_file,
        epochs=args.epochs,
        base_channels=args.base_channels,
        restart_threshold=args.restart_threshold,
        restart_check_epoch=args.restart_check_epoch,
        max_total_runs=args.max_total_runs,
    )

    # Build data pairs
    data_loader  = DataLoader(root_path=args.root)
    train_pairs  = [data_loader.get_pair_path_from_id(hid) for hid in train]
    val_pairs    = [data_loader.get_pair_path_from_id(hid) for hid in val]
    test_pairs   = [data_loader.get_pair_path_from_id(hid) for hid in test]

    # Baseline memory snapshot
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ram_baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ── CodeCarbon tracker ────────────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name=f"power_est_{args.model}",
        output_dir=out_dir,
        log_level="critical",
        measure_power_secs=10,
        save_to_file=True,
        allow_multiple_runs=True,
    )

    print(f"\n[power] Starting measurement — model={args.model}  "
          f"epochs={args.epochs}  device={device}")

    t0 = time.perf_counter()
    tracker.start()

    run_bayes(
        yaml_file=yaml_file,
        db_path=db_path,
        device=device,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        n_trials=1,
        checkpoint_every=args.checkpoint_every,
        metadata_root=args.metadata_root,
        out_dir=out_dir,
        model_filter=args.model,
        stop_after_first=True,
    )

    emissions_kg = tracker.stop()
    elapsed      = time.perf_counter() - t0

    # ── Collect metrics ───────────────────────────────────────────────────────
    final       = tracker.final_emissions_data
    energy_kwh  = float(final.energy_consumed)          # kWh
    avg_power_w = energy_kwh * 3_600_000 / elapsed      # W = kWh → J → J/s

    scale           = args.full_run_epochs / args.epochs
    extrap_energy   = energy_kwh * scale
    extrap_time_s   = elapsed    * scale

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    # ru_maxrss on Linux is in kB; on macOS it's bytes — guard both
    peak_ram_kb  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_ram_mb  = peak_ram_kb / 1024          # Linux: kB → MB

    return dict(
        model           = args.model,
        device          = str(device),
        epochs          = args.epochs,
        full_run_epochs = args.full_run_epochs,
        elapsed_s       = elapsed,
        time_per_epoch  = elapsed / args.epochs,
        extrap_time_s   = extrap_time_s,
        energy_kwh      = energy_kwh,
        avg_power_w     = avg_power_w,
        extrap_energy   = extrap_energy,
        emissions_kg    = float(emissions_kg),
        peak_vram_mb    = peak_vram_mb,
        peak_ram_mb     = peak_ram_mb,
        out_dir         = out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(m: dict) -> str:
    title   = f"Power Estimation — {m['model']}"
    width   = 52
    border  = "═" * width
    lines = [
        f"╔{border}╗",
        f"║   {title:<{width - 3}}║",
        f"╚{border}╝",
        "",
        f"  Model              :  {m['model']}",
        f"  Device             :  {m['device']}",
        f"  Measured epochs    :  {m['epochs']}",
        f"  Extrapolation base :  {m['full_run_epochs']} epochs",
        "",
        "── Training Time " + "─" * (width - 16),
        f"  Per epoch          :  {m['time_per_epoch']:.2f} s",
        f"  For {m['epochs']:,} epochs      :  {_fmt_duration(m['elapsed_s'])}",
        f"  Extrap {m['full_run_epochs']:,} epochs :  {_fmt_duration(m['extrap_time_s'])}",
        "",
        "── Power & Energy " + "─" * (width - 17),
        f"  Avg power draw     :  {m['avg_power_w']:.1f} W",
        f"  Energy (measured)  :  {m['energy_kwh']:.6f} kWh",
        f"  Energy (extrap)    :  {m['extrap_energy']:.6f} kWh",
        f"  CO₂ emissions      :  {m['emissions_kg'] * 1000:.2f} g CO₂eq",
        "",
        "── Memory (peak) " + "─" * (width - 16),
        f"  GPU VRAM           :  {m['peak_vram_mb']:.0f} MB",
        f"  RAM                :  {m['peak_ram_mb']:.0f} MB",
        "",
        f"  Report saved → {m['out_dir']}/report.txt",
    ]
    return "\n".join(lines)


def print_report(m: dict) -> None:
    report = build_report(m)
    print("\n" + report + "\n")

    report_path = os.path.join(m["out_dir"], "report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args   = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train, val, test = load_splits(args.splits_path)

    metrics = run_power_measurement(args, train, val, test, device)
    print_report(metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
