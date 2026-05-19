"""Bayesian feature-selection search for EX3.

Each trial samples (top_k, correlation_threshold), greedily selects a
SHAP-driven feature subset from the EX2 outputs, then trains a fresh
film_simple_res model with the EX1 champion's hyperparameters frozen.
The trial's objective is best_val_loss (SSIM+L1).

Mirrors the structure of utils/model_utils/bayes_search.py (one Optuna study
backed by SQLite + a human-readable YAML audit log, lockfile-protected
appends, WAL-mode SQLite for safe concurrent multi-GPU sessions). Reuses the
locking/YAML helpers from that module rather than duplicating them.
"""

import os
import sqlite3 as _sqlite3
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim

from ex3.feature_selection import map_aggregated_to_raw_indices, select_features
from utils.metadata.combined_metadata_utils import (
    CombinedMetadataUtils, SubsetCombinedMetadataUtil,
)
from utils.model_utils.bayes_search import (
    _append_trial_to_yaml, _assert_metadata_coverage,
    _load_yaml, _save_yaml,
)
from utils.model_utils.grid_search import CROP_AXES, _build_model, _build_scheduler
from utils.model_utils.loss_functions import ssim_l1_loss
from utils.model_utils.train_eval import fit_feature_based_3D

optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_NAME_EX3 = "bayes_film_simple_res_features"


# ─────────────────────────────────────────────────────────────────────────────
# Suggest space
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_feature_params(trial: optuna.Trial, top_k_max: int) -> dict:
    """Sample (top_k, correlation_threshold) for one trial."""
    return {
        "top_k":                 trial.suggest_int("top_k", 1, top_k_max),
        "correlation_threshold": trial.suggest_float("correlation_threshold", 0.5, 1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# YAML persistence
# ─────────────────────────────────────────────────────────────────────────────

def init_yaml_ex3(
    yaml_file: str,
    epochs: int,
    base_channels: int,
    source_shap_model: str,
    champion_id: str,
    fixed_hyperparams: dict,
    restart_threshold: float | None = None,
    restart_check_epoch: int = 250,
    max_total_runs: int = 3,
) -> None:
    """Create bayes_search.yaml with a self-describing params header.

    Records the frozen EX1 champion HPs and the SHAP source run so the audit
    trail is reproducible without external lookups. No-op if file exists.
    """
    if os.path.exists(yaml_file):
        return
    os.makedirs(os.path.dirname(yaml_file) or ".", exist_ok=True)
    config = {
        "params": {
            "study_name":          STUDY_NAME_EX3,
            "epochs":              epochs,
            "base_channels":       base_channels,
            "source_shap_model":   source_shap_model,
            "champion_id":         champion_id,
            "fixed_hyperparams":   fixed_hyperparams,
            "restart_threshold":   restart_threshold,
            "restart_check_epoch": restart_check_epoch,
            "max_total_runs":      max_total_runs,
            "created_at":          datetime.now().isoformat(timespec="seconds"),
        },
        "trials": [],
    }
    _save_yaml(yaml_file, config)
    print(f"[bayes_ex3] Created {yaml_file}")


def _cleanup_incomplete_trials_ex3(storage: str, yaml_file: str) -> None:
    """Mark interrupted Optuna trials as FAIL and drop incomplete YAML entries.

    Single-study version of bayes_search._cleanup_incomplete_trials.
    """
    cleaned_optuna = 0
    try:
        study = optuna.load_study(study_name=STUDY_NAME_EX3, storage=storage)
        for frozen in study.trials:
            if frozen.state == optuna.trial.TrialState.RUNNING:
                study.tell(frozen.number, state=optuna.trial.TrialState.FAIL)
                cleaned_optuna += 1
                print(f"[bayes_ex3] Marked interrupted trial {frozen.number} as FAIL")
    except Exception:
        pass

    config = _load_yaml(yaml_file)
    trials = config.get("trials", [])
    complete = [t for t in trials if t.get("results", {}).get("completed", False)]
    removed = len(trials) - len(complete)
    if removed > 0:
        config["trials"] = complete
        _save_yaml(yaml_file, config)
        print(f"[bayes_ex3] Removed {removed} incomplete YAML entry(s)")

    if cleaned_optuna or removed:
        print(f"[bayes_ex3] Cleanup done — {cleaned_optuna} Optuna trial(s) failed, "
              f"{removed} YAML entry(s) dropped")


# ─────────────────────────────────────────────────────────────────────────────
# Core trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_feature_trial(
    feature_params: dict,
    fixed_hyperparams: dict,
    trial_id: str,
    base_channels: int,
    epochs: int,
    device: torch.device,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    checkpoint_every: int,
    trial_out_dir: str,
    imp_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    full_loader: CombinedMetadataUtils,
    restart_threshold: float | None,
    restart_check_epoch: int,
    max_total_runs: int,
) -> tuple[float, str, dict]:
    """Apply feature selection, build model, train. Returns (best_val_loss,
    model_path, selection_dict)."""
    top_k    = int(feature_params["top_k"])
    corr_thr = float(feature_params["correlation_threshold"])

    # ── Selection ──────────────────────────────────────────────────────────
    kept_aggregated, dropped_for_corr = select_features(
        imp_df, corr_df,
        imp_thr=0.0,       # disabled — we use top_k instead
        corr_thr=corr_thr,
        top_k=top_k,
    )
    raw_indices, chosen_raw_names = map_aggregated_to_raw_indices(
        kept_aggregated, full_loader.feature_names,
    )
    loader = SubsetCombinedMetadataUtil(base=full_loader, indices=raw_indices)

    selection = {
        "source_model":            None,   # filled in by run_bayes_features
        "top_k":                   top_k,
        "correlation_threshold":   corr_thr,
        "kept_aggregated":         kept_aggregated,
        "dropped_for_correlation": dropped_for_corr,
        "raw_indices":             raw_indices,
        "raw_feature_names":       chosen_raw_names,
        "n_features_original":     full_loader.n_features,
        "n_features_aggregated":   len(kept_aggregated),
        "n_features_raw":          loader.n_features,
    }

    # ── Model ──────────────────────────────────────────────────────────────
    model = _build_model(
        fixed_hyperparams,
        base_channels=base_channels,
        cond_dim=loader.n_features,
        device=device,
    )
    lr    = float(fixed_hyperparams["learning_rate"])
    wd    = float(fixed_hyperparams["weight_decay"])
    betas = (float(fixed_hyperparams["beta1"]), float(fixed_hyperparams["beta2"]))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    scheduler = _build_scheduler(optimizer, fixed_hyperparams["lr_scheduler"], epochs)

    # ── Train ──────────────────────────────────────────────────────────────
    _, loss_history, _, best_model, best_val_loss = fit_feature_based_3D(
        model=model,
        device=device,
        training_pairs=train_pairs,
        validation_pairs=val_pairs,
        epochs=epochs,
        loss_func=ssim_l1_loss,
        metadata_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        crop_axes=CROP_AXES,
        checkpoint_every=checkpoint_every,
        restart_threshold=restart_threshold,
        restart_check_epoch=restart_check_epoch,
        max_total_runs=max_total_runs,
        use_autocast=False,
    )

    # ── Persist artifacts (mirror train_reduced.py output layout) ──────────
    os.makedirs(trial_out_dir, exist_ok=True)
    model_path = os.path.join(trial_out_dir, "model.pth")
    torch.save(best_model.state_dict(), model_path)

    pd.DataFrame({
        "epoch":      range(len(loss_history)),
        "train_loss": loss_history,
    }).to_csv(os.path.join(trial_out_dir, "loss_history.csv"), index=False)

    return float(best_val_loss), model_path, selection


# ─────────────────────────────────────────────────────────────────────────────
# Public driver
# ─────────────────────────────────────────────────────────────────────────────

def run_bayes_features(
    yaml_file: str,
    db_path: str,
    device: torch.device,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    source_shap_model: str,
    fixed_hyperparams: dict,
    n_trials: int = 30,
    checkpoint_every: int = 250,
    metadata_root: str = "data/metadata",
    out_dir: str = "out/ex3/bayes_search",
    stop_after_first: bool = False,
) -> int:
    """Run up to n_trials Bayesian feature-selection trials.

    Loads the EX2 SHAP outputs once, opens (or creates) a single Optuna
    study, and loops ask/train/tell. Resumable across sessions via the
    SQLite DB + YAML audit log.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Read YAML params (written by init_yaml_ex3) ────────────────────────
    config = _load_yaml(yaml_file)
    params = config.get("params", {})
    epochs              = int(params["epochs"])
    base_channels       = int(params["base_channels"])
    restart_threshold   = params.get("restart_threshold")
    restart_check_epoch = int(params.get("restart_check_epoch", 250))
    max_total_runs      = int(params.get("max_total_runs", 3))
    if restart_threshold is not None:
        restart_threshold = float(restart_threshold)

    print(f"[bayes_ex3] epochs={epochs}  base_channels={base_channels}  "
          f"restart_threshold={restart_threshold}  restart_check_epoch={restart_check_epoch}  "
          f"max_total_runs={max_total_runs}")

    # ── Load SHAP outputs once ─────────────────────────────────────────────
    shap_dir = os.path.join("ex2", "results", source_shap_model, "analysis")
    imp_csv  = os.path.join(shap_dir, "feature_importance.csv")
    corr_csv = os.path.join(shap_dir, "shap_correlation.csv")
    if not (os.path.isfile(imp_csv) and os.path.isfile(corr_csv)):
        raise FileNotFoundError(
            f"[bayes_ex3] Missing SHAP outputs at {shap_dir}. "
            f"Expected {imp_csv} and {corr_csv}."
        )
    imp_df  = pd.read_csv(imp_csv)
    corr_df = pd.read_csv(corr_csv, index_col=0)
    top_k_max = int(len(imp_df))
    print(f"[bayes_ex3] SHAP source: {source_shap_model}  "
          f"({top_k_max} aggregated features available)")

    # ── Full metadata loader + coverage pre-flight ─────────────────────────
    full_loader = CombinedMetadataUtils(
        csv_path=os.path.join(metadata_root, "metadata_combined.csv")
    )
    _assert_metadata_coverage(full_loader, train_pairs, "train")
    _assert_metadata_coverage(full_loader, val_pairs, "val")

    # ── SQLite WAL mode + 60s busy_timeout for safe concurrent sessions ────
    with _sqlite3.connect(db_path, timeout=60) as _conn:
        _conn.execute("PRAGMA journal_mode=WAL")
    storage = f"sqlite:///{db_path}?timeout=60"

    _cleanup_incomplete_trials_ex3(storage, yaml_file)

    study = optuna.create_study(
        study_name=STUDY_NAME_EX3,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=None),
        load_if_exists=True,
    )

    completed = 0
    for _ in range(n_trials):
        trial    = study.ask()
        fp       = _suggest_feature_params(trial, top_k_max=top_k_max)
        trial_id = f"feat_trial_{trial.number + 1:03d}"
        trial_out_dir = os.path.join(out_dir, trial_id)

        print(f"\n[bayes_ex3] Starting {trial_id}  "
              f"top_k={fp['top_k']}  corr_thr={fp['correlation_threshold']:.3f}")

        # Breadcrumb entry — flipped to completed=True on success
        entry: dict = {
            "id":                  trial_id,
            "optuna_trial_number": trial.number + 1,
            "study_name":          STUDY_NAME_EX3,
            "source_shap_model":   source_shap_model,
            "fixed_hyperparams":   fixed_hyperparams,
            "feature_params":      fp,
            "results":             {"completed": False},
        }
        _append_trial_to_yaml(yaml_file, entry)

        try:
            best_val_loss, model_path, selection = _run_one_feature_trial(
                feature_params=fp,
                fixed_hyperparams=fixed_hyperparams,
                trial_id=trial_id,
                base_channels=base_channels,
                epochs=epochs,
                device=device,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                checkpoint_every=checkpoint_every,
                trial_out_dir=trial_out_dir,
                imp_df=imp_df,
                corr_df=corr_df,
                full_loader=full_loader,
                restart_threshold=restart_threshold,
                restart_check_epoch=restart_check_epoch,
                max_total_runs=max_total_runs,
            )
            selection["source_model"] = source_shap_model

            # Persist per-trial selection.json (same schema as train_reduced.py)
            import json
            with open(os.path.join(trial_out_dir, "selection.json"), "w") as f:
                json.dump(selection, f, indent=2)

            study.tell(trial, best_val_loss)
            entry.update({
                "kept_aggregated":       selection["kept_aggregated"],
                "n_features_aggregated": selection["n_features_aggregated"],
                "n_features_raw":        selection["n_features_raw"],
                "results": {
                    "best_val_loss": best_val_loss,
                    "completed":     True,
                    "trained_at":    datetime.now().isoformat(timespec="seconds"),
                    "model_path":    model_path,
                },
            })
            print(f"[bayes_ex3] Done {trial_id}  best_val_loss={best_val_loss:.6f}  → {model_path}")

        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            entry["results"] = {"completed": False, "error": str(e)}
            print(f"[bayes_ex3] FAILED {trial_id}: {e}")

        _append_trial_to_yaml(yaml_file, entry)
        completed += 1

        if stop_after_first:
            break

    print(f"\n[bayes_ex3] Session complete. Ran {completed} trial(s).")
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Read-back helper
# ─────────────────────────────────────────────────────────────────────────────

def get_ex3_champion_entry(yaml_file: str) -> dict | None:
    """Return the best completed trial from the EX3 YAML, or None if empty."""
    config = _load_yaml(yaml_file)
    best: dict | None = None
    best_loss = np.inf
    for entry in config.get("trials", []):
        results = entry.get("results", {})
        if not results.get("completed", False):
            continue
        loss = float(results["best_val_loss"])
        if loss < best_loss:
            best_loss = loss
            best = entry
    return best
