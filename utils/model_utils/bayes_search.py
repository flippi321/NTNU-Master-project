import os
import tempfile
from datetime import datetime

import optuna
import torch
import torch.optim as optim
import yaml

from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.model_utils.grid_search import CROP_AXES, _build_model, _build_scheduler
from utils.model_utils.loss_functions import ssim_l1_loss
from utils.model_utils.train_eval import fit_3D, fit_feature_based_3D

optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_NAMES: dict[str, str] = {
    "film_simple":  "bayes_film_simple",
    "film_complex": "bayes_film_complex",
    "std":          "bayes_std",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _type_key_from_entry(entry: dict) -> str:
    if entry["model_type"] == "std":
        return "std"
    return f"film_{entry.get('film_generator_type', 'unknown')}"


def _suggest_hyperparams(trial: optuna.Trial, model_type_key: str) -> dict:
    """Suggest hyperparameters for a trial; returns dict compatible with _build_model()."""
    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    beta1        = trial.suggest_float("beta1", 0.85, 0.93)
    beta2        = trial.suggest_float("beta2", 0.95, 0.999)
    scheduler    = trial.suggest_categorical("scheduler", ["cosine", "plateau"])

    params: dict = dict(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        lr_scheduler=scheduler,
    )

    if model_type_key == "std":
        params["model_type"] = "std"
    else:
        gen_type = "simple" if model_type_key == "film_simple" else "complex"
        params["model_type"] = "film"
        params["film_generator_type"] = gen_type
        params["mlp_hidden"] = trial.suggest_int("mlp_hidden", 32, 256, log=True)

    return params


def _make_trial_id(model_type_key: str, trial_number: int) -> str:
    return f"{model_type_key}_trial_{trial_number:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# YAML persistence
# ─────────────────────────────────────────────────────────────────────────────

def init_yaml(
    yaml_file: str,
    epochs: int,
    base_channels: int,
    restart_threshold: float | None = None,
    restart_check_epoch: int = 250,
    max_total_runs: int = 3,
) -> None:
    """Create bayes_search.yaml with params header. No-op if already exists."""
    if os.path.exists(yaml_file):
        return
    os.makedirs(os.path.dirname(yaml_file) or ".", exist_ok=True)
    config = {
        "params": {
            "epochs": epochs,
            "base_channels": base_channels,
            "restart_threshold": restart_threshold,
            "restart_check_epoch": restart_check_epoch,
            "max_total_runs": max_total_runs,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "trials": [],
    }
    _save_yaml(yaml_file, config)
    print(f"[bayes_search] Created {yaml_file}")


def _load_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        return yaml.safe_load(f) or {"params": {}, "trials": []}


def _save_yaml(yaml_file: str, config: dict) -> None:
    """Atomically write config to yaml_file using a temp file + os.replace."""
    dir_ = os.path.dirname(yaml_file) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, yaml_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_trial_to_yaml(yaml_file: str, entry: dict) -> None:
    """Atomically append or update a trial entry in the YAML file."""
    config = _load_yaml(yaml_file)
    trials = config.setdefault("trials", [])
    for i, t in enumerate(trials):
        if t.get("id") == entry["id"]:
            trials[i] = entry
            _save_yaml(yaml_file, config)
            return
    trials.append(entry)
    _save_yaml(yaml_file, config)


# ─────────────────────────────────────────────────────────────────────────────
# Core training
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_trial(
    model_type_key: str,
    hyperparams: dict,
    trial_id: str,
    base_channels: int,
    epochs: int,
    device: torch.device,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    checkpoint_every: int,
    metadata_root: str,
    out_dir: str,
    combined_loader: "CombinedMetadataUtils | None",
    restart_threshold: float | None,
    restart_check_epoch: int,
    max_total_runs: int,
) -> tuple[float, str, "float | None"]:
    """Build model, train, save checkpoint. Returns (best_val_loss, model_path, final_train_loss)."""
    lr         = float(hyperparams["learning_rate"])
    wd         = float(hyperparams["weight_decay"])
    betas      = (float(hyperparams["beta1"]), float(hyperparams["beta2"]))
    sched_type = hyperparams["lr_scheduler"]

    if model_type_key == "std":
        model     = _build_model(hyperparams, base_channels=base_channels)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        scheduler = _build_scheduler(optimizer, sched_type, epochs)
        _, loss_history, _, best_model, best_val_loss = fit_3D(
            model=model,
            device=device,
            training_pairs=train_pairs,
            validation_pairs=val_pairs,
            epochs=epochs,
            loss_func=ssim_l1_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            crop_axes=CROP_AXES,
            checkpoint_every=checkpoint_every,
            restart_threshold=restart_threshold,
            restart_check_epoch=restart_check_epoch,
            max_total_runs=max_total_runs,
        )
    else:
        model     = _build_model(hyperparams, base_channels=base_channels, cond_dim=combined_loader.n_features)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        scheduler = _build_scheduler(optimizer, sched_type, epochs)
        _, loss_history, _, best_model, best_val_loss = fit_feature_based_3D(
            model=model,
            device=device,
            training_pairs=train_pairs,
            validation_pairs=val_pairs,
            epochs=epochs,
            loss_func=ssim_l1_loss,
            metadata_loader=combined_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            crop_axes=CROP_AXES,
            checkpoint_every=checkpoint_every,
            restart_threshold=restart_threshold,
            restart_check_epoch=restart_check_epoch,
            max_total_runs=max_total_runs,
        )

    model_path = os.path.join(out_dir, f"{trial_id}_({best_val_loss:.4f}).pth")
    torch.save(best_model.state_dict(), model_path)

    final_train_loss = float(loss_history[-1]) if loss_history else None
    return float(best_val_loss), model_path, final_train_loss


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_incomplete_trials(
    active_types: list[str],
    storage: str,
    yaml_file: str,
) -> None:
    """Mark interrupted Optuna trials as FAIL and remove incomplete YAML entries."""
    cleaned_optuna = 0
    for type_key in active_types:
        study_name = STUDY_NAMES[type_key]
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            continue
        for frozen in study.trials:
            if frozen.state == optuna.trial.TrialState.RUNNING:
                study.tell(frozen.number, state=optuna.trial.TrialState.FAIL)
                cleaned_optuna += 1
                print(f"[bayes_search] Marked interrupted trial {frozen.number} in {study_name} as FAIL")

    config = _load_yaml(yaml_file)
    trials = config.get("trials", [])
    complete = [t for t in trials if t.get("results", {}).get("completed", False)]
    removed = len(trials) - len(complete)
    if removed > 0:
        config["trials"] = complete
        _save_yaml(yaml_file, config)
        print(f"[bayes_search] Removed {removed} incomplete YAML entry(s)")

    if cleaned_optuna or removed:
        print(f"[bayes_search] Cleanup done — {cleaned_optuna} Optuna trial(s) failed, {removed} YAML entry(s) dropped")


def run_bayes(
    yaml_file: str,
    db_path: str,
    device: torch.device,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    n_trials: int = 20,
    checkpoint_every: int = 250,
    metadata_root: str = "data/metadata",
    out_dir: str = "out/bayes_search",
    model_filter: str | None = None,
    stop_after_first: bool = False,
) -> int:
    """
    Run up to n_trials Bayesian optimization trials using Optuna TPE.

    One study per model type, backed by a shared SQLite database at db_path.
    Resumable: calling run_bayes() again continues from where it left off.
    After each trial, appends results to yaml_file.

    Training config (epochs, base_channels, restart_*) is read from the YAML
    params section written by init_yaml — there is no duplication.

    Parameters
    ----------
    yaml_file        : path to bayes_search.yaml (human-readable audit log)
    db_path          : path to SQLite file, e.g. "data/bayes_search.db"
    n_trials         : total NEW trials to run this session across all active types
    checkpoint_every : how often (in epochs) to run validation inside each trial
    model_filter     : None (all types), 'std', 'film_simple', or 'film_complex'
    stop_after_first : run exactly one trial then return

    Returns
    -------
    int  Number of trials completed in this call.
    """
    os.makedirs(out_dir, exist_ok=True)

    # All fixed training config lives in the YAML params written by init_yaml
    config = _load_yaml(yaml_file)
    params = config.get("params", {})
    epochs               = int(params["epochs"])
    base_channels        = int(params["base_channels"])
    restart_threshold    = params.get("restart_threshold")
    restart_check_epoch  = int(params.get("restart_check_epoch", 250))
    max_total_runs       = int(params.get("max_total_runs", 3))

    if restart_threshold is not None:
        restart_threshold = float(restart_threshold)

    print(
        f"[bayes_search] epochs={epochs}  base_channels={base_channels}  "
        f"restart_threshold={restart_threshold}  restart_check_epoch={restart_check_epoch}  "
        f"max_total_runs={max_total_runs}"
    )

    active_types = [k for k in STUDY_NAMES if model_filter is None or k == model_filter]
    if not active_types:
        raise ValueError(
            f"Unknown model_filter: {model_filter!r}. "
            "Use None, 'std', 'film_simple', or 'film_complex'."
        )

    storage = f"sqlite:///{db_path}"
    _cleanup_incomplete_trials(active_types, storage, yaml_file)
    _combined_loader: CombinedMetadataUtils | None = None

    def _get_loader() -> CombinedMetadataUtils:
        nonlocal _combined_loader
        if _combined_loader is None:
            _combined_loader = CombinedMetadataUtils(
                csv_path=os.path.join(metadata_root, "metadata_combined.csv")
            )
        return _combined_loader

    completed = 0
    
    for i in range(n_trials):
        # Round-robin across active model types
        model_type_key = active_types[i % len(active_types)]

        study = optuna.create_study(
            study_name=STUDY_NAMES[model_type_key],
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
        )

        trial = study.ask()
        hyperparams = _suggest_hyperparams(trial, model_type_key)
        trial_id = _make_trial_id(model_type_key, trial.number + 1)

        print(f"\n[bayes_search] Starting {trial_id}")
        film_info = (
            f"  film_type={hyperparams['film_generator_type']}  mlp_hidden={hyperparams.get('mlp_hidden')}"
            if model_type_key != "std" else ""
        )
        print(
            f"  lr={hyperparams['learning_rate']:.2e}  wd={hyperparams['weight_decay']:.2e}"
            f"  beta1={hyperparams['beta1']:.3f}  beta2={hyperparams['beta2']:.3f}"
            f"  sched={hyperparams['lr_scheduler']}{film_info}"
        )

        entry: dict = {
            "id": trial_id,
            "optuna_trial_number": trial.number + 1,
            "study_name": STUDY_NAMES[model_type_key],
            "model_type": hyperparams["model_type"],
            "learning_rate": hyperparams["learning_rate"],
            "weight_decay": hyperparams["weight_decay"],
            "beta1": hyperparams["beta1"],
            "beta2": hyperparams["beta2"],
            "lr_scheduler": hyperparams["lr_scheduler"],
        }
        if model_type_key != "std":
            entry["film_generator_type"] = hyperparams["film_generator_type"]
            entry["mlp_hidden"] = hyperparams["mlp_hidden"]

        try:
            loader = None if model_type_key == "std" else _get_loader()
            best_val_loss, model_path, final_train_loss = _run_one_trial(
                model_type_key=model_type_key,
                hyperparams=hyperparams,
                trial_id=trial_id,
                base_channels=base_channels,
                epochs=epochs,
                device=device,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                checkpoint_every=checkpoint_every,
                metadata_root=metadata_root,
                out_dir=out_dir,
                combined_loader=loader,
                restart_threshold=restart_threshold,
                restart_check_epoch=restart_check_epoch,
                max_total_runs=max_total_runs,
            )
            study.tell(trial, best_val_loss)
            entry["results"] = {
                "best_val_loss": best_val_loss,
                "final_train_loss": final_train_loss,
                "completed": True,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "model_path": model_path,
            }
            print(f"[bayes_search] Done {trial_id}  best_val_loss={best_val_loss:.6f}  → {model_path}")

        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            entry["results"] = {"completed": False, "error": str(e)}
            print(f"[bayes_search] FAILED {trial_id}: {e}")

        _append_trial_to_yaml(yaml_file, entry)
        completed += 1

        if stop_after_first:
            break

    print(f"\n[bayes_search] Session complete. Ran {completed} trial(s).")
    return completed


def get_bayes_champion_entries(yaml_file: str) -> dict[str, dict]:
    """
    Return the best completed trial entry per model type from bayes_search.yaml.

    Returns
    -------
    dict mapping type key ('std', 'film_simple', 'film_complex') -> entry dict
    """
    config = _load_yaml(yaml_file)
    best: dict[str, dict] = {}
    for entry in config.get("trials", []):
        results = entry.get("results", {})
        if not results.get("completed", False):
            continue
        key = _type_key_from_entry(entry)
        val_loss = float(results["best_val_loss"])
        if key not in best or val_loss < float(best[key]["results"]["best_val_loss"]):
            best[key] = entry
    return best


def select_bayes_champions(
    yaml_file: str,
    device: torch.device,
    metadata_root: str = "data/metadata",
) -> dict[str, torch.nn.Module]:
    """
    Load and return the best-performing model per type in eval mode.

    Rebuilds architecture from YAML entry, loads .pth weights from model_path.
    Mirrors grid_search.select_champions() exactly.

    Returns
    -------
    dict mapping type key -> model (eval mode, on device)
    """
    config = _load_yaml(yaml_file)
    base_channels = int(config.get("params", {}).get("base_channels", 32))

    best = get_bayes_champion_entries(yaml_file)
    if not best:
        print("[select_bayes_champions] No completed trials found.")
        return {}

    _combined_loader: CombinedMetadataUtils | None = None
    champions: dict[str, torch.nn.Module] = {}

    for key, entry in best.items():
        model_path = entry["results"]["model_path"]
        if entry["model_type"] == "film":
            if _combined_loader is None:
                _combined_loader = CombinedMetadataUtils(
                    csv_path=os.path.join(metadata_root, "metadata_combined.csv")
                )
            model = _build_model(entry, base_channels=base_channels, cond_dim=_combined_loader.n_features)
        else:
            model = _build_model(entry, base_channels=base_channels)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        champions[key] = model
        print(
            f"[select_bayes_champions] {key}: {entry['id']}"
            f"  val_loss={entry['results']['best_val_loss']:.6f}  ← {model_path}"
        )

    return champions
