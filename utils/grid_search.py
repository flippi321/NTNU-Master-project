import importlib
import itertools
import os
from datetime import datetime

import torch
import torch.optim as optim
import yaml

from utils.combined_metadata_loader import CombinedMetadataLoader
from utils.loss_functions import ssim_l1_loss
from utils.train_eval import fit_3D, fit_feature_based_3D

MODEL_REGISTRY = {
    "std":  ("models.unet_3d_std",  "UNet3D"),
    "film": ("models.unet_3d_film", "FiLMUNet3D"),
}

CROP_AXES = ((16, 10, 0), (17, 11, 17))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _matches_filter(entry: dict, model_filter: str | None) -> bool:
    if model_filter is None:
        return True
    if model_filter == "std":
        return entry["model_type"] == "std"
    if model_filter == "film":
        return entry["model_type"] == "film"
    if model_filter == "film_simple":
        return entry["model_type"] == "film" and entry.get("film_generator_type") == "simple"
    if model_filter == "film_complex":
        return entry["model_type"] == "film" and entry.get("film_generator_type") == "complex"
    raise ValueError(f"Unknown model_filter: {model_filter!r}. "
                     "Use None, 'std', 'film', 'film_simple', or 'film_complex'.")


def _build_model(entry: dict, base_channels: int, cond_dim: int = 0) -> torch.nn.Module:
    model_type = entry["model_type"]
    module_name, class_name = MODEL_REGISTRY[model_type]
    cls = getattr(importlib.import_module(module_name), class_name)
    if model_type == "film":
        use_simple = entry.get("film_generator_type") == "simple"
        mlp_hidden = int(entry["mlp_hidden"])
        return cls(base=base_channels, cond_dim=cond_dim, use_simple=use_simple, mlp_hidden=mlp_hidden)
    return cls(base=base_channels)


def _build_scheduler(optimizer, scheduler_type: str, epochs: int):
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    raise ValueError(f"Unknown lr_scheduler: {scheduler_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_hyperparameter_grid(
    grid_file: str,
    epochs: int = 4000,
    base_channels: int = 32,
    lrs: list = [5e-5, 1e-4],
    weight_decays: list = [1e-5, 1e-4],
    betas: list = [[0.9, 0.999], [0.9, 0.95]],
    schedulers: list = ["cosine", "plateau"],
    mlp_hiddens: list = [64, 128],
) -> None:
    """Create grid_search.yaml. Does nothing if the file already exists."""
    if os.path.exists(grid_file):
        print(f"[create_hyperparameter_grid] {grid_file} already exists — skipping.")
        return

    params = dict(epochs=epochs, base_channels=base_channels, lrs=lrs,
                  weight_decays=weight_decays, betas=betas,
                  schedulers=schedulers, mlp_hiddens=mlp_hiddens)
    entries = []

    n = 1
    for lr, wd, b, sched in itertools.product(lrs, weight_decays, betas, schedulers):
        entries.append({"id": f"std_{n:03d}", "model_type": "std",
                        "learning_rate": lr, "weight_decay": wd,
                        "betas": list(b), "lr_scheduler": sched})
        n += 1

    n = 1
    for lr, wd, b, sched, mlp in itertools.product(lrs, weight_decays, betas, schedulers, mlp_hiddens):
        entries.append({"id": f"film_simple_{n:03d}", "model_type": "film",
                        "film_generator_type": "simple", "mlp_hidden": mlp,
                        "learning_rate": lr, "weight_decay": wd,
                        "betas": list(b), "lr_scheduler": sched})
        n += 1

    n = 1
    for lr, wd, b, sched, mlp in itertools.product(lrs, weight_decays, betas, schedulers, mlp_hiddens):
        entries.append({"id": f"film_complex_{n:03d}", "model_type": "film",
                        "film_generator_type": "complex", "mlp_hidden": mlp,
                        "learning_rate": lr, "weight_decay": wd,
                        "betas": list(b), "lr_scheduler": sched})
        n += 1

    os.makedirs(os.path.dirname(grid_file) or ".", exist_ok=True)
    with open(grid_file, "w") as f:
        yaml.dump({"params": params, "grid": entries}, f,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[create_hyperparameter_grid] Created {grid_file} with {len(entries)} entries.")


def run_pipeline(
    grid_file: str,
    device: torch.device,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    checkpoint_every: int = 250,
    metadata_root: str = "data/metadata",
    out_dir: str = "out/grid_search",
    model_filter: str | None = None,
    stop_after_first: bool = False,
) -> int:
    """
    Scan the YAML grid file for untrained entries and train them sequentially.

    Parameters
    ----------
    grid_file : str
        Path to grid_search.yaml.
    device : torch.device
        Torch device to train on.
    train_pairs, val_pairs : list of (x_path, y_path) tuples.
    model_filter : str | None
        Restrict to a model type: None (all), 'std', 'film', 'film_simple', 'film_complex'.
    stop_after_first : bool
        Train one entry then return immediately.

    Returns
    -------
    int  Number of models trained in this call.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(grid_file) as f:
        config = yaml.safe_load(f)

    yaml_params   = config.get("params", {})
    epochs        = int(yaml_params["epochs"])
    base_channels = int(yaml_params["base_channels"])
    grid          = config["grid"]

    _combined_loader: CombinedMetadataLoader | None = None
    trained_count = 0

    for entry in grid:
        if not _matches_filter(entry, model_filter):
            continue

        results = entry.setdefault("results", {})
        if results.get("completed", False):
            print(f"[pipeline] Skipping {entry['id']} — already completed")
            continue

        run_id   = entry["id"]
        lr       = float(entry["learning_rate"])
        wd       = float(entry["weight_decay"])
        betas    = (float(entry["betas"][0]), float(entry["betas"][1]))
        sched_type = entry["lr_scheduler"]

        print(f"\n[pipeline] Starting {run_id}")
        print(f"  type={entry['model_type']}  film_type={entry.get('film_generator_type')}  "
              f"lr={lr}  wd={wd}  betas={betas}  sched={sched_type}  "
              f"mlp={entry.get('mlp_hidden')}  epochs={epochs}  base={base_channels}")

        # Baseline non-feature aware Model
        if entry["model_type"] == "std":
            model = _build_model(entry, base_channels=base_channels)
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
            )

        # Feature-aware Model with CombinedMetadataLoader
        else: 
            if _combined_loader is None:
                _combined_loader = CombinedMetadataLoader(
                    csv_path=os.path.join(metadata_root, "metadata_combined.csv")
                )
            model = _build_model(entry, base_channels=base_channels, cond_dim=_combined_loader.n_features)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
            scheduler = _build_scheduler(optimizer, sched_type, epochs)
            _, loss_history, _, best_model, best_val_loss = fit_feature_based_3D(
                model=model,
                device=device,
                training_pairs=train_pairs,
                validation_pairs=val_pairs,
                epochs=epochs,
                loss_func=ssim_l1_loss,
                metadata_loader=_combined_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                crop_axes=CROP_AXES,
                checkpoint_every=checkpoint_every,
            )

        model_path = os.path.join(out_dir, f"{run_id}_best.pth")
        torch.save(best_model.state_dict(), model_path)

        results.update(dict(
            best_val_loss=float(best_val_loss),
            final_train_loss=float(loss_history[-1]) if loss_history else None,
            completed=True,
            trained_at=datetime.now().isoformat(timespec='seconds'),
            model_path=model_path,
        ))

        with open(grid_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"[pipeline] Done {run_id}  best_val_loss={best_val_loss:.6f}  → {model_path}")
        trained_count += 1

        if stop_after_first:
            break

    if trained_count == 0:
        print("[pipeline] Nothing to train — all matching entries are complete.")
    else:
        print(f"\n[pipeline] Finished. Trained {trained_count} model(s).")

    return trained_count
