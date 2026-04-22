import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.combined_metadata_loader import CombinedMetadataLoader, SubsetCombinedMetadataLoader
from utils.data_converter import DataConverter
from utils.loss_functions import ssim_l1_loss
from utils.train_eval import fit_feature_based_3D, get_metadata_features

# Shared constants (mirror grid_search.py)
_CROP_AXES    = ((16, 10, 0), (17, 11, 17))
_BASE_CHANNELS = 32


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — gradient-based feature importance
# ─────────────────────────────────────────────────────────────────────────────

def gradient_importance(
    model: torch.nn.Module,
    pairs: list[tuple[str, str]],
    cond_loader: CombinedMetadataLoader,
    crop_axes=_CROP_AXES,
) -> np.ndarray:
    """
    Compute per-feature importance as mean |d(loss)/d(cond)| over all pairs.

    Returns
    -------
    np.ndarray of shape (n_features,) with non-negative importance scores.
    """
    dc = DataConverter()
    data_device = next(model.parameters()).device
    n_features  = cond_loader.n_features
    accum       = np.zeros(n_features, dtype=np.float64)
    count       = 0

    model.eval()

    for x_path, y_path in tqdm(pairs, desc="Gradient importance", unit="vol"):
        x_full = dc.load_path_as_tensor(x_path, data_device)
        y_full = dc.load_path_as_tensor(y_path, data_device)
        x = dc.get_volume_with_3d_change(x_full, crop_axes, remove_mode=True)
        y = dc.get_volume_with_3d_change(y_full, crop_axes, remove_mode=True)

        cond_np = cond_loader.get(x_path)
        if cond_np is None:
            cond_np = np.zeros(n_features, dtype=np.float32)
        cond = torch.tensor(cond_np, dtype=torch.float32, device=data_device).unsqueeze(0)
        cond.requires_grad_(True)

        out   = model(x, cond)
        y_hat = out[0] if isinstance(out, (tuple, list)) else out

        loss_val = ssim_l1_loss(y_hat, y)
        loss     = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val
        loss.backward()

        if cond.grad is not None:
            accum += cond.grad.detach().abs().cpu().numpy().squeeze()
            count += 1

        cond.requires_grad_(False)
        del x_full, y_full, x, y, cond, out, y_hat, loss
        torch.cuda.empty_cache()

    if count == 0:
        return accum
    return accum / count

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — train a FiLM model with a given feature subset
# ─────────────────────────────────────────────────────────────────────────────

def train_with_features(
    feature_indices: list[int],
    champion_entry: dict,
    base_loader: CombinedMetadataLoader,
    device: torch.device,
    training_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
    epochs: int,
    out_dir: str,
    run_label: str,
    checkpoint_every: int = 250,
    crop_axes=_CROP_AXES,
    base_channels: int = _BASE_CHANNELS,
    hide_progress: bool = False,
) -> tuple[float, str]:
    """
    Build and train a FiLM model conditioned on `feature_indices` only.
    Hyperparameters are taken from `champion_entry`.

    Returns
    -------
    (best_val_loss, checkpoint_path)
    """
    import importlib
    from utils.grid_search import MODEL_REGISTRY

    subset_loader = SubsetCombinedMetadataLoader(base_loader, feature_indices)
    cond_dim      = subset_loader.n_features

    # Build model
    model_type = champion_entry["model_type"]
    module_name, class_name = MODEL_REGISTRY[model_type]
    cls = getattr(importlib.import_module(module_name), class_name)
    use_simple = champion_entry.get("film_generator_type") == "simple"
    mlp_hidden = int(champion_entry["mlp_hidden"])
    model = cls(base=base_channels, cond_dim=cond_dim, use_simple=use_simple, mlp_hidden=mlp_hidden)

    lr    = float(champion_entry["learning_rate"])
    wd    = float(champion_entry["weight_decay"])
    betas = (float(champion_entry["betas"][0]), float(champion_entry["betas"][1]))
    sched_type = champion_entry["lr_scheduler"]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {sched_type!r}")

    _, _, _, best_model, best_val_loss = fit_feature_based_3D(
        model=model,
        device=next(model.parameters()).device,
        training_pairs=training_pairs,
        validation_pairs=validation_pairs,
        epochs=epochs,
        loss_func=ssim_l1_loss,
        metadata_loader=subset_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        crop_axes=crop_axes,
        checkpoint_every=checkpoint_every,
        hide_progress=hide_progress,
    )

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{run_label}_best.pth")
    torch.save(best_model.state_dict(), ckpt_path)
    del model, best_model
    torch.cuda.empty_cache()
    return float(best_val_loss), ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Sequential Forward Selection
# ─────────────────────────────────────────────────────────────────────────────

def run_sfs(
    candidate_indices: list[int],
    train_fn,
    results_path: str,
    max_features: int = 5,
    min_improvement: float = 1e-4,
) -> list[int]:
    """
    Sequential Forward Selection over `candidate_indices`.

    Parameters
    ----------
    candidate_indices : list[int]
        Feature indices to search over (e.g. top-K from gradient importance).
    train_fn : callable(feature_indices: list[int], run_label: str) -> float
        Trains a model on the given feature subset and returns best_val_loss.
        The caller is responsible for caching / saving checkpoints.
    results_path : str
        JSON file for checkpoint state. Created on first run; loaded to resume.
    max_features : int
        Maximum features to select.
    min_improvement : float
        Stop early if the best step gain is less than this.

    Returns
    -------
    list[int] — selected feature indices in order of selection.
    """
    state = _load_sfs_state(results_path)
    selected: list[int] = list(state.get("selected", []))
    steps: list[dict]   = state.get("steps", [])

    remaining = [i for i in candidate_indices if i not in selected]
    prev_loss = steps[-1]["val_loss"] if steps else float("inf")

    print(f"[SFS] Resuming — already selected {len(selected)} feature(s): {selected}")

    for step_num in range(len(selected) + 1, max_features + 1):
        if not remaining:
            print("[SFS] No candidates left.")
            break

        # Find or load current step's tried candidates
        step_record = next((s for s in steps if s["step"] == step_num), None)
        tried: dict[str, float] = step_record["candidates_tried"] if step_record else {}

        print(f"\n[SFS] Step {step_num}: trying {len(remaining)} candidate(s)  "
              f"(prev_loss={prev_loss:.6f})")

        cand_bar = tqdm(list(remaining), desc=f"SFS step {step_num}", unit="candidate")
        for feat_idx in cand_bar:
            key = str(feat_idx)
            if key in tried:
                cand_bar.write(f"  [skip] feature {feat_idx} already evaluated  loss={tried[key]:.6f}")
                continue

            trial = selected + [feat_idx]
            label = f"sfs_step{step_num}_feat{'_'.join(str(i) for i in sorted(trial))}"
            cand_bar.set_postfix_str(f"feat={feat_idx}")
            cand_bar.write(f"  Training with features {trial}  (label={label})")
            val_loss = train_fn(trial, label)
            tried[key] = val_loss
            cand_bar.write(f"  feature {feat_idx} → val_loss={val_loss:.6f}")

            # Persist after each candidate so nothing is lost on crash
            _save_sfs_state(results_path, selected, steps, step_num, tried)

        best_feat = min(tried, key=tried.__getitem__)
        best_loss = tried[best_feat]
        improvement = prev_loss - best_loss

        step_entry = {
            "step":             step_num,
            "selected_after":   selected + [int(best_feat)],
            "best_feature":     int(best_feat),
            "val_loss":         best_loss,
            "improvement":      improvement,
            "candidates_tried": tried,
            "completed_at":     datetime.now().isoformat(timespec="seconds"),
        }
        # Replace any partial record for this step
        steps = [s for s in steps if s["step"] != step_num] + [step_entry]
        selected.append(int(best_feat))
        remaining.remove(int(best_feat))
        prev_loss = best_loss

        _save_sfs_state(results_path, selected, steps, step_num, tried)
        print(f"[SFS] Step {step_num} done — added feature {best_feat}  "
              f"val_loss={best_loss:.6f}  improvement={improvement:.6f}")

        if improvement < min_improvement:
            print(f"[SFS] Early stop: improvement {improvement:.6f} < {min_improvement}")
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sfs_state(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"selected": [], "steps": []}


def _save_sfs_state(path: str, selected, steps, current_step, tried):
    state = {
        "selected": selected,
        "steps":    steps,
        "in_progress_step": {
            "step":             current_step,
            "candidates_tried": tried,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
