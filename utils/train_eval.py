import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.data_converter import DataConverter
from tqdm import tqdm
import copy

# ---------------------------------------------
# General Helper Functions
# ---------------------------------------------

def build_optimizer(model, lr=1e-4, wd=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def cap_logged_loss(loss: torch.Tensor, loss_cap: float = 1e6) -> float:
    """
    Cap a tensor loss.
    Returns a float value suitable for logging.
    """
    capped = torch.clamp(loss, max=loss_cap)
    return float(capped.item())

def get_middle_slice_3D(volume: torch.Tensor | np.ndarray):
    """
    volume: (1, 1, D, H, W)
    -> numpy slice (D, W) in [0,1] from the middle of W
    """
    if isinstance(volume, torch.Tensor):
        t = volume.detach().cpu()
    else:
        t = torch.tensor(volume)
    _, _, D, H, W = t.shape
    mid = W // 2
    sl = t[0, 0, :, :, mid]
    return sl.clamp(0, 1).numpy()


# ---------------------------------------------
# 3D Training Loop
# ---------------------------------------------

def fit_3D(
    model,
    device: torch.device,
    training_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
    epochs=1000,
    loss_func=None,
    dataConverter: DataConverter = DataConverter(),
    optimizer=None,
    save_every: int = None,
    checkpoint_every: int = 100,
    crop_axes: list[tuple, tuple] = None,
):
    """
    Train a 3D model on full volumes using HuntDataLoader.
    - Expects model(x) -> either:
        * y_hat
        * (y_hat, delta)   where delta is a residual volume previously used for TV regularization
    - loss_func is optional:
        * If provided and accepts (y_hat, y), we use it.
        * Otherwise we fall back to L1.
    - Snapshots show the mid-axial slice (H,W) for input, target, and recon.
    """
    
    # Set up values
    optimizer = optimizer or build_optimizer(model)
    saved_snapshots, loss_history = [], []
    best_val_loss = np.inf
    best_model = copy.deepcopy(model)
    model.train()

    prog_bar = tqdm(range(epochs), desc="Training 3D Residual U-Net")

    for i in prog_bar:
        # pick a random pair volume
        patient_id = random.randint(0, len(training_pairs) - 1)
        x_path, y_path = training_pairs[patient_id][0], training_pairs[patient_id][1]

        # Load full volumes as tensors
        x_full = dataConverter.load_path_as_tensor(x_path, device)
        y_full = dataConverter.load_path_as_tensor(y_path, device)

        # Crop if specified
        if crop_axes is not None:
            x = dataConverter.get_volume_with_3d_change(tensor=x_full, crop_axes=crop_axes, remove_mode=True)
            y = dataConverter.get_volume_with_3d_change(tensor=y_full, crop_axes=crop_axes, remove_mode=True)

        # Make sure they have the same depth
        if (x.shape[2] != y.shape[2]) or (x.shape[3] != y.shape[3]) or (x.shape[4] != y.shape[4]):
            print(f"Warning: unequal dimentions in training pair for patient {patient_id} (D: {x.shape[2]} vs {y.shape[2]}, H: {x.shape[3]} vs {y.shape[3]}, W: {x.shape[4]} vs {y.shape[4]}). Skipped pair...")
            continue

        # forward (support both y_hat or (y_hat, delta))
        out = model(x)

        # Res unet returns delta as well, we only need the recon
        y_hat = out[0] if isinstance(out, (tuple, list)) else out

        # --- compute loss ---
        crit_out = loss_func(y_hat, y)
        
        # TODO Sjekk hvem av disse som er tilfelle ved ssim
        loss = crit_out[0] if isinstance(crit_out, (tuple, list)) else crit_out
        capped_loss = cap_logged_loss(loss)
    
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- log ---
        loss_history.append(capped_loss)

        # --- Save Snapshot ---
        if save_every and (i % save_every == 0 or i == 0 or i == epochs - 1):
            # Add padding for the exported
            if crop_axes is not None:
                y_hat_padded = dataConverter.get_volume_with_3d_change(tensor=y_hat, crop_axes=crop_axes, remove_mode=False)
            
            with torch.no_grad():
                x_np = get_middle_slice_3D(x_full)
                y_np = get_middle_slice_3D(y_full)
                recon_np = get_middle_slice_3D(y_hat_padded)

            saved_snapshots.append({"iter": i, "x": x_np, "y": y_np, "recon": recon_np, "loss": capped_loss})

        # --- Validation ---
        if (i % checkpoint_every == 0 or i == 0 or i == epochs - 1):
            # Check if we got new best
            if len(validation_pairs) > 0:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for vx_path, vy_path in validation_pairs:
                        val_x = dataConverter.load_path_as_tensor(vx_path, device)
                        val_y = dataConverter.load_path_as_tensor(vy_path, device)

                        if crop_axes is not None:
                            val_x = dataConverter.get_volume_with_3d_change(tensor=val_x, crop_axes=crop_axes, remove_mode=True)
                            val_y = dataConverter.get_volume_with_3d_change(tensor=val_y, crop_axes=crop_axes, remove_mode=True)

                        vout = model(val_x)
                        if isinstance(vout, (tuple, list)) and len(vout) >= 2:
                            vy_hat, _ = vout[0], vout[1]
                        else:
                            vy_hat, _ = vout, None

                        vloss = None
                        if loss_func is not None:
                            try:
                                vcrit_out = loss_func(vy_hat, val_y)
                                vloss = vcrit_out[0] if isinstance(vcrit_out, (tuple, list)) else vcrit_out
                            except TypeError:
                                pass

                        if vloss is None:
                            vloss = F.l1_loss(vy_hat, val_y)

                        val_losses.append(float(vloss.item()))

                avg_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else np.inf

                # check for new best
                if avg_loss < best_val_loss:
                    prog_bar.set_postfix_str(f"Best loss on val {avg_loss:.6f}, (Iter {i})")
                    best_val_loss = avg_loss
                    best_model = copy.deepcopy(model)

                # Restore training mode after validation
                model.train()
                best_model = copy.deepcopy(model)

    return model, loss_history, saved_snapshots, best_model