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

# TODO Evt finne en annen løsning her
def mid_axial_slice_5d(t5):
    """
    t5: (B, C, D, H, W) torch tensor
    -> numpy slice (H, W) in [0,1] from the middle of D
    """
    if isinstance(t5, torch.Tensor):
        t = t5.detach().cpu()
    else:
        t = torch.tensor(t5)
    _, _, D, H, W = t.shape
    mid = D // 2
    sl = t[0, 0, mid]  # (H, W)
    sl = sl.clamp(0, 1).numpy()
    return sl

# ---------------------------------------------
# 3D Training Loop
# ---------------------------------------------

def fit_3D(
    model,
    device: torch.device,
    training_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
    epochs=1,
    loss_func=None,
    dataConverter: DataConverter = DataConverter(),
    optimizer=None,
    print_every: int = None,
    save_every: int = None,
    crop_axes: list[list, list] = None,
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

    for i in tqdm(range(epochs), desc="Training 3D Residual U-Net"):
        # pick a random pair volume
        patient_id = random.randint(0, len(training_pairs) - 1)
        x_path, y_path = training_pairs[patient_id][0], training_pairs[patient_id][1]

        # Load full volumes as tensors
        # TODO Legg til cropping med crop_axes
        x = dataConverter.load_path_as_tensor(x_path)
        y = dataConverter.load_path_as_tensor(y_path)

        # Make sure they have the same depth
        if (len(x) != len(y)):
            print(f"Warning: unequal depth in training pair for patient {patient_id} ({len(x)} != {len(y)}). Skipped pair...")
            continue

        # forward (support both y_hat or (y_hat, delta))
        out = model(x)

        # Unpack output
        # TODO Kanskje fjerne en av disse?
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            y_hat, _ = out[0], out[1]
        else:
            y_hat, _ = out, None    # Uhhhhhhhhhhhh?? TODO selvmord

        # --- compute loss ---
        crit_out = loss_func(y_hat, y)
        # TODO Sjekk hvem av disse som er tilfelle ved ssim
        if isinstance(crit_out, (tuple, list)):
            loss = crit_out[0]
        else:
            loss = crit_out
        loss_val = cap_logged_loss(loss)
    
        # Update model
        loss.backward()
        optimizer.step()

        # --- log ---
        loss_history.append(loss_val)

        if print_every and (i % print_every == 0):
            print(f"[Iter {i}] total: {loss_val:.6f}")

        # --- snapshot ---
        if save_every and (i % save_every == 0 or i == epochs - 1):
            with torch.no_grad():
                x_np = mid_axial_slice_5d(x)
                y_np = mid_axial_slice_5d(y)
                recon_np = mid_axial_slice_5d(y_hat)

            saved_snapshots.append({"iter": i, "x": x_np, "y": y_np, "recon": recon_np, "loss": loss_val})
            print(f"Saved snapshot at iter {i} (mid-axial slice)")

            # Check if we got new best
            if len(validation_pairs) > 0:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for vx_path, vy_path in validation_pairs:
                        val_x = dataConverter.load_path_as_tensor(vx_path)
                        val_y = dataConverter.load_path_as_tensor(vy_path)

                        vout = model(val_x)
                        if isinstance(vout, (tuple, list)) and len(vout) >= 2:
                            vy_hat, _ = vout[0], vout[1]
                        else:
                            vy_hat, _ = vout, None

                        # validation loss uses same loss_func fallback logic
                        vloss = None
                        if loss_func is not None:
                            try:
                                vcrit_out = loss_func(vy_hat, val_y)
                                if isinstance(vcrit_out, (tuple, list)):
                                    vloss = vcrit_out[0]
                                else:
                                    vloss = vcrit_out
                            except TypeError:
                                pass

                        if vloss is None:
                            vloss = F.l1_loss(vy_hat, val_y)

                        val_losses.append(float(vloss.item()))

                avg_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else np.inf

                # check for new best
                if avg_loss < best_val_loss:
                    print(
                        f"Found new best loss on validation set: "
                        f"{avg_loss:.6f} (prev {best_val_loss:.6f})"
                    )
                    best_val_loss = avg_loss
                    best_model = copy.deepcopy(model)
        
        # Finally reset optimizer for next iter
        optimizer.zero_grad()
    
    return model, loss_history, saved_snapshots, best_model