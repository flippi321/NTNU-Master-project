import torch
import random
import numpy as np
import torch.optim as optim
from utils.mri.data_converter import DataConverter
from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
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
    _, _, _, _, W = t.shape
    mid = W // 2
    sl = t[0, 0, :, :, mid]
    return sl.clamp(0, 1).numpy()

# Track which patient paths we've already warned about for non-finite metadata,
# so a 485-patient training run doesn't print the same message hundreds of times.
_NAN_COND_WARNED: set[str] = set()


def get_metadata_features(loader: CombinedMetadataUtils, path: str) -> torch.Tensor:
    features = loader.get(path)
    if features is None:
        # Fail loudly. Silently substituting zeros lets a coverage gap quietly
        # turn off conditioning for a subset of patients — run_bayes runs a
        # _assert_metadata_coverage pre-flight so reaching this branch is a bug.
        raise KeyError(
            f"No metadata for {path}; pre-flight coverage check should have caught this."
        )
    else:
        # Some patients have NaN/inf entries in the combined metadata CSV (e.g.
        # missing FastSurfer columns). Letting them through corrupts the model
        # the first time that patient is sampled — MetadataTokenizer.mlp(cond)
        # produces NaN tokens, NaN propagates into K/V, and the params turn
        # NaN permanently. Sanitize at the boundary; warn once per patient.
        features = np.asarray(features, dtype=np.float32)
        bad_mask = ~np.isfinite(features)
        if bad_mask.any():
            if path not in _NAN_COND_WARNED:
                _NAN_COND_WARNED.add(path)
                bad_idx = np.where(bad_mask)[0].tolist()
                print(
                    f"Warning: non-finite metadata features for {path} "
                    f"at indices {bad_idx} — replacing with 0."
                )
            features = np.where(bad_mask, 0.0, features).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32)

# ---------------------------------------------
# 3D Training Loop
# ---------------------------------------------

def fit_3D(
    model,
    device: torch.device,
    training_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
    epochs=2000,
    loss_func=None,
    dataConverter: DataConverter = DataConverter(),
    optimizer=None,
    scheduler=None,
    snapshot_every: int = None,
    checkpoint_every: int = 100,
    crop_axes: list[tuple, tuple] = None,
    restart_threshold: float | None = None,
    restart_check_epoch: int = 250,
    max_total_runs: int = 3,
):
    """
    Train a 3D model on full volumes using HuntDataLoader.
    - Expects model(x) -> either:
        * y_hat
        * (y_hat, delta)   where delta is a residual volume previously used for TV regularization
    - Snapshots show the mid-axial slice (H,W) for input, target, and recon.
    - If restart_threshold is set, checks val_loss at restart_check_epoch and restarts from
      fresh weights if the threshold is not met, up to max_total_runs total attempts.
    """

    # Models that self-manage placement expose either `device` (single-GPU
    # FiLM/Meta) or `dev0` (multi-GPU std). Honor those before falling back.
    data_device = getattr(model, 'device', getattr(model, 'dev0', device))

    optimizer = optimizer or build_optimizer(model)

    saved_snapshots, loss_history = [], []
    best_val_loss = np.inf
    best_model = copy.deepcopy(model)
    total_runs = max_total_runs if restart_threshold is not None else 1

    for run in range(total_runs):
        if run > 0:
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer.state.clear()
            best_val_loss = np.inf
            best_model = copy.deepcopy(model)
            print(f"[restart] Starting run {run + 1}/{total_runs} with fresh random weights.")

        model.train()
        restart_triggered = False
        restart_checked = False
        desc = "Training 3D U-Net" if total_runs == 1 else f"Training 3D U-Net [run {run + 1}/{total_runs}]"
        prog_bar = tqdm(range(epochs), desc=desc)

        for i in prog_bar:
            # pick a random pair volume
            patient_id = random.randint(0, len(training_pairs) - 1)
            x_path, y_path = training_pairs[patient_id][0], training_pairs[patient_id][1]

            # Load full volumes as tensors
            x_full = dataConverter.load_path_as_tensor(x_path, data_device)
            y_full = dataConverter.load_path_as_tensor(y_path, data_device)

            # Crop if specified
            if crop_axes is not None:
                x = dataConverter.get_volume_with_3d_change(tensor=x_full, crop_axes=crop_axes, remove_mode=True)
                y = dataConverter.get_volume_with_3d_change(tensor=y_full, crop_axes=crop_axes, remove_mode=True)
            else:
                x, y = x_full, y_full

            # Make sure they have the same depth
            if (x.shape[2] != y.shape[2]) or (x.shape[3] != y.shape[3]) or (x.shape[4] != y.shape[4]):
                print(f"Warning: unequal dimentions in training pair for patient {patient_id} (D: {x.shape[2]} vs {y.shape[2]}, H: {x.shape[3]} vs {y.shape[3]}, W: {x.shape[4]} vs {y.shape[4]}). Skipped pair...")
                continue

            out = model(x)

            # Res unet returns delta as well, we only need the recon
            y_hat = out[0] if isinstance(out, (tuple, list)) else out

            # Collapse detection: if output collapses to all-black, trigger restart when possible
            if i % checkpoint_every == 0 and float(y_hat.abs().mean()) < 1e-4:
                print(f"[collapse] Epoch {i}: mean output = {float(y_hat.abs().mean()):.2e} — model collapsed to black.")
                if restart_threshold is not None and run < total_runs - 1:
                    restart_triggered = True
                    break

            # --- compute loss ---
            crit_out = loss_func(y_hat, y)

            # TODO Sjekk hvem av disse som er tilfelle ved ssim
            loss = crit_out[0] if isinstance(crit_out, (tuple, list)) else crit_out
            capped_loss = cap_logged_loss(loss)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- log ---
            loss_history.append(capped_loss)

            # --- Save Snapshot ---
            if snapshot_every and i > 0 and (i % snapshot_every == 0 or i == epochs - 1):

                # Add padding for the exported
                if crop_axes is not None:
                    # Move y_hat back to cpu/device for padding op
                    y_hat_padded = dataConverter.get_volume_with_3d_change(
                        tensor=y_hat.to(device), crop_axes=crop_axes, remove_mode=False
                    )
                else:
                    y_hat_padded = y_hat

                with torch.no_grad():
                    x_np = get_middle_slice_3D(x_full)
                    y_np = get_middle_slice_3D(y_full)
                    recon_np = get_middle_slice_3D(y_hat_padded)

                saved_snapshots.append({"iter": i, "x": x_np, "y": y_np, "recon": recon_np, "loss": capped_loss})

            # --- Validation ---
            # Guard against i == 0: validating before any meaningful training would
            # capture the near-untrained baseline as best_val_loss and any later
            # checkpoint would have to beat that to update.
            if i > 0 and (i % checkpoint_every == 0 or i == epochs - 1):
                # Check if we got new best
                if len(validation_pairs) > 0:
                    model.eval()
                    val_losses = []

                    with torch.no_grad():
                        for vx_path, vy_path in validation_pairs:
                            val_x = dataConverter.load_path_as_tensor(vx_path, data_device)
                            val_y = dataConverter.load_path_as_tensor(vy_path, data_device)

                            if crop_axes is not None:
                                val_x = dataConverter.get_volume_with_3d_change(tensor=val_x, crop_axes=crop_axes, remove_mode=True)
                                val_y = dataConverter.get_volume_with_3d_change(tensor=val_y, crop_axes=crop_axes, remove_mode=True)

                            vout = model(val_x)
                            vy_hat = vout[0] if isinstance(vout, (tuple, list)) else vout

                            vloss = None
                            try:
                                vcrit_out = loss_func(vy_hat, val_y)
                                vloss = vcrit_out[0] if isinstance(vcrit_out, (tuple, list)) else vcrit_out
                            except TypeError:
                                pass

                            val_losses.append(float(vloss.item()))

                    avg_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else np.inf

                    # check for new best
                    if avg_loss < best_val_loss:
                        prog_bar.set_postfix_str(f"Best loss on val {avg_loss:.6f}, (Iter {i})")
                        best_val_loss = avg_loss
                        best_model = copy.deepcopy(model)

                    # Restore training mode after validation
                    model.train()

                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(avg_loss)
                        else:
                            scheduler.step()

                    # --- Restart check (fires once, at the first validation at or after restart_check_epoch) ---
                    if restart_threshold is not None and not restart_checked and i >= restart_check_epoch:
                        restart_checked = True
                        if best_val_loss > restart_threshold:
                            if run < total_runs - 1:
                                print(
                                    f"[restart] Run {run + 1}/{total_runs}: "
                                    f"val_loss={best_val_loss:.6f} > threshold={restart_threshold} "
                                    f"at epoch {i + 1}. Restarting with fresh weights."
                                )
                                restart_triggered = True
                                break
                            else:
                                print(
                                    f"[restart] Run {run + 1}/{total_runs}: "
                                    f"val_loss={best_val_loss:.6f} > threshold={restart_threshold} "
                                    f"at epoch {i + 1}. Max runs reached — continuing to completion."
                                )

        if not restart_triggered:
            break

    return model, loss_history, saved_snapshots, best_model, best_val_loss




# ---------------------------------------------
# Feature based 3D Training Loop
# ---------------------------------------------

def fit_feature_based_3D(
    model,
    device: torch.device,
    training_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
    epochs=2000,
    loss_func=None,
    dataConverter: DataConverter = DataConverter(),
    metadata_loader: CombinedMetadataUtils = CombinedMetadataUtils(),
    optimizer=None,
    scheduler=None,
    snapshot_every: int = None,
    checkpoint_every: int = 100,
    crop_axes: list[tuple, tuple] = None,
    hide_progress: bool = False,
    restart_threshold: float | None = None,
    restart_check_epoch: int = 250,
    max_total_runs: int = 3,
    debug_nan: bool = False,
    use_autocast: bool = False,
):
    """
    Train a 3D model on full volumes using HuntDataLoader.
    - Expects model(x) -> either:
        * y_hat
        * (y_hat, delta)   where delta is a residual volume previously used for TV regularization
    - Snapshots show the mid-axial slice (H,W) for input, target, and recon.
    - If restart_threshold is set, checks val_loss at restart_check_epoch and restarts from
      fresh weights if the threshold is not met, up to max_total_runs total attempts.
    """

    # Models that self-manage placement expose either `device` (single-GPU
    # FiLM/Meta) or `dev0` (multi-GPU std). Honor those before falling back.
    data_device = getattr(model, 'device', getattr(model, 'dev0', device))

    optimizer = optimizer or build_optimizer(model)
    device_gpu_available = device.type == "cuda" or device.type == "cuda:0" or device.type == "cuda:1"

    saved_snapshots, loss_history = [], []
    best_val_loss = np.inf
    best_model = copy.deepcopy(model)
    total_runs = max_total_runs if restart_threshold is not None else 1

    for run in range(total_runs):
        if run > 0:
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer.state.clear()
            best_val_loss = np.inf
            best_model = copy.deepcopy(model)
            print(f"[restart] Starting run {run + 1}/{total_runs} with fresh random weights.")

        model.train()
        amp_enabled = device_gpu_available and use_autocast
        scaler = torch.GradScaler(enabled=amp_enabled, device=device)
        restart_triggered = False
        restart_checked = False
        base_desc = "Training 3D U-Net with Features"
        desc = base_desc if total_runs == 1 else f"{base_desc} [run {run + 1}/{total_runs}]"
        prog_bar = tqdm(range(epochs), desc=desc, disable=hide_progress)

        for i in prog_bar:
            # pick a random pair volume
            patient_id = random.randint(0, len(training_pairs) - 1)
            x_path, y_path = training_pairs[patient_id]

            # Load full volumes as tensors
            x_full = dataConverter.load_path_as_tensor(x_path, data_device)
            y_full = dataConverter.load_path_as_tensor(y_path, data_device)

            # Crop if specified
            if crop_axes is not None:
                x = dataConverter.get_volume_with_3d_change(tensor=x_full, crop_axes=crop_axes, remove_mode=True)
                y = dataConverter.get_volume_with_3d_change(tensor=y_full, crop_axes=crop_axes, remove_mode=True)
            else:
                x, y = x_full, y_full

            # Make sure they have the same depth
            if (x.shape[2] != y.shape[2]) or (x.shape[3] != y.shape[3]) or (x.shape[4] != y.shape[4]):
                print(f"Warning: unequal dimentions in training pair for patient {patient_id} (D: {x.shape[2]} vs {y.shape[2]}, H: {x.shape[3]} vs {y.shape[3]}, W: {x.shape[4]} vs {y.shape[4]}). Skipped pair...")
                continue

            cond = get_metadata_features(metadata_loader, x_path).unsqueeze(0).to(data_device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(enabled=amp_enabled, device_type=device.type):
                out = model(x, cond)

                # Res unet returns delta as well, we only need the recon
                y_hat = out[0] if isinstance(out, (tuple, list)) else out

                # Collapse detection: if output collapses to all-black, trigger restart when possible
                if i % checkpoint_every == 0 and float(y_hat.abs().mean()) < 1e-4:
                    print(f"[collapse] Epoch {i}: mean output = {float(y_hat.abs().mean()):.2e} — model collapsed to black.")
                    if restart_threshold is not None and run < total_runs - 1:
                        restart_triggered = True
                        break

                # --- compute loss ---
                crit_out = loss_func(y_hat, y)

                # TODO Sjekk hvem av disse som er tilfelle ved ssim
                loss = crit_out[0] if isinstance(crit_out, (tuple, list)) else crit_out

            # --- NaN diagnostic: check forward output and loss ---
            # Skip the entire backward/step path if the forward already produced
            # NaN/inf — we have nothing useful to backprop and `optimizer.step()`
            # would corrupt every parameter. Note we must NOT call
            # `scaler.update()` here because no `scaler.step()` was made this iter
            # (PyTorch raises a RuntimeError in that case).
            if debug_nan:
                yhat_bad = bool(torch.isnan(y_hat).any().item() or torch.isinf(y_hat).any().item())
                loss_bad = not bool(torch.isfinite(loss).item())
                if yhat_bad or loss_bad:
                    with torch.no_grad():
                        x_min, x_max = float(x.min()), float(x.max())
                        c_min, c_max = float(cond.min()), float(cond.max())
                        c_mean, c_std = float(cond.mean()), float(cond.std())
                        if torch.isfinite(y_hat).any():
                            yh_finite = y_hat[torch.isfinite(y_hat)]
                            yh_stats = (float(yh_finite.min()), float(yh_finite.max()))
                        else:
                            yh_stats = (float("nan"), float("nan"))
                    print(
                        f"[debug_nan] iter={i} run={run+1} "
                        f"yhat_bad={yhat_bad} loss_bad={loss_bad} "
                        f"loss={float(loss) if torch.isfinite(loss).item() else 'nan/inf'} "
                        f"x=[{x_min:.3g},{x_max:.3g}] "
                        f"cond=[{c_min:.3g},{c_max:.3g}] mean={c_mean:.3g} std={c_std:.3g} "
                        f"yhat_finite_range={yh_stats} "
                        f"scaler_scale={float(scaler.get_scale())}"
                    )
                    loss_history.append(float("nan"))
                    optimizer.zero_grad(set_to_none=True)
                    del x_full, y_full, x, y, cond, out, y_hat, crit_out, loss
                    if device_gpu_available:
                        torch.cuda.synchronize()
                    continue

            capped_loss = cap_logged_loss(loss)
            loss_history.append(capped_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # --- NaN diagnostic: scan gradients before clip + step ---
            # We only LOG here. `scaler.step(optimizer)` already skips the
            # optimizer step when any grad is inf/NaN — we just want to know
            # which parameter(s) went bad first.
            if debug_nan:
                bad_grad_params = []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        bad_grad_params.append(name)
                        if len(bad_grad_params) >= 5:
                            break
                if bad_grad_params:
                    print(
                        f"[debug_nan] iter={i} run={run+1} non-finite grads in: "
                        f"{bad_grad_params} (and possibly more) "
                        f"scaler_scale={float(scaler.get_scale())}"
                    )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # --- Save Snapshot ---
            if snapshot_every and i > 0 and (i % snapshot_every == 0 or i == epochs - 1):
                # Add padding for the exported
                with torch.no_grad():
                    if crop_axes is not None:
                        y_hat_padded = dataConverter.get_volume_with_3d_change(y_hat, crop_axes, remove_mode=False)
                    else:
                        y_hat_padded = y_hat

                    saved_snapshots.append({
                        "iter": i,
                        "x": get_middle_slice_3D(x_full),
                        "y": get_middle_slice_3D(y_full),
                        "recon": get_middle_slice_3D(y_hat_padded),
                        "loss": capped_loss
                    })

            # free ASAP
            del x_full, y_full, x, y, cond, out, y_hat, crit_out, loss
            if device_gpu_available:
                torch.cuda.synchronize()

            # --- Validation ---
            # See note in fit_3D: skip i == 0 so best_val_loss isn't pinned to a
            # near-untrained baseline.
            if i > 0 and (i % checkpoint_every == 0 or i == epochs - 1) and len(validation_pairs) > 0:
                model.eval()
                val_losses = []
                # Phase 1d: count non-finite val losses so the restart line
                # ("val_loss=inf > threshold=...") has explanatory context.
                bad_val_count = 0

                with torch.no_grad():
                    for pair_idx, (vx_path, vy_path) in enumerate(validation_pairs):
                        val_x = dataConverter.load_path_as_tensor(vx_path, data_device)
                        val_y = dataConverter.load_path_as_tensor(vy_path, data_device)

                        if crop_axes is not None:
                            val_x = dataConverter.get_volume_with_3d_change(tensor=val_x, crop_axes=crop_axes, remove_mode=True)
                            val_y = dataConverter.get_volume_with_3d_change(tensor=val_y, crop_axes=crop_axes, remove_mode=True)

                        vcond = get_metadata_features(metadata_loader, vx_path).unsqueeze(0).to(data_device)

                        with torch.autocast(enabled=amp_enabled, device_type=device.type):
                            vout = model(val_x, vcond)
                            vy_hat = vout[0] if isinstance(vout, (tuple, list)) else vout
                            vcrit_out = loss_func(vy_hat, val_y)
                            vloss = vcrit_out[0] if isinstance(vcrit_out, (tuple, list)) else vcrit_out

                        vloss_val = float(vloss.item())
                        if debug_nan and not np.isfinite(vloss_val):
                            bad_val_count += 1
                            if bad_val_count <= 3:
                                if torch.isfinite(vy_hat).any():
                                    yh_finite = vy_hat[torch.isfinite(vy_hat)]
                                    yh_stats = (float(yh_finite.min()), float(yh_finite.max()))
                                else:
                                    yh_stats = (float("nan"), float("nan"))
                                print(
                                    f"[debug_nan] val iter={i} run={run+1} "
                                    f"pair_idx={pair_idx} vx={vx_path} "
                                    f"vloss={vloss_val} "
                                    f"vy_hat_finite_range={yh_stats} "
                                    f"vcond=[{float(vcond.min()):.3g},{float(vcond.max()):.3g}] "
                                    f"mean={float(vcond.mean()):.3g} std={float(vcond.std()):.3g}"
                                )
                        val_losses.append(vloss_val)
                        del val_x, val_y, vcond, vout, vy_hat, vcrit_out, vloss

                if device_gpu_available:
                    torch.cuda.empty_cache()
                avg_loss = float(np.mean(val_losses)) if val_losses else np.inf
                if debug_nan and (bad_val_count > 0 or not np.isfinite(avg_loss)):
                    print(
                        f"[debug_nan] val iter={i} run={run+1} "
                        f"non_finite_pairs={bad_val_count}/{len(val_losses)} "
                        f"avg_loss={avg_loss}"
                    )
                if avg_loss < best_val_loss:
                    if not hide_progress:
                        prog_bar.set_postfix_str(f"Best loss on val {avg_loss:.6f}, (Iter {i})")
                    else:
                        print(f"Best loss on val {avg_loss:.6f}, (Iter {i})")
                    best_val_loss = avg_loss
                    best_model = copy.deepcopy(model)

                # Restore training mode after validation
                model.train()

                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)

                # --- Restart check (fires once, at the first validation at or after restart_check_epoch) ---
                if restart_threshold is not None and not restart_checked and i >= restart_check_epoch:
                    restart_checked = True
                    if best_val_loss > restart_threshold:
                        if run < total_runs - 1:
                            print(
                                f"[restart] Run {run + 1}/{total_runs}: "
                                f"val_loss={best_val_loss:.6f} > threshold={restart_threshold} "
                                f"at epoch {i + 1}. Restarting with fresh weights."
                            )
                            restart_triggered = True
                            break
                        else:
                            print(
                                f"[restart] Run {run + 1}/{total_runs}: "
                                f"val_loss={best_val_loss:.6f} > threshold={restart_threshold} "
                                f"at epoch {i + 1}. Max runs reached — continuing to completion."
                            )

        if not restart_triggered:
            break

    return model, loss_history, saved_snapshots, best_model, best_val_loss