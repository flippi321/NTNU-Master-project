# Evaluate one or more trained models end-to-end:
#   inference → FastSurfer segmentation → Dice + Volumetric Similarity → SSIM + L1
#
# All outputs are written to out/<model_id>/ and are fully cached:
# re-running skips any step whose output already exists.  Stage skipping plus
# merge-on-write means SSIM/L1 can be added retroactively to models that were
# already evaluated for Dice/VS — no re-inference required.
#
# Example:
#   python evaluate_models.py \
#       --models out/bayes_search/film_simple_res_trial_012_\(0.0520\).pth \
#                out/bayes_search/std_trial_036_\(0.0538\).pth \
#                out/bayes_search/meta_trial_018_\(0.0703\).pth \
#       --device cuda:1
#
# Retroactively add SSIM/L1 to existing evaluations (reuses cached predictions):
#   python evaluate_models.py --models ... --device cuda:1 \
#       --skip_inference --skip_segmentation --skip_dice
#
# Skip individual stages if already done:
#   python evaluate_models.py --models ... --device cuda:1 --skip_inference
#   python evaluate_models.py --models ... --device cuda:1 --skip_segmentation
#   python evaluate_models.py --models ... --device cuda:1 --skip_dice
#   python evaluate_models.py --models ... --device cuda:1 --skip_ssim_l1

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.model_utils.grid_search import _build_model
from utils.model_utils.loss_functions import l1_loss, ssim_loss
from utils.model_utils.train_eval import get_metadata_features
from utils.mri.data_converter import DataConverter
from utils.mri.data_loader import DataLoader

# ─── Constants ────────────────────────────────────────────────────────────────

CROP_AXES      = ((16, 10, 0), (17, 11, 17))
THRESHOLD      = 1e-3
SPLITS_PATH    = "data/metadata/splits.json"
YAML_PATH      = "out/bayes_search/bayes_search.yaml"
BASE_CHANNELS  = 16
GT_SEG_DIR     = "out/hunt4_gt/segmentations"
FASTSURFER_DIR = os.path.join(os.path.expanduser("~"), "FastSurfer")

# Tissue label sets (DKT parcellation — identical to EX1)
GM_LABELS = set([
    *range(1000, 1036), *range(2000, 2036),
    10, 11, 12, 13, 17, 18, 26, 28,
    49, 50, 51, 52, 53, 54, 58, 60,
])
WM_LABELS  = set([2, 41, 7, 46, 77, 85, 251, 252, 253, 254, 255])
CSF_LABELS = set([4, 5, 14, 15, 24, 31, 43, 44, 63, 72])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_model_id(path: str) -> str:
    """film_simple_res_trial_012_(0.0520).pth  →  film_simple_res_trial_012"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_\(\d+\.\d+\)$", "", stem)


def parse_model_family(model_id: str) -> str:
    """film_simple_res_trial_012  →  film_simple_res"""
    m = re.match(r"^(.+?)_trial_", model_id)
    if not m:
        raise ValueError(f"Cannot determine model family from id: {model_id!r}")
    return m.group(1)


def load_yaml_trials(yaml_path: str) -> list[dict]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("trials", [])


def find_yaml_entry(trials: list[dict], model_id: str) -> dict | None:
    for t in trials:
        if t.get("id") == model_id:
            return t
        mp = t.get("results", {}).get("model_path", "")
        if model_id in os.path.basename(mp):
            return t
    return None


def to_binary_mask(seg_vol: np.ndarray, label_set: set) -> np.ndarray:
    mask = np.zeros(seg_vol.shape, dtype=bool)
    for label in label_set:
        mask |= (seg_vol == label)
    return mask


def dice_coefficient(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    total = mask_pred.sum() + mask_gt.sum()
    return float("nan") if total == 0 else 2.0 * float(intersection) / float(total)


def volumetric_similarity(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    v_pred, v_gt = float(mask_pred.sum()), float(mask_gt.sum())
    denom = v_pred + v_gt
    return float("nan") if denom == 0 else 1.0 - abs(v_pred - v_gt) / denom


def _patch_torchvision() -> None:
    spec = importlib.util.find_spec("torchvision")
    if spec is None:
        return
    meta_path = os.path.join(os.path.dirname(spec.origin), "_meta_registrations.py")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            content = f.read()
        if "# PATCHED" not in content:
            with open(meta_path, "w") as f:
                f.write("# PATCHED - disabled for Python 3.14 / torch compatibility\n")


def run_fastsurfer_seg(input_path: str, output_path: str, fs_device: str) -> bool:
    abs_out = os.path.abspath(output_path)
    sid     = os.path.basename(os.path.dirname(abs_out))
    sd      = os.path.dirname(os.path.dirname(abs_out))
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(FASTSURFER_DIR, "FastSurferCNN", "run_prediction.py"),
        "--t1",              os.path.abspath(input_path),
        "--asegdkt_segfile", abs_out,
        "--sd",              sd,
        "--sid",             sid,
        "--device",          fs_device,
        "--batch_size",      "1",
    ]
    env = {**os.environ, "PYTHONPATH": FASTSURFER_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] FastSurfer failed on {input_path}\n{result.stderr[-800:]}")
        return False
    return True


def _summary_stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std":  float(arr.std())  if len(arr) else float("nan"),
        "min":  float(arr.min())  if len(arr) else float("nan"),
        "max":  float(arr.max())  if len(arr) else float("nan"),
    }


# ─── Per-model pipeline ───────────────────────────────────────────────────────

def run_inference(
    model_id: str,
    model_family: str,
    model: torch.nn.Module,
    test_pairs: list[tuple[str, str]],
    comb_loader: CombinedMetadataUtils,
    dc: DataConverter,
    device: torch.device,
) -> None:
    pred_dir = os.path.join("out", model_id, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    is_conditional = model_family != "std"
    n_skipped = 0

    for hunt3_path, hunt4_path in tqdm(test_pairs, desc=f"Inference  {model_id}"):
        patient_id = os.path.basename(os.path.dirname(hunt3_path))
        pred_path  = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")
        if os.path.isfile(pred_path):
            n_skipped += 1
            continue

        x      = dc.load_path_as_tensor(hunt3_path, device=device)
        x_crop = dc.get_volume_with_3d_change(x, CROP_AXES, remove_mode=True)

        with torch.no_grad():
            if is_conditional:
                cond = get_metadata_features(comb_loader, hunt3_path).unsqueeze(0).to(device)
                out  = model(x_crop, cond)
            else:
                out  = model(x_crop)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            pred[pred.abs() < THRESHOLD] = 0.0
            pred_full = dc.get_volume_with_3d_change(pred.cpu(), CROP_AXES, remove_mode=False)

        ref = nib.load(hunt4_path)
        nib.save(
            nib.Nifti1Image(pred_full.squeeze().numpy().astype(np.float32), ref.affine, ref.header),
            pred_path,
        )
        del x, x_crop, out, pred, pred_full
        torch.cuda.empty_cache()

    print(f"  Inference done  ({n_skipped} cached)")


def run_segmentation(
    model_id: str,
    test_pairs: list[tuple[str, str]],
    fs_device: str,
) -> None:
    pred_dir = os.path.join("out", model_id, "predictions")
    seg_dir  = os.path.join("out", model_id, "segmentations")
    os.makedirs(GT_SEG_DIR, exist_ok=True)

    # GT segmentations (shared across all models)
    n_skip = 0
    for hunt3_path, hunt4_path in tqdm(test_pairs, desc="Seg GT"):
        patient_id  = os.path.basename(os.path.dirname(hunt3_path))
        gt_seg_path = os.path.join(GT_SEG_DIR, patient_id, "gt_seg.nii.gz")
        if os.path.isfile(gt_seg_path):
            n_skip += 1
            continue
        run_fastsurfer_seg(hunt4_path, gt_seg_path, fs_device)
    print(f"  GT seg done  ({n_skip} cached)")

    # Prediction segmentations
    n_skip = 0
    for hunt3_path, _ in tqdm(test_pairs, desc=f"Seg pred  {model_id}"):
        patient_id = os.path.basename(os.path.dirname(hunt3_path))
        pred_path  = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")
        seg_path   = os.path.join(seg_dir, patient_id, "pred_seg.nii.gz")
        if os.path.isfile(seg_path):
            n_skip += 1
            continue
        if not os.path.isfile(pred_path):
            print(f"  [WARN] No prediction for {patient_id} — skipping seg")
            continue
        run_fastsurfer_seg(pred_path, seg_path, fs_device)
    print(f"  Pred seg done  ({n_skip} cached)")


def run_dice(
    model_id: str,
    test_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    seg_dir = os.path.join("out", model_id, "segmentations")
    rows = []

    for hunt3_path, _ in tqdm(test_pairs, desc=f"Dice  {model_id}"):
        patient_id    = os.path.basename(os.path.dirname(hunt3_path))
        pred_seg_path = os.path.join(seg_dir,    patient_id, "pred_seg.nii.gz")
        gt_seg_path   = os.path.join(GT_SEG_DIR, patient_id, "gt_seg.nii.gz")

        if not os.path.isfile(pred_seg_path) or not os.path.isfile(gt_seg_path):
            print(f"  [WARN] Missing seg for {patient_id}")
            continue

        pred_seg = np.round(nib.load(pred_seg_path).get_fdata()).astype(np.int32)
        gt_seg   = np.round(nib.load(gt_seg_path).get_fdata()).astype(np.int32)

        gm_p  = to_binary_mask(pred_seg, GM_LABELS)
        gm_g  = to_binary_mask(gt_seg,   GM_LABELS)
        wm_p  = to_binary_mask(pred_seg, WM_LABELS)
        wm_g  = to_binary_mask(gt_seg,   WM_LABELS)
        csf_p = to_binary_mask(pred_seg, CSF_LABELS)
        csf_g = to_binary_mask(gt_seg,   CSF_LABELS)

        rows.append({
            "patient_id": patient_id,
            "dice_GM":    dice_coefficient(gm_p,  gm_g),
            "dice_WM":    dice_coefficient(wm_p,  wm_g),
            "dice_CSF":   dice_coefficient(csf_p, csf_g),
            "vs_GM":      volumetric_similarity(gm_p,  gm_g),
            "vs_WM":      volumetric_similarity(wm_p,  wm_g),
            "vs_CSF":     volumetric_similarity(csf_p, csf_g),
        })

    return pd.DataFrame(rows)


def run_ssim_l1(
    model_id: str,
    test_pairs: list[tuple[str, str]],
    dc: DataConverter,
    device: torch.device,
) -> pd.DataFrame:
    """Compute per-patient SSIM and L1 losses from cached predictions vs HUNT4 target."""
    pred_dir = os.path.join("out", model_id, "predictions")
    rows = []

    for hunt3_path, hunt4_path in tqdm(test_pairs, desc=f"SSIM/L1  {model_id}"):
        patient_id = os.path.basename(os.path.dirname(hunt3_path))
        pred_path  = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")
        if not os.path.isfile(pred_path):
            print(f"  [WARN] No prediction for {patient_id} — skipping SSIM/L1")
            continue

        pred_full = dc.load_path_as_tensor(pred_path,  device=device)
        y_full    = dc.load_path_as_tensor(hunt4_path, device=device)
        pred_c    = dc.get_volume_with_3d_change(pred_full, CROP_AXES, remove_mode=True)
        y_c       = dc.get_volume_with_3d_change(y_full,    CROP_AXES, remove_mode=True)

        with torch.no_grad():
            ssim_v = float(ssim_loss(pred_c, y_c).item())
            l1_v   = float(l1_loss(pred_c, y_c).item())

        rows.append({"patient_id": patient_id, "ssim_loss": ssim_v, "l1_loss": l1_v})
        del pred_full, y_full, pred_c, y_c
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def _merge_dice_csv(out_dir: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new per-patient results into existing dice_results.csv on patient_id.
    Existing columns are preserved; new columns are added; overlapping cells
    are overwritten by new_df.
    """
    csv_path = os.path.join(out_dir, "dice_results.csv")
    if not os.path.isfile(csv_path):
        return new_df

    existing = pd.read_csv(csv_path, dtype={"patient_id": str})
    if "patient_id" not in existing.columns or "patient_id" not in new_df.columns:
        return new_df

    new_df = new_df.copy()
    new_df["patient_id"]   = new_df["patient_id"].astype(str)
    existing["patient_id"] = existing["patient_id"].astype(str)

    merged = existing.merge(new_df, on="patient_id", how="outer", suffixes=("_old", ""))
    for col in new_df.columns:
        if col == "patient_id":
            continue
        old_col = f"{col}_old"
        if old_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[old_col])
            merged = merged.drop(columns=[old_col])
    return merged


def save_meta(
    model_id: str,
    model_path: str,
    model_family: str,
    entry: dict | None,
    dice_df: pd.DataFrame | None = None,
    ssim_df: pd.DataFrame | None = None,
) -> None:
    """Merge new results into existing meta.json + dice_results.csv (if any)."""
    out_dir = os.path.join("out", model_id)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    csv_path  = os.path.join(out_dir, "dice_results.csv")

    # ── Load existing meta (preserve fields we won't overwrite) ────────────
    existing = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)

    # ── Hyperparameter snapshot ────────────────────────────────────────────
    results     = entry.get("results", {}) if entry else {}
    hyperparams = existing.get("hyperparams", {})
    if entry:
        for k in ("learning_rate", "weight_decay", "beta1", "beta2", "lr_scheduler",
                  "film_generator_type", "mlp_hidden", "residual",
                  "n_meta_tokens", "num_heads"):
            if k in entry:
                hyperparams[k] = entry[k]

    # ── Merge per-patient frames ───────────────────────────────────────────
    parts = [df for df in (dice_df, ssim_df) if df is not None and not df.empty]
    if parts:
        new_df = parts[0]
        for p in parts[1:]:
            new_df = new_df.merge(p, on="patient_id", how="outer")
        merged_df = _merge_dice_csv(out_dir, new_df)
    else:
        merged_df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else None

    # ── Summary stats: recompute from merged_df where possible ─────────────
    def stats(col):
        if merged_df is None or col not in merged_df.columns:
            return None
        return _summary_stats(merged_df[col].tolist())

    meta = dict(existing)
    meta.update({
        "model_id":     model_id,
        "model_path":   model_path,
        "model_family": model_family,
        "hyperparams":  hyperparams,
        "val_loss":     results.get("best_val_loss",  existing.get("val_loss")),
        "test_loss":    results.get("best_test_loss", existing.get("test_loss")),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })
    if merged_df is not None:
        meta["n_patients"] = int(len(merged_df))

    if dice_df is not None and not dice_df.empty:
        meta["dice_summary"] = {"GM": stats("dice_GM"), "WM": stats("dice_WM"), "CSF": stats("dice_CSF")}
        meta["vs_summary"]   = {"GM": stats("vs_GM"),   "WM": stats("vs_WM"),   "CSF": stats("vs_CSF")}

    if ssim_df is not None and not ssim_df.empty:
        meta["ssim_summary"] = stats("ssim_loss")
        meta["l1_summary"]   = stats("l1_loss")

    # ── Write ──────────────────────────────────────────────────────────────
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    if merged_df is not None:
        sort_col = "dice_GM" if "dice_GM" in merged_df.columns else "patient_id"
        merged_df.sort_values(sort_col, na_position="last").reset_index(drop=True).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models: inference → segmentation → Dice")
    parser.add_argument("--models",           nargs="+", required=True, help="Paths to .pth model files")
    parser.add_argument("--device",           default="cuda:0",         help="PyTorch device (default: cuda:0)")
    parser.add_argument("--skip_inference",   action="store_true",      help="Skip inference stage")
    parser.add_argument("--skip_segmentation",action="store_true",      help="Skip FastSurfer segmentation")
    parser.add_argument("--skip_dice",        action="store_true",      help="Skip Dice computation")
    parser.add_argument("--skip_ssim_l1",     action="store_true",      help="Skip SSIM/L1 computation")
    args = parser.parse_args()

    device = torch.device(args.device)

    # FastSurfer expects "cuda", "cuda:0", "cuda:1", etc.
    fs_device = args.device if args.device.startswith("cuda") else "cpu"

    # Patch torchvision if needed (required for FastSurfer)
    _patch_torchvision()

    # Load test split (same 113 patients used in training/EX1)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_ids  = splits["test"]
    dl        = DataLoader(root_path="data/")
    test_pairs = [dl.get_pair_path_from_id(hid) for hid in test_ids]
    print(f"Test set: {len(test_pairs)} patients")

    # Load YAML trial metadata
    trials = load_yaml_trials(YAML_PATH)

    # Shared metadata loader (lazy-loads on first .get() call)
    comb_loader = CombinedMetadataUtils()
    dc          = DataConverter()

    for model_path in args.models:
        if not os.path.isfile(model_path):
            print(f"\n[ERROR] Model file not found: {model_path}")
            continue

        model_id     = parse_model_id(model_path)
        model_family = parse_model_family(model_id)
        entry        = find_yaml_entry(trials, model_id)

        print(f"\n{'='*60}")
        print(f"Model  : {model_id}")
        print(f"Family : {model_family}")
        if entry:
            print(f"Val loss: {entry.get('results', {}).get('best_val_loss', 'N/A'):.4f}")

        # ── Stage 1: Inference ──────────────────────────────────────────────
        if not args.skip_inference:
            if entry is None:
                print("[WARN] No YAML entry found — cannot build model; skipping inference")
            else:
                model = _build_model(entry, base_channels=BASE_CHANNELS,
                                     cond_dim=comb_loader.n_features, device=device)
                model.load_state_dict(torch.load(model_path, weights_only=False))
                model.eval()
                run_inference(model_id, model_family, model, test_pairs, comb_loader, dc, device)
                del model
                torch.cuda.empty_cache()

        # ── Stage 2: FastSurfer segmentation ────────────────────────────────
        if not args.skip_segmentation:
            if not os.path.isdir(FASTSURFER_DIR):
                print(f"[ERROR] FastSurfer not found at {FASTSURFER_DIR}. Clone it first.")
            else:
                run_segmentation(model_id, test_pairs, fs_device)

        # ── Stage 3: Dice + Volumetric Similarity ───────────────────────────
        dice_df: pd.DataFrame | None = None
        if not args.skip_dice:
            dice_df = run_dice(model_id, test_pairs)
            if dice_df.empty:
                print("  [WARN] No dice results computed.")
                dice_df = None

        # ── Stage 4: SSIM + L1 (uses cached predictions) ────────────────────
        ssim_df: pd.DataFrame | None = None
        if not args.skip_ssim_l1:
            ssim_df = run_ssim_l1(model_id, test_pairs, dc, device)
            if ssim_df.empty:
                print("  [WARN] No SSIM/L1 results computed.")
                ssim_df = None

        # ── Persist (merges with existing meta.json + dice_results.csv) ─────
        if dice_df is not None or ssim_df is not None:
            save_meta(model_id, model_path, model_family, entry,
                      dice_df=dice_df, ssim_df=ssim_df)

            if dice_df is not None:
                print(f"\n  Dice/VS ({len(dice_df)} patients):")
                for tissue in ("GM", "WM", "CSF"):
                    d = dice_df[f"dice_{tissue}"].mean()
                    v = dice_df[f"vs_{tissue}"].mean()
                    print(f"    {tissue}: Dice={d:.4f}  VS={v:.4f}")
            if ssim_df is not None:
                print(f"  SSIM loss : {ssim_df['ssim_loss'].mean():.4f} ± {ssim_df['ssim_loss'].std():.4f}")
                print(f"  L1   loss : {ssim_df['l1_loss'].mean():.4f} ± {ssim_df['l1_loss'].std():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
