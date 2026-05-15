# Compute GradientSHAP feature attributions for a feature-conditioned model.
#
# For each validation subject (107 patients, matching EX1's split), this script
# explains the model's prediction quality (SSIM+L1 composite loss against HUNT4
# target) in terms of its 38 input conditioning features.  A random sample of
# training subjects' conditioning vectors is used as the GradientSHAP baseline
# distribution.
#
# Raw per-column SHAP values are saved to ex2/results/<model_id>/.  One-hot
# aggregation (e.g. Smoking Status dummies → single feature) happens later in
# shap_analyse.py so the raw values stay available.
#
# Example (run from project root):
#   python ex2/shap_compute.py \
#       --model out/bayes_search/film_simple_res_trial_012_\(0.0520\).pth \
#       --device cuda:1
#
#   python ex2/shap_compute.py \
#       --model out/bayes_search/meta_trial_018_\(0.0703\).pth \
#       --device cuda:0 --n_background 30 --n_samples 3

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime

# Reduce CUDA memory fragmentation across the per-subject SHAP loop.
# Must be set BEFORE `import torch`. Set both env-var names because PyTorch
# 2.5+ renamed the canonical form.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF",      "expandable_segments:True")

# Make project root importable when this script is invoked as
#   python ex2/shap_compute.py …
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

try:
    from captum.attr import GradientShap
except ImportError:
    print("ERROR: captum is not installed. Install it with:\n    pip install captum")
    sys.exit(1)

import torch.utils.checkpoint as _checkpoint
from pytorch_msssim import ssim as _ssim

from utils.metadata.combined_metadata_utils import CombinedMetadataUtils
from utils.model_utils.grid_search import _build_model
from utils.mri.data_converter import DataConverter
from utils.mri.data_loader import DataLoader


def per_sample_ssim_l1(pred: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.8) -> torch.Tensor:
    """Per-sample SSIM+L1 composite loss for a 5D batch (B,1,D,H,W). Returns (B,)."""
    ssim_per = _ssim(pred, target, win_size=11, win_sigma=1.5,
                     K=(0.01, 0.03), data_range=1.0, size_average=False)  # (B,)
    ssim_loss_per = 1.0 - ssim_per
    l1_per = torch.abs(pred - target).mean(dim=(1, 2, 3, 4))  # (B,)
    return alpha * ssim_loss_per + (1.0 - alpha) * l1_per


class _CheckpointedModel(torch.nn.Module):
    """Wraps a feature-conditioned model so its forward is gradient-checkpointed.
    Captum still gets correct gradients via recomputation during backward; in
    exchange, intermediate activations aren't stored, cutting peak VRAM ~3-5×.
    """
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # use_reentrant=False is required for compatibility with autograd.grad,
        # which captum uses internally.
        return _checkpoint.checkpoint(self.inner, x, cond, use_reentrant=False)

# ─── Constants ────────────────────────────────────────────────────────────────

CROP_AXES     = ((16, 10, 0), (17, 11, 17))
SPLITS_PATH   = "data/metadata/splits.json"
YAML_PATH     = "out/bayes_search/bayes_search.yaml"
BASE_CHANNELS = 16
SEED          = 69         # hardcoded: identical to EX1's split seed
SSIM_L1_ALPHA = 0.8        # identical to EX1 training loss


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_model_id(path: str) -> str:
    """film_simple_res_trial_012_(0.0520).pth → film_simple_res_trial_012"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_\(\d+\.\d+\)$", "", stem)


def parse_model_family(model_id: str) -> str:
    m = re.match(r"^(.+?)_trial_", model_id)
    if not m:
        raise ValueError(f"Cannot determine model family from id: {model_id!r}")
    return m.group(1)


def load_yaml_trials(yaml_path: str) -> list[dict]:
    with open(yaml_path) as f:
        return yaml.safe_load(f).get("trials", [])


def find_yaml_entry(trials: list[dict], model_id: str) -> dict | None:
    for t in trials:
        if t.get("id") == model_id:
            return t
        mp = t.get("results", {}).get("model_path", "")
        if model_id in os.path.basename(mp):
            return t
    return None


def cond_vector(loader: CombinedMetadataUtils, hunt_id) -> np.ndarray | None:
    """Look up a subject's 38-dim conditioning vector by integer/string ID."""
    sid = str(hunt_id)
    vec = loader.get(sid)
    if vec is None:
        # try zero-padded form
        vec = loader.get(sid.zfill(5))
    return vec


def stack_cond_vectors(loader: CombinedMetadataUtils, ids: list) -> np.ndarray:
    """(N, 38) float32 array of conditioning vectors. Missing → zeros (warned)."""
    rows = []
    for hid in ids:
        v = cond_vector(loader, hid)
        if v is None:
            print(f"  [WARN] no metadata for {hid} — using zeros")
            v = np.zeros(loader.n_features, dtype=np.float32)
        rows.append(v.astype(np.float32))
    return np.stack(rows, axis=0)


# ─── SHAP computation per subject ─────────────────────────────────────────────

def compute_shap_for_subject(
    model: torch.nn.Module,
    is_conditional: bool,            # always True here (we refuse std models)
    hunt3_path: str,
    hunt4_path: str,
    cond_vec: np.ndarray,            # (38,)
    baselines: torch.Tensor,         # (N, 38) on device
    dc: DataConverter,
    device: torch.device,
    n_samples: int,
) -> np.ndarray:
    """Return (38,) numpy SHAP values for this subject."""
    # Fix x and y_target for this subject
    x_full = dc.load_path_as_tensor(hunt3_path, device=device)
    y_full = dc.load_path_as_tensor(hunt4_path, device=device)
    x_fixed  = dc.get_volume_with_3d_change(x_full, CROP_AXES, remove_mode=True)
    y_target = dc.get_volume_with_3d_change(y_full, CROP_AXES, remove_mode=True)
    del x_full, y_full

    cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 38)

    def forward_fn(cond_b: torch.Tensor) -> torch.Tensor:
        # We drive the n_samples loop in Python so cond_b is always batch=1.
        out  = model(x_fixed, cond_b)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return per_sample_ssim_l1(pred, y_target, alpha=SSIM_L1_ALPHA)  # (1,)

    # Drive captum's n_samples in an outer Python loop with n_samples=1 per
    # call. Mathematically equivalent to calling once with n_samples=N (both
    # average independent random interpolations), but peak VRAM drops to
    # exactly ONE forward+backward instead of N×.
    shap_accum = torch.zeros_like(cond_t)
    for _ in range(n_samples):
        explainer = GradientShap(forward_fn)
        attr = explainer.attribute(cond_t, baselines=baselines, n_samples=1)  # (1, 38)
        shap_accum = shap_accum + attr.detach()
        del attr, explainer
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    shap = (shap_accum / float(n_samples)).squeeze(0).cpu().numpy()

    del shap_accum, cond_t, x_fixed, y_target, forward_fn
    gc.collect()
    torch.cuda.empty_cache()
    return shap


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute GradientSHAP feature attributions.")
    parser.add_argument("--model",        required=True, help="Path to a single .pth model file")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_background", type=int, default=50,
                        help="Training subjects sampled for the GradientSHAP baseline (default: 50)")
    parser.add_argument("--n_samples",    type=int, default=2,
                        help="captum n_samples for gradient estimation (default: 2). "
                             "Each adds one extra forward+backward pass per subject; "
                             "memory stays flat (we drive the loop in Python).")
    parser.add_argument("--force",        action="store_true",
                        help="Recompute even if output CSVs already exist")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device)

    # ── Identify model ────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] Model file not found: {args.model}")
    model_id     = parse_model_id(args.model)
    model_family = parse_model_family(model_id)

    if model_family == "std":
        sys.exit("[ERROR] std models are unconditional — no features to attribute.")

    out_dir = os.path.join("ex2", "results", model_id)
    os.makedirs(out_dir, exist_ok=True)
    shap_csv = os.path.join(out_dir, "shap_values.csv")
    feat_csv = os.path.join(out_dir, "feature_values.csv")
    meta_pth = os.path.join(out_dir, "meta.json")

    if os.path.isfile(shap_csv) and os.path.isfile(feat_csv) and not args.force:
        sys.exit(f"[SKIP] {shap_csv} already exists. Use --force to recompute.")

    # ── Splits ────────────────────────────────────────────────────────────
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    train_ids, val_ids = splits["train"], splits["val"]
    print(f"Validation subjects: {len(val_ids)}   Training pool: {len(train_ids)}")

    dl = DataLoader(root_path="data/")

    # ── Metadata loader & baseline draw ───────────────────────────────────
    loader = CombinedMetadataUtils()
    n_features = loader.n_features
    feature_names = loader.feature_names
    print(f"Conditioning vector size: {n_features} features")

    rng = np.random.default_rng(SEED)
    bg_size = min(args.n_background, len(train_ids))
    bg_ids  = list(rng.choice(train_ids, size=bg_size, replace=False))
    bg_arr  = stack_cond_vectors(loader, bg_ids)
    baselines = torch.tensor(bg_arr, dtype=torch.float32, device=device)
    print(f"Baseline distribution: {baselines.shape}")

    # ── Model ─────────────────────────────────────────────────────────────
    trials = load_yaml_trials(YAML_PATH)
    entry  = find_yaml_entry(trials, model_id)
    if entry is None:
        sys.exit(f"[ERROR] No YAML entry for {model_id}")
    val_loss = entry.get("results", {}).get("best_val_loss")

    inner = _build_model(entry, base_channels=BASE_CHANNELS,
                         cond_dim=n_features, device=device)
    inner.load_state_dict(torch.load(args.model, weights_only=False))
    inner.eval()
    # Captum needs gradients on the input (cond) only; freezing parameters
    # avoids allocating .grad buffers for the model weights each iteration.
    for p in inner.parameters():
        p.requires_grad_(False)

    # Wrap in checkpoint to cut peak activation memory ~3-5×. Compatible with
    # captum (autograd.grad) because we use use_reentrant=False.
    model = _CheckpointedModel(inner).to(device).eval()
    print(f"Loaded {model_family} model (checkpointed)  val_loss={val_loss:.4f}")

    # ── Per-subject SHAP loop ─────────────────────────────────────────────
    dc = DataConverter()
    shap_rows, feat_rows = [], []
    for hid in tqdm(val_ids, desc="SHAP"):
        hunt3_path, hunt4_path = dl.get_pair_path_from_id(str(hid))
        cond_vec = cond_vector(loader, hid)
        if cond_vec is None:
            print(f"  [WARN] no metadata for val subject {hid} — skipping")
            continue
        if not (os.path.isfile(hunt3_path) and os.path.isfile(hunt4_path)):
            print(f"  [WARN] missing scan for {hid} — skipping")
            continue

        try:
            shap_vals = compute_shap_for_subject(
                model=model,
                is_conditional=True,
                hunt3_path=hunt3_path,
                hunt4_path=hunt4_path,
                cond_vec=cond_vec,
                baselines=baselines,
                dc=dc,
                device=device,
                n_samples=args.n_samples,
            )
        except torch.cuda.OutOfMemoryError as e:
            # Strip traceback locals: they keep failed-iteration tensors alive
            msg = str(e).split("\n", 1)[0]
            print(f"  [OOM] SHAP failed for {hid}: {msg}")
            del e
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            msg = str(e).split("\n", 1)[0]
            print(f"  [ERROR] SHAP failed for {hid}: {msg}")
            del e
            gc.collect()
            torch.cuda.empty_cache()
            continue

        shap_rows.append({"patient_id": str(hid),
                          **dict(zip(feature_names, shap_vals.tolist()))})
        feat_rows.append({"patient_id": str(hid),
                          **dict(zip(feature_names, cond_vec.tolist()))})

    # ── Save ──────────────────────────────────────────────────────────────
    pd.DataFrame(shap_rows).to_csv(shap_csv, index=False)
    pd.DataFrame(feat_rows).to_csv(feat_csv, index=False)

    meta = {
        "model_id":      model_id,
        "model_family":  model_family,
        "model_path":    args.model,
        "target_metric": f"ssim_l1_alpha{SSIM_L1_ALPHA}",
        "n_background":  bg_size,
        "n_samples":     args.n_samples,
        "n_validation":  len(shap_rows),
        "feature_names": feature_names,
        "val_loss":      val_loss,
        "computed_at":   datetime.now().isoformat(timespec="seconds"),
        "seed":          SEED,
    }
    with open(meta_pth, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {shap_csv}")
    print(f"Saved: {feat_csv}")
    print(f"Saved: {meta_pth}")
    print(f"Done: SHAP for {len(shap_rows)} validation subjects.")


if __name__ == "__main__":
    main()
