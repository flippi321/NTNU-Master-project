# Select the top-N completed trials from the EX3 Bayesian feature-selection
# sweep and write a meta.json beside each chosen trial's model.pth so the
# generic evaluation pipeline (model_analysis/evaluate_models.py) can pick
# them up via its ex3-layout branch (resolve_model_context).
#
# The script does NOT run inference / segmentation itself — it prints the
# exact evaluate_models.py command to run next.
#
# Example:
#   python ex3/select_top_n.py
#   python ex3/select_top_n.py --top-n 10 --device cuda:1

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import yaml


# Map (film_generator_type, residual) → ex3 model_type string accepted by
# evaluate_models.py's _EX3_FILM_TYPES mapping.
_FILM_GEN_RES_TO_TYPE = {
    ("simple",  False): "film_simple",
    ("complex", False): "film_complex",
    ("simple",  True):  "film_simple_res",
    ("complex", True):  "film_complex_res",
}


def derive_model_type(fixed_hp: dict) -> str:
    mt  = fixed_hp.get("model_type")
    if mt == "film":
        gen = fixed_hp.get("film_generator_type")
        res = bool(fixed_hp.get("residual", False))
        key = (gen, res)
        if key not in _FILM_GEN_RES_TO_TYPE:
            raise ValueError(f"Unknown film variant: generator={gen!r}, residual={res!r}")
        return _FILM_GEN_RES_TO_TYPE[key]
    if mt == "meta":
        return "meta"
    if mt == "std":
        return "std"
    raise ValueError(f"Unknown model_type in fixed_hyperparams: {mt!r}")


def make_display_name(trial_id: str, top_k: int, corr_thr: float) -> str:
    """feat_trial_004 → 'feat_004 (k=14, ρ=0.84)'"""
    short = trial_id.replace("feat_trial_", "feat_")
    return f"{short} (k={top_k}, ρ={corr_thr:.2f})"


def build_meta(entry: dict, base_channels: int) -> dict:
    fixed_hp     = entry.get("fixed_hyperparams", {})
    feature_p    = entry.get("feature_params", {})
    results      = entry.get("results", {})
    trial_id     = entry["id"]

    hp_out = {
        "base_channels": base_channels,
        "lr":            fixed_hp.get("learning_rate"),
        "weight_decay":  fixed_hp.get("weight_decay"),
        "beta1":         fixed_hp.get("beta1"),
        "beta2":         fixed_hp.get("beta2"),
        "scheduler":     fixed_hp.get("lr_scheduler"),
    }
    if "mlp_hidden" in fixed_hp:
        hp_out["mlp_hidden"] = fixed_hp["mlp_hidden"]
    if "n_meta_tokens" in fixed_hp:
        hp_out["n_meta_tokens"] = fixed_hp["n_meta_tokens"]
    if "num_heads" in fixed_hp:
        hp_out["num_heads"] = fixed_hp["num_heads"]

    return {
        "run_name":       trial_id,
        "source_model":   entry.get("source_shap_model"),
        "model_type":     derive_model_type(fixed_hp),
        "hyperparams":    hp_out,
        "feature_params": {
            "top_k":                 feature_p.get("top_k"),
            "correlation_threshold": feature_p.get("correlation_threshold"),
        },
        "n_features_used": entry.get("n_features_raw"),
        "best_val_loss":   float(results["best_val_loss"]),
        "display_name":    make_display_name(
            trial_id, feature_p["top_k"], feature_p["correlation_threshold"]
        ),
        "trained_at":      results.get("trained_at"),
        "selected_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select top-N ex3 bayes trials by val_loss and prep them for evaluate_models.py"
    )
    parser.add_argument("--yaml",   default="out/ex3/bayes_search/bayes_search.yaml",
                        help="Path to the ex3 bayes_search YAML log")
    parser.add_argument("--top-n",  type=int, default=5,
                        help="How many top trials to select (default: 5)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device string to embed in the suggested evaluate_models.py command")
    args = parser.parse_args()

    if not os.path.isfile(args.yaml):
        print(f"[ERROR] YAML file not found: {args.yaml}", file=sys.stderr)
        sys.exit(1)

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    base_channels = int(cfg.get("params", {}).get("base_channels", 16))
    trials_all    = cfg.get("trials", [])

    completed = [t for t in trials_all if t.get("results", {}).get("completed")]
    if not completed:
        print("[ERROR] No completed trials found in YAML.", file=sys.stderr)
        sys.exit(1)

    completed.sort(key=lambda t: float(t["results"]["best_val_loss"]))
    top = completed[: args.top_n]

    print(f"\nFound {len(completed)} completed trial(s); selecting top {len(top)}:\n")
    print(f"  {'Trial':<18} {'val_loss':>10}  {'top_k':>5}  {'corr_thr':>9}  {'n_feat_raw':>11}")
    print(f"  {'-'*18} {'-'*10}  {'-'*5}  {'-'*9}  {'-'*11}")
    for t in top:
        fp = t.get("feature_params", {})
        print(f"  {t['id']:<18} "
              f"{float(t['results']['best_val_loss']):>10.6f}  "
              f"{fp.get('top_k', '?'):>5}  "
              f"{fp.get('correlation_threshold', float('nan')):>9.4f}  "
              f"{t.get('n_features_raw', '?'):>11}")

    print("\nWriting meta.json beside each selected model.pth...")
    model_paths: list[str] = []
    for entry in top:
        model_path = entry.get("results", {}).get("model_path")
        if not model_path:
            print(f"  [WARN] {entry['id']}: missing results.model_path — skipping")
            continue
        if not os.path.isfile(model_path):
            print(f"  [WARN] {entry['id']}: model.pth not found at {model_path} — skipping")
            continue

        meta      = build_meta(entry, base_channels)
        trial_dir = os.path.dirname(model_path)
        meta_path = os.path.join(trial_dir, "meta.json")

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        model_paths.append(model_path)
        print(f"  Wrote: {meta_path}  ({meta['display_name']})")

    if not model_paths:
        print("\n[ERROR] No meta.json files written — nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    print("\nNext step — run the evaluation pipeline (FastSurfer + Dice + SSIM/L1):\n")
    print(f"  python model_analysis/evaluate_models.py --device {args.device} --models \\")
    for i, mp in enumerate(model_paths):
        suffix = " \\" if i < len(model_paths) - 1 else ""
        print(f"    {mp}{suffix}")
    print("\nAfter that completes, generate the ex3_analysis figures with:\n")
    print(f"  python model_analysis/analyse_results.py \\")
    print(f"    --all \\")
    print(f"    --root out/ex3/bayes_search \\")
    print(f"    --extra-models out/film_simple_res_trial_012 \\")
    print(f"    --out_dir out/ex3_analysis\n")


if __name__ == "__main__":
    main()
