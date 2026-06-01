# NTNU-Master-project

Predicting longitudinal change in brain MRI with 3D U-Nets. Given a baseline
scan (HUNT3), the models predict the follow-up scan (HUNT4), optionally
conditioned on participant health metadata. The repository contains the model
definitions, training/evaluation utilities, and the experiment notebooks and
scripts used in the thesis.

## Models

Four 3D U-Net variants live in [models/](models/):

- `unet_3d_std` — baseline 3D U-Net.
- `unet_3d_res` — residual variant that predicts the HUNT3→HUNT4 delta.
- `unet_3d_film` — FiLM-conditioned U-Net that modulates features on metadata.
- `unet_3d_meta` — metadata-conditioned variant.

## Experiments

- **EX1 — Architecture comparison.** Bayesian hyperparameter search (Optuna TPE)
  across the U-Net variants. See `EX1-compare-unets.ipynb` / `EX1-compare-unets.py`.
- **EX2 — Feature evaluation.** SHAP-based analysis of metadata features in [ex2/](ex2/).
- **EX3 — Feature selection.** Search over `top_k` and correlation threshold in [ex3/](ex3/).
- **EX3.5 — Joint search.** Unified hyperparameter + feature search in [ex3_5/](ex3_5/)
  (`EX3.5-train-best-model.py`).
- **EX4 — Subgroup analysis.** Per-subgroup performance comparison in `EX4_subsets.ipynb`.

Supporting scripts: `run_bayes.py`, `run_bayes_ex3.py` (Optuna search runners) and
`estimate_power_usage.py` (CodeCarbon energy estimate for a training run).

## Repository layout

```
models/          3D U-Net architectures
utils/
  mri/           MRI loading and volume preprocessing
  metadata/      health-data / FastSurfer metadata handling and splits
  model_utils/   training loop, loss functions, Bayesian/grid search
  analysis/      exploratory dataset analysis
model_analysis/  model evaluation and result aggregation
ex2/ ex3/ ex3_5/ feature evaluation / selection experiment code
*.ipynb          experiment notebooks
```

`freesurfer_commands.sh` documents the FastSurfer / Singularity commands used to
produce the segmentations referenced during evaluation.

## Setup

```bash
pip install -r requirements.txt
```

## Data availability

The HUNT MRI dataset and associated health metadata are access-controlled and
are **not** included in this repository (the `data/` and `out/` directories are
git-ignored). The notebooks reference these paths but cannot be re-run without
authorised access to the data. To respect participant privacy, no subject
identifiers are stored in the committed notebooks or scripts; saved notebook
outputs have been cleared for the same reason.
