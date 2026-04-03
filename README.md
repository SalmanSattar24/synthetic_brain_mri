# Synthetic Brain MRI Generation (ADNI, Conditional T1)

This repository contains a 5-step pipeline for conditional synthesis of T1-weighted brain MRI volumes (Normal/MCI/AD), targeting MNI-aligned `128^3` outputs.

## Pipeline Overview

1. **Step 1**: Data preprocessing (N4, skull stripping, registration, normalization)
2. **Step 2**: 2D DDPM baseline
3. **Step 3**: 3D latent diffusion model
4. **Step 4**: Three-layer validation (FID, biomarker realism, classification utility)
5. **Step 5**: Benchmarking against alternatives

## Quick Start

1. Install dependencies from `requirements.txt`.
2. Verify `config/config.yaml` paths and hyperparameters.
3. Run preflight checks:
   - `python run_preflight.py --config config/config.yaml`
4. Run Step 1:
   - `python run_step1.py --config config/config.yaml`
5. Run the full 5-step sequence:
  - `python run_all_steps.py --config config/config.yaml --profile config --preflight --start-step 1 --end-step 5`

## Run profiles

- `config`: uses values exactly from your config file
- `smoke`: auto-overrides to a very small fast run
- `full`: auto-removes data caps (`max_volumes`, `max_real_volumes`, etc.)

Examples:

- Smoke debug:
  - `python run_all_steps.py --config config/config.smoke.yaml --profile smoke --preflight`
- Full real-data run:
  - `python run_all_steps.py --config config/config.full.yaml --profile full --preflight`

## Google Colab (single master notebook)

- Use `notebooks/master_pipeline_colab.ipynb` for one-notebook orchestration.
- The notebook mounts Google Drive, patches config paths for Colab, installs dependencies, and executes `run_all_steps.py`.
- Step 1 supports:
  - smoke test mode (`smoke_test: true`, default 5 scans)
  - parallel preprocessing workers (`num_workers`)

## Current implementation status

- Step 1: implemented preprocessing pipeline (N4, HD-BET hook, registration + normalization)
- Step 2: implemented 2D DDPM baseline training/sampling
- Step 3: implemented 3D latent diffusion-style training/sampling
- Step 4: implemented validation suite with proxy FID + biomarker proxy + classification utility
- Step 5: implemented benchmarking suite against DCGAN/StyleGAN2/VAE3D-style baselines

> Note: Step 4/5 use computationally lightweight proxy evaluations to support rapid iteration and Colab workflows. You can swap in full FreeSurfer and publication-grade metrics later without changing orchestration.

## Notes

- ADNI path is set to:
  `C:/All-Code/AD_Research/ADNI1_Complete_1Yr_3T/ADNI`
- For skull stripping, install and expose **HD-BET** CLI in your environment PATH.
- For full MNI registration, set `step1.registration.mni_template_path` to a valid MNI template file.

## Real-data readiness checklist

Before running with full ADNI data, confirm:

1. `run_preflight.py` exits successfully (no hard failures)
2. `paths.adni_root` points to your real ADNI directory
3. Sufficient disk space for `results/` artifacts
4. (Optional but recommended) `hd-bet` in PATH and `antspyx` installed
