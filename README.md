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
   - `python run_step1_preprocessing.py --config config/config.yaml`
5. Run the full 5-step sequence:
   - `python run_all_steps.py --config config/config.yaml --profile config --preflight --start-step 1 --end-step 5`

## Step runner scripts (descriptive names)

- `run_step1_preprocessing.py` → ADNI preprocessing (N4, skull-strip, registration, normalization)
- `run_step2_ddpm_baseline.py` → 2D DDPM baseline training/sampling
- `run_step3_3d_latent_diffusion.py` → 3D latent diffusion-style training/sampling
- `run_step4_validation.py` → validation metrics and utility evaluation
- `run_step5_benchmarking.py` → model comparison benchmarking

## Run profiles

- `config`: uses values exactly from your config file
- `smoke`: auto-overrides to a very small fast run
- `full`: auto-removes data caps (`max_volumes`, `max_real_volumes`, etc.)

## Resume support

- Add `--resume-auto` to skip already-completed steps based on existing summary artifacts.
- Example:
  - `python run_all_steps.py --config config/config.full.yaml --profile full --preflight --resume-auto`

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

### Colab full-run instructions

1. Upload/sync this repo to Drive (or clone it in Colab).
2. Ensure your ADNI folder is accessible at a mounted path (example):
  - `/content/drive/MyDrive/AD_Research/ADNI1_Complete_1Yr_3T/ADNI`
3. Open `notebooks/master_pipeline_colab.ipynb` and run cells top-to-bottom.
4. For a full run, set in notebook config patch cell:
  - `cfg['step1']['smoke_test'] = False`
  - increase `cfg['step1']['num_workers']` carefully (CPU/RAM dependent)
5. Use the notebook's pipeline command (already configured) with preflight:
  - `run_all_steps.py --profile config --preflight --start-step 1 --end-step 5`
6. If Colab disconnects or times out, re-run with resume enabled:
  - `run_all_steps.py --profile config --preflight --resume-auto`

> Recommended flow on Colab: run once in smoke mode first, then switch to full mode.

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

## Dataset requirements (required vs optional)

### Required to run current pipeline

- ADNI T1 NIfTI scans under `paths.adni_root`
  - Current workspace check: **420** NIfTI files detected.

### Missing (for full research-plan fidelity)

- **Diagnosis/clinical labels table** (e.g., CN/MCI/AD mapping per subject/session)
  - Needed for true conditional training/evaluation by disease group.
  - Current workspace check: no `.csv`/`.tsv` metadata files detected in `ADNI1_Complete_1Yr_3T`.

- **MNI template file** (`.nii` or `.nii.gz`) for full template-based registration
  - Current workspace check: no MNI template file detected.

### Optional but recommended

- FreeSurfer outputs (or pipeline integration) for publication-grade biomarker validation.
- Additional ADNI phases (ADNI2/GO/ADNI3) for scale and diversity.

## Enabling conditional CN/MCI/AD training

Conditional mode is now available in Steps 2 and 3.

1. Prepare a CSV with columns:
  - subject id: one of `subject_id`, `ptid`, `participant_id`, `subject`
  - diagnosis: one of `diagnosis`, `dx`, `group`, `label`
2. Supported diagnosis mapping by default:
  - `CN`/`NORMAL`/`CONTROL` → 0
  - `MCI`/`EMCI`/`LMCI` → 1
  - `AD`/`ALZHEIMER` → 2
3. Enable in config:
  - `step2.conditional.enabled: true`
  - `step2.conditional.labels_csv: <path-to-labels.csv>`
  - `step3.conditional.enabled: true`
  - `step3.conditional.labels_csv: <path-to-labels.csv>`

`run_preflight.py` will fail fast if conditional mode is enabled but labels CSV is missing.

## Real-data readiness checklist

Before running with full ADNI data, confirm:

1. `run_preflight.py` exits successfully (no hard failures)
2. `paths.adni_root` points to your real ADNI directory
3. Sufficient disk space for `results/` artifacts
4. (Optional but recommended) `hd-bet` in PATH and `antspyx` installed
