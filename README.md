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
3. Run Step 1:
   - `python run_step1.py --config config/config.yaml`
4. Run the full 5-step sequence:
  - `python run_all_steps.py --config config/config.yaml --start-step 1 --end-step 5`

## Google Colab (single master notebook)

- Use `notebooks/master_pipeline_colab.ipynb` for one-notebook orchestration.
- The notebook mounts Google Drive, patches config paths for Colab, installs dependencies, and executes `run_all_steps.py`.
- Step 1 supports:
  - smoke test mode (`smoke_test: true`, default 5 scans)
  - parallel preprocessing workers (`num_workers`)

## Notes

- ADNI path is set to:
  `C:/All-Code/AD_Research/ADNI1_Complete_1Yr_3T/ADNI`
- For skull stripping, install and expose **HD-BET** CLI in your environment PATH.
- For full MNI registration, set `step1.registration.mni_template_path` to a valid MNI template file.
