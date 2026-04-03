from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from src.step3.pipeline import discover_real_volumes, run_step3


class TestStep3Pipeline(unittest.TestCase):
    def test_discover_real_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a_preproc.nii.gz").write_bytes(b"x")
            (root / "b_preproc.nii.gz").write_bytes(b"y")
            out = discover_real_volumes(root, max_volumes=1)
            self.assertEqual(len(out), 1)

    def test_run_step3_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            step1_out = root / "results" / "step1"
            step1_out.mkdir(parents=True, exist_ok=True)

            # synthetic preprocessed volumes in [-1, 1]
            for i in range(2):
                arr = np.random.uniform(-1, 1, size=(24, 24, 24)).astype(np.float32)
                nib.save(nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32)), str(step1_out / f"sub{i}_preproc.nii.gz"))

            cfg = {
                "project": {"seed": 7},
                "paths": {"step1_output": str(step1_out)},
                "step3": {
                    "output_dir": str(root / "results" / "step3"),
                    "max_volumes": 2,
                    "volume_shape": [16, 16, 16],
                    "batch_size": 1,
                    "autoencoder_epochs": 1,
                    "diffusion_epochs": 1,
                    "learning_rate": 1e-4,
                    "latent_channels": 4,
                    "train_timesteps": 20,
                    "sample_count": 2,
                    "num_workers": 0,
                },
            }
            cfg_path = root / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            summary = run_step3(cfg_path)
            self.assertTrue(summary.exists())
            self.assertTrue((root / "results" / "step3" / "samples" / "generated_volumes.npy").exists())


if __name__ == "__main__":
    unittest.main()
