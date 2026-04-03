from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from src.step4.validation_evaluation_pipeline import run_step4


class TestStep4Pipeline(unittest.TestCase):
    def test_run_step4_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            step1_out = root / "results" / "step1"
            step3_out = root / "results" / "step3"
            (step3_out / "samples").mkdir(parents=True, exist_ok=True)
            step1_out.mkdir(parents=True, exist_ok=True)

            for i in range(3):
                arr = np.random.uniform(-1, 1, size=(16, 16, 16)).astype(np.float32)
                nib.save(nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32)), str(step1_out / f"real{i}_preproc.nii.gz"))

            synth = np.random.uniform(-1, 1, size=(3, 1, 16, 16, 16)).astype(np.float32)
            np.save(step3_out / "samples" / "generated_volumes.npy", synth)

            cfg = {
                "project": {"seed": 7},
                "paths": {"step1_output": str(step1_out)},
                "step3": {"output_dir": str(step3_out)},
                "step4": {
                    "output_dir": str(root / "results" / "step4"),
                    "max_real_volumes": 3,
                    "max_synth_volumes": 3,
                },
            }
            cfg_path = root / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            summary = run_step4(cfg_path)
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
