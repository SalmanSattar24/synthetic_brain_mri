from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from src.step5.benchmarking_suite_pipeline import run_step5


class TestStep5Pipeline(unittest.TestCase):
    def test_run_step5_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            step1_out = root / "results" / "step1"
            step3_out = root / "results" / "step3"
            (step3_out / "samples").mkdir(parents=True, exist_ok=True)
            step1_out.mkdir(parents=True, exist_ok=True)

            for i in range(4):
                arr = np.random.uniform(-1, 1, size=(16, 16, 16)).astype(np.float32)
                nib.save(nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32)), str(step1_out / f"real{i}_preproc.nii.gz"))

            synth = np.random.uniform(-1, 1, size=(4, 1, 16, 16, 16)).astype(np.float32)
            np.save(step3_out / "samples" / "generated_volumes.npy", synth)

            cfg = {
                "project": {"seed": 7},
                "paths": {"step1_output": str(step1_out)},
                "step3": {"output_dir": str(step3_out)},
                "step5": {
                    "output_dir": str(root / "results" / "step5"),
                    "max_real_volumes": 4,
                    "benchmark_count": 3,
                    "volume_shape": [16, 16, 16],
                },
            }
            cfg_path = root / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            summary = run_step5(cfg_path)
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
