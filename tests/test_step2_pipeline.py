from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


class TestStep2Pipeline(unittest.TestCase):
    def test_load_step2_config(self) -> None:
        from src.step2.pipeline import load_step2_config

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = {
                "project": {"seed": 123},
                "paths": {"step1_output": str(root / "results" / "step1")},
                "step2": {
                    "output_dir": str(root / "results" / "step2"),
                    "image_size": 64,
                    "slices_per_volume": 8,
                    "train_batch_size": 4,
                    "num_epochs": 1,
                    "learning_rate": 1e-4,
                    "num_workers": 0,
                    "train_timesteps": 100,
                    "sample_count": 2,
                },
            }
            config_path = root / "config.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            out = load_step2_config(config_path)
            self.assertEqual(out.image_size, 64)
            self.assertEqual(out.slices_per_volume, 8)
            self.assertEqual(out.seed, 123)

    def test_discover_preprocessed_volumes(self) -> None:
        from src.step2.pipeline import discover_preprocessed_volumes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir(parents=True, exist_ok=True)
            (root / "b").mkdir(parents=True, exist_ok=True)
            (root / "a" / "x_preproc.nii.gz").write_bytes(b"x")
            (root / "b" / "y_preproc.nii.gz").write_bytes(b"y")
            (root / "b" / "z.nii.gz").write_bytes(b"z")

            all_paths = discover_preprocessed_volumes(root)
            self.assertEqual(len(all_paths), 2)
            limited = discover_preprocessed_volumes(root, max_volumes=1)
            self.assertEqual(len(limited), 1)

    def test_slice_dataset_len(self) -> None:
        try:
            import nibabel as nib
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"nibabel unavailable: {exc}")
            return

        from src.step2.pipeline import ADNISliceDataset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "vol_preproc.nii.gz"
            arr = np.random.uniform(-1, 1, size=(32, 32, 20)).astype(np.float32)
            img = nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32))
            nib.save(img, str(p))

            ds = ADNISliceDataset([p], image_size=64, slices_per_volume=10)
            self.assertEqual(len(ds), 10)
            x, cls = ds[0]
            self.assertEqual(tuple(x.shape), (1, 64, 64))
            self.assertEqual(cls, 0)


if __name__ == "__main__":
    unittest.main()
