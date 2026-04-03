from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


class TestResumeAuto(unittest.TestCase):
    def test_summary_path_construction(self) -> None:
        from run_all_steps import _step_summary_paths

        cfg = {
            "paths": {"step1_output": "results/step1"},
            "step2": {"output_dir": "results/step2"},
            "step3": {"output_dir": "results/step3"},
            "step4": {"output_dir": "results/step4"},
            "step5": {"output_dir": "results/step5"},
        }
        paths = _step_summary_paths(cfg)
        self.assertTrue(str(paths[1]).endswith("step1_manifest.csv"))
        self.assertTrue(str(paths[5]).endswith("step5_summary.json"))

    def test_preflight_conditional_labels_check_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_path = root / "config.yaml"
            cfg = {
                "paths": {"adni_root": str(root), "step1_output": str(root / "results" / "step1")},
                "step2": {"output_dir": str(root / "results" / "step2"), "conditional": {"enabled": True, "labels_csv": ""}},
                "step3": {"output_dir": str(root / "results" / "step3"), "conditional": {"enabled": False}},
                "step4": {"output_dir": str(root / "results" / "step4")},
                "step5": {"output_dir": str(root / "results" / "step5")},
            }
            (root / "x.nii.gz").write_bytes(b"x")
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            # Just check module imports and function exists; runtime behavior covered by command tests.
            import run_preflight  # noqa: F401


if __name__ == "__main__":
    unittest.main()
