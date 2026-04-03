from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic brain MRI pipeline end-to-end")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--profile",
        type=str,
        default="config",
        choices=["config", "smoke", "full"],
        help="Run profile: config (as-is), smoke (small fast run), full (remove data caps).",
    )
    parser.add_argument("--preflight", action="store_true", help="Run environment/data preflight checks first.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=5)
    args = parser.parse_args()

    profile = "smoke" if args.smoke_test else args.profile

    effective_config = args.config
    tmp_config_path: Path | None = None
    if profile != "config":
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if profile == "smoke":
            cfg.setdefault("step1", {})["smoke_test"] = True
            cfg["step1"].setdefault("smoke_test_cases", 5)
            cfg.setdefault("step2", {})["max_volumes"] = min(int(cfg.get("step2", {}).get("max_volumes", 12) or 12), 12)
            cfg["step2"]["num_epochs"] = 1
            cfg.setdefault("step3", {})["max_volumes"] = min(int(cfg.get("step3", {}).get("max_volumes", 6) or 6), 6)
            cfg["step3"]["autoencoder_epochs"] = 1
            cfg["step3"]["diffusion_epochs"] = 1
            cfg.setdefault("step4", {})["max_real_volumes"] = 8
            cfg["step4"]["max_synth_volumes"] = 8
            cfg.setdefault("step5", {})["max_real_volumes"] = 12
            cfg["step5"]["benchmark_count"] = 4
        elif profile == "full":
            cfg.setdefault("step1", {})["smoke_test"] = False
            cfg["step1"]["max_cases"] = None
            cfg.setdefault("step2", {})["max_volumes"] = None
            cfg.setdefault("step3", {})["max_volumes"] = None
            cfg.setdefault("step4", {})["max_real_volumes"] = None
            cfg["step4"]["max_synth_volumes"] = None
            cfg.setdefault("step5", {})["max_real_volumes"] = None

        tmp_config_path = Path("results") / "_tmp" / f"config.{profile}.yaml"
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        effective_config = tmp_config_path

    if args.preflight:
        run_cmd([sys.executable, "run_preflight.py", "--config", str(effective_config)])

    steps = {
        1: [sys.executable, "run_step1_preprocessing.py", "--config", str(effective_config)],
        2: [sys.executable, "run_step2_ddpm_baseline.py", "--config", str(effective_config)],
        3: [sys.executable, "run_step3_3d_latent_diffusion.py", "--config", str(effective_config)],
        4: [sys.executable, "run_step4_validation.py", "--config", str(effective_config)],
        5: [sys.executable, "run_step5_benchmarking.py", "--config", str(effective_config)],
    }

    if args.smoke_test:
        steps[1].append("--smoke-test")
    if args.num_workers is not None:
        steps[1].extend(["--num-workers", str(args.num_workers)])

    for step in range(args.start_step, args.end_step + 1):
        run_cmd(steps[step])

    if tmp_config_path is not None and tmp_config_path.exists():
        tmp_config_path.unlink(missing_ok=True)

    print("All requested steps completed.")


if __name__ == "__main__":
    main()
