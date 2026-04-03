from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic brain MRI pipeline end-to-end")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=5)
    args = parser.parse_args()

    steps = {
        1: [sys.executable, "run_step1.py", "--config", str(args.config)],
        2: [sys.executable, "run_step2.py", "--config", str(args.config)],
        3: [sys.executable, "run_step3.py"],
        4: [sys.executable, "run_step4.py"],
        5: [sys.executable, "run_step5.py"],
    }

    if args.smoke_test:
        steps[1].append("--smoke-test")
    if args.num_workers is not None:
        steps[1].extend(["--num-workers", str(args.num_workers)])

    for step in range(args.start_step, args.end_step + 1):
        run_cmd(steps[step])

    print("All requested steps completed.")


if __name__ == "__main__":
    main()
