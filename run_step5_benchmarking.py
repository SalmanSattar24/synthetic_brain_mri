import argparse
from pathlib import Path

from src.step5.pipeline import run_step5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 5 (benchmarking)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()

    out = run_step5(args.config)
    print(f"Step 5 complete: {out}")


if __name__ == "__main__":
    main()
