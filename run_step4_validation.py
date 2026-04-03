import argparse
from pathlib import Path

from src.step4.pipeline import run_step4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 4 (validation suite)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()

    out = run_step4(args.config)
    print(f"Step 4 complete: {out}")


if __name__ == "__main__":
    main()
