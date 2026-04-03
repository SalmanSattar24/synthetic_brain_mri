import argparse
from pathlib import Path

from src.step2.pipeline import run_step2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 2 (2D DDPM baseline)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()

    out = run_step2(args.config)
    print(f"Step 2 complete: {out}")


if __name__ == "__main__":
    main()
