import argparse
from pathlib import Path

from src.step3.pipeline import run_step3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3 (3D latent diffusion)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()

    out = run_step3(args.config)
    print(f"Step 3 complete: {out}")


if __name__ == "__main__":
    main()
