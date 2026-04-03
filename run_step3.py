from pathlib import Path

from src.step3.pipeline import run_step3


if __name__ == "__main__":
    out = run_step3(Path("results/step3"))
    print(f"Step 3 complete: {out}")
