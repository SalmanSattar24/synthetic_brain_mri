from pathlib import Path

from src.step5.pipeline import run_step5


if __name__ == "__main__":
    out = run_step5(Path("results/step5"))
    print(f"Step 5 complete: {out}")
