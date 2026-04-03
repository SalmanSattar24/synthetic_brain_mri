from pathlib import Path

from src.step4.pipeline import run_step4


if __name__ == "__main__":
    out = run_step4(Path("results/step4"))
    print(f"Step 4 complete: {out}")
