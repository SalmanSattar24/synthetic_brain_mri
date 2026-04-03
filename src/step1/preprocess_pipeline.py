from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import yaml
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


try:
    import ants  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    ants = None


@dataclass
class Step1Config:
    adni_root: Path
    step1_output: Path
    file_globs: List[str]
    max_cases: Optional[int]
    smoke_test: bool
    smoke_test_cases: int
    num_workers: int
    overwrite: bool

    n4_enabled: bool
    n4_shrink_factor: int
    n4_num_control_points: int
    n4_max_iterations: Sequence[int]

    skull_strip_enabled: bool
    hdbet_device: str
    hdbet_mode: str

    registration_enabled: bool
    target_shape: Sequence[int]
    target_spacing: Sequence[float]
    mni_template_path: Optional[Path]

    intensity_norm_enabled: bool
    percentile_clip: Sequence[float]
    output_range: Sequence[float]


def load_config(config_path: Path) -> Step1Config:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    step1 = cfg["step1"]

    template_raw = step1["registration"].get("mni_template_path", "")
    template_path = Path(template_raw) if template_raw else None

    return Step1Config(
        adni_root=Path(paths["adni_root"]),
        step1_output=Path(paths["step1_output"]),
        file_globs=list(step1["file_globs"]),
        max_cases=step1.get("max_cases"),
        smoke_test=bool(step1.get("smoke_test", False)),
        smoke_test_cases=int(step1.get("smoke_test_cases", 5)),
        num_workers=int(step1.get("num_workers", 1)),
        overwrite=bool(step1.get("overwrite", False)),
        n4_enabled=bool(step1["n4"].get("enabled", True)),
        n4_shrink_factor=int(step1["n4"].get("shrink_factor", 4)),
        n4_num_control_points=int(step1["n4"].get("num_control_points", 4)),
        n4_max_iterations=tuple(step1["n4"].get("max_iterations", [50, 50, 30, 20])),
        skull_strip_enabled=bool(step1["skull_strip"].get("enabled", True)),
        hdbet_device=str(step1["skull_strip"].get("hdbet_device", "cpu")),
        hdbet_mode=str(step1["skull_strip"].get("hdbet_mode", "fast")),
        registration_enabled=bool(step1["registration"].get("enabled", True)),
        target_shape=tuple(step1["registration"].get("target_shape", [128, 128, 128])),
        target_spacing=tuple(step1["registration"].get("target_spacing", [1.5, 1.5, 1.5])),
        mni_template_path=template_path,
        intensity_norm_enabled=bool(step1["intensity_normalization"].get("enabled", True)),
        percentile_clip=tuple(step1["intensity_normalization"].get("percentile_clip", [0.5, 99.5])),
        output_range=tuple(step1["intensity_normalization"].get("output_range", [-1.0, 1.0])),
    )


def discover_nifti_files(adni_root: Path, file_globs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in file_globs:
        files.extend(adni_root.glob(pattern))

    unique_files = sorted({p.resolve() for p in files if p.is_file()})
    return unique_files


def n4_bias_correct(image: sitk.Image, shrink_factor: int, max_iterations: Sequence[int]) -> sitk.Image:
    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(max_iterations))
    if shrink_factor > 1:
        image_small = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
        mask_small = sitk.Shrink(mask_image, [shrink_factor] * mask_image.GetDimension())
        corrected_small = corrector.Execute(image_small, mask_small)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        return image / sitk.Exp(log_bias_field)

    return corrector.Execute(image, mask_image)


def skull_strip_with_hdbet(input_path: Path, output_path: Path, device: str, mode: str) -> bool:
    hdbet_cmd = shutil.which("hd-bet")
    if not hdbet_cmd:
        LOGGER.warning("HD-BET not found in PATH. Skipping skull stripping for %s", input_path)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        hdbet_cmd,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-device",
        device,
        "-mode",
        mode,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.warning("HD-BET failed for %s: %s", input_path, result.stderr.strip())
        return False

    return True


def register_to_mni(image_path: Path, cfg: Step1Config) -> sitk.Image:
    # Full ANTs registration if template and antspyx are available.
    if cfg.mni_template_path and cfg.mni_template_path.exists() and ants is not None:
        try:
            fixed = ants.image_read(str(cfg.mni_template_path))
            moving = ants.image_read(str(image_path))
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
            warped = reg["warpedmovout"]
            warped.to_file(str(image_path))
            return sitk.ReadImage(str(image_path))
        except Exception as exc:  # pragma: no cover - runtime path
            LOGGER.warning("ANTs registration failed for %s (%s). Falling back to resampling.", image_path, exc)

    # Fallback skeleton: deterministic resampling to target grid.
    image = sitk.ReadImage(str(image_path))
    resampled = sitk.Resample(
        image,
        list(map(int, cfg.target_shape)),
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        list(map(float, cfg.target_spacing)),
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )
    return resampled


def normalize_to_range(
    image: sitk.Image,
    low_pct: float,
    high_pct: float,
    out_min: float,
    out_max: float,
) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)

    if hi <= lo:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        clipped = np.clip(arr, lo, hi)
        norm01 = (clipped - lo) / (hi - lo)
        scaled = norm01 * (out_max - out_min) + out_min

    out = sitk.GetImageFromArray(scaled)
    out.CopyInformation(image)
    return out


def process_single_scan(scan_path: Path, cfg: Step1Config) -> Optional[Path]:
    rel = scan_path.relative_to(cfg.adni_root)
    out_dir = cfg.step1_output / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scan_path.stem}_preproc.nii.gz"

    if out_path.exists() and not cfg.overwrite:
        LOGGER.info("Skipping existing output: %s", out_path)
        return out_path

    image = sitk.ReadImage(str(scan_path))

    if cfg.n4_enabled:
        image = n4_bias_correct(
            image=image,
            shrink_factor=cfg.n4_shrink_factor,
            max_iterations=cfg.n4_max_iterations,
        )

    # Write temporary image for CLI tools that need file IO.
    tmp_path = out_dir / f"{scan_path.stem}_tmp.nii.gz"
    sitk.WriteImage(image, str(tmp_path))

    if cfg.skull_strip_enabled:
        hdbet_out = out_dir / f"{scan_path.stem}_hdbet.nii.gz"
        ok = skull_strip_with_hdbet(
            input_path=tmp_path,
            output_path=hdbet_out,
            device=cfg.hdbet_device,
            mode=cfg.hdbet_mode,
        )
        if ok and hdbet_out.exists():
            tmp_path = hdbet_out

    if cfg.registration_enabled:
        image = register_to_mni(tmp_path, cfg)
    else:
        image = sitk.ReadImage(str(tmp_path))

    if cfg.intensity_norm_enabled:
        image = normalize_to_range(
            image,
            low_pct=float(cfg.percentile_clip[0]),
            high_pct=float(cfg.percentile_clip[1]),
            out_min=float(cfg.output_range[0]),
            out_max=float(cfg.output_range[1]),
        )

    sitk.WriteImage(image, str(out_path))

    # Best-effort cleanup
    for maybe_tmp in [
        out_dir / f"{scan_path.stem}_tmp.nii.gz",
        out_dir / f"{scan_path.stem}_hdbet.nii.gz",
    ]:
        if maybe_tmp.exists():
            maybe_tmp.unlink(missing_ok=True)

    return out_path


def _process_single_scan_safe(scan_path: Path, cfg: Step1Config) -> tuple[str, str, str]:
    try:
        out = process_single_scan(scan_path, cfg)
        return str(scan_path), str(out) if out else "", "ok" if out else "skipped"
    except Exception as exc:  # pragma: no cover - runtime path
        LOGGER.exception("Failed processing %s: %s", scan_path, exc)
        return str(scan_path), "", f"error: {exc}"


def run_step1(config_path: Path) -> None:
    cfg = load_config(config_path)
    cfg.step1_output.mkdir(parents=True, exist_ok=True)

    scans = discover_nifti_files(cfg.adni_root, cfg.file_globs)
    if cfg.smoke_test:
        scans = scans[: cfg.smoke_test_cases]
        LOGGER.info("Smoke test mode enabled: using first %d scans", len(scans))
    elif cfg.max_cases:
        scans = scans[: cfg.max_cases]

    LOGGER.info("Found %d scans under %s", len(scans), cfg.adni_root)

    manifest_path = cfg.step1_output / "step1_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_scan", "output_scan", "status"])

        if cfg.num_workers <= 1:
            for scan in tqdm(scans, desc="Step1 preprocessing", unit="scan"):
                row = _process_single_scan_safe(scan, cfg)
                writer.writerow(list(row))
        else:
            LOGGER.info("Running Step 1 with %d workers", cfg.num_workers)
            with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
                futures = {executor.submit(_process_single_scan_safe, scan, cfg): scan for scan in scans}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Step1 preprocessing", unit="scan"):
                    row = future.result()
                    writer.writerow(list(row))

    LOGGER.info("Step 1 complete. Manifest: %s", manifest_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 1 ADNI preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override config and run on a small subset for fast debugging.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional max number of cases to process.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional number of parallel workers.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.smoke_test or args.max_cases is not None or args.num_workers is not None:
        cfg = load_config(args.config)
        if args.smoke_test:
            cfg.smoke_test = True
        if args.max_cases is not None:
            cfg.max_cases = args.max_cases
        if args.num_workers is not None:
            cfg.num_workers = max(1, int(args.num_workers))

        # Save temporary overridden config to avoid changing source config.
        tmp_cfg_path = cfg.step1_output / "_tmp_step1_override.yaml"
        tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)

        raw_cfg["step1"]["smoke_test"] = cfg.smoke_test
        raw_cfg["step1"]["max_cases"] = cfg.max_cases
        raw_cfg["step1"]["num_workers"] = cfg.num_workers

        with tmp_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw_cfg, f, sort_keys=False)

        run_step1(config_path=tmp_cfg_path)
        tmp_cfg_path.unlink(missing_ok=True)
        return

    run_step1(config_path=args.config)


if __name__ == "__main__":
    main()
