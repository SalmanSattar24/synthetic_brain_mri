from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import yaml


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for synthetic_brain_mri pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    adni_root = Path(paths.get("adni_root", ""))

    report: dict[str, object] = {
        "config": str(args.config),
        "hard_failures": [],
        "warnings": [],
        "checks": {},
    }

    # Data path and scan count checks.
    adni_exists = adni_root.exists() and adni_root.is_dir()
    report["checks"]["adni_root_exists"] = adni_exists
    if not adni_exists:
        report["hard_failures"].append(f"ADNI root missing: {adni_root}")
        scan_count = 0
    else:
        scan_count = len(list(adni_root.rglob("*.nii"))) + len(list(adni_root.rglob("*.nii.gz")))
        report["checks"]["adni_scan_count"] = scan_count
        if scan_count == 0:
            report["hard_failures"].append(f"No NIfTI scans found under: {adni_root}")

    # Output directories are writable.
    output_paths = [
        Path(cfg.get("step2", {}).get("output_dir", "results/step2")),
        Path(cfg.get("step3", {}).get("output_dir", "results/step3")),
        Path(cfg.get("step4", {}).get("output_dir", "results/step4")),
        Path(cfg.get("step5", {}).get("output_dir", "results/step5")),
    ]
    writable = True
    for p in output_paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            writable = False
            report["hard_failures"].append(f"Cannot create output dir {p}: {exc}")
    report["checks"]["output_dirs_writable"] = writable

    # Conditional label checks.
    s2_cond = cfg.get("step2", {}).get("conditional", {})
    s3_cond = cfg.get("step3", {}).get("conditional", {})
    for name, cond in (("step2", s2_cond), ("step3", s3_cond)):
        enabled = bool(cond.get("enabled", False))
        labels_csv = str(cond.get("labels_csv", "")).strip()
        check_key = f"{name}_conditional"
        if not enabled:
            report["checks"][check_key] = {"enabled": False}
            continue

        labels_ok = bool(labels_csv) and Path(labels_csv).exists()
        report["checks"][check_key] = {
            "enabled": True,
            "labels_csv": labels_csv,
            "labels_csv_exists": labels_ok,
        }
        if not labels_ok:
            report["hard_failures"].append(
                f"{name} conditional enabled but labels CSV missing: {labels_csv or '<empty>'}"
            )

    # Core python dependencies.
    required = [
        "numpy",
        "yaml",
        "nibabel",
        "SimpleITK",
        "torch",
        "diffusers",
        "sklearn",
    ]
    dep_results = {}
    for mod in required:
        ok, detail = _check_import(mod)
        dep_results[mod] = {"ok": ok, "detail": detail}
        if not ok:
            report["hard_failures"].append(f"Missing dependency: {mod} ({detail})")
    report["checks"]["python_dependencies"] = dep_results

    # Optional dependencies.
    ants_ok, ants_detail = _check_import("ants")
    report["checks"]["optional_antspyx"] = {"ok": ants_ok, "detail": ants_detail}
    if not ants_ok:
        report["warnings"].append("antspyx not available: registration will fall back to deterministic resampling.")

    hd_bet_cli = shutil.which("hd-bet") is not None
    hd_bet_mod_ok, _ = _check_import("HD_BET")
    hd_bet_available = hd_bet_cli or hd_bet_mod_ok
    report["checks"]["optional_hd_bet"] = {
        "available": hd_bet_available,
        "cli": hd_bet_cli,
        "module": hd_bet_mod_ok,
    }
    if not hd_bet_available:
        report["warnings"].append("HD-BET not detected (CLI/module): skull stripping will be skipped.")

    print(json.dumps(report, indent=2))

    if report["hard_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
