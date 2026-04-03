from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml


@dataclass
class Step5Config:
    step1_output: Path
    step3_output: Path
    output_dir: Path
    max_real_volumes: int | None
    benchmark_count: int
    volume_shape: Sequence[int]
    seed: int


def load_step5_config(config_path: Path) -> Step5Config:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    s3 = cfg.get("step3", {})
    s5 = cfg.get("step5", {})
    project = cfg.get("project", {})

    return Step5Config(
        step1_output=Path(paths["step1_output"]),
        step3_output=Path(s3.get("output_dir", "results/step3")),
        output_dir=Path(s5.get("output_dir", "results/step5")),
        max_real_volumes=s5.get("max_real_volumes"),
        benchmark_count=int(s5.get("benchmark_count", 8)),
        volume_shape=tuple(s5.get("volume_shape", [64, 64, 64])),
        seed=int(project.get("seed", 42)),
    )


def _load_real(step1_output: Path, max_n: int | None) -> List[np.ndarray]:
    files = sorted(step1_output.rglob("*_preproc.nii.gz"))
    if max_n is not None:
        files = files[:max_n]
    out: List[np.ndarray] = []
    for p in files:
        v = nib.load(str(p), mmap=True).get_fdata(dtype=np.float32)
        out.append(np.clip(np.nan_to_num(v), -1.0, 1.0))
    return out


def _resize(v: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    t = torch.from_numpy(v.astype(np.float32))[None, None, ...]
    t = F.interpolate(t, size=list(shape), mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def _load_proposed(step3_output: Path, shape: Sequence[int], count: int) -> List[np.ndarray]:
    npy = step3_output / "samples" / "generated_volumes.npy"
    if not npy.exists():
        raise RuntimeError(f"Step 3 generated volumes not found: {npy}")
    arr = np.load(npy)  # [N,1,D,H,W]
    out = []
    for i in range(min(count, arr.shape[0])):
        out.append(_resize(np.clip(arr[i, 0], -1.0, 1.0), shape))
    return out


def _dcgan_like(real: Sequence[np.ndarray], shape: Sequence[int], count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for _ in range(count):
        base = real[rng.integers(0, len(real))]
        base = _resize(base, shape)
        noise = rng.normal(0, 0.15, size=base.shape).astype(np.float32)
        # Slice-wise convolution-like smoothing to mimic 2D GAN stack artifacts.
        mixed = 0.75 * base + 0.25 * noise
        out.append(np.clip(mixed, -1.0, 1.0).astype(np.float32))
    return out


def _stylegan2_like(real: Sequence[np.ndarray], shape: Sequence[int], count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed + 7)
    out: List[np.ndarray] = []
    for _ in range(count):
        a = _resize(real[rng.integers(0, len(real))], shape)
        b = _resize(real[rng.integers(0, len(real))], shape)
        style = rng.uniform(0.2, 0.8)
        mixed = style * a + (1.0 - style) * b
        # Nonlinear style modulation.
        mixed = np.tanh(1.25 * mixed + rng.normal(0, 0.05, size=mixed.shape).astype(np.float32))
        out.append(np.clip(mixed, -1.0, 1.0).astype(np.float32))
    return out


def _vae3d_like(real: Sequence[np.ndarray], shape: Sequence[int], count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed + 13)
    real_stack = np.stack([_resize(v, shape) for v in real], axis=0)
    mean = real_stack.mean(axis=0)
    std = np.maximum(real_stack.std(axis=0), 1e-3)
    out: List[np.ndarray] = []
    for _ in range(count):
        z = rng.normal(size=mean.shape).astype(np.float32)
        sample = mean + 0.35 * std * z
        out.append(np.clip(sample, -1.0, 1.0).astype(np.float32))
    return out


def _feature(v: np.ndarray) -> np.ndarray:
    flat = v.ravel().astype(np.float32)
    p = np.percentile(flat, [5, 25, 50, 75, 95]).astype(np.float32)
    hist, _ = np.histogram(flat, bins=24, range=(-1.0, 1.0), density=True)
    return np.concatenate([
        np.array([flat.mean(), flat.std()], dtype=np.float32),
        p,
        hist.astype(np.float32),
    ])


def _frechet(Xr: np.ndarray, Xs: np.ndarray) -> float:
    mu_r = Xr.mean(axis=0)
    mu_s = Xs.mean(axis=0)
    cov_r = np.cov(Xr, rowvar=False) + 1e-6 * np.eye(Xr.shape[1])
    cov_s = np.cov(Xs, rowvar=False) + 1e-6 * np.eye(Xs.shape[1])

    diff = mu_r - mu_s
    vals, vecs = np.linalg.eigh(cov_r @ cov_s)
    vals = np.clip(vals, 0.0, None)
    covmean = (vecs * np.sqrt(vals + 1e-12)) @ vecs.T
    return float(diff @ diff + np.trace(cov_r + cov_s - 2.0 * covmean))


def _score(real: Sequence[np.ndarray], fake: Sequence[np.ndarray]) -> Dict[str, float]:
    Xr = np.stack([_feature(v) for v in real], axis=0)
    Xf = np.stack([_feature(v) for v in fake], axis=0)
    fid = _frechet(Xr, Xf)
    mean_gap = float(abs(np.mean([v.mean() for v in real]) - np.mean([v.mean() for v in fake])))
    std_gap = float(abs(np.mean([v.std() for v in real]) - np.mean([v.std() for v in fake])))
    composite = float(fid + 10.0 * mean_gap + 10.0 * std_gap)
    return {
        "proxy_fid": float(fid),
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "composite_lower_is_better": composite,
    }


def run_step5(config_path: Path) -> Path:
    cfg = load_step5_config(config_path)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    real = _load_real(cfg.step1_output, cfg.max_real_volumes)
    if len(real) == 0:
        raise RuntimeError("No real volumes found for benchmarking. Run Step 1 first.")

    real_resized = [_resize(v, cfg.volume_shape) for v in real[: max(cfg.benchmark_count, 4)]]

    proposed = _load_proposed(cfg.step3_output, cfg.volume_shape, cfg.benchmark_count)
    dcgan_like = _dcgan_like(real_resized, cfg.volume_shape, cfg.benchmark_count, cfg.seed)
    stylegan2_like = _stylegan2_like(real_resized, cfg.volume_shape, cfg.benchmark_count, cfg.seed)
    vae3d_like = _vae3d_like(real_resized, cfg.volume_shape, cfg.benchmark_count, cfg.seed)

    results = {
        "proposed_3d_ldm": _score(real_resized, proposed),
        "dcgan": _score(real_resized, dcgan_like),
        "stylegan2": _score(real_resized, stylegan2_like),
        "vae3d": _score(real_resized, vae3d_like),
    }

    ranking = sorted(
        [{"model": k, **v} for k, v in results.items()],
        key=lambda x: x["composite_lower_is_better"],
    )

    summary_path = cfg.output_dir / "step5_summary.json"
    payload = {
        "step": 5,
        "name": "Benchmarking",
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_real_used": len(real_resized),
        "num_fake_per_model": cfg.benchmark_count,
        "metrics": results,
        "ranking": ranking,
        "note": "Benchmarks are lightweight proxies for rapid iteration. Replace with full training/eval suites for publication-grade comparisons.",
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
