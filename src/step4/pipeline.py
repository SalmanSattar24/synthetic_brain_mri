from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import nibabel as nib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class Step4Config:
    step1_output: Path
    step3_output: Path
    output_dir: Path
    max_real_volumes: int | None
    max_synth_volumes: int | None
    seed: int


def load_step4_config(config_path: Path) -> Step4Config:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    s4 = cfg.get("step4", {})
    s3 = cfg.get("step3", {})
    project = cfg.get("project", {})

    return Step4Config(
        step1_output=Path(paths["step1_output"]),
        step3_output=Path(s3.get("output_dir", "results/step3")),
        output_dir=Path(s4.get("output_dir", "results/step4")),
        max_real_volumes=s4.get("max_real_volumes"),
        max_synth_volumes=s4.get("max_synth_volumes"),
        seed=int(project.get("seed", 42)),
    )


def _load_real_volumes(step1_output: Path, max_n: int | None) -> List[np.ndarray]:
    files = sorted(step1_output.rglob("*_preproc.nii.gz"))
    if max_n is not None:
        files = files[:max_n]
    out: List[np.ndarray] = []
    for p in files:
        vol = nib.load(str(p), mmap=True).get_fdata(dtype=np.float32)
        out.append(np.clip(np.nan_to_num(vol), -1.0, 1.0))
    return out


def _load_synth_volumes(step3_output: Path, max_n: int | None) -> List[np.ndarray]:
    npy = step3_output / "samples" / "generated_volumes.npy"
    if not npy.exists():
        raise RuntimeError(f"Synthetic volume array not found: {npy}. Run Step 3 first.")
    arr = np.load(npy)  # [N,1,D,H,W]
    vols = [np.clip(np.nan_to_num(v[0]), -1.0, 1.0) for v in arr]
    if max_n is not None:
        vols = vols[:max_n]
    return vols


def _volume_features(v: np.ndarray, hist_bins: int = 32) -> np.ndarray:
    flat = v.astype(np.float32).ravel()
    p = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)
    mean = np.array([flat.mean(), flat.std()], dtype=np.float32)
    hist, _ = np.histogram(flat, bins=hist_bins, range=(-1.0, 1.0), density=True)
    hist = hist.astype(np.float32)
    return np.concatenate([mean, p, hist], axis=0)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    # Stable proxy FID using eigen decomposition instead of scipy sqrtm.
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    vals, vecs = np.linalg.eigh(cov_prod)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    covmean = (vecs * np.sqrt(vals + 1e-12)) @ vecs.T
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _hippocampal_proxy(v: np.ndarray) -> float:
    # Approximate bilateral hippocampal region with central-temporal box proxy.
    x, y, z = v.shape
    x0, x1 = int(0.25 * x), int(0.75 * x)
    y0, y1 = int(0.55 * y), int(0.90 * y)
    z0, z1 = int(0.30 * z), int(0.70 * z)
    roi = v[x0:x1, y0:y1, z0:z1]
    return float(np.mean(roi))


def run_step4(config_path: Path) -> Path:
    cfg = load_step4_config(config_path)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    real = _load_real_volumes(cfg.step1_output, cfg.max_real_volumes)
    synth = _load_synth_volumes(cfg.step3_output, cfg.max_synth_volumes)
    if not real or not synth:
        raise RuntimeError("Step 4 requires non-empty real and synthetic volume sets.")

    real_feat = np.stack([_volume_features(v) for v in real], axis=0)
    synth_feat = np.stack([_volume_features(v) for v in synth], axis=0)

    mu_r = real_feat.mean(axis=0)
    mu_s = synth_feat.mean(axis=0)
    sigma_r = np.cov(real_feat, rowvar=False) + 1e-6 * np.eye(real_feat.shape[1])
    sigma_s = np.cov(synth_feat, rowvar=False) + 1e-6 * np.eye(synth_feat.shape[1])
    proxy_fid = _frechet_distance(mu_r, sigma_r, mu_s, sigma_s)

    real_hip = np.array([_hippocampal_proxy(v) for v in real], dtype=np.float32)
    synth_hip = np.array([_hippocampal_proxy(v) for v in synth], dtype=np.float32)

    # Classification utility: distinguish real vs synthetic on handcrafted features.
    X = np.concatenate([real_feat, synth_feat], axis=0)
    y = np.concatenate([
        np.ones(real_feat.shape[0], dtype=np.int64),
        np.zeros(synth_feat.shape[0], dtype=np.int64),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=cfg.seed,
        stratify=y,
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_score))

    summary_path = cfg.output_dir / "step4_summary.json"
    payload = {
        "step": 4,
        "name": "Three-layer validation",
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_real": len(real),
        "num_synthetic": len(synth),
        "proxy_fid": proxy_fid,
        "hippocampal_proxy": {
            "real_mean": float(real_hip.mean()),
            "real_std": float(real_hip.std()),
            "synthetic_mean": float(synth_hip.mean()),
            "synthetic_std": float(synth_hip.std()),
            "abs_mean_gap": float(abs(real_hip.mean() - synth_hip.mean())),
            "note": "Proxy ROI metric; replace with FreeSurfer biomarkers for production validation.",
        },
        "classification_utility": {
            "task": "real_vs_synthetic",
            "roc_auc": auc,
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
