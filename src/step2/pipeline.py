from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset


@dataclass
class Step2Config:
    step1_output: Path
    output_dir: Path
    max_volumes: int | None
    image_size: int
    slices_per_volume: int
    train_batch_size: int
    num_epochs: int
    learning_rate: float
    num_workers: int
    seed: int
    train_timesteps: int
    sample_count: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_step2_config(config_path: Path) -> Step2Config:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    s2 = cfg.get("step2", {})

    return Step2Config(
        step1_output=Path(paths["step1_output"]),
        output_dir=Path(s2.get("output_dir", "results/step2")),
        max_volumes=s2.get("max_volumes"),
        image_size=int(s2.get("image_size", 128)),
        slices_per_volume=int(s2.get("slices_per_volume", 16)),
        train_batch_size=int(s2.get("train_batch_size", 8)),
        num_epochs=int(s2.get("num_epochs", 1)),
        learning_rate=float(s2.get("learning_rate", 1e-4)),
        num_workers=int(s2.get("num_workers", 2)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
        train_timesteps=int(s2.get("train_timesteps", 1000)),
        sample_count=int(s2.get("sample_count", 8)),
    )


def discover_preprocessed_volumes(step1_output: Path, max_volumes: int | None = None) -> List[Path]:
    volumes = sorted(step1_output.rglob("*_preproc.nii.gz"))
    if max_volumes is not None:
        volumes = volumes[:max_volumes]
    return volumes


def _select_slice_indices(depth: int, slices_per_volume: int) -> np.ndarray:
    if depth <= 0:
        return np.array([], dtype=np.int64)
    if slices_per_volume <= 1:
        return np.array([depth // 2], dtype=np.int64)
    margin = max(1, depth // 10)
    start, end = margin, max(margin + 1, depth - margin)
    idx = np.linspace(start, end - 1, num=slices_per_volume)
    return np.clip(np.round(idx).astype(np.int64), 0, depth - 1)


def _resize_slice(slice_2d: np.ndarray, image_size: int) -> np.ndarray:
    t = torch.from_numpy(slice_2d.astype(np.float32))[None, None, ...]
    t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


class ADNISliceDataset(Dataset[torch.Tensor]):
    def __init__(self, volume_paths: Sequence[Path], image_size: int, slices_per_volume: int):
        self.image_size = image_size
        self.samples: List[Tuple[Path, int]] = []

        for p in volume_paths:
            data = nib.load(str(p), mmap=True).get_fdata(dtype=np.float32)
            if data.ndim != 3:
                continue
            slice_indices = _select_slice_indices(data.shape[2], slices_per_volume)
            self.samples.extend((p, int(i)) for i in slice_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p, k = self.samples[idx]
        data = nib.load(str(p), mmap=True).get_fdata(dtype=np.float32)
        slice_2d = data[:, :, k]

        # Data is expected near [-1, 1] after Step 1. Clamp for robustness.
        slice_2d = np.clip(slice_2d, -1.0, 1.0)
        slice_2d = _resize_slice(slice_2d, self.image_size)
        tensor = torch.from_numpy(slice_2d[None, ...]).float()
        return tensor


def _train_ddpm(dataset: ADNISliceDataset, cfg: Step2Config, device: torch.device) -> tuple[UNet2DModel, list[float]]:
    loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = UNet2DModel(
        sample_size=cfg.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=cfg.train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    epoch_losses: list[float] = []
    model.train()
    for _ in range(cfg.num_epochs):
        running = 0.0
        n_batches = 0
        for clean in loader:
            clean = clean.to(device)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device).long()
            noisy = scheduler.add_noise(clean, noise, timesteps)
            noise_pred = model(noisy, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        epoch_losses.append(running / max(1, n_batches))

    return model, epoch_losses


@torch.no_grad()
def _sample_images(model: UNet2DModel, cfg: Step2Config, device: torch.device) -> np.ndarray:
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=cfg.train_timesteps)
    x = torch.randn(cfg.sample_count, 1, cfg.image_size, cfg.image_size, device=device)

    for t in scheduler.timesteps:
        residual = model(x, t).sample
        x = scheduler.step(residual, t, x).prev_sample

    x = x.clamp(-1.0, 1.0).cpu().numpy()
    return x


def run_step2(config_path: Path) -> Path:
    cfg = load_step2_config(config_path)
    _set_seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.output_dir / "checkpoints"
    sample_dir = cfg.output_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    volume_paths = discover_preprocessed_volumes(cfg.step1_output, cfg.max_volumes)
    if not volume_paths:
        raise RuntimeError(
            f"No preprocessed volumes found in {cfg.step1_output}. "
            "Run Step 1 first or adjust config paths."
        )

    dataset = ADNISliceDataset(
        volume_paths=volume_paths,
        image_size=cfg.image_size,
        slices_per_volume=cfg.slices_per_volume,
    )
    if len(dataset) == 0:
        raise RuntimeError("Step 2 dataset is empty after slice extraction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = _train_ddpm(dataset, cfg, device)
    samples = _sample_images(model, cfg, device)

    model_path = ckpt_dir / "ddpm_unet2d.pt"
    torch.save(model.state_dict(), model_path)

    np.save(sample_dir / "samples.npy", samples)

    summary_path = cfg.output_dir / "step2_summary.json"
    payload = {
        "step": 2,
        "name": "2D DDPM baseline",
        "status": "complete",
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "device": str(device),
        "num_input_volumes": len(volume_paths),
        "num_training_slices": len(dataset),
        "num_epochs": cfg.num_epochs,
        "epoch_losses": losses,
        "checkpoint": str(model_path),
        "sample_array": str(sample_dir / "samples.npy"),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
