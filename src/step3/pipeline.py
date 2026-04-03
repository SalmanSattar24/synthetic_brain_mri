from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import random
from dataclasses import dataclass
from typing import List, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Step3Config:
    step1_output: Path
    output_dir: Path
    max_volumes: int | None
    volume_shape: Sequence[int]
    batch_size: int
    autoencoder_epochs: int
    diffusion_epochs: int
    learning_rate: float
    latent_channels: int
    train_timesteps: int
    sample_count: int
    num_workers: int
    seed: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_step3_config(config_path: Path) -> Step3Config:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    s3 = cfg.get("step3", {})
    project = cfg.get("project", {})

    return Step3Config(
        step1_output=Path(paths["step1_output"]),
        output_dir=Path(s3.get("output_dir", "results/step3")),
        max_volumes=s3.get("max_volumes"),
        volume_shape=tuple(s3.get("volume_shape", [64, 64, 64])),
        batch_size=int(s3.get("batch_size", 2)),
        autoencoder_epochs=int(s3.get("autoencoder_epochs", 1)),
        diffusion_epochs=int(s3.get("diffusion_epochs", 1)),
        learning_rate=float(s3.get("learning_rate", 1e-4)),
        latent_channels=int(s3.get("latent_channels", 8)),
        train_timesteps=int(s3.get("train_timesteps", 200)),
        sample_count=int(s3.get("sample_count", 4)),
        num_workers=int(s3.get("num_workers", 0)),
        seed=int(project.get("seed", 42)),
    )


def discover_real_volumes(step1_output: Path, max_volumes: int | None = None) -> List[Path]:
    volumes = sorted(step1_output.rglob("*_preproc.nii.gz"))
    if max_volumes is not None:
        volumes = volumes[:max_volumes]
    return volumes


def _resize_volume(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    t = torch.from_numpy(volume.astype(np.float32))[None, None, ...]
    t = F.interpolate(t, size=list(target_shape), mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


class ADNI3DVolumeDataset(Dataset[torch.Tensor]):
    def __init__(self, volume_paths: Sequence[Path], target_shape: Sequence[int]):
        self.volume_paths = list(volume_paths)
        self.target_shape = tuple(int(x) for x in target_shape)

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.volume_paths[idx]
        vol = nib.load(str(p), mmap=True).get_fdata(dtype=np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=-1.0)
        vol = np.clip(vol, -1.0, 1.0)
        vol = _resize_volume(vol, self.target_shape)
        return torch.from_numpy(vol[None, ...]).float()


class TinyAutoencoder3D(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class TinyLatentDenoiser3D(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_channels + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Broadcast scalar timestep embedding as extra channel.
        t_embed = t.float().view(-1, 1, 1, 1, 1) / 1000.0
        t_embed = t_embed.expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3], z_noisy.shape[4])
        return self.net(torch.cat([z_noisy, t_embed], dim=1))


def run_step3(config_path: Path) -> Path:
    cfg = load_step3_config(config_path)
    _set_seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.output_dir / "checkpoints"
    sample_dir = cfg.output_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    volumes = discover_real_volumes(cfg.step1_output, cfg.max_volumes)
    if not volumes:
        raise RuntimeError(f"No preprocessed volumes found in {cfg.step1_output}. Run Step 1 first.")

    dataset = ADNI3DVolumeDataset(volumes, cfg.volume_shape)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = TinyAutoencoder3D(latent_channels=cfg.latent_channels).to(device)
    denoiser = TinyLatentDenoiser3D(latent_channels=cfg.latent_channels).to(device)

    ae_opt = torch.optim.AdamW(autoencoder.parameters(), lr=cfg.learning_rate)
    ae_losses: list[float] = []

    autoencoder.train()
    for _ in range(cfg.autoencoder_epochs):
        epoch_loss = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            x_hat = autoencoder(x)
            loss = F.l1_loss(x_hat, x) + F.mse_loss(x_hat, x)

            ae_opt.zero_grad(set_to_none=True)
            loss.backward()
            ae_opt.step()

            epoch_loss += float(loss.item())
            n += 1
        ae_losses.append(epoch_loss / max(1, n))

    scheduler = DDPMScheduler(num_train_timesteps=cfg.train_timesteps)
    denoise_opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.learning_rate)
    diff_losses: list[float] = []

    autoencoder.eval()
    denoiser.train()
    for _ in range(cfg.diffusion_epochs):
        epoch_loss = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                z = autoencoder.encoder(x)

            noise = torch.randn_like(z)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],), device=device).long()
            z_noisy = scheduler.add_noise(z, noise, t)
            noise_pred = denoiser(z_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            denoise_opt.zero_grad(set_to_none=True)
            loss.backward()
            denoise_opt.step()

            epoch_loss += float(loss.item())
            n += 1
        diff_losses.append(epoch_loss / max(1, n))

    # Infer latent spatial shape from one batch.
    with torch.no_grad():
        example_x = next(iter(loader)).to(device)
        latent_shape = tuple(autoencoder.encoder(example_x).shape[1:])

    denoiser.eval()
    with torch.no_grad():
        z = torch.randn((cfg.sample_count, *latent_shape), device=device)
        for t in scheduler.timesteps:
            t_batch = torch.full((cfg.sample_count,), int(t), device=device, dtype=torch.long)
            noise_pred = denoiser(z, t_batch)
            z = scheduler.step(noise_pred, t, z).prev_sample

        x_syn = autoencoder.decoder(z).clamp(-1.0, 1.0).cpu().numpy()  # [N,1,D,H,W]

    samples_npy = sample_dir / "generated_volumes.npy"
    np.save(samples_npy, x_syn)

    # Also save first sample as NIfTI for quick visual checks.
    first_vol = x_syn[0, 0]
    nib.save(nib.Nifti1Image(first_vol.astype(np.float32), affine=np.eye(4, dtype=np.float32)), str(sample_dir / "sample_000.nii.gz"))

    ae_ckpt = ckpt_dir / "autoencoder3d.pt"
    denoise_ckpt = ckpt_dir / "latent_denoiser3d.pt"
    torch.save(autoencoder.state_dict(), ae_ckpt)
    torch.save(denoiser.state_dict(), denoise_ckpt)

    summary_path = cfg.output_dir / "step3_summary.json"
    payload = {
        "step": 3,
        "name": "3D Latent Diffusion Model",
        "status": "complete",
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "device": str(device),
        "num_input_volumes": len(volumes),
        "train_volume_shape": list(cfg.volume_shape),
        "latent_shape": list(latent_shape),
        "autoencoder_epoch_losses": ae_losses,
        "diffusion_epoch_losses": diff_losses,
        "autoencoder_checkpoint": str(ae_ckpt),
        "denoiser_checkpoint": str(denoise_ckpt),
        "samples_npy": str(samples_npy),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
