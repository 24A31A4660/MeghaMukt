"""training/trainer.py — Full training loop with AMP, early stopping, and TensorBoard."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_psnr, compute_ssim
from models.losses import CombinedLoss
from utils.checkpoint import (
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.logger import TBLogger, setup_logger

log = setup_logger("trainer")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Cloud reconstruction training loop.

    Features:
    - Automatic Mixed Precision (AMP) with GradScaler
    - Gradient clipping
    - Early stopping
    - Best + periodic checkpoint saving
    - Resume from checkpoint
    - TensorBoard logging
    - Per-epoch PSNR/SSIM tracking
    """

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Enable cuDNN auto-tuner for max conv performance
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Move model to device
        self.model = model.to(device)

        tcfg = cfg["training"]
        lcfg = cfg["loss"]

        # Optimiser
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )

        # LR Scheduler — cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["epochs"],
            eta_min=tcfg["learning_rate"] * 0.01,
        )

        # Loss
        self.criterion = CombinedLoss(
            w_l1=lcfg["l1"],
            w_ssim=lcfg["ssim"],
            w_perceptual=lcfg["perceptual"],
            w_edge=lcfg["edge"],
            n_bands=cfg["model"]["output_channels"],
        ).to(device)

        # AMP
        self.use_amp = tcfg["amp"] and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=tcfg["early_stopping_patience"])

        # Paths
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb = TBLogger(Path(cfg["paths"]["tensorboard"]))

        # State
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0

    def resume_if_exists(self) -> None:
        """Load the latest checkpoint if it exists."""
        ckpt = find_latest_checkpoint(self.ckpt_dir)
        if ckpt is not None:
            log.info("Resuming from checkpoint: %s", ckpt.name)
            payload = load_checkpoint(
                ckpt, self.model, self.optimizer, self.scheduler,
                device=str(self.device),
            )
            self.start_epoch = payload["epoch"] + 1
            self.best_val_loss = payload.get("metrics", {}).get("val_loss", float("inf"))
            log.info("Resumed at epoch %d (best_val_loss=%.6f)", self.start_epoch, self.best_val_loss)
        else:
            log.info("No checkpoint found. Starting fresh.")

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch. Returns average metrics dict."""
        self.model.train()
        running: dict[str, list[float]] = {}
        n_bands = self.cfg["model"]["output_channels"]

        pbar = tqdm(
            loader, desc=f"Epoch {epoch:03d} [Train]",
            unit="batch", dynamic_ncols=True, leave=False,
        )

        for batch in pbar:
            cloudy = batch["cloudy"].to(self.device)       # [B, C_in, H, W]
            clear  = batch["clear"].to(self.device)        # [B, C_out, H, W]
            mask   = batch["mask"].to(self.device)         # [B, H, W]

            # Split cloudy into optical bands and cloud mask channel
            optical = cloudy[:, :n_bands]                  # [B, 6, H, W]
            mask_ch = cloudy[:, n_bands:]                  # [B, 1, H, W]

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                pred = self.model(optical, mask_ch)        # [B, 6, H, W]
                loss, components = self.criterion(pred, clear, mask)

            self.scaler.scale(loss).backward()

            # Gradient clipping and norm monitoring
            self.scaler.unscale_(self.optimizer)
            if self.cfg["training"]["grad_clip"] > 0:
                norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg["training"]["grad_clip"],
                )
            else:
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
            
            grad_norm = norm.item() if isinstance(norm, torch.Tensor) else norm
            running.setdefault("grad_norm", []).append(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            for k, v in components.items():
                running.setdefault(k, []).append(v)

            # TensorBoard per-step logging
            self.tb.scalar("train/loss_step", loss.item(), self.global_step)
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Average over epoch
        avg = {k: float(np.mean(v)) for k, v in running.items()}
        return avg

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Validate for one epoch. Returns average metrics dict."""
        import torchvision.utils as vutils
        self.model.eval()
        running: dict[str, list[float]] = {}
        psnr_list: list[float] = []
        ssim_list: list[float] = []
        n_bands = self.cfg["model"]["output_channels"]
        first_batch = True

        pbar = tqdm(
            loader, desc=f"Epoch {epoch:03d} [Val]  ",
            unit="batch", dynamic_ncols=True, leave=False,
        )

        for batch in pbar:
            cloudy = batch["cloudy"].to(self.device)
            clear  = batch["clear"].to(self.device)
            mask   = batch["mask"].to(self.device)

            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            with autocast(enabled=self.use_amp):
                pred = self.model(optical, mask_ch)
                loss, components = self.criterion(pred, clear, mask)

            for k, v in components.items():
                running.setdefault(k, []).append(v)

            # Save cloudy, prediction, ground truth every 5 epochs
            if first_batch and (epoch + 1) % 5 == 0:
                mon_dir = Path(self.cfg["paths"]["outputs"]) / "monitoring"
                mon_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract RGB (B04, B03, B02 -> indices 2, 1, 0)
                c_rgb = torch.clamp(optical[0, [2, 1, 0]], 0, 1).cpu()
                p_rgb = torch.clamp(pred[0, [2, 1, 0]], 0, 1).cpu()
                t_rgb = torch.clamp(clear[0, [2, 1, 0]], 0, 1).cpu()
                
                vutils.save_image(c_rgb, mon_dir / f"epoch_{epoch+1:03d}_cloudy.png")
                vutils.save_image(p_rgb, mon_dir / f"epoch_{epoch+1:03d}_pred.png")
                vutils.save_image(t_rgb, mon_dir / f"epoch_{epoch+1:03d}_target.png")
                
                self.tb.image("val_monitor/cloudy", c_rgb, epoch)
                self.tb.image("val_monitor/pred", p_rgb, epoch)
                self.tb.image("val_monitor/target", t_rgb, epoch)
                first_batch = False

            # PSNR/SSIM on first sample in batch
            p = pred[0].cpu().numpy()           # [C, H, W]
            t = clear[0].cpu().numpy()
            p_hwc = np.transpose(p, (1, 2, 0))
            t_hwc = np.transpose(t, (1, 2, 0))
            psnr_list.append(compute_psnr(p_hwc, t_hwc))
            ssim_list.append(compute_ssim(p_hwc, t_hwc))

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = {k: float(np.mean(v)) for k, v in running.items()}
        avg["val/psnr"] = float(np.mean(psnr_list))
        avg["val/ssim"] = float(np.mean(ssim_list))
        return avg

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Full training loop."""
        tcfg = self.cfg["training"]
        total_epochs = tcfg["epochs"]

        if tcfg.get("resume", False):
            self.resume_if_exists()

        log.info("Model parameters: %s", f"{self.model.count_parameters():,}")
        log.info("Device: %s | AMP: %s", self.device, self.use_amp)
        log.info("Starting training for %d epochs (from epoch %d)", total_epochs, self.start_epoch)

        for epoch in range(self.start_epoch, total_epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            for k, v in train_metrics.items():
                self.tb.scalar(f"train/{k}", v, epoch)

            # Validate
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.val_epoch(val_loader, epoch)
                for k, v in val_metrics.items():
                    self.tb.scalar(f"val/{k}", v, epoch)

            # Learning rate
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb.scalar("train/lr", lr, epoch)

            elapsed = time.time() - t0
            val_loss = val_metrics.get("loss/total", train_metrics.get("loss/total", 0))

            gpu_mem = 0.0
            if self.device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats(self.device)

            # Log summary
            log.info(
                "Epoch %03d | Train: %.4f | Val: %.4f | L1: %.4f | SSIM: %.4f | Perc: %.4f | Edge: %.4f | "
                "LR: %.6f | Grad: %.2f | GPU: %dMB | %.1fs",
                epoch + 1,
                train_metrics.get("loss/total", 0),
                val_loss,
                val_metrics.get("loss/l1", train_metrics.get("loss/l1", 0)),
                val_metrics.get("loss/ssim", train_metrics.get("loss/ssim", 0)),
                val_metrics.get("loss/perceptual", train_metrics.get("loss/perceptual", 0)),
                val_metrics.get("loss/edge", train_metrics.get("loss/edge", 0)),
                lr,
                train_metrics.get("grad_norm", 0),
                int(gpu_mem),
                elapsed,
            )

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.ckpt_dir / "best.pth",
                    epoch, self.model, self.optimizer, self.scheduler,
                    metrics={"val_loss": val_loss, **val_metrics},
                    config=self.cfg,
                )
                log.info("  >> Saved best model (val_loss=%.5f)", val_loss)

            # Periodic checkpoint
            if (epoch + 1) % tcfg.get("checkpoint_interval", 5) == 0:
                save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch:04d}.pth",
                    epoch, self.model, self.optimizer, self.scheduler,
                    metrics={"val_loss": val_loss, **val_metrics},
                    config=self.cfg,
                )

            # Early stopping
            if val_loader is not None and self.early_stopping.step(val_loss):
                log.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch, self.early_stopping.patience,
                )
                break

        self.tb.close()
        log.info("Training complete. Best val_loss=%.5f", self.best_val_loss)
