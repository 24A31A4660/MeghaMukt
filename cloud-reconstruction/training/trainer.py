"""training/trainer.py — Full training loop with AMP, gradient accumulation, and TensorBoard."""
from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_psnr, compute_ssim, compute_sam
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

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
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
    - Gradient accumulation for effective larger batch sizes
    - Gradient clipping
    - Cosine Annealing LR scheduler
    - Early stopping
    - Best + periodic checkpoint saving (best_model.pth, latest_checkpoint.pth)
    - Resume from checkpoint
    - TensorBoard logging
    - Per-epoch PSNR/SSIM/SAM tracking
    """

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Enable cuDNN auto-tuner and TF32 on CUDA devices for faster training.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

        # Move model to device
        self.model = model.to(device)

        tcfg = cfg["training"]
        lcfg = cfg["loss"]

        # Optimizer — AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
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
            w_charbonnier=lcfg.get("charbonnier", 0.1),
            n_bands=cfg["model"]["output_channels"],
        ).to(device)

        # AMP
        self.use_amp = tcfg["amp"] and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = tcfg.get("gradient_accumulation", 1)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=tcfg["early_stopping_patience"])

        # Paths
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb = TBLogger(Path(cfg["paths"]["tensorboard"]))

        # State
        self.start_epoch = 0
        # Best-model tracking — priority: SSIM (high) > PSNR (high) > LPIPS (low) > loss (low)
        self.best_val_ssim  = -1.0
        self.best_val_psnr  = 0.0
        self.best_val_lpips = float("inf")
        self.best_val_loss  = float("inf")
        self.global_step = 0
        self.report_path = Path(cfg["paths"]["outputs"]) / "training_report.json"
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.report_path.exists():
            self.report_path.write_text("[]", encoding="utf-8")

    def resume_if_exists(self) -> None:
        """Load the latest checkpoint if it exists."""
        ckpt = find_latest_checkpoint(self.ckpt_dir)
        if ckpt is not None:
            log.info("Resuming from checkpoint: %s", ckpt.name)
            payload = load_checkpoint(
                ckpt,
                self.model,
                self.optimizer,
                self.scheduler,
                scaler=self.scaler,
                device=str(self.device),
            )
            self.start_epoch    = payload.get("epoch", 0) + 1
            self.best_val_ssim  = payload.get("metrics", {}).get("val/ssim",  -1.0)
            self.best_val_psnr  = payload.get("metrics", {}).get("val/psnr",   0.0)
            self.best_val_lpips = payload.get("metrics", {}).get("val/lpips", float("inf"))
            self.best_val_loss  = payload.get("metrics", {}).get("val_loss",  float("inf"))
            log.info("Resumed at epoch %d (best SSIM=%.4f, PSNR=%.2f)",
                     self.start_epoch, self.best_val_ssim, self.best_val_psnr)
        else:
            log.info("No checkpoint found. Starting fresh.")

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch with gradient accumulation. Returns average metrics dict."""
        self.model.train()
        running: dict[str, list[float]] = {}
        n_bands = self.cfg["model"]["output_channels"]

        pbar = tqdm(
            loader, desc=f"Epoch {epoch:03d} [Train]",
            unit="batch", dynamic_ncols=True, leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            cloudy = batch["cloudy"].to(self.device, non_blocking=True)
            clear  = batch["clear"].to(self.device, non_blocking=True)
            mask   = batch["mask"].to(self.device, non_blocking=True)

            # Split cloudy into optical bands and cloud mask channel
            optical = cloudy[:, :n_bands]                  # [B, 6, H, W]
            mask_ch = cloudy[:, n_bands:]                  # [B, 1, H, W]

            with autocast("cuda", enabled=self.use_amp):
                pred = self.model(optical, mask_ch)        # [B, 6, H, W]
                loss, components = self.criterion(pred, clear, mask)
                # Scale loss by accumulation steps
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            # Step optimizer every grad_accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                # Gradient clipping
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
                self.optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics (unscaled loss for logging)
            for k, v in components.items():
                running.setdefault(k, []).append(v)

            # TensorBoard per-step logging
            self.tb.scalar("train/loss_step", components["loss/total"], self.global_step)
            self.global_step += 1

            pbar.set_postfix(loss=f"{components['loss/total']:.4f}")

        # Average over epoch
        avg = {k: float(np.mean(v)) for k, v in running.items()}
        return avg

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Validate for one epoch and save monitoring visuals plus richer metrics."""
        import torchvision.utils as vutils

        self.model.eval()
        running: dict[str, list[float]] = {}
        psnr_list: list[float] = []
        ssim_list: list[float] = []
        rmse_list: list[float] = []
        mae_list: list[float] = []
        sam_list: list[float] = []
        infer_ms_list: list[float] = []
        n_bands = self.cfg["model"]["output_channels"]

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch:03d} [Val]  ",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        mon_dir = Path(self.cfg["paths"]["outputs"]) / "monitoring"
        mon_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(pbar):
            cloudy = batch["cloudy"].to(self.device, non_blocking=True)
            clear = batch["clear"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.time()
            with autocast("cuda", enabled=self.use_amp):
                pred = self.model(optical, mask_ch)
                loss, components = self.criterion(pred, clear, mask)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_ms = (time.time() - t0) * 1000.0

            for k, v in components.items():
                running.setdefault(k, []).append(v)

            pred_cpu = pred.detach().cpu()
            clear_cpu = clear.detach().cpu()
            for sample_idx in range(pred_cpu.shape[0]):
                p = pred_cpu[sample_idx].numpy()
                t = clear_cpu[sample_idx].numpy()
                p_hwc = np.transpose(p, (1, 2, 0))
                t_hwc = np.transpose(t, (1, 2, 0))
                psnr_list.append(compute_psnr(p_hwc, t_hwc))
                ssim_list.append(compute_ssim(p_hwc, t_hwc))
                rmse_list.append(float(np.sqrt(np.mean((p_hwc - t_hwc) ** 2))))
                mae_list.append(float(np.mean(np.abs(p_hwc - t_hwc))))
                sam_list.append(compute_sam(p_hwc, t_hwc))
            infer_ms_list.append(infer_ms / max(1, pred_cpu.shape[0]))

            # Save a compact monitoring panel for the first sample of each epoch.
            if batch_idx == 0:
                c_rgb = torch.clamp(optical[0, [2, 1, 0]], 0, 1).cpu()
                p_rgb = torch.clamp(pred[0, [2, 1, 0]], 0, 1).cpu()
                t_rgb = torch.clamp(clear[0, [2, 1, 0]], 0, 1).cpu()
                diff = torch.abs(pred[0].cpu() - clear[0].cpu()).mean(dim=0, keepdim=True)
                diff = torch.clamp(diff, 0, 1)
                mask_rgb = mask[0].unsqueeze(0).repeat(3, 1, 1).float().cpu()

                vutils.save_image(c_rgb, mon_dir / f"epoch_{epoch + 1:03d}_cloudy.png")
                vutils.save_image(mask_rgb, mon_dir / f"epoch_{epoch + 1:03d}_mask.png")
                vutils.save_image(p_rgb, mon_dir / f"epoch_{epoch + 1:03d}_pred.png")
                vutils.save_image(t_rgb, mon_dir / f"epoch_{epoch + 1:03d}_target.png")
                vutils.save_image(diff.repeat(3, 1, 1), mon_dir / f"epoch_{epoch + 1:03d}_diff.png")

                self.tb.image("val_monitor/cloudy", c_rgb, epoch)
                self.tb.image("val_monitor/mask", mask_rgb, epoch)
                self.tb.image("val_monitor/pred", p_rgb, epoch)
                self.tb.image("val_monitor/target", t_rgb, epoch)
                self.tb.image("val_monitor/diff", diff.repeat(3, 1, 1), epoch)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = {k: float(np.mean(v)) for k, v in running.items()}
        avg["val/psnr"] = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg["val/ssim"] = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg["val/rmse"] = float(np.mean(rmse_list)) if rmse_list else 0.0
        avg["val/mae"] = float(np.mean(mae_list)) if mae_list else 0.0
        avg["val/sam"] = float(np.mean(sam_list)) if sam_list else 0.0
        avg["val/inference_ms"] = float(np.mean(infer_ms_list)) if infer_ms_list else 0.0
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
        log.info("Device: %s | AMP: %s | Grad Accum: %d", self.device, self.use_amp, self.grad_accum_steps)
        log.info("Effective batch size: %d", tcfg["batch_size"] * self.grad_accum_steps)
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
            remaining_epochs = max(0, total_epochs - (epoch + 1))
            eta_seconds = elapsed * remaining_epochs
            eta_text = str(timedelta(seconds=int(eta_seconds)))

            gpu_mem = 0.0
            if self.device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats(self.device)

            # Log summary
            log.info(
                "Epoch %03d | Train: %.4f | Val: %.4f | PSNR: %.2f | SSIM: %.4f | "
                "RMSE: %.4f | MAE: %.4f | SAM: %.2f° | "
                "LR: %.6f | Grad: %.2f | GPU: %dMB | %.1fs | ETA: %s",
                epoch + 1,
                train_metrics.get("loss/total", 0),
                val_loss,
                val_metrics.get("val/psnr", 0),
                val_metrics.get("val/ssim", 0),
                val_metrics.get("val/rmse", 0),
                val_metrics.get("val/mae", 0),
                val_metrics.get("val/sam", 0),
                lr,
                train_metrics.get("grad_norm", 0),
                int(gpu_mem),
                elapsed,
                eta_text,
            )

            report_entry = {
                "epoch": epoch + 1,
                "train_loss": train_metrics.get("loss/total", 0),
                "val_loss": val_loss,
                "psnr": val_metrics.get("val/psnr", 0),
                "ssim": val_metrics.get("val/ssim", 0),
                "rmse": val_metrics.get("val/rmse", 0),
                "mae": val_metrics.get("val/mae", 0),
                "sam": val_metrics.get("val/sam", 0),
                "lr": lr,
                "epoch_time_seconds": round(elapsed, 3),
                "eta_seconds": round(eta_seconds, 3),
                "eta_hms": eta_text,
            }
            history = json.loads(self.report_path.read_text(encoding="utf-8"))
            history.append(report_entry)
            self.report_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            # ── Best model selection ──────────────────────────────────────────
            # Priority: SSIM (high) > PSNR (high) > LPIPS (low) > val_loss (low)
            # The model with the lowest loss is NOT always the best visual result.
            # ─────────────────────────────────────────────────────────────────────
            cur_ssim  = val_metrics.get("val/ssim",  0.0)
            cur_psnr  = val_metrics.get("val/psnr",  0.0)
            cur_lpips = val_metrics.get("val/lpips", float("inf"))

            should_save_best = False
            if cur_ssim > self.best_val_ssim + 1e-4:
                should_save_best = True
            elif abs(cur_ssim - self.best_val_ssim) < 1e-4:
                if cur_psnr > self.best_val_psnr + 0.01:
                    should_save_best = True
                elif abs(cur_psnr - self.best_val_psnr) < 0.01:
                    if cur_lpips < self.best_val_lpips - 1e-4:
                        should_save_best = True
                    elif abs(cur_lpips - self.best_val_lpips) < 1e-4 and val_loss < self.best_val_loss - 1e-6:
                        should_save_best = True

            if should_save_best:
                self.best_val_ssim  = cur_ssim
                self.best_val_psnr  = cur_psnr
                self.best_val_lpips = cur_lpips
                self.best_val_loss  = val_loss
                save_checkpoint(
                    self.ckpt_dir / "best_model.pth",
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    metrics={"val_loss": val_loss, **val_metrics},
                    config=self.cfg,
                    scaler_state_dict=self.scaler.state_dict(),
                )
                log.info(
                    "  >> Saved best_model.pth  SSIM=%.4f  PSNR=%.2f dB  LPIPS=%.4f  loss=%.5f",
                    self.best_val_ssim, self.best_val_psnr, self.best_val_lpips, self.best_val_loss,
                )

            # Always save latest checkpoint
            save_checkpoint(
                self.ckpt_dir / "latest_checkpoint.pth",
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                metrics={"val_loss": val_loss, **val_metrics},
                config=self.cfg,
                scaler_state_dict=self.scaler.state_dict(),
            )

            # Periodic checkpoint
            if (epoch + 1) % tcfg.get("checkpoint_interval", 5) == 0:
                save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch:04d}.pth",
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    metrics={"val_loss": val_loss, **val_metrics},
                    config=self.cfg,
                    scaler_state_dict=self.scaler.state_dict(),
                )

            # Early stopping
            if val_loader is not None and self.early_stopping.step(val_loss):
                log.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch, self.early_stopping.patience,
                )
                break

        self.tb.close()
        log.info("Training complete. Best val_loss=%.5f, SSIM=%.4f, PSNR=%.2f",
                 self.best_val_loss, self.best_val_ssim, self.best_val_psnr)
