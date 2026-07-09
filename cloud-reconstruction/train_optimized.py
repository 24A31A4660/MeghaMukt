#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING PIPELINE — Swin U-Net
Optimized for RTX 4050 (6 GB VRAM)

Features:
  - Swin U-Net (Swin Transformer Tiny encoder)
  - AdamW optimizer with weight decay
  - Cosine Annealing LR scheduler (3e-4 → 1e-6)
  - AMP (Automatic Mixed Precision) + GradScaler
  - Gradient accumulation (effective batch = 16)
  - Gradient clipping (max_norm = 1.0)
  - TF32 + cuDNN benchmark
  - Early stopping (patience 15)
  - Best model saving (best_model.pth)
  - Latest checkpoint saving (latest_checkpoint.pth)
  - Automatic resume from checkpoint
  - TensorBoard logging
  - PSNR, SSIM, RMSE, MAE, SAM per-epoch validation
"""

import os
import sys
import json
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from models.registry import build_model
from preprocessing.loader import build_dataloaders
from models.losses import CombinedLoss
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.logger import setup_logger


# LPIPS for perceptual quality metric during validation
try:
    import lpips as _lpips_lib
    _LPIPS_FN = _lpips_lib.LPIPS(net="alex", verbose=False)
    _LPIPS_AVAILABLE = True
except Exception:
    _LPIPS_FN = None
    _LPIPS_AVAILABLE = False


# ============================================================================
# TRAINING CONFIGURATION FOR RTX 4050
# ============================================================================

def get_optimal_config(base_config: dict, device: torch.device) -> dict:
    """
    Automatically optimize configuration for RTX 4050.

    Key settings:
    - Batch size: 4 (6GB VRAM constraint with Swin U-Net + 384×384 patches)
    - Gradient accumulation: 4 (effective batch = 16)
    - AMP: Enabled for memory efficiency
    - TF32: Enabled for 2-3x speedup on RTX 40xx
    - cuDNN benchmark: Enabled
    - Optimizer: AdamW with weight decay
    - Scheduler: CosineAnnealingLR
    """
    config = base_config.copy()

    # GPU OPTIMIZATIONS
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Ensure training section exists
    if "training" not in config:
        config["training"] = {}

    # Use config values as defaults but don't override what's already set
    config["training"].setdefault("batch_size", 4)
    config["training"].setdefault("gradient_accumulation", 4)
    config["training"].setdefault("num_workers", 2)
    config["training"].setdefault("pin_memory", True)
    config["training"].setdefault("persistent_workers", True)
    config["training"].setdefault("prefetch_factor", 2)
    config["training"].setdefault("amp", True)
    config["training"].setdefault("grad_clip", 1.0)
    config["training"].setdefault("weight_decay", 1e-5)
    config["training"].setdefault("learning_rate", 3e-4)
    config["training"].setdefault("early_stopping_patience", 15)
    config["training"].setdefault("resume", True)

    return config


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, RMSE, MAE, SAM, LPIPS metrics."""
    from evaluation.metrics import compute_psnr, compute_ssim, compute_sam

    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Per-sample metrics
    psnr_vals = []
    ssim_vals = []
    sam_vals = []
    for i in range(pred_np.shape[0]):
        p_hwc = np.transpose(pred_np[i], (1, 2, 0))
        t_hwc = np.transpose(target_np[i], (1, 2, 0))
        psnr_vals.append(compute_psnr(p_hwc, t_hwc))
        ssim_vals.append(compute_ssim(p_hwc, t_hwc))
        sam_vals.append(compute_sam(p_hwc, t_hwc))

    # Batch-level metrics
    mse = np.mean((pred_np - target_np) ** 2, axis=(1, 2, 3))
    rmse_val = float(np.sqrt(np.mean(mse)))
    mae_val = float(np.mean(np.abs(pred_np - target_np)))

    # LPIPS — uses RGB subset (bands 2,1,0)
    lpips_val = 0.0
    if _LPIPS_AVAILABLE and _LPIPS_FN is not None:
        try:
            # Extract RGB: B04(R)=ch2, B03(G)=ch1, B02(B)=ch0
            pred_rgb   = pred[:, [2, 1, 0]]   # [B, 3, H, W]
            target_rgb = target[:, [2, 1, 0]]
            # LPIPS expects [-1, 1]; our images are [0, 1]
            p_scaled = pred_rgb   * 2 - 1
            t_scaled = target_rgb * 2 - 1
            _fn = _LPIPS_FN.to(pred.device)
            with torch.no_grad():
                lp = _fn(p_scaled, t_scaled)  # [B, 1, 1, 1]
            lpips_val = float(lp.mean().item())
        except Exception:
            lpips_val = 0.0

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "rmse": rmse_val,
        "mae": mae_val,
        "sam": float(np.mean(sam_vals)),
        "lpips": lpips_val,
    }


def get_gpu_memory_usage(device: torch.device) -> float:
    """Get GPU memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0.0


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, config,
                scaler, log, grad_accum_steps: int = 4):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        n_bands = config["model"]["output_channels"]
        optical = cloudy[:, :n_bands]
        mask_ch = cloudy[:, n_bands:]

        # Forward pass with AMP
        with autocast("cuda", enabled=config["training"]["amp"] and device.type == "cuda"):
            pred = model(optical, mask_ch)
            loss, loss_components = loss_fn(pred, clear, mask)
            loss = loss / grad_accum_steps  # Scale for accumulation

        # ── NaN / Inf guard ──────────────────────────────────────────────────
        if torch.isnan(loss) or torch.isinf(loss):
            log.error(
                "CRITICAL: NaN/Inf loss at batch %d! "
                "Components: %s. Skipping batch.",
                batch_idx, loss_components
            )
            optimizer.zero_grad(set_to_none=True)
            continue
        # ─────────────────────────────────────────────────────────────────────

        scaler.scale(loss).backward()

        # Step optimizer every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            if config["training"]["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps  # Unscale for logging
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            log.info("  Batch %d/%d - Loss: %.4f", batch_idx + 1, len(train_loader),
                     loss.item() * grad_accum_steps)

    return total_loss / max(num_batches, 1)


def validate_epoch(model, val_loader, loss_fn, device, config):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            n_bands = config["model"]["output_channels"]
            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            with autocast("cuda", enabled=config["training"]["amp"] and device.type == "cuda"):
                pred = model(optical, mask_ch)
                loss, _ = loss_fn(pred, clear, mask)

            # Skip NaN validation batches gracefully
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()

            metrics = compute_metrics(pred, clear, mask)
            for key, val in metrics.items():
                all_metrics[key].append(val)

    avg_metrics = {
        key: float(np.mean(vals)) for key, vals in all_metrics.items()
    }

    n_batches = max(len(val_loader), 1)
    return total_loss / n_batches, avg_metrics


def save_visual_panels(model, val_loader, device, config, epoch: int, output_dir: Path):
    """Save comparison panels for up to 4 validation samples."""
    try:
        from evaluation.visualizer import create_comparison_panel
        model.eval()
        panel_dir = output_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        n_bands = config["model"]["output_channels"]

        with torch.no_grad():
            batch = next(iter(val_loader))
            cloudy  = batch["cloudy"].to(device)
            clear   = batch["clear"].to(device)
            mask    = batch["mask"].to(device)
            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]
            pred = model(optical, mask_ch)

        # Save up to 4 panels
        for i in range(min(4, cloudy.shape[0])):
            panel_path = panel_dir / f"epoch_{epoch+1:04d}_sample_{i}.png"
            create_comparison_panel(
                cloudy       = cloudy[i].cpu().numpy(),
                mask         = mask[i].cpu().numpy(),
                pred         = pred[i].cpu().numpy(),
                ground_truth = clear[i].cpu().numpy(),
                save_path    = str(panel_path),
                epoch        = epoch + 1,
                sample       = i,
            )
    except Exception as e:
        pass  # Visual panels are best-effort; never crash training


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch

    # Create directories
    for key in ("checkpoints", "logs", "outputs", "tensorboard"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup logger
    log = setup_logger("train")
    log.info("=" * 80)
    log.info("SWIN U-NET TRAINING PIPELINE — RTX 4050 Optimized")
    log.info("=" * 80)
    log.info("Device: %s", device)

    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(device))
        log.info("VRAM: %.1f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)

    # Optimize configuration
    config = get_optimal_config(config, device)
    grad_accum = config["training"]["gradient_accumulation"]
    effective_batch = config["training"]["batch_size"] * grad_accum

    log.info("\nConfiguration:")
    log.info("  Model: %s", config["model"]["name"])
    log.info("  Batch size: %d (× %d accum = %d effective)",
             config["training"]["batch_size"], grad_accum, effective_batch)
    log.info("  Patch size: %d", config["patch"]["size"])
    log.info("  AMP: %s", config["training"]["amp"])
    log.info("  Grad Clip: %.1f", config["training"]["grad_clip"])
    log.info("  Learning Rate: %.2e", config["training"]["learning_rate"])
    log.info("  Epochs: %d", config["training"]["epochs"])

    # Load data
    log.info("\nLoading dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(config)

    log.info("  Train batches: %d", len(train_loader))
    log.info("  Validation batches: %d", len(val_loader) if val_loader else 0)

    # Build model
    log.info("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Model: %s", config["model"]["name"])
    log.info("  Total parameters: %.2fM", total_params / 1e6)
    log.info("  Trainable parameters: %.2fM", trainable_params / 1e6)

    # Setup training
    log.info("\nSetting up training...")

    loss_fn = CombinedLoss(
        w_l1=config["loss"]["l1"],
        w_ssim=config["loss"]["ssim"],
        w_perceptual=config["loss"]["perceptual"],
        w_charbonnier=config["loss"].get("charbonnier", 0.1),
        n_bands=config["model"]["output_channels"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6,
    )

    scaler = GradScaler() if config["training"]["amp"] else GradScaler(enabled=False)

    writer = SummaryWriter(config["paths"]["tensorboard"])

    # Resume from checkpoint
    start_epoch = 0
    # Best-model tracking — priority: SSIM (high) > PSNR (high) > LPIPS (low) > loss (low)
    # The model with the lowest loss is NOT necessarily the one that looks best visually.
    best_val_ssim  = -1.0
    best_val_psnr  = 0.0
    best_val_lpips = float("inf")
    best_val_loss  = float("inf")

    if config["training"]["resume"] and not args.no_resume:
        ckpt_dir = Path(config["paths"]["checkpoints"])
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            try:
                checkpoint = load_checkpoint(latest_ckpt, model, optimizer, scheduler, scaler,
                                             device=str(device))
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_val_ssim  = checkpoint.get("metrics", {}).get("val/ssim", -1.0)
                best_val_psnr  = checkpoint.get("metrics", {}).get("val/psnr", 0.0)
                best_val_lpips = checkpoint.get("metrics", {}).get("val/lpips", float("inf"))
                best_val_loss  = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
                log.info("Resumed from checkpoint: Epoch %d (val_loss=%.5f)",
                         start_epoch, best_val_loss)
            except Exception as e:
                log.warning("Failed to resume checkpoint: %s", e)

    # Training loop
    log.info("\n" + "=" * 80)
    log.info("TRAINING (AMP, TF32, Gradient Accumulation, Cosine Annealing)")
    log.info("=" * 80 + "\n")

    training_report = []
    early_stopping_count = 0
    start_time = time.time()
    diagnostic_report = None
    
    # Quality Gate trackers
    consecutive_val_loss_inc = 0
    consecutive_psnr_dec = 0
    consecutive_ssim_dec = 0
    nan_count = 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start = time.time()

        try:
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                     device, config, scaler, log, grad_accum)

            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device, config)
        except torch.cuda.OutOfMemoryError:
            log.error("CRITICAL: CUDA OutOfMemoryError during epoch %d. Pausing training.", epoch + 1)
            diagnostic_report = {"error": "CUDA OOM", "epoch": epoch + 1}
            break
        except Exception as e:
            log.error("CRITICAL: Exception during epoch %d: %s", epoch + 1, e)
            diagnostic_report = {"error": str(e), "epoch": epoch + 1}
            break

        # Check for NaN validation
        if np.isnan(val_loss):
            nan_count += 1
            if nan_count > 3:
                log.error("CRITICAL: NaN loss exceeded threshold. Pausing training.")
                diagnostic_report = {"error": "NaN loss", "epoch": epoch + 1}
                break
        else:
            nan_count = 0

        # Step scheduler
        scheduler.step()

        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining_epochs = config["training"]["epochs"] - epoch - 1
        eta_sec = (elapsed / max(1, epoch + 1 - start_epoch)) * remaining_epochs

        # Log to TensorBoard
        lr = optimizer.param_groups[0]["lr"]
        gpu_mem = get_gpu_memory_usage(device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/PSNR", val_metrics.get("psnr", 0), epoch)
        writer.add_scalar("Metrics/SSIM", val_metrics.get("ssim", 0), epoch)
        writer.add_scalar("Metrics/RMSE", val_metrics.get("rmse", 0), epoch)
        writer.add_scalar("Metrics/MAE",  val_metrics.get("mae",  0), epoch)
        writer.add_scalar("Metrics/SAM",  val_metrics.get("sam",  0), epoch)
        writer.add_scalar("Metrics/LPIPS", val_metrics.get("lpips", 0), epoch)
        writer.add_scalar("Training/LR", lr, epoch)
        writer.add_scalar("Training/GPU_Memory_MB", gpu_mem, epoch)

        # Console output
        log.info(
            "Epoch %3d/%d | Loss: %.4f/%.4f | PSNR: %.2f | SSIM: %.4f | "
            "RMSE: %.4f | MAE: %.4f | SAM: %.2f° | LR: %.2e | GPU: %.0fMB | "
            "Time: %.1fs | ETA: %.1fh",
            epoch + 1, config["training"]["epochs"],
            train_loss, val_loss,
            val_metrics.get("psnr", 0), val_metrics.get("ssim", 0),
            val_metrics.get("rmse", 0), val_metrics.get("mae", 0),
            val_metrics.get("sam", 0),
            lr, gpu_mem, epoch_time,
            eta_sec / 3600 if eta_sec > 0 else 0,
        )

        # Report
        report_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "psnr": round(val_metrics.get("psnr", 0), 4),
            "ssim": round(val_metrics.get("ssim", 0), 4),
            "rmse": round(val_metrics.get("rmse", 0), 4),
            "mae":  round(val_metrics.get("mae",  0), 4),
            "sam":  round(val_metrics.get("sam",  0), 4),
            "lpips": round(val_metrics.get("lpips", 0), 4),
            "lr": lr,
            "gpu_memory_mb": round(gpu_mem, 1),
            "epoch_time_sec": round(epoch_time, 2),
            "eta_hours": round(eta_sec / 3600, 2) if eta_sec > 0 else 0,
        }
        training_report.append(report_entry)
        report_path = Path(config["paths"]["outputs"]) / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(training_report, f, indent=2)

        # ── Best model selection ──────────────────────────────────────────
        # Priority: SSIM (high) > PSNR (high) > LPIPS (low) > val_loss (low)
        # Rationale: loss convergence does not always equal perceptual quality.
        # ─────────────────────────────────────────────────────────────────────
        cur_ssim  = val_metrics.get("ssim",  0.0)
        cur_psnr  = val_metrics.get("psnr",  0.0)
        cur_lpips = val_metrics.get("lpips", float("inf"))

        is_best_ssim  = cur_ssim  > best_val_ssim  + 1e-4
        is_best_psnr  = cur_psnr  > best_val_psnr  + 0.01

        is_best = False
        if cur_ssim > best_val_ssim + 1e-4:
            # Primary: strictly better SSIM
            is_best = True
        elif abs(cur_ssim - best_val_ssim) < 1e-4:
            if cur_psnr > best_val_psnr + 0.01:
                # Secondary: same SSIM, better PSNR (≥0.01 dB)
                is_best = True
            elif abs(cur_psnr - best_val_psnr) < 0.01:
                if cur_lpips < best_val_lpips - 1e-4:
                    # Tertiary: same SSIM+PSNR, better LPIPS
                    is_best = True
                elif abs(cur_lpips - best_val_lpips) < 1e-4 and val_loss < best_val_loss - 1e-6:
                    # Last resort: all perceptual metrics tied, lower loss
                    is_best = True

        if is_best:
            best_val_ssim  = cur_ssim
            best_val_psnr  = cur_psnr
            best_val_lpips = cur_lpips
            best_val_loss  = val_loss

        ckpt_dir = Path(config["paths"]["checkpoints"])
        ckpt_meta = {
            "val_loss": val_loss,
            **{f"val/{k}": v for k, v in val_metrics.items()}
        }

        # ── Save best_model.pth (primary composite best) ──
        if is_best:
            save_checkpoint(
                ckpt_dir / "best_model.pth",
                epoch, model, optimizer, scheduler,
                metrics=ckpt_meta, config=config,
                scaler_state_dict=scaler.state_dict(),
            )
            log.info("  >> SAVED best_model.pth  SSIM=%.4f  PSNR=%.2f dB  LPIPS=%.4f  loss=%.5f",
                     best_val_ssim, best_val_psnr, best_val_lpips, best_val_loss)

        # ── Save best_ssim.pth (best SSIM independently) ──
        if is_best_ssim:
            save_checkpoint(
                ckpt_dir / "best_ssim.pth",
                epoch, model, optimizer, scheduler,
                metrics=ckpt_meta, config=config,
                scaler_state_dict=scaler.state_dict(),
            )
            log.info("  >> SAVED best_ssim.pth  SSIM=%.4f", cur_ssim)

        # ── Save best_psnr.pth (best PSNR independently) ──
        if is_best_psnr:
            save_checkpoint(
                ckpt_dir / "best_psnr.pth",
                epoch, model, optimizer, scheduler,
                metrics=ckpt_meta, config=config,
                scaler_state_dict=scaler.state_dict(),
            )
            log.info("  >> SAVED best_psnr.pth  PSNR=%.2f dB", cur_psnr)

        # ── Always save latest ──
        save_checkpoint(
            ckpt_dir / "latest_checkpoint.pth",
            epoch, model, optimizer, scheduler,
            metrics=ckpt_meta, config=config,
            scaler_state_dict=scaler.state_dict(),
        )

        # ── Periodic epoch_XXXX.pth every checkpoint_interval epochs ──
        ckpt_interval = config["training"].get("checkpoint_interval", 5)
        if (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                epoch, model, optimizer, scheduler,
                metrics=ckpt_meta, config=config,
                scaler_state_dict=scaler.state_dict(),
            )
            log.info("  Saved periodic checkpoint: epoch_%04d.pth", epoch + 1)

        # ── Visual comparison panels every checkpoint_interval epochs ──
        if (epoch + 1) % ckpt_interval == 0 or epoch == 0:
            save_visual_panels(
                model, val_loader, device, config, epoch,
                Path(config["paths"]["outputs"]),
            )
            log.info("  Saved comparison panels: outputs/panels/epoch_%04d_*.png", epoch + 1)

        # ── Quality Gates Check ──────────────────────────────────────────
        # Check for catastrophic failures only (NaN/Inf checked previously, OOM checked previously)
        q_fail = None

        if q_fail:
            log.error("QUALITY GATE FAILED: %s", q_fail)
            diagnostic_report = {"error": f"Quality Gate Failed: {q_fail}", "epoch": epoch + 1}
            break

        # Early stopping
        if is_best:
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            if early_stopping_count >= config["training"]["early_stopping_patience"]:
                log.info("\nEarly stopping triggered (patience: %d epochs)",
                         config["training"]["early_stopping_patience"])
                break
    else:
        diagnostic_report = None

    if diagnostic_report:
        diag_path = Path(config["paths"]["outputs"]) / "diagnostic_report.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic_report, f, indent=2)
        log.error("Training paused due to Quality Gate or Error. Diagnostic report saved to %s", diag_path)

    # Final report
    total_time = time.time() - start_time
    log.info("\n" + "=" * 80)
    log.info("TRAINING COMPLETE")
    log.info("=" * 80)
    log.info("  Total time: %.2f hours", total_time / 3600)
    log.info("  Best SSIM:       %.4f",  best_val_ssim)
    log.info("  Best PSNR:       %.2f dB", best_val_psnr)
    log.info("  Best LPIPS:      %.4f",  best_val_lpips)
    log.info("  Best val loss:   %.6f", best_val_loss)

    final_report = {
        "training_date": datetime.now().isoformat(),
        "total_time_hours": round(total_time / 3600, 2),
        "model": config["model"]["name"],
        "best_ssim":     float(best_val_ssim),
        "best_psnr":     float(best_val_psnr),
        "best_lpips":    float(best_val_lpips),
        "best_val_loss": float(best_val_loss),
        "config": {
            "batch_size": config["training"]["batch_size"],
            "gradient_accumulation": config["training"]["gradient_accumulation"],
            "effective_batch": config["training"]["batch_size"] * config["training"]["gradient_accumulation"],
            "learning_rate": config["training"]["learning_rate"],
            "patch_size": config["patch"]["size"],
            "amp": config["training"]["amp"],
            "scheduler": "CosineAnnealingLR",
        },
    }

    final_path = Path(config["paths"]["outputs"]) / "final_report.json"
    with open(final_path, "w") as f:
        json.dump(final_report, f, indent=2)

    log.info("\nFinal report: %s", final_path)
    log.info("Best model:   %s", ckpt_dir / "best_model.pth")
    log.info("Latest:       %s", ckpt_dir / "latest_checkpoint.pth")


if __name__ == "__main__":
    main()
