#!/usr/bin/env python3
"""
overfit_test.py -- Verify the FULL PIPELINE works end-to-end.

Uses 50-100 image pairs and trains for 30-50 epochs.
A healthy model should nearly memorise these pairs:
    SSIM  > 0.95
    PSNR  > 35 dB
    LPIPS ~= 0

If it cannot overfit 100 images, there is still a bug somewhere
(data loading, preprocessing, normalization, band order, loss, etc.).

Do NOT start a 200-epoch full training until this passes.

Usage:
    python overfit_test.py                     # default: 80 pairs, 40 epochs
    python overfit_test.py --n-samples 50 --epochs 50
    python overfit_test.py --patch-size 256    # smaller patch for faster test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Overfit test for Swin U-Net pipeline.")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--n-samples",   type=int,   default=80,    help="Number of pairs to overfit")
    p.add_argument("--epochs",      type=int,   default=40,    help="Number of epochs to train")
    p.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate (higher helps overfit faster)")
    p.add_argument("--batch-size",  type=int,   default=4,     help="Batch size")
    p.add_argument("--patch-size",  type=int,   default=None,  help="Override patch size (e.g. 256 for speed)")
    p.add_argument("--no-pretrain", action="store_true",       help="Skip pretrained weights")
    p.add_argument("--save-visuals",action="store_true",       help="Save comparison images every 10 epochs")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = {
    "ssim": (">=", 0.95, "higher is better"),
    "psnr": (">=", 35.0, "dB, higher is better"),
    "rmse": ("<=", 0.05, "lower is better"),
}


def check_target(name, value):
    if name not in TARGETS:
        return None
    op, threshold, note = TARGETS[name]
    if op == ">=":
        passed = value >= threshold
    else:
        passed = value <= threshold
    status = "[PASS]" if passed else "[FAIL]"
    return f"  {status} {name.upper()}: {value:.4f} {op} {threshold} ({note})"


# ─────────────────────────────────────────────────────────────────────────────
# Data verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_batch(batch, n_bands=6):
    """Verify that cloudy/clear/mask are correctly paired and normalised."""
    cloudy = batch["cloudy"]
    clear  = batch["clear"]
    mask   = batch["mask"]

    print("\n  Data Verification:")
    print(f"    cloudy shape : {tuple(cloudy.shape)}")
    print(f"    clear  shape : {tuple(clear.shape)}")
    print(f"    mask   shape : {tuple(mask.shape)}")

    # Range checks
    c_min, c_max = cloudy[:, :n_bands].min().item(), cloudy[:, :n_bands].max().item()
    t_min, t_max = clear.min().item(), clear.max().item()
    m_min, m_max = mask.min().item(), mask.max().item()

    print(f"    cloudy range : [{c_min:.3f}, {c_max:.3f}]  (expect [0,1])")
    print(f"    clear  range : [{t_min:.3f}, {t_max:.3f}]  (expect [0,1])")
    print(f"    mask   range : [{m_min:.3f}, {m_max:.3f}]  (expect {0,1})")

    ok = True
    if c_max > 1.5 or c_min < -0.5:
        print("    [WARN] Cloudy values out of [0,1] -- check normalisation!")
        ok = False
    if t_max > 1.5 or t_min < -0.5:
        print("    [WARN] Clear values out of [0,1] -- check normalisation!")
        ok = False
    if not (0.0 <= m_min and m_max <= 1.0):
        print("    [WARN] Mask not binary -- check cloud detector!")
        ok = False

    # Cloud fraction
    cloud_frac = mask.float().mean().item()
    print(f"    cloud fraction: {cloud_frac*100:.1f}%")

    # Band correlation (cloudy optical vs clear: should be high in clear regions)
    opt = cloudy[:, :n_bands]
    clear_region = (1 - mask.unsqueeze(1)).expand_as(opt)
    if clear_region.sum() > 100:
        corr_vals = []
        for b in range(n_bands):
            pred_b   = opt[:, b][clear_region[:, 0].bool()].numpy()
            target_b = clear[:, b][clear_region[:, 0].bool()].numpy()
            if len(pred_b) > 10:
                corr = float(np.corrcoef(pred_b[:1000], target_b[:1000])[0, 1])
                corr_vals.append(corr)
        if corr_vals:
            avg_corr = np.mean(corr_vals)
            status = "[OK] " if avg_corr > 0.7 else "[WARN]"
            print(f"    {status} clear-region band correlation: {avg_corr:.3f}  (expect >0.7 if paired correctly)")
    if ok:
        print("    [OK] Data looks correctly paired and normalised.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Fast per-batch PSNR, SSIM, RMSE (no LPIPS — too slow for every step)."""
    from evaluation.metrics import compute_psnr, compute_ssim
    pred_np   = torch.clamp(pred, 0, 1).detach().cpu().numpy()
    target_np = torch.clamp(target, 0, 1).detach().cpu().numpy()

    psnr_list, ssim_list, rmse_list = [], [], []
    for i in range(pred_np.shape[0]):
        p = np.transpose(pred_np[i], (1, 2, 0))
        t = np.transpose(target_np[i], (1, 2, 0))
        psnr_list.append(compute_psnr(p, t))
        ssim_list.append(compute_ssim(p, t))
        rmse_list.append(float(np.sqrt(np.mean((p - t) ** 2))))

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "rmse": float(np.mean(rmse_list)),
    }


def compute_lpips_final(pred_list, target_list) -> float:
    """Compute LPIPS on the collected outputs (run once at end)."""
    from evaluation.metrics import compute_lpips
    scores = []
    for p, t in zip(pred_list, target_list):
        scores.append(compute_lpips(
            np.transpose(p, (1, 2, 0)),
            np.transpose(t, (1, 2, 0)),
        ))
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Visual output
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(cloudy, pred, clear, mask, epoch, output_dir, n_bands=6):
    """Save a comparison panel for the first sample in the batch."""
    try:
        from evaluation.visualizer import create_comparison_panel
        import torchvision.utils as vutils

        output_dir.mkdir(parents=True, exist_ok=True)

        c_rgb = torch.clamp(cloudy[0, [2, 1, 0]], 0, 1).cpu()
        p_rgb = torch.clamp(pred[0, [2, 1, 0]], 0, 1).cpu()
        t_rgb = torch.clamp(clear[0, [2, 1, 0]], 0, 1).cpu()

        vutils.save_image(
            torch.stack([c_rgb, p_rgb, t_rgb]),
            output_dir / f"epoch_{epoch:03d}.png",
            nrow=3,
            padding=4,
        )
    except Exception as e:
        print(f"    [warn] Could not save visual: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU monitoring
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_stats(device):
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        util_pct = reserved / total * 100
        print(f"  GPU: {alloc:.0f}MB alloc / {reserved:.0f}MB reserved / {total:.0f}MB total ({util_pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 64)
    print("  OVERFIT TEST -- Swin U-Net Pipeline Verification")
    print("=" * 64)
    print(f"  Pairs:      {args.n_samples}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override for overfit test
    if args.patch_size:
        cfg["patch"]["size"] = args.patch_size
        cfg["inference"]["patch_size"] = args.patch_size
    if args.no_pretrain:
        cfg["model"]["pretrained"] = False
    # Disable gradient checkpointing for overfit test (we want full speed)
    cfg["model"]["gradient_checkpointing"] = False
    cfg["training"]["amp"] = True

    patch_size = cfg["patch"]["size"]
    print(f"  Patch size: {patch_size}x{patch_size}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(device)}")
    print()

    # Dataset
    from preprocessing.loader import build_dataloaders
    dataset_dir = Path(cfg["dataset"]["output_dir"])
    if not dataset_dir.exists() or not (dataset_dir / "train").exists():
        print(f"[ERROR] Dataset not found at: {dataset_dir.resolve()}")
        print("  Run prepare_dataset.py first, or check config.yaml dataset.output_dir")
        return 1

    print("Loading dataset...")
    full_loaders = build_dataloaders(
        dataset_dir=dataset_dir,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_workers=0,        # 0 workers for stability during test
        pin_memory=False,     # no pinning (num_workers=0)
    )
    full_train_ds = full_loaders["train"].dataset

    # Subset: take at most N samples
    n = min(args.n_samples, len(full_train_ds))
    if n < len(full_train_ds):
        indices = list(range(n))   # take first N for reproducibility
        subset = Subset(full_train_ds, indices)
    else:
        subset = full_train_ds

    print(f"  Using {n} / {len(full_train_ds)} available patches")

    overfit_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Verify a batch
    sample_batch = next(iter(overfit_loader))
    verify_batch(sample_batch, n_bands=cfg["model"]["output_channels"])

    # Build model
    print("\nBuilding model...")
    from models.registry import build_model
    model = build_model(cfg).to(device)
    params = model.count_parameters()
    print(f"  Parameters: {params / 1e6:.2f}M")

    # Loss
    from models.losses import CombinedLoss
    loss_fn = CombinedLoss(
        w_l1=cfg["loss"]["l1"],
        w_ssim=cfg["loss"]["ssim"],
        w_perceptual=cfg["loss"]["perceptual"],
        w_charbonnier=cfg["loss"].get("charbonnier", 0.1),
        w_spectral=cfg["loss"].get("spectral", 0.05),
        n_bands=cfg["model"]["output_channels"],
    ).to(device)

    # Optimizer — higher LR to overfit faster
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = GradScaler()
    n_bands = cfg["model"]["output_channels"]

    # Output dir for visuals
    output_dir = Path("outputs") / "overfit_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    history = []
    best_psnr, best_ssim, best_rmse = 0.0, 0.0, float("inf")

    # === TRAINING LOOP ===
    print(f"\n{'=' * 64}")
    print("  Training (overfit mode)...")
    print(f"{'=' * 64}")
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'PSNR':>8} | {'SSIM':>8} | {'RMSE':>8} | {'Time':>6}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr, epoch_ssim, epoch_rmse = [], [], []
        t0 = time.time()

        for batch in overfit_loader:
            cloudy = batch["cloudy"].to(device)
            clear  = batch["clear"].to(device)
            mask   = batch["mask"].to(device)

            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(optical, mask_ch)
                loss, _ = loss_fn(pred, clear, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            m = batch_metrics(pred, clear)
            epoch_psnr.append(m["psnr"])
            epoch_ssim.append(m["ssim"])
            epoch_rmse.append(m["rmse"])

        avg_loss = epoch_loss / len(overfit_loader)
        avg_psnr = float(np.mean(epoch_psnr))
        avg_ssim = float(np.mean(epoch_ssim))
        avg_rmse = float(np.mean(epoch_rmse))
        elapsed  = time.time() - t0

        best_psnr = max(best_psnr, avg_psnr)
        best_ssim = max(best_ssim, avg_ssim)
        best_rmse = min(best_rmse, avg_rmse)

        # Flags for easy reading
        psnr_flag = " *" if avg_psnr >= 35.0 else ""
        ssim_flag = " *" if avg_ssim >= 0.95 else ""

        print(f"  {epoch:5d} | {avg_loss:8.4f} | {avg_psnr:7.2f}{psnr_flag} | {avg_ssim:7.4f}{ssim_flag} | {avg_rmse:8.4f} | {elapsed:5.1f}s")

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "psnr": avg_psnr, "ssim": avg_ssim, "rmse": avg_rmse
        })

        # Save visuals
        if args.save_visuals and epoch % 10 == 0:
            save_comparison(cloudy, pred, clear, mask, epoch, output_dir / "visuals", n_bands)

    # === COMPUTE LPIPS ON FINAL EPOCH ===
    print("\n  Computing LPIPS on final epoch...")
    model.eval()
    pred_list, target_list = [], []
    with torch.no_grad():
        for batch in overfit_loader:
            cloudy = batch["cloudy"].to(device)
            clear  = batch["clear"].to(device)
            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(optical, mask_ch)
            pred_list.extend(torch.clamp(pred, 0, 1).cpu().numpy())
            target_list.extend(torch.clamp(clear, 0, 1).cpu().numpy())
            if len(pred_list) >= 20:   # limit LPIPS to 20 samples (slow)
                break

    lpips_score = compute_lpips_final(pred_list[:20], target_list[:20])

    # Save a final comparison
    with torch.no_grad():
        sample = next(iter(overfit_loader))
        cloudy_s = sample["cloudy"].to(device)
        clear_s  = sample["clear"].to(device)
        mask_s   = sample["mask"].to(device)
        optical_s = cloudy_s[:, :n_bands]
        mask_ch_s = cloudy_s[:, n_bands:]
        with autocast("cuda", enabled=(device.type == "cuda")):
            pred_s = model(optical_s, mask_ch_s)

    save_comparison(cloudy_s, pred_s, clear_s, mask_s, 999, output_dir, n_bands)

    # === GPU STATS ===
    print()
    print_gpu_stats(device)

    # === FINAL REPORT ===
    total_time = time.time() - t_start
    print(f"\n{'=' * 64}")
    print("  OVERFIT TEST RESULTS")
    print(f"{'=' * 64}")
    print(f"  Total training time: {total_time/60:.1f} min")
    print(f"  Best PSNR:  {best_psnr:.2f} dB  (target >= 35)")
    print(f"  Best SSIM:  {best_ssim:.4f}   (target >= 0.95)")
    print(f"  Best RMSE:  {best_rmse:.4f}   (target <= 0.05)")
    print(f"  LPIPS:      {lpips_score:.4f}   (target ~= 0)")
    print()

    results = {
        "psnr": best_psnr,
        "ssim": best_ssim,
        "rmse": best_rmse,
        "lpips": lpips_score,
    }

    all_pass = True
    for name, value in results.items():
        line = check_target(name, value)
        if line:
            print(line)
            if "[FAIL]" in line:
                all_pass = False

    print()
    if all_pass:
        print("  [PASS] Pipeline is working correctly!")
        print("         Safe to launch full 200-epoch training.")
        print("         Run: python train_optimized.py")
    else:
        print("  [FAIL] Model could not overfit the training data.")
        print("         This indicates a bug in: data loading,")
        print("         preprocessing, normalisation, band order, or loss.")
        print()
        print("  Debugging checklist:")
        print("  1. Verify cloudy/clear/mask are correctly paired")
        print("  2. Check normalisation: values should be in [0, 1]")
        print("  3. Verify band ordering is consistent (B02=0, B03=1, B04=2, ...)")
        print("  4. Run: python sanity_check.py --verbose")
        print("  5. Inspect outputs/overfit_test/ for visual evidence")

    print(f"{'=' * 64}")
    print(f"  Comparison panel saved to: {(output_dir / 'epoch_999.png').resolve()}")
    print(f"{'=' * 64}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
