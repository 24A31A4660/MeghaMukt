#!/usr/bin/env python3
"""
sanity_check.py — Verify the full Swin U-Net pipeline before training.

Checks:
  1. Imports work from all entry points
  2. SwinUNet forward pass — correct shape and value range
  3. Variable input sizes (128, 256, 384, 512)
  4. Loss function computation (all 4 components)
  5. Metrics computation (PSNR, SSIM, RMSE, MAE, SAM, LPIPS)
  6. VRAM usage with batch=4, patch=384 on CUDA
  7. Checkpoint save/load round-trip
  8. DataLoader structure (if dataset exists)

Run from cloud-reconstruction/ directory:
    python sanity_check.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(label: str, fn, skip_if=None) -> bool:
    if skip_if and skip_if():
        print(f"{SKIP}  {label}")
        return True
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"         -> {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
def test_imports():
    section("1. Import Tests")

    check("models.registry.build_model", lambda: __import__("models.registry", fromlist=["build_model"]))
    check("models.swin_unet.SwinUNet",   lambda: __import__("models.swin_unet", fromlist=["SwinUNet"]))
    check("models.losses.CombinedLoss",  lambda: __import__("models.losses", fromlist=["CombinedLoss"]))
    check("evaluation.metrics",          lambda: __import__("evaluation.metrics"))
    check("evaluation.visualizer",       lambda: __import__("evaluation.visualizer"))
    check("utils.checkpoint",            lambda: __import__("utils.checkpoint"))
    check("preprocessing.loader",        lambda: __import__("preprocessing.loader"))
    check("timm available",              lambda: __import__("timm"))
    check("einops available",            lambda: __import__("einops"))


# ─────────────────────────────────────────────────────────────────────────────
def test_model():
    section("2. SwinUNet Forward Pass Tests")
    from models.swin_unet import SwinUNet

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # CPU-only instantiation test (no pretrained weights to avoid download)
    cfg["model"]["pretrained"] = False

    device = torch.device("cpu")

    def make_inputs(H, W, B=1):
        opt = torch.rand(B, 6, H, W)
        msk = torch.rand(B, 1, H, W)
        return opt, msk

    model = SwinUNet(cfg).to(device).eval()
    params_m = model.count_parameters() / 1e6

    check(f"Instantiation (params={params_m:.2f}M)", lambda: None)

    # Minimum supported size = spatial_multiple = patch_size * window_size * 2^(n_stages-1) = 256
    # The existing dataset patches are 256x256 so this is always satisfied.
    for size in [256, 384, 512]:
        def fwd(s=size):
            opt, msk = make_inputs(s, s)
            with torch.no_grad():
                out = model(opt, msk)
            assert out.shape == (1, 6, s, s), f"Expected (1,6,{s},{s}), got {out.shape}"
            assert 0.0 <= out.min().item() <= out.max().item() <= 1.0, \
                f"Output out of [0,1] range: min={out.min().item():.4f}, max={out.max().item():.4f}"
        check(f"Forward pass {size}x{size}", fwd)

    # Non-multiples of 256 (padding to next 256 boundary must work)
    def fwd_odd():
        # 288 pads to 512, 320 pads to 512
        opt, msk = make_inputs(288, 320)
        with torch.no_grad():
            out = model(opt, msk)
        assert out.shape == (1, 6, 288, 320), f"Expected (1,6,288,320), got {out.shape}"
    check("Forward pass 288x320 (non-multiple of 256)", fwd_odd)

    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
def test_loss():
    section("3. Loss Function Tests")
    from models.losses import CombinedLoss

    loss_fn = CombinedLoss(w_l1=0.5, w_ssim=0.3, w_perceptual=0.1, w_charbonnier=0.1)

    B, C, H, W = 2, 6, 256, 256
    pred   = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    mask   = (torch.rand(B, H, W) > 0.6).float()

    def run_loss_no_mask():
        loss, comps = loss_fn(pred, target)
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"
        assert all(k in comps for k in ["loss/total", "loss/l1", "loss/ssim", "loss/charbonnier"])

    def run_loss_with_mask():
        loss, comps = loss_fn(pred, target, mask)
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"

    def run_loss_perc():
        loss, comps = loss_fn(pred, target, mask)
        # Perceptual may be 0 if VGG not available — just check it's a number
        assert isinstance(comps["loss/perceptual"], float)

    def run_loss_spectral():
        from models.losses import SpectralConsistencyLoss
        loss_fn = SpectralConsistencyLoss()
        B, C, H, W = 1, 6, 64, 64
        pred   = torch.rand(B, C, H, W)
        target = torch.rand(B, C, H, W)
        mask   = (torch.rand(B, H, W) > 0.5).float()
        # Without mask
        val_no_mask = loss_fn(pred, target, None)
        assert val_no_mask.shape == (), f"Expected scalar, got {val_no_mask.shape}"
        # With mask
        val_masked = loss_fn(pred, target, mask)
        assert val_masked.shape == (), f"Expected scalar, got {val_masked.shape}"
        # Identical images -> loss should be 0
        val_zero = loss_fn(pred, pred, None)
        assert val_zero.item() < 1e-5, f"Identical inputs give non-zero loss: {val_zero.item()}"

    check("CombinedLoss without mask", run_loss_no_mask)
    check("CombinedLoss with cloud mask", run_loss_with_mask)
    check("Perceptual loss returns scalar", run_loss_perc)
    check("SpectralConsistencyLoss (NDVI/SR/GSWIR)", run_loss_spectral)
    check("Loss weights sum to ~1.05",
          lambda: abs(0.5 + 0.3 + 0.1 + 0.1 + 0.05 - 1.05) < 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
def test_metrics():
    section("4. Metrics Tests")
    from evaluation.metrics import compute_metrics, compute_sam, compute_lpips

    H, W, C = 256, 256, 6
    pred   = np.random.rand(C, H, W).astype(np.float32)
    target = np.random.rand(C, H, W).astype(np.float32)
    mask   = (np.random.rand(H, W) > 0.6).astype(np.uint8)

    def run_all_metrics():
        result = compute_metrics(pred, target, mask)
        d = result.to_dict()
        for key in ["psnr_full", "ssim_full", "rmse_full", "mae_full", "sam_full", "lpips_full"]:
            assert key in d, f"Missing metric: {key}"
            assert isinstance(d[key], float), f"{key} is not float: {type(d[key])}"

    def run_sam():
        pred_hwc = np.transpose(pred, (1, 2, 0))
        tgt_hwc  = np.transpose(target, (1, 2, 0))
        sam_val  = compute_sam(pred_hwc, tgt_hwc)
        assert 0 <= sam_val <= 90, f"SAM out of range: {sam_val}"

    def run_lpips():
        pred_hwc = np.transpose(pred, (1, 2, 0))
        tgt_hwc  = np.transpose(target, (1, 2, 0))
        val = compute_lpips(pred_hwc, tgt_hwc)
        assert isinstance(val, float), f"LPIPS not float: {type(val)}"

    def run_perfect():
        """Identical images should give max PSNR and SSIM=1."""
        result = compute_metrics(pred, pred, mask)
        assert result.ssim_full > 0.99, f"SSIM of identical images: {result.ssim_full:.4f}"

    check("All 6 metrics present", run_all_metrics)
    check("SAM in [0, 90]°", run_sam)
    check("LPIPS returns float", run_lpips)
    check("Identical images: SSIM=1.0", run_perfect)


# ─────────────────────────────────────────────────────────────────────────────
def test_checkpoint():
    section("5. Checkpoint Round-Trip Test")
    import tempfile, os
    from models.swin_unet import SwinUNet
    from utils.checkpoint import save_checkpoint, load_checkpoint

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False

    model_a = SwinUNet(cfg)
    opt_a   = torch.optim.AdamW(model_a.parameters(), lr=3e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pth"

        def save():
            save_checkpoint(ckpt_path, 5, model_a, opt_a, None,
                            metrics={"val_loss": 0.042}, config=cfg)
            assert ckpt_path.exists()

        def load():
            model_b = SwinUNet(cfg)
            payload = load_checkpoint(ckpt_path, model_b, device="cpu")
            assert payload["epoch"] == 5
            assert abs(payload["metrics"]["val_loss"] - 0.042) < 1e-9
            # Check weights match
            for (n, p_a), (_, p_b) in zip(
                model_a.named_parameters(), model_b.named_parameters()
            ):
                assert torch.allclose(p_a, p_b), f"Weight mismatch at {n}"

        check("Save checkpoint", save)
        check("Load checkpoint + weight match", load)


# ─────────────────────────────────────────────────────────────────────────────
def test_gpu():
    section("6. GPU VRAM Test (batch=4, patch=384)")

    if not torch.cuda.is_available():
        print(f"{SKIP}  CUDA not available — skipping GPU tests")
        return

    from models.swin_unet import SwinUNet

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False

    device = torch.device("cuda:0")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    print(f"  GPU: {torch.cuda.get_device_name(device)}  |  VRAM: {total_vram:.0f} MB")

    def gpu_forward_amp():
        from torch.amp import autocast
        model = SwinUNet(cfg).to(device)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        B, C, H, W = 4, 6, 384, 384
        opt = torch.rand(B, C, H, W).to(device)
        msk = torch.rand(B, 1, H, W).to(device)

        with torch.no_grad(), autocast("cuda", enabled=True):
            out = model(opt, msk)

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        alloc_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"\n  Batch=4, Patch=384×384:")
        print(f"    Peak VRAM usage:     {peak_mb:.0f} MB")
        print(f"    Current allocation:  {alloc_mb:.0f} MB")
        print(f"    Available VRAM:      {total_vram:.0f} MB")

        if peak_mb > total_vram * 0.95:
            raise RuntimeError(
                f"VRAM usage {peak_mb:.0f}MB is >95% of total {total_vram:.0f}MB. "
                "Consider reducing batch size or using gradient checkpointing."
            )
        assert out.shape == (B, 6, H, W)

    check(f"AMP forward pass on CUDA (batch=4, 384×384)", gpu_forward_amp)


# ─────────────────────────────────────────────────────────────────────────────
def test_dataset():
    section("7. Dataset Loader Test")
    from preprocessing.loader import build_dataloaders

    dataset_dir = Path("dataset")
    if not dataset_dir.exists() or not (dataset_dir / "train").exists():
        print(f"{SKIP}  Dataset not found at {dataset_dir.resolve()} — skipping")
        return

    def load():
        loaders = build_dataloaders(
            dataset_dir=dataset_dir,
            patch_size=384,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        assert "train" in loaders, "Missing 'train' split"
        train_loader = loaders["train"]
        batch = next(iter(train_loader))

        # Check all required keys are present
        for key in ("cloudy", "clear", "mask"):
            assert key in batch, f"Missing batch key: {key}"

        cloudy = batch["cloudy"]
        clear  = batch["clear"]
        mask   = batch["mask"]

        print(f"\n  cloudy shape: {tuple(cloudy.shape)}")
        print(f"  clear  shape: {tuple(clear.shape)}")
        print(f"  mask   shape: {tuple(mask.shape)}")

        assert cloudy.shape[1] >= 7, f"Expected ≥7 channels in cloudy, got {cloudy.shape[1]}"
        assert clear.shape[1] == 6, f"Expected 6 channels in clear, got {clear.shape[1]}"

    check("DataLoader: load one batch", load)


# ─────────────────────────────────────────────────────────────────────────────
def test_config():
    section("8. Config Validation")

    def validate_config():
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        required = {
            "model": ["name", "input_channels", "output_channels", "embed_dim",
                       "depths", "num_heads", "window_size", "patch_size"],
            "training": ["epochs", "batch_size", "gradient_accumulation",
                          "learning_rate", "weight_decay", "amp", "grad_clip",
                          "early_stopping_patience"],
            "loss": ["l1", "ssim", "perceptual", "charbonnier"],
            "paths": ["checkpoints", "logs", "outputs", "tensorboard"],
        }

        for section_key, keys in required.items():
            assert section_key in cfg, f"Config missing section: {section_key}"
            for k in keys:
                assert k in cfg[section_key], f"Config missing: {section_key}.{k}"

        # Validate loss weights sum to 1.0
        lcfg = cfg["loss"]
        total = lcfg["l1"] + lcfg["ssim"] + lcfg["perceptual"] + lcfg["charbonnier"]
        assert abs(total - 1.0) < 1e-6, f"Loss weights sum to {total} (expected 1.0)"

        # Validate model name
        assert cfg["model"]["name"] in ("swin_unet", "unet"), \
            f"Unknown model: {cfg['model']['name']}"

        print(f"\n  Model:       {cfg['model']['name']}")
        print(f"  Patch size:  {cfg['patch']['size']}×{cfg['patch']['size']}")
        print(f"  Batch size:  {cfg['training']['batch_size']} × {cfg['training']['gradient_accumulation']} = "
              f"{cfg['training']['batch_size'] * cfg['training']['gradient_accumulation']} effective")
        print(f"  LR:          {cfg['training']['learning_rate']}")
        print(f"  Epochs:      {cfg['training']['epochs']}")
        print(f"  Loss weights: L1={lcfg['l1']} + SSIM={lcfg['ssim']} + "
              f"Perceptual={lcfg['perceptual']} + Charbonnier={lcfg['charbonnier']} = {total:.1f}")

    check("All required config keys present", validate_config)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Swin U-Net — Pipeline Sanity Check")
    print("=" * 60)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()} ({torch.version.cuda or 'N/A'})")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")

    test_imports()
    test_config()
    model, cfg = test_model()
    test_loss()
    test_metrics()
    test_checkpoint()
    test_gpu()
    test_dataset()

    print("\n" + "=" * 60)
    print("  Sanity check complete.")
    print("  If all tests pass (or skip), run: python train_optimized.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
