#!/usr/bin/env python3
import os
import json
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.registry import build_model
from preprocessing.loader import build_dataloaders
from utils.checkpoint import load_checkpoint
from train_optimized import compute_metrics
import matplotlib.pyplot as plt

def create_diff_panel(cloudy, pred, gt, mask, save_path, idx):
    """Generate the Cloudy -> AI -> GT -> Diff map panel."""
    # Convert from [C, H, W] to [H, W, C] and select RGB bands (B04, B03, B02 -> channels 2, 1, 0)
    def to_rgb(tensor):
        img = tensor.transpose(1, 2, 0)
        rgb = img[:, :, [2, 1, 0]]
        # Simple stretch for visualization
        rgb = np.clip(rgb * 3.0, 0, 1)
        return rgb

    c_rgb = to_rgb(cloudy)
    p_rgb = to_rgb(pred)
    g_rgb = to_rgb(gt)

    # Difference map (Absolute error between Pred and GT, averaged over RGB)
    diff = np.mean(np.abs(p_rgb - g_rgb), axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(c_rgb)
    axes[0].set_title(f"Cloudy Input (Test #{idx})")
    axes[0].axis('off')

    axes[1].imshow(p_rgb)
    axes[1].set_title("AI Reconstruction")
    axes[1].axis('off')

    axes[2].imshow(g_rgb)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')

    im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[3].set_title("Difference Map")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running final evaluation on {device}")

    _, _, test_loader = build_dataloaders(config)
    if not test_loader:
        print("No test_loader available. Falling back to val_loader.")
        _, test_loader, _ = build_dataloaders(config)

    model = build_model(config).to(device)
    model.eval()

    ckpt_path = Path(config["paths"]["checkpoints"]) / "best_model.pth"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        return

    print(f"Loading {ckpt_path}...")
    load_checkpoint(ckpt_path, model, device=str(device))

    panels_dir = Path(config["paths"]["outputs"]) / "final_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    panel_count = 0

    print("Evaluating test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            cloudy = batch["cloudy"].to(device)
            clear = batch["clear"].to(device)
            mask = batch["mask"].to(device)

            n_bands = config["model"]["output_channels"]
            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            pred = model(optical, mask_ch)

            metrics = compute_metrics(pred, clear, mask)
            all_metrics.append(metrics)

            # Generate panels
            for i in range(cloudy.shape[0]):
                if panel_count < 30:
                    save_path = panels_dir / f"test_panel_{panel_count:02d}.png"
                    create_diff_panel(
                        cloudy[i].cpu().numpy(),
                        pred[i].cpu().numpy(),
                        clear[i].cpu().numpy(),
                        mask[i].cpu().numpy(),
                        save_path,
                        panel_count
                    )
                    panel_count += 1

    final_metrics = {}
    for k in all_metrics[0].keys():
        final_metrics[k] = float(np.mean([m[k] for m in all_metrics]))

    out_path = Path(config["paths"]["outputs"]) / "final_test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*50)
    print("FINAL TEST METRICS")
    print("="*50)
    for k, v in final_metrics.items():
        print(f"{k.upper():<10}: {v:.4f}")
    print("="*50)
    print(f"Saved {panel_count} comparison panels to {panels_dir}")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
