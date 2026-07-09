#!/usr/bin/env python3
"""
validate.py — Run validation / test evaluation on a checkpoint.

Usage:
    python validate.py                                    # uses best_model.pth
    python validate.py --checkpoint checkpoints_swin/epoch_0050.pth
    python validate.py --split test                       # evaluate on test set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.confidence import compute_confidence_map, compute_difference_map
from evaluation.metrics import compute_metrics
from evaluation.visualizer import create_comparison_panel
from models.registry import build_model
from preprocessing.loader import build_dataloaders
from utils.checkpoint import find_best_checkpoint, load_checkpoint
from utils.logger import setup_logger

log = setup_logger("validate")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    dev_type = cfg["device"]["type"]
    if dev_type == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_type)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate cloud reconstruction model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: best_model.pth)")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--save-panels", action="store_true", help="Save comparison panels")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)

    # Load model via registry
    model = build_model(cfg)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(Path(cfg["paths"]["checkpoints"]))
    if ckpt_path is None or not ckpt_path.exists():
        log.error("No checkpoint found. Train the model first.")
        return 1

    payload = load_checkpoint(ckpt_path, model, device=str(device), strict=False)
    model = model.to(device).eval()
    log.info("Loaded checkpoint: %s (epoch %d)", ckpt_path.name, payload.get("epoch", -1))

    # Build dataloader
    loaders = build_dataloaders(
        dataset_dir=Path(cfg["dataset"]["output_dir"]),
        patch_size=cfg["patch"]["size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    loader = loaders.get(args.split)
    if loader is None:
        log.error("No %s data found.", args.split)
        return 1

    n_bands = cfg["model"]["output_channels"]
    use_amp = cfg["training"]["amp"] and device.type == "cuda"

    all_metrics: list[dict] = []
    output_dir = Path(cfg["paths"]["outputs"]) / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating [{args.split}]", unit="batch")):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            cloudy = batch["cloudy"].to(device)
            clear  = batch["clear"].to(device)
            mask   = batch["mask"].to(device)

            optical = cloudy[:, :n_bands]
            mask_ch = cloudy[:, n_bands:]

            with autocast("cuda", enabled=use_amp):
                pred = model(optical, mask_ch)

            # Evaluate each sample in batch
            for i in range(pred.shape[0]):
                p = pred[i].cpu().numpy()
                t = clear[i].cpu().numpy()
                m = mask[i].cpu().numpy()
                c = cloudy[i].cpu().numpy()

                result = compute_metrics(p, t, m.astype(np.uint8))
                all_metrics.append(result.to_dict())

                # Save panels for first few samples
                if args.save_panels and batch_idx < 5:
                    stem = batch["stem"][i] if "stem" in batch else f"batch{batch_idx}_{i}"
                    diff = compute_difference_map(p, t)
                    conf = compute_confidence_map(p, c[:n_bands], m.astype(np.uint8))
                    create_comparison_panel(
                        cloudy_input=c[:n_bands],
                        cloud_mask=m.astype(np.uint8),
                        prediction=p,
                        ground_truth=t,
                        difference_map=diff,
                        confidence_map=conf,
                        output_path=output_dir / f"{stem}_comparison.jpg",
                    )

    # Aggregate metrics
    if all_metrics:
        avg_metrics = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0].keys()
        }
        log.info("=" * 60)
        log.info("EVALUATION RESULTS [%s] — %d patches", args.split.upper(), len(all_metrics))
        log.info("=" * 60)
        for k, v in avg_metrics.items():
            log.info("  %s: %.4f", k, v)
        log.info("=" * 60)

        # Save to JSON
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({"average": avg_metrics, "per_sample": all_metrics}, f, indent=2)
        log.info("Results saved to: %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
