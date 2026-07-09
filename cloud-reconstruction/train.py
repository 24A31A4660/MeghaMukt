#!/usr/bin/env python3
"""
train.py — Train the cloud reconstruction model.

Usage:
    python train.py                            # default config
    python train.py --epochs 50 --batch 4      # override
    python train.py --resume                   # resume from latest checkpoint
    python train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.registry import build_model
from preprocessing.loader import build_dataloaders
from training.trainer import Trainer
from utils.logger import setup_logger

log = setup_logger("train")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    dev_type = cfg["device"]["type"]
    if dev_type == "auto":
        if torch.cuda.is_available():
            idx = cfg["device"]["cuda_id"]
            return torch.device(f"cuda:{idx}")
        return torch.device("cpu")
    return torch.device(dev_type)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train cloud reconstruction model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", action="store_true", default=None, help="Force resume")
    parser.add_argument("--no-resume", action="store_true", help="Force fresh start")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch is not None:
        cfg["training"]["batch_size"] = args.batch
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.resume is not None:
        cfg["training"]["resume"] = True
    if args.no_resume:
        cfg["training"]["resume"] = False

    # Create directories
    for key in ("checkpoints", "logs", "outputs", "tensorboard"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # Device
    device = resolve_device(cfg)
    log.info("Device: %s", device)

    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(device))
        log.info("VRAM: %.1f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)

    # Build model via registry (supports both swin_unet and unet)
    model = build_model(cfg)
    log.info("Model: %s", model.get_config())
    log.info("Parameters: %s", f"{model.count_parameters():,}")

    # Build dataloaders
    source_dir = Path(cfg["dataset"]["source_dir"])
    if "RICE" in source_dir.name or "RICE" in str(source_dir):
        from preprocessing.rice_loader import build_rice_dataloaders
        dataset_dir = source_dir
        if not dataset_dir.exists():
            log.error("RICE dataset directory '%s' not found.", dataset_dir)
            return 1

        loaders = build_rice_dataloaders(
            dataset_dir=dataset_dir,
            patch_size=cfg["patch"]["size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
            pin_memory=cfg["training"]["pin_memory"],
            append_mask=(cfg["model"]["input_channels"] == 4)
        )
    else:
        dataset_dir = Path(cfg["dataset"]["output_dir"])
        if not dataset_dir.exists():
            log.error(
                "Dataset directory '%s' not found. Run prepare_dataset.py first.",
                dataset_dir,
            )
            return 1

        loaders = build_dataloaders(
            dataset_dir=dataset_dir,
            patch_size=cfg["patch"]["size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
            pin_memory=cfg["training"]["pin_memory"],
        )

    if "train" not in loaders:
        log.error("No training data found in %s/train/", dataset_dir)
        return 1

    log.info("Train batches: %d", len(loaders["train"]))
    if "validation" in loaders:
        log.info("Val batches: %d", len(loaders["validation"]))

    # Train
    trainer = Trainer(cfg, model, device)
    trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders.get("validation"),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
