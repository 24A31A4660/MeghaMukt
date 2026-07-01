"""utils/logger.py — TensorBoard + rotating file logger."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def setup_logger(
    name: str = "cloud_recon",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TBLogger:
    """Thin wrapper around SummaryWriter with graceful fallback when TensorBoard is unavailable."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self._writer: Optional["SummaryWriter"] = None
        if HAS_TB:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def image(self, tag: str, img_tensor, step: int) -> None:
        if self._writer is not None:
            self._writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
