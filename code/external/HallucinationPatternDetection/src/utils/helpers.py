"""Shared utilities: config loading, seeding, logging, IO helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# --------------------------------------------------------------- env tweaks
# Set at import time so every script that touches the project gets a
# clean console regardless of launch mode. `setdefault` means
# user-set values win.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------- seeding ---------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for python, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ----------------------------- config ----------------------------------
def load_config(path: str | os.PathLike = "config/config.yaml") -> Dict[str, Any]:
    """Load the YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found at {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ----------------------------- logging ---------------------------------
def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to console + optional file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        ensure_dir(log_dir)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------- io --------------------------------------
def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


# ----------------------------- timing ----------------------------------
class Timer:
    """Context manager that measures wall-clock duration."""

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.time() - (self.start or time.time())
        msg = f"[{self.label}] elapsed {self.elapsed:.2f}s"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
