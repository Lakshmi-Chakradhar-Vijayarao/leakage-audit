"""01 — download / prepare datasets and save normalized JSONL items.

Outputs:
  data/processed/<dataset>.jsonl

Idempotent: re-running skips datasets whose JSONL already exists AND
contains at least one item. A previous run that produced an empty
file will be re-attempted.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so `src.*` imports resolve when running the
# script directly with `python scripts/01_download_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, set_seed, get_logger, ensure_dir
from src.data.dataset_loader import (
    DATASET_LOADERS, save_items_jsonl,
)
from src.data.prompt_generator import generate_synthetic_dataset


def _existing_nonempty(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            for _ in f:
                return True
    except Exception:
        return False
    return False


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    log = get_logger("download_data", log_dir=cfg["paths"]["logs"])

    out_dir = ensure_dir(cfg["paths"]["data_processed"])
    cache_dir = ensure_dir(cfg["paths"]["cache_dir"])

    for ds_name, ds_cfg in cfg["datasets"].items():
        out_path = out_dir / f"{ds_name}.jsonl"
        if _existing_nonempty(out_path):
            log.info(f"[skip] {out_path} already exists (non-empty)")
            continue
        if out_path.exists():
            log.warning(
                f"[retry] {out_path} exists but is empty — deleting and re-downloading"
            )
            out_path.unlink()
        log.info(f"[load] {ds_name}")
        try:
            if ds_name == "synthetic":
                items = generate_synthetic_dataset(
                    n_samples=ds_cfg.get("n_samples", 400),
                    seed=cfg["seed"],
                )
            else:
                loader = DATASET_LOADERS[ds_name]
                items = loader(
                    n_samples=ds_cfg.get("n_samples"),
                    cache_dir=str(cache_dir),
                    seed=cfg["seed"],
                )
        except Exception as e:
            log.exception(f"[fail] could not load {ds_name}: {e}")
            continue

        if not items:
            log.error(
                f"[fail] {ds_name} produced 0 items. The dataset mirrors may "
                f"have changed shape — please open an issue or extend "
                f"src/data/dataset_loader.py with the new column names."
            )
            continue

        save_items_jsonl(items, out_path)
        log.info(f"[saved] {out_path}  (n={len(items)})")


if __name__ == "__main__":
    main()
