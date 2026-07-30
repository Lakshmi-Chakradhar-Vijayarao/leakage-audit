"""02 — extract per-layer hidden states for every (model, dataset) pair.

Outputs:
  results/hidden_states/<model>__<dataset>.pt
"""
from __future__ import annotations

import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import load_config, set_seed, get_logger, ensure_dir, Timer
from src.data.dataset_loader import load_items_jsonl
from src.models.model_loader import load_quantized_model
from src.models.hidden_state_extractor import (
    HiddenStateExtractor, ExtractionConfig, extract_hidden_states_for_dataset,
)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    log = get_logger("extract_hidden_states", log_dir=cfg["paths"]["logs"])

    processed_dir = Path(cfg["paths"]["data_processed"])
    out_dir = ensure_dir(cfg["paths"]["hidden_states"])

    ex_cfg = ExtractionConfig(
        pool=cfg["extraction"]["pool"],
        capture_attention=cfg["extraction"]["capture_attention"],
        attention_layers=cfg["extraction"]["attention_layers"],
        batch_size=cfg["extraction"]["batch_size"],
        max_input_length=cfg["extraction"]["max_input_length"],
    )

    failed_models: list = []
    for model_cfg in cfg["models"]:
        short = model_cfg["short_name"]
        log.info(f"[model] {short}: loading 4-bit (hf_id={model_cfg['hf_id']})...")
        try:
            with Timer(f"load {short}", logger=log):
                bundle = load_quantized_model(
                    model_cfg, cfg["quantization"],
                    device=cfg["device"], cache_dir=cfg["paths"]["cache_dir"],
                )
        except Exception as e:
            log.error(
                f"[skip] failed to load model '{short}' ({model_cfg['hf_id']}): "
                f"{type(e).__name__}: {e}"
            )
            log.error(
                "[hint] If this is a 401/GatedRepoError, you either need to "
                "(a) run `huggingface-cli login` and accept the model's "
                "license on its HF page, or (b) switch this model's hf_id "
                "in config/config.yaml to an ungated mirror."
            )
            failed_models.append(short)
            continue

        for ds_name in cfg["datasets"]:
            items_path = processed_dir / f"{ds_name}.jsonl"
            if not items_path.exists():
                log.warning(f"[skip] {items_path} missing")
                continue
            out_path = out_dir / f"{short}__{ds_name}.pt"
            if out_path.exists():
                log.info(f"[skip] {out_path} already exists")
                continue
            items = load_items_jsonl(items_path)
            if not items:
                log.warning(f"[skip] {items_path} has 0 items")
                continue
            log.info(f"[extract] {short} x {ds_name}: n={len(items)}")
            try:
                with Timer(f"{short}__{ds_name}", logger=log):
                    extract_hidden_states_for_dataset(bundle, items, out_path, ex_cfg)
            except Exception as e:
                log.exception(f"[error] {short} x {ds_name}: {e}")

        # release model VRAM
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"[model] {short}: released")

    if failed_models:
        log.warning(f"[summary] models skipped due to load failures: {failed_models}")


if __name__ == "__main__":
    main()
