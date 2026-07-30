"""04 — INSIDE / semantic entropy from K stochastic completions.

For each (model, dataset) we sample K completions per prompt, capture
the last-token hidden state of a chosen layer for each completion, and
compute the EigenScore (log-det of regularized covariance).

Outputs:
  results/metrics/inside__<model>__<dataset>.json
"""
from __future__ import annotations

import sys
import gc
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.utils import load_config, set_seed, get_logger, ensure_dir, save_json, Timer
from src.data.dataset_loader import load_items_jsonl
from src.models.model_loader import load_quantized_model
from src.detection.inside_method import inside_batch
from src.analysis.metrics import score_to_metrics


@torch.inference_mode()
def _sample_hidden(bundle, prompt: str, K: int, max_new_tokens: int,
                   temperature: float, top_p: float, layer: int) -> np.ndarray:
    """Return [K, D] hidden states of the *last generated token* at `layer`."""
    tok = bundle.tokenizer
    enc = tok(prompt, return_tensors="pt").to(bundle.device)
    prompt_len = enc["input_ids"].shape[1]
    out = bundle.model.generate(
        **enc,
        do_sample=True,
        temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=K,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )
    steps = out.hidden_states
    if not steps:
        return np.zeros((K, bundle.hidden_size), dtype=np.float32)
    last_step = steps[-1]
    li = layer if layer >= 0 else len(last_step) + layer
    h = last_step[li]
    if h.shape[1] > 1:
        h = h[:, -1, :]
    else:
        h = h.squeeze(1)
    return h.float().cpu().numpy()


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    log = get_logger("inside", log_dir=cfg["paths"]["logs"])

    processed_dir = Path(cfg["paths"]["data_processed"])
    out_dir = ensure_dir(cfg["paths"]["metrics"])
    ins_cfg = cfg["inside"]; gen_cfg = cfg["generation"]

    failed_models: list = []
    for model_cfg in cfg["models"]:
        short = model_cfg["short_name"]
        log.info(f"[model] {short}: loading 4-bit for INSIDE (hf_id={model_cfg['hf_id']})...")
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
                "[hint] If this is a 401/GatedRepoError, run `huggingface-cli login` "
                "or switch to an ungated mirror in config/config.yaml."
            )
            failed_models.append(short)
            continue

        for ds_name in cfg["datasets"]:
            items_path = processed_dir / f"{ds_name}.jsonl"
            if not items_path.exists():
                continue
            out_path = out_dir / f"inside__{short}__{ds_name}.json"
            if out_path.exists():
                log.info(f"[skip] {out_path.name} already exists")
                continue
            items = load_items_jsonl(items_path)
            if not items:
                log.warning(f"[skip] {items_path} has 0 items")
                continue
            scores: List[float] = []
            labels: List[int] = []
            log.info(f"[inside] {short} x {ds_name}: n={len(items)}")
            for it in tqdm(items, desc=f"inside:{short}:{ds_name}"):
                try:
                    H = _sample_hidden(
                        bundle, it.prompt, K=ins_cfg["n_samples"],
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
                        layer=ins_cfg["layer"],
                    )
                except Exception:
                    H = np.zeros((ins_cfg["n_samples"], bundle.hidden_size), dtype=np.float32)
                s = inside_batch([H], alpha=ins_cfg["alpha"], mode="eigen")[0]
                scores.append(float(s))
                labels.append(int(it.label))

            scores_arr = np.array(scores)
            labels_arr = np.array(labels)
            metrics = score_to_metrics(labels_arr, scores_arr)
            save_json({
                "model": short, "dataset": ds_name,
                "method": "INSIDE",
                "layer": ins_cfg["layer"],
                "n_samples_per_prompt": ins_cfg["n_samples"],
                "scores": scores_arr.tolist(),
                "labels": labels_arr.tolist(),
                **metrics,
            }, out_path)
            log.info(f"[done] {out_path.name}  AUROC={metrics['auroc']:.3f}")

        del bundle; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if failed_models:
        log.warning(f"[summary] models skipped due to load failures: {failed_models}")


if __name__ == "__main__":
    main()
