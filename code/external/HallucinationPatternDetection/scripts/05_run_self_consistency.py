"""05 — self-consistency hallucination score (K stochastic samples).

Outputs:
  results/metrics/selfconsistency__<model>__<dataset>.json
"""
from __future__ import annotations

import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.utils import load_config, set_seed, get_logger, ensure_dir, save_json, Timer
from src.data.dataset_loader import load_items_jsonl
from src.models.model_loader import load_quantized_model
from src.detection.self_consistency import run_self_consistency
from src.analysis.metrics import score_to_metrics


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    log = get_logger("self_consistency", log_dir=cfg["paths"]["logs"])
    sc_cfg = cfg["self_consistency"]
    gen_cfg = cfg["generation"]
    processed_dir = Path(cfg["paths"]["data_processed"])
    out_dir = ensure_dir(cfg["paths"]["metrics"])

    failed_models: list = []
    for model_cfg in cfg["models"]:
        short = model_cfg["short_name"]
        log.info(f"[model] {short}: loading 4-bit for self-consistency (hf_id={model_cfg['hf_id']})...")
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
            out_path = out_dir / f"selfconsistency__{short}__{ds_name}.json"
            if out_path.exists():
                log.info(f"[skip] {out_path.name} already exists")
                continue
            items = load_items_jsonl(items_path)
            if not items:
                log.warning(f"[skip] {items_path} has 0 items")
                continue
            log.info(f"[sc] {short} x {ds_name}: n={len(items)}")
            try:
                with Timer(f"sc {short}__{ds_name}", logger=log):
                    res = run_self_consistency(
                        bundle, items,
                        n_samples=sc_cfg["n_samples"],
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                        embedding_model=sc_cfg["embedding_model"],
                    )
            except Exception:
                log.exception(f"[error] {short} x {ds_name}")
                continue

            recs = res["records"]
            score_exact = np.array([r["hallucination_score_exact"] for r in recs])
            score_sem = np.array([r["hallucination_score_semantic"] for r in recs])
            labels = np.array([r["label"] for r in recs])
            metrics_exact = score_to_metrics(labels, score_exact)
            # semantic might contain NaNs if embedder failed
            mask = ~np.isnan(score_sem)
            metrics_sem = (
                score_to_metrics(labels[mask], score_sem[mask])
                if mask.sum() > 1 else
                {"auroc": float("nan"), "auprc": float("nan")}
            )
            save_json({
                "model": short, "dataset": ds_name,
                "method": "self_consistency",
                "n_samples_per_prompt": sc_cfg["n_samples"],
                "records": recs,
                "score_exact_metrics": metrics_exact,
                "score_semantic_metrics": metrics_sem,
            }, out_path)
            log.info(
                f"[done] {out_path.name}  "
                f"AUROC(exact)={metrics_exact['auroc']:.3f}  "
                f"AUROC(sem)={metrics_sem['auroc']:.3f}"
            )
        del bundle; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if failed_models:
        log.warning(f"[summary] models skipped due to load failures: {failed_models}")


if __name__ == "__main__":
    main()
