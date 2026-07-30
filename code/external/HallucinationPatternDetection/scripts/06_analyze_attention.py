"""06 — analyze the attention-entropy signal collected during extraction.

For every saved hidden-state file with an `attention_entropy` tensor,
score AUROC of the head-averaged entropy as a hallucination signal.

Outputs:
  results/metrics/attention__<model>__<dataset>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.utils import load_config, get_logger, ensure_dir, save_json
from src.analysis.metrics import score_to_metrics


def main() -> None:
    cfg = load_config()
    log = get_logger("attention_analysis", log_dir=cfg["paths"]["logs"])
    hs_dir = Path(cfg["paths"]["hidden_states"])
    out_dir = ensure_dir(cfg["paths"]["metrics"])

    for fp in sorted(hs_dir.glob("*.pt")):
        data = torch.load(fp, map_location="cpu")
        if "attention_entropy" not in data:
            continue
        ent = data["attention_entropy"].numpy()    # [N, L_keep, H]
        labels = data["labels"].numpy()
        layers_kept = data.get("attention_layers_kept", list(range(ent.shape[1])))
        # one AUROC per kept layer (head-averaged entropy)
        per_layer = {}
        per_sample = ent.mean(axis=-1)             # [N, L_keep]
        for li in range(ent.shape[1]):
            m = score_to_metrics(labels, per_sample[:, li])
            per_layer[str(layers_kept[li])] = m
        # best layer
        best = max(per_layer, key=lambda k: per_layer[k]["auroc"] if not np.isnan(per_layer[k]["auroc"]) else -1)
        model_short = data.get("model_short_name", "")
        dataset_name = fp.stem.split("__", 1)[1] if "__" in fp.stem else ""
        out_path = out_dir / f"attention__{fp.stem}.json"
        save_json({
            "model": model_short, "dataset": dataset_name,
            "method": "attention_entropy",
            "per_layer": per_layer,
            "best_layer": best,
            "best_auroc": per_layer[best]["auroc"],
        }, out_path)
        log.info(f"[done] {out_path.name}  best_layer={best}  auroc={per_layer[best]['auroc']:.3f}")


if __name__ == "__main__":
    main()
