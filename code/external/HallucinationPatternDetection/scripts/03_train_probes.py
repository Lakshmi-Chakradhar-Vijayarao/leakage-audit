"""03 — train SAPLMA-style probes on every saved hidden-state file.

Outputs:
  results/probes/<model>__<dataset>__<probe>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.utils import load_config, set_seed, get_logger, ensure_dir, save_json, Timer
from src.detection.probes import train_layerwise_probes
from src.detection.saplma import saplma_probe_per_layer


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    log = get_logger("train_probes", log_dir=cfg["paths"]["logs"])

    hs_dir = Path(cfg["paths"]["hidden_states"])
    out_dir = ensure_dir(cfg["paths"]["probes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_cfg = cfg["probes"]

    files = sorted(hs_dir.glob("*.pt"))
    if not files:
        log.warning(f"[empty] no hidden state files in {hs_dir}")
        return

    for fp in files:
        log.info(f"[load] {fp.name}")
        data = torch.load(fp, map_location="cpu")
        X = data["hidden_states"].numpy()         # [N, L+1, D]
        y = data["labels"].numpy()                # [N]
        if len(np.unique(y)) < 2:
            log.warning(f"[skip] {fp.name} has only one class")
            continue

        for ptype in p_cfg["types"]:
            out_path = out_dir / f"{fp.stem}__{ptype}.json"
            if out_path.exists():
                log.info(f"[skip] {out_path.name} already exists")
                continue
            log.info(f"[probe] {fp.stem} | {ptype}")
            with Timer(f"probe {fp.stem} {ptype}", logger=log):
                result = saplma_probe_per_layer(
                    X, y,
                    probe_type=ptype,
                    n_seeds=p_cfg["n_seeds"],
                    device=device,
                    test_size=p_cfg["test_size"],
                    val_size=p_cfg["val_size"],
                    epochs=p_cfg["epochs"],
                    batch_size=p_cfg["batch_size"],
                    lr=p_cfg["lr"],
                    weight_decay=p_cfg["weight_decay"],
                    mlp_hidden=p_cfg["mlp_hidden"],
                    mlp_dropout=p_cfg["mlp_dropout"],
                )
            result["model"] = data.get("model_short_name", "")
            result["dataset"] = fp.stem.split("__", 1)[1] if "__" in fp.stem else ""
            result["probe_type"] = ptype
            save_json(result, out_path)
            log.info(
                f"[done] {out_path.name}  best_layer={result['best_layer']} "
                f"best_auroc={result['best_auroc']:.3f}"
            )


if __name__ == "__main__":
    main()
