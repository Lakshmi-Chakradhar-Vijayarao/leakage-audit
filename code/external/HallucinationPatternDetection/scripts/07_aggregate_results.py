"""07 — aggregate all probe / INSIDE / self-consistency / attention
results into a single tidy CSV plus an executive markdown summary.

Outputs:
  results/tables/all_results.csv
  results/tables/best_per_method.csv
  results/tables/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import load_config, get_logger, ensure_dir


def _row_from_probe(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for li, m in payload.get("per_layer", {}).items():
        rows.append({
            "model": payload.get("model", ""),
            "dataset": payload.get("dataset", ""),
            "method": f"probe_{payload.get('probe_type','')}",
            "layer": int(li),
            "auroc": m.get("auroc_mean"),
            "auroc_std": m.get("auroc_std"),
            "accuracy": m.get("accuracy_mean"),
            "f1": m.get("f1_mean"),
            "auprc": m.get("auprc_mean"),
        })
    return rows


def _row_from_scalar(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": payload.get("model", ""),
        "dataset": payload.get("dataset", ""),
        "method": payload.get("method", ""),
        "layer": payload.get("layer", -1) if isinstance(payload.get("layer"), int) else -1,
        "auroc": payload.get("auroc"),
        "auprc": payload.get("auprc"),
        "accuracy": payload.get("accuracy_at_best_thr"),
        "f1": payload.get("f1_at_best_thr"),
    }


def main() -> None:
    cfg = load_config()
    log = get_logger("aggregate", log_dir=cfg["paths"]["logs"])
    probe_dir = Path(cfg["paths"]["probes"])
    metrics_dir = Path(cfg["paths"]["metrics"])
    tables_dir = ensure_dir(cfg["paths"]["tables"])

    rows: List[Dict[str, Any]] = []

    # probes -----------------------------------------------------------
    for fp in sorted(probe_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.extend(_row_from_probe(payload))

    # other methods ----------------------------------------------------
    for fp in sorted(metrics_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if fp.name.startswith("attention__"):
            best = payload.get("best_layer")
            per = payload.get("per_layer", {})
            best_m = per.get(str(best), {})
            rows.append({
                "model": payload.get("model", ""),
                "dataset": payload.get("dataset", ""),
                "method": "attention_entropy",
                "layer": int(best) if best and str(best).lstrip("-").isdigit() else -1,
                "auroc": best_m.get("auroc"),
                "auprc": best_m.get("auprc"),
                "accuracy": best_m.get("accuracy_at_best_thr"),
                "f1": best_m.get("f1_at_best_thr"),
            })
        elif fp.name.startswith("inside__"):
            rows.append(_row_from_scalar(payload))
        elif fp.name.startswith("selfconsistency__"):
            ex = payload.get("score_exact_metrics", {})
            sem = payload.get("score_semantic_metrics", {})
            rows.append({
                "model": payload.get("model", ""),
                "dataset": payload.get("dataset", ""),
                "method": "self_consistency_exact",
                "layer": -1,
                "auroc": ex.get("auroc"), "auprc": ex.get("auprc"),
                "accuracy": ex.get("accuracy_at_best_thr"),
                "f1": ex.get("f1_at_best_thr"),
            })
            rows.append({
                "model": payload.get("model", ""),
                "dataset": payload.get("dataset", ""),
                "method": "self_consistency_semantic",
                "layer": -1,
                "auroc": sem.get("auroc"), "auprc": sem.get("auprc"),
                "accuracy": sem.get("accuracy_at_best_thr"),
                "f1": sem.get("f1_at_best_thr"),
            })

    if not rows:
        log.warning("[empty] no results to aggregate")
        return

    df = pd.DataFrame(rows)
    all_path = tables_dir / "all_results.csv"
    df.to_csv(all_path, index=False)
    log.info(f"[saved] {all_path}  ({len(df)} rows)")

    # best per (model, dataset, method)
    df_valid = df.dropna(subset=["auroc"])
    if not df_valid.empty:
        idx = df_valid.groupby(["model", "dataset", "method"])["auroc"].idxmax()
        best = df_valid.loc[idx].reset_index(drop=True)
        best_path = tables_dir / "best_per_method.csv"
        best.to_csv(best_path, index=False)
        log.info(f"[saved] {best_path}")
    else:
        best = df_valid

    # markdown summary -------------------------------------------------
    md_lines: List[str] = ["# Hallucination Pattern Detection — results summary\n"]
    if not best.empty:
        for ds in sorted(best["dataset"].unique()):
            md_lines.append(f"## Dataset: {ds}\n")
            sub = best[best["dataset"] == ds].sort_values(["model", "method"])
            md_lines.append(sub[["model", "method", "layer", "auroc", "accuracy", "f1"]].to_markdown(index=False))
            md_lines.append("")
    md_path = tables_dir / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
