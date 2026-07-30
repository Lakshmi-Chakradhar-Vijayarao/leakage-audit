"""Higher-level analyses operating on saved hidden states + probe results.

These produce summary tables: per-dataset detection performance,
cross-model comparison, and intrinsic layer-geometry statistics
(separation between truthful and hallucinated clouds).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


# ----------------------- per-dataset summary ---------------------------
def per_dataset_summary(
    results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a tidy DataFrame of (model, dataset, method, layer, metric).

    `results` shape:
      results[model_short_name][dataset_name][method] = layerwise_dict
    """
    rows: List[Dict[str, Any]] = []
    for model, by_ds in results.items():
        for ds, by_method in by_ds.items():
            for method, payload in by_method.items():
                if "per_layer" in payload:
                    for li, m in payload["per_layer"].items():
                        rows.append({
                            "model": model, "dataset": ds, "method": method,
                            "layer": int(li),
                            "auroc": m.get("auroc_mean"),
                            "auroc_std": m.get("auroc_std"),
                            "accuracy": m.get("accuracy_mean"),
                            "f1": m.get("f1_mean"),
                            "auprc": m.get("auprc_mean"),
                        })
                else:
                    rows.append({
                        "model": model, "dataset": ds, "method": method,
                        "layer": payload.get("layer", -1),
                        **{k: payload[k] for k in ("auroc", "accuracy", "f1", "auprc") if k in payload},
                    })
    return pd.DataFrame(rows)


# --------------------- cross-model comparison --------------------------
def cross_model_comparison(
    summary_df: pd.DataFrame,
    metric: str = "auroc",
) -> pd.DataFrame:
    """For each (model, dataset, method) take the *best layer* and pivot
    into a comparison table."""
    if summary_df.empty:
        return summary_df
    keep = summary_df.dropna(subset=[metric])
    if keep.empty:
        return keep
    idx = keep.groupby(["model", "dataset", "method"])[metric].idxmax()
    best = keep.loc[idx].reset_index(drop=True)
    return best


# --------------------- layer geometry analytics ------------------------
def layer_geometry_stats(
    hidden_states: np.ndarray,     # [N, L+1, D]
    labels: np.ndarray,            # [N]
) -> Dict[str, np.ndarray]:
    """Per-layer geometric separation between truthful (1) and hallucinated (0).

    Returns:
      mean_distance_between_class_centroids  : [L+1]
      within_class_spread_truthful           : [L+1]
      within_class_spread_hallucinated       : [L+1]
      separation_ratio                       : centroid_dist / mean_within
    """
    N, L1, D = hidden_states.shape
    pos = labels == 1
    neg = labels == 0
    cd: List[float] = []
    sp_pos: List[float] = []
    sp_neg: List[float] = []
    for li in range(L1):
        Xi = hidden_states[:, li, :]
        mu_pos = Xi[pos].mean(axis=0) if pos.any() else np.zeros(D)
        mu_neg = Xi[neg].mean(axis=0) if neg.any() else np.zeros(D)
        cd.append(float(np.linalg.norm(mu_pos - mu_neg)))
        sp_pos.append(
            float(np.linalg.norm(Xi[pos] - mu_pos, axis=1).mean()) if pos.any() else 0.0
        )
        sp_neg.append(
            float(np.linalg.norm(Xi[neg] - mu_neg, axis=1).mean()) if neg.any() else 0.0
        )
    cd_a = np.array(cd); sp_a = (np.array(sp_pos) + np.array(sp_neg)) / 2
    ratio = cd_a / np.where(sp_a > 0, sp_a, 1.0)
    return {
        "centroid_distance": cd_a,
        "spread_truthful": np.array(sp_pos),
        "spread_hallucinated": np.array(sp_neg),
        "separation_ratio": ratio,
    }
