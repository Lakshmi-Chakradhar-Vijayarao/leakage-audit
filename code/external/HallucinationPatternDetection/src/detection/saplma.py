"""SAPLMA — Statement Accuracy Prediction from Language Model Activations.

Azaria & Mitchell 2023. SAPLMA is functionally a per-layer probe over the
*final-token* hidden representation of a statement, where the label is
whether the statement is true or false. Our `train_layerwise_probes`
already implements this on the per-(prompt, answer) hidden states. The
helper below is a thin convenience wrapper that names the result
"SAPLMA" and pulls out the *best-layer* operating point for reporting.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .probes import train_layerwise_probes


def saplma_probe_per_layer(
    hidden_states: np.ndarray,   # [N, L+1, D]
    labels: np.ndarray,          # [N]
    probe_type: str = "mlp",
    n_seeds: int = 3,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """SAPLMA = per-layer probe; returns layerwise dict + best layer summary."""
    layerwise = train_layerwise_probes(
        hidden_states, labels, probe_type=probe_type,
        n_seeds=n_seeds, device=device, **kwargs,
    )
    # determine the best layer by AUROC mean.
    best_layer = -1
    best_auc = -np.inf
    for li, m in layerwise["per_layer"].items():
        v = m.get("auroc_mean", float("nan"))
        if not np.isnan(v) and v > best_auc:
            best_auc = v
            best_layer = int(li)
    layerwise["best_layer"] = best_layer
    layerwise["best_auroc"] = float(best_auc)
    layerwise["method"] = "SAPLMA"
    return layerwise
