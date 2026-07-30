"""Metric helpers shared across detection methods."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def score_to_metrics(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Treat `score` as a continuous hallucination signal (higher = more
    hallucinated) and reduce to binary metrics by optimal-threshold search.

    Returns AUROC/AUPRC on the *raw* score, plus accuracy/F1 at the
    Youden-optimal threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    # because score = hallucination signal, label of hallucinated = 0,
    # we predict "hallucinated" when score is high — i.e. we ROC over (1 - y_true)
    target = 1 - y_true
    try:
        auroc = float(roc_auc_score(target, score))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(target, score))
    except ValueError:
        auprc = float("nan")
    # Youden-J optimal threshold
    try:
        fpr, tpr, thr = roc_curve(target, score)
        j = tpr - fpr
        best = int(np.argmax(j))
        thr_best = float(thr[best])
        y_pred_hall = (score >= thr_best).astype(int)        # 1 = hallucinated
        y_pred_truth = 1 - y_pred_hall                       # back to truthful=1
        acc = float(accuracy_score(y_true, y_pred_truth))
        f1 = float(f1_score(y_true, y_pred_truth, zero_division=0))
    except Exception:
        thr_best, acc, f1 = float("nan"), float("nan"), float("nan")
    return {
        "auroc": auroc, "auprc": auprc,
        "accuracy_at_best_thr": acc, "f1_at_best_thr": f1,
        "best_threshold": thr_best,
    }


def summarize_layerwise(layerwise: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a layerwise-probe result to a flat summary: best layer, peak AUROC, etc."""
    per_layer = layerwise.get("per_layer", {})
    layers = sorted(int(k) for k in per_layer.keys())
    aurocs = [per_layer[str(li)]["auroc_mean"] if str(li) in per_layer else per_layer[li]["auroc_mean"]
              for li in layers]
    accs = [per_layer[str(li)]["accuracy_mean"] if str(li) in per_layer else per_layer[li]["accuracy_mean"]
            for li in layers]
    if not aurocs:
        return {"best_layer": -1, "best_auroc": float("nan"), "best_accuracy": float("nan")}
    best_idx = int(np.nanargmax(aurocs))
    return {
        "layers": layers,
        "auroc_per_layer": aurocs,
        "accuracy_per_layer": accs,
        "best_layer": layers[best_idx],
        "best_auroc": float(aurocs[best_idx]),
        "best_accuracy": float(accs[best_idx]),
    }


def aggregate_across_models(
    per_model: Dict[str, Dict[str, Any]],
    metric: str = "best_auroc",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, summary in per_model.items():
        out[name] = float(summary.get(metric, float("nan")))
    return out
