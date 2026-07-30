"""Linear / MLP probes trained on per-layer hidden states.

These are the workhorse of SAPLMA-style hallucination detection: a
small classifier is fit on top of the frozen hidden representation
to predict the truthful/hallucinated label. Probe accuracy as a
function of layer index is the central diagnostic.

We keep the API sklearn-friendly *and* expose a pure PyTorch loop for
the MLP so we can use GPU when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ============================ models ===================================
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = (256, 64),
        n_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================ training =================================
@dataclass
class ProbeMetrics:
    accuracy: float
    f1: float
    auroc: float
    auprc: float
    n_train: int
    n_val: int
    n_test: int
    history: Dict[str, List[float]] = field(default_factory=dict)


def _split(
    X: np.ndarray, y: np.ndarray, test_size: float, val_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # relative val size w.r.t. remaining tv
    rel = val_size / max(1e-9, 1 - test_size)
    if rel <= 0:
        return X_tv, np.empty((0, X.shape[1])), X_te, y_tv, np.array([], dtype=int), y_te
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=rel, stratify=y_tv, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float("nan")
    return float(acc), float(f1), float(auroc), float(auprc)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    probe_type: str = "linear",
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mlp_hidden: List[int] = (256, 64),
    mlp_dropout: float = 0.2,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, ProbeMetrics]:
    """Fit a probe on (X, y) and return (model, metrics on held-out test)."""
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_tr, X_va, X_te, y_tr, y_va, y_te = _split(X, y, test_size, val_size, seed)

    in_dim = X.shape[1]
    torch.manual_seed(seed)

    if probe_type == "linear_sklearn":
        # sklearn logistic regression — fast baseline
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        acc, f1, auroc, auprc = _metrics(y_te, y_prob)
        return clf, ProbeMetrics(acc, f1, auroc, auprc, len(y_tr), len(y_va), len(y_te))

    if probe_type == "linear":
        model: nn.Module = LinearProbe(in_dim, n_classes=2)
    elif probe_type == "mlp":
        model = MLPProbe(in_dim, list(mlp_hidden), n_classes=2, dropout=mlp_dropout)
    else:
        raise ValueError(f"Unknown probe_type {probe_type}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    has_val = len(y_va) > 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_val = -1.0
    best_state: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= max(1, len(train_loader.dataset))
        history["train_loss"].append(epoch_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_va).to(device)
                yv = torch.from_numpy(y_va).to(device)
                vlogits = model(xv)
                vloss = F.cross_entropy(vlogits, yv).item()
                vprob = F.softmax(vlogits, dim=-1)[:, 1].cpu().numpy()
                try:
                    vauc = roc_auc_score(y_va, vprob)
                except ValueError:
                    vauc = float("nan")
            history["val_loss"].append(vloss)
            history["val_auroc"].append(float(vauc))
            if not np.isnan(vauc) and vauc > best_val:
                best_val = float(vauc)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if has_val:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_te).to(device)
        logits = model(xt)
        prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    acc, f1, auroc, auprc = _metrics(y_te, prob)

    return model, ProbeMetrics(
        acc, f1, auroc, auprc, len(y_tr), len(y_va), len(y_te), history
    )


def evaluate_probe(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu"
) -> Tuple[float, float, float, float]:
    """Evaluate a torch probe on (X, y) and return (acc, f1, auroc, auprc)."""
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).to(device)
        logits = model(xt)
        prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return _metrics(y, prob)


def train_layerwise_probes(
    hidden_states: np.ndarray,           # [N, L+1, D]
    labels: np.ndarray,                  # [N]
    probe_type: str = "linear",
    n_seeds: int = 3,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train one probe per layer index. Returns a dict keyed by layer.

    For each layer we run `n_seeds` repeats with different splits and
    report mean and std of every metric.
    """
    N, L1, D = hidden_states.shape
    results: Dict[str, Any] = {"layers": list(range(L1)), "per_layer": {}}
    for li in range(L1):
        Xi = hidden_states[:, li, :]
        seed_metrics = {"accuracy": [], "f1": [], "auroc": [], "auprc": []}
        for s in range(n_seeds):
            _, m = train_probe(Xi, labels, probe_type=probe_type,
                               seed=42 + s, device=device, **kwargs)
            seed_metrics["accuracy"].append(m.accuracy)
            seed_metrics["f1"].append(m.f1)
            seed_metrics["auroc"].append(m.auroc)
            seed_metrics["auprc"].append(m.auprc)
        agg = {
            f"{k}_mean": float(np.nanmean(v)) for k, v in seed_metrics.items()
        }
        agg.update({f"{k}_std": float(np.nanstd(v)) for k, v in seed_metrics.items()})
        agg["seed_values"] = seed_metrics
        results["per_layer"][li] = agg
    return results
