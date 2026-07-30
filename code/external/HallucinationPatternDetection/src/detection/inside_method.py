"""INSIDE-style semantic entropy from hidden states.

Implementation follows the spirit of Chen et al., "INSIDE: LLMs'
Internal States Retain the Power of Hallucination Detection"
(2024). For each input we collect K stochastic completions, take
the hidden state of the last generated token at a target layer
for each completion, form the K x D matrix, and use the log
determinant of its regularized covariance as a hallucination score
(higher = more diverse internal states = more likely hallucinated).
We expose both the EigenScore (log-det) and a simpler variance score.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def compute_inside_score(
    hidden_matrix: np.ndarray,  # [K, D]
    alpha: float = 1e-3,
    mode: str = "eigen",
) -> float:
    """Compute INSIDE-style score from a K x D matrix of hidden vectors.

    mode:
      "eigen"   - log-det of regularized covariance (recommended);
                  higher = more semantic diversity = more uncertain.
      "trace"   - trace of covariance (sum of variances).
      "fronorm" - Frobenius norm of the centered matrix.
    """
    if hidden_matrix.ndim != 2:
        raise ValueError(f"hidden_matrix must be 2D, got {hidden_matrix.shape}")
    K, D = hidden_matrix.shape
    if K < 2:
        return 0.0
    X = hidden_matrix - hidden_matrix.mean(axis=0, keepdims=True)
    # cov in K-space ([K,K]) is cheaper than D-space when D >> K, and
    # has the same non-zero eigenvalues.
    C = X @ X.T / max(1, D)
    C = C + alpha * np.eye(K, dtype=C.dtype)
    if mode == "eigen":
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            # extremely rare numerical edge case
            return float(np.linalg.norm(X))
        return float(logdet)
    if mode == "trace":
        return float(np.trace(C))
    if mode == "fronorm":
        return float(np.linalg.norm(X))
    raise ValueError(f"Unknown mode {mode}")


def inside_batch(
    hidden_per_sample: Sequence[np.ndarray],  # list of [K, D]
    alpha: float = 1e-3,
    mode: str = "eigen",
) -> np.ndarray:
    """Apply `compute_inside_score` to a list of K x D matrices.

    Returns an array of length len(hidden_per_sample).
    """
    return np.array(
        [compute_inside_score(h, alpha=alpha, mode=mode) for h in hidden_per_sample],
        dtype=np.float64,
    )
