"""
SpectralEntropy — Novel hallucination detection metric.

Motivation:
    EigenScore uses the log-determinant (sum of log eigenvalues) as a
    divergence measure. However, it is sensitive to the *magnitude* of
    eigenvalues and can be dominated by a single large eigenvalue.

    SpectralEntropy instead normalizes eigenvalues into a probability
    distribution and computes Shannon entropy over them.  This captures
    *how uniformly* the variance is spread across the embedding dimensions:

    - Low SpectralEntropy  → one dominant eigenvalue → responses are
      semantically clustered → model is confident → less hallucination.
    - High SpectralEntropy → eigenvalues uniformly distributed → responses
      span many directions → model is uncertain → likely hallucination.

    This is complementary to EigenScore:
    EigenScore measures total semantic spread (volume), while
    SpectralEntropy measures the uniformity of that spread (shape).
"""

import torch
import numpy as np
from typing import List
from core.embeddings import build_embedding_matrix
from core.eigenscore import compute_covariance, compute_eigenvalues
from utils.config import CFG


def spectral_entropy(
    hidden_states_list: List[torch.Tensor],
    apply_clip: bool = True,
    layer_index: int = None,
) -> dict:
    """
    Compute SpectralEntropy for K responses.

    Returns:
        score        : float, higher → more hallucination
        eigenvalues  : raw eigenvalues
        probs        : normalized eigenvalue distribution (for visualization)
        max_entropy  : log(K), theoretical max entropy (for normalization)
        normalized_score : score / max_entropy ∈ [0, 1]
    """
    from core.feature_clip import clip_features, update_memory_bank

    eps = CFG.spectral.epsilon

    processed = []
    for hs in hidden_states_list:
        update_memory_bank(hs)
        if apply_clip:
            clipped = hs.clone()
            clipped[-2] = clip_features(hs[-2])
            processed.append(clipped)
        else:
            processed.append(hs)

    Z = build_embedding_matrix(processed, layer_index)
    Sigma = compute_covariance(Z)
    eigenvalues = compute_eigenvalues(Sigma, alpha=CFG.spectral.alpha)

    # Normalize to probability distribution
    eig_pos = torch.clamp(eigenvalues, min=0.0)
    total = eig_pos.sum() + eps
    probs = (eig_pos / total).cpu().numpy()

    # Shannon entropy: H = -sum(p * log(p))
    log_probs = np.log(probs + eps)
    entropy = float(-np.sum(probs * log_probs))

    K = len(hidden_states_list)
    max_entropy = float(np.log(K))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "score": entropy,
        "normalized_score": normalized,
        "eigenvalues": eigenvalues.cpu().tolist(),
        "probs": probs.tolist(),
        "max_entropy": max_entropy,
    }


def combined_score(eigen_result: dict, spectral_result: dict, w1: float = 0.5, w2: float = 0.5) -> float:
    """
    Weighted combination of EigenScore and SpectralEntropy.
    Both are normalized to [0,1] before combining.
    EigenScore is negated (more negative = less hallucination).
    """
    # Normalize EigenScore: clip to [-10, 0] range then scale
    e_norm = float(np.clip(-eigen_result["score"] / 10.0, 0, 1))
    s_norm = float(spectral_result["normalized_score"])
    # Higher combined score → more likely hallucination
    return w1 * (1 - e_norm) + w2 * s_norm