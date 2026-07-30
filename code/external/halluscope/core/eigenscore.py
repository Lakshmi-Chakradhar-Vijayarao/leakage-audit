import torch
from typing import List
from core.embeddings import build_embedding_matrix
from utils.config import CFG


def compute_covariance(Z: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix of sentence embeddings.
    Args:
        Z : (hidden_dim, K)
    Returns:
        Sigma : (K, K)
    """
    d, K = Z.shape
    # Centering matrix: Jd = I_d - (1/d) * 1_d * 1_d^T
    J = torch.eye(d, device=Z.device) - (1.0 / d) * torch.ones(d, d, device=Z.device)
    Sigma = Z.T @ J @ Z  # (K, K)
    return Sigma


def compute_eigenvalues(Sigma: torch.Tensor, alpha: float = None) -> torch.Tensor:
    """
    Compute eigenvalues of regularized covariance matrix via SVD.
    Returns:
        eigenvalues : (K,) sorted descending
    """
    alpha = alpha or CFG.eigenscore.alpha
    K = Sigma.shape[0]
    reg = Sigma + alpha * torch.eye(K, device=Sigma.device)
    # SVD is more numerically stable than eig for symmetric matrices
    _, S, _ = torch.linalg.svd(reg)
    return S  # already descending


def eigenscore(
    hidden_states_list: List[torch.Tensor],
    apply_clip: bool = True,
    layer_index: int = None
) -> dict:
    """
    Compute EigenScore for a list of hidden states (K responses).
    Returns dict with score and eigenvalues for UI display.
    """
    from core.feature_clip import clip_features, update_memory_bank

    # Update memory bank and optionally clip
    processed = []
    for hs in hidden_states_list:
        update_memory_bank(hs)
        if apply_clip:
            # Clip penultimate layer, rebuild
            clipped = hs.clone()
            clipped[-2] = clip_features(hs[-2])
            processed.append(clipped)
        else:
            processed.append(hs)

    Z = build_embedding_matrix(processed, layer_index)
    Sigma = compute_covariance(Z)
    eigenvalues = compute_eigenvalues(Sigma)

    K = len(hidden_states_list)
    score = (1.0 / K) * torch.sum(torch.log(eigenvalues)).item()

    return {
        "score": score,
        "eigenvalues": eigenvalues.cpu().tolist(),
    }