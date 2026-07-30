import torch
from typing import List, Optional
from utils.config import CFG


def get_sentence_embedding(
    hidden_states: torch.Tensor,
    layer_index: Optional[int] = None
) -> torch.Tensor:
    """
    Extract sentence embedding from hidden states.
    Uses last token of the middle layer (same as INSIDE paper).

    Args:
        hidden_states : (num_layers, seq_len, hidden_dim)
        layer_index   : override layer selection (None → mid layer)
    Returns:
        embedding : (hidden_dim,)
    """
    num_layers = hidden_states.shape[0]
    idx = layer_index if layer_index is not None else num_layers // 2
    # Last token embedding of the chosen layer
    return hidden_states[idx, -1, :]  # (hidden_dim,)


def build_embedding_matrix(
    hidden_states_list: List[torch.Tensor],
    layer_index: Optional[int] = None
) -> torch.Tensor:
    """
    Build Z matrix from K responses.
    Returns:
        Z : (hidden_dim, K)  — each column is one sentence embedding
    """
    embeddings = [
        get_sentence_embedding(h, layer_index) for h in hidden_states_list
    ]
    Z = torch.stack(embeddings, dim=1).float()  # (hidden_dim, K)
    return Z