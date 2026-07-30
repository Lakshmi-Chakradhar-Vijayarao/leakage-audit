"""Visualizations for attention-pattern hallucination signals.

Figures are saved via the shared `save_paper_figure` helper so the output
file has zero margins on all four borders and (when the extension is
`.pdf`) is a vector graphic.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_paper_style, save_paper_figure


def plot_attention_entropy(
    entropy_tensor: np.ndarray,   # [N, L_keep, H]
    labels: np.ndarray,           # [N]
    layer_labels: Sequence[int],
    title: str = "Attention entropy (last answer token, per kept layer)",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """For each kept layer, plot the head-averaged entropy distribution
    split by label."""
    apply_paper_style()
    N, L, H = entropy_tensor.shape
    # Average over heads -> per-sample, per-layer scalar
    per_sample = entropy_tensor.mean(axis=-1)    # [N, L]
    pos = labels == 1
    fig, axes = plt.subplots(1, L, figsize=(5.2 * L, 5.0), sharey=True)
    if L == 1:
        axes = [axes]
    for li, ax in enumerate(axes):
        ax.hist(per_sample[pos, li], bins=30, alpha=0.7,
                color="#4393c3", label="Truthful", edgecolor="white", linewidth=0.4)
        ax.hist(per_sample[~pos, li], bins=30, alpha=0.7,
                color="#d6604d", label="Hallucinated", edgecolor="white", linewidth=0.4)
        ax.set_title(f"layer {layer_labels[li]}", pad=6)
        ax.set_xlabel("entropy")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if li == 0:
            ax.set_ylabel("count")
            ax.legend(frameon=False, loc="best")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


def plot_attention_heatmap(
    attn_matrix: np.ndarray,         # [T, T]
    tokens: Sequence[str],
    title: str = "Attention",
    out_path: Optional[str] = None,
    dpi: int = 300,
    max_tokens: int = 64,
) -> plt.Figure:
    """Simple [T, T] attention heatmap for qualitative inspection."""
    apply_paper_style()
    T = min(attn_matrix.shape[0], max_tokens)
    A = attn_matrix[:T, :T]
    fig, ax = plt.subplots(figsize=(max(8, T * 0.2), max(8, T * 0.2)))
    im = ax.imshow(A, cmap="magma", aspect="auto")
    # Token labels can be very dense (one tick per token); keep them small,
    # but large enough to stay legible after column scaling.
    ax.set_xticks(range(T)); ax.set_xticklabels(tokens[:T], rotation=90, fontsize=10)
    ax.set_yticks(range(T)); ax.set_yticklabels(tokens[:T], fontsize=10)
    ax.set_title(title, pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig
