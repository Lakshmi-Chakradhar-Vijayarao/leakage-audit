"""Layer-wise diagnostic plots.

All figures saved by this module are written in vector PDF (when callers
pass a `.pdf` extension) with zero margins on all four borders.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_paper_style, save_paper_figure


def plot_layerwise_auroc(
    layerwise: Dict[str, Dict],     # per_layer dict, keyed by layer index (int or str)
    title: str = "Probe AUROC vs layer",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    apply_paper_style()
    layers = sorted([int(k) for k in layerwise.keys()])
    auroc = [layerwise[k if k in layerwise else str(k)]["auroc_mean"] for k in layers]
    std = [layerwise[k if k in layerwise else str(k)]["auroc_std"] for k in layers]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    auroc = np.array(auroc); std = np.array(std)
    ax.plot(layers, auroc, "o-", color="#1f78b4", label="AUROC",
            linewidth=2.6, markersize=8)
    ax.fill_between(layers, auroc - std, auroc + std, alpha=0.25, color="#1f78b4")
    ax.axhline(0.5, ls="--", color="grey", alpha=0.65, label="chance", linewidth=1.8)
    ax.set_xlabel("layer index (0 = embeddings)")
    ax.set_ylabel("AUROC")
    ax.set_title(title, pad=10)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


def plot_layerwise_heatmap(
    matrix: np.ndarray,             # [n_models, n_layers]
    row_labels: Sequence[str],
    col_labels: Sequence[int],
    title: str = "Layer-wise AUROC across models",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.42),
                                    max(3.4, len(row_labels) * 0.75 + 1.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    # Show ~10 layer ticks max so the labels stay legible at the bigger
    # font sizes (matplotlib otherwise overlays them).
    step = max(1, len(col_labels) // 10)
    ticks = list(range(0, len(col_labels), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([col_labels[i] for i in ticks])
    ax.set_xlabel("layer index")
    ax.set_title(title, pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("AUROC", fontsize=18)
    cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


def plot_separation_ratio(
    geom: Dict[str, np.ndarray],
    title: str = "Class separation across layers",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    apply_paper_style()
    layers = np.arange(len(geom["centroid_distance"]))
    # A bit taller than strictly necessary so the suptitle has breathing
    # room above the two subplot titles at the larger font sizes.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    ax.plot(layers, geom["centroid_distance"], color="#1f78b4",
            label=r"$\|\mu_1-\mu_0\|$", linewidth=2.6)
    ax.plot(layers, geom["spread_truthful"], color="#4393c3", ls="--",
            label="spread (truthful)", linewidth=2.4)
    ax.plot(layers, geom["spread_hallucinated"], color="#d6604d", ls="--",
            label="spread (hallucinated)", linewidth=2.4)
    ax.set_xlabel("layer index"); ax.set_ylabel("distance (L2)")
    ax.legend(frameon=False, loc="best")
    ax.set_title("Centroid distance vs within-class spread", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(layers, geom["separation_ratio"], color="#33a02c", linewidth=2.6)
    ax.set_xlabel("layer index"); ax.set_ylabel("separation ratio")
    ax.set_title("Centroid distance / mean within-class spread", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig
