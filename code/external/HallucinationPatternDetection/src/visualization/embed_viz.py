"""2-D projections of per-layer hidden states (t-SNE / UMAP / PCA).

Each function takes a [N, D] slice (a single layer's representations) and
produces a scatter plot colored by label. Figures are saved (PDF by
default, with zero margins) via
`src.visualization.style.save_paper_figure`. The output format is taken
from the extension of `out_path`, so passing a `.pdf` path yields a
vector PDF.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .style import apply_paper_style, save_paper_figure


def _scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cls0 = labels == 0
    cls1 = labels == 1
    ax.scatter(coords[cls0, 0], coords[cls0, 1],
               s=42, alpha=0.78, c="#d6604d",
               edgecolors="white", linewidths=0.4,
               label="Hallucinated (0)")
    ax.scatter(coords[cls1, 0], coords[cls1, 1],
               s=42, alpha=0.78, c="#4393c3",
               edgecolors="white", linewidths=0.4,
               label="Truthful (1)")
    ax.set_title(title, pad=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


def plot_tsne_layer(
    X: np.ndarray, y: np.ndarray,
    title: str = "t-SNE",
    perplexity: int = 30,
    seed: int = 42,
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    n = X.shape[0]
    p = max(5, min(perplexity, (n - 1) // 3))
    # scikit-learn ≥1.6 renamed n_iter → max_iter
    import inspect
    _tsne_params = inspect.signature(TSNE).parameters
    iter_key = "max_iter" if "max_iter" in _tsne_params else "n_iter"
    tsne = TSNE(
        n_components=2, perplexity=p, init="pca",
        learning_rate="auto", random_state=seed, **{iter_key: 1000},
    )
    coords = tsne.fit_transform(X)
    return _scatter(coords, y, title, out_path, dpi)


def plot_umap_layer(
    X: np.ndarray, y: np.ndarray,
    title: str = "UMAP",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("umap-learn is required for plot_umap_layer") from e
    reducer = umap.UMAP(
        n_components=2, n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
        min_dist=min_dist, random_state=seed,
    )
    coords = reducer.fit_transform(X)
    return _scatter(coords, y, title, out_path, dpi)


def plot_pca_layer(
    X: np.ndarray, y: np.ndarray,
    title: str = "PCA",
    seed: int = 42,
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return _scatter(coords, y, title, out_path, dpi)
