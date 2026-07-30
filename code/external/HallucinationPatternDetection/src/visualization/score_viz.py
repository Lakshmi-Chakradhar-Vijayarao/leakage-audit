"""Score-distribution, ROC, and cross-method comparison plots.

All saved figures use the shared `save_paper_figure` helper so they land
on disk with zero margins on every border, and -- when the caller passes
a `.pdf` extension -- as vector graphics.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .style import apply_paper_style, save_paper_figure


def plot_score_distribution(
    score: np.ndarray, labels: np.ndarray,
    title: str = "Score distribution by label",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    pos = labels == 1
    ax.hist(score[pos], bins=30, alpha=0.7, color="#4393c3",
            label="Truthful", edgecolor="white", linewidth=0.4)
    ax.hist(score[~pos], bins=30, alpha=0.7, color="#d6604d",
            label="Hallucinated", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("hallucination score (higher = more hallucinated)")
    ax.set_ylabel("count")
    ax.set_title(title, pad=10)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


def plot_roc_curves(
    method_scores: Dict[str, np.ndarray],   # {method: score per sample (higher = hallucinated)}
    labels: np.ndarray,                     # 1 = truthful, 0 = hallucinated
    title: str = "ROC curves",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Treat each `score` as a hallucination signal. Target is (1 - labels)."""
    apply_paper_style()
    target = 1 - labels
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2"]
    for i, (name, sc) in enumerate(method_scores.items()):
        try:
            fpr, tpr, _ = roc_curve(target, sc)
            ax.plot(fpr, tpr, label=name, linewidth=2.6,
                    color=palette[i % len(palette)])
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.65, linewidth=1.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, pad=10)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig


# ------------------------------------------------------------------ helpers
# Concise display names for the legend. Raw method names like
# ``self_consistency_semantic`` are wider than the chart itself, and an
# oversized legend (combined with ``bbox_inches="tight"``) stretches the
# saved figure horizontally and squashes the bars.
_METHOD_DISPLAY = {
    "linear":                    "linear probe",
    "probe_linear":              "linear probe",
    "mlp":                       "MLP probe",
    "probe_mlp":                 "MLP probe",
    "inside":                    "INSIDE",
    "INSIDE":                    "INSIDE",
    "self_consistency":          "self-consistency",
    "self_consistency_exact":    "self-cons. (exact)",
    "self_consistency_semantic": "self-cons. (sem.)",
    "attention":                 "attn. entropy",
    "attention_entropy":         "attn. entropy",
}

# Preferred ordering. We list both the raw aggregator names and the
# probe-prefixed variants so the table sorts correctly regardless of
# which writer produced the CSV.
_METHOD_ORDER = [
    "probe_linear", "linear",
    "probe_mlp",    "mlp",
    "INSIDE",       "inside",
    "self_consistency_exact",
    "self_consistency_semantic",
    "self_consistency",
    "attention_entropy", "attention",
]


def plot_method_comparison(
    df: pd.DataFrame,
    metric: str = "auroc",
    title: str = "Method comparison",
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Grouped bar chart of ``metric`` per (model x method) for one dataset.

    Expects ``df`` to contain at least the columns ``[model, method, <metric>]``.
    The caller is expected to filter ``df`` to a single dataset and put
    the dataset name in ``title``, so the x-axis only carries short model
    names (drawn horizontally).

    Implementation notes:

    * Long method names like ``self_consistency_semantic`` are abbreviated
      in the legend via ``_METHOD_DISPLAY``. Without this, the legend
      becomes wider than the chart and ``bbox_inches="tight"`` then
      stretches the saved figure horizontally, squashing the bars to a
      single x position.
    * The legend sits below the axes in compact rows, capped at three
      columns, so its bounding box never exceeds the chart width.
    * No per-bar value labels are drawn: at 6+ methods per model the
      bars are narrow enough that the labels would stack on top of each
      other. The exact numbers live in ``results/tables/all_results.csv``.
    """
    apply_paper_style()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=18)
        ax.set_axis_off()
        save_paper_figure(fig, out_path, dpi=dpi)
        return fig

    # Aggregate over any remaining dimensions (layers, seeds, ...) so each
    # cell of the pivot is the best score that method achieved per model.
    pivot = (
        df.pivot_table(index="model", columns="method",
                       values=metric, aggfunc="max")
          .fillna(0.0)
    )
    # Stable column order across runs.
    ordered = [c for c in _METHOD_ORDER if c in pivot.columns]
    ordered += sorted([c for c in pivot.columns if c not in ordered])
    pivot = pivot[ordered]
    # Display-name rename only happens for the legend; the underlying
    # values are unchanged.
    pivot = pivot.rename(columns=_METHOD_DISPLAY)

    n_models = len(pivot)
    n_methods = len(pivot.columns)

    # Figure dimensions tuned so each individual bar has a few tenths of
    # an inch of width even with 6+ methods per model. Capped so
    # downstream scaling does not make text microscopic.
    fig_w = max(9.0, min(13.0, 1.6 * n_models + 1.0 * n_methods))
    fig_h = 5.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Stable qualitative palette (matplotlib tab10 picks); same method
    # gets the same colour across datasets.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    colors = palette[:n_methods]

    pivot.plot(
        kind="bar", ax=ax, color=colors,
        edgecolor="black", linewidth=0.6, width=0.82,
    )

    ax.set_ylabel(metric.upper())
    ax.set_xlabel("")  # model names are self-explanatory under the bars
    ax.set_title(title, pad=10)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(
        0.5, ls="--", color="grey", alpha=0.65, linewidth=1.8, label="chance",
    )
    # Horizontal x-tick labels -- model short names fit without rotation.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.tick_params(axis="x", pad=6)
    ax.yaxis.grid(True, ls=":", color="grey", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Legend below the axes, in compact rows. ncol is chosen so that the
    # legend never exceeds the chart width: with abbreviated method names
    # at most ~18 chars wide, 3 columns is the comfortable ceiling --
    # otherwise ``bbox_inches="tight"`` would expand the saved figure
    # horizontally and squash every bar to one pixel.
    legend_ncol = min(3, n_methods + 1)
    ax.legend(
        title="method",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=legend_ncol,
        handletextpad=0.6,
        columnspacing=1.6,
        borderpad=0.3,
        labelspacing=0.4,
    )

    fig.tight_layout()
    save_paper_figure(fig, out_path, dpi=dpi)
    return fig
