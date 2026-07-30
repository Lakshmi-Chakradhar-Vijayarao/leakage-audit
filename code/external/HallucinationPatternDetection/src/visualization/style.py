"""Shared figure-style helpers for the visualization package.

Everything that the visualization modules need for publication-quality
output is centralized here so the behaviour is identical across every
plot type:

  * `apply_paper_style()`  -- sets matplotlib rcParams with text/font sizes
    that remain legible when a figure is scaled down for print or slides.
  * `save_paper_figure()`  -- wraps `Figure.savefig` with `bbox_inches="tight"`
    and `pad_inches=0`, so the resulting file has zero margin on all four
    borders and embeds cleanly wherever it is used.

The default output format is PDF (vector, so it rescales without quality
loss). The format is detected from the file extension of `out_path`, so
callers that pass a `.png` path still get a PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- font sizes
# These sizes are intentionally on the larger side because figures are
# usually scaled down for display -- to a single column (~3.3 in) or a
# double column (~7 in). At that scale, 8-10 pt matplotlib defaults
# become illegible. The values below remain comfortably readable.
_PAPER_RC_PARAMS = {
    # Body / default text -- noticeably above matplotlib's 10 pt default so
    # every label remains legible after the figure is scaled down.
    "font.size":             17,
    "axes.titlesize":        22,
    "axes.labelsize":        20,
    "xtick.labelsize":       16,
    "ytick.labelsize":       16,
    "legend.fontsize":       16,
    "legend.title_fontsize": 17,
    "figure.titlesize":      22,
    # Slightly thicker strokes so the figure still has presence at small
    # column widths.
    "axes.linewidth":   1.2,
    "lines.linewidth":  2.4,
    "lines.markersize": 8,
    "patch.linewidth":  0.8,
    # DejaVu Sans is bundled with matplotlib, so rendering is identical
    # on every machine without installing any system fonts.
    "font.family":      "DejaVu Sans",
    # Keep PDF text as actual text (selectable / searchable), not as
    # outlined paths.
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.0,
}

_STYLE_APPLIED = False


def apply_paper_style() -> None:
    """Idempotently apply the shared figure-style rcParams.

    Safe to call from every visualization function; subsequent calls are
    cheap and have no effect once the style is in place.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    mpl.rcParams.update(_PAPER_RC_PARAMS)
    _STYLE_APPLIED = True


def save_paper_figure(
    fig: plt.Figure,
    out_path: Optional[str],
    dpi: int = 300,
) -> None:
    """Save `fig` to `out_path` with zero margins on all four borders.

    The output format is taken from the extension of `out_path`. PDF is
    recommended because it is a vector format -- no resolution loss when
    the figure is rescaled.

    Parameters
    ----------
    fig
        The matplotlib figure to save.
    out_path
        Destination file. The parent directory is created if missing. If
        `None`, the figure is not written to disk (useful for notebook use).
    dpi
        Used only for raster-format outputs (PNG, JPEG, etc.). Ignored by
        PDF / SVG / EPS.
    """
    if out_path is None:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `bbox_inches="tight"` crops empty whitespace around the axes, and
    # `pad_inches=0` removes the small default padding that `tight` leaves
    # behind. Together they guarantee zero margin on all four borders.
    fig.savefig(
        out.as_posix(),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.0,
    )


__all__ = ["apply_paper_style", "save_paper_figure"]
