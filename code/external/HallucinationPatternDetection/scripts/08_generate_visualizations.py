"""08 — generate every figure from existing pipeline outputs.

This stage is the *visualization-only* entry point. It consumes the
outputs of the earlier pipeline stages (hidden states, trained probes,
per-method metrics, aggregated tables) and writes the complete figure
set to ``results/figures/``.

The default output format is **vector PDF** (overridable via
``--figure-format`` or ``viz.figure_format`` in ``config/config.yaml``).
Every figure is saved with zero margins on all four borders.

Outputs (default ``.pdf``; ``--figure-format png`` switches to raster):

  results/figures/
    tsne_<model>__<dataset>__layer<best>.pdf
    pca_<model>__<dataset>__layer<best>.pdf
    auroc_<model>__<dataset>__<probe>.pdf
    heatmap_<dataset>__<probe>.pdf
    separation_<model>__<dataset>.pdf
    attention_entropy_<model>__<dataset>.pdf
    score_dist_<method>__<model>__<dataset>.pdf
    method_comparison_<dataset>.pdf

Running modes
-------------

  1. As part of the end-to-end pipeline (automatic):

         python scripts/run_pipeline.py

     ``run_pipeline.py`` invokes this script as the final stage (id ``08``)
     after every earlier stage has finished.

  2. Standalone -- regenerate all figures from the *existing* results on
     disk, without re-running any of the heavy stages:

         python scripts/08_generate_visualizations.py

     Equivalent, via the runner:

         python scripts/run_pipeline.py --only 08 --skip-preflight

     Useful flags::

         python scripts/08_generate_visualizations.py --figure-format pdf
         python scripts/08_generate_visualizations.py --figure-format png
         python scripts/08_generate_visualizations.py --figures-dir results/figures_v2
         python scripts/08_generate_visualizations.py --dpi 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, get_logger, ensure_dir
from src.visualization import (
    apply_paper_style,
    plot_tsne_layer, plot_pca_layer,
    plot_layerwise_auroc, plot_layerwise_heatmap, plot_separation_ratio,
    plot_attention_entropy,
    plot_score_distribution, plot_method_comparison,
)
from src.analysis.pattern_analyzer import layer_geometry_stats


# ---------------------------------------------------------------- helpers
_VALID_FORMATS = {"pdf", "png", "svg", "eps", "jpg", "jpeg"}


def _safe_load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_format(fmt: str) -> str:
    fmt = (fmt or "pdf").lower().lstrip(".")
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"unsupported figure_format '{fmt}'; "
            f"choose one of {sorted(_VALID_FORMATS)}"
        )
    return fmt


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate all figures from existing pipeline outputs. "
            "Defaults to vector PDF with zero margins."
        )
    )
    p.add_argument(
        "--figure-format",
        default=None,
        help=(
            "Output format for every figure (pdf, png, svg, eps, jpg). "
            "Defaults to viz.figure_format from config/config.yaml, which is "
            "'pdf' out of the box."
        ),
    )
    p.add_argument(
        "--figures-dir",
        default=None,
        help=(
            "Destination directory for the generated figures. Defaults to "
            "paths.figures from config/config.yaml (results/figures)."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=(
            "Override the savefig DPI (only matters for raster formats like "
            "PNG; ignored for vector PDF / SVG / EPS)."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = load_config()
    log = get_logger("visualize", log_dir=cfg["paths"]["logs"])

    # Resolve runtime parameters: CLI overrides config; config provides
    # sensible defaults. The default figure format is PDF (vector, so
    # figures rescale without quality loss).
    fmt = _normalize_format(
        args.figure_format
        or cfg.get("viz", {}).get("figure_format", "pdf")
    )
    dpi = int(args.dpi if args.dpi is not None else cfg["viz"].get("dpi", 300))

    hs_dir = Path(cfg["paths"]["hidden_states"])
    probe_dir = Path(cfg["paths"]["probes"])
    metrics_dir = Path(cfg["paths"]["metrics"])
    fig_dir = ensure_dir(args.figures_dir or cfg["paths"]["figures"])
    tables_dir = Path(cfg["paths"]["tables"])

    # Apply the shared figure style up-front so every figure -- including ones
    # produced before any plot function has been called -- inherits the
    # larger font sizes.
    apply_paper_style()

    log.info(
        f"[viz] format={fmt}  dpi={dpi}  figures_dir={fig_dir}  "
        f"hidden_states_dir={hs_dir}"
    )

    n_generated = 0

    # ---------- 1) per-(model, dataset) projections + geometry ---------
    for fp in sorted(hs_dir.glob("*.pt")):
        try:
            data = torch.load(fp, map_location="cpu")
        except Exception as e:
            log.warning(f"[viz] failed to load {fp.name}: {e}")
            continue
        X = data["hidden_states"].numpy()       # [N, L+1, D]
        y = data["labels"].numpy()
        model = data.get("model_short_name", fp.stem.split("__", 1)[0])
        dataset = fp.stem.split("__", 1)[1] if "__" in fp.stem else ""

        # geometry
        geom = layer_geometry_stats(X, y)
        plot_separation_ratio(
            geom,
            title=f"{model} on {dataset}: class separation across layers",
            out_path=str(fig_dir / f"separation_{model}__{dataset}.{fmt}"),
            dpi=dpi,
        )
        n_generated += 1

        # best layer per-model heuristic: argmax separation_ratio
        best_layer = int(np.nanargmax(geom["separation_ratio"]))
        Xi = X[:, best_layer, :]
        plot_pca_layer(
            Xi, y,
            title=f"PCA — {model} / {dataset} / layer {best_layer}",
            seed=cfg["seed"],
            out_path=str(fig_dir / f"pca_{model}__{dataset}__layer{best_layer}.{fmt}"),
            dpi=dpi,
        )
        n_generated += 1
        try:
            plot_tsne_layer(
                Xi, y,
                title=f"t-SNE — {model} / {dataset} / layer {best_layer}",
                perplexity=cfg["viz"]["tsne_perplexity"],
                seed=cfg["seed"],
                out_path=str(fig_dir / f"tsne_{model}__{dataset}__layer{best_layer}.{fmt}"),
                dpi=dpi,
            )
            n_generated += 1
        except Exception as e:
            log.warning(f"[tsne] failed for {model}__{dataset}: {e}")
        # attention entropy if present
        if "attention_entropy" in data:
            ent = data["attention_entropy"].numpy()
            layers_kept = data.get(
                "attention_layers_kept", list(range(ent.shape[1]))
            )
            plot_attention_entropy(
                ent, y, layers_kept,
                title=f"{model} / {dataset}: attention entropy",
                out_path=str(fig_dir / f"attention_entropy_{model}__{dataset}.{fmt}"),
                dpi=dpi,
            )
            n_generated += 1
        plt.close("all")

    # ---------- 2) per-probe layerwise AUROC + heatmap -----------------
    probe_files = list(probe_dir.glob("*.json"))
    # accumulate matrices for heatmaps: per (dataset, probe_type) we
    # collect a [n_models, n_layers] auroc matrix.
    matrices: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(probe_files):
        try:
            payload = _safe_load_json(fp)
        except Exception as e:
            log.warning(f"[viz] failed to read probe file {fp.name}: {e}")
            continue
        per_layer = payload.get("per_layer", {})
        if not per_layer:
            continue
        # ensure keys are ints
        keyed: Dict[int, Any] = {int(k): v for k, v in per_layer.items()}
        plot_layerwise_auroc(
            keyed,
            title=(
                f"{payload.get('model','')} / "
                f"{payload.get('dataset','')} / "
                f"{payload.get('probe_type','')}"
            ),
            out_path=str(
                fig_dir
                / f"auroc_{payload.get('model','m')}__{payload.get('dataset','d')}__{payload.get('probe_type','p')}.{fmt}"
            ),
            dpi=dpi,
        )
        n_generated += 1
        plt.close("all")
        # accumulate for the cross-model heatmap
        key = f"{payload.get('dataset','')}_{payload.get('probe_type','')}"
        matrices.setdefault(key, {"models": [], "rows": [], "layers": []})
        layers = sorted(keyed.keys())
        if len(layers) > len(matrices[key]["layers"]):
            matrices[key]["layers"] = layers
        matrices[key]["models"].append(payload.get("model", ""))
        matrices[key]["rows"].append([keyed[li]["auroc_mean"] for li in layers])

    for key, blob in matrices.items():
        ds, probe = key.rsplit("_", 1)
        # Models may have different layer counts → pad shorter rows with NaN
        max_cols = max(len(r) for r in blob["rows"])
        padded = []
        for r in blob["rows"]:
            if len(r) < max_cols:
                padded.append(r + [float("nan")] * (max_cols - len(r)))
            else:
                padded.append(r)
        rows = np.array(padded, dtype=float)
        all_layers = (
            blob["layers"] if len(blob["layers"]) >= max_cols
            else list(range(max_cols))
        )
        plot_layerwise_heatmap(
            rows, blob["models"], all_layers,
            title=f"Layer-wise AUROC — {probe} probe on {ds}",
            out_path=str(fig_dir / f"heatmap_{ds}__{probe}.{fmt}"),
            dpi=dpi,
        )
        n_generated += 1
        plt.close("all")

    # ---------- 3) score distributions for INSIDE ---------------------
    for fp in sorted(metrics_dir.glob("inside__*.json")):
        try:
            payload = _safe_load_json(fp)
        except Exception as e:
            log.warning(f"[viz] failed to read metric file {fp.name}: {e}")
            continue
        score = np.array(payload.get("scores", []))
        labels = np.array(payload.get("labels", []))
        if score.size == 0:
            continue
        # hallucination score is the EigenScore itself (higher = more)
        plot_score_distribution(
            score, labels,
            title=(
                f"INSIDE EigenScore — "
                f"{payload.get('model','')} / {payload.get('dataset','')}"
            ),
            out_path=str(
                fig_dir
                / f"score_dist_inside__{payload.get('model','')}__{payload.get('dataset','')}.{fmt}"
            ),
            dpi=dpi,
        )
        n_generated += 1
        plt.close("all")

    # ---------- 4) method-comparison from aggregated table -------------
    all_csv = tables_dir / "all_results.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        for ds in df["dataset"].dropna().unique():
            sub = df[df["dataset"] == ds]
            plot_method_comparison(
                sub, metric="auroc",
                title=f"Method comparison — {ds}",
                out_path=str(fig_dir / f"method_comparison_{ds}.{fmt}"),
                dpi=dpi,
            )
            n_generated += 1
            plt.close("all")

    log.info(f"[done] wrote {n_generated} figure(s) ({fmt}) to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
