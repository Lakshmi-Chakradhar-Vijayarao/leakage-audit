"""
Paper 2 -- quantifies Case Study 4's winner's-curse severity directly
from data already shipped in code/external/HallucinationPatternDetection/
results/probes/*.json (33 layers x 3 seeds each, auroc per layer per
seed). The paper previously claimed this required re-extracting hidden
states from the quantized checkpoint from scratch; a fresh review found
the winner's-curse estimate is fully computable from files already in
this repo.

CORRECTION (a later independent adversarial review found this). The first
version of this script set

    naive_best_auroc = d["best_auroc"]

i.e. the repo's own reported best -- the max over ALL THREE seeds, INCLUDING
the seed later used as the "held-out" evaluation. Whenever the leave-one-out
argmax happened to coincide with the full-sample argmax (true in 7 of 24
cells), the "honest" held-out estimate was then arithmetically forced to
equal the naive one, pinning the winner's-curse estimate at exactly 0.0000
regardless of the true effect. That is not a held-out estimate at all.

FIXED PROTOCOL (genuinely leave-one-seed-out, both sides):
  For each of the 3 seeds in turn, treat it as the held-out evaluation seed.
    - SELECTION seeds = the other two, and ONLY the other two.
    - naive_r  = max over layers of the mean AUROC on those two seeds
                 (the number a practitioner would report if they selected
                 and reported on the same data -- computed here without ever
                 touching the held-out seed).
    - honest_r = the held-out seed's AUROC at that same selected layer.
    - winners_curse_r = naive_r - honest_r.
  The cell's estimate is the mean over the 3 rotations. No quantity on
  either side of the subtraction is a function of the held-out seed's own
  AUROC except honest_r itself.

Also added (this section previously carried no CI or significance test at
all, unlike every other section of the paper): a BCa bootstrap 95% CI over
the per-cell estimates, matching the convention used in code/44, code/46 and
code/49, plus a Wilcoxon signed-rank test against zero.

Also added: an explicit ceiling-saturation audit. Many of these 24 cells sit
at AUROC >= 0.975 at EVERY layer, where there is essentially no room for a
selection effect to appear. The full 24-cell mean and the non-saturated
subset's mean are both reported so a reader can see how much of the headline
average is a ceiling artifact.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "code" / "external" / "HallucinationPatternDetection" / "results" / "probes"
OUT_PATH = ROOT / "results" / "case_study_4_winners_curse.json"

SATURATION_THRESHOLD = 0.975
RNG_GLOBAL = np.random.default_rng(2026)


def quantify_file(path):
    d = json.load(open(path))
    layers = d["layers"]
    per_layer = d["per_layer"]
    n_seeds = len(per_layer[str(layers[0])]["seed_values"]["auroc"])

    per_rotation = []
    for held_out_seed in range(n_seeds):
        other_seeds = [s for s in range(n_seeds) if s != held_out_seed]
        # Selection AND the naive baseline both use ONLY the other seeds.
        mean_over_others = {
            layer: float(np.mean([per_layer[str(layer)]["seed_values"]["auroc"][s] for s in other_seeds]))
            for layer in layers
        }
        selected_layer = max(mean_over_others, key=mean_over_others.get)
        naive_r = mean_over_others[selected_layer]
        honest_r = float(per_layer[str(selected_layer)]["seed_values"]["auroc"][held_out_seed])
        per_rotation.append({
            "held_out_seed": held_out_seed,
            "selected_layer": selected_layer,
            "naive_selection_auroc": naive_r,
            "held_out_auroc": honest_r,
            "winners_curse": naive_r - honest_r,
        })

    naive_mean = float(np.mean([r["naive_selection_auroc"] for r in per_rotation]))
    honest_mean = float(np.mean([r["held_out_auroc"] for r in per_rotation]))
    winners_curse = float(np.mean([r["winners_curse"] for r in per_rotation]))

    all_layer_means = [per_layer[str(layer)]["auroc_mean"] for layer in layers]
    worst_layer_auroc = float(min(all_layer_means))
    mean_over_all_layers = float(np.mean(all_layer_means))

    return {
        "reported_best_layer_full_sample": d["best_layer"],
        "reported_best_auroc_full_sample": d["best_auroc"],
        "naive_selection_auroc_loo": naive_mean,
        "honest_held_out_auroc": honest_mean,
        "winners_curse_estimate": winners_curse,
        "mean_over_all_layers": mean_over_all_layers,
        "worst_layer_auroc": worst_layer_auroc,
        "spread_best_minus_worst": float(d["best_auroc"] - worst_layer_auroc),
        # Two distinct, both-reported saturation criteria:
        #   all_layers_saturated  -- EVERY layer is already >= threshold, so
        #                            there is no spread at all for selection
        #                            to exploit (the strict reading).
        #   operating_point_saturated -- the SELECTED layer itself sits at
        #                            >= threshold, so the reported number has
        #                            almost no headroom above it. This is the
        #                            criterion that actually compresses the
        #                            measurable winner's curse, and it is the
        #                            one used for the headline subset split.
        "all_layers_saturated": bool(worst_layer_auroc >= SATURATION_THRESHOLD),
        "operating_point_saturated": bool(naive_mean >= SATURATION_THRESHOLD),
        "ceiling_saturated": bool(naive_mean >= SATURATION_THRESHOLD),
        "per_rotation": per_rotation,
    }


def bca_ci(vals, n_resamples=10000):
    vals = np.asarray(vals, dtype=float)
    if np.allclose(vals, vals[0]):
        return [float(vals[0]), float(vals[0])]
    res = bootstrap((vals,), np.mean, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def summarize(name, cells):
    vals = [c["winners_curse_estimate"] for c in cells]
    if not vals:
        return None
    ci = bca_ci(vals)
    try:
        _, p = wilcoxon(vals)
        p = float(p)
    except ValueError:
        p = None
    s = {
        "n_cells": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "bca_ci_95": ci,
        "wilcoxon_p_vs_zero": p,
    }
    print(f"\n{name} (n={s['n_cells']} cells):")
    print(f"  mean winner's-curse = {s['mean']:+.4f}  BCa 95% CI {ci}  Wilcoxon p={p}")
    print(f"  median {s['median']:+.4f}  min {s['min']:+.4f}  max {s['max']:+.4f}")
    return s


def main():
    out = {"per_cell": {}}
    files = sorted(PROBES_DIR.glob("*.json"))
    print(f"Found {len(files)} probe result files", flush=True)
    for f in files:
        key = f.stem
        r = quantify_file(f)
        out["per_cell"][key] = r
        print(f"  {key}: wc={r['winners_curse_estimate']:+.4f}  "
              f"naive_loo={r['naive_selection_auroc_loo']:.4f}  "
              f"held_out={r['honest_held_out_auroc']:.4f}  "
              f"worst_layer={r['worst_layer_auroc']:.4f}  "
              f"{'CEILING' if r['ceiling_saturated'] else 'non-saturated'}", flush=True)

    cells = list(out["per_cell"].values())
    sat = [c for c in cells if c["ceiling_saturated"]]
    nonsat = [c for c in cells if not c["ceiling_saturated"]]

    out["summary_all_cells"] = summarize("ALL CELLS", cells)
    out["summary_ceiling_saturated"] = summarize(
        f"CEILING-SATURATED (selected-layer AUROC >= {SATURATION_THRESHOLD})", sat)
    out["summary_non_saturated"] = summarize("NON-SATURATED", nonsat)
    out["summary_all_layers_saturated"] = summarize(
        f"ALL-LAYERS-SATURATED (every layer >= {SATURATION_THRESHOLD}, strict reading)",
        [c for c in cells if c["all_layers_saturated"]])
    out["summary_not_all_layers_saturated"] = summarize(
        "NOT-ALL-LAYERS-SATURATED (strict reading complement)",
        [c for c in cells if not c["all_layers_saturated"]])

    by_dataset = {}
    for k, v in out["per_cell"].items():
        ds = k.split("__")[1]
        by_dataset.setdefault(ds, []).append(v)
    out["composition_by_dataset"] = {
        ds: {
            "n_cells": len(vs),
            "n_ceiling_saturated": int(sum(c["ceiling_saturated"] for c in vs)),
            "mean_winners_curse": float(np.mean([c["winners_curse_estimate"] for c in vs])),
        }
        for ds, vs in sorted(by_dataset.items())
    }
    print("\nComposition by dataset:")
    for ds, v in out["composition_by_dataset"].items():
        print(f"  {ds:14s} n={v['n_cells']}  saturated={v['n_ceiling_saturated']}  "
              f"mean_wc={v['mean_winners_curse']:+.4f}")

    out["saturation_threshold"] = SATURATION_THRESHOLD
    out["ceiling_saturated_cells"] = sorted(
        k for k, v in out["per_cell"].items() if v["ceiling_saturated"])

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
