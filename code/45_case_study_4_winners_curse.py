"""
Paper 2 -- quantifies Case Study 4's winner's-curse severity directly
from data already shipped in code/external/HallucinationPatternDetection/
results/probes/*.json (3 seeds per cell; 33 probed layers for llama3.1-8b
and mistral-7b, 29 for qwen2.5-7b -- see LAYER_COUNTS below). The paper
previously claimed this required re-extracting hidden states from the
quantized checkpoint from scratch; a fresh review found the winner's-curse
estimate is fully computable from files already in this repo.

CORRECTION 1 (an adversarial review found this). The first version of this
script set naive_best_auroc = d["best_auroc"], i.e. the repo's own reported
best -- the max over ALL THREE seeds, INCLUDING the seed later used as the
"held-out" evaluation.

CORRECTION 2 (A SECOND, INDEPENDENT REVIEW FOUND THAT CORRECTION 1 DID NOT
WORK). Rebuilding the estimator as a leave-one-seed-out rotation removed the
stated cause but left the defect itself intact. The rotation estimator is
ALGEBRAICALLY PINNED TO EXACTLY ZERO whenever the selected layer is the same
in all three rotations, which is true in 7 of the 24 cells.

  Proof. Suppose every rotation selects the same layer L. Let a_0, a_1, a_2
  be the three seeds' AUROC at L and S = a_0 + a_1 + a_2, abar = S/3. In
  rotation r the two selection seeds are the ones other than r, so
      naive_r  = (S - a_r) / 2
      honest_r = a_r
      wc_r     = naive_r - honest_r = (S - 3 a_r) / 2 = (3/2)(abar - a_r).
  The cell estimate is the mean over rotations:
      mean_r wc_r = (3/2) * mean_r (abar - a_r) = (3/2)(abar - abar) = 0,
  identically, for ANY data. The per-rotation values are 1.5x the held-out
  seed's deviation from the seed mean -- pure noise that sums to zero by
  construction -- so in these cells the estimator is not measuring a small
  effect, it is measuring nothing at all. The float residues of +-3.70e-17
  seen in two cells are summation-order noise around that algebraic zero, and
  must be treated as exact zeros, not as tiny signed observations.

  This matters because ALL 7 degenerate cells fall in the NON-SATURATED
  subgroup (7 of its 12 cells), which is the subgroup the paper leans on. Its
  previously-reported mean of +0.0065 is 5 genuine measurements averaging
  +0.0157 diluted by 7 algebraic zeros, and its Wilcoxon p moves from 0.047
  to 0.0625 once the two float residues are correctly counted as zeros
  (scipy drops exact zeros; it was ranking +-3.7e-17 as real observations).

THE FIX IS NOT ANOTHER PATCH OF THE SAME STRUCTURE. Three things are done.

  (1) DEGENERACY IS DETECTED AND DECLARED, not hidden. Every cell records
      `selected_layers_across_rotations`, `estimator_degenerate` (all three
      rotations picked the same layer) and a snapped estimate in which any
      |wc| < 1e-12 becomes exactly 0.0. The headline number is computed over
      the 17 NON-DEGENERATE cells, with BCa CI and Wilcoxon; the 7 degenerate
      cells are reported separately and are disclosed in the paper rather
      than averaged in.

  (2) A PERMUTATION NULL characterizes the estimator INCLUDING its
      degeneracy. Note first that a GLOBAL relabelling of seeds leaves this
      estimator exactly invariant -- it already averages over all three
      choices of held-out seed -- so a naive seed-label shuffle is a no-op
      and would produce a meaningless null. The shuffle is therefore applied
      INDEPENDENTLY WITHIN EACH LAYER, which destroys the layer x seed
      structure that makes an argmax stable while preserving every layer's
      marginal set of three AUROC values. Under that null the argmax is
      noise-driven, which is the regime in which a winner's curse is
      LARGEST; the null therefore answers the question the degenerate cells
      cannot: "what would this estimator have reported here if the layer
      ranking had been noise?" A degenerate cell whose null distribution is
      centred well above zero is a cell where the exact zero reflects a
      stable argmax defeating the estimator, NOT an absence of selection
      effect. Reported per cell and pooled, with a permutation p-value.

  (3) A SECOND, STRUCTURALLY DIFFERENT ESTIMATOR that cannot be pinned:
      the standard bootstrap estimate of the bias of a max-of-means.
      Resample the 3 seeds with replacement, take the argmax layer on the
      resample, and difference the resample's mean at that layer against the
      FULL-SAMPLE mean at that same layer. This never telescopes, because
      the two sides are computed on different seed multisets by
      construction. It is coarse at 3 seeds and is reported as a
      cross-check on sign and order of magnitude, not as the headline.

Also retained from the previous revision: BCa bootstrap 95% CIs and Wilcoxon
signed-rank tests (this section previously carried neither), and an explicit
ceiling-saturation audit -- many cells sit at AUROC >= 0.975 at every layer,
where there is no room for a selection effect to appear.
"""
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "code" / "external" / "HallucinationPatternDetection" / "results" / "probes"
OUT_PATH = ROOT / "results" / "case_study_4_winners_curse.json"

SATURATION_THRESHOLD = 0.975
RNG_GLOBAL = np.random.default_rng(2026)

# Anything below this in absolute value is the float residue of an algebraic
# zero, not an observation. See CORRECTION 2 above.
ZERO_SNAP = 1e-12
N_PERM = 2000
N_BOOT_BIAS = 2000

# Per-model probed-layer counts. The paper previously stated a flat "33
# layers" for all 24 cells; the 8 qwen2.5-7b cells probe 29. Asserted below
# against the shipped files so the paper's stated counts cannot drift.
LAYER_COUNTS = {"llama3.1-8b": 33, "mistral-7b": 33, "qwen2.5-7b": 29}


def _rotation_estimate(A):
    """Leave-one-seed-out rotation estimator on an (n_layers, n_seeds) matrix.

    Returns (estimate, per_rotation_records, selected_layer_indices)."""
    n_layers, n_seeds = A.shape
    recs, sel = [], []
    for held in range(n_seeds):
        others = [s for s in range(n_seeds) if s != held]
        mean_others = A[:, others].mean(axis=1)
        li = int(np.argmax(mean_others))
        sel.append(li)
        recs.append({"held_out_seed": held, "layer_index": li,
                     "naive_selection_auroc": float(mean_others[li]),
                     "held_out_auroc": float(A[li, held]),
                     "winners_curse": float(mean_others[li] - A[li, held])})
    est = float(np.mean([r["winners_curse"] for r in recs]))
    return est, recs, sel


def _permutation_null(A, rng, n_perm=N_PERM):
    """Null distribution of the rotation estimator under per-layer seed shuffles.

    A GLOBAL seed permutation is a no-op here (the estimator already averages
    over every choice of held-out seed), so the shuffle is applied
    INDEPENDENTLY WITHIN EACH LAYER. That preserves each layer's marginal set
    of three AUROC values while destroying the layer x seed structure that
    makes an argmax stable across rotations -- i.e. it puts the estimator in
    the noise-driven-argmax regime where a winner's curse is largest."""
    vals, degen = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        B = np.take_along_axis(A, rng.permuted(
            np.tile(np.arange(A.shape[1]), (A.shape[0], 1)), axis=1), axis=1)
        est, _, sel = _rotation_estimate(B)
        vals[b] = 0.0 if abs(est) < ZERO_SNAP else est
        degen[b] = float(len(set(sel)) == 1)
    return vals, float(degen.mean())


def _bootstrap_max_bias(A, rng, n_boot=N_BOOT_BIAS):
    """Bootstrap estimate of the selection bias of a max-of-means.

    Structurally cannot be pinned to zero: the resample's mean at the
    resample-selected layer and the full-sample mean at that same layer are
    computed on different seed multisets by construction, so nothing
    telescopes. Coarse at n_seeds=3; reported as a sign/magnitude cross-check."""
    n_seeds = A.shape[1]
    full_mean = A.mean(axis=1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        m = A[:, idx].mean(axis=1)
        li = int(np.argmax(m))
        out[b] = m[li] - full_mean[li]
    return float(out.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def quantify_file(path):
    d = json.load(open(path))
    layers = d["layers"]
    per_layer = d["per_layer"]
    n_seeds = len(per_layer[str(layers[0])]["seed_values"]["auroc"])

    model = path.stem.split("__")[0]
    if model in LAYER_COUNTS:
        assert len(layers) == LAYER_COUNTS[model], (
            f"{path.stem}: expected {LAYER_COUNTS[model]} probed layers for {model}, "
            f"found {len(layers)}")

    # (n_layers, n_seeds) AUROC matrix -- the only input any estimator needs.
    A = np.array([[per_layer[str(l)]["seed_values"]["auroc"][s] for s in range(n_seeds)]
                  for l in layers], dtype=float)

    winners_curse_raw, recs, sel_idx = _rotation_estimate(A)
    degenerate = len(set(sel_idx)) == 1
    winners_curse = 0.0 if abs(winners_curse_raw) < ZERO_SNAP else winners_curse_raw

    per_rotation = [{**r, "selected_layer": layers[r["layer_index"]]} for r in recs]
    for r in per_rotation:
        r.pop("layer_index")
    selected_layers = [layers[i] for i in sel_idx]

    # Per-cell RNG seeded from a STABLE hash of the cell name. NOT Python's
    # builtin hash(): string hashing is salted per interpreter process
    # (PYTHONHASHSEED), so seeding from it would make the permutation null and
    # the bootstrap cross-check silently irreproducible between runs -- in a
    # paper whose entire subject is numbers that cannot be re-derived from what
    # ships. zlib.crc32 is stable across processes, platforms and versions.
    rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
    null_vals, null_degen_rate = _permutation_null(A, rng)
    boot_bias, boot_ci = _bootstrap_max_bias(A, rng)
    # One-sided permutation p: how often does a noise-driven argmax produce an
    # estimate at least as large as the observed one?
    perm_p = float((np.sum(null_vals >= winners_curse) + 1) / (len(null_vals) + 1))

    naive_mean = float(np.mean([r["naive_selection_auroc"] for r in per_rotation]))
    honest_mean = float(np.mean([r["held_out_auroc"] for r in per_rotation]))

    all_layer_means = [per_layer[str(layer)]["auroc_mean"] for layer in layers]
    worst_layer_auroc = float(min(all_layer_means))
    mean_over_all_layers = float(np.mean(all_layer_means))

    return {
        "reported_best_layer_full_sample": d["best_layer"],
        "reported_best_auroc_full_sample": d["best_auroc"],
        "n_probed_layers": len(layers),
        "naive_selection_auroc_loo": naive_mean,
        "honest_held_out_auroc": honest_mean,
        "winners_curse_estimate": winners_curse,
        "winners_curse_estimate_unsnapped": winners_curse_raw,
        "selected_layers_across_rotations": selected_layers,
        "estimator_degenerate": bool(degenerate),
        "degeneracy_reason": (
            "All 3 rotations selected the same layer, so the rotation estimator is "
            "algebraically 0 for ANY data (see module docstring). This cell carries no "
            "information about the winner's curse and is excluded from the headline mean."
            if degenerate else None),
        "permutation_null_mean": float(np.mean(null_vals)),
        "permutation_null_ci_95": [float(np.percentile(null_vals, 2.5)),
                                   float(np.percentile(null_vals, 97.5))],
        "permutation_null_degenerate_rate": null_degen_rate,
        "permutation_p_one_sided": perm_p,
        "bootstrap_max_bias": boot_bias,
        "bootstrap_max_bias_ci_95": boot_ci,
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
    """Summarize a group of cells.

    `winners_curse_estimate` is already SNAPPED: any |value| < ZERO_SNAP is
    exactly 0.0, so scipy's Wilcoxon drops it as a true zero instead of
    ranking a +-3.7e-17 float residue as a real observation. That snapping
    alone moves the non-saturated subgroup's p from 0.047 to 0.0625."""
    vals = [c["winners_curse_estimate"] for c in cells]
    if not vals:
        return None
    n_degen = int(sum(c["estimator_degenerate"] for c in cells))
    ci = bca_ci(vals)
    try:
        _, p = wilcoxon(vals)
        p = float(p)
    except ValueError:
        p = None
    s = {
        "n_cells": len(vals),
        "n_degenerate_cells_included": n_degen,
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "bca_ci_95": ci,
        "wilcoxon_p_vs_zero": p,
        "n_nonzero_for_wilcoxon": int(sum(v != 0.0 for v in vals)),
        "permutation_null_mean_over_group": float(np.mean(
            [c["permutation_null_mean"] for c in cells])),
        "bootstrap_max_bias_mean_over_group": float(np.mean(
            [c["bootstrap_max_bias"] for c in cells])),
    }
    print(f"\n{name} (n={s['n_cells']} cells, {n_degen} degenerate):")
    print(f"  mean winner's-curse = {s['mean']:+.4f}  BCa 95% CI {ci}  Wilcoxon p={p} "
          f"(n_nonzero={s['n_nonzero_for_wilcoxon']})")
    print(f"  median {s['median']:+.4f}  min {s['min']:+.4f}  max {s['max']:+.4f}")
    print(f"  permutation-null mean {s['permutation_null_mean_over_group']:+.4f}  "
          f"bootstrap max-bias {s['bootstrap_max_bias_mean_over_group']:+.4f}")
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
              f"{'DEGENERATE' if r['estimator_degenerate'] else 'ok        '}  "
              f"null={r['permutation_null_mean']:+.4f}  "
              f"boot={r['bootstrap_max_bias']:+.4f}  "
              f"worst_layer={r['worst_layer_auroc']:.4f}  "
              f"{'CEILING' if r['ceiling_saturated'] else 'non-saturated'}", flush=True)

    cells = list(out["per_cell"].values())
    sat = [c for c in cells if c["ceiling_saturated"]]
    nonsat = [c for c in cells if not c["ceiling_saturated"]]
    degen = [c for c in cells if c["estimator_degenerate"]]
    nondegen = [c for c in cells if not c["estimator_degenerate"]]

    # ── HEADLINE: non-degenerate cells only ──────────────────────────────────
    out["summary_non_degenerate"] = summarize(
        "*** HEADLINE: NON-DEGENERATE CELLS (rotation argmax varied) ***", nondegen)
    out["summary_degenerate_excluded"] = summarize(
        "DEGENERATE CELLS (constant argmax -> estimate algebraically 0; DISCLOSED, NOT AVERAGED IN)",
        degen)
    out["summary_non_saturated_non_degenerate"] = summarize(
        "NON-SATURATED and NON-DEGENERATE", [c for c in nonsat if not c["estimator_degenerate"]])

    # ── retained for continuity with the previous revision, now labelled ─────
    out["summary_all_cells"] = summarize("ALL 24 CELLS (contaminated by 7 algebraic zeros)", cells)
    out["summary_ceiling_saturated"] = summarize(
        f"CEILING-SATURATED (selected-layer AUROC >= {SATURATION_THRESHOLD})", sat)
    out["summary_non_saturated"] = summarize(
        "NON-SATURATED (contaminated: 7 of its 12 cells are algebraic zeros)", nonsat)
    out["summary_all_layers_saturated"] = summarize(
        f"ALL-LAYERS-SATURATED (every layer >= {SATURATION_THRESHOLD}, strict reading)",
        [c for c in cells if c["all_layers_saturated"]])
    out["summary_not_all_layers_saturated"] = summarize(
        "NOT-ALL-LAYERS-SATURATED (strict reading complement)",
        [c for c in cells if not c["all_layers_saturated"]])

    out["degeneracy_audit"] = {
        "n_cells_total": len(cells),
        "n_degenerate": len(degen),
        "degenerate_cells": sorted(k for k, v in out["per_cell"].items()
                                   if v["estimator_degenerate"]),
        "zero_snap_threshold": ZERO_SNAP,
        "explanation": (
            "A cell is degenerate when all 3 leave-one-seed-out rotations select the SAME "
            "layer. Then naive_r = (S - a_r)/2 and honest_r = a_r, so the rotation mean is "
            "(3/2)(abar - abar) = 0 identically, for any data. Such a cell measures nothing "
            "and is excluded from the headline. Every degenerate cell here falls in the "
            "NON-SATURATED subgroup (7 of its 12 cells), which is why that subgroup's "
            "contaminated mean understates the effect among cells the estimator can see."),
        "permutation_null_mean_over_degenerate_cells": float(np.mean(
            [c["permutation_null_mean"] for c in degen])) if degen else None,
        "permutation_null_interpretation": (
            "The per-layer-shuffle null is what this estimator reports when the layer "
            "ranking is noise, i.e. the regime of MAXIMAL winner's curse. Degenerate cells "
            "whose null sits well above zero are cells where the exact zero reflects a "
            "STABLE argmax defeating the estimator, not an absent selection effect."),
    }
    out["per_model_layer_counts"] = LAYER_COUNTS
    out["method_notes"] = {
        "n_permutations": N_PERM,
        "n_bootstrap_max_bias": N_BOOT_BIAS,
        "permutation_scheme": (
            "Seed labels shuffled INDEPENDENTLY WITHIN EACH LAYER. A global seed relabelling "
            "leaves the rotation estimator exactly invariant and would be a no-op null."),
    }

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
