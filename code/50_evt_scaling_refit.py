"""
Paper 2 -- honest re-fit of the K-scaling ("winner's curse") relationship
reported in Appendix A's ninth check.

WHY THIS SCRIPT EXISTS. `code/47_selection_multiplicity_sweep.py` fits the
extreme-value-theory (EVT) prediction

    gap  ~=  c * sigma_val * sqrt(2 ln K)

but at lines 230-231 it collapses sigma_val to a single CONSTANT -- the mean
of the per-K-cell `mean_val_auc_std` across all ten K values -- and uses that
constant inside the predictor for every cell. That is not the model the theory
states. The theory's sigma is the noise SD of the validation estimates *in the
cell being predicted*, and this sweep measures exactly that per cell. Using the
constant silently converts a one-parameter EVT model into "gap is proportional
to sqrt(2 ln K)", which is a materially different (and much more flattering)
claim, because the measured per-cell sigma_val varies by ~2.5x across the sweep
AND moves in the OPPOSITE direction from the gap (sigma_val falls from ~0.049
at K=10 to ~0.020 at K=225 while the gap rises from +0.0023 to +0.0052).

This script refits the same data three ways and reports R^2 for each:
  (a) c * sigma_K * sqrt(2 ln K)   -- the theoretically correct EVT form,
                                      using each cell's OWN measured sigma_K
  (b) a + b * ln K                 -- two-parameter logarithmic
  (c) a * K^b                      -- two-parameter power law
plus, for reference, the constant-sigma version code/47 actually fit, so the
difference between the two is reproducible rather than asserted.

R^2 is computed as 1 - SS_res/SS_tot against the observed gaps on the same
K>1 cells code/47 fits on (K=1 is excluded there because the gap is identically
zero by construction: selecting the best of one candidate is not a selection).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = ROOT / "results" / "selection_multiplicity_sweep.json"
OUT_PATH = ROOT / "results" / "evt_scaling_refit.json"


def r_squared(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None


def main():
    d = json.load(open(SWEEP_PATH))
    cells = d["sweep_A_K"]
    Ks = np.array(sorted(int(k) for k in cells), dtype=float)
    gaps = np.array([cells[str(int(k))]["gap_mean"] for k in Ks])
    sigmas = np.array([cells[str(int(k))]["mean_val_auc_std"] for k in Ks])

    valid = Ks > 1
    Kv, gv, sv = Ks[valid], gaps[valid], sigmas[valid]

    print("Per-cell measured sigma_val (the quantity the EVT form needs):")
    for k, g, s in zip(Ks, gaps, sigmas):
        print(f"  K={int(k):4d}  gap={g:+.5f}  sigma_val={s:.5f}")
    print(f"\nsigma_val range over fitted cells: {sv.min():.5f}-{sv.max():.5f} "
          f"({sv.max() / sv.min():.2f}x variation)")
    print(f"corr(sigma_val, gap) over fitted cells: {np.corrcoef(sv, gv)[0, 1]:+.4f}"
          "   <- theory needs this to be POSITIVE\n")

    fits = {}

    # (a) theoretically correct EVT form: per-cell sigma_K
    pred_a = sv * np.sqrt(2 * np.log(Kv))
    c_a = float(np.sum(gv * pred_a) / np.sum(pred_a ** 2))
    fits["evt_per_cell_sigma"] = {
        "form": "gap = c * sigma_K * sqrt(2 ln K), sigma_K measured per cell",
        "params": {"c": c_a},
        "r_squared": r_squared(gv, c_a * pred_a),
    }

    # code/47's actual (mis-specified) fit: constant sigma = mean over all cells
    sigma_const = float(np.mean(sigmas))
    pred_const = sigma_const * np.sqrt(2 * np.log(np.maximum(Ks, 1.01)))[valid]
    c_const = float(np.sum(gv * pred_const) / np.sum(pred_const ** 2))
    fits["evt_constant_sigma_as_previously_fit"] = {
        "form": "gap = c * mean(sigma) * sqrt(2 ln K)  [what code/47 fit]",
        "params": {"c": c_const, "sigma_const": sigma_const},
        "r_squared": r_squared(gv, c_const * pred_const),
    }

    # (b) two-parameter logarithmic
    A = np.vstack([np.ones_like(Kv), np.log(Kv)]).T
    coef_b, *_ = np.linalg.lstsq(A, gv, rcond=None)
    fits["log_two_parameter"] = {
        "form": "gap = a + b * ln K",
        "params": {"a": float(coef_b[0]), "b": float(coef_b[1])},
        "r_squared": r_squared(gv, A @ coef_b),
    }

    # (c) two-parameter power law (fit in log space on strictly positive gaps)
    pos = gv > 0
    Ap = np.vstack([np.ones(pos.sum()), np.log(Kv[pos])]).T
    coef_c, *_ = np.linalg.lstsq(Ap, np.log(gv[pos]), rcond=None)
    a_c, b_c = float(np.exp(coef_c[0])), float(coef_c[1])
    fits["power_law_two_parameter"] = {
        "form": "gap = a * K^b",
        "params": {"a": a_c, "b": b_c},
        "r_squared": r_squared(gv[pos], a_c * Kv[pos] ** b_c),
        "n_cells_fit": int(pos.sum()),
    }

    for name, f in fits.items():
        print(f"{name}:\n  {f['form']}\n  params={f['params']}\n  R^2={f['r_squared']:.4f}")

    out = {
        "n_cells_fit": int(valid.sum()),
        "K_values_fit": [int(k) for k in Kv],
        "gaps": [float(g) for g in gv],
        "sigma_val_per_cell": [float(s) for s in sv],
        "sigma_val_ratio_max_over_min": float(sv.max() / sv.min()),
        "corr_sigma_val_vs_gap": float(np.corrcoef(sv, gv)[0, 1]),
        "fits": fits,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
