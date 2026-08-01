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
  (a) c * sigma_K * sqrt(2 ln K)   -- the "theoretically correct" EVT form,
                                      using each cell's OWN measured sigma_K
  (b) a + b * ln K                 -- two-parameter logarithmic
  (c) a * K^b                      -- two-parameter power law
plus, for reference, the constant-sigma version code/47 actually fit, so the
difference between the two is reproducible rather than asserted.

R^2 is computed as 1 - SS_res/SS_tot against the observed gaps on the same
K>1 cells code/47 fits on (K=1 is excluded there because the gap is identically
zero by construction: selecting the best of one candidate is not a selection).

──────────────────────────────────────────────────────────────────────────────
THE EVT "FALSIFICATION" IS RETRACTED AND DOWNGRADED (this revision). A second
independent adversarial review established that fit (a) does not test extreme
value theory at all, because `mean_val_auc_std` IS NOT THE QUANTITY EVT's
sigma REFERS TO.

  What EVT needs.  gap ~ c * sigma * sqrt(2 ln K) is derived for the expected
  maximum of K estimates that are exchangeable draws around a COMMON mean,
  with sigma their SAMPLING-NOISE standard deviation. K independent candidates,
  each estimated with noise sigma.

  What this harness measures.  In code/47, `val_aucs` is one validation AUROC
  per EPOCH of a SINGLE training trajectory, and mean_val_auc_std is
  np.std(val_aucs) over that trajectory. Successive epochs are not
  exchangeable and are not centred on a common mean: they are a LEARNING
  CURVE. Its dispersion is dominated by the systematic rise from
  initialization toward convergence, not by estimation noise around a fixed
  target.

  The two anomalies this script previously reported as evidence AGAINST EVT
  are exactly what that misspecification predicts, mechanically:
    - sigma falls 2.66x as K rises. Of course it does. More epochs means a
      larger fraction of the trajectory is spent converged and flat, so the
      SD of the whole curve shrinks. This is a statement about learning
      curves, not about selection noise, which should if anything be flat.
    - corr(sigma, gap) = -0.84, where EVT needs it POSITIVE. Same cause: the
      gap rises with K because there are more candidates to select over,
      while the trajectory SD falls with K for the reason above. Two
      quantities driven in opposite directions by the same variable will
      anticorrelate regardless of whether EVT holds.

  Therefore fit (a)'s R^2 = -0.399 CANNOT distinguish "EVT is wrong" from
  "sigma is the wrong quantity to test it with", and the previous revision's
  language -- that the extreme-value form is "falsified on this data once fit
  as stated" -- claimed a discrimination the design cannot make. The honest
  verdict is UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED. Testing EVT properly
  would require a sigma estimated across independent, exchangeable candidate
  evaluations at a fixed point in training -- e.g. K independently-seeded
  models, or repeated resampled validation splits at a fixed epoch -- which
  this harness does not produce. That is a measurement gap, not a theoretical
  obstacle, and it is no longer listed among the obstacles to deriving a
  formal bound.

  WHAT IS UNAFFECTED. Fit (b), gap = a + b ln K with R^2 = 0.970, is a purely
  empirical description of the K axis that never references sigma. It stands
  exactly as reported. So does the paper's use of it.
──────────────────────────────────────────────────────────────────────────────

ALSO ADDED (same review, a separate and smaller point): Sweep A's ACHIEVED
operating point is not constant across the K sweep even though its NOMINAL
target is fixed at AUROC_0 = 0.80. LEAKY's mean runs from 0.6269 at K=1 to
0.7574 at K=225. Since this paper's own SS5 shows the operating point is a
strong severity modifier, a K law fit across a moving operating point is
partly confounded with it. The fit is therefore ALSO reported restricted to
K >= 10, where the achieved operating point is nearly flat (0.7565-0.7616).
The law survives that restriction essentially unchanged, which is the reason
it is safe to disclose the confound rather than to qualify the result.
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
    achieved = np.array([cells[str(int(k))]["leaky_mean"] for k in Ks])

    valid = Ks > 1
    Kv, gv, sv, av = Ks[valid], gaps[valid], sigmas[valid], achieved[valid]

    print("Per-cell sigma (NOTE: trajectory dispersion, NOT the sampling-noise SD")
    print("EVT's derivation requires -- see module docstring):")
    for k, g, s, a in zip(Ks, gaps, sigmas, achieved):
        print(f"  K={int(k):4d}  gap={g:+.5f}  sigma_traj={s:.5f}  achieved_AUROC={a:.4f}")
    print(f"\nsigma range over fitted cells: {sv.min():.5f}-{sv.max():.5f} "
          f"({sv.max() / sv.min():.2f}x variation)")
    print(f"corr(sigma, gap) over fitted cells: {np.corrcoef(sv, gv)[0, 1]:+.4f}")
    print("  Both facts are predicted by the misspecification (longer trajectories are")
    print("  flatter, so their SD falls with K while the gap rises with K). Neither is")
    print("  evidence about EVT.\n")

    fits = {}

    # (a) theoretically correct EVT form: per-cell sigma_K
    pred_a = sv * np.sqrt(2 * np.log(Kv))
    c_a = float(np.sum(gv * pred_a) / np.sum(pred_a ** 2))
    fits["evt_per_cell_sigma"] = {
        "form": "gap = c * sigma_K * sqrt(2 ln K), sigma_K measured per cell",
        "params": {"c": c_a},
        "r_squared": r_squared(gv, c_a * pred_a),
        "verdict": "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED",
        "why_not_a_falsification": (
            "sigma_K here is np.std over one training trajectory's per-EPOCH validation "
            "AUROCs. EVT's sigma is the SAMPLING-NOISE SD of K exchangeable estimates around "
            "a common mean. A learning curve is neither exchangeable nor centred on a common "
            "mean, so this R^2 cannot distinguish 'EVT is wrong' from 'this is the wrong "
            "sigma'. Both anomalies previously cited as evidence against EVT (sigma falling "
            "2.66x with K, and corr(sigma, gap) = -0.84) follow mechanically from the "
            "misspecification. A real test needs sigma across independent, exchangeable "
            "candidate evaluations at a fixed point in training, which this harness does not "
            "produce. RETRACTED: the previous revision described this as a falsification."),
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

    # (d) the same logarithmic law restricted to K >= 10, where Sweep A's
    # ACHIEVED operating point is nearly flat. See the docstring: the nominal
    # target is fixed at 0.80 throughout, but the achieved LEAKY mean is not,
    # and this paper's own SS5 makes the operating point a strong severity
    # modifier. The restriction removes that confound; the law survives it.
    restrict = Ks >= 10
    Ar = np.vstack([np.ones(restrict.sum()), np.log(Ks[restrict])]).T
    coef_d, *_ = np.linalg.lstsq(Ar, gaps[restrict], rcond=None)
    fits["log_two_parameter_K_ge_10"] = {
        "form": "gap = a + b * ln K, restricted to K >= 10",
        "params": {"a": float(coef_d[0]), "b": float(coef_d[1])},
        "r_squared": r_squared(gaps[restrict], Ar @ coef_d),
        "n_cells_fit": int(restrict.sum()),
        "achieved_operating_point_range": [float(achieved[restrict].min()),
                                           float(achieved[restrict].max())],
        "why": ("Sweep A's achieved operating point moves from 0.6269 (K=1) to 0.7574 (K=225) "
                "even though its nominal target is fixed at 0.80, so the full-range K law is "
                "partly confounded with the operating-point relationship SS5 reports. "
                "Restricted to K >= 10 the achieved operating point spans only "
                "0.7565-0.7616 and the law is essentially unchanged, which is why the "
                "confound is disclosed rather than treated as a qualification."),
    }

    for name, f in fits.items():
        print(f"{name}:\n  {f['form']}\n  params={f['params']}\n  R^2={f['r_squared']:.4f}")
        if "verdict" in f:
            print(f"  VERDICT={f['verdict']}")

    out = {
        "n_cells_fit": int(valid.sum()),
        "K_values_fit": [int(k) for k in Kv],
        "gaps": [float(g) for g in gv],
        "sigma_val_per_cell": [float(s) for s in sv],
        "sigma_val_ratio_max_over_min": float(sv.max() / sv.min()),
        "corr_sigma_val_vs_gap": float(np.corrcoef(sv, gv)[0, 1]),
        "sigma_misspecification": {
            "what_is_measured": ("np.std over one training trajectory's per-epoch validation "
                                 "AUROCs (code/47, train_to_best_checkpoint)"),
            "what_EVT_requires": ("the sampling-noise SD of K exchangeable candidate estimates "
                                  "around a common mean"),
            "consequence": ("fit (a) is a test of the wrong quantity; its R^2 is not evidence "
                            "for or against extreme value theory"),
            "verdict": "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED",
            "unaffected": ("the empirical log law gap = a + b ln K (R^2 = 0.970) never "
                           "references sigma and stands as reported"),
        },
        "achieved_operating_point_per_cell": {
            str(int(k)): float(a) for k, a in zip(Ks, achieved)},
        "achieved_operating_point_full_range": [float(achieved.min()), float(achieved.max())],
        "fits": fits,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
