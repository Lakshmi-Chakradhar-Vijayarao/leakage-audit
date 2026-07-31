"""
Paper 2 -- the joint severity surface: does a SINGLE empirical model in
(K, AUROC_0) describe the inflation this paper measures?

WHY THIS EXISTS. An independent adversarial review observed that this paper
reports two strong one-at-a-time regularities -- gap rises with the candidate
count K, gap falls sharply with the operating point AUROC_0 -- and never
attempts to combine them, and that a joint function

    E[AUROC_argmax - AUROC(l*)] <= phi(K, n_sel, AUROC_0)

with phi decreasing in n_sel and AUROC_0 would unify the five mechanisms and
turn the checklist into a calculator. The review is right that the paper had
not attempted this. This script attempts it.

WHAT THIS IS, AND EMPHATICALLY IS NOT. This is an EMPIRICAL JOINT FIT
measured inside one synthetic harness, holding the leakage mechanism fixed.
It is NOT a bound, NOT a theorem, and NOT derived from anything. It carries
no guarantee for any cell it was not fit on, no guarantee for any other
mechanism, and no guarantee outside this harness's generative process. A
formally-derived, theoretically-justified upper bound remains future work and
is named as such in the paper's Limitations. Nothing below should be quoted
as "phi".

WHY A NEW GRID RATHER THAN A REFIT OF THE EXISTING SWEEPS. code/47's Sweep A
varies K at a fixed AUROC_0=0.80; its Sweep C varies AUROC_0 at a fixed K=45.
Together they form a CROSS through the (K, AUROC_0) plane, not a grid: there
is no cell in which both coordinates differ from the reference. A joint model
can be fit to a cross, but the fit cannot see -- let alone test -- an
interaction between the two axes, which is exactly the term that decides
whether a single separable function is adequate. So this script runs the
missing cells: a full K x AUROC_0 factorial, using code/47's harness
unchanged (same generative process, same LEAKY/CLEAN_MATCHED contrast, same
decoupled data/split/fold/init seed streams, same guardrail).

GRID: K in {3, 15, 45, 135} x AUROC_0 in {0.70, 0.80, 0.90, 0.95, 0.985},
n=100 seeds/cell, capacity 128, N_SAMPLES=700, K_CV=5 -- so n_val=112 is held
FIXED throughout. This is deliberate and is a stated limitation of the fit:
the review's phi has three arguments and this surface only spans two of them.
code/47's Sweep D varies n_val separately and finds it moves the gap in the
direction OPPOSITE to the winner's-curse prediction, tracking the operating
point instead, so n_val is not an independent third axis in this harness in
the way the theory assumes; folding it in would fit an artifact.

MODELS COMPARED (all least-squares on the 20 cell means, with z0 =
Phi^-1(AUROC_0), the natural monotone transform of the operating point --
it is the coordinate in which the harness's own binormal calibration is
linear, so a linear term in z0 is the parsimonious choice, not a fitted one):
  M0  gap = a                              (intercept only; the null)
  M1  gap = a + b ln K                     (this paper's existing K law)
  M2  gap = a + c z0                       (operating point only)
  M3  gap = a + b ln K + c z0              (additive joint)
  M4  gap = a + b ln K + c z0 + d ln K z0  (joint with interaction)
  M5  gap = exp(a + b ln K + c z0)         (multiplicative/log-linear, fit by
                                            NLS on the raw gaps)
Reported per model: R^2, adjusted R^2, and -- because 20 cells against 4
parameters invites overfitting -- LEAVE-ONE-CELL-OUT predictive R^2, which is
the number that actually says whether the surface generalizes within its own
grid. An F-test compares M4 against M3 for the interaction term.

OUT-OF-SAMPLE CHECK, AND EXACTLY HOW OUT-OF-SAMPLE IT IS. The five Sweep A
cells at K in {5, 10, 25, 75, 225}, AUROC_0=0.80, are NOT in this grid, so
their K values are genuinely held out from the fit and the models are scored
against them without refitting. They are NOT independent draws, though, and
we say so rather than let "out-of-sample" imply more than it is: Sweep A runs
200 seeds using the same seed->stream mapping this grid uses, so seeds 0-99
of every Sweep A cell are the same replicates the grid's cells draw from.
The check therefore tests generalization ACROSS K, which is what it is for,
and not generalization across data.

INTERNAL CONSISTENCY CHECK. The (K=45, AUROC_0) column of this grid is the
same configuration as code/47's Sweep C and must reproduce it cell for cell;
any divergence means the harness drifted and the run should be discarded.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "joint_severity_surface.json"
SWEEP_PATH = ROOT / "results" / "selection_multiplicity_sweep.json"

# Import code/47's harness verbatim -- same generative process, same decoupled
# seed streams, same LEAKY/CLEAN_MATCHED contrast. Nothing is reimplemented.
_SPEC = importlib.util.spec_from_file_location(
    "sweep47", Path(__file__).resolve().parent / "47_selection_multiplicity_sweep.py")
_M = importlib.util.module_from_spec(_SPEC)
sys.modules["sweep47"] = _M
_SPEC.loader.exec_module(_M)

run_sweep_cell = _M.run_sweep_cell
CAPACITY = _M.CAPACITY
DEFAULT_N_SAMPLES = _M.DEFAULT_N_SAMPLES

K_VALUES = [3, 15, 45, 135]
AUROC0_VALUES = [0.70, 0.80, 0.90, 0.95, 0.985]
N_SEEDS = 100


# ── model fitting ───────────────────────────────────────────────────────────

def _design(name, lnK, z0):
    if name == "M0":
        return np.column_stack([np.ones_like(lnK)])
    if name == "M1":
        return np.column_stack([np.ones_like(lnK), lnK])
    if name == "M2":
        return np.column_stack([np.ones_like(lnK), z0])
    if name == "M3":
        return np.column_stack([np.ones_like(lnK), lnK, z0])
    if name == "M4":
        return np.column_stack([np.ones_like(lnK), lnK, z0, lnK * z0])
    raise ValueError(name)


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_linear(name, lnK, z0, y):
    X = _design(name, lnK, z0)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    n, p = X.shape
    r2 = _r2(y, yhat)
    adj = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    # leave-one-cell-out predictive R^2
    loo = np.empty_like(y)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        loo[i] = X[i] @ b
    return {"coefficients": beta.tolist(), "n_params": int(p), "r_squared": r2,
            "adj_r_squared": float(adj), "loo_r_squared": _r2(y, loo),
            "ss_res": float(np.sum((y - yhat) ** 2)),
            "residuals": (y - yhat).tolist()}


def fit_loglinear(lnK, z0, y):
    """gap = exp(a + b lnK + c z0), fit by Gauss-Newton on the raw gaps.
    Initialized from an OLS fit on log(max(gap, eps)) over positive cells."""
    pos = y > 1e-6
    X = np.column_stack([np.ones_like(lnK), lnK, z0])
    b0, *_ = np.linalg.lstsq(X[pos], np.log(y[pos]), rcond=None)
    b = b0.copy()
    for _ in range(200):
        f = np.exp(X @ b)
        J = f[:, None] * X
        r = y - f
        try:
            step, *_ = np.linalg.lstsq(J, r, rcond=None)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    yhat = np.exp(X @ b)
    n, p = len(y), 3
    r2 = _r2(y, yhat)
    loo = np.empty_like(y)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        pm = pos & m
        bb, *_ = np.linalg.lstsq(X[pm], np.log(y[pm]), rcond=None)
        for _ in range(200):
            f = np.exp(X[m] @ bb)
            J = f[:, None] * X[m]
            try:
                step, *_ = np.linalg.lstsq(J, y[m] - f, rcond=None)
            except np.linalg.LinAlgError:
                break
            bb = bb + step
            if np.max(np.abs(step)) < 1e-12:
                break
        loo[i] = float(np.exp(X[i] @ bb))
    return {"coefficients": b.tolist(), "n_params": p, "r_squared": r2,
            "adj_r_squared": float(1 - (1 - r2) * (n - 1) / (n - p)),
            "loo_r_squared": _r2(y, loo),
            "ss_res": float(np.sum((y - yhat) ** 2)),
            "residuals": (y - yhat).tolist()}


def main():
    t0 = time.time()
    cells = {}
    print(f"Joint severity surface: {len(K_VALUES)}x{len(AUROC0_VALUES)} grid, "
          f"n={N_SEEDS} seeds/cell, capacity={CAPACITY}, "
          f"N_SAMPLES={DEFAULT_N_SAMPLES}, K_CV=5 (n_val=112 held fixed)", flush=True)
    for a0 in AUROC0_VALUES:
        for K in K_VALUES:
            r = run_sweep_cell(N_SEEDS, CAPACITY, K, DEFAULT_N_SAMPLES, a0)
            cells[f"{K}|{a0}"] = {**r, "K": K, "target_auroc": a0}
            print(f"  K={K:4d} AUROC0={a0:<6}: gap={r['gap_mean']:+.5f} "
                  f"CI={[round(x, 5) for x in r['gap_bca_ci_95']]} "
                  f"p={r['wilcoxon_p']:.3g} leaky={r['leaky_mean']:.4f}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    keys = list(cells)
    lnK = np.array([np.log(cells[k]["K"]) for k in keys])
    z0 = np.array([norm.ppf(cells[k]["target_auroc"]) for k in keys])
    y = np.array([cells[k]["gap_mean"] for k in keys])

    fits = {name: fit_linear(name, lnK, z0, y) for name in ["M0", "M1", "M2", "M3", "M4"]}
    fits["M5_loglinear"] = fit_loglinear(lnK, z0, y)

    # F-test: does the interaction term (M4) improve on the additive model (M3)?
    n = len(y)
    ss3, ss4 = fits["M3"]["ss_res"], fits["M4"]["ss_res"]
    f_stat = ((ss3 - ss4) / 1) / (ss4 / (n - 4))
    f_p = float(1 - f_dist.cdf(f_stat, 1, n - 4))

    # ── internal consistency: the K=45 column must reproduce code/47's Sweep C ──
    sweep = json.load(open(SWEEP_PATH))
    consistency = {}
    for a0 in AUROC0_VALUES:
        ref = sweep["sweep_C_operating_point"].get(str(a0))
        if ref is not None:
            here = cells[f"45|{a0}"]["gap_mean"]
            consistency[str(a0)] = {"sweep_C": ref["gap_mean"], "grid_K45": here,
                                    "abs_diff": abs(ref["gap_mean"] - here)}

    # ── out-of-sample: Sweep A cells at K not in this grid, AUROC_0 = 0.80 ──
    oos_K = [5, 10, 25, 75, 225]
    oos = {"K": oos_K, "observed": [], "predicted": {}}
    for K in oos_K:
        oos["observed"].append(sweep["sweep_A_K"][str(K)]["gap_mean"])
    oos["observed"] = np.array(oos["observed"])
    ln_o = np.log(np.array(oos_K, dtype=float))
    z_o = np.full(len(oos_K), norm.ppf(0.80))
    for name in ["M1", "M3", "M4"]:
        b = np.array(fits[name]["coefficients"])
        pred = _design(name, ln_o, z_o) @ b
        oos["predicted"][name] = {"values": pred.tolist(),
                                  "r_squared_vs_observed": _r2(oos["observed"], pred)}
    b5 = np.array(fits["M5_loglinear"]["coefficients"])
    pred5 = np.exp(np.column_stack([np.ones_like(ln_o), ln_o, z_o]) @ b5)
    oos["predicted"]["M5_loglinear"] = {"values": pred5.tolist(),
                                        "r_squared_vs_observed": _r2(oos["observed"], pred5)}
    oos["observed"] = oos["observed"].tolist()

    out = {
        "framing": (
            "EMPIRICAL JOINT FIT, not a bound and not a theorem. Measured in one "
            "synthetic harness with the leakage mechanism held fixed. Carries no "
            "guarantee outside this grid, this harness, or this mechanism. A "
            "formally-derived upper bound phi(K, n_sel, AUROC_0) remains future work."),
        "grid": {"K_values": K_VALUES, "auroc0_values": AUROC0_VALUES,
                 "n_seeds_per_cell": N_SEEDS, "capacity": CAPACITY,
                 "n_samples": DEFAULT_N_SAMPLES, "k_cv": 5, "n_val_fixed": 112},
        "n_val_note": (
            "n_val is held FIXED at 112 across the whole grid, so this surface spans "
            "two of the review's three arguments, not three. code/47's Sweep D varies "
            "n_val in isolation and finds the gap moves OPPOSITE to the 1/sqrt(n_val) "
            "prediction, tracking the operating point instead; there is therefore no "
            "independent n_val axis in this harness to fold in honestly."),
        "cells": cells,
        "predictors": {"lnK": lnK.tolist(), "z0_probit_of_auroc0": z0.tolist(),
                       "gap": y.tolist(), "cell_order": keys},
        "fits": fits,
        "interaction_f_test": {"model_compared": "M4 (with lnK*z0) vs M3 (additive)",
                               "f_stat": float(f_stat), "df": [1, n - 4], "p": f_p},
        "internal_consistency_vs_sweep_C": consistency,
        "out_of_sample_sweep_A_cells": oos,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== fits (20 cells) ===")
    for name, fr in fits.items():
        print(f"  {name:14s} p={fr['n_params']}  R^2={fr['r_squared']:+.4f}  "
              f"adjR^2={fr['adj_r_squared']:+.4f}  LOO-R^2={fr['loo_r_squared']:+.4f}")
    print(f"  interaction F-test: F={f_stat:.3f}, p={f_p:.4g}")
    print("\n=== internal consistency vs code/47 Sweep C (K=45 column) ===")
    for k, v in consistency.items():
        print(f"  AUROC0={k}: sweep_C={v['sweep_C']:+.5f} grid={v['grid_K45']:+.5f} "
              f"|diff|={v['abs_diff']:.2e}")
    print("\n=== out-of-sample (Sweep A cells not in this grid) ===")
    for name, v in oos["predicted"].items():
        print(f"  {name:14s} out-of-sample R^2 = {v['r_squared_vs_observed']:+.4f}")
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
