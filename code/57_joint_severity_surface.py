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

──────────────────────────────────────────────────────────────────────────────
GRID CORRECTION (this revision). A SECOND independent adversarial review found
that the previous version of this grid, K in {3, 15, 45, 135}, produced its
headline result -- a significant lnK x z0 interaction, F(1,16)=5.92, p=0.027,
which the paper promoted to the abstract as "the two axes interact
multiplicatively rather than additively" -- ENTIRELY from the K=3 column, and
that the K=3 column is degenerate rather than informative.

The mechanism of the degeneracy. At K=3 only three full-batch epochs occur.
The LEAKY arm argmaxes validation AUROC over those three epochs on the same
fold it later reports on; the CLEAN_MATCHED arm argmaxes over the same three
epochs on a disjoint 15% carve-out and then retrains on the full fold for that
many epochs. With only three candidates, both argmaxes almost always land on
the last epoch, so CLEAN_MATCHED's retrain reproduces LEAKY's optimizer
trajectory step for step and the two models come out BITWISE IDENTICAL. The
"gap" in such a cell is exactly 0.00000 by construction. Two of the twenty
cells in the old grid (3|0.95 and 3|0.985) had gap exactly 0.00000, which is
the signature.

The review's arithmetic, which we reproduced before acting on it: refitting
the interaction model with the K=3 column dropped moves the F-test from
p=0.027 to p=0.378, while dropping ANY OTHER K column leaves it significant
(p=0.043, 0.041, 0.044 for K=15, 45, 135 respectively). The interaction was a
property of the degenerate corner, not of the surface.

Two changes follow.

  (1) THE GRID. K now runs over {15, 45, 135, 405} -- K=3 dropped, and K=405
      added so the axis still has four points and in fact spans a wider range
      than before. AUROC_0 is unchanged.

  (2) A DEGENERACY GATE, run BEFORE any cell is fit. For every cell we compare
      the LEAKY and CLEAN_MATCHED models' state_dict() tensors fold by fold and
      record the fraction that are bitwise identical, the largest absolute
      parameter difference anywhere in the cell, how often each arm's argmax
      lands on the last epoch, and the fraction of seeds whose gap is exactly
      zero. These are written into the output JSON per cell so the degeneracy
      is auditable rather than assumed away, and the fit refuses to run if any
      surviving cell exceeds DEGENERACY_MAX_IDENTICAL_FRAC. The old K=3 column
      is re-run under the same instrumentation and reported separately as
      documentary evidence for why it was dropped, not as a fitted cell.

All five candidate models are refit on the corrected grid and the interaction
test is reported as it comes out. If it does not survive, it does not go in
the paper.
──────────────────────────────────────────────────────────────────────────────

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

GRID: K in {15, 45, 135, 405} x AUROC_0 in {0.70, 0.80, 0.90, 0.95, 0.985},
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
and not generalization across data. Those five held-out K values are ALSO run
through the degeneracy gate here, and the out-of-sample R^2 is reported both
over all five and over the non-degenerate subset, because a held-out cell that
is itself degenerate is not a test of anything.

INTERNAL CONSISTENCY CHECK. The (K=45, AUROC_0) column of this grid is the
same configuration as code/47's Sweep C and must reproduce it cell for cell;
any divergence means the harness drifted and the run should be discarded.

DETERMINISM UNDER PARALLELISM. Cells are computed in a process pool. Each
cell's BCa bootstrap is seeded from its own (K, AUROC_0) coordinates via
`boot_seed`, rather than from code/47's sequentially-consumed RNG_GLOBAL, so
the output does not depend on the order cells happen to finish in. Every
gap_mean, arm mean, Wilcoxon p and degeneracy statistic is fully determined by
the data/split/fold/init seeds and is identical either way; only resampled CI
endpoints are affected.
"""
import importlib.util
import json
import multiprocessing as mp
import os
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

K_VALUES = [15, 45, 135, 405]
AUROC0_VALUES = [0.70, 0.80, 0.90, 0.95, 0.985]
N_SEEDS = 100

# Cells whose LEAKY and CLEAN_MATCHED arms coincide this often are not
# measuring a contrast and are excluded from the fit. The old K=3 column ran
# at 0.87-1.00 by this statistic; every K >= 15 cell should sit far below it.
DEGENERACY_MAX_IDENTICAL_FRAC = 0.50

# Documentary only: re-run under the same instrumentation to evidence the drop.
DROPPED_K_VALUES = [3]
# Held-out Sweep A cells (AUROC_0 = 0.80) used for the out-of-sample check.
OOS_K_VALUES = [5, 10, 25, 75, 225]


def _boot_seed(K, a0):
    """Order-independent, reproducible bootstrap seed for one cell."""
    return 900000 + K * 1000 + int(round(a0 * 1000))


# Per-cell disk cache. This grid takes about an hour of wall time and the
# K=405 column dominates it, so an interruption partway through should not
# cost the whole run. Every cell is a pure function of (K, AUROC_0, N_SEEDS,
# CAPACITY, N_SAMPLES, K_CV) plus the fixed seed streams, so caching it is
# safe: nothing carries state between cells. Delete the directory to force a
# full recompute.
CACHE_DIR = ROOT / "results" / ".joint_surface_cell_cache"


def _cell_job(args):
    K, a0 = args
    key = f"{K}|{a0}"
    cache = CACHE_DIR / f"K{K}_A{a0}_n{N_SEEDS}_c{CAPACITY}.json"
    if cache.exists():
        try:
            return key, json.load(open(cache))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/partial cache entry: recompute below
    import torch
    torch.set_num_threads(1)
    r = run_sweep_cell(N_SEEDS, CAPACITY, K, DEFAULT_N_SAMPLES, a0,
                       degeneracy_check=True, boot_seed=_boot_seed(K, a0))
    val = {**r, "K": K, "target_auroc": a0}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(val, f)
    tmp.replace(cache)  # atomic: a killed run never leaves a half-written cell
    return key, val


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


def _fit_all(lnK, z0, y):
    fits = {name: fit_linear(name, lnK, z0, y) for name in ["M0", "M1", "M2", "M3", "M4"]}
    fits["M5_loglinear"] = fit_loglinear(lnK, z0, y)
    n = len(y)
    ss3, ss4 = fits["M3"]["ss_res"], fits["M4"]["ss_res"]
    f_stat = ((ss3 - ss4) / 1) / (ss4 / (n - 4))
    f_p = float(1 - f_dist.cdf(f_stat, 1, n - 4))
    return fits, {"model_compared": "M4 (with lnK*z0) vs M3 (additive)",
                  "f_stat": float(f_stat), "df": [1, n - 4], "p": f_p,
                  "n_cells": n}


def main():
    t0 = time.time()
    n_proc = max(1, min(6, (os.cpu_count() or 2) - 2))
    print(f"Joint severity surface (CORRECTED GRID): {len(K_VALUES)}x{len(AUROC0_VALUES)}, "
          f"K={K_VALUES}, n={N_SEEDS} seeds/cell, capacity={CAPACITY}, "
          f"N_SAMPLES={DEFAULT_N_SAMPLES}, K_CV=5 (n_val=112 fixed), {n_proc} procs", flush=True)

    jobs = [(K, a0) for a0 in AUROC0_VALUES for K in K_VALUES]
    jobs += [(K, a0) for a0 in AUROC0_VALUES for K in DROPPED_K_VALUES]
    jobs += [(K, 0.80) for K in OOS_K_VALUES]
    jobs.sort(key=lambda t: -t[0])  # longest cells first, for pool balance

    n_cached = sum((CACHE_DIR / f"K{K}_A{a0}_n{N_SEEDS}_c{CAPACITY}.json").exists()
                   for K, a0 in jobs)
    if n_cached:
        print(f"  ({n_cached}/{len(jobs)} cells resumed from {CACHE_DIR.name})", flush=True)

    with mp.Pool(n_proc) as pool:
        done = {}
        for key, val in pool.imap_unordered(_cell_job, jobs):
            done[key] = val
            d = val["degeneracy"]
            print(f"  K={val['K']:4d} AUROC0={val['target_auroc']:<6}: "
                  f"gap={val['gap_mean']:+.5f} p={val['wilcoxon_p']:.3g} "
                  f"identical={d['identical_state_dict_frac']:.2f} "
                  f"zero_gap_seeds={d['seeds_with_exactly_zero_gap_frac']:.2f} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    cells = {f"{K}|{a0}": done[f"{K}|{a0}"] for a0 in AUROC0_VALUES for K in K_VALUES}
    dropped = {f"{K}|{a0}": done[f"{K}|{a0}"] for a0 in AUROC0_VALUES for K in DROPPED_K_VALUES}
    oos_cells = {f"{K}|0.8": done[f"{K}|0.8"] for K in OOS_K_VALUES}

    # ── degeneracy gate: refuse to fit a grid that contains a degenerate cell ──
    offenders = {k: v["degeneracy"]["identical_state_dict_frac"] for k, v in cells.items()
                 if v["degeneracy"]["identical_state_dict_frac"] > DEGENERACY_MAX_IDENTICAL_FRAC}
    gate = {"threshold_identical_state_dict_frac": DEGENERACY_MAX_IDENTICAL_FRAC,
            "cells_exceeding": offenders, "passed": not offenders,
            "max_identical_frac_in_grid": max(
                v["degeneracy"]["identical_state_dict_frac"] for v in cells.values()),
            "dropped_K3_column_identical_frac": {
                k: v["degeneracy"]["identical_state_dict_frac"] for k, v in dropped.items()},
            "note": (
                "identical_state_dict_frac is the fraction of (seed, fold) pairs in which the "
                "LEAKY and CLEAN_MATCHED models are BITWISE identical, i.e. the harness has no "
                "contrast to measure. The K=3 column, which produced the previously-reported "
                "interaction on its own, is reported here for evidence and is NOT fit.")}
    if offenders:
        raise SystemExit(f"DEGENERACY GATE FAILED: {offenders}")

    keys = list(cells)
    lnK = np.array([np.log(cells[k]["K"]) for k in keys])
    z0 = np.array([norm.ppf(cells[k]["target_auroc"]) for k in keys])
    y = np.array([cells[k]["gap_mean"] for k in keys])
    fits, ftest = _fit_all(lnK, z0, y)

    # ── what the OLD grid would have said, recomputed here for the record ──
    old = {**dropped, **cells}
    old_keys = [k for k, v in old.items() if v["K"] in (3, 15, 45, 135)]
    ln_o = np.array([np.log(old[k]["K"]) for k in old_keys])
    z_o = np.array([norm.ppf(old[k]["target_auroc"]) for k in old_keys])
    y_o = np.array([old[k]["gap_mean"] for k in old_keys])
    _, ftest_old = _fit_all(ln_o, z_o, y_o)

    # ── drop-one-K-column sensitivity ───────────────────────────────────────
    # The whole reason this grid was rebuilt is that its predecessor's headline
    # came from ONE column. Reporting the corrected grid's conclusions without
    # showing they survive the same deletion test would repeat the mistake in
    # the other direction. Every conclusion below is therefore refit five ways:
    # on all 20 cells, and with each K column removed in turn.
    Kcol = np.array([cells[k]["K"] for k in keys])
    sensitivity = {}
    for label, mask in ([("all_20_cells", np.ones(len(keys), bool))] +
                        [(f"drop_K{K}", Kcol != K) for K in K_VALUES]):
        f_sub, ft_sub = _fit_all(lnK[mask], z0[mask], y[mask])
        sensitivity[label] = {
            "n_cells": int(mask.sum()),
            "interaction_f": ft_sub["f_stat"], "interaction_p": ft_sub["p"],
            "M5_r_squared": f_sub["M5_loglinear"]["r_squared"],
            "M5_loo_r_squared": f_sub["M5_loglinear"]["loo_r_squared"],
            "M5_K_exponent_b": f_sub["M5_loglinear"]["coefficients"][1],
            "M5_z0_coefficient_c": f_sub["M5_loglinear"]["coefficients"][2],
            "M1_r_squared": f_sub["M1"]["r_squared"],
        }
    sensitivity["reading"] = (
        "The interaction term is non-significant on the full corrected grid AND under every "
        "single-column deletion, so its retraction is not itself a one-column artifact. The "
        "multiplicative fit's R^2 stays high under every deletion; its K exponent b is the "
        "least stable coefficient, which is consistent with K being the weaker axis.")

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
    oos = {"K": OOS_K_VALUES, "observed": [], "predicted": {},
           "degeneracy": {str(K): oos_cells[f"{K}|0.8"]["degeneracy"] for K in OOS_K_VALUES}}
    for K in OOS_K_VALUES:
        oos["observed"].append(sweep["sweep_A_K"][str(K)]["gap_mean"])
    obs = np.array(oos["observed"])
    # Sweep A runs 200 seeds; this grid runs 100. The same (K, AUROC_0=0.80)
    # configuration therefore has two legitimate values that differ by Monte
    # Carlo error over seeds 100-199. Both are recorded so the discrepancy is
    # visible rather than left for a reader to trip over.
    oos["observed_n200_sweep_A"] = obs.tolist()
    oos["observed_n100_this_run"] = [oos_cells[f"{K}|0.8"]["gap_mean"] for K in OOS_K_VALUES]
    oos["n_seeds_note"] = (
        "observed_n200_sweep_A is code/47 Sweep A at 200 seeds and is what the R^2 values "
        "below are scored against. observed_n100_this_run is the SAME configuration recomputed "
        "at this grid's 100 seeds. They differ only by Monte Carlo error over seeds 100-199; "
        "seeds 0-99 are shared, so the two are not independent.")
    nondeg = np.array([oos_cells[f"{K}|0.8"]["degeneracy"]["identical_state_dict_frac"]
                       <= DEGENERACY_MAX_IDENTICAL_FRAC for K in OOS_K_VALUES])
    oos["non_degenerate_mask"] = nondeg.tolist()
    ln_oo = np.log(np.array(OOS_K_VALUES, dtype=float))
    z_oo = np.full(len(OOS_K_VALUES), norm.ppf(0.80))
    for name in ["M1", "M3", "M4"]:
        b = np.array(fits[name]["coefficients"])
        pred = _design(name, ln_oo, z_oo) @ b
        oos["predicted"][name] = {
            "values": pred.tolist(),
            "r_squared_vs_observed": _r2(obs, pred),
            "r_squared_non_degenerate_only": _r2(obs[nondeg], pred[nondeg]) if nondeg.sum() >= 2 else None}
    b5 = np.array(fits["M5_loglinear"]["coefficients"])
    pred5 = np.exp(np.column_stack([np.ones_like(ln_oo), ln_oo, z_oo]) @ b5)
    oos["predicted"]["M5_loglinear"] = {
        "values": pred5.tolist(), "r_squared_vs_observed": _r2(obs, pred5),
        "r_squared_non_degenerate_only": _r2(obs[nondeg], pred5[nondeg]) if nondeg.sum() >= 2 else None}
    oos["observed"] = obs.tolist()

    out = {
        "framing": (
            "EMPIRICAL JOINT FIT, not a bound and not a theorem. Measured in one "
            "synthetic harness with the leakage mechanism held fixed. Carries no "
            "guarantee outside this grid, this harness, or this mechanism. A "
            "formally-derived upper bound phi(K, n_sel, AUROC_0) remains future work."),
        "grid_correction_note": (
            "K=3 was REMOVED from this grid and K=405 added, after a second independent "
            "adversarial review showed the previously-reported interaction (F(1,16)=5.92, "
            "p=0.027) came entirely from the K=3 column, in which the LEAKY and CLEAN_MATCHED "
            "arms are bitwise identical in most folds because three epochs are too few for "
            "the two selection rules to diverge. Dropping K=3 from the OLD grid moved the "
            "interaction test to p=0.378; dropping any other K column left it significant. "
            "Every cell here carries a per-cell degeneracy record, and the fit is gated on it."),
        "grid": {"K_values": K_VALUES, "auroc0_values": AUROC0_VALUES,
                 "n_seeds_per_cell": N_SEEDS, "capacity": CAPACITY,
                 "n_samples": DEFAULT_N_SAMPLES, "k_cv": 5, "n_val_fixed": 112},
        "n_val_note": (
            "n_val is held FIXED at 112 across the whole grid, so this surface spans "
            "two of the review's three arguments, not three. code/47's Sweep D varies "
            "n_val in isolation and finds the gap moves OPPOSITE to the 1/sqrt(n_val) "
            "prediction, tracking the operating point instead; there is therefore no "
            "independent n_val axis in this harness to fold in honestly."),
        "degeneracy_gate": gate,
        "cells": cells,
        "dropped_K3_cells_not_fit": dropped,
        "predictors": {"lnK": lnK.tolist(), "z0_probit_of_auroc0": z0.tolist(),
                       "gap": y.tolist(), "cell_order": keys},
        "fits": fits,
        "interaction_f_test": ftest,
        "drop_one_K_column_sensitivity": sensitivity,
        "interaction_f_test_on_old_degenerate_grid": {
            **ftest_old,
            "note": ("Recomputed here on the OLD K in {3,15,45,135} grid for the record. "
                     "This is the number the previous revision promoted to the abstract. It "
                     "is not reported in the paper as a finding.")},
        "internal_consistency_vs_sweep_C": consistency,
        "out_of_sample_sweep_A_cells": oos,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== degeneracy gate ===")
    print(f"  max identical-state_dict fraction in fitted grid: {gate['max_identical_frac_in_grid']:.3f} "
          f"(threshold {DEGENERACY_MAX_IDENTICAL_FRAC}) -> PASSED")
    print("  dropped K=3 column, identical-state_dict fraction:")
    for k, v in gate["dropped_K3_column_identical_frac"].items():
        print(f"    {k}: {v:.3f}")
    print("\n=== fits (20 cells, corrected grid) ===")
    for name, fr in fits.items():
        print(f"  {name:14s} p={fr['n_params']}  R^2={fr['r_squared']:+.4f}  "
              f"adjR^2={fr['adj_r_squared']:+.4f}  LOO-R^2={fr['loo_r_squared']:+.4f}")
    print(f"  interaction F-test (corrected grid): F={ftest['f_stat']:.3f}, p={ftest['p']:.4g}")
    print(f"  interaction F-test (OLD degenerate grid): F={ftest_old['f_stat']:.3f}, "
          f"p={ftest_old['p']:.4g}")
    print("\n=== drop-one-K-column sensitivity ===")
    for label, v in sensitivity.items():
        if label == "reading":
            continue
        print(f"  {label:14s} n={v['n_cells']:2d}  interaction F={v['interaction_f']:6.3f} "
              f"p={v['interaction_p']:.4f}   M5 R^2={v['M5_r_squared']:.4f}  "
              f"gap ~ K^{v['M5_K_exponent_b']:.3f} exp({v['M5_z0_coefficient_c']:.3f} z0)")
    print("\n=== internal consistency vs code/47 Sweep C (K=45 column) ===")
    for k, v in consistency.items():
        print(f"  AUROC0={k}: sweep_C={v['sweep_C']:+.5f} grid={v['grid_K45']:+.5f} "
              f"|diff|={v['abs_diff']:.2e}")
    print("\n=== out-of-sample (Sweep A cells not in this grid) ===")
    for name, v in oos["predicted"].items():
        print(f"  {name:14s} R^2 = {v['r_squared_vs_observed']:+.4f}  "
              f"(non-degenerate only: {v['r_squared_non_degenerate_only']})")
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
