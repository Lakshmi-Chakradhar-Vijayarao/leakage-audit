"""
Paper 2 -- Case Study 2 (GUARDIAN) full 32-layer decomposition.

An earlier version of this paper's Appendix A disclosed this as unavailable
("no per-layer data exists"). That was wrong: GUARDIAN's own sibling project
(`~/Desktop/guardian/results/hidden_states/mistral_7b_tqa_hidden_states.npz`)
contains the full (700, 32, 4096) raw Mistral-7B hidden states -- all 700
samples (the 400-sample CV-selection pool and the 300-sample held-out set),
at every layer, in full dimensionality. This reproduces the paper's already-
published L11 numbers exactly under the paper's ORIGINAL sequential-split
protocol (CV=0.804, held-out=0.616) as a sanity check, then extends the same
computation to all 32 layers.

CORRECTION (independent adversarial review found this): the paper's own
`H[:400]` / `H[400:]` split is a fixed, unrandomized sequential slice of
whatever order the underlying dataset happened to store samples in -- not a
random train/held-out partition. A hidden-state probe can separate the two
halves at AUROC 0.734-0.776, i.e. the two halves are systematically
different populations, not exchangeable draws.

SECOND CORRECTION (a subsequent independent review found this): the first
correction's `selection_specific_component()` was itself computed at a
hardcoded layer (11) in every randomized-split rep, not at that rep's own
CV-argmax layer. Since the whole point of "selection-specific" is
gap-at-whatever-layer-the-procedure-would-actually-select minus the mean
gap, hardcoding the layer measures L11's own idiosyncrasy under each
random split, not selection optimism -- silently biasing the result toward
whatever L11 happens to look like, not what argmax-driven selection
actually costs. Fixed: every call site now uses each split's own
`l_star = argmax(cv_auroc)`.

Recomputing the "selection-specific" component of the optimism gap
(gap_at_the_layer_that_would_be_selected - mean_gap_across_all_layers)
under the sequential split (which happens to select L11) gives the
original -0.0038 (no real selection optimism there, an artifact of
comparing two non-exchangeable populations). Reversing the sequential
split (using H[400:] to select and H[:400] as held-out) selects a
different layer and flips sign, itself evidence the sequential-split
number tracks which half is easier, not a stable selection-optimism
estimate. Under randomized stratified splits (each rep's own selected
layer), the corrected result is reported directly -- see the module
docstring is deliberately not pre-stating this number, since it is
exactly what this rerun exists to measure correctly for the first time.

This script now reports, and disclosure-labels, all three: (1) the original
sequential split (kept for continuity with the previously published
numbers, explicitly labeled as protocol-confounded), (2) N_REPS randomized
stratified splits (the corrected primary result, each rep using its own
selected layer), and (3) the reversed sequential split (diagnostic showing
the sign flip).

THIRD CORRECTION (a further independent adversarial review found two things):

(a) N_REPS WAS 8, WITH NO STATED STOPPING RULE. Eight is a small enough
    number that a reader cannot rule out its having been chosen after
    inspecting the result. It is now 50. The binding constraint is not
    runtime (50 reps takes ~15 min on one core) but the size of the shipped
    derived replay artifact, which stores every per-layer, per-sample probe
    score for every rep so code/51 can re-derive the headline number without
    the 171 MB raw hidden-state cache: that artifact grows ~100 KB/rep, and
    50 reps keeps it around 5 MB, which is the largest we were willing to
    put in a submission archive. The RNG stream is unchanged, so reps 0-7
    are bit-identical to the previously reported 8.

    This does NOT fix the deeper statistical problem, which is now disclosed
    in the paper rather than left implicit: all reps resample splits of the
    SAME underlying 700 samples. The SD across reps therefore measures
    split-to-split variability conditional on this one dataset, not sampling
    variability across independent draws from the population, so the
    one-sample t-test's i.i.d. assumption is violated and its p-value is a
    within-dataset robustness statistic, not a population-generalizing
    significance test. More reps tighten the SD estimate; they cannot
    manufacture independent samples that do not exist.

(b) A LIVE ALTERNATIVE EXPLANATION FOR THE ~ZERO GENERAL GAP WAS UNTESTED.
    The probe here is LogisticRegression at d=4096, n=400, max_iter=1000,
    default C=1.0, on UNSCALED raw hidden states. That configuration could
    plausibly be so heavily regularized toward its prior that both the CV and
    the held-out estimate are dominated by regularization rather than by
    genuine fit -- which would be a different story from "CV and held-out
    estimates are exchangeable under a random split." The `regularization_
    robustness` block below tests this directly: StandardScaler + a sweep of
    C over five decades {0.01, 0.1, 1, 10, 100}, N_REPS_ROBUST randomized
    splits each, reporting the selection-specific component and the general
    gap at every setting. Convergence is also checked explicitly (max
    `n_iter_` and any ConvergenceWarning) rather than assumed.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

GUARDIAN_ROOT = Path(os.path.expanduser(os.environ.get("GUARDIAN_ROOT", "~/Desktop/guardian")))
GUARDIAN_NPZ = GUARDIAN_ROOT / "results" / "hidden_states" / "mistral_7b_tqa_hidden_states.npz"
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "case_study_2_layer_decomposition.json"
# Small, shipped, derived artifact so a reader can re-derive every number in
# OUT_PATH without the 171 MB raw hidden-state cache (see code/51).
ARTIFACT_PATH = ROOT / "results" / "case_study_2_probe_scores.npz"

N_TRAIN_SELECT = 400
RANDOM_STATE = 42
# Raised from 8 to 50 (third correction, item (a) in the module docstring).
# Stopping point is set by the size of the shipped replay artifact
# (~100 KB/rep), not by runtime. Reps 0-7 are bit-identical to the previously
# reported 8 because the RNG stream is unchanged.
N_REPS = int(os.environ.get("N_REPS_OVERRIDE", 50))
# Reps used for the regularization-robustness sweep (item (b)). Fewer, because
# it is 5 C-values wide; it reuses the same RNG stream, so its rep r is the
# same split as the primary result's rep r.
N_REPS_ROBUST = int(os.environ.get("N_REPS_ROBUST_OVERRIDE", 20))
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# Every probe score computed anywhere in this script is recorded here and
# written to ARTIFACT_PATH at the end, keyed by split-variant name.
ARTIFACT = {}


def _fit_probe(X_tr, y_tr, scale, C, conv):
    """Fit the layer probe. `scale`/`C` are the knobs the regularization-
    robustness sweep varies; the primary result uses the original (scale=False,
    C=1.0) configuration so its numbers are unchanged. `conv` accumulates
    convergence diagnostics so max_iter adequacy is checked, not assumed."""
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr = sc.transform(X_tr)
    else:
        sc = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(max_iter=1000, C=C).fit(X_tr, y_tr)
        if any(issubclass(x.category, ConvergenceWarning) for x in w):
            conv["n_convergence_warnings"] += 1
    conv["max_n_iter"] = max(conv["max_n_iter"], int(np.max(clf.n_iter_)))
    conv["n_fits"] += 1
    return clf, sc


def _score(clf, sc, X):
    return clf.predict_proba(sc.transform(X) if sc is not None else X)[:, 1]


def per_layer_gaps(H_sel, y_sel, H_ho, y_ho, n_layers, cv, variant=None,
                   scale=False, C=1.0, conv=None):
    """Per-layer CV-on-selection-pool vs. fit-on-pool/score-on-held-out AUROC.

    The CV number is the mean of the per-fold AUROCs (identical to what
    `cross_val_score(..., scoring='roc_auc')` returned in the previous version
    of this script, which is replaced here by an explicit fold loop only so the
    per-sample scores and fold assignments can be recorded for the shipped
    derived artifact -- the arithmetic is unchanged).
    """
    if conv is None:
        conv = {"n_fits": 0, "max_n_iter": 0, "n_convergence_warnings": 0}
    n_sel = len(y_sel)
    fold_id = np.full(n_sel, -1, dtype=np.int16)
    cv_scores = np.zeros((n_layers, n_sel), dtype=np.float32)
    ho_scores = np.zeros((n_layers, len(y_ho)), dtype=np.float32)

    splits = list(cv.split(np.zeros(n_sel), y_sel))
    for f, (_, te) in enumerate(splits):
        fold_id[te] = f

    per_layer = []
    for l in range(n_layers):
        X_sel = H_sel[:, l, :]
        X_ho = H_ho[:, l, :]
        fold_aurocs = []
        for f, (tr, te) in enumerate(splits):
            clf_f, sc_f = _fit_probe(X_sel[tr], y_sel[tr], scale, C, conv)
            s = _score(clf_f, sc_f, X_sel[te])
            cv_scores[l, te] = s
            fold_aurocs.append(roc_auc_score(y_sel[te], s))
        cv_auroc = float(np.mean(fold_aurocs))
        clf, sc_full = _fit_probe(X_sel, y_sel, scale, C, conv)
        s_ho = _score(clf, sc_full, X_ho)
        ho_scores[l] = s_ho
        heldout_auroc = float(roc_auc_score(y_ho, s_ho))
        per_layer.append({
            "layer": l, "cv_auroc_selectpool": cv_auroc,
            "heldout_auroc": heldout_auroc, "optimism_gap": cv_auroc - heldout_auroc,
        })

    if variant is not None:
        ARTIFACT[f"{variant}__y_sel"] = np.asarray(y_sel, dtype=np.int8)
        ARTIFACT[f"{variant}__y_ho"] = np.asarray(y_ho, dtype=np.int8)
        ARTIFACT[f"{variant}__fold_id"] = fold_id
        ARTIFACT[f"{variant}__cv_scores"] = cv_scores
        ARTIFACT[f"{variant}__ho_scores"] = ho_scores
    return per_layer


def selection_specific_component(per_layer, l_star):
    """gap_at_the_selected_layer minus the mean gap across all layers.

    KNOWN, DELIBERATE MECHANICAL PROPERTY (disclosed after an independent
    review noted it could be over-read): this quantity is positive in
    expectation even under a pure-noise null. l_star = argmax_l cv_l, and
    gap_l = cv_l - ho_l shares that same +cv_l term, so selecting the argmax
    of cv necessarily selects a layer whose gap is upward-biased relative to
    the mean gap. That is not a defect -- it is exactly the winner's curse
    this metric exists to measure, and it is why the metric is the right one.
    But it means a reader must NOT read the measured component's statistical
    significance as evidence of anything beyond argmax-over-a-noisy-statistic:
    a significant positive value is the expected signature of selection on a
    noisy criterion, not additional evidence that layer choice carries some
    further, separate pathology."""
    gaps = np.array([r["optimism_gap"] for r in per_layer])
    return float(gaps[l_star] - gaps.mean()), gaps


def random_stratified_split(rng, y, n_sel):
    """One randomized stratified selection/held-out partition. Factored out of
    main() so the regularization-robustness sweep can reproduce exactly the
    same splits (same seed -> same stream -> rep r is the same partition)."""
    n = len(y)
    idx = rng.permutation(n)
    pos_idx = idx[y[idx] == 1]
    neg_idx = idx[y[idx] == 0]
    n_pos_sel = int(round(n_sel * y.mean()))
    n_neg_sel = n_sel - n_pos_sel
    sel_idx = np.concatenate([pos_idx[:n_pos_sel], neg_idx[:n_neg_sel]])
    ho_idx = np.concatenate([pos_idx[n_pos_sel:], neg_idx[n_neg_sel:]])
    rng.shuffle(sel_idx)
    rng.shuffle(ho_idx)
    return sel_idx, ho_idx


def main():
    d = np.load(GUARDIAN_NPZ)
    H, y = d["hidden_states"], d["labels"]
    valid = y >= 0
    H, y = H[valid], y[valid]
    n_layers = H.shape[1]
    n = len(y)
    print(f"Loaded {GUARDIAN_NPZ.name}: {H.shape}, n_valid={valid.sum()}, hall_rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # (1) Original sequential split -- kept for continuity, labeled as confounded.
    print("\n=== (1) ORIGINAL sequential split H[:400]/H[400:] (protocol-confounded) ===", flush=True)
    H_sel, y_sel = H[:N_TRAIN_SELECT], y[:N_TRAIN_SELECT]
    H_ho, y_ho = H[N_TRAIN_SELECT:], y[N_TRAIN_SELECT:]
    per_layer_seq = per_layer_gaps(H_sel, y_sel, H_ho, y_ho, n_layers, cv, variant="sequential")
    for r in per_layer_seq:
        print(f"  L{r['layer']:02d}: CV={r['cv_auroc_selectpool']:.4f}  held-out={r['heldout_auroc']:.4f}  gap={r['optimism_gap']:+.4f}", flush=True)
    l_star_seq = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer_seq]))
    sel_component_seq, gaps_seq = selection_specific_component(per_layer_seq, l_star_seq)
    print(f"  Selected layer (own argmax): L{l_star_seq}", flush=True)

    # (2) Reversed sequential split -- diagnostic for population-difference artifact.
    # FIXED (second review): use this split's OWN argmax layer, not a hardcoded 11 --
    # a different split can select a different layer, and "selection-specific"
    # is only meaningful evaluated at the layer that split's own procedure picks.
    print("\n=== (2) REVERSED sequential split H[400:]=select / H[:400]=held-out ===", flush=True)
    per_layer_rev = per_layer_gaps(H_ho, y_ho, H_sel, y_sel, n_layers, cv, variant="reversed")
    l_star_rev = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer_rev]))
    sel_component_rev, gaps_rev = selection_specific_component(per_layer_rev, l_star_rev)
    print(f"  Selected layer (own argmax): L{l_star_rev} | gap at that layer: {gaps_rev[l_star_rev]:+.4f} "
          f"| selection-specific component (reversed): {sel_component_rev:+.4f}", flush=True)

    # (3) Randomized stratified splits (N_REPS reps) -- corrected primary result.
    # FIXED (second review): each rep must use ITS OWN argmax-selected layer, not a
    # hardcoded 11 -- the entire quantity being measured is "what does the argmax
    # procedure's own choice cost," which requires evaluating at that rep's choice.
    print(f"\n=== (3) {N_REPS} randomized stratified 400/{n-N_TRAIN_SELECT} splits (corrected) ===", flush=True)
    rand_sel_components = []
    rand_gaps_at_selected = []
    rand_selected_layers = []
    rand_mean_gap_all_layers = []
    rng = np.random.default_rng(RANDOM_STATE)
    conv_primary = {"n_fits": 0, "max_n_iter": 0, "n_convergence_warnings": 0}
    for rep in range(N_REPS):
        sel_idx, ho_idx = random_stratified_split(rng, y, N_TRAIN_SELECT)
        per_layer_rand = per_layer_gaps(H[sel_idx], y[sel_idx], H[ho_idx], y[ho_idx], n_layers, cv,
                                        variant=f"rand{rep}", conv=conv_primary)
        l_star_rand = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer_rand]))
        comp, gaps_rand = selection_specific_component(per_layer_rand, l_star_rand)
        rand_sel_components.append(comp)
        rand_gaps_at_selected.append(float(gaps_rand[l_star_rand]))
        rand_selected_layers.append(l_star_rand)
        rand_mean_gap_all_layers.append(float(gaps_rand.mean()))
        print(f"  rep {rep}: selected L{l_star_rand}, gap={gaps_rand[l_star_rand]:+.4f}  "
              f"selection-specific component={comp:+.4f}", flush=True)

    rand_sel_components = np.array(rand_sel_components)
    rand_mean_gap_arr = np.array(rand_mean_gap_all_layers)
    t_stat, t_p = ttest_1samp(rand_sel_components, 0.0)
    w_stat, w_p = wilcoxon(rand_sel_components)
    t_stat_gen, t_p_gen = ttest_1samp(rand_mean_gap_arr, 0.0)

    # ── (4) Regularization-robustness sweep (third correction, item (b)) ──
    # Does the ~zero general gap survive feature scaling and a five-decade C
    # sweep, or was the original (unscaled, C=1) probe simply pinned to its
    # prior? Same RNG seed => rep r here is the same partition as rep r above.
    print(f"\n=== (4) Regularization robustness: StandardScaler x C in {C_GRID}, "
          f"{N_REPS_ROBUST} reps each ===", flush=True)
    robustness = {}
    for C in C_GRID:
        rng_r = np.random.default_rng(RANDOM_STATE)
        conv_c = {"n_fits": 0, "max_n_iter": 0, "n_convergence_warnings": 0}
        comps, gens, layers_sel = [], [], []
        for rep in range(N_REPS_ROBUST):
            sel_idx, ho_idx = random_stratified_split(rng_r, y, N_TRAIN_SELECT)
            pl = per_layer_gaps(H[sel_idx], y[sel_idx], H[ho_idx], y[ho_idx], n_layers, cv,
                                variant=None, scale=True, C=C, conv=conv_c)
            ls = int(np.argmax([r["cv_auroc_selectpool"] for r in pl]))
            comp, gaps_c = selection_specific_component(pl, ls)
            comps.append(comp)
            gens.append(float(gaps_c.mean()))
            layers_sel.append(ls)
        comps = np.array(comps); gens = np.array(gens)
        tC, pC = ttest_1samp(comps, 0.0)
        robustness[str(C)] = {
            "scaled": True, "C": C, "n_reps": N_REPS_ROBUST,
            "selection_specific_component_mean": float(comps.mean()),
            "selection_specific_component_sd": float(comps.std(ddof=1)),
            "selection_specific_component_t": float(tC),
            "selection_specific_component_p": float(pC),
            "general_gap_mean": float(gens.mean()),
            "general_gap_sd": float(gens.std(ddof=1)),
            "selected_layer_per_rep": layers_sel,
            "convergence": conv_c,
        }
        print(f"  C={C:<6g} sel-specific={comps.mean():+.4f} (SD {comps.std(ddof=1):.4f}, "
              f"t={tC:+.2f}, p={pC:.4g})  general gap={gens.mean():+.4f} "
              f"(SD {gens.std(ddof=1):.4f})  max n_iter={conv_c['max_n_iter']} "
              f"conv_warns={conv_c['n_convergence_warnings']}/{conv_c['n_fits']}", flush=True)

    out = {
        "sanity_check_L11_sequential": {
            "cv": per_layer_seq[11]["cv_auroc_selectpool"], "heldout": per_layer_seq[11]["heldout_auroc"],
            "matches_published_0.804_0.616": abs(per_layer_seq[11]["cv_auroc_selectpool"] - 0.804) < 0.01
            and abs(per_layer_seq[11]["heldout_auroc"] - 0.616) < 0.01,
        },
        "original_sequential_split": {
            "per_layer": per_layer_seq,
            "l_star_by_cv": l_star_seq,
            "gap_at_L11": float(gaps_seq[11]),
            "mean_gap_all_32_layers": float(gaps_seq.mean()),
            "selection_specific_component": sel_component_seq,
            "note": "CONFOUNDED: sequential split, not a random partition. Two halves are separable by a hidden-state probe (AUROC 0.734-0.776), so this component conflates selection optimism with a population-difference artifact.",
        },
        "reversed_sequential_split": {
            "l_star_by_cv": l_star_rev,
            "gap_at_selected_layer": float(gaps_rev[l_star_rev]),
            "selection_specific_component": sel_component_rev,
            "note": "Diagnostic only, evaluated at THIS split's own argmax layer (not a hardcoded L11 -- an earlier version of this script had that bug). Sign flip vs. the original sequential split confirms the sequential-split number is protocol-confounded, not a stable selection-optimism estimate.",
        },
        "randomized_stratified_splits": {
            "n_reps": N_REPS,
            "selected_layer_per_rep": rand_selected_layers,
            "gap_at_selected_layer_per_rep": rand_gaps_at_selected,
            "mean_gap_all_32_layers_per_rep": rand_mean_gap_all_layers,
            "mean_gap_all_32_layers_mean": float(np.mean(rand_mean_gap_all_layers)),
            "mean_gap_all_32_layers_sd": float(np.std(rand_mean_gap_all_layers, ddof=1)),
            "selection_specific_component_per_rep": rand_sel_components.tolist(),
            "selection_specific_component_mean": float(rand_sel_components.mean()),
            "selection_specific_component_sd": float(rand_sel_components.std(ddof=1)),
            "selection_specific_component_t": float(t_stat),
            "selection_specific_component_t_p": float(t_p),
            "selection_specific_component_wilcoxon_p": float(w_p),
            "general_gap_t": float(t_stat_gen),
            "general_gap_t_p": float(t_p_gen),
            "convergence": conv_primary,
            "note": "CORRECTED PRIMARY RESULT: each rep's selection-specific component is now evaluated at THAT REP'S OWN argmax-selected layer (an earlier version of this script hardcoded L11 for every rep, which does not measure selection optimism -- it measures L11's own idiosyncrasy under each random split). Mean selection-specific component across randomized stratified splits, replacing the original sequential-split-derived +0.188 claim.",
            "independence_caveat": "All reps resample splits of the SAME 700 samples. The reported SD is split-to-split variability conditional on this one dataset, not sampling variability across independent draws from the population. The one-sample t-test's i.i.d. assumption is therefore violated; read t/p as a within-dataset robustness statistic, not a population-generalizing significance test. N_REPS was raised from 8 to 50 to make the SD estimate tight and to remove any suggestion that a small rep count was chosen after inspection; the stopping point is set by the size of the shipped replay artifact (~100 KB/rep), not by runtime.",
            "mechanical_positivity_caveat": "The selection-specific component is positive in expectation even under a pure-noise null, because l_star = argmax_l cv_l and gap_l = cv_l - ho_l share the +cv_l term. This is the winner's curse the metric is designed to measure, not a defect -- but its statistical significance is not evidence of anything beyond argmax-over-a-noisy-statistic.",
        },
        "regularization_robustness": {
            "grid": C_GRID,
            "n_reps": N_REPS_ROBUST,
            "by_C": robustness,
            "note": "Tests the alternative explanation that the near-zero general gap under randomized splits is an artifact of an unscaled, default-C, d=4096/n=400 LogisticRegression being pinned to its prior rather than genuinely fitting. Adds StandardScaler and sweeps C over five decades, on the same splits as the primary result (shared RNG seed). Convergence is measured (max n_iter_, ConvergenceWarning count), not assumed.",
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

    np.savez_compressed(ARTIFACT_PATH, n_layers=np.int32(n_layers),
                        n_reps=np.int32(N_REPS), **ARTIFACT)
    size_mb = ARTIFACT_PATH.stat().st_size / 1e6
    print(f"Saved derived artifact: {ARTIFACT_PATH} ({size_mb:.2f} MB) -- "
          f"replay with code/51_case_study_2_replay_from_artifact.py, no 171 MB "
          f"raw hidden-state cache required.")
    print(f"Sequential-split selection-specific component (at L{l_star_seq}): {sel_component_seq:+.4f} (CONFOUNDED)")
    print(f"Reversed-split selection-specific component (at L{l_star_rev}):    {sel_component_rev:+.4f} (sign-flip diagnostic)")
    print(f"Randomized-split selection-specific component (own-layer, per rep, N_REPS={N_REPS}): "
          f"{rand_sel_components.mean():+.4f} (SD {rand_sel_components.std(ddof=1):.4f}, "
          f"t={t_stat:+.2f}, p={t_p:.4g}; Wilcoxon p={w_p:.4g}) (CORRECTED)")
    print(f"Randomized-split general gap (mean over all {n_layers} layers): "
          f"{rand_mean_gap_arr.mean():+.4f} (SD {rand_mean_gap_arr.std(ddof=1):.4f}, "
          f"t={t_stat_gen:+.2f}, p={t_p_gen:.4g})")
    print(f"Primary-probe convergence: max n_iter_={conv_primary['max_n_iter']} of max_iter=1000, "
          f"{conv_primary['n_convergence_warnings']} ConvergenceWarnings over {conv_primary['n_fits']} fits")
    print(f"Randomized-split selected layers per rep: {rand_selected_layers}")
    print("NOTE: reps share one 700-sample dataset -- the t-test is a within-dataset "
          "robustness statistic, not a population-generalizing significance test.")


if __name__ == "__main__":
    main()
