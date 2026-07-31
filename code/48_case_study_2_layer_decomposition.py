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
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

GUARDIAN_ROOT = Path(os.path.expanduser(os.environ.get("GUARDIAN_ROOT", "~/Desktop/guardian")))
GUARDIAN_NPZ = GUARDIAN_ROOT / "results" / "hidden_states" / "mistral_7b_tqa_hidden_states.npz"
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "case_study_2_layer_decomposition.json"
# Small, shipped, derived artifact so a reader can re-derive every number in
# OUT_PATH without the 171 MB raw hidden-state cache (see code/51).
ARTIFACT_PATH = ROOT / "results" / "case_study_2_probe_scores.npz"

N_TRAIN_SELECT = 400
RANDOM_STATE = 42
N_REPS = 8

# Every probe score computed anywhere in this script is recorded here and
# written to ARTIFACT_PATH at the end, keyed by split-variant name.
ARTIFACT = {}


def per_layer_gaps(H_sel, y_sel, H_ho, y_ho, n_layers, cv, variant=None):
    """Per-layer CV-on-selection-pool vs. fit-on-pool/score-on-held-out AUROC.

    The CV number is the mean of the per-fold AUROCs (identical to what
    `cross_val_score(..., scoring='roc_auc')` returned in the previous version
    of this script, which is replaced here by an explicit fold loop only so the
    per-sample scores and fold assignments can be recorded for the shipped
    derived artifact -- the arithmetic is unchanged).
    """
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
            clf_f = LogisticRegression(max_iter=1000).fit(X_sel[tr], y_sel[tr])
            s = clf_f.predict_proba(X_sel[te])[:, 1]
            cv_scores[l, te] = s
            fold_aurocs.append(roc_auc_score(y_sel[te], s))
        cv_auroc = float(np.mean(fold_aurocs))
        clf = LogisticRegression(max_iter=1000).fit(X_sel, y_sel)
        s_ho = clf.predict_proba(X_ho)[:, 1]
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
    gaps = np.array([r["optimism_gap"] for r in per_layer])
    return float(gaps[l_star] - gaps.mean()), gaps


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
    for rep in range(N_REPS):
        idx = rng.permutation(n)
        # stratify manually: shuffle within each class then interleave proportional to overall rate
        pos_idx = idx[y[idx] == 1]
        neg_idx = idx[y[idx] == 0]
        n_pos_sel = int(round(N_TRAIN_SELECT * y.mean()))
        n_neg_sel = N_TRAIN_SELECT - n_pos_sel
        sel_idx = np.concatenate([pos_idx[:n_pos_sel], neg_idx[:n_neg_sel]])
        ho_idx = np.concatenate([pos_idx[n_pos_sel:], neg_idx[n_neg_sel:]])
        rng.shuffle(sel_idx); rng.shuffle(ho_idx)
        per_layer_rand = per_layer_gaps(H[sel_idx], y[sel_idx], H[ho_idx], y[ho_idx], n_layers, cv, variant=f"rand{rep}")
        l_star_rand = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer_rand]))
        comp, gaps_rand = selection_specific_component(per_layer_rand, l_star_rand)
        rand_sel_components.append(comp)
        rand_gaps_at_selected.append(float(gaps_rand[l_star_rand]))
        rand_selected_layers.append(l_star_rand)
        rand_mean_gap_all_layers.append(float(gaps_rand.mean()))
        print(f"  rep {rep}: selected L{l_star_rand}, gap={gaps_rand[l_star_rand]:+.4f}  "
              f"selection-specific component={comp:+.4f}", flush=True)

    rand_sel_components = np.array(rand_sel_components)
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
            "note": "CORRECTED PRIMARY RESULT: each rep's selection-specific component is now evaluated at THAT REP'S OWN argmax-selected layer (an earlier version of this script hardcoded L11 for every rep, which does not measure selection optimism -- it measures L11's own idiosyncrasy under each random split). Mean selection-specific component across randomized stratified splits, replacing the original sequential-split-derived +0.188 claim.",
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
    print(f"Randomized-split selection-specific component (own-layer, per rep): {rand_sel_components.mean():+.4f} (SD {rand_sel_components.std(ddof=1):.4f}) (CORRECTED)")
    print(f"Randomized-split selected layers per rep: {rand_selected_layers}")


if __name__ == "__main__":
    main()
