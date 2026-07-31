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
different populations, not exchangeable draws. Recomputing the
"selection-specific" component of the optimism gap
(gap_at_selected_layer - mean_gap_across_all_layers) under this sequential
split gives -0.0038 (no real selection optimism, an artifact of comparing
two non-exchangeable populations), while randomized stratified splits give
+0.0163 (SD 0.0120) -- the true effect, about 11.5x smaller than this
paper's original claimed +0.188. Reversing the sequential split (using
H[400:] to select and H[:400] as held-out) flips L11's gap sign entirely
(to -0.0344), which is itself strong evidence the sequential split's number
was measuring a population-difference artifact, not selection optimism.

This script now reports, and disclosure-labels, all three: (1) the original
sequential split (kept for continuity with the previously published
numbers, explicitly labeled as protocol-confounded), (2) N_REPS randomized
stratified splits (the corrected primary result), and (3) the reversed
sequential split (diagnostic showing the sign flip).
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

GUARDIAN_ROOT = Path(os.path.expanduser(os.environ.get("GUARDIAN_ROOT", "~/Desktop/guardian")))
GUARDIAN_NPZ = GUARDIAN_ROOT / "results" / "hidden_states" / "mistral_7b_tqa_hidden_states.npz"
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "case_study_2_layer_decomposition.json"

N_TRAIN_SELECT = 400
RANDOM_STATE = 42
N_REPS = 8


def per_layer_gaps(H_sel, y_sel, H_ho, y_ho, n_layers, cv):
    per_layer = []
    for l in range(n_layers):
        X_sel = H_sel[:, l, :]
        X_ho = H_ho[:, l, :]
        cv_auroc = float(cross_val_score(
            LogisticRegression(max_iter=1000), X_sel, y_sel, cv=cv, scoring="roc_auc"
        ).mean())
        clf = LogisticRegression(max_iter=1000).fit(X_sel, y_sel)
        heldout_auroc = float(roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1]))
        per_layer.append({
            "layer": l, "cv_auroc_selectpool": cv_auroc,
            "heldout_auroc": heldout_auroc, "optimism_gap": cv_auroc - heldout_auroc,
        })
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
    per_layer_seq = per_layer_gaps(H_sel, y_sel, H_ho, y_ho, n_layers, cv)
    for r in per_layer_seq:
        print(f"  L{r['layer']:02d}: CV={r['cv_auroc_selectpool']:.4f}  held-out={r['heldout_auroc']:.4f}  gap={r['optimism_gap']:+.4f}", flush=True)
    l_star_seq = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer_seq]))
    sel_component_seq, gaps_seq = selection_specific_component(per_layer_seq, 11)

    # (2) Reversed sequential split -- diagnostic for population-difference artifact.
    print("\n=== (2) REVERSED sequential split H[400:]=select / H[:400]=held-out ===", flush=True)
    per_layer_rev = per_layer_gaps(H_ho, y_ho, H_sel, y_sel, n_layers, cv)
    sel_component_rev, gaps_rev = selection_specific_component(per_layer_rev, 11)
    print(f"  L11 gap (reversed): {gaps_rev[11]:+.4f} | selection-specific component (reversed): {sel_component_rev:+.4f}", flush=True)

    # (3) Randomized stratified splits (N_REPS reps) -- corrected primary result.
    print(f"\n=== (3) {N_REPS} randomized stratified 400/{n-N_TRAIN_SELECT} splits (corrected) ===", flush=True)
    rand_sel_components = []
    rand_gaps_at_l11 = []
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
        per_layer_rand = per_layer_gaps(H[sel_idx], y[sel_idx], H[ho_idx], y[ho_idx], n_layers, cv)
        comp, gaps_rand = selection_specific_component(per_layer_rand, 11)
        rand_sel_components.append(comp)
        rand_gaps_at_l11.append(float(gaps_rand[11]))
        print(f"  rep {rep}: L11 gap={gaps_rand[11]:+.4f}  selection-specific component={comp:+.4f}", flush=True)

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
            "gap_at_L11": float(gaps_rev[11]),
            "selection_specific_component": sel_component_rev,
            "note": "Diagnostic only. Sign flip vs. the original sequential split confirms the sequential-split number is protocol-confounded, not a stable selection-optimism estimate.",
        },
        "randomized_stratified_splits": {
            "n_reps": N_REPS,
            "gap_at_L11_per_rep": rand_gaps_at_l11,
            "selection_specific_component_per_rep": rand_sel_components.tolist(),
            "selection_specific_component_mean": float(rand_sel_components.mean()),
            "selection_specific_component_sd": float(rand_sel_components.std(ddof=1)),
            "note": "CORRECTED PRIMARY RESULT: mean selection-specific component across randomized stratified splits, replacing the original sequential-split-derived +0.188 claim.",
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Sequential-split selection-specific component: {sel_component_seq:+.4f} (CONFOUNDED)")
    print(f"Reversed-split selection-specific component:    {sel_component_rev:+.4f} (sign-flip diagnostic)")
    print(f"Randomized-split selection-specific component:  {rand_sel_components.mean():+.4f} (SD {rand_sel_components.std(ddof=1):.4f}) (CORRECTED)")


if __name__ == "__main__":
    main()
