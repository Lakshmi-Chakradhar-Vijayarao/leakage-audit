"""
Paper 2 -- Case Study 2 (GUARDIAN): re-derive every number in
`results/case_study_2_layer_decomposition.json` from the small, shipped
derived artifact `results/case_study_2_probe_scores.npz`, with NO access to
the 171 MB raw hidden-state cache.

WHY THIS EXISTS. `code/48` reads
`~/Desktop/guardian/results/hidden_states/mistral_7b_tqa_hidden_states.npz`
-- a 171 MB file that exists only on the authors' machine and is far too
large to ship with a submission. That made Case Study 2's decomposition
unreproducible for a reader in practice, even though the analysis itself is
simple. `code/48` now also writes, per split variant (the sequential split,
the reversed sequential split, and each of the 8 randomized stratified reps):

    <variant>__y_sel      (n_sel,)             selection-pool labels
    <variant>__y_ho       (n_ho,)              held-out labels
    <variant>__fold_id    (n_sel,)             StratifiedKFold assignment
    <variant>__cv_scores  (32, n_sel)          each sample's probe score from
                                               the fold-model that did NOT
                                               train on it
    <variant>__ho_scores  (32, n_ho)           held-out scores from the probe
                                               fit on the whole selection pool

Everything the paper reports for this case study is a function of those five
arrays, so this script recomputes all of it and asserts agreement with the
committed JSON. Running it is the reader-facing reproduction path.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "results" / "case_study_2_probe_scores.npz"
JSON_PATH = ROOT / "results" / "case_study_2_layer_decomposition.json"

# The artifact stores probe scores as float32 to keep it under 1 MB, so replay
# agrees with the committed JSON to ~1e-6 rather than bitwise. Every number the
# paper quotes is reported to 3-4 decimals, so this is three orders of magnitude
# tighter than any reported precision.
TOL = 1e-5


def per_layer_from_artifact(a, variant, n_layers):
    y_sel = a[f"{variant}__y_sel"]
    y_ho = a[f"{variant}__y_ho"]
    fold_id = a[f"{variant}__fold_id"]
    cv_scores = a[f"{variant}__cv_scores"]
    ho_scores = a[f"{variant}__ho_scores"]

    per_layer = []
    for l in range(n_layers):
        fold_aurocs = [
            roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
            for f in sorted(set(fold_id.tolist()))
        ]
        cv_auroc = float(np.mean(fold_aurocs))
        heldout_auroc = float(roc_auc_score(y_ho, ho_scores[l]))
        per_layer.append({
            "layer": l, "cv_auroc_selectpool": cv_auroc,
            "heldout_auroc": heldout_auroc, "optimism_gap": cv_auroc - heldout_auroc,
        })
    return per_layer


def selection_specific_component(per_layer):
    gaps = np.array([r["optimism_gap"] for r in per_layer])
    l_star = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer]))
    return l_star, float(gaps[l_star] - gaps.mean()), gaps


def check(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {name}: replay={got:+.6f}  committed={want:+.6f}")
    return ok


def main():
    a = np.load(ARTIFACT_PATH)
    n_layers = int(a["n_layers"])
    n_reps = int(a["n_reps"])
    ref = json.load(open(JSON_PATH))
    print(f"Replaying Case Study 2 from {ARTIFACT_PATH.name} "
          f"({ARTIFACT_PATH.stat().st_size / 1e6:.2f} MB), {n_layers} layers, {n_reps} reps\n")

    all_ok = True

    # (1) sequential split
    pl_seq = per_layer_from_artifact(a, "sequential", n_layers)
    l_seq, comp_seq, gaps_seq = selection_specific_component(pl_seq)
    print("(1) sequential split:")
    all_ok &= check("L11 CV AUROC", pl_seq[11]["cv_auroc_selectpool"],
                    ref["sanity_check_L11_sequential"]["cv"])
    all_ok &= check("L11 held-out AUROC", pl_seq[11]["heldout_auroc"],
                    ref["sanity_check_L11_sequential"]["heldout"])
    all_ok &= check("mean gap over all layers", float(gaps_seq.mean()),
                    ref["original_sequential_split"]["mean_gap_all_32_layers"])
    all_ok &= check("selection-specific component", comp_seq,
                    ref["original_sequential_split"]["selection_specific_component"])
    print(f"       selected layer: replay=L{l_seq}  committed=L{ref['original_sequential_split']['l_star_by_cv']}")
    all_ok &= l_seq == ref["original_sequential_split"]["l_star_by_cv"]

    # (2) reversed sequential split
    pl_rev = per_layer_from_artifact(a, "reversed", n_layers)
    l_rev, comp_rev, _ = selection_specific_component(pl_rev)
    print("\n(2) reversed sequential split:")
    all_ok &= check("selection-specific component", comp_rev,
                    ref["reversed_sequential_split"]["selection_specific_component"])
    print(f"       selected layer: replay=L{l_rev}  committed=L{ref['reversed_sequential_split']['l_star_by_cv']}")
    all_ok &= l_rev == ref["reversed_sequential_split"]["l_star_by_cv"]

    # (3) randomized stratified reps
    print(f"\n(3) {n_reps} randomized stratified splits:")
    comps, gaps_at_sel, layers_sel, mean_gaps = [], [], [], []
    for rep in range(n_reps):
        pl = per_layer_from_artifact(a, f"rand{rep}", n_layers)
        l_star, comp, gaps = selection_specific_component(pl)
        comps.append(comp)
        gaps_at_sel.append(float(gaps[l_star]))
        layers_sel.append(l_star)
        mean_gaps.append(float(gaps.mean()))
    r = ref["randomized_stratified_splits"]
    print(f"       selected layers: replay={layers_sel}  committed={r['selected_layer_per_rep']}")
    all_ok &= layers_sel == list(r["selected_layer_per_rep"])
    all_ok &= check("selection-specific component (mean)", float(np.mean(comps)),
                    r["selection_specific_component_mean"])
    all_ok &= check("selection-specific component (SD)", float(np.std(comps, ddof=1)),
                    r["selection_specific_component_sd"])
    all_ok &= check("general gap across all layers (mean)", float(np.mean(mean_gaps)),
                    r["mean_gap_all_32_layers_mean"])
    all_ok &= check("general gap across all layers (SD)", float(np.std(mean_gaps, ddof=1)),
                    r["mean_gap_all_32_layers_sd"])

    print("\n" + ("ALL CHECKS PASSED -- every Case Study 2 number in the paper is "
                  "reproducible from the shipped derived artifact alone."
                  if all_ok else "MISMATCH -- see FAIL lines above."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
