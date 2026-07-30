"""
Paper 2 -- Case Study 2 (GUARDIAN) full 32-layer decomposition.

An earlier version of this paper's Appendix A disclosed this as unavailable
("no per-layer data exists"). That was wrong: GUARDIAN's own sibling project
(`~/Desktop/guardian/results/hidden_states/mistral_7b_tqa_hidden_states.npz`)
contains the full (700, 32, 4096) raw Mistral-7B hidden states -- all 700
samples (the 400-sample CV-selection pool and the 300-sample held-out set),
at every layer, in full dimensionality. This reproduces the paper's already-
published L11 numbers exactly (CV=0.804, held-out=0.616) as a sanity check,
then extends the same computation to all 32 layers to show whether the
18.8-point selection-optimism gap is a property of L11 specifically or of
the CV-based argmax-selection procedure in general.
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


def main():
    d = np.load(GUARDIAN_NPZ)
    H, y = d["hidden_states"], d["labels"]
    valid = y >= 0
    H, y = H[valid], y[valid]
    n_layers = H.shape[1]
    print(f"Loaded {GUARDIAN_NPZ.name}: {H.shape}, n_valid={valid.sum()}, hall_rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    H_sel, y_sel = H[:N_TRAIN_SELECT], y[:N_TRAIN_SELECT]
    H_ho, y_ho = H[N_TRAIN_SELECT:], y[N_TRAIN_SELECT:]

    per_layer = []
    for l in range(n_layers):
        X_sel = H_sel[:, l, :]
        X_ho = H_ho[:, l, :]

        cv_auroc = float(cross_val_score(
            LogisticRegression(max_iter=1000), X_sel, y_sel, cv=cv, scoring="roc_auc"
        ).mean())

        clf = LogisticRegression(max_iter=1000).fit(X_sel, y_sel)
        heldout_auroc = float(roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1]))

        gap = cv_auroc - heldout_auroc
        per_layer.append({
            "layer": l, "cv_auroc_train400": cv_auroc,
            "heldout_auroc_300": heldout_auroc, "optimism_gap": gap,
        })
        print(f"  L{l:02d}: CV={cv_auroc:.4f}  held-out={heldout_auroc:.4f}  gap={gap:+.4f}", flush=True)

    l_star_cv = int(np.argmax([r["cv_auroc_train400"] for r in per_layer]))
    l_star_allfold = int(np.argmax([
        cross_val_score(LogisticRegression(max_iter=1000), H[:, l, :], y, cv=cv, scoring="roc_auc").mean()
        for l in range(n_layers)
    ]))

    gaps = np.array([r["optimism_gap"] for r in per_layer])
    out = {
        "sanity_check_L11": {"cv": per_layer[11]["cv_auroc_train400"], "heldout": per_layer[11]["heldout_auroc_300"],
                              "matches_published_0.804_0.616": abs(per_layer[11]["cv_auroc_train400"] - 0.804) < 0.01
                              and abs(per_layer[11]["heldout_auroc_300"] - 0.616) < 0.01},
        "per_layer": per_layer,
        "l_star_by_train_cv": l_star_cv,
        "l_star_by_all_data_cv": l_star_allfold,
        "gap_summary": {
            "mean": float(gaps.mean()), "sd": float(gaps.std(ddof=1)),
            "min": float(gaps.min()), "max": float(gaps.max()),
            "at_selected_layer_L11": float(gaps[11]),
            "n_layers_with_larger_gap_than_L11": int((gaps > gaps[11]).sum()),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Gap at selected L11: {gaps[11]:+.4f} | Mean gap all 32 layers: {gaps.mean():+.4f} (SD {gaps.std(ddof=1):.4f})")
    print(f"Layers with a LARGER optimism gap than L11: {int((gaps > gaps[11]).sum())}/32")


if __name__ == "__main__":
    main()
