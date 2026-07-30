"""
Paper 2 -- quantifies a fifth label-leakage mechanism a fresh audit found
sitting in the very file this paper already audits for Mechanism 3:
`MultiHaluDet/run_pipeline.py::stage_4_ensemble` calls
`find_best_thresholds(probs, y_test)` (src/utils/metrics.py), which
sweeps 81 candidate F1 thresholds and the ROC-optimal (Youden) threshold
directly against the TEST labels, then reports every threshold-dependent
metric (F1, accuracy, MCC, Cohen's kappa, balanced accuracy, and the ECE
computed from those predictions) at that test-label-selected threshold.
AUROC is threshold-free and unaffected, but every other headline number
in that pipeline is test-set-optimized operating-point selection --
a structurally distinct mechanism from Mechanism 3 (checkpoint/fold
selection): here the choice is which decision threshold to report at,
not which model/checkpoint to keep.

This uses the SAME isotropic-Gaussian calibration this paper's severity
harness already validates (AUROC=0.80 Bayes-optimal target, binormal
identity) and a simple logistic-regression classifier (deliberately not
the full SweepMLP+meta-learner machinery from code/02d -- threshold
selection leakage does not depend on which classifier produced the
probabilities, and a simple classifier keeps this demonstration fast and
self-contained) to compare two protocols on the same test-set
predictions:
  LEAKY:  threshold selected by find_best_thresholds(probs_test, y_test)
          (identical to MultiHaluDet's own code), then F1/accuracy
          computed at that threshold on y_test.
  HONEST: threshold selected on an independent, held-out validation
          split (never touching the test labels), then applied as-is
          to the test set.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "mechanism5_threshold_selection.json"

N_SAMPLES = 700
TEST_SIZE = 0.20
VAL_SIZE_OF_TRAIN = 0.20
N_SEEDS = 200
TARGET_AUROCS = [0.70, 0.80, 0.90, 0.95, 0.985]
FEAT_DIM = 64
RNG_GLOBAL = np.random.default_rng(2026)


def make_synthetic_data(seed, target_auroc):
    rng = np.random.default_rng(seed)
    j_target = 2 * (norm.ppf(target_auroc)) ** 2
    class_sep = np.sqrt(j_target / FEAT_DIM)
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    X_pos = class_sep + rng.standard_normal((n_pos, FEAT_DIM))
    X_neg = -class_sep + rng.standard_normal((n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float64)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def find_best_thresholds(probs, labels):
    """Verbatim port of MultiHaluDet/src/utils/metrics.py::find_best_thresholds."""
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    youden_j = tpr - fpr
    best_threshold_youden = thresholds_roc[np.argmax(youden_j)]
    best_f1, best_threshold_f1 = 0, 0.5
    for thresh in np.linspace(0.1, 0.9, 81):
        preds_temp = (probs >= thresh).astype(int)
        f1_temp = f1_score(labels, preds_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1, best_threshold_f1 = f1_temp, thresh
    return {"youden": best_threshold_youden, "f1": best_threshold_f1}


def run_one_seed(seed, target_auroc):
    X, y = make_synthetic_data(seed, target_auroc)
    n = len(y)
    n_test = int(n * TEST_SIZE)
    n_val = int((n - n_test) * VAL_SIZE_OF_TRAIN)
    rng = np.random.default_rng(seed + 50000)
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    clf = LogisticRegression(max_iter=2000).fit(X[train_idx], y[train_idx])
    probs_val = clf.predict_proba(X[val_idx])[:, 1]
    probs_test = clf.predict_proba(X[test_idx])[:, 1]

    # LEAKY: threshold selected ON the test labels themselves
    leaky_thresh = find_best_thresholds(probs_test, y[test_idx])
    leaky_f1 = f1_score(y[test_idx], (probs_test >= leaky_thresh["f1"]).astype(int), zero_division=0)
    leaky_acc = accuracy_score(y[test_idx], (probs_test >= leaky_thresh["youden"]).astype(int))

    # HONEST: threshold selected on an independent validation split, applied as-is
    honest_thresh = find_best_thresholds(probs_val, y[val_idx])
    honest_f1 = f1_score(y[test_idx], (probs_test >= honest_thresh["f1"]).astype(int), zero_division=0)
    honest_acc = accuracy_score(y[test_idx], (probs_test >= honest_thresh["youden"]).astype(int))

    return {"leaky_f1": leaky_f1, "honest_f1": honest_f1, "leaky_acc": leaky_acc, "honest_acc": honest_acc}


def bca_ci(a, b, n_resamples=10000):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples,
                     method="BCa", random_state=RNG_GLOBAL)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    out = {"capacities": {}}
    for target_auroc in TARGET_AUROCS:
        leaky_f1s, honest_f1s, leaky_accs, honest_accs = [], [], [], []
        for seed in range(N_SEEDS):
            r = run_one_seed(seed, target_auroc)
            leaky_f1s.append(r["leaky_f1"]); honest_f1s.append(r["honest_f1"])
            leaky_accs.append(r["leaky_acc"]); honest_accs.append(r["honest_acc"])

        gap_f1 = np.array(leaky_f1s) - np.array(honest_f1s)
        gap_acc = np.array(leaky_accs) - np.array(honest_accs)
        _, p_f1 = wilcoxon(leaky_f1s, honest_f1s)
        _, p_acc = wilcoxon(leaky_accs, honest_accs)
        ci_f1 = bca_ci(leaky_f1s, honest_f1s)
        ci_acc = bca_ci(leaky_accs, honest_accs)

        print(f"AUROC target={target_auroc}: "
              f"F1 leaky={np.mean(leaky_f1s):.4f} honest={np.mean(honest_f1s):.4f} "
              f"gap={gap_f1.mean():+.4f} CI={ci_f1} p={p_f1:.4g} | "
              f"Acc gap={gap_acc.mean():+.4f} CI={ci_acc} p={p_acc:.4g}", flush=True)

        out["capacities"][str(target_auroc)] = {
            "leaky_f1_mean": float(np.mean(leaky_f1s)), "honest_f1_mean": float(np.mean(honest_f1s)),
            "f1_gap_mean": float(gap_f1.mean()), "f1_gap_bca_ci_95": ci_f1, "f1_gap_wilcoxon_p": float(p_f1),
            "leaky_acc_mean": float(np.mean(leaky_accs)), "honest_acc_mean": float(np.mean(honest_accs)),
            "acc_gap_mean": float(gap_acc.mean()), "acc_gap_bca_ci_95": ci_acc, "acc_gap_wilcoxon_p": float(p_acc),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
