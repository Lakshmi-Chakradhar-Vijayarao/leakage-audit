"""
Paper 2 -- the discriminating experiment for the isotropic-vs-anisotropic
sweep comparison: hold the feature dimension FIXED at d=64 and vary ONLY the
within-class covariance eigenspectrum.

WHY THIS EXISTS. §4.3 compares two generative processes for Case Study 3:
code/02d (isotropic, d=64) and code/27 (real anisotropic covariance, d=414).
Their capacity-by-capacity significance patterns reshuffle rather than
replicating, and the paper already discloses that TWO things differ between
them at once -- covariance shape AND dimensionality (a 6.5x change in the
dimension-to-sample ratio). With both moving, "the covariance shape changed
the answer" and "the dimensionality changed the answer" are not separable.
An independent adversarial review named the clean experiment that separates
them: fix d and sweep only the eigenspectrum. That is this script.

DESIGN. d = 64 throughout, matching code/02d exactly, so the only thing that
changes across cells is the shape of the within-class covariance:

    Sigma(beta) = Q diag(lambda) Q^T,     lambda_i proportional to i^(-beta)

with a single FIXED random orthonormal Q shared by every cell (so the
eigenBASIS is held constant too -- only the eigenVALUE profile moves), and
lambda renormalized so trace(Sigma) = d in every cell. Fixing the trace means
total within-class variance is identical everywhere; only its distribution
across directions differs. beta = 0 reproduces the isotropic case exactly and
is the built-in control.

Cells: beta in {0.0, 0.5, 1.0, 2.0} plus `real_top64`, whose eigenvalue
profile is the top-64 eigenvalues of the real Mistral-7B/HaluEval pooled
within-class covariance code/27 uses, renormalized to trace 64 -- i.e. the
real spectrum's SHAPE transplanted into the isotropic sweep's dimensionality.

Mean-difference direction: a single fixed unit direction, identical across
cells, with equal components in the eigenbasis Q. Equal weighting is the
neutral choice: it does not preferentially align delta_mu with either the
high- or the low-variance directions of any particular spectrum, which a
random direction would do differently for each beta. Its length is then set
per cell so the Mahalanobis Fisher ratio

    J = delta_mu^T Sigma^-1 delta_mu

equals 2*Phi^-1(0.80)^2 exactly, i.e. every cell is calibrated to the same
Bayes-optimal AUROC = 0.80 the rest of this paper's synthetic sweeps target.

CALIBRATION CHECK. This script deliberately does NOT call
code/sanity_checks.py's assert_calibration(): that guardrail's bias
correction assumes identity within-class covariance and is not directly
applicable to an anisotropic generator (see its module docstring). Because
Sigma here is CONSTRUCTED rather than estimated, the correct check is
available exactly and is asserted directly instead: the realized Mahalanobis
J of the population parameters is verified against J_target to 1e-9, and the
realized J of each drawn sample is recorded.

Everything else -- SweepMLP, train_to_best_checkpoint, train_fixed_epochs,
the LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO logic, N_SEEDS, EPOCHS -- is an exact
port of code/02d, so any difference across cells is attributable to the
eigenspectrum alone.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

_SPEC = importlib.util.spec_from_file_location(
    "sweep02d", Path(__file__).resolve().parent / "02d_corrected_capacity_placebo_sweep.py")
_M = importlib.util.module_from_spec(_SPEC)
sys.modules["sweep02d"] = _M
_SPEC.loader.exec_module(_M)

train_to_best_checkpoint = _M.train_to_best_checkpoint
train_fixed_epochs = _M.train_fixed_epochs
extract_features = _M.extract_features

ROOT = Path(__file__).resolve().parent.parent
REAL_FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "eigenspectrum_sweep_fixed_dim.json"

FEAT_DIM = 64            # FIXED -- the whole point of this script
N_SAMPLES = 700
TARGET_AUROC = 0.80
TEST_SIZE = 0.20
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
N_INNER_FOLDS = 5
N_SEEDS = int(os.environ.get("N_SEEDS_OVERRIDE", 100))
CAPACITIES = [int(c) for c in os.environ.get("CAPACITIES", "128").split(",")]
BETAS = [0.0, 0.5, 1.0, 2.0]
J_TARGET = 2 * (norm.ppf(TARGET_AUROC)) ** 2
CONDITIONS = ["leaky", "clean", "clean_matched", "placebo"]
RNG_GLOBAL = np.random.default_rng(2026)

# One fixed orthonormal eigenbasis, shared by every cell.
_q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((FEAT_DIM, FEAT_DIM)))
Q = _q


def real_top64_spectrum():
    """Top-64 eigenvalues of the real pooled within-class covariance code/27
    uses -- the real spectrum's shape, transplanted to d=64."""
    d = np.load(REAL_FEATS_PATH)
    X = np.hstack([d["X_seq"].reshape(d["X_seq"].shape[0], -1), d["X_glob"]]).astype(np.float64)
    y = d["y"]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    Xc = np.vstack([X[y == 1] - X[y == 1].mean(axis=0), X[y == 0] - X[y == 0].mean(axis=0)])
    ev = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1][:FEAT_DIM]
    return np.clip(ev, 1e-6, None)


def build_cell(spectrum_name, lam_raw):
    """Sigma with the given eigenvalue profile (trace normalized to d), and a
    delta_mu along the fixed equal-weight-in-eigenbasis direction, scaled so
    the Mahalanobis J hits J_TARGET exactly."""
    lam = np.asarray(lam_raw, dtype=np.float64)
    lam = lam * (FEAT_DIM / lam.sum())              # trace(Sigma) = d in every cell
    chol = Q @ np.diag(np.sqrt(lam))                # Sigma = chol @ chol.T
    u_eig = np.ones(FEAT_DIM) / np.sqrt(FEAT_DIM)   # equal weight in the eigenbasis
    direction = Q @ u_eig
    # J for delta = s*direction is s^2 * sum(u_eig^2 / lam); solve for s.
    j_unit = float(np.sum(u_eig ** 2 / lam))
    s = float(np.sqrt(J_TARGET / j_unit))
    delta_mu = s * direction
    # Exact calibration check (population, not a plug-in estimate).
    sigma_inv = Q @ np.diag(1.0 / lam) @ Q.T
    j_realized = float(delta_mu @ sigma_inv @ delta_mu)
    assert abs(j_realized - J_TARGET) < 1e-9, (spectrum_name, j_realized, J_TARGET)
    return {
        "name": spectrum_name, "delta_mu": delta_mu, "chol": chol,
        "eigenvalues": lam, "j_realized": j_realized,
        "condition_number": float(lam.max() / lam.min()),
        "participation_ratio": float(lam.sum() ** 2 / np.sum(lam ** 2)),
    }


def make_synthetic_data(cell, seed):
    rng = np.random.default_rng(seed)
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    X_pos = cell["delta_mu"] / 2 + rng.standard_normal((n_pos, FEAT_DIM)) @ cell["chol"].T
    X_neg = -cell["delta_mu"] / 2 + rng.standard_normal((n_neg, FEAT_DIM)) @ cell["chol"].T
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def run_one_seed(cell, seed, hidden):
    X, y = make_synthetic_data(cell, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(seed + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = hidden // 2
    oof = {k: np.zeros((len(y_train), feat_dim_out)) for k in CONDITIONS}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in CONDITIONS}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold
        m, _ = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx],
                                        X_train[val_idx], y_train[val_idx], hidden, EPOCHS, fold_seed)
        oof["leaky"][val_idx] = extract_features(m, X_train[val_idx])
        test_feat["leaky"] += extract_features(m, X_test)

        tr2_idx, es_idx = train_test_split(tr_idx, test_size=ES_HOLD_FRACTION,
                                           stratify=y_train[tr_idx], random_state=fold_seed)
        m, best_epoch = train_to_best_checkpoint(X_train[tr2_idx], y_train[tr2_idx],
                                                 X_train[es_idx], y_train[es_idx], hidden, EPOCHS, fold_seed)
        oof["clean"][val_idx] = extract_features(m, X_train[val_idx])
        test_feat["clean"] += extract_features(m, X_test)

        m = train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed)
        oof["clean_matched"][val_idx] = extract_features(m, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(m, X_test)

        m, _ = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx], X_train[val_idx],
                                        rng.permutation(y_train[val_idx]), hidden, EPOCHS, fold_seed)
        oof["placebo"][val_idx] = extract_features(m, X_train[val_idx])
        test_feat["placebo"] += extract_features(m, X_test)

    aucs = {}
    for k in oof:
        test_feat[k] /= N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    return aucs


def gap_stats(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gap = a - b
    _, p = wilcoxon(a, b)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=RNG_GLOBAL)
    return {"mean": float(gap.mean()), "wilcoxon_p": float(p),
            "bca_ci_95": [float(res.confidence_interval.low),
                          float(res.confidence_interval.high)]}


def main():
    t0 = time.time()
    cells = [build_cell(f"powerlaw_beta{b}", (np.arange(1, FEAT_DIM + 1.0)) ** (-b)) for b in BETAS]
    cells.append(build_cell("real_top64", real_top64_spectrum()))

    print(f"d={FEAT_DIM} FIXED; {len(cells)} eigenspectra; capacities={CAPACITIES}; "
          f"N_SEEDS={N_SEEDS}; all calibrated to Mahalanobis J={J_TARGET:.4f} "
          f"(AUROC={TARGET_AUROC})\n")
    for c in cells:
        print(f"  {c['name']:20s} cond={c['condition_number']:10.2f}  "
              f"participation_ratio={c['participation_ratio']:6.2f}  J={c['j_realized']:.6f}")

    out = {"feat_dim": FEAT_DIM, "n_samples": N_SAMPLES, "n_seeds": N_SEEDS,
           "target_auroc": TARGET_AUROC, "j_target": J_TARGET,
           "capacities": CAPACITIES, "by_capacity": {}}

    for hidden in CAPACITIES:
        out["by_capacity"][str(hidden)] = {}
        for c in cells:
            acc = {k: [] for k in CONDITIONS}
            for seed in range(N_SEEDS):
                a = run_one_seed(c, seed, hidden)
                for k in CONDITIONS:
                    acc[k].append(a[k])
            cell_out = {
                "condition_number": c["condition_number"],
                "participation_ratio": c["participation_ratio"],
                "j_realized_population": c["j_realized"],
                "means": {k: float(np.mean(v)) for k, v in acc.items()},
                "leaky_minus_clean_matched": gap_stats(acc["leaky"], acc["clean_matched"]),
                "clean_matched_minus_placebo": gap_stats(acc["clean_matched"], acc["placebo"]),
            }
            out["by_capacity"][str(hidden)][c["name"]] = cell_out
            g = cell_out["leaky_minus_clean_matched"]
            print(f"  [cap {hidden}] {c['name']:20s} LEAKY-CLEAN_MATCHED={g['mean']:+.4f} "
                  f"CI=[{g['bca_ci_95'][0]:+.4f},{g['bca_ci_95'][1]:+.4f}] p={g['wilcoxon_p']:.4g} "
                  f"LEAKY_mean={cell_out['means']['leaky']:.4f}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    out["note"] = (
        "Discriminating experiment for the isotropic (code/02d, d=64) vs. anisotropic "
        "(code/27, d=414) comparison, which confounds covariance shape with dimensionality. "
        "Here d=64 is fixed and only the within-class eigenvalue profile varies, with the "
        "eigenbasis, the total within-class variance (trace), the mean-difference direction "
        "and the calibrated Mahalanobis J all held constant. beta=0 is the isotropic control; "
        "real_top64 transplants the real feature covariance's spectral shape into d=64."
    )
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
