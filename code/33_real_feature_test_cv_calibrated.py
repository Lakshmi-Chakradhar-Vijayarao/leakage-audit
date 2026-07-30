"""
Paper 2 -- fix a CRITICAL flaw in code/31's real-feature calibration a fresh
review found (independently re-verified by permutation test before acting):
the analytic alpha-calibration formula (binormal AUROC identity, J = delta_mu^T
Sigma_inv delta_mu, alpha = sqrt(J_target/J_real)) is badly biased at this
data's actual dimensionality regime (n=400 samples, d=414 features, i.e.
d approx equal to n). Confirmed directly: permuting the labels and
recomputing J with the identical plug-in formula gives J_perm ~= 2.3-2.5,
which ALREADY EXCEEDS J_target=1.42 (the value defined to correspond to
AUROC=0.80) -- meaning pure label noise, with the true label-feature
relationship completely destroyed, registers as "more separated" by this
estimator than the paper's own calibration target. The plug-in Mahalanobis
distance is inverting a near-singular n~=d covariance matrix, and this
inflates the quadratic form on noise alone. code/31's alpha=0.1455 was
computed from an estimate (J_real=66.93) that is contaminated by exactly
this same noise-driven inflation, so its claim of hitting AUROC=0.80 was
not trustworthy.

FIX: replace the analytic formula with an EMPIRICAL, cross-validated
calibration. For a candidate alpha, apply the identical affine class-mean
rescaling code/31 uses (real per-sample deviations from each sample's own
class mean are left completely untouched -- only the class-mean offset is
shrunk toward the midpoint), then measure the ACTUALLY ACHIEVED AUROC via
5-fold cross-validated, strongly-regularized (C=0.01) logistic regression
-- out-of-fold, so a sample's held-out prediction never uses that same
sample's contribution to the fold's fitted decision boundary. Verified this
approach is NOT subject to the same bias: under label permutation, this
CV-AUROC estimator gives 0.504 +/- 0.043 (10 permutations) -- correctly
centered at chance, unlike the analytic formula's 0.86. Binary-search alpha
so the measured (not theoretical) CV AUROC hits the target 0.80 within
tolerance.

Everything downstream -- SweepMLP, train_to_best_checkpoint, the 5-fold OOF
+ meta-learner severity-comparison architecture, capacities, N_SEEDS -- is
an exact, unmodified port of code/31/code/25. Only the calibration mechanism
changes.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "real_feature_test_cv_calibrated.json"

CAPACITIES = [128, 384]
N_SEEDS = 100
N_INNER_FOLDS = 5
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
TARGET_AUROC = 0.80
CALIB_C = 0.01
CALIB_SEEDS = 5
CALIB_TOL = 0.005
CALIB_MAX_ITER = 20


class SweepMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def features(self, x):
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[3](h)
        h = self.net[4](h)
        return h


def load_raw_real_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


def apply_calibration(X, y, alpha, mu_pos, mu_neg, midpoint):
    X_calibrated = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        deviation = X[mask] - mu_cls  # real, unmodified per-sample noise
        new_class_mean = midpoint + alpha * (mu_cls - midpoint)
        X_calibrated[mask] = new_class_mean + deviation
    return X_calibrated


def measure_cv_auroc(X_calibrated, y, n_seeds=CALIB_SEEDS, C=CALIB_C):
    aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs = cross_val_predict(
            LogisticRegression(max_iter=5000, C=C), X_calibrated, y, cv=skf, method="predict_proba"
        )[:, 1]
        aucs.append(roc_auc_score(y, probs))
    return float(np.mean(aucs)), float(np.std(aucs))


def calibrate_alpha_empirically(X, y):
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2

    lo, hi = 0.0, 1.0
    history = []
    for it in range(CALIB_MAX_ITER):
        mid = (lo + hi) / 2
        Xc = apply_calibration(X, y, mid, mu_pos, mu_neg, midpoint)
        auc_mean, auc_std = measure_cv_auroc(Xc, y)
        history.append({"alpha": mid, "cv_auroc_mean": auc_mean, "cv_auroc_std": auc_std})
        print(f"  [calib iter {it}] alpha={mid:.5f} -> CV AUROC={auc_mean:.4f} +/- {auc_std:.4f}", flush=True)
        if abs(auc_mean - TARGET_AUROC) < CALIB_TOL:
            break
        if auc_mean < TARGET_AUROC:
            lo = mid
        else:
            hi = mid
    else:
        mid = (lo + hi) / 2

    return mid, mu_pos, mu_neg, midpoint, history


def sanity_check_permutation_null(X, y, alpha, mu_pos, mu_neg, midpoint, n_perms=10):
    """Confirm the CV-based calibration measurement is NOT subject to the
    same noise-driven inflation as the analytic plug-in formula: under label
    permutation, CV AUROC at the SAME alpha should sit near chance."""
    rng = np.random.default_rng(2026)
    perm_aucs = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        mu_pos_p = X[y_perm == 1].mean(axis=0)
        mu_neg_p = X[y_perm == 0].mean(axis=0)
        mid_p = (mu_pos_p + mu_neg_p) / 2
        Xc_perm = apply_calibration(X, y_perm, alpha, mu_pos_p, mu_neg_p, mid_p)
        auc, _ = measure_cv_auroc(Xc_perm, y_perm, n_seeds=1)
        perm_aucs.append(auc)
    return float(np.mean(perm_aucs)), float(np.std(perm_aucs))


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel_for_selection, hidden, epochs, seed):
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state, best_epoch = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xs)).numpy()
        try:
            auc = roc_auc_score(y_sel_for_selection, probs)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed_epochs(X_tr, y_tr, hidden, n_epochs, seed):
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    n_epochs = max(n_epochs, 1)
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        return model.features(torch.tensor(X, dtype=torch.float32)).numpy()


def run_one_seed(X, y, hidden, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(seed + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    conditions = ["leaky", "clean", "clean_matched", "placebo"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        model_leaky, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["leaky"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky"] += extract_features(model_leaky, X_test)

        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean, best_epoch = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["clean"][val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat["clean"] += extract_features(model_clean, X_test)

        model_clean_matched = train_fixed_epochs(
            X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed,
        )
        oof["clean_matched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(model_clean_matched, X_test)

        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted,
            hidden, EPOCHS, fold_seed,
        )
        oof["placebo"][val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat["placebo"] += extract_features(model_placebo, X_test)

    aucs = {}
    for k in oof:
        test_feat[k] /= N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    return aucs


def decision_rule(gap_lp_mean, gap_lp_p, gap_cmp_mean):
    if gap_lp_p < 0.05 and gap_lp_mean > gap_cmp_mean * 1.5:
        return "GENUINE_LEAK_CONFIRMED"
    elif abs(gap_lp_mean - gap_cmp_mean) < 0.001 and gap_lp_p > 0.05:
        return "CONFOUND_CONFIRMED_NO_REAL_LEAK"
    else:
        return "MIXED"


def main():
    t0 = time.time()
    X, y = load_raw_real_features()

    print("=== Empirical (CV-based) alpha calibration ===", flush=True)
    alpha, mu_pos, mu_neg, midpoint, calib_history = calibrate_alpha_empirically(X, y)
    final_auc_mean, final_auc_std = measure_cv_auroc(
        apply_calibration(X, y, alpha, mu_pos, mu_neg, midpoint), y, n_seeds=10
    )
    print(f"Final alpha={alpha:.5f}, CV AUROC (10-seed re-check)={final_auc_mean:.4f} +/- {final_auc_std:.4f}", flush=True)

    print("\n=== Permutation-null sanity check at this alpha ===", flush=True)
    perm_mean, perm_std = sanity_check_permutation_null(X, y, alpha, mu_pos, mu_neg, midpoint)
    print(f"Permuted-label CV AUROC at alpha={alpha:.5f}: {perm_mean:.4f} +/- {perm_std:.4f} (expect ~0.50)", flush=True)

    X_calibrated = apply_calibration(X, y, alpha, mu_pos, mu_neg, midpoint).astype(np.float32)
    print(f"\nCalibrated real feature matrix: {X_calibrated.shape}, hall_rate={1-y.mean():.3f}", flush=True)

    out = {
        "calibration_alpha": float(alpha),
        "calibration_method": "empirical_cv_logreg_C0.01_bisection",
        "calibration_achieved_cv_auroc": {"mean": final_auc_mean, "std": final_auc_std},
        "calibration_permutation_null_check": {"mean": perm_mean, "std": perm_std},
        "calibration_search_history": calib_history,
        "capacities": {},
    }
    for hidden in CAPACITIES:
        print(f"\n{'='*60}\nCapacity {hidden}, N_SEEDS={N_SEEDS}, CV-calibrated real features (alpha={alpha:.4f})\n{'='*60}", flush=True)
        all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
        for seed in range(N_SEEDS):
            aucs = run_one_seed(X_calibrated, y, hidden, seed)
            for k, v in aucs.items():
                all_aucs[k].append(v)
            if (seed + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{seed+1}/{N_SEEDS}] elapsed={elapsed:.0f}s "
                      f"leaky={np.mean(all_aucs['leaky']):.4f} clean_matched={np.mean(all_aucs['clean_matched']):.4f} "
                      f"placebo={np.mean(all_aucs['placebo']):.4f}", flush=True)

        arrs = {k: np.array(v) for k, v in all_aucs.items()}
        gap_lcm = arrs["leaky"] - arrs["clean_matched"]
        gap_cmp = arrs["clean_matched"] - arrs["placebo"]
        gap_lp = arrs["leaky"] - arrs["placebo"]
        _, p_lcm = wilcoxon(arrs["leaky"], arrs["clean_matched"])
        _, p_cmp = wilcoxon(arrs["clean_matched"], arrs["placebo"])
        _, p_lp = wilcoxon(arrs["leaky"], arrs["placebo"])
        verdict = decision_rule(gap_lp.mean(), p_lp, gap_cmp.mean())

        print(f"\n--- Capacity {hidden} summary (CV-calibrated real features) ---")
        for k, v in arrs.items():
            print(f"  {k:15s} AUROC: mean={v.mean():.4f} std={v.std():.4f}")
        print(f"  leaky - clean_matched: mean={gap_lcm.mean():+.4f} p={p_lcm:.4g}")
        print(f"  clean_matched - placebo: mean={gap_cmp.mean():+.4f} p={p_cmp:.4g}")
        print(f"  leaky - placebo: mean={gap_lp.mean():+.4f} p={p_lp:.4g}")
        print(f"  DECISION RULE VERDICT: {verdict}")

        out["capacities"][str(hidden)] = {
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "leaky_minus_clean_matched": {"mean": float(gap_lcm.mean()), "wilcoxon_p": float(p_lcm)},
            "clean_matched_minus_placebo": {"mean": float(gap_cmp.mean()), "wilcoxon_p": float(p_cmp)},
            "leaky_minus_placebo": {"mean": float(gap_lp.mean()), "wilcoxon_p": float(p_lp)},
            "decision_rule_verdict": verdict,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
