"""
Paper 2 -- fixes the CRITICAL flaw a fresh, independent review found in
code/33's real-feature calibration: `apply_calibration` estimates
mu_pos/mu_neg/midpoint from ALL 400 samples (train and test combined)
before any split. Because each class's per-sample deviations sum to
exactly zero over the full sample used to estimate the class means, any
subsequent train/test split forces the two halves' leftover class-mean
directions into an algebraic identity: n_tr*mean_tr = -n_te*mean_te.
This was independently verified: at alpha=0 (classes collapsed to an
identical mean), cos(delta_mu_train, delta_mu_test) = -0.9999 and a
plain linear model fit on train and scored on test gives AUROC=0.06 --
not "zero separation," but near-perfect, mechanically-induced
ANTI-correlation. code/33's own downstream 0.9535 AUROC at alpha=0 (and
very plausibly all of its reported alpha=0.2031 numbers too) is
therefore contaminated by this artifact, not evidence of real
higher-order structure the calibration "never touches." This is the
exact leakage pattern this paper's own checklist names: a full-dataset,
label-dependent transform applied before the split it is later
evaluated on.

FIX: center on train indices only, apply the resulting affine map to
both splits. This script:
  1. Reproduces the diagnostic (cosine + train-fit-to-test-eval AUROC)
     for both centering methods, confirming the fix removes the
     artifact.
  2. Reconciles the permutation-null discrepancy: code/33's own
     sanity_check_permutation_null gave 0.1202+/-0.0170 (saved in
     results/real_feature_test_cv_calibrated.json), not the 0.504+/-0.043
     the paper text claimed -- both numbers are real, but the paper
     conflated two different checks. This reruns both explicitly and
     labels them correctly.
  3. Re-runs the CV-based alpha bisection and the full downstream
     LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO severity pipeline (both fully
     unmodified from code/33) using the calibration selected below, at
     capacities 128 and 384, 100 seeds each.

SECOND CORRECTION (a later independent review): train-only centering fixes
the *estimation* of the class means but NOT the *application* of the
shrinkage -- `mask = y == cls` ranges over the whole array, so each test
point's own label still decides which class offset gets subtracted from it.
That is a residual, smaller-scale instance of exactly the Mechanism-1
pattern this paper audits for, committed once again by this paper's own
instrument. The fix is `apply_calibration_label_free`: estimate the
class-mean axis u, the midpoint c, and the pooled within-class SD along u
from train indices only, then apply an identical, label-free map to every
row (shrink the projection onto u by alpha and re-inject
sqrt(1-alpha^2)*s_w of fresh independent noise along u, preserving the
marginal spread). Every number this script now reports uses that transform.
A simpler label-free candidate -- pure directional shrinkage toward the
midpoint, with no noise re-injection -- was tested and REJECTED: for any
alpha>0 it is an invertible linear map and therefore cannot reduce
separability at all (measured: AUROC ~0.99 at every alpha>0). Both the
superseded label-conditional transform and the rejected directional-shrink
transform are retained in the Part-1 diagnostic so the comparison is
reproducible.
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
OUT_PATH = ROOT / "results" / "calibration_leakage_diagnostic.json"
SEVERITY_OUT_PATH = ROOT / "results" / "real_feature_test_train_only_calibrated.json"

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


def load_raw_real_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


def apply_calibration_full_sample(X, y, alpha):
    """The code/33 (buggy) version: class means estimated from ALL samples."""
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2
    Xc = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        deviation = X[mask] - mu_cls
        new_class_mean = midpoint + alpha * (mu_cls - midpoint)
        Xc[mask] = new_class_mean + deviation
    return Xc


def apply_calibration_train_only(X, y, alpha, train_idx):
    """PARTIAL FIX (superseded, retained for the diagnostic table): class means
    estimated from train indices only, applied to both splits. A later
    independent review found this still conditions the *application* of the
    shrinkage on each point's own label (`mask = y == cls` ranges over the full
    array, test points included) -- a residual, smaller-scale instance of
    exactly the Mechanism-1 pattern this paper audits for. Superseded by
    `apply_calibration_label_free` below, which is used for every number this
    paper now reports."""
    y_tr = y[train_idx]
    mu_pos = X[train_idx][y_tr == 1].mean(axis=0)
    mu_neg = X[train_idx][y_tr == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2
    Xc = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        deviation = X[mask] - mu_cls
        new_class_mean = midpoint + alpha * (mu_cls - midpoint)
        Xc[mask] = new_class_mean + deviation
    return Xc


def apply_calibration_directional_shrink(X, y, alpha, train_idx, seed=0):
    """Label-free candidate #1 (REJECTED -- documented because it does not work).

    Shrink every point's own projection onto the train-estimated class-mean
    axis toward the train-estimated midpoint:  x -> x - (1-alpha)<x-c,u>u.
    This consults no label at application time, but for any alpha>0 it is an
    *invertible* linear map, so it cannot reduce linear separability at all --
    it rescales the discriminative axis and the within-class spread along that
    axis by the identical factor, leaving d' along u unchanged. Measured
    behaviour is a step function (AUROC ~0.99 at every alpha>0, collapsing only
    at alpha=0 where the map becomes singular), i.e. useless as a difficulty
    dial. Kept in the diagnostic so the rejection is reproducible."""
    y_tr = y[train_idx]
    mu_pos = X[train_idx][y_tr == 1].mean(axis=0)
    mu_neg = X[train_idx][y_tr == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    u = diff / (np.linalg.norm(diff) + 1e-12)
    c = (mu_pos + mu_neg) / 2
    proj = (X - c) @ u
    return X - (1.0 - alpha) * np.outer(proj, u)


def apply_calibration_label_free(X, y, alpha, train_idx, seed=0):
    """FINAL FIX: a fully label-free difficulty dial (no per-sample label is
    consulted when the transform is applied to any point, train or test).

    Everything the transform needs -- the class-mean axis u, the midpoint c,
    and the pooled within-class SD along u, s_w -- is estimated from the
    TRAIN indices only. The map applied to every row of X is then:

        p  = <x - c, u>
        p' = alpha*p + sqrt(1 - alpha^2) * s_w * eps,   eps ~ N(0,1) i.i.d.
        x' = x + (p' - p) * u

    This shrinks the between-class mean separation along u by exactly alpha
    while preserving the marginal spread along u (alpha^2 s_w^2 +
    (1-alpha^2) s_w^2 = s_w^2), so d' along u is alpha * ||dmu|| / s_w:
    monotone in alpha, hitting the chance level exactly at alpha=0. Unlike
    `apply_calibration_directional_shrink` it is not invertible (fresh
    independent noise is injected), so it genuinely dials difficulty; unlike
    `apply_calibration_train_only` it never looks at a point's own label."""
    y_tr = y[train_idx]
    mu_pos = X[train_idx][y_tr == 1].mean(axis=0)
    mu_neg = X[train_idx][y_tr == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    u = diff / (np.linalg.norm(diff) + 1e-12)
    c = (mu_pos + mu_neg) / 2
    proj_tr = (X[train_idx] - c) @ u
    s_w = float(np.sqrt(0.5 * (proj_tr[y_tr == 1].var() + proj_tr[y_tr == 0].var())))
    p = (X - c) @ u
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(len(X))
    p_new = alpha * p + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * s_w * eps
    return X + np.outer(p_new - p, u)


def diagnostic_cosine_and_auroc(X, y, calibration_fn, alpha, n_splits=20):
    """For a given calibration method, measure cos(delta_mu_train, delta_mu_test)
    and train-fit -> test-eval AUROC across many random splits."""
    cos_list, auc_list = [], []
    for seed in range(n_splits):
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed
        )
        if calibration_fn is apply_calibration_full_sample:
            Xc = calibration_fn(X, y, alpha)
        elif calibration_fn is apply_calibration_train_only:
            Xc = calibration_fn(X, y, alpha, idx_tr)
        else:
            Xc = calibration_fn(X, y, alpha, idx_tr, seed)
        y_tr, y_te = y[idx_tr], y[idx_te]
        Xtr, Xte = Xc[idx_tr], Xc[idx_te]
        dmu_tr = Xtr[y_tr == 1].mean(axis=0) - Xtr[y_tr == 0].mean(axis=0)
        dmu_te = Xte[y_te == 1].mean(axis=0) - Xte[y_te == 0].mean(axis=0)
        cos = float(np.dot(dmu_tr, dmu_te) / (np.linalg.norm(dmu_tr) * np.linalg.norm(dmu_te) + 1e-12))
        cos_list.append(cos)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(Xte)[:, 1])
        auc_list.append(auc)
    return float(np.mean(cos_list)), float(np.std(cos_list)), float(np.mean(auc_list)), float(np.std(auc_list))


def measure_cv_auroc_full_sample(X_calibrated, y, n_seeds=CALIB_SEEDS, C=CALIB_C):
    aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs = cross_val_predict(
            LogisticRegression(max_iter=5000, C=C), X_calibrated, y, cv=skf, method="predict_proba"
        )[:, 1]
        aucs.append(roc_auc_score(y, probs))
    return float(np.mean(aucs)), float(np.std(aucs))


def reconcile_permutation_null(X, y):
    """Rerun BOTH permutation checks that code/33 conflated, explicitly labeled."""
    rng = np.random.default_rng(2026)

    # Check A: plain uncalibrated features vs permuted labels (this is what the
    # paper's "0.504" prose actually describes -- a check on whether the CV
    # scoring method itself is unbiased under a pure null).
    check_a = []
    for i in range(10):
        y_perm = rng.permutation(y)
        auc, _ = measure_cv_auroc_full_sample(X, y_perm, n_seeds=1)
        check_a.append(auc)

    # Check B: code/33's actual sanity_check_permutation_null -- recompute
    # calibration parameters from PERMUTED labels at the real alpha, apply,
    # then score against those same permuted labels. This is what
    # calibration_permutation_null_check in the saved JSON actually is.
    alpha = 0.203125
    check_b = []
    for i in range(10):
        y_perm = rng.permutation(y)
        Xc_perm = apply_calibration_full_sample(X, y_perm, alpha)
        auc, _ = measure_cv_auroc_full_sample(Xc_perm, y_perm, n_seeds=1)
        check_b.append(auc)

    return {
        "check_A_plain_features_vs_permuted_labels": {"mean": float(np.mean(check_a)), "std": float(np.std(check_a))},
        "check_B_full_sample_calibration_from_permuted_labels": {"mean": float(np.mean(check_b)), "std": float(np.std(check_b))},
    }


# ---- Downstream severity pipeline: exact, unmodified port of code/33 ----

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
    """Train-only calibration is applied HERE, inside the split, unlike code/33
    which calibrated the whole X once before any split existed."""
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_calibrated = apply_calibration_label_free(
        X, y, ALPHA_TRAIN_ONLY, X_train_idx, seed=seed
    ).astype(np.float32)
    X_train, X_test = X_calibrated[X_train_idx], X_calibrated[X_test_idx]

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


ALPHA_TRAIN_ONLY = None  # set in main() after calibration search


def calibrate_alpha_train_only_empirically(X, y):
    """Bisection using the label-free calibration + CV logistic regression (fit
    on train fold, scored on held-out fold), so the calibration search itself
    never lets test information leak into the estimated axis/midpoint/spread,
    and no point's own label is consulted when the transform is applied."""
    lo, hi = 0.0, 1.0
    history = []
    for it in range(CALIB_MAX_ITER):
        mid = (lo + hi) / 2
        aucs = []
        for seed in range(CALIB_SEEDS):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_aucs = []
            for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xc = apply_calibration_label_free(X, y, mid, tr_idx, seed=seed * 10 + fold_i)
                clf = LogisticRegression(max_iter=5000, C=CALIB_C).fit(Xc[tr_idx], y[tr_idx])
                fold_aucs.append(roc_auc_score(y[te_idx], clf.predict_proba(Xc[te_idx])[:, 1]))
            aucs.append(np.mean(fold_aucs))
        auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
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
    return mid, history


def main():
    global ALPHA_TRAIN_ONLY
    t0 = time.time()
    X, y = load_raw_real_features()

    out = {}

    print("=== Part 1: cosine + train-fit->test-eval AUROC diagnostic ===", flush=True)
    diag = {}
    for alpha in [0.0, 0.2031, 1.0]:
        cos_full, cos_full_sd, auc_full, auc_full_sd = diagnostic_cosine_and_auroc(
            X, y, apply_calibration_full_sample, alpha
        )
        cos_tr, cos_tr_sd, auc_tr, auc_tr_sd = diagnostic_cosine_and_auroc(
            X, y, apply_calibration_train_only, alpha
        )
        cos_lf, cos_lf_sd, auc_lf, auc_lf_sd = diagnostic_cosine_and_auroc(
            X, y, apply_calibration_label_free, alpha
        )
        cos_ds, cos_ds_sd, auc_ds, auc_ds_sd = diagnostic_cosine_and_auroc(
            X, y, apply_calibration_directional_shrink, alpha
        )
        print(f"alpha={alpha}: FULL-SAMPLE cos={cos_full:.4f}+/-{cos_full_sd:.4f} auc={auc_full:.4f}+/-{auc_full_sd:.4f}"
              f"  |  TRAIN-ONLY(label-cond) cos={cos_tr:.4f}+/-{cos_tr_sd:.4f} auc={auc_tr:.4f}+/-{auc_tr_sd:.4f}"
              f"  |  LABEL-FREE cos={cos_lf:.4f}+/-{cos_lf_sd:.4f} auc={auc_lf:.4f}+/-{auc_lf_sd:.4f}"
              f"  |  DIR-SHRINK(rejected) cos={cos_ds:.4f}+/-{cos_ds_sd:.4f} auc={auc_ds:.4f}+/-{auc_ds_sd:.4f}", flush=True)
        diag[str(alpha)] = {
            "full_sample_centering": {"cos_mean": cos_full, "cos_std": cos_full_sd, "auc_mean": auc_full, "auc_std": auc_full_sd},
            "train_only_centering": {"cos_mean": cos_tr, "cos_std": cos_tr_sd, "auc_mean": auc_tr, "auc_std": auc_tr_sd},
            "label_free_axis_noising": {"cos_mean": cos_lf, "cos_std": cos_lf_sd, "auc_mean": auc_lf, "auc_std": auc_lf_sd},
            "directional_shrink_rejected": {"cos_mean": cos_ds, "cos_std": cos_ds_sd, "auc_mean": auc_ds, "auc_std": auc_ds_sd},
        }
    out["centering_diagnostic"] = diag

    print("\n=== Part 2: reconcile the permutation-null discrepancy ===", flush=True)
    recon = reconcile_permutation_null(X, y)
    print(json.dumps(recon, indent=2), flush=True)
    out["permutation_null_reconciliation"] = recon

    print("\n=== Part 3: calibrate alpha with label-free axis-noising calibration ===", flush=True)
    alpha, calib_history = calibrate_alpha_train_only_empirically(X, y)
    ALPHA_TRAIN_ONLY = alpha
    print(f"Converged alpha={alpha:.5f}", flush=True)
    out["train_only_calibration_alpha"] = alpha
    out["train_only_calibration_search_history"] = calib_history

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved diagnostic: {OUT_PATH}", flush=True)

    print("\n=== Part 4: full downstream severity pipeline, label-free calibration ===", flush=True)
    severity_out = {"calibration_alpha": float(alpha), "calibration_method": "label_free_axis_noising", "capacities": {}}
    for hidden in CAPACITIES:
        print(f"\n{'='*60}\nCapacity {hidden}, N_SEEDS={N_SEEDS}, train-only-calibrated (alpha={alpha:.4f})\n{'='*60}", flush=True)
        all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
        for seed in range(N_SEEDS):
            aucs = run_one_seed(X, y, hidden, seed)
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

        print(f"\n--- Capacity {hidden} summary (train-only-calibrated real features) ---")
        for k, v in arrs.items():
            print(f"  {k:15s} AUROC: mean={v.mean():.4f} std={v.std():.4f}")
        print(f"  leaky - clean_matched: mean={gap_lcm.mean():+.4f} p={p_lcm:.4g}")
        print(f"  clean_matched - placebo: mean={gap_cmp.mean():+.4f} p={p_cmp:.4g}")
        print(f"  leaky - placebo: mean={gap_lp.mean():+.4f} p={p_lp:.4g}")
        print(f"  DECISION RULE VERDICT: {verdict}")

        severity_out["capacities"][str(hidden)] = {
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "leaky_minus_clean_matched": {"mean": float(gap_lcm.mean()), "wilcoxon_p": float(p_lcm)},
            "clean_matched_minus_placebo": {"mean": float(gap_cmp.mean()), "wilcoxon_p": float(p_cmp)},
            "leaky_minus_placebo": {"mean": float(gap_lp.mean()), "wilcoxon_p": float(p_lp)},
            "decision_rule_verdict": verdict,
        }

    with open(SEVERITY_OUT_PATH, "w") as f:
        json.dump(severity_out, f, indent=2)
    print(f"\nSaved severity results: {SEVERITY_OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
