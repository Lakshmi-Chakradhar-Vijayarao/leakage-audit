"""
Paper 2 -- second, structurally different generative process for Case Study
3's synthetic severity reconstruction, closing a Priority-1 gap a fresh
review flagged: the original sweep (code/02d_corrected_capacity_placebo_sweep.py)
draws each class from an ISOTROPIC Gaussian (identity covariance, classes
differ only in a constant-offset mean). Real LLM hidden-state features are
almost certainly anisotropic and internally correlated across dimensions --
if the small checkpoint-selection-leakage severity estimate (+0.0009 to
+0.0034 AUROC) is an artifact of the isotropic assumption interacting with
how early-stopping selects checkpoints in a simple, uncorrelated feature
space, a differently-shaped generative process should show a different
severity, or none at all.

DESIGN: rather than inventing an arbitrary anisotropic covariance, or
building and training a small transformer from scratch (a much larger
undertaking with its own new sources of bugs), this calibrates the
anisotropic covariance DIRECTLY from the real Mistral-7B/HaluEval features
already used for the real-feature validation (results/
real_features_mistral7b_halueval.npz, n=400): estimate the real
pooled within-class covariance matrix Sigma and the real class-mean
difference direction Delta_mu, then rescale Delta_mu (keeping Sigma's real
shape/correlation structure exactly as observed) so the binormal AUROC
identity AUROC = Phi(sqrt(J/2)), J = Delta_mu^T Sigma^-1 Delta_mu, hits the
SAME calibration target (AUROC=0.80) as the original isotropic sweep. This
gives a synthetic generative process with a realistic, empirically-derived
covariance shape while keeping the controlled, repeatable, N_SEEDS=100
sweep methodology the rest of Case Study 3 depends on.

Everything else -- SweepMLP architecture, train_to_best_checkpoint,
train_fixed_epochs, the LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO run_one_seed
logic, capacities, N_SEEDS -- is an EXACT port of
code/02d_corrected_capacity_placebo_sweep.py, changing only
make_synthetic_data(). Any difference in the resulting severity estimate
is therefore attributable to the covariance structure, not to some other
procedural change.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
REAL_FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "anisotropic_covariance_capacity_sweep.json"

N_SAMPLES = 700
TARGET_AUROC = 0.80
TEST_SIZE = 0.20
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
N_SEEDS = 100
N_INNER_FOLDS = 5
CAPACITIES = [16, 48, 128, 384]


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


def fit_anisotropic_generator():
    """Estimate real pooled within-class covariance + mean-difference
    direction from the real Mistral-7B/HaluEval features, then rescale the
    mean difference so the binormal-AUROC identity hits TARGET_AUROC while
    keeping the real covariance's shape exactly as observed."""
    d = np.load(REAL_FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    # Standardize real features first so the covariance scale is comparable
    # to the original isotropic sweep's unit-variance-per-dim convention.
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    delta_mu = mu_pos - mu_neg

    # Pooled within-class covariance (real, anisotropic, correlated).
    Xc = np.vstack([X[y == 1] - mu_pos, X[y == 0] - mu_neg])
    Sigma = np.cov(Xc, rowvar=False)
    # Regularize (real n=400 << 414 dims, Sigma is otherwise singular).
    Sigma += np.eye(Sigma.shape[0]) * 0.05 * np.trace(Sigma) / Sigma.shape[0]

    Sigma_inv = np.linalg.pinv(Sigma)
    J_real = float(delta_mu @ Sigma_inv @ delta_mu)
    J_target = 2 * (norm.ppf(TARGET_AUROC)) ** 2
    alpha = np.sqrt(J_target / J_real)
    delta_mu_scaled = delta_mu * alpha

    achieved_auroc = norm.cdf(np.sqrt(J_target / 2))
    print(f"[calibration] real J={J_real:.4f}, target J={J_target:.4f} (AUROC={TARGET_AUROC}), "
          f"rescale alpha={alpha:.4f}, achieved AUROC (by construction)={achieved_auroc:.4f}")

    L = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * 1e-6)
    return delta_mu_scaled, L, X.shape[1]


DELTA_MU, CHOL_L, FEAT_DIM = fit_anisotropic_generator()


def make_synthetic_data(seed):
    rng = np.random.default_rng(seed)
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    z_pos = rng.standard_normal((n_pos, FEAT_DIM))
    z_neg = rng.standard_normal((n_neg, FEAT_DIM))
    X_pos = (DELTA_MU / 2) + z_pos @ CHOL_L.T
    X_neg = (-DELTA_MU / 2) + z_neg @ CHOL_L.T
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


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


def run_one_seed(seed, hidden):
    X, y = make_synthetic_data(seed)
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


def main():
    t0 = time.time()
    out = {"capacities": {}, "feat_dim": FEAT_DIM}
    for hidden in CAPACITIES:
        print(f"\n{'='*60}\nCapacity {hidden}, N_SEEDS={N_SEEDS}, anisotropic covariance (real-derived)\n{'='*60}", flush=True)
        all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
        for seed in range(N_SEEDS):
            aucs = run_one_seed(seed, hidden)
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
        _, p_lcm = wilcoxon(arrs["leaky"], arrs["clean_matched"])
        _, p_cmp = wilcoxon(arrs["clean_matched"], arrs["placebo"])

        print(f"\n--- Capacity {hidden} summary (anisotropic covariance) ---")
        for k, v in arrs.items():
            print(f"  {k:15s} AUROC: mean={v.mean():.4f} std={v.std():.4f}")
        print(f"  leaky - clean_matched: mean={gap_lcm.mean():+.4f} p={p_lcm:.4g}")
        print(f"  clean_matched - placebo: mean={gap_cmp.mean():+.4f} p={p_cmp:.4g}")

        out["capacities"][str(hidden)] = {
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "leaky_minus_clean_matched": {"mean": float(gap_lcm.mean()), "wilcoxon_p": float(p_lcm)},
            "clean_matched_minus_placebo": {"mean": float(gap_cmp.mean()), "wilcoxon_p": float(p_cmp)},
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
