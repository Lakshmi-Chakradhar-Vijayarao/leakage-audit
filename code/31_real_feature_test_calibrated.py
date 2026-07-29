"""
Paper 2 -- fix a CRITICAL flaw a fresh review caught in code/25's "corrected
architecture" real-feature validation: even with the architecture bug
fixed, the real Mistral-7B/HaluEval features are separable enough that
LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO all sit at AUROC ~0.985 -- the exact
ceiling problem Appendix A's own first correction-history entry describes
("a task with no room to fail leaves no room for a leakage bug to show
inflation either"). The synthetic sweep was deliberately calibrated DOWN
to AUROC=0.80 for this reason; code/25 never applied the same discipline
to the real features, so its near-zero gap and "adequately powered null"
claim compare an MDE computed at one operating point (0.80, synthetic)
to an effect measured at a different one (0.985, real) -- not a valid
comparison, and on a scale-invariant reading (relative error reduction,
or the Fisher-J the paper's own binormal identity already uses
elsewhere) the observed real-feature gap is consistent with, not against,
the synthetic estimate transferring.

FIX: keep the REAL per-sample noise/covariance exactly as observed (no
Gaussian resampling, unlike code/27's anisotropic sweep) and rescale ONLY
the between-class mean separation, by the same alpha code/27 already
computes (fit from the real pooled covariance and the binormal AUROC
identity), so the calibrated real-feature task targets the same AUROC=0.80
regime as the synthetic sweep:

    new_x_i = midpoint + alpha * (class_mean_i - midpoint) + (x_i - class_mean_i)

This shifts each sample's class mean toward the midpoint by (1-alpha)
while leaving that exact sample's own real deviation from its class mean
(the real noise) completely untouched. Everything else -- SweepMLP,
train_to_best_checkpoint, the 5-fold OOF + meta-learner architecture,
capacities, N_SEEDS -- is an exact port of code/25.
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
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "real_feature_test_calibrated.json"

CAPACITIES = [128, 384]
N_SEEDS = 100
N_INNER_FOLDS = 5
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
TARGET_AUROC = 0.80


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


def load_calibrated_real_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2
    delta_mu = mu_pos - mu_neg

    Xc = np.vstack([X[y == 1] - mu_pos, X[y == 0] - mu_neg])
    Sigma = np.cov(Xc, rowvar=False)
    Sigma += np.eye(Sigma.shape[0]) * 0.05 * np.trace(Sigma) / Sigma.shape[0]
    Sigma_inv = np.linalg.pinv(Sigma)
    J_real = float(delta_mu @ Sigma_inv @ delta_mu)
    J_target = 2 * (norm.ppf(TARGET_AUROC)) ** 2
    alpha = np.sqrt(J_target / J_real)
    print(f"[calibration] real J={J_real:.4f}, target J={J_target:.4f} (AUROC={TARGET_AUROC}), "
          f"rescale alpha={alpha:.4f}", flush=True)

    X_calibrated = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        deviation = X[mask] - mu_cls  # real, unmodified per-sample noise
        new_class_mean = midpoint + alpha * (mu_cls - midpoint)
        X_calibrated[mask] = new_class_mean + deviation

    return X_calibrated.astype(np.float32), y, alpha


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
    X, y, alpha = load_calibrated_real_features()
    print(f"Calibrated real feature matrix: {X.shape}, hall_rate={1-y.mean():.3f}", flush=True)

    out = {"capacities": {}, "calibration_alpha": float(alpha)}
    for hidden in CAPACITIES:
        print(f"\n{'='*60}\nCapacity {hidden}, N_SEEDS={N_SEEDS}, calibrated real features (alpha={alpha:.4f})\n{'='*60}", flush=True)
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

        print(f"\n--- Capacity {hidden} summary (calibrated real features) ---")
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
