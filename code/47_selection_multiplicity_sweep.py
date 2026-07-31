"""
Paper 2 -- the reviewer's highest-leverage suggestion: the existing
capacity sweep (code/02d) varies a parameter (MLP hidden width) that
does not actually drive checkpoint-selection winner's-curse severity.
The mechanism is a winner's-curse over K candidate checkpoints, so its
magnitude should scale with K (number of candidates) and n_val
(selection-set size), not capacity. This sweeps the parameters that
actually matter, holding capacity fixed at 128 (the isotropic sweep's
significant cell) throughout, with independently decoupled seeds
(data/split/fold/init -- previously conflated into one `seed` variable,
the root cause of Appendix A's third correction-history issue).

Sweep A -- K (number of candidate checkpoints, i.e. EPOCHS): {1,5,15,45,135}.
  Pre-registered prediction: gap ~ c * sigma_val * sqrt(2*ln(K)), the
  extreme-value-theory scaling for the expected max of K noisy estimates.
  K=1 has no selection at all and must give a gap of ~0 by construction.

Sweep B -- n_val (selection-set size, via N_SAMPLES): {350, 700, 2800}.
  Pre-registered prediction: gap ~ 1/sqrt(n_val).

Sweep C -- operating point (TARGET_AUROC): {0.70, 0.80, 0.90, 0.95, 0.985}.
  The last matches MultiHaluDet's actual reported AUROC (0.9855).

Everything else -- SweepMLP, the isotropic-Gaussian generative process,
LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO logic -- is an exact port of
code/02d_corrected_capacity_placebo_sweep.py, with the single `seed`
variable decoupled into `data_seed` (which synthetic sample), `split_seed`
(train/test split), `fold_seed` (inner CV fold assignment), and
`init_seed` (torch model initialization).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanity_checks import assert_calibration

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "selection_multiplicity_sweep.json"

CAPACITY = 128
TEST_SIZE = 0.20
ES_HOLD_FRACTION = 0.15
N_INNER_FOLDS = 5
DEFAULT_EPOCHS = 45
DEFAULT_N_SAMPLES = 700
DEFAULT_TARGET_AUROC = 0.80
FEAT_DIM = 64
RNG_GLOBAL = np.random.default_rng(2026)


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
        h = self.net[0](x); h = self.net[1](h); h = self.net[3](h); h = self.net[4](h)
        return h


def make_synthetic_data(data_seed, n_samples, target_auroc):
    """FIXED (independent adversarial review found this, same bug as
    code/46): per-dimension mean difference must be class_sep, not
    2*class_sep, to realize ||delta_mu||^2 = j_target as intended (matching
    code/02d's mean_pos=CLASS_SEP/2, mean_neg=-CLASS_SEP/2 convention). The
    original +-class_sep version silently quadrupled the realized Fisher J."""
    rng = np.random.default_rng(data_seed)
    j_target = 2 * (norm.ppf(target_auroc)) ** 2
    class_sep = np.sqrt(j_target / FEAT_DIM)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    X_pos = class_sep / 2 + rng.standard_normal((n_pos, FEAT_DIM))
    X_neg = -class_sep / 2 + rng.standard_normal((n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    assert_calibration(X, y, target_auroc)
    return X[perm], y[perm]


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel, hidden, epochs, init_seed):
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state, best_epoch = -1.0, None, 0
    val_aucs = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xs)).numpy()
        try:
            auc = roc_auc_score(y_sel, probs)
        except ValueError:
            auc = 0.5
        val_aucs.append(auc)
        if auc > best_auc:
            best_auc, best_state, best_epoch = auc, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1
    model.load_state_dict(best_state)
    return model, best_epoch, float(np.std(val_aucs))


def train_fixed_epochs(X_tr, y_tr, hidden, n_epochs, init_seed):
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(max(n_epochs, 1)):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
    model.eval()
    return model


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        return model.features(torch.tensor(X, dtype=torch.float32)).numpy()


def run_one_seed(data_seed, split_seed, fold_seed_base, init_seed_base, hidden, epochs, n_samples, target_auroc):
    X, y = make_synthetic_data(data_seed, n_samples, target_auroc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    rng = np.random.default_rng(fold_seed_base + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=fold_seed_base)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    conditions = ["leaky", "clean", "clean_matched", "placebo"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}
    val_stds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        init_seed = init_seed_base * 100 + fold
        model_leaky, _, val_std = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], hidden, epochs, init_seed)
        val_stds.append(val_std)
        oof["leaky"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky"] += extract_features(model_leaky, X_test)

        tr2_idx, es_idx = train_test_split(tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=init_seed)
        model_clean, best_epoch, _ = train_to_best_checkpoint(X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], hidden, epochs, init_seed)
        oof["clean"][val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat["clean"] += extract_features(model_clean, X_test)

        model_clean_matched = train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, init_seed)
        oof["clean_matched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(model_clean_matched, X_test)

        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo, _, _ = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted, hidden, epochs, init_seed)
        oof["placebo"][val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat["placebo"] += extract_features(model_placebo, X_test)

    aucs = {}
    for k in oof:
        test_feat[k] /= N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    aucs["_val_auc_std"] = float(np.mean(val_stds))
    return aucs


def run_sweep_cell(n_seeds, hidden, epochs, n_samples, target_auroc):
    all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
    val_stds = []
    for seed in range(n_seeds):
        data_seed = seed
        split_seed = seed + 100000
        fold_seed_base = seed + 200000
        init_seed_base = seed + 300000
        aucs = run_one_seed(data_seed, split_seed, fold_seed_base, init_seed_base, hidden, epochs, n_samples, target_auroc)
        for k in all_aucs:
            all_aucs[k].append(aucs[k])
        val_stds.append(aucs["_val_auc_std"])
    arrs = {k: np.array(v) for k, v in all_aucs.items()}
    gap = arrs["leaky"] - arrs["clean_matched"]
    _, p = wilcoxon(arrs["leaky"], arrs["clean_matched"]) if not np.allclose(gap, 0) else (None, 1.0)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa", random_state=RNG_GLOBAL)
    return {
        "gap_mean": float(gap.mean()), "gap_bca_ci_95": [float(res.confidence_interval.low), float(res.confidence_interval.high)],
        "wilcoxon_p": float(p), "mean_val_auc_std": float(np.mean(val_stds)),
        "leaky_mean": float(arrs["leaky"].mean()), "clean_matched_mean": float(arrs["clean_matched"].mean()),
        "placebo_mean": float(arrs["placebo"].mean()),
    }


def main():
    t0 = time.time()
    N_SEEDS = 100
    out = {"sweep_A_K": {}, "sweep_B_n_val": {}, "sweep_C_operating_point": {}}

    print("=== Sweep A: K (number of candidate checkpoints) ===", flush=True)
    SWEEP_A_K_VALUES = [1, 3, 5, 10, 15, 25, 45, 75, 135, 225]
    SWEEP_A_N_SEEDS = 200
    for K in SWEEP_A_K_VALUES:
        r = run_sweep_cell(SWEEP_A_N_SEEDS, CAPACITY, K, DEFAULT_N_SAMPLES, DEFAULT_TARGET_AUROC)
        out["sweep_A_K"][str(K)] = r
        print(f"  K={K}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} p={r['wilcoxon_p']:.4g} "
              f"val_auc_std={r['mean_val_auc_std']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

    # Fit gap ~ c * sigma_val * sqrt(2 ln K)
    Ks = np.array(SWEEP_A_K_VALUES, dtype=float)
    gaps = np.array([out["sweep_A_K"][str(int(k))]["gap_mean"] for k in Ks])
    sigma_val = np.mean([out["sweep_A_K"][str(int(k))]["mean_val_auc_std"] for k in Ks])
    predictor = sigma_val * np.sqrt(2 * np.log(np.maximum(Ks, 1.01)))
    valid = Ks > 1
    if valid.sum() >= 2:
        c_fit = float(np.sum(gaps[valid] * predictor[valid]) / np.sum(predictor[valid] ** 2))
        pred_gaps = c_fit * predictor
        ss_res = np.sum((gaps[valid] - pred_gaps[valid]) ** 2)
        ss_tot = np.sum((gaps[valid] - gaps[valid].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    else:
        c_fit, r2 = None, None
    out["sweep_A_fit"] = {"c_fit": c_fit, "r_squared": r2, "sigma_val_used": float(sigma_val)}
    print(f"  Extreme-value fit: gap ~= {c_fit} * sigma_val * sqrt(2 ln K), R^2={r2}", flush=True)

    print("\n=== Sweep B: n_val (selection-set size, via N_SAMPLES) ===", flush=True)
    for n_samples in [350, 700, 2800]:
        r = run_sweep_cell(N_SEEDS, CAPACITY, DEFAULT_EPOCHS, n_samples, DEFAULT_TARGET_AUROC)
        n_val_approx = int(n_samples * (1 - TEST_SIZE) / N_INNER_FOLDS)
        out["sweep_B_n_val"][str(n_samples)] = {**r, "n_val_approx": n_val_approx}
        print(f"  N_SAMPLES={n_samples} (n_val~={n_val_approx}): gap={r['gap_mean']:+.4f} "
              f"CI={r['gap_bca_ci_95']} p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    print("\n=== Sweep C: operating point (TARGET_AUROC) ===", flush=True)
    for target_auroc in [0.70, 0.80, 0.90, 0.95, 0.985]:
        r = run_sweep_cell(N_SEEDS, CAPACITY, DEFAULT_EPOCHS, DEFAULT_N_SAMPLES, target_auroc)
        out["sweep_C_operating_point"][str(target_auroc)] = r
        print(f"  TARGET_AUROC={target_auroc}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} "
              f"p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
