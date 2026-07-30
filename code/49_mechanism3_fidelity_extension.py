"""
Paper 2 -- Mechanism 3 fidelity extension.

The existing severity harness (code/33, code/43) captures MultiHaluDet's
per-fold checkpoint-selection leak (best-val-AUC checkpoint kept) but not
two other things the *actual* vendored trainer does every epoch, reacting
to the same validation fold's AUC score
(`code/external/MultiHaluDet/src/training/trainer.py`, lines 76-149):
  1. `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`, stepped on
     that fold's val AUC every epoch (`scheduler.step(auc_score)`).
  2. Early stopping tied to the SAME patience counter as the LR schedule
     (`patience_counter >= config.patience` breaks training).

This is a continuous, epoch-by-epoch reactive coupling to validation
feedback, not just a final argmax checkpoint pick -- a materially
different (and untested) leakage pathway from Mechanism 3 as already
quantified. This reruns the LEAKY vs. CLEAN_MATCHED vs. PLACEBO comparison
with this scheduler+patience mechanic ported in, on the same train-only-
calibrated real features (alpha=0.1328, code/43) at capacity 128, to see
whether it adds severity beyond the checkpoint-selection-only harness.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "mechanism3_fidelity_extension.json"

N_SEEDS = 100
HIDDEN = 128
N_INNER_FOLDS = 5
MAX_EPOCHS = 60
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
ALPHA_TRAIN_ONLY = 0.1328
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-5
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


def load_raw_real_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


def apply_calibration_train_only(X, y, alpha, train_idx):
    mu_pos = X[train_idx][y[train_idx] == 1].mean(axis=0)
    mu_neg = X[train_idx][y[train_idx] == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2
    X_calibrated = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        deviation = X[mask] - mu_cls
        new_class_mean = midpoint + alpha * (mu_cls - midpoint)
        X_calibrated[mask] = new_class_mean + deviation
    return X_calibrated


def train_with_scheduler_and_earlystop(X_tr, y_tr, X_sel, y_sel, hidden, max_epochs, seed):
    """Faithful port of MultiHaluDet's trainer.py mechanics: ReduceLROnPlateau
    stepped on the SAME validation fold's AUC every epoch, plus early
    stopping tied to the same patience counter, plus best-val-AUC checkpoint
    tracking -- all three reacting to the identical val signal."""
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state, best_epoch = -1.0, None, 0
    patience_counter = 0
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xs)).numpy()
        try:
            auc = roc_auc_score(y_sel, probs)
        except ValueError:
            auc = 0.5
        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= LR_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed_epochs(X_tr, y_tr, hidden, n_epochs, seed):
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(max(n_epochs, 1)):
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
    X_train_idx, X_test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_calibrated = apply_calibration_train_only(X, y, ALPHA_TRAIN_ONLY, X_train_idx).astype(np.float32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_calibrated[X_train_idx])
    X_test = scaler.transform(X_calibrated[X_test_idx])
    y_train, y_test = y[X_train_idx], y[X_test_idx]

    rng = np.random.default_rng(seed + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    conditions = ["leaky_plus_lrsched", "clean_matched_plus_lrsched", "placebo_plus_lrsched"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        # LEAKY: scheduler+early-stop react to the SAME val fold later scored as OOF.
        model_leaky, _ = train_with_scheduler_and_earlystop(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], hidden, MAX_EPOCHS, fold_seed,
        )
        oof["leaky_plus_lrsched"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky_plus_lrsched"] += extract_features(model_leaky, X_test)

        # CLEAN: scheduler+early-stop react to a genuine, disjoint ES-holdout
        # carved from tr_idx (never touches val_idx) -- gives an independently
        # selected epoch count, not LEAKY's own.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        _, best_epoch_clean = train_with_scheduler_and_earlystop(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], hidden, MAX_EPOCHS, fold_seed,
        )

        # CLEAN_MATCHED: budget-matched to LEAKY (retrained on the FULL tr_idx,
        # same seed as LEAKY), but for CLEAN's independently-selected epoch
        # count, with no scheduler reacting to val_idx.
        model_clean_matched = train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, best_epoch_clean, fold_seed)
        oof["clean_matched_plus_lrsched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched_plus_lrsched"] += extract_features(model_clean_matched, X_test)

        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo, _ = train_with_scheduler_and_earlystop(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted, hidden, MAX_EPOCHS, fold_seed,
        )
        oof["placebo_plus_lrsched"][val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat["placebo_plus_lrsched"] += extract_features(model_placebo, X_test)

    aucs = {}
    for k in oof:
        test_feat[k] /= N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    return aucs


def bca_ci(a, b, n_resamples=10000):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples,
                     method="BCa", random_state=RNG_GLOBAL)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    X, y = load_raw_real_features()
    print(f"Loaded real features: X={X.shape}, y={y.shape}, hall_rate={y.mean():.3f}")

    results = {k: [] for k in ["leaky_plus_lrsched", "clean_matched_plus_lrsched", "placebo_plus_lrsched"]}
    for seed in range(N_SEEDS):
        aucs = run_one_seed(X, y, HIDDEN, seed)
        for k in results:
            results[k].append(aucs[k])
        if (seed + 1) % 10 == 0:
            print(f"  seed {seed + 1}/{N_SEEDS} done", flush=True)

    gap_lp_cmp = np.array(results["leaky_plus_lrsched"]) - np.array(results["clean_matched_plus_lrsched"])
    gap_cmp_placebo = np.array(results["clean_matched_plus_lrsched"]) - np.array(results["placebo_plus_lrsched"])
    _, p_lp_cmp = wilcoxon(results["leaky_plus_lrsched"], results["clean_matched_plus_lrsched"])
    _, p_cmp_placebo = wilcoxon(results["clean_matched_plus_lrsched"], results["placebo_plus_lrsched"])
    ci_lp_cmp = bca_ci(results["leaky_plus_lrsched"], results["clean_matched_plus_lrsched"])
    ci_cmp_placebo = bca_ci(results["clean_matched_plus_lrsched"], results["placebo_plus_lrsched"])

    out = {
        "n_seeds": N_SEEDS, "hidden": HIDDEN, "alpha_train_only": ALPHA_TRAIN_ONLY,
        "means": {k: float(np.mean(v)) for k, v in results.items()},
        "leaky_plus_lrsched_minus_clean_matched_plus_lrsched": {
            "gap_mean": float(gap_lp_cmp.mean()), "bca_ci_95": ci_lp_cmp, "wilcoxon_p": float(p_lp_cmp),
        },
        "clean_matched_plus_lrsched_minus_placebo_plus_lrsched": {
            "gap_mean": float(gap_cmp_placebo.mean()), "bca_ci_95": ci_cmp_placebo, "wilcoxon_p": float(p_cmp_placebo),
        },
        "raw_per_seed": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nLEAKY_PLUS_LRSCHED vs CLEAN_MATCHED_PLUS_LRSCHED: gap={gap_lp_cmp.mean():+.4f} "
          f"CI={ci_lp_cmp} p={p_lp_cmp:.4g}")
    print(f"CLEAN_MATCHED_PLUS_LRSCHED vs PLACEBO_PLUS_LRSCHED: gap={gap_cmp_placebo.mean():+.4f} "
          f"CI={ci_cmp_placebo} p={p_cmp_placebo:.4g}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
