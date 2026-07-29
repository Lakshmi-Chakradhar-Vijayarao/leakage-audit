"""
Paper 2 -- elite-review follow-up on SS4.3's real-feature validation.
Two things this closes, reusing the ALREADY-CACHED real Mistral-7B
features (results/real_features_mistral7b_halueval.npz) -- no GPU, no
model loading, no re-extraction needed:

1. **Exact MDE, not a back-of-envelope normal approximation.** The
   original run (code/03_real_feature_leakage_test.py) never saved the
   full per-seed AUROC arrays, only the mean gap and Wilcoxon p -- so
   S4.3's power-check paragraph had to back-calculate an approximate
   per-seed SD from p and the mean gap. This script saves the full
   per-seed arrays for all four conditions and derives the MDE directly
   from the empirical per-seed gap SD.

2. **A precise mechanistic diagnostic for the CLEAN_MATCHED-underperforms-
   PLACEBO anomaly.** Elite review flagged that our original hypothesis
   ("CLEAN_MATCHED trains on less data than PLACEBO, from the held-out
   carve-out") does not actually match the code: CLEAN_MATCHED retrains
   on the FULL tr_idx (matching PLACEBO and LEAKY's data size exactly),
   for a FIXED epoch count (`best_epoch`) inherited from CLEAN's
   early-stopping run on a smaller, disjoint fold. The real candidate
   mechanism is therefore an EPOCH-COUNT selection difference: PLACEBO's
   train_to_best_checkpoint() picks whichever epoch happens to maximize
   AUROC against permuted validation labels -- which, at high enough
   capacity, is not a random pick but tends to select later, more-fit-to-
   real-training-data epochs, purely from overfitting dynamics unrelated
   to the permutation being informative. CLEAN's early-stopping (on a
   smaller es_idx fold) may stop earlier due to noisier held-out signal
   at that reduced sample size. This script captures both best_epoch
   values per seed and reports their paired distribution and correlation
   with the AUC gap, to test this mechanism directly rather than leaving
   it as an untested hypothesis.

CORRECTION (post-review): this script reuses code/03's single-fold
architecture, which does not match code/02d_corrected_capacity_placebo_sweep.py
(see the correction note in code/03_real_feature_leakage_test.py). Under
the corrected 5-fold-OOF-plus-meta-learner architecture
(code/25_real_feature_leakage_test_corrected_architecture.py), the
CLEAN_MATCHED-vs-PLACEBO anomaly this script investigates is not
significant and does not reproduce. The epoch-count mechanism documented
below is retained for the historical record (Appendix A) but does not
describe the paper's reported result.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "real_feature_leakage_diagnostics.json"

CAPACITY = 128
N_SEEDS = 100
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
RANDOM_STATE = 42


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


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel, hidden, epochs, seed):
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state, best_epoch = -1.0, None, 0
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
        if auc > best_auc:
            best_auc, best_state, best_epoch = auc, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1
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
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
    model.eval()
    return model


def eval_auc(model, X, y):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
    return float(roc_auc_score(y, probs))


def run_one_seed(X, y, hidden, fold_seed):
    rng = np.random.default_rng(fold_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=fold_seed, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    tr_idx, val_idx = next(iter(skf.split(X_train, y_train)))

    aucs = {}
    model_leaky, leaky_epoch = train_to_best_checkpoint(
        X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], hidden, EPOCHS, fold_seed)
    aucs["leaky"] = eval_auc(model_leaky, X_test, y_test)

    tr2_idx, es_idx = train_test_split(
        tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed)
    model_clean, clean_best_epoch = train_to_best_checkpoint(
        X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], hidden, EPOCHS, fold_seed)
    aucs["clean"] = eval_auc(model_clean, X_test, y_test)

    model_clean_matched = train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, clean_best_epoch, fold_seed)
    aucs["clean_matched"] = eval_auc(model_clean_matched, X_test, y_test)

    y_val_permuted = rng.permutation(y_train[val_idx])
    model_placebo, placebo_best_epoch = train_to_best_checkpoint(
        X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted, hidden, EPOCHS, fold_seed)
    aucs["placebo"] = eval_auc(model_placebo, X_test, y_test)

    return aucs, leaky_epoch, clean_best_epoch, placebo_best_epoch


def main():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob])
    print(f"Combined feature matrix: {X.shape}, hall_rate={1-y.mean():.3f}")

    all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
    leaky_epochs, clean_epochs, placebo_epochs = [], [], []
    for seed in range(N_SEEDS):
        aucs, le, ce, pe = run_one_seed(X, y, CAPACITY, seed)
        for k, v in aucs.items():
            all_aucs[k].append(v)
        leaky_epochs.append(le)
        clean_epochs.append(ce)
        placebo_epochs.append(pe)
        if (seed + 1) % 10 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}: clean_matched={np.mean(all_aucs['clean_matched']):.4f} "
                  f"placebo={np.mean(all_aucs['placebo']):.4f} "
                  f"clean_epoch={np.mean(clean_epochs):.1f} placebo_epoch={np.mean(placebo_epochs):.1f}",
                  flush=True)

    clean_epochs = np.array(clean_epochs)
    placebo_epochs = np.array(placebo_epochs)
    epoch_diff = placebo_epochs - clean_epochs
    gap_clean_matched_placebo = np.array(all_aucs["clean_matched"]) - np.array(all_aucs["placebo"])

    _, epoch_wilcoxon_p = wilcoxon(epoch_diff) if not np.allclose(epoch_diff, 0) else (None, 1.0)
    corr, corr_p = pearsonr(epoch_diff, gap_clean_matched_placebo)

    print(f"\n=== Epoch-count diagnostic ===")
    print(f"CLEAN best_epoch: mean={clean_epochs.mean():.2f}, sd={clean_epochs.std():.2f}")
    print(f"PLACEBO best_epoch: mean={placebo_epochs.mean():.2f}, sd={placebo_epochs.std():.2f}")
    print(f"PLACEBO - CLEAN epoch diff: mean={epoch_diff.mean():.2f}, Wilcoxon p={epoch_wilcoxon_p:.4g}")
    print(f"Correlation(epoch_diff, AUC_gap[clean_matched-placebo]): r={corr:.4f}, p={corr_p:.4g}")

    def gap_stats_exact(a, b):
        arr_a, arr_b = np.array(all_aucs[a]), np.array(all_aucs[b])
        gap = arr_a - arr_b
        try:
            _, p = wilcoxon(gap)
        except ValueError:
            p = 1.0
        n = len(gap)
        sd = float(gap.std(ddof=1))
        se = sd / np.sqrt(n)
        # 80%-power two-sided MDE for a one-sample (paired-gap) test
        from scipy.stats import norm as _norm
        mde = se * (_norm.ppf(0.975) + _norm.ppf(0.80))
        return {"mean_a": float(arr_a.mean()), "mean_b": float(arr_b.mean()),
                "mean_gap": float(gap.mean()), "std_gap_exact": sd, "wilcoxon_p": float(p),
                "n": n, "exact_mde_80pct_power": float(mde)}

    gaps_exact = {
        "leaky_minus_placebo": gap_stats_exact("leaky", "placebo"),
        "clean_minus_placebo": gap_stats_exact("clean", "placebo"),
        "clean_matched_minus_placebo": gap_stats_exact("clean_matched", "placebo"),
        "leaky_minus_clean_matched": gap_stats_exact("leaky", "clean_matched"),
    }
    print(f"\nExact (not back-calculated) gap stats:")
    for k, v in gaps_exact.items():
        print(f"  {k}: gap={v['mean_gap']:+.4f} sd={v['std_gap_exact']:.4f} "
              f"p={v['wilcoxon_p']:.4g} MDE(80% power)={v['exact_mde_80pct_power']:.4f}")

    out = {
        "n_seeds": N_SEEDS, "capacity": CAPACITY,
        "mean_aucs": {k: float(np.mean(v)) for k, v in all_aucs.items()},
        "per_seed_aucs": {k: [float(x) for x in v] for k, v in all_aucs.items()},
        "gaps_exact": gaps_exact,
        "epoch_diagnostic": {
            "clean_best_epoch_mean": float(clean_epochs.mean()), "clean_best_epoch_sd": float(clean_epochs.std()),
            "placebo_best_epoch_mean": float(placebo_epochs.mean()), "placebo_best_epoch_sd": float(placebo_epochs.std()),
            "leaky_best_epoch_mean": float(np.mean(leaky_epochs)),
            "placebo_minus_clean_epoch_diff_mean": float(epoch_diff.mean()),
            "epoch_diff_wilcoxon_p": float(epoch_wilcoxon_p) if epoch_wilcoxon_p is not None else None,
            "corr_epoch_diff_vs_auc_gap": float(corr), "corr_p": float(corr_p),
            "per_seed_clean_epoch": [int(x) for x in clean_epochs],
            "per_seed_placebo_epoch": [int(x) for x in placebo_epochs],
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
