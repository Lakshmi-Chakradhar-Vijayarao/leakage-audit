"""
Paper 2 — Bulletproofing check: does the Case-Study-3 leakage gap scale with model capacity?

WHY THIS EXISTS
---------------
02_synthetic_leakage_ablation.py found a small, non-significant LEAKY-minus-
CLEAN AUROC gap (+0.0019, Wilcoxon p=0.17) for MultiHaluDet's checkpoint-
selection-leakage mechanism, using a small TinyMLP (hidden=48). The
obvious, most damaging reviewer question: "your 'this leakage is mild'
conclusion depends entirely on an arbitrarily small MLP -- MultiHaluDet's
ACTUAL model has hidden_dim=384 (an 8x larger penultimate representation)
plus 6 transformer layers, multi-scale branches, and heavy augmentation.
Why should your conclusion transfer?"

This script directly answers that by sweeping model capacity -- including
a hidden_dim=384 condition matching MultiHaluDet's actual config value --
and checking whether the LEAKY-minus-CLEAN gap grows as capacity
increases. Same exact mechanism (checkpoint selection on val_idx vs. a
disjoint carve-out), same synthetic task calibration (AUROC~0.80 via the
Fisher-ratio/AUROC identity), fewer seeds per capacity level (10 instead
of 20) to keep total runtime reasonable while still supporting a paired
test at each level.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 700
FEAT_DIM = 64
N_INNER_FOLDS = 5
TARGET_AUROC = 0.80
FISHER_J = (2 * norm.ppf(TARGET_AUROC)) ** 2
CLASS_SEP = float(np.sqrt(FISHER_J / FEAT_DIM))
TEST_SIZE = 0.20
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
N_SEEDS = 10
HIDDEN_SIZES = [16, 48, 128, 384]  # 384 matches MultiHaluDet's actual config.hidden_dim


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


def make_synthetic_data(seed):
    rng = np.random.default_rng(seed)
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    mean_pos = np.full(FEAT_DIM, CLASS_SEP / 2)
    mean_neg = np.full(FEAT_DIM, -CLASS_SEP / 2)
    X_pos = rng.normal(mean_pos, 1.0, size=(n_pos, FEAT_DIM))
    X_neg = rng.normal(mean_neg, 1.0, size=(n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel, hidden, epochs, seed):
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xs)).numpy()
        auc = roc_auc_score(y_sel, probs)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
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

    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = hidden // 2
    oof_leaky = np.zeros((len(y_train), feat_dim_out))
    oof_clean = np.zeros((len(y_train), feat_dim_out))
    test_feat_leaky = np.zeros((len(y_test), feat_dim_out))
    test_feat_clean = np.zeros((len(y_test), feat_dim_out))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        model_leaky = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof_leaky[val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat_leaky += extract_features(model_leaky, X_test)

        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof_clean[val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat_clean += extract_features(model_clean, X_test)

    test_feat_leaky /= N_INNER_FOLDS
    test_feat_clean /= N_INNER_FOLDS

    clf_leaky = LogisticRegression(max_iter=2000).fit(oof_leaky, y_train)
    auc_leaky = roc_auc_score(y_test, clf_leaky.predict_proba(test_feat_leaky)[:, 1])
    clf_clean = LogisticRegression(max_iter=2000).fit(oof_clean, y_train)
    auc_clean = roc_auc_score(y_test, clf_clean.predict_proba(test_feat_clean)[:, 1])
    return auc_leaky, auc_clean


def main():
    print(f"Calibration: target AUROC={TARGET_AUROC} -> J={FISHER_J:.3f} -> CLASS_SEP={CLASS_SEP:.4f}")
    summary = {}
    for hidden in HIDDEN_SIZES:
        leaky_aucs, clean_aucs = [], []
        for seed in range(N_SEEDS):
            al, ac = run_one_seed(seed, hidden)
            leaky_aucs.append(al)
            clean_aucs.append(ac)
        leaky_aucs = np.array(leaky_aucs)
        clean_aucs = np.array(clean_aucs)
        gaps = leaky_aucs - clean_aucs
        try:
            stat, p = wilcoxon(leaky_aucs, clean_aucs)
        except ValueError:
            stat, p = float("nan"), 1.0  # all-zero differences
        print(f"\nhidden={hidden:4d}  LEAKY={leaky_aucs.mean():.4f}  CLEAN={clean_aucs.mean():.4f}  "
              f"gap={gaps.mean():+.4f} (std {gaps.std():.4f})  p={p:.4f}  "
              f"positive in {int((gaps>0).sum())}/{N_SEEDS}")
        summary[hidden] = {
            "leaky_mean": float(leaky_aucs.mean()), "clean_mean": float(clean_aucs.mean()),
            "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
            "wilcoxon_p": float(p), "positive_gap_seeds": int((gaps > 0).sum()), "n_seeds": N_SEEDS,
        }

    print("\n=== Capacity sweep summary ===")
    gaps_by_hidden = [summary[h]["gap_mean"] for h in HIDDEN_SIZES]
    trend = "GROWING with capacity" if gaps_by_hidden[-1] > gaps_by_hidden[0] * 1.5 else "NOT clearly growing with capacity"
    print(f"Gap at hidden=16:  {summary[16]['gap_mean']:+.4f}")
    print(f"Gap at hidden=384 (MultiHaluDet's actual config.hidden_dim): {summary[384]['gap_mean']:+.4f}")
    print(f"Trend: {trend}")

    import json
    from pathlib import Path
    out = {"config": {"n_samples": N_SAMPLES, "feat_dim": FEAT_DIM, "n_inner_folds": N_INNER_FOLDS,
                       "epochs": EPOCHS, "n_seeds": N_SEEDS, "hidden_sizes": HIDDEN_SIZES},
           "by_hidden_size": summary, "trend": trend}
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "capacity_sweep_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'capacity_sweep_ablation.json'}")


if __name__ == "__main__":
    main()
