"""
Paper 2 -- fresh-review follow-up on Case Study 3's synthetic reconstruction
(code/02d_corrected_capacity_placebo_sweep.py).

An independent review identified a plausible fourth confound in the
LEAKY-vs-CLEAN_MATCHED comparison that section 4.3 treats as the paper's
decisive test: CLEAN_MATCHED trains on the full tr_idx (matching LEAKY's
budget) but with NO adaptive checkpoint selection during that run --
train_fixed_epochs() blindly trains to a pre-computed epoch count inherited
from an earlier, separate run. LEAKY, by contrast, gets full budget AND an
adaptive best-of-45 checkpoint search against val_idx during the same run.
So the LEAKY-vs-CLEAN_MATCHED gap could reflect "having any adaptive
selection mechanism at all," not "specifically reusing val_idx's real
labels downstream as out-of-fold features."

This script adds a fifth condition, CLEAN_MATCHED_ADAPTIVE, that isolates
this: full tr_idx training budget (matches LEAKY/PLACEBO), WITH ongoing
adaptive checkpoint selection during that same run (matches LEAKY's
selection mechanism exactly), but selecting against es_idx -- the same
disjoint, honestly-held-out fold CLEAN uses -- whose labels are real but
which is discarded after selection, never reused downstream as features.

LEAKY and CLEAN_MATCHED_ADAPTIVE now differ in exactly one respect: whether
the fold used for adaptive checkpoint selection is the same fold later
reused as out-of-fold features (LEAKY) or a disjoint fold that is not
(CLEAN_MATCHED_ADAPTIVE). If LEAKY still beats CLEAN_MATCHED_ADAPTIVE by
roughly the same margin as it beats the original CLEAN_MATCHED, that is
direct evidence the effect is fold-reuse leakage specifically, not merely
"adaptive selection helps." If the gap collapses relative to
CLEAN_MATCHED_ADAPTIVE, the alternative-confound hypothesis is supported
and the original severity estimate is partly an artifact of selection
mechanism, not fold reuse.
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

N_SAMPLES = 700
FEAT_DIM = 64
N_INNER_FOLDS = 5
TARGET_AUROC = 0.80

FISHER_J = 2 * (norm.ppf(TARGET_AUROC)) ** 2
CLASS_SEP = float(np.sqrt(FISHER_J / FEAT_DIM))

TEST_SIZE = 0.20
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
N_SEEDS = 100
CAPACITIES = [128, 384]  # the two capacities with a real residual to explain


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
    conditions = ["leaky", "clean_matched", "clean_matched_adaptive", "placebo"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        # LEAKY: adaptive selection on val_idx's TRUE labels (the same fold
        # reused downstream as out-of-fold features). Full training budget.
        model_leaky, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["leaky"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky"] += extract_features(model_leaky, X_test)

        # es_idx: disjoint honest carve-out, used for selection only, never
        # reused downstream -- same construction CLEAN/CLEAN_MATCHED use.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )

        # CLEAN_MATCHED (existing, from 02d): select best_epoch on the
        # smaller-budget tr2_idx/es_idx run, then blindly retrain on the
        # full tr_idx for exactly that many epochs -- NO adaptive selection
        # during the full-budget run itself.
        _, best_epoch = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        model_clean_matched = train_fixed_epochs(
            X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed,
        )
        oof["clean_matched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(model_clean_matched, X_test)

        # CLEAN_MATCHED_ADAPTIVE (new control): full tr_idx training budget
        # AND ongoing adaptive checkpoint selection during that same run,
        # exactly like LEAKY -- but selecting against es_idx's true labels
        # (disjoint, discarded after selection) instead of val_idx's (reused
        # downstream). Isolates fold-reuse specifically from "having
        # adaptive selection" in general.
        model_cma, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["clean_matched_adaptive"][val_idx] = extract_features(model_cma, X_train[val_idx])
        test_feat["clean_matched_adaptive"] += extract_features(model_cma, X_test)

        # PLACEBO: adaptive selection on val_idx's PERMUTED labels, full budget.
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
    results_by_capacity = {}

    for hidden in CAPACITIES:
        print(f"\n{'='*70}\nHIDDEN={hidden}  N_SEEDS={N_SEEDS}\n{'='*70}")
        all_aucs = {k: [] for k in ["leaky", "clean_matched", "clean_matched_adaptive", "placebo"]}
        for seed in range(N_SEEDS):
            aucs = run_one_seed(seed, hidden)
            for k, v in aucs.items():
                all_aucs[k].append(v)
            if (seed + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{seed+1}/{N_SEEDS}] elapsed={elapsed:.0f}s  "
                      f"leaky={aucs['leaky']:.4f} clean_matched={aucs['clean_matched']:.4f} "
                      f"clean_matched_adaptive={aucs['clean_matched_adaptive']:.4f} placebo={aucs['placebo']:.4f}")

        arrs = {k: np.array(v) for k, v in all_aucs.items()}

        def gap_stats(a, b):
            gap = arrs[a] - arrs[b]
            stat, p = wilcoxon(arrs[a], arrs[b])
            return {"mean": float(gap.mean()), "std": float(gap.std()),
                    "wilcoxon_p": float(p), "positive_seeds": int((gap > 0).sum())}

        gaps = {
            "leaky_minus_clean_matched": gap_stats("leaky", "clean_matched"),
            "leaky_minus_clean_matched_adaptive": gap_stats("leaky", "clean_matched_adaptive"),
            "clean_matched_adaptive_minus_placebo": gap_stats("clean_matched_adaptive", "placebo"),
            "clean_matched_minus_clean_matched_adaptive": gap_stats("clean_matched", "clean_matched_adaptive"),
        }

        print(f"\n--- HIDDEN={hidden} summary ---")
        for k, v in arrs.items():
            print(f"  {k:26s} AUROC: mean={v.mean():.4f} std={v.std():.4f}")
        for k, v in gaps.items():
            print(f"  gap[{k:40s}]: mean={v['mean']:+.4f} p={v['wilcoxon_p']:.4f} "
                  f"pos={v['positive_seeds']}/{N_SEEDS}")

        results_by_capacity[str(hidden)] = {
            "hidden": hidden,
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "gaps": gaps,
        }

    out = {
        "config": {"n_samples": N_SAMPLES, "feat_dim": FEAT_DIM,
                   "n_seeds": N_SEEDS, "n_inner_folds": N_INNER_FOLDS,
                   "epochs": EPOCHS, "capacities": CAPACITIES,
                   "target_auroc": TARGET_AUROC, "fisher_j_corrected": FISHER_J},
        "by_capacity": results_by_capacity,
    }
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "epoch_forcing_confound_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
