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

──────────────────────────────────────────────────────────────────────────────
CORRECTION (a later independent adversarial review; this revision). THE CONTROL
DESCRIBED ABOVE DID NOT DO WHAT IT SAYS. `es_idx` was carved OUT OF `tr_idx`
via train_test_split(tr_idx, ...), but the arm was then TRAINED ON THE FULL
`tr_idx` -- so every point the "held-out" selection fold contained was inside
that arm's own training set the whole time. Selecting a checkpoint on data the
model is concurrently fitting is not honest selection: in-sample AUROC is
near-monotone in training, so the argmax simply runs to the end of the budget.
The signature is visible in the arm's own kept-checkpoint epoch, which this
script did not previously record: under the broken construction the arm's mean
best epoch is ~38.6 of 45 (near-total convergence), versus ~20.5-20.7 for the
genuinely honest arms. The control therefore measured "train to convergence,"
not "adaptive selection against an honest signal," and could not test the
alternative explanation it was built to test. Any conclusion drawn from it --
including the paper's previous "this alternative is directionally unsupported"
-- was unsupported in turn.

THE FIX mirrors `code/43`'s construction, which already separates these
correctly: there, the arm whose checkpoint is selected on `es_idx` TRAINS ON
`tr2_idx`, the complement of `es_idx` within `tr_idx`, so training data and
selection data are disjoint by construction. CLEAN_MATCHED_ADAPTIVE now does
the same: it trains on `tr2_idx` and selects on `es_idx`. This costs a
disclosed ~15% reduction in training-set size relative to LEAKY
(ES_HOLD_FRACTION=0.15) rather than an exact data-budget match -- the same
trade this paper's other honest-selection arms already make, and the same one
`code/49` documents. An exact budget match and an honest selection signal
cannot both be had within a single fold: the selection points must come from
somewhere.

BOTH ARMS ARE RUN AND BOTH ARE REPORTED. The broken construction is retained
under the explicit name `clean_matched_adaptive_contaminated` so the size of
the artifact is a shipped number rather than an assertion, and so the paper's
correction history is reproducible. `clean_matched_adaptive` now refers to the
corrected, disjoint arm, and is the only one any reported conclusion rests on.
Mean kept-checkpoint epoch is now recorded for every arm (`best_epoch_stats`),
which is the diagnostic that exposes this class of defect directly.
──────────────────────────────────────────────────────────────────────────────
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
    conditions = ["leaky", "clean_matched", "clean_matched_adaptive",
                  "clean_matched_adaptive_contaminated", "placebo"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}
    best_epochs = {k: [] for k in conditions}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        # LEAKY: adaptive selection on val_idx's TRUE labels (the same fold
        # reused downstream as out-of-fold features). Full training budget.
        model_leaky, be_leaky = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["leaky"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky"] += extract_features(model_leaky, X_test)
        best_epochs["leaky"].append(be_leaky)

        # es_idx: disjoint honest carve-out, used for selection only, never
        # reused downstream -- same construction CLEAN/CLEAN_MATCHED use.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )

        # CLEAN_MATCHED (existing, from 02d): select best_epoch on the
        # smaller-budget tr2_idx/es_idx run, then blindly retrain on the
        # full tr_idx for exactly that many epochs -- NO adaptive selection
        # during the full-budget run itself.
        model_cma, best_epoch = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        model_clean_matched = train_fixed_epochs(
            X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed,
        )
        oof["clean_matched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(model_clean_matched, X_test)
        best_epochs["clean_matched"].append(best_epoch)

        # CLEAN_MATCHED_ADAPTIVE (CORRECTED). Ongoing adaptive checkpoint
        # selection during the run, exactly like LEAKY -- but selecting against
        # es_idx's true labels, and TRAINING ON tr2_idx, the complement of
        # es_idx within tr_idx, so selection data and training data are
        # DISJOINT. This mirrors code/43's construction (its `model_clean`
        # trains on tr2_idx and selects on es_idx) and is the same model
        # object CLEAN_MATCHED already derives its epoch count from, so it
        # costs no additional training. Isolates fold reuse from "having
        # adaptive selection" at a disclosed ~15% training-budget cost.
        oof["clean_matched_adaptive"][val_idx] = extract_features(model_cma, X_train[val_idx])
        test_feat["clean_matched_adaptive"] += extract_features(model_cma, X_test)
        best_epochs["clean_matched_adaptive"].append(best_epoch)

        # CLEAN_MATCHED_ADAPTIVE_CONTAMINATED (the superseded construction,
        # retained as documentary evidence -- see the docstring's CORRECTION).
        # Trains on the FULL tr_idx while selecting on es_idx, which is a
        # SUBSET of tr_idx: the selection signal is in-sample, so the argmax
        # runs to the end of the budget and the arm measures convergence, not
        # honest selection. No reported conclusion rests on it.
        model_cma_contam, be_contam = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["clean_matched_adaptive_contaminated"][val_idx] = extract_features(
            model_cma_contam, X_train[val_idx])
        test_feat["clean_matched_adaptive_contaminated"] += extract_features(model_cma_contam, X_test)
        best_epochs["clean_matched_adaptive_contaminated"].append(be_contam)

        # PLACEBO: adaptive selection on val_idx's PERMUTED labels, full budget.
        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo, be_plc = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted,
            hidden, EPOCHS, fold_seed,
        )
        oof["placebo"][val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat["placebo"] += extract_features(model_placebo, X_test)
        best_epochs["placebo"].append(be_plc)

    aucs = {}
    for k in oof:
        test_feat[k] /= N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    return aucs, {k: float(np.mean(v)) for k, v in best_epochs.items()}


def main():
    t0 = time.time()
    results_by_capacity = {}

    for hidden in CAPACITIES:
        print(f"\n{'='*70}\nHIDDEN={hidden}  N_SEEDS={N_SEEDS}\n{'='*70}")
        conds = ["leaky", "clean_matched", "clean_matched_adaptive",
                 "clean_matched_adaptive_contaminated", "placebo"]
        all_aucs = {k: [] for k in conds}
        all_best_epochs = {k: [] for k in conds}
        for seed in range(N_SEEDS):
            aucs, be = run_one_seed(seed, hidden)
            for k, v in aucs.items():
                all_aucs[k].append(v)
            for k, v in be.items():
                all_best_epochs[k].append(v)
            if (seed + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{seed+1}/{N_SEEDS}] elapsed={elapsed:.0f}s  "
                      f"leaky={aucs['leaky']:.4f} clean_matched={aucs['clean_matched']:.4f} "
                      f"cma={aucs['clean_matched_adaptive']:.4f} "
                      f"cma_contam={aucs['clean_matched_adaptive_contaminated']:.4f} "
                      f"placebo={aucs['placebo']:.4f}", flush=True)

        arrs = {k: np.array(v) for k, v in all_aucs.items()}
        be_arrs = {k: np.array(v) for k, v in all_best_epochs.items()}

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
            "leaky_minus_clean_matched_adaptive_contaminated": gap_stats(
                "leaky", "clean_matched_adaptive_contaminated"),
            "clean_matched_adaptive_contaminated_minus_placebo": gap_stats(
                "clean_matched_adaptive_contaminated", "placebo"),
        }

        # Placebo-relative retention: what fraction of CLEAN_MATCHED's advantage
        # over PLACEBO does the adaptive-selection control recover? This is the
        # quantity SS4.3 quotes. Reported for BOTH the corrected arm and the
        # superseded contaminated one, so the size of the artifact is visible.
        def retention(arm):
            return float(100 * (arrs[arm].mean() - arrs["placebo"].mean())
                         / (arrs["clean_matched"].mean() - arrs["placebo"].mean()))

        retentions = {
            "clean_matched_adaptive": retention("clean_matched_adaptive"),
            "clean_matched_adaptive_contaminated": retention("clean_matched_adaptive_contaminated"),
        }

        # The diagnostic that exposes the defect the correction fixes: an arm
        # that selects its checkpoint on data it is concurrently training on
        # runs to the end of the budget, because in-sample AUROC is
        # near-monotone in training.
        best_epoch_stats = {
            "mean_best_epoch": {k: float(v.mean()) for k, v in be_arrs.items()},
            "sd_best_epoch": {k: float(v.std()) for k, v in be_arrs.items()},
            "epochs_budget": EPOCHS,
            "note": (
                "Mean epoch of the kept checkpoint, averaged over folds then seeds. "
                "clean_matched_adaptive_contaminated selects on es_idx while training on the "
                "full tr_idx, of which es_idx is a subset, so its argmax runs to near the end "
                "of the 45-epoch budget. The corrected clean_matched_adaptive arm trains on "
                "tr2_idx (disjoint from es_idx) and stops where an honest signal says to, in "
                "line with the other honest arms."),
        }

        print(f"\n--- HIDDEN={hidden} summary ---")
        for k, v in arrs.items():
            print(f"  {k:38s} AUROC: mean={v.mean():.4f} std={v.std():.4f}  "
                  f"mean_best_epoch={be_arrs[k].mean():.1f}/{EPOCHS}")
        for k, v in gaps.items():
            print(f"  gap[{k:50s}]: mean={v['mean']:+.4f} p={v['wilcoxon_p']:.4g} "
                  f"pos={v['positive_seeds']}/{N_SEEDS}")
        for k, v in retentions.items():
            print(f"  placebo-relative retention[{k:38s}] = {v:.1f}%")

        results_by_capacity[str(hidden)] = {
            "hidden": hidden,
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "gaps": gaps,
            "placebo_relative_retention_pct": retentions,
            "best_epoch_stats": best_epoch_stats,
        }

    out = {
        "config": {"n_samples": N_SAMPLES, "feat_dim": FEAT_DIM,
                   "n_seeds": N_SEEDS, "n_inner_folds": N_INNER_FOLDS,
                   "epochs": EPOCHS, "capacities": CAPACITIES,
                   "es_hold_fraction": ES_HOLD_FRACTION,
                   "target_auroc": TARGET_AUROC, "fisher_j_corrected": FISHER_J},
        "arm_definitions": {
            "leaky": "trains on tr_idx; checkpoint argmaxed on val_idx's true labels "
                     "(the fold later reused as OOF features).",
            "clean_matched": "epoch count selected on the tr2_idx/es_idx run, then blindly "
                             "retrained on the full tr_idx for that many epochs; no adaptive "
                             "selection during the full-budget run.",
            "clean_matched_adaptive": "CORRECTED. Trains on tr2_idx, checkpoint argmaxed on "
                                      "es_idx -- DISJOINT from its training set, mirroring "
                                      "code/43's construction. Costs ~15% training budget.",
            "clean_matched_adaptive_contaminated": "SUPERSEDED, retained as evidence only. "
                                                   "Trains on the full tr_idx while argmaxing "
                                                   "on es_idx, a SUBSET of tr_idx, so the "
                                                   "selection signal is in-sample.",
            "placebo": "trains on tr_idx; checkpoint argmaxed on val_idx's PERMUTED labels.",
        },
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
