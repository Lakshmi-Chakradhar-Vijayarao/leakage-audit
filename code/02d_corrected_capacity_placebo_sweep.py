"""
Paper 2 -- Bulletproofing round 3: fix the calibration bug AND the training-
budget confound found by the second adversarial review round, and rerun the
capacity sweep (not just the single flagship capacity) at adequate power.

THREE BUGS THIS SCRIPT FIXES (each independently confirmed against raw data
before writing this script -- see chat/session notes, not reproduced here):

1. CALIBRATION FORMULA BUG. 02c_placebo_and_power_check.py line 48 computed
   FISHER_J = (2 * norm.ppf(TARGET_AUROC)) ** 2, which is the inversion of
   the WRONG AUROC identity AUROC = Phi(sqrt(J)/2) (this is actually the
   equal-prior Bayes-ACCURACY formula, not AUROC -- the same bug independently
   found and fixed in the companion GEOM-PROOF paper's Proposition 1). The
   correct binormal AUROC identity is AUROC = Phi(sqrt(J/2)), whose correct
   inversion is FISHER_J = 2 * (norm.ppf(TARGET_AUROC)) ** 2. Using the old
   formula, J=2.833 was intended to calibrate an AUROC=0.80 task; under the
   CORRECT formula, J=2.833 actually gives a Bayes-optimal AUROC of 0.883,
   not 0.80. This is independently confirmed by the empirical LEAKY/CLEAN
   AUROCs observed in the original run, which cluster at ~0.87-0.88, not
   ~0.80. This script uses the corrected inversion (J=1.417 for a genuine
   0.80-AUROC task).

2. THE PLACEBO SCRIPT'S OWN DECISION RULE RETURNED "MIXED," NOT "CONFIRMED."
   Plugging the actual 02c output into its own pre-registered branching logic
   (line 235: p_lp < 0.05 AND gap_lp.mean() > gap_cp.mean() * 1.5) gives
   0.01278 !> 0.01597 -- False. The paper's text asserted "genuine
   label-peeking confirmed" despite the script's own logic landing in the
   "mixed -- do not force a narrative" branch. This script reruns the
   comparison under the CORRECTED calibration and lets the same
   pre-registered rule speak for itself on the new numbers.

3. TRAINING-BUDGET CONFOUND. LEAKY and PLACEBO train on the full tr_idx
   (~448 samples at hidden=384); CLEAN trains on tr2_idx = tr_idx minus a
   15% carve-out (~381 samples) -- so LEAKY-CLEAN previously conflated the
   leak with an ~18% larger training set for LEAKY. This script adds a
   CLEAN_MATCHED condition: select the best epoch count e* via early-stopping
   AUC on a disjoint held-out fold (same as CLEAN), then RETRAIN FROM SCRATCH
   on the FULL tr_idx for exactly e* epochs (no selection during the
   retrain). CLEAN_MATCHED gets the same training-data budget as LEAKY while
   still choosing its epoch count from a fold LEAKY never sees.

ALSO: reruns across the FULL capacity sweep (16/48/128/384), not just the
flagship 384, at N_SEEDS=100 throughout -- directly answering the review's
"re-run the full sweep at n=100 or delete the capacity-dependence claim"
demand, instead of powering up only the endpoint.

FOURTH FIX (added later, after an independent review pointed out that
code/47's seed-decoupling was never retrofitted to this script -- the
paper's PRIMARY severity harness): SEED DECOUPLING. Every earlier version
of this script drove the synthetic data generation, the outer train/test
split AND the inner StratifiedKFold fold assignment from one shared `seed`
variable, so across the 100 replicates those three sources of variation were
perfectly confounded: no two replicates share a dataset with a different
split, and no two share a split with a different fold assignment. That is
the exact confound `code/47_selection_multiplicity_sweep.py` introduced
`data_seed`/`split_seed`/`fold_seed_base`/`init_seed_base` to fix, and it is
now retrofitted here using code/47's identical scheme:

    data_seed      = i            (which synthetic sample is drawn)
    split_seed     = i + 100000   (the outer train/test partition)
    fold_seed_base = i + 200000   (inner CV fold assignment + placebo RNG)
    init_seed_base = i + 300000   (torch init and the ES carve-out split)

so no single draw can produce the data, the split and the folds together.
See `seeds_for()` below. The `--coupled-seed` flag reproduces the previous,
confounded behaviour bit-for-bit so the two can be compared directly; the
previous run's output is retained as
`results/corrected_capacity_placebo_sweep_coupled_seed_legacy.json`.
"""
import json
import sys
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

# ── FIX 1: corrected binormal AUROC identity inversion ──────────────────────
# AUROC = Phi(sqrt(J/2))  =>  J = 2 * Phi^-1(AUROC)^2
FISHER_J = 2 * (norm.ppf(TARGET_AUROC)) ** 2
CLASS_SEP = float(np.sqrt(FISHER_J / FEAT_DIM))
print(f"[calibration] TARGET_AUROC={TARGET_AUROC} -> J={FISHER_J:.4f} "
      f"(corrected; old buggy formula gave J=2.833, actual AUROC=0.883) "
      f"-> CLASS_SEP={CLASS_SEP:.4f}")
print(f"[calibration] Sanity check: Phi(sqrt(J/2)) = {norm.cdf(np.sqrt(FISHER_J/2)):.4f} "
      f"(should equal {TARGET_AUROC})")

TEST_SIZE = 0.20
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
N_SEEDS = 100
CAPACITIES = [16, 48, 128, 384]

# ── FIX 4: seed decoupling (code/47's scheme, retrofitted here) ─────────────
# COUPLED_SEEDS=True restores the previous, confounded single-`seed` behaviour
# bit-for-bit (`--coupled-seed` on the command line) so the two runs can be
# compared directly rather than asserted to agree.
COUPLED_SEEDS = "--coupled-seed" in sys.argv


def seeds_for(i):
    """Decouple replicate `i` into four independent seed streams.

    Identical to code/47_selection_multiplicity_sweep.py's scheme, which was
    written specifically to fix this confound and was never retrofitted here
    until now. Returns (data_seed, split_seed, fold_seed_base, init_seed_base).
    """
    if COUPLED_SEEDS:
        return i, i, i, i
    return i, i + 100000, i + 200000, i + 300000


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
    """Train on (X_tr, y_tr); keep whichever epoch maximizes AUC on
    (X_sel, y_sel_for_selection). Returns (model, best_epoch_index)."""
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
    """Train for exactly n_epochs with NO selection -- used for CLEAN_MATCHED's
    full-budget retrain once the epoch count has been chosen elsewhere."""
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


def run_one_seed(data_seed, split_seed, fold_seed_base, init_seed_base, hidden):
    X, y = make_synthetic_data(data_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(fold_seed_base + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=fold_seed_base)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in
           ["leaky", "clean", "clean_matched", "placebo"]}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in
                 ["leaky", "clean", "clean_matched", "placebo"]}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # torch init and the ES carve-out split get their own stream, so a
        # replicate's initialization no longer co-varies with its data draw.
        fold_seed = init_seed_base * 100 + fold

        # LEAKY: select on val_idx's TRUE labels (same fold reused downstream);
        # trains on the FULL tr_idx.
        model_leaky, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["leaky"][val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat["leaky"] += extract_features(model_leaky, X_test)

        # CLEAN (original, budget-confounded): select on a disjoint carve-out's
        # TRUE labels; trains on tr2_idx only (~85% of tr_idx).
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean, best_epoch = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed,
        )
        oof["clean"][val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat["clean"] += extract_features(model_clean, X_test)

        # CLEAN_MATCHED (budget-fixed): SAME epoch-selection procedure as CLEAN
        # (disjoint es_idx decides best_epoch), but then retrain from scratch
        # on the FULL tr_idx for exactly best_epoch epochs -- same training
        # budget as LEAKY, epoch count still chosen from a fold LEAKY never sees.
        model_clean_matched = train_fixed_epochs(
            X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed,
        )
        oof["clean_matched"][val_idx] = extract_features(model_clean_matched, X_train[val_idx])
        test_feat["clean_matched"] += extract_features(model_clean_matched, X_test)

        # PLACEBO: select on val_idx's PERMUTED labels (same held-out indices/
        # size/capacity/training-budget as LEAKY). Training itself still uses
        # the real, unpermuted y_train[tr_idx] -- this is zero SELECTION-signal,
        # not zero signal overall, which is why placebo AUROC sits ~0.73-0.74,
        # not 0.5. Only the checkpoint-selection criterion is permuted.
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
        print(f"\n{'='*70}")
        print(f"HIDDEN={hidden}  N_SEEDS={N_SEEDS}")
        print(f"{'='*70}")
        all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
        for seed in range(N_SEEDS):
            aucs = run_one_seed(*seeds_for(seed), hidden)
            for k, v in aucs.items():
                all_aucs[k].append(v)
            if (seed + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{seed+1}/{N_SEEDS}] elapsed={elapsed:.0f}s  "
                      f"leaky={aucs['leaky']:.4f} clean={aucs['clean']:.4f} "
                      f"clean_matched={aucs['clean_matched']:.4f} placebo={aucs['placebo']:.4f}")

        arrs = {k: np.array(v) for k, v in all_aucs.items()}

        def gap_stats(a, b):
            gap = arrs[a] - arrs[b]
            stat, p = wilcoxon(arrs[a], arrs[b])
            return {"mean": float(gap.mean()), "std": float(gap.std()),
                    "wilcoxon_p": float(p), "positive_seeds": int((gap > 0).sum())}

        gaps = {
            "leaky_minus_clean": gap_stats("leaky", "clean"),
            "leaky_minus_clean_matched": gap_stats("leaky", "clean_matched"),
            "leaky_minus_placebo": gap_stats("leaky", "placebo"),
            "clean_minus_placebo": gap_stats("clean", "placebo"),
            "clean_matched_minus_placebo": gap_stats("clean_matched", "placebo"),
        }

        print(f"\n--- HIDDEN={hidden} summary ---")
        for k, v in arrs.items():
            print(f"  {k:15s} AUROC: mean={v.mean():.4f} std={v.std():.4f}")
        for k, v in gaps.items():
            print(f"  gap[{k:28s}]: mean={v['mean']:+.4f} p={v['wilcoxon_p']:.4f} "
                  f"pos={v['positive_seeds']}/{N_SEEDS}")

        # Pre-registered decision rule (unchanged from 02c), applied to the
        # BUDGET-MATCHED comparison this time.
        gap_lp = gaps["leaky_minus_placebo"]
        gap_cmp = gaps["clean_matched_minus_placebo"]
        if gap_lp["wilcoxon_p"] < 0.05 and gap_lp["mean"] > gap_cmp["mean"] * 1.5:
            verdict = "GENUINE_LEAK_CONFIRMED"
        elif abs(gap_lp["mean"] - gap_cmp["mean"]) < 0.001 and gap_lp["wilcoxon_p"] > 0.05:
            verdict = "CONFOUND_CONFIRMED_NO_REAL_LEAK"
        else:
            verdict = "MIXED"
        print(f"  VERDICT (budget-matched): {verdict}")

        results_by_capacity[str(hidden)] = {
            "hidden": hidden,
            "aucs": {k: v.tolist() for k, v in arrs.items()},
            "gaps": gaps,
            "verdict_budget_matched": verdict,
        }

    out = {
        "config": {"n_samples": N_SAMPLES, "feat_dim": FEAT_DIM,
                   "n_seeds": N_SEEDS, "n_inner_folds": N_INNER_FOLDS,
                   "epochs": EPOCHS, "capacities": CAPACITIES,
                   "target_auroc": TARGET_AUROC, "fisher_j_corrected": FISHER_J,
                   "class_sep": CLASS_SEP,
                   "calibration_sanity_check_auroc": float(norm.cdf(np.sqrt(FISHER_J/2))),
                   "seed_scheme": "coupled_legacy" if COUPLED_SEEDS else "decoupled",
                   "seed_scheme_note": (
                       "decoupled: data_seed=i, split_seed=i+100000, "
                       "fold_seed_base=i+200000, init_seed_base=i+300000 (code/47's "
                       "scheme, retrofitted here). coupled_legacy: a single i drove "
                       "data generation, the outer split and the fold assignment "
                       "simultaneously -- retained only for comparison.")},
        "by_capacity": results_by_capacity,
    }
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("corrected_capacity_placebo_sweep_coupled_seed_legacy.json"
                          if COUPLED_SEEDS else "corrected_capacity_placebo_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
