"""
Paper 2 — Bulletproofing round 2: resolve the adversarial-review confound.

An independent adversarial review of the capacity-sweep result (Tier-1
review, see review notes) identified two real problems:

1. "Monotonically" is factually wrong: the absolute LEAKY/CLEAN AUROCs
   wander non-monotonically across capacity levels (e.g. CLEAN at 128
   units is LOWER than CLEAN at 16 units); only the endpoint-ordered GAP
   trends, and every single level is individually non-significant
   (p >= 0.105 at n=10 seeds). "7x larger effect" is 7x a number
   indistinguishable from zero.

2. A genuine unaddressed confound: does the LEAKY-CLEAN gap growing with
   capacity reflect real label-peeking severity, or just generic
   checkpoint-selection VARIANCE inflating with capacity (over-parameterized
   models have noisier per-epoch validation AUC regardless of whether the
   selection criterion contains any real signal at all)? The gap's own std
   already grows with capacity (0.0063 at 16u -> 0.0093 at 384u) -- exactly
   what a pure-variance confound would produce.

This script resolves both with a single addition: a PLACEBO condition
where checkpoint selection uses the SAME held-out fold as LEAKY (same
size, same held-out indices, same capacity, same everything) but the
selection AUC is computed against RANDOMLY PERMUTED labels -- i.e. a
selection criterion with a real (identical) variance profile but ZERO
real signal. If LEAKY beats PLACEBO by about as much as it beats CLEAN,
the capacity effect is genuine label-peeking. If LEAKY is statistically
indistinguishable from PLACEBO, the entire "leak severity" story is a
capacity-driven-variance artifact, not real. Run at hidden=384 only
(the flagship capacity) with 100 seeds (vs. the original 10) to actually
resolve significance rather than leave the paper's centerpiece at
p=0.105/n=10.
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
N_SEEDS = 100          # 10x the original capacity-sweep run, to actually resolve significance
HIDDEN = 384            # flagship capacity only -- matches MultiHaluDet's real config


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
    (X_sel, y_sel_for_selection). For LEAKY/CLEAN, y_sel_for_selection is
    the TRUE label of the selection split. For PLACEBO, it is a PERMUTED
    version of those same true labels -- same variance profile, zero
    signal."""
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
        try:
            auc = roc_auc_score(y_sel_for_selection, probs)
        except ValueError:
            auc = 0.5  # degenerate (all-one-class permutation draw); treat as chance
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        return model.features(torch.tensor(X, dtype=torch.float32)).numpy()


def run_one_seed(seed):
    X, y = make_synthetic_data(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(seed + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = HIDDEN // 2
    oof_leaky = np.zeros((len(y_train), feat_dim_out))
    oof_clean = np.zeros((len(y_train), feat_dim_out))
    oof_placebo = np.zeros((len(y_train), feat_dim_out))
    test_feat_leaky = np.zeros((len(y_test), feat_dim_out))
    test_feat_clean = np.zeros((len(y_test), feat_dim_out))
    test_feat_placebo = np.zeros((len(y_test), feat_dim_out))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        # LEAKY: select on val_idx's TRUE labels (same fold reused downstream)
        model_leaky = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            HIDDEN, EPOCHS, fold_seed,
        )
        oof_leaky[val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat_leaky += extract_features(model_leaky, X_test)

        # CLEAN: select on a disjoint carve-out's TRUE labels
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            HIDDEN, EPOCHS, fold_seed,
        )
        oof_clean[val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat_clean += extract_features(model_clean, X_test)

        # PLACEBO: select on val_idx's PERMUTED labels (same held-out indices/size/
        # capacity as LEAKY, zero real signal in the selection criterion)
        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted,
            HIDDEN, EPOCHS, fold_seed,
        )
        oof_placebo[val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat_placebo += extract_features(model_placebo, X_test)

    test_feat_leaky /= N_INNER_FOLDS
    test_feat_clean /= N_INNER_FOLDS
    test_feat_placebo /= N_INNER_FOLDS

    clf_leaky = LogisticRegression(max_iter=2000).fit(oof_leaky, y_train)
    auc_leaky = roc_auc_score(y_test, clf_leaky.predict_proba(test_feat_leaky)[:, 1])
    clf_clean = LogisticRegression(max_iter=2000).fit(oof_clean, y_train)
    auc_clean = roc_auc_score(y_test, clf_clean.predict_proba(test_feat_clean)[:, 1])
    clf_placebo = LogisticRegression(max_iter=2000).fit(oof_placebo, y_train)
    auc_placebo = roc_auc_score(y_test, clf_placebo.predict_proba(test_feat_placebo)[:, 1])
    return auc_leaky, auc_clean, auc_placebo


def main():
    print(f"Calibration: target AUROC={TARGET_AUROC} -> J={FISHER_J:.3f} -> CLASS_SEP={CLASS_SEP:.4f}")
    print(f"Running N_SEEDS={N_SEEDS} at HIDDEN={HIDDEN} (flagship capacity, 10x original seed count)")
    leaky_aucs, clean_aucs, placebo_aucs = [], [], []
    for seed in range(N_SEEDS):
        al, ac, ap = run_one_seed(seed)
        leaky_aucs.append(al)
        clean_aucs.append(ac)
        placebo_aucs.append(ap)
        if (seed + 1) % 10 == 0:
            print(f"  [{seed+1}/{N_SEEDS}] leaky={al:.4f} clean={ac:.4f} placebo={ap:.4f}")

    leaky_aucs = np.array(leaky_aucs)
    clean_aucs = np.array(clean_aucs)
    placebo_aucs = np.array(placebo_aucs)

    gap_lc = leaky_aucs - clean_aucs        # original comparison
    gap_lp = leaky_aucs - placebo_aucs      # NEW: leaky vs. pure-noise-floor placebo
    gap_cp = clean_aucs - placebo_aucs      # sanity: clean should ALSO beat pure noise somewhat

    stat_lc, p_lc = wilcoxon(leaky_aucs, clean_aucs)
    stat_lp, p_lp = wilcoxon(leaky_aucs, placebo_aucs)
    stat_cp, p_cp = wilcoxon(clean_aucs, placebo_aucs)

    print("\n=== Results at HIDDEN=384, N_SEEDS=100 ===")
    print(f"LEAKY   AUROC: mean={leaky_aucs.mean():.4f} std={leaky_aucs.std():.4f}")
    print(f"CLEAN   AUROC: mean={clean_aucs.mean():.4f} std={clean_aucs.std():.4f}")
    print(f"PLACEBO AUROC: mean={placebo_aucs.mean():.4f} std={placebo_aucs.std():.4f}")
    print(f"\nGap LEAKY-CLEAN:   mean={gap_lc.mean():+.4f} std={gap_lc.std():.4f} p={p_lc:.4f} (positive in {int((gap_lc>0).sum())}/{N_SEEDS})")
    print(f"Gap LEAKY-PLACEBO: mean={gap_lp.mean():+.4f} std={gap_lp.std():.4f} p={p_lp:.4f} (positive in {int((gap_lp>0).sum())}/{N_SEEDS})")
    print(f"Gap CLEAN-PLACEBO: mean={gap_cp.mean():+.4f} std={gap_cp.std():.4f} p={p_cp:.4f} (positive in {int((gap_cp>0).sum())}/{N_SEEDS})")

    print("\n=== Interpretation ===")
    if p_lc < 0.05:
        print(f"LEAKY-CLEAN gap IS significant at n={N_SEEDS} (p={p_lc:.4f}) -- the original")
        print("capacity-sweep finding (non-significant at n=10) is REAL, just underpowered before.")
    else:
        print(f"LEAKY-CLEAN gap remains non-significant even at n={N_SEEDS} (p={p_lc:.4f}) --")
        print("the capacity-sweep 'growing severity' claim does NOT survive a proper power check.")

    if p_lp < 0.05 and gap_lp.mean() > gap_cp.mean() * 1.5:
        print(f"LEAKY beats the pure-noise PLACEBO by more than CLEAN does (gap_lp={gap_lp.mean():+.4f} vs")
        print(f"gap_cp={gap_cp.mean():+.4f}) -- genuine label-peeking confirmed, not just capacity variance.")
    elif abs(gap_lp.mean() - gap_cp.mean()) < 0.001 and p_lp > 0.05:
        print("LEAKY is statistically indistinguishable from the pure-noise PLACEBO, and CLEAN")
        print("does about as well against PLACEBO as LEAKY does -- CONFOUND CONFIRMED: the")
        print("LEAKY-CLEAN gap is consistent with generic capacity-driven checkpoint-selection")
        print("variance, not demonstrated label-peeking severity. This mechanism's claimed")
        print("severity growth should be substantially walked back in the paper.")
    else:
        print("Mixed result -- report all three gaps plainly, do not force a clean narrative.")

    import json
    from pathlib import Path
    out = {
        "config": {"n_samples": N_SAMPLES, "feat_dim": FEAT_DIM, "hidden": HIDDEN,
                   "n_seeds": N_SEEDS, "n_inner_folds": N_INNER_FOLDS, "epochs": EPOCHS},
        "leaky_aucs": leaky_aucs.tolist(), "clean_aucs": clean_aucs.tolist(),
        "placebo_aucs": placebo_aucs.tolist(),
        "gap_leaky_clean": {"mean": float(gap_lc.mean()), "std": float(gap_lc.std()),
                             "wilcoxon_p": float(p_lc), "positive_seeds": int((gap_lc > 0).sum())},
        "gap_leaky_placebo": {"mean": float(gap_lp.mean()), "std": float(gap_lp.std()),
                              "wilcoxon_p": float(p_lp), "positive_seeds": int((gap_lp > 0).sum())},
        "gap_clean_placebo": {"mean": float(gap_cp.mean()), "std": float(gap_cp.std()),
                              "wilcoxon_p": float(p_cp), "positive_seeds": int((gap_cp > 0).sum())},
    }
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "placebo_and_power_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'placebo_and_power_check.json'}")


if __name__ == "__main__":
    main()
