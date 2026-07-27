"""
Paper 2 — Controlled synthetic ablation of the MultiHaluDet-style leakage mechanism.

WHY THIS EXISTS
---------------
MultiHaluDet's released code (run_pipeline.py::stage_2_3_train_oof +
src/training/trainer.py::train_deep_model_fold) has a verified pattern:
inside each inner CV fold, a deep model is trained on tr_idx, and the
CHECKPOINT KEPT is whichever epoch maximizes AUROC on val_idx (early
stopping / best-epoch selection). That same checkpoint's features on
val_idx are then stored as the "out-of-fold" features fed to the
downstream meta-learner. This means the "held-out" fold's features come
from a model that was explicitly selected because it performs well on
that exact fold's labels -- a real leakage channel, distinct from (but in
the same family as) the classic "probe fit on all labels, scored on all
labels" bug found in HaRP.

Re-running MultiHaluDet's actual pipeline requires a 7B-parameter model
(Mistral-7B or Llama-2-7B) on real HaluEval/TriviaQA data -- not feasible
on this machine, and not worth spending the Kaggle GPU budget already
committed to Paper 3. Instead, this script isolates the MECHANISM itself
in a controlled, synthetic setting: same training loop shape (mini-batch
SGD, per-epoch validation AUC, keep-the-best-checkpoint, extract downstream
features from that checkpoint), same CV structure (5-fold inner, held-out
outer test), applied to synthetic two-class Gaussian data where the TRUE
achievable AUROC is known by construction. This lets us report a clean,
repeated-seed estimate of exactly how much AUROC inflation this checkpoint-
selection-on-the-same-fold pattern causes, as a general, citable number --
independent of any one paper's specific dataset.

Two conditions, otherwise identical:
  LEAKY: best-checkpoint selected by AUC on val_idx (the same fold whose
         features get used downstream) -- reproduces MultiHaluDet's code.
  CLEAN: best-checkpoint selected by AUC on a separate early-stopping
         split carved out of tr_idx (never touches val_idx labels) --
         the honest fix, output features for val_idx are still extracted
         from the resulting model, but that model was never tuned using
         val_idx's own labels.

Repeated across many random seeds to get a distribution of the leaky-minus-
clean AUROC gap, not just a single point estimate.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 700          # matches typical N in this literature (HaRP, MultiHaluDet-scale)
FEAT_DIM = 64            # synthetic "deep embedding" dimensionality
N_INNER_FOLDS = 5

# Calibrate class separation via GEOM-PROOF's own closed-form result
# (AUROC ~ Phi(sqrt(J)/2), Fisher ratio J, equal-covariance Gaussians -- see
# geom-proof/src/fisher.py / paper/main.tex). We invert it to target a
# REALISTIC, non-saturated probe AUROC (0.80, matching HaRP's 0.775-0.804
# range) rather than guessing a separation constant -- a saturated 1.0
# AUROC synthetic task leaves no room for a leakage bug to show inflation,
# which is exactly the calibration mistake the first version of this
# script made (CLASS_SEP=1.1 -> J~64x too large -> AUROC=1.0000 for both
# conditions, gap undetectable by construction).
TARGET_AUROC = 0.80
FISHER_J = (2 * norm.ppf(TARGET_AUROC)) ** 2          # invert AUROC = Phi(sqrt(J)/2)
CLASS_SEP = float(np.sqrt(FISHER_J / FEAT_DIM))        # per-dim mean gap, spread evenly, unit variance
print(f"Calibration: target AUROC={TARGET_AUROC} -> Fisher J={FISHER_J:.3f} -> "
      f"CLASS_SEP={CLASS_SEP:.4f} (per-dim mean separation across {FEAT_DIM} dims)")
TEST_SIZE = 0.20
EPOCHS = 45              # matches MultiHaluDet's config.epochs
ES_HOLD_FRACTION = 0.15  # fraction of tr_idx carved out for the CLEAN early-stopping split
N_SEEDS = 20


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=48):
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
        return h  # penultimate-layer embedding, matches "extract_features" pattern


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


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel, epochs, seed):
    """Train on (X_tr, y_tr); after every epoch, score on (X_sel, y_sel) and
    keep whichever epoch's weights maximize AUC there. Mirrors
    MultiHaluDet's train_deep_model_fold best-checkpoint-by-val-AUC logic
    exactly, parameterized by which split plays the "selection" role."""
    torch.manual_seed(seed)
    model = TinyMLP(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)

    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = crit(out, yt)
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


def run_one_seed(seed):
    X, y = make_synthetic_data(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)

    feat_dim_out = 24  # TinyMLP.features() output dim (hidden//2)
    oof_leaky = np.zeros((len(y_train), feat_dim_out))
    oof_clean = np.zeros((len(y_train), feat_dim_out))
    test_feat_leaky = np.zeros((len(y_test), feat_dim_out))
    test_feat_clean = np.zeros((len(y_test), feat_dim_out))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        # ---- LEAKY: checkpoint selected using val_idx's own labels ----
        model_leaky = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx],
            X_train[val_idx], y_train[val_idx],  # <- selection split == the "held-out" fold
            EPOCHS, fold_seed,
        )
        oof_leaky[val_idx] = extract_features(model_leaky, X_train[val_idx])
        test_feat_leaky += extract_features(model_leaky, X_test)

        # ---- CLEAN: checkpoint selected using a separate carve-out, never val_idx ----
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx],
            X_train[es_idx], y_train[es_idx],   # <- selection split excludes val_idx entirely
            EPOCHS, fold_seed,
        )
        oof_clean[val_idx] = extract_features(model_clean, X_train[val_idx])
        test_feat_clean += extract_features(model_clean, X_test)

    test_feat_leaky /= N_INNER_FOLDS
    test_feat_clean /= N_INNER_FOLDS

    # Downstream meta-learner: plain logistic regression on the OOF features (matches
    # MultiHaluDet's stage_4_ensemble spirit, minus the full stacking complexity).
    clf_leaky = LogisticRegression(max_iter=2000).fit(oof_leaky, y_train)
    auc_leaky = roc_auc_score(y_test, clf_leaky.predict_proba(test_feat_leaky)[:, 1])

    clf_clean = LogisticRegression(max_iter=2000).fit(oof_clean, y_train)
    auc_clean = roc_auc_score(y_test, clf_clean.predict_proba(test_feat_clean)[:, 1])

    return auc_leaky, auc_clean


def main():
    leaky_aucs, clean_aucs = [], []
    for seed in range(N_SEEDS):
        auc_leaky, auc_clean = run_one_seed(seed)
        leaky_aucs.append(auc_leaky)
        clean_aucs.append(auc_clean)
        print(f"seed={seed:2d}  leaky={auc_leaky:.4f}  clean={auc_clean:.4f}  "
              f"gap={auc_leaky - auc_clean:+.4f}")

    leaky_aucs = np.array(leaky_aucs)
    clean_aucs = np.array(clean_aucs)
    gaps = leaky_aucs - clean_aucs

    stat, p = wilcoxon(leaky_aucs, clean_aucs)

    print("\n=== Summary over", N_SEEDS, "seeds ===")
    print(f"LEAKY AUROC: mean={leaky_aucs.mean():.4f} std={leaky_aucs.std():.4f}")
    print(f"CLEAN AUROC: mean={clean_aucs.mean():.4f} std={clean_aucs.std():.4f}")
    print(f"Gap (leaky - clean): mean={gaps.mean():+.4f} std={gaps.std():.4f} "
          f"min={gaps.min():+.4f} max={gaps.max():+.4f}")
    print(f"Wilcoxon signed-rank test: stat={stat:.2f} p={p:.6f}")
    print(f"Positive gap in {int((gaps > 0).sum())}/{N_SEEDS} seeds")

    import json
    out = {
        "config": {
            "n_samples": N_SAMPLES, "feat_dim": FEAT_DIM, "class_sep": CLASS_SEP,
            "n_inner_folds": N_INNER_FOLDS, "epochs": EPOCHS, "n_seeds": N_SEEDS,
        },
        "leaky_aucs": leaky_aucs.tolist(),
        "clean_aucs": clean_aucs.tolist(),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "positive_gap_seeds": int((gaps > 0).sum()),
    }
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_leakage_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'synthetic_leakage_ablation.json'}")


if __name__ == "__main__":
    main()
