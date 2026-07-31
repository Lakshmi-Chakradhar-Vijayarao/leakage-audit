"""
Paper 2 -- Mechanism 3 fidelity extension.

The existing severity harness (code/33, code/43) captures MultiHaluDet's
per-fold checkpoint-selection leak (best-val-AUC checkpoint kept) but not
two other things the *actual* vendored trainer does every epoch, reacting
to the same validation fold's AUC score
(`code/external/MultiHaluDet/src/training/trainer.py`, lines 76-149):
  1. `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`, stepped on
     that fold's val AUC every epoch (`scheduler.step(auc_score)`).
  2. Early stopping driven by a no-improvement counter over the same val
     AUC (`patience_counter >= config.patience` breaks training), where
     `config.patience = 15` (`src/config.py:19`) -- a DIFFERENT threshold
     from the scheduler's own internal `patience=3`.

This is a continuous, epoch-by-epoch reactive coupling to validation
feedback, not just a final argmax checkpoint pick -- a materially
different (and untested) leakage pathway from Mechanism 3 as already
quantified. This reruns the LEAKY vs. CLEAN_MATCHED vs. PLACEBO comparison
with this scheduler+patience mechanic ported in, on the same train-only-
calibrated real features (alpha read from code/43's output) at capacity 128, to see
whether it adds severity beyond the checkpoint-selection-only harness.

CORRECTION (independent adversarial review found this): an earlier version
of CLEAN_MATCHED_PLUS_LRSCHED discarded the model already trained with a
real ReduceLROnPlateau scheduler on a disjoint carve-out (tr2_idx/es_idx)
and retrained from scratch with `train_fixed_epochs` -- plain constant-LR
Adam, no scheduler at all -- merely matched to LEAKY's epoch *count*. That
confounded "reuses the leaky val fold as the scheduler's feedback signal"
with "has an adaptive LR/early-stopping mechanism at all," since LEAKY and
PLACEBO both had an active scheduler while the control did not. The fix:
CLEAN_MATCHED_PLUS_LRSCHED now IS that first model -- trained with the
same real scheduler+early-stopping mechanic, but fed by the disjoint
es_idx carve-out instead of the reused val_idx fold. This costs a small,
disclosed reduction in training-set size (tr2_idx is ~85% of tr_idx, the
ES_HOLD_FRACTION=0.15 carve-out) rather than an exact data-budget match,
in exchange for actually isolating the mechanism under test.

SECOND CORRECTION (a later independent adversarial review found this): this
script used a single constant, LR_PATIENCE=3, for BOTH the LR scheduler's
reduction threshold AND the early-stopping break -- i.e. it stopped training
after 3 non-improving epochs. The audited repository does not do that. In
`code/external/MultiHaluDet/src/training/trainer.py:76` the scheduler is
constructed with its own `patience=3`, while the early-stopping break at
line 149 tests `patience_counter >= config.patience` with
`config.patience = 15` (`src/config.py:19`). Stopping at 3 truncates every
run drastically and exaggerates how much of a difference the validation
signal makes. Fixed: LR_PATIENCE=3 for the scheduler (unchanged),
ES_PATIENCE=15 for the early-stopping break, counted separately.

THIRD CORRECTION (same review): the calibration transform this harness runs
on is now `apply_calibration_label_free` (code/43). The previous
`apply_calibration_train_only` estimated class means from train indices only
but still chose which class offset to subtract from each point -- test points
included -- by looking at that point's own label, a residual instance of the
Mechanism-1 pattern this paper audits for. The alpha used here is read from
code/43's output rather than hardcoded, so the two harnesses cannot drift.

FOURTH CORRECTION (a further independent adversarial review): calling this a
"fidelity extension" while silently differing from the audited trainer on
half a dozen optimizer/data-pipeline settings was itself an overclaim. That
review enumerated the gaps; the tractable ones are now ported, and the
remainder are enumerated explicitly here and in the paper (Appendix A, issue
14) rather than left undisclosed. Exactly what is and is not ported:

  PORTED FROM THE AUDITED REPO (config.py / trainer.py / run_pipeline.py):
    - learning_rate = 2e-4          (config.py:17; this script previously
                                     used 2e-3, a 10x mismatch)
    - AdamW                         (trainer.py:75; previously plain Adam)
    - batch_size = 28 mini-batching (config.py:15; previously full-batch,
                                     i.e. exactly one gradient step/epoch)
    - warmup_epochs = 5             (config.py:21; linear LR ramp over the
                                     first 5 epochs -- trainer.py:88-91 --
                                     during which the scheduler is NOT
                                     stepped: trainer.py:134 gates
                                     `scheduler.step` on
                                     `epoch >= config.warmup_epochs`)
    - grad_clip = 0.5               (config.py:30, trainer.py:113)
    - min_lr = 1e-7                 (config.py:20; previously 1e-5)
    - epochs = 45                   (config.py:16; previously 60)
    - RobustScaler on the input features (run_pipeline.py:81,85; previously
                                     StandardScaler)
    - StandardScaler on the deep OOF features before the meta-learner
                                    (run_pipeline.py:130; previously none)
    - and, from before this correction: ReduceLROnPlateau(mode='max',
      factor=0.5, patience=3) stepped on the reused val fold's AUROC,
      early-stopping at config.patience=15 on the same signal,
      best-val-AUROC checkpoint retention, weight_decay=6e-5, 5 inner folds,
      test_size=0.20.

  DELIBERATELY NOT PORTED (out of scope for this harness; each would change
  the objective or the model class, not just the optimizer schedule):
    - EMA of the weights (config.use_ema, decay 0.999) -- the audited trainer
      evaluates and checkpoints under the EMA shadow weights.
    - The composite loss: 0.45*BCE + 0.35*FocalLoss(gamma=2) +
      0.20*AsymmetricLoss(gamma_neg=3, gamma_pos=1, clip=0.03), plus a
      0.20-weighted ContrastiveLoss(temp=0.04) auxiliary term on the
      embedding. This harness uses plain BCEWithLogitsLoss.
    - BCE pos_weight class rebalancing (trainer.py:51,70) -- porting it alone,
      without the other three loss terms it is only 45% of, would not be
      more faithful than omitting it.
    - label_smoothing = 0.02.
    - mixup (alpha 0.15) and cutmix (alpha 0.15), each applied to ~33% of
      batches.
    - The model itself: the audited pipeline trains a 6-layer, 8-head,
      384-hidden multi-scale transformer (src/models/multihaludet.py); this
      harness trains the 3-layer MLP the rest of this paper's severity
      sweeps use, so that the fidelity extension is comparable to the
      checkpoint-selection-only harness it is meant to be contrasted with.
    - SWA is listed in config.py (use_swa, swa_start=30, swa_lr=5e-5) but is
      never referenced anywhere in the audited repository's Python sources,
      so there is nothing to port.

  This means "fidelity extension" here refers specifically to fidelity of
  the *validation-signal coupling and optimizer schedule*, not of the loss
  function or the model class. The quantity being measured is the
  incremental severity of continuous scheduler/early-stopping coupling over
  a final-checkpoint argmax, holding this paper's own harness architecture
  fixed; it is not an estimate of MultiHaluDet's own reported number's
  inflation, which would require their model and loss as well.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / os.environ.get("OUT_NAME", "mechanism3_fidelity_extension.json")
N_SEEDS_OVERRIDE = int(os.environ.get("N_SEEDS_OVERRIDE", 0))
CALIB_PATH = ROOT / "results" / "calibration_leakage_diagnostic.json"

N_SEEDS = N_SEEDS_OVERRIDE or 100
HIDDEN = 128
N_INNER_FOLDS = 5          # config.py:5   n_inner_folds
MAX_EPOCHS = 45            # config.py:16  epochs (was 60 before the 4th correction)
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20           # config.py:6   test_size
# --- ported optimizer/data-pipeline settings (4th correction; see docstring) ---
LEARNING_RATE = 2e-4       # config.py:17  (was 2e-3 -- a 10x mismatch)
BATCH_SIZE = 28            # config.py:15  (was full-batch: 1 step/epoch)
WARMUP_EPOCHS = 5          # config.py:21  (linear ramp; scheduler not stepped)
GRAD_CLIP = 0.5            # config.py:30
WEIGHT_DECAY = 6e-5        # config.py:18
# Read from code/43's output so this harness and the checkpoint-selection-only
# harness can never drift apart on the calibration strength.
ALPHA_TRAIN_ONLY = float(json.load(open(CALIB_PATH))["train_only_calibration_alpha"])
# MultiHaluDet uses TWO different patience thresholds against the same val
# signal. Keeping them distinct here is the point of the second correction
# documented in the module docstring.
LR_PATIENCE = 3    # trainer.py:76 -- ReduceLROnPlateau's own reduction threshold
# config.py:19 -- early-stopping break threshold. Overridable so the effect of
# the patience correction alone can be measured against the same calibration
# (see the ablation reported in Appendix A, issue 10); the paper's reported
# number always uses the repo-faithful value 15.
ES_PATIENCE = int(os.environ.get("ES_PATIENCE_OVERRIDE", 15))
LR_FACTOR = 0.5
MIN_LR = 1e-7              # config.py:20 (was 1e-5 before the 4th correction)
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


def apply_calibration_label_free(X, y, alpha, train_idx, seed=0):
    """Identical to code/43's `apply_calibration_label_free`. Axis u, midpoint c
    and pooled within-class SD along u are all estimated from TRAIN indices
    only; the map applied to every row consults no per-sample label:
        p' = alpha*p + sqrt(1-alpha^2)*s_w*eps,   p = <x-c,u>,  eps ~ N(0,1).
    Shrinks between-class separation along u by alpha while preserving the
    marginal spread along u, so difficulty is monotone in alpha and hits chance
    exactly at alpha=0."""
    y_tr = y[train_idx]
    mu_pos = X[train_idx][y_tr == 1].mean(axis=0)
    mu_neg = X[train_idx][y_tr == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    u = diff / (np.linalg.norm(diff) + 1e-12)
    c = (mu_pos + mu_neg) / 2
    proj_tr = (X[train_idx] - c) @ u
    s_w = float(np.sqrt(0.5 * (proj_tr[y_tr == 1].var() + proj_tr[y_tr == 0].var())))
    p = (X - c) @ u
    eps = np.random.default_rng(seed).standard_normal(len(X))
    p_new = alpha * p + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * s_w * eps
    return X + np.outer(p_new - p, u)


def apply_calibration_train_only_SUPERSEDED(X, y, alpha, train_idx, seed=0):
    """The superseded label-conditional transform, retained ONLY so the 2x2
    ablation in Appendix A (issue 10) -- {ES_PATIENCE 3, 15} x {old calibration,
    label-free calibration} -- is reproducible. Never used for a reported
    number. Its defect: `mask = y == cls` ranges over the full array, so each
    test point's own label decides which class offset is subtracted from it."""
    y_tr = y[train_idx]
    mu_pos = X[train_idx][y_tr == 1].mean(axis=0)
    mu_neg = X[train_idx][y_tr == 0].mean(axis=0)
    midpoint = (mu_pos + mu_neg) / 2
    Xc = np.zeros_like(X)
    for cls, mu_cls in [(1, mu_pos), (0, mu_neg)]:
        mask = y == cls
        Xc[mask] = midpoint + alpha * (mu_cls - midpoint) + (X[mask] - mu_cls)
    return Xc


# Selected by env var for the 2x2 ablation only; defaults to the label-free
# transform, which is what every reported number uses.
CALIBRATION_FN = (apply_calibration_train_only_SUPERSEDED
                  if os.environ.get("USE_SUPERSEDED_CALIBRATION") == "1"
                  else apply_calibration_label_free)
if os.environ.get("USE_SUPERSEDED_CALIBRATION") == "1":
    ALPHA_TRAIN_ONLY = float(os.environ.get("ALPHA_OVERRIDE", 0.1328))


def train_with_scheduler_and_earlystop(X_tr, y_tr, X_sel, y_sel, hidden, max_epochs, seed):
    """Port of MultiHaluDet's trainer.py mechanics: ReduceLROnPlateau
    (its own patience=3) stepped on the SAME validation fold's AUC every
    epoch, plus early stopping on a no-improvement counter over that same
    signal at the repo's own separate threshold (config.patience=15), plus
    best-val-AUC checkpoint tracking -- all three reacting to the identical
    val signal, at the two distinct thresholds the repo actually uses.

    Since the 4th correction this also matches the audited optimizer schedule:
    AdamW at lr=2e-4, batch_size=28 mini-batching, a 5-epoch linear LR warmup
    during which the scheduler is NOT stepped, grad clipping at 0.5, and
    min_lr=1e-7. See the module docstring for the exhaustive
    ported/not-ported list -- notably the loss function and model class are
    NOT ported."""
    torch.manual_seed(seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    n_tr = Xt.shape[0]
    shuffle_gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, best_epoch = -1.0, None, 0
    patience_counter = 0
    for ep in range(max_epochs):
        # trainer.py:88-91 -- linear LR warmup, applied before the epoch's steps.
        if ep < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (ep + 1) / WARMUP_EPOCHS
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr
        model.train()
        perm = torch.randperm(n_tr, generator=shuffle_gen)
        for start in range(0, n_tr, BATCH_SIZE):
            bidx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(Xt[bidx]), yt[bidx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xs)).numpy()
        try:
            auc = roc_auc_score(y_sel, probs)
        except ValueError:
            auc = 0.5
        # trainer.py:134-135 -- the scheduler is NOT stepped during warmup.
        if ep >= WARMUP_EPOCHS:
            scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= ES_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        return model.features(torch.tensor(X, dtype=torch.float32)).numpy()


def run_one_seed(X, y, hidden, seed):
    X_train_idx, X_test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_calibrated = CALIBRATION_FN(
        X, y, ALPHA_TRAIN_ONLY, X_train_idx, seed=seed
    ).astype(np.float32)
    # run_pipeline.py:81,85 -- the audited pipeline scales its input features
    # with RobustScaler, not StandardScaler (4th correction).
    scaler = RobustScaler()
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

        # CLEAN_MATCHED: scheduler+early-stop react to a genuine, disjoint
        # ES-holdout carved from tr_idx (never touches val_idx). FIXED
        # (independent adversarial review found this): this used to be
        # discarded in favor of retraining on the full tr_idx with
        # train_fixed_epochs -- plain constant-LR Adam, no scheduler at all,
        # confounding "reuses the leaky val fold" with "has an adaptive
        # LR/early-stop mechanism at all." Now this model IS the control:
        # same real scheduler+early-stopping mechanic as LEAKY, but fed by
        # the disjoint es_idx carve-out instead of the reused val_idx fold.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed,
        )
        model_clean_matched, _ = train_with_scheduler_and_earlystop(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], hidden, MAX_EPOCHS, fold_seed,
        )
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
        # run_pipeline.py:130 -- the audited pipeline standardizes the deep OOF
        # features (fit on train, applied to test) before the meta-learner.
        deep_scaler = StandardScaler()
        oof_k = deep_scaler.fit_transform(oof[k])
        test_k = deep_scaler.transform(test_feat[k])
        clf = LogisticRegression(max_iter=2000).fit(oof_k, y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_k)[:, 1])
    return aucs


def bca_ci(a, b, n_resamples=10000):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples,
                     method="BCa", random_state=RNG_GLOBAL)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_permutation_p(a, b, n_perm=10000, rng=None):
    """Sign-flip permutation test on the paired differences -- reported
    alongside Wilcoxon because the Wilcoxon signed-rank statistic degrades
    when many |differences| are tied, which they are here (see tie counts)."""
    rng = rng or np.random.default_rng(2026)
    diff = np.asarray(a) - np.asarray(b)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((null >= obs - 1e-15).mean())


def tie_diagnostics(a, b):
    diff = np.asarray(a) - np.asarray(b)
    absd = np.abs(diff)
    _, counts = np.unique(absd, return_counts=True)
    return {
        "n_pairs": int(len(diff)),
        "n_zero_diffs": int((diff == 0).sum()),
        "n_tied_abs_diffs": int((counts[counts > 1]).sum()),
    }


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
        "lr_patience": LR_PATIENCE, "es_patience": ES_PATIENCE,
        # FIXED: this field was hardcoded to "label_free_axis_noising" even in the
        # 2x2 ablation's superseded-calibration cells, mislabelling two of the four
        # result files. It now reports whichever transform actually ran.
        "calibration_method": CALIBRATION_FN.__name__,
        "leaky_plus_lrsched_minus_clean_matched_plus_lrsched": {
            "gap_mean": float(gap_lp_cmp.mean()), "bca_ci_95": ci_lp_cmp, "wilcoxon_p": float(p_lp_cmp),
            "paired_permutation_p": paired_permutation_p(
                results["leaky_plus_lrsched"], results["clean_matched_plus_lrsched"]),
            **tie_diagnostics(results["leaky_plus_lrsched"], results["clean_matched_plus_lrsched"]),
        },
        "clean_matched_plus_lrsched_minus_placebo_plus_lrsched": {
            "gap_mean": float(gap_cmp_placebo.mean()), "bca_ci_95": ci_cmp_placebo, "wilcoxon_p": float(p_cmp_placebo),
            "paired_permutation_p": paired_permutation_p(
                results["clean_matched_plus_lrsched"], results["placebo_plus_lrsched"]),
            **tie_diagnostics(results["clean_matched_plus_lrsched"], results["placebo_plus_lrsched"]),
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
