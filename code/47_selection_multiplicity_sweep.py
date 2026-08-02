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

Sweep B -- TOTAL SAMPLE SIZE (N_SAMPLES): {350, 700, 2800}.
  RELABELLED (an independent adversarial review found this mislabelled). This
  sweep was previously described, here and in the paper, as an "n_val sweep."
  It is not one. Varying N_SAMPLES moves the train-set size, the test-set size
  AND the per-fold validation-set size simultaneously and proportionally, so
  the resulting gaps cannot be attributed to n_val alone: any comparison
  against the winner's-curse prediction gap ~ 1/sqrt(n_val) is confounded by
  the training set (hence the achievable fit) and the test set (hence the
  granularity and variance of the reported AUROC) moving with it. It is
  reported below as a sample-size sweep, with that confound stated, and the
  paper's extreme-value-theory discussion is qualified accordingly.

Sweep C -- operating point (TARGET_AUROC): {0.70, 0.80, 0.90, 0.95, 0.985}.
  The last matches MultiHaluDet's actual reported AUROC (0.9855).

Sweep D -- n_val ISOLATED, via the number of inner CV folds K_CV in
  {2, 3, 5, 10}, at fixed N_SAMPLES=700. This is the sweep Sweep B was
  mislabelled as. Holding the dataset and the outer train/test split fixed and
  varying only K_CV changes the per-fold validation-set size
  (n_val = n_train/K_CV = 280/187/112/56) without touching the total sample,
  the training pool, or the test set the metric is reported on.
  Disclosed residual coupling, inherent to cross-validation rather than to
  this design: the per-fold TRAINING subset size necessarily moves inversely
  (n_tr_fold = n_train - n_val = 280/373/448/504), because a fold's train and
  validation parts partition the same fixed pool. So Sweep D isolates n_val
  from total/test-set size but not from per-fold train size. It is still a
  strictly cleaner test of the 1/sqrt(n_val) prediction than Sweep B, and it
  costs nothing beyond the CV structure the harness already has.

DEGENERACY INSTRUMENTATION (this revision; an independent adversarial review
found the omission). `state_dicts_identical` and the per-cell degeneracy record
below were added in an earlier round and wired into code/57's joint K x
operating-point grid, which gates its fit on them -- but they were NEVER
ENABLED AT THIS SCRIPT'S OWN CALL SITES, even though Sweep A is what produces
the headline `gap = a + b ln K` law. By that same bitwise check the small-K
cells of Sweep A are largely degenerate: at K=3, 5, 10 the LEAKY and
CLEAN_MATCHED models come out bitwise identical in roughly 98%, 90% and 65% of
(seed, fold) pairs respectively, because three to ten full-batch epochs are too
few for the two selection rules to land on different epochs. A cell like that
contributes a gap that is near-zero BY CONSTRUCTION, and feeding it to a
log-linear fit manufactures most of the fit's apparent dynamic range.

`degeneracy_check=True` is now passed at EVERY call site (Sweeps A, B, C and D),
so every cell this script ships carries an auditable degeneracy record. The
check consumes no RNG and does not touch the training path, so every previously
reported gap, arm mean, Wilcoxon p and bootstrap CI reproduces bit for bit; only
new fields are added. code/50 consumes these records and reports the K law both
as originally fit (all K > 1 cells, now labelled contaminated) and refit on the
non-degenerate cells only, using the SAME threshold code/57 already ships
(identical_state_dict_frac <= 0.50).

Everything else -- SweepMLP, the isotropic-Gaussian generative process,
LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO logic -- is an exact port of
code/02d_corrected_capacity_placebo_sweep.py, with the single `seed`
variable decoupled into `data_seed` (which synthetic sample), `split_seed`
(train/test split), `fold_seed` (inner CV fold assignment), and
`init_seed` (torch model initialization).
"""
import json
import os
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


def state_dicts_identical(m1, m2):
    """Bitwise equality of two models' parameters.

    DEGENERACY DIAGNOSTIC (added after an independent adversarial review). The
    LEAKY and CLEAN_MATCHED arms differ only in WHICH epoch each one's
    selection rule stops at: LEAKY argmaxes validation AUROC on the same fold
    it later reports features on, CLEAN_MATCHED argmaxes on a disjoint 15%
    carve-out and then retrains for that many epochs on the full fold. When the
    epoch budget K is small enough that both argmaxes land on the last epoch,
    CLEAN_MATCHED's retrain reproduces LEAKY's trajectory step for step and the
    two models come out BITWISE IDENTICAL -- so the measured gap is exactly
    0.00000 by construction, not by measurement. Any cell with a high identical
    fraction is telling you the harness has no contrast to measure there, and
    must not be fed to a model fit as if it were a real observation."""
    s1, s2 = m1.state_dict(), m2.state_dict()
    return all(torch.equal(s1[k], s2[k]) for k in s1)


def run_one_seed(data_seed, split_seed, fold_seed_base, init_seed_base, hidden, epochs, n_samples,
                 target_auroc, n_inner_folds=N_INNER_FOLDS, degeneracy_check=False):
    X, y = make_synthetic_data(data_seed, n_samples, target_auroc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    rng = np.random.default_rng(fold_seed_base + 10000)
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=fold_seed_base)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    conditions = ["leaky", "clean", "clean_matched", "placebo"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conditions}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conditions}
    val_stds = []
    degen_identical, degen_maxdiff, leaky_last, clean_last = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        init_seed = init_seed_base * 100 + fold
        model_leaky, leaky_best_epoch, val_std = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], hidden, epochs, init_seed)
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

        if degeneracy_check:
            degen_identical.append(float(state_dicts_identical(model_leaky, model_clean_matched)))
            sl, sc = model_leaky.state_dict(), model_clean_matched.state_dict()
            degen_maxdiff.append(float(max((sl[k] - sc[k]).abs().max().item() for k in sl)))
            leaky_last.append(float(leaky_best_epoch == epochs))
            clean_last.append(float(best_epoch == epochs))

        y_val_permuted = rng.permutation(y_train[val_idx])
        model_placebo, _, _ = train_to_best_checkpoint(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted, hidden, epochs, init_seed)
        oof["placebo"][val_idx] = extract_features(model_placebo, X_train[val_idx])
        test_feat["placebo"] += extract_features(model_placebo, X_test)

    aucs = {}
    for k in oof:
        test_feat[k] /= n_inner_folds
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    aucs["_val_auc_std"] = float(np.mean(val_stds))
    if degeneracy_check:
        aucs["_degen_identical_frac"] = float(np.mean(degen_identical))
        aucs["_degen_max_param_diff"] = float(np.max(degen_maxdiff))
        aucs["_leaky_argmax_is_last_frac"] = float(np.mean(leaky_last))
        aucs["_clean_argmax_is_last_frac"] = float(np.mean(clean_last))
    return aucs


def run_sweep_cell(n_seeds, hidden, epochs, n_samples, target_auroc, n_inner_folds=N_INNER_FOLDS,
                   degeneracy_check=False, boot_seed=None):
    """One (K, AUROC_0, ...) cell.

    `boot_seed`, when given, seeds this cell's BCa bootstrap independently
    instead of drawing from the module-level RNG_GLOBAL. This exists so that
    cells can be computed in any order (e.g. in parallel) and still reproduce
    bit for bit; RNG_GLOBAL is consumed sequentially and therefore makes CIs
    depend on cell ORDER. gap_mean, the Wilcoxon p and every arm mean are fully
    deterministic given the data/split/fold/init seeds and are unaffected
    either way -- only the resampled CI endpoints depend on this."""
    all_aucs = {k: [] for k in ["leaky", "clean", "clean_matched", "placebo"]}
    val_stds = []
    degen = {k: [] for k in ["_degen_identical_frac", "_degen_max_param_diff",
                             "_leaky_argmax_is_last_frac", "_clean_argmax_is_last_frac"]}
    for seed in range(n_seeds):
        data_seed = seed
        split_seed = seed + 100000
        fold_seed_base = seed + 200000
        init_seed_base = seed + 300000
        aucs = run_one_seed(data_seed, split_seed, fold_seed_base, init_seed_base, hidden, epochs, n_samples,
                            target_auroc, n_inner_folds=n_inner_folds, degeneracy_check=degeneracy_check)
        for k in all_aucs:
            all_aucs[k].append(aucs[k])
        val_stds.append(aucs["_val_auc_std"])
        if degeneracy_check:
            for k in degen:
                degen[k].append(aucs[k])
    arrs = {k: np.array(v) for k, v in all_aucs.items()}
    gap = arrs["leaky"] - arrs["clean_matched"]
    _, p = wilcoxon(arrs["leaky"], arrs["clean_matched"]) if not np.allclose(gap, 0) else (None, 1.0)
    rng_boot = RNG_GLOBAL if boot_seed is None else np.random.default_rng(boot_seed)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa", random_state=rng_boot)
    out = {
        "gap_mean": float(gap.mean()), "gap_bca_ci_95": [float(res.confidence_interval.low), float(res.confidence_interval.high)],
        "wilcoxon_p": float(p), "mean_val_auc_std": float(np.mean(val_stds)),
        "leaky_mean": float(arrs["leaky"].mean()), "clean_matched_mean": float(arrs["clean_matched"].mean()),
        "placebo_mean": float(arrs["placebo"].mean()),
    }
    if degeneracy_check:
        out["degeneracy"] = {
            "identical_state_dict_frac": float(np.mean(degen["_degen_identical_frac"])),
            "max_param_abs_diff": float(np.max(degen["_degen_max_param_diff"])),
            "leaky_argmax_is_last_epoch_frac": float(np.mean(degen["_leaky_argmax_is_last_frac"])),
            "clean_argmax_is_last_epoch_frac": float(np.mean(degen["_clean_argmax_is_last_frac"])),
            "seeds_with_exactly_zero_gap_frac": float(np.mean(np.abs(gap) == 0.0)),
        }
    return out


def run_sweep_D(out, n_seeds, t0):
    """n_val isolated via the inner-CV fold count, at fixed N_SAMPLES.

    This is the sweep Sweep B was mislabelled as. N_SAMPLES (and therefore the
    outer train/test split, and the test set the metric is reported on) is held
    fixed at DEFAULT_N_SAMPLES; only K_CV moves, which moves n_val = n_train /
    K_CV. Residual coupling disclosed in the module docstring: per-fold train
    size moves inversely, because a fold's train and val parts partition one
    fixed pool. Nothing here can be decoupled further without leaving
    cross-validation entirely."""
    print("\n=== Sweep D: n_val ISOLATED via K_CV (N_SAMPLES fixed) ===", flush=True)
    n_train = int(DEFAULT_N_SAMPLES * (1 - TEST_SIZE))
    for k_cv in [2, 3, 5, 10]:
        r = run_sweep_cell(n_seeds, CAPACITY, DEFAULT_EPOCHS, DEFAULT_N_SAMPLES,
                           DEFAULT_TARGET_AUROC, n_inner_folds=k_cv, degeneracy_check=True)
        n_val = n_train // k_cv
        out.setdefault("sweep_D_n_val_isolated", {})[str(k_cv)] = {
            **r, "k_cv": k_cv, "n_val_exact": n_val,
            "n_train_fold_exact": n_train - n_val, "n_samples": DEFAULT_N_SAMPLES,
        }
        print(f"  K_CV={k_cv} (n_val={n_val}, n_tr_fold={n_train - n_val}): "
              f"gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} p={r['wilcoxon_p']:.4g} "
              f"val_auc_std={r['mean_val_auc_std']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)
    out["sweep_D_note"] = (
        "True n_val sweep: total sample size, outer train/test split and test set all held "
        "fixed at N_SAMPLES=700; only the inner CV fold count K_CV varies, which varies n_val. "
        "Residual coupling (inherent to CV, disclosed rather than claimed away): per-fold "
        "training-subset size moves inversely with n_val. Compare against sweep_B_sample_size, "
        "which was previously mislabelled an n_val sweep and in which train, test and val all "
        "move together."
    )
    return out


def main():
    t0 = time.time()
    N_SEEDS = 100
    only = os.environ.get("ONLY_SWEEP", "").upper()

    if only == "D":
        # Additive rerun: keep every previously committed cell, add Sweep D.
        # Sweeps A/B/C are deterministic given their seeds and are not affected
        # by this change, so they are not recomputed; only the mislabelled key
        # is migrated in place.
        out = json.load(open(OUT_PATH))
        if "sweep_B_n_val" in out:
            out["sweep_B_sample_size"] = out.pop("sweep_B_n_val")
            for cell in out["sweep_B_sample_size"].values():
                n = None
                for k, v in out["sweep_B_sample_size"].items():
                    if v is cell:
                        n = int(k)
                cell["n_train_approx"] = int(n * (1 - TEST_SIZE))
                cell["n_test_approx"] = int(n * TEST_SIZE)
            out["sweep_B_confound_note"] = (
                "CONFOUNDED, and relabelled from 'sweep_B_n_val' accordingly. Varying N_SAMPLES "
                "moves train, test AND validation set sizes together, so these gaps are not "
                "attributable to n_val alone and do not constitute a test of gap ~ 1/sqrt(n_val). "
                "See sweep_D_n_val_isolated, which holds N_SAMPLES (and hence train and test "
                "sizes) fixed and varies only the inner CV fold count."
            )
        run_sweep_D(out, N_SEEDS, t0)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved (Sweep D merged into existing cells): {OUT_PATH}")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")
        return

    out = {"sweep_A_K": {}, "sweep_B_sample_size": {}, "sweep_C_operating_point": {}}

    print("=== Sweep A: K (number of candidate checkpoints) ===", flush=True)
    SWEEP_A_K_VALUES = [1, 3, 5, 10, 15, 25, 45, 75, 135, 225]
    SWEEP_A_N_SEEDS = 200
    for K in SWEEP_A_K_VALUES:
        r = run_sweep_cell(SWEEP_A_N_SEEDS, CAPACITY, K, DEFAULT_N_SAMPLES, DEFAULT_TARGET_AUROC,
                           degeneracy_check=True)
        out["sweep_A_K"][str(K)] = r
        print(f"  K={K}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} p={r['wilcoxon_p']:.4g} "
              f"val_auc_std={r['mean_val_auc_std']:.4f} "
              f"identical={r['degeneracy']['identical_state_dict_frac']:.3f}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

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

    print("\n=== Sweep B: TOTAL SAMPLE SIZE (N_SAMPLES) -- NOT an isolated n_val sweep ===", flush=True)
    for n_samples in [350, 700, 2800]:
        r = run_sweep_cell(N_SEEDS, CAPACITY, DEFAULT_EPOCHS, n_samples, DEFAULT_TARGET_AUROC,
                           degeneracy_check=True)
        n_val_approx = int(n_samples * (1 - TEST_SIZE) / N_INNER_FOLDS)
        out["sweep_B_sample_size"][str(n_samples)] = {
            **r, "n_val_approx": n_val_approx,
            "n_train_approx": int(n_samples * (1 - TEST_SIZE)),
            "n_test_approx": int(n_samples * TEST_SIZE),
        }
        print(f"  N_SAMPLES={n_samples} (n_val~={n_val_approx}, n_train~="
              f"{int(n_samples * (1 - TEST_SIZE))}, n_test~={int(n_samples * TEST_SIZE)}): "
              f"gap={r['gap_mean']:+.4f} "
              f"CI={r['gap_bca_ci_95']} p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)
    out["sweep_B_confound_note"] = (
        "CONFOUNDED, and relabelled from 'sweep_B_n_val' accordingly. Varying N_SAMPLES moves "
        "train, test AND validation set sizes together, so these gaps are not attributable to "
        "n_val alone and do not constitute a test of gap ~ 1/sqrt(n_val). See sweep_D_n_val_"
        "isolated, which holds N_SAMPLES (and hence train and test sizes) fixed and varies only "
        "the inner CV fold count."
    )

    print("\n=== Sweep C: operating point (TARGET_AUROC) ===", flush=True)
    for target_auroc in [0.70, 0.80, 0.90, 0.95, 0.985]:
        r = run_sweep_cell(N_SEEDS, CAPACITY, DEFAULT_EPOCHS, DEFAULT_N_SAMPLES, target_auroc,
                           degeneracy_check=True)
        out["sweep_C_operating_point"][str(target_auroc)] = r
        print(f"  TARGET_AUROC={target_auroc}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} "
              f"p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    run_sweep_D(out, N_SEEDS, t0)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
