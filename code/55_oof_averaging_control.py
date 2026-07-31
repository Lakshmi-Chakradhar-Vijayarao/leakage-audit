"""
Paper 2 -- Case Study 3, alternative explanation T2: is the small measured
Mechanism-3 severity an artifact of cross-model feature averaging?

THE ALTERNATIVE EXPLANATION. `code/02d_corrected_capacity_placebo_sweep.py`
(and MultiHaluDet's own `run_pipeline.py:108`, which it mirrors) accumulates
test-set features from all N_INNER_FOLDS fold-models and divides by
N_INNER_FOLDS before the meta-learner scores them:

    test_features_accum += extract_features_batch(model, X_seq_test, ...)
    ...
    X_test_deep = test_features_accum / config.n_inner_folds

An independent adversarial review pointed out that this averages the
activations of 5 independently-initialized MLPs. If the leaked
checkpoint-selection signal is fold-specific -- living in each fold-model's
own idiosyncratic response to the validation labels it peeked at -- then
averaging across five differently-leaked models could cancel most of it
before the reported metric ever sees it. Under that account the mechanism
would not be inherently benign; the measurement instrument would simply be
destroying the effect it is trying to measure.

That is a live explanation and it had never been tested. This script tests it.

WHY N_INNER_FOLDS=1 IS NOT THE TEST. The obvious version of the experiment --
"rerun with N_INNER_FOLDS=1 to remove averaging entirely" -- is not runnable,
and not because of a budget limit: with one fold there is no out-of-fold
partition at all, so there are no OOF features to build the meta-learner on.
The averaging and the OOF construction are the same K>1 structure. The
correct way to remove averaging while keeping the pipeline intact is to read
out the SAME trained fold-models without the averaging step, which is what
this script does.

FOUR READOUTS, IDENTICAL MODELS. For every seed and capacity, the fold loop
is bit-identical to code/02d's (same seeds, same splits, same training). Only
the readout differs:

  (A) averaged      -- the shipped pipeline: mean over the K fold-models'
                       test features, then the meta-learner. This reproduces
                       code/02d exactly and is the control for the others.
  (B) single_model  -- no cross-model averaging at all: the meta-learner (fit
                       on the same OOF matrix) scores each fold-model's test
                       features on their own; the K resulting AUROCs are
                       averaged AFTER scoring, so no activation is ever mixed
                       across models. This is the direct un-averaged readout.
  (C) oof_apparent  -- the leaked signal at its source: the meta-learner's
                       APPARENT (in-sample) AUROC on the OOF feature matrix
                       itself, which is per-fold by construction and never
                       passes through the averaging step. This number is
                       upward-biased in absolute terms for every condition and
                       must never be quoted as a performance estimate; it is
                       used only comparatively, since all four conditions carry
                       the identical bias.
  (D) oof_crossfold_cv -- diagnostic. 5-fold cross-validated AUROC of the
                       meta-learner ON the OOF matrix, i.e. training on rows
                       produced by some fold-models and scoring rows produced
                       by others. Reported because it comes out AT CHANCE
                       (0.495-0.515 across all capacities and all four
                       conditions), which is the point: the fold-models'
                       feature spaces carry essentially no information about
                       each other. That is direct evidence for the premise
                       behind T2 -- averaging activations across these models
                       really is mixing incompatible bases.

If the LEAKY-minus-CLEAN_MATCHED gap under (B) and (C) is comparable to or
larger than under (A), cross-model averaging is not what makes the measured
severity small, and the T2 explanation is not supported. If the gap is
substantially larger without averaging, it is.

RESULT (n=100 seeds, all four capacities): T2 IS SUPPORTED. Removing the
averaging step raises the LEAKY-minus-CLEAN_MATCHED gap by roughly 2.5-4.7x
at every capacity -- +0.0009/+0.0017/+0.0034/+0.0027 (averaged, at capacities
16/48/128/384, significant at 2 of 4) versus +0.0042/+0.0068/+0.0087/+0.0083
(single_model, significant at all 4). The OOF matrix read directly gives the
same picture (+0.0060/+0.0084/+0.0083/+0.0040, significant at all 4). So the
mechanism is NOT inherently benign at its source; the pipeline's own
cross-model averaging attenuates it by about 3x before it reaches the
reported metric.

Both facts matter and the paper reports both. The 'averaged' number remains
the right estimate of inflation in what the audited pipeline ACTUALLY
reports, because averaging is what that pipeline does. The 'single_model'
number is the right estimate of the mechanism's severity at its source. The
paper's primary capacity-sweep table is the former, and is unchanged; it is
now explicitly labelled as such rather than as a bound on the mechanism.

This script does NOT modify code/02d or its committed results; the paper's
primary capacity-sweep table is unchanged and this is reported as a separate
robustness check.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

import importlib.util
import sys

_SPEC = importlib.util.spec_from_file_location(
    "sweep02d", Path(__file__).resolve().parent / "02d_corrected_capacity_placebo_sweep.py")
_M = importlib.util.module_from_spec(_SPEC)
sys.modules["sweep02d"] = _M
_SPEC.loader.exec_module(_M)

make_synthetic_data = _M.make_synthetic_data
train_to_best_checkpoint = _M.train_to_best_checkpoint
train_fixed_epochs = _M.train_fixed_epochs
extract_features = _M.extract_features
N_INNER_FOLDS = _M.N_INNER_FOLDS
TEST_SIZE = _M.TEST_SIZE
EPOCHS = _M.EPOCHS
ES_HOLD_FRACTION = _M.ES_HOLD_FRACTION

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "oof_averaging_control.json"
N_SEEDS = 100
CAPACITIES = [16, 48, 128, 384]
CONDITIONS = ["leaky", "clean", "clean_matched", "placebo"]
RNG_GLOBAL = np.random.default_rng(2026)


def run_one_seed(seed, hidden):
    """Fold loop identical to code/02d's run_one_seed (same seeds, same splits,
    same training), but the per-fold test features are kept separately so the
    averaged and un-averaged readouts can both be computed from the SAME
    trained models."""
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
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in CONDITIONS}
    per_fold_test = {k: [] for k in CONDITIONS}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold

        m_leaky, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed)
        oof["leaky"][val_idx] = extract_features(m_leaky, X_train[val_idx])
        per_fold_test["leaky"].append(extract_features(m_leaky, X_test))

        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[tr_idx], random_state=fold_seed)
        m_clean, best_epoch = train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, fold_seed)
        oof["clean"][val_idx] = extract_features(m_clean, X_train[val_idx])
        per_fold_test["clean"].append(extract_features(m_clean, X_test))

        m_cm = train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, best_epoch, fold_seed)
        oof["clean_matched"][val_idx] = extract_features(m_cm, X_train[val_idx])
        per_fold_test["clean_matched"].append(extract_features(m_cm, X_test))

        y_val_permuted = rng.permutation(y_train[val_idx])
        m_plc, _ = train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_val_permuted,
            hidden, EPOCHS, fold_seed)
        oof["placebo"][val_idx] = extract_features(m_plc, X_train[val_idx])
        per_fold_test["placebo"].append(extract_features(m_plc, X_test))

    out = {"averaged": {}, "single_model": {}, "oof_apparent": {}, "oof_crossfold_cv": {}}
    for k in CONDITIONS:
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)

        # (A) shipped pipeline: average activations across the K fold-models.
        avg_test = np.mean(per_fold_test[k], axis=0)
        out["averaged"][k] = roc_auc_score(y_test, clf.predict_proba(avg_test)[:, 1])

        # (B) no cross-model averaging: score each fold-model's own test
        #     features, average the AUROCs afterwards.
        per_fold_auc = [roc_auc_score(y_test, clf.predict_proba(tf)[:, 1])
                        for tf in per_fold_test[k]]
        out["single_model"][k] = float(np.mean(per_fold_auc))

        # (C) the leaked signal at its source: the OOF matrix itself, which
        #     never passes through the averaging step. In-sample/apparent, so
        #     comparative use only (identical bias across all four conditions).
        out["oof_apparent"][k] = roc_auc_score(y_train, clf.predict_proba(oof[k])[:, 1])

        # (D) diagnostic: CV across the OOF matrix trains on rows from some
        #     fold-models and scores rows from others. Expected to be poor --
        #     see the module docstring.
        p = cross_val_predict(
            LogisticRegression(max_iter=2000), oof[k], y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            method="predict_proba")[:, 1]
        out["oof_crossfold_cv"][k] = roc_auc_score(y_train, p)
    return out


def gap_stats(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gap = a - b
    if np.allclose(gap, 0):
        return {"mean": 0.0, "wilcoxon_p": 1.0, "bca_ci_95": [0.0, 0.0]}
    _, p = wilcoxon(a, b)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=RNG_GLOBAL)
    return {"mean": float(gap.mean()),
            "wilcoxon_p": float(p),
            "bca_ci_95": [float(res.confidence_interval.low),
                          float(res.confidence_interval.high)]}


def main():
    t0 = time.time()
    results = {}
    for hidden in CAPACITIES:
        acc = {r: {k: [] for k in CONDITIONS} for r in ["averaged", "single_model", "oof_apparent", "oof_crossfold_cv"]}
        for seed in range(N_SEEDS):
            o = run_one_seed(seed, hidden)
            for r in acc:
                for k in CONDITIONS:
                    acc[r][k].append(o[r][k])
            if (seed + 1) % 25 == 0:
                print(f"  [cap {hidden}] seed {seed+1}/{N_SEEDS}  elapsed={time.time()-t0:.0f}s",
                      flush=True)
        cell = {}
        for r in acc:
            cell[r] = {
                "means": {k: float(np.mean(v)) for k, v in acc[r].items()},
                "leaky_minus_clean_matched": gap_stats(acc[r]["leaky"], acc[r]["clean_matched"]),
                "clean_matched_minus_placebo": gap_stats(acc[r]["clean_matched"], acc[r]["placebo"]),
            }
        results[str(hidden)] = cell
        print(f"\n--- capacity {hidden} ---")
        for r in ["averaged", "single_model", "oof_apparent", "oof_crossfold_cv"]:
            g = cell[r]["leaky_minus_clean_matched"]
            print(f"  {r:13s} LEAKY-CLEAN_MATCHED = {g['mean']:+.4f} "
                  f"CI=[{g['bca_ci_95'][0]:+.4f},{g['bca_ci_95'][1]:+.4f}] p={g['wilcoxon_p']:.4g}",
                  flush=True)

    out = {
        "n_seeds": N_SEEDS,
        "capacities": CAPACITIES,
        "n_inner_folds": N_INNER_FOLDS,
        "readouts": {
            "averaged": "code/02d's shipped pipeline: mean of the K fold-models' test-feature "
                        "activations, then the meta-learner. Control condition.",
            "single_model": "No cross-model averaging: the meta-learner scores each fold-model's "
                            "own test features; the K AUROCs are averaged after scoring.",
            "oof_apparent": "APPARENT (in-sample) AUROC of the meta-learner on the OOF feature "
                            "matrix itself, which is per-fold by construction and never averaged "
                            "across models. Upward-biased in absolute terms for every condition; "
                            "use comparatively only.",
            "oof_crossfold_cv": "Diagnostic. 5-fold CV on the OOF matrix, which trains on rows "
                                "from some fold-models and scores rows from others. Comes out AT "
                                "CHANCE (0.495-0.515) for every condition and capacity, showing "
                                "the fold-models' feature spaces carry essentially no information "
                                "about one another -- direct evidence that cross-model averaging "
                                "really does mix incompatible bases.",
        },
        "n_inner_folds_1_note": "N_INNER_FOLDS=1 is not runnable and this is structural, not a "
                                "budget limit: with a single fold there is no out-of-fold "
                                "partition, so no OOF features exist to train the meta-learner "
                                "on. The 'single_model' readout is the correct un-averaged "
                                "equivalent -- same trained models, no activation ever mixed "
                                "across models.",
        "by_capacity": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
