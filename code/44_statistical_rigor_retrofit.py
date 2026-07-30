"""
Paper 2 -- statistical rigor retrofit (post-hoc, no reruns needed).

The reviewer found zero confidence intervals anywhere on this paper's
estimand (severity gaps), Wilcoxon used without tie/zero diagnostics, and
two significant real-feature gaps + one anisotropic-sweep capacity
omitted from the text despite being in the shipped JSONs.

This reads the ALREADY-SAVED per-seed AUC arrays from every severity
sweep this paper has run (isotropic, anisotropic, adaptive-selection
control, train-only-calibrated real-feature test) and adds, for every
condition pair already reported: a BCa bootstrap 95% CI (10,000
resamples) on the mean paired gap, and a paired permutation test
p-value alongside the existing Wilcoxon. No new experiments are run --
this is entirely a statistics pass over existing arrays.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "statistical_rigor_retrofit.json"

N_PERM = 10000
RNG = np.random.default_rng(2026)


def bca_ci(a, b):
    """BCa bootstrap 95% CI on the mean of paired differences a - b."""
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=10000,
                     method="BCa", random_state=RNG)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_permutation_test(a, b, n_perm=N_PERM):
    """Sign-flip permutation test on paired differences (assumption-free
    alternative to Wilcoxon; exact for exchangeable pairs under H0)."""
    diff = np.asarray(a) - np.asarray(b)
    observed = diff.mean()
    signs = RNG.choice([-1, 1], size=(n_perm, len(diff)))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(observed)).mean())
    return p


def wilcoxon_diagnostics(a, b):
    diff = np.asarray(a) - np.asarray(b)
    n_zero = int((diff == 0).sum())
    n_ties = len(diff) - len(np.unique(np.abs(diff[diff != 0])))
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return {"wilcoxon_p": float(p), "n_zero_diffs": n_zero, "n_tied_abs_diffs": n_ties, "n_pairs": len(diff)}


def process_capacity_dict(aucs_by_condition, pairs):
    """pairs: list of (name, cond_a, cond_b) to compute stats for."""
    out = {}
    for name, ca, cb in pairs:
        a, b = aucs_by_condition[ca], aucs_by_condition[cb]
        gap_mean = float(np.mean(a) - np.mean(b))
        ci_low, ci_high = bca_ci(a, b)
        perm_p = paired_permutation_test(a, b)
        wilc = wilcoxon_diagnostics(a, b)
        out[name] = {
            "gap_mean": gap_mean,
            "bca_ci_95": [ci_low, ci_high],
            "paired_permutation_p": perm_p,
            **wilc,
        }
    return out


def retrofit_file(path, label, pairs, capacities_key="by_capacity"):
    d = json.load(open(path))
    result = {}
    caps = d.get(capacities_key)
    if caps is None:
        # some files use "capacities" with string-keyed hidden sizes containing "aucs"
        caps = d.get("capacities")
    for cap, capdata in caps.items():
        aucs = capdata.get("aucs")
        if aucs is None:
            continue
        result[str(cap)] = process_capacity_dict(aucs, pairs)
    return {"source_file": str(path.relative_to(ROOT)), "label": label, "by_capacity": result}


def main():
    out = {}

    pairs_standard = [
        ("leaky_minus_clean_matched", "leaky", "clean_matched"),
        ("clean_matched_minus_placebo", "clean_matched", "placebo"),
        ("leaky_minus_placebo", "leaky", "placebo"),
    ]

    print("Retrofitting isotropic sweep (code/02d)...", flush=True)
    out["isotropic_sweep"] = retrofit_file(
        ROOT / "results" / "corrected_capacity_placebo_sweep.json", "isotropic", pairs_standard
    )

    print("Retrofitting anisotropic sweep (code/27)...", flush=True)
    out["anisotropic_sweep"] = retrofit_file(
        ROOT / "results" / "anisotropic_covariance_capacity_sweep.json", "anisotropic", pairs_standard
    )

    print("Retrofitting adaptive-selection control (code/22)...", flush=True)
    try:
        d22 = json.load(open(ROOT / "results" / "epoch_forcing_confound_control.json"))
        print("  (code/22 JSON keys:", list(d22.keys()), ") -- inspect manually if schema differs", flush=True)
    except FileNotFoundError:
        print("  epoch_forcing_confound_control.json not found, skipping", flush=True)

    print("Retrofitting train-only-calibrated real-feature test (code/43)...", flush=True)
    out["real_feature_train_only_calibrated"] = retrofit_file(
        ROOT / "results" / "real_feature_test_train_only_calibrated.json", "real_feature_train_only", pairs_standard
    )

    print("Retrofitting bias-corrected (leaky, pre-train-only-fix) real-feature test (code/33)...", flush=True)
    out["real_feature_cv_calibrated_biased_centering"] = retrofit_file(
        ROOT / "results" / "real_feature_test_cv_calibrated.json", "real_feature_cv_calibrated", pairs_standard
    )

    # Explicitly surface the anisotropic sweep's all-4-capacities CM-PLACEBO numbers
    # (previously only 2 of 4 were reported in the paper text).
    d27 = json.load(open(ROOT / "results" / "anisotropic_covariance_capacity_sweep.json"))
    cm_placebo_all_capacities = {}
    for cap, capdata in d27["capacities"].items():
        cm_placebo_all_capacities[cap] = capdata["clean_matched_minus_placebo"]
    out["anisotropic_cm_minus_placebo_all_capacities"] = cm_placebo_all_capacities

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

    print("\n=== Summary: BCa CIs and permutation p-values ===")
    for key in ["isotropic_sweep", "anisotropic_sweep", "real_feature_train_only_calibrated"]:
        print(f"\n--- {key} ---")
        for cap, stats in out[key]["by_capacity"].items():
            for name, s in stats.items():
                print(f"  cap={cap} {name}: gap={s['gap_mean']:+.4f} "
                      f"BCa95%CI=[{s['bca_ci_95'][0]:+.4f},{s['bca_ci_95'][1]:+.4f}] "
                      f"perm_p={s['paired_permutation_p']:.4g} wilcoxon_p={s['wilcoxon_p']:.4g}")

    print("\n--- anisotropic CLEAN_MATCHED-PLACEBO, all 4 capacities (previously 2 of 4 reported) ---")
    for cap, v in out["anisotropic_cm_minus_placebo_all_capacities"].items():
        print(f"  capacity {cap}: mean={v['mean']:+.4f} p={v['wilcoxon_p']:.4g}")


if __name__ == "__main__":
    main()
