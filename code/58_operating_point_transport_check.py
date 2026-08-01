"""
Paper 2 -- does the operating-point severity relationship TRANSPORT?

WHY THIS EXISTS. An independent adversarial review pointed out that this paper
promotes a strong operating-point regularity -- severity falls by 48.6x as
AUROC_0 rises from 0.70 to 0.985, measured in the synthetic Sweep C of
code/47 -- to the abstract, the contributions list, SS5 and the Conclusion,
and at one point framed it as moving severity "more than any difference we
measure between mechanisms." The review then observed that the paper's OWN
shipped JSONs already contain the test of that claim, and that the paper never
performs it.

Three measurements of the SAME contrast (LEAKY minus CLEAN_MATCHED) exist at
achieved operating points within 0.003 AUROC of each other:

  * code/47 Sweep C, synthetic isotropic features, target AUROC_0 = 0.95:
    achieved LEAKY 0.9433, gap +0.00065.
  * code/43 real-feature checkpoint-only harness (label-free calibration,
    capacity 128): achieved LEAKY 0.9403, gap +0.0093.
  * code/49 fidelity extension, same real features and same calibration,
    with the LR scheduler and early stopping also coupled to the fold:
    achieved LEAKY 0.9424, gap +0.0338.

If the operating point were the dominant driver of severity that the
48.6x decline suggests, three measurements agreeing to within 0.003 in
operating point should agree in gap. They differ by 14x and 52x. Appendix A
already notes that two of these operating points match closely (0.9424 versus
0.9403) but stops there and never takes the next step of comparing predicted
against observed severity.

WHAT THIS SCRIPT DOES. It takes the Sweep C measurement at the matched
operating point as the "law's" prediction and divides the two real-feature
measurements by it, so the discrepancy is a computed artifact rather than a
claim. It also reports the same comparison against an interpolated Sweep C
prediction at each real harness's exact achieved operating point, fit in the
probit coordinate z0 = Phi^-1(AUROC_0) in which Sweep C is close to
log-linear, so the result cannot be dismissed as an artifact of comparing
against the nearest grid point.

WHAT IT LICENSES, AND WHAT IT DOES NOT. It does NOT show the operating-point
relationship is wrong: within the synthetic harness Sweep C is a clean,
monotone, strongly-fit regularity, and it is reported as such. It shows the
relationship is a property OF THAT HARNESS -- of its isotropic generative
process and its single fixed leakage mechanic -- and does not transport across
harnesses or across mechanisms at a fixed operating point. The claim the paper
is entitled to is therefore the weaker one: the operating point is a strong
severity modifier WITHIN a harness, and must be matched before any two
severity numbers are compared. It is not a general severity law, and the
48.6x figure must not be quoted as one.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT_PATH = R / "operating_point_transport_check.json"


def main():
    sweep = json.load(open(R / "selection_multiplicity_sweep.json"))["sweep_C_operating_point"]
    rf = json.load(open(R / "real_feature_test_train_only_calibrated.json"))["capacities"]["128"]
    fx = json.load(open(R / "mechanism3_fidelity_extension.json"))

    # ── the synthetic relationship, as measured ──────────────────────────────
    targets = sorted(float(k) for k in sweep)
    sc = [{"target_auroc": t,
           "achieved_leaky_auroc": sweep[str(t)]["leaky_mean"],
           "gap": sweep[str(t)]["gap_mean"]} for t in targets]
    decline = sc[0]["gap"] / sc[-1]["gap"]

    # Log-linear interpolation of Sweep C in the probit coordinate, so a
    # prediction can be made at ANY operating point rather than only at grid
    # points. Fit on achieved (not nominal) operating points, since that is
    # what the real harnesses are being matched on.
    z = norm.ppf(np.array([c["achieved_leaky_auroc"] for c in sc]))
    lg = np.log(np.array([c["gap"] for c in sc]))
    coef = np.polyfit(z, lg, 1)
    interp_r2 = float(1 - np.sum((lg - np.polyval(coef, z)) ** 2) / np.sum((lg - lg.mean()) ** 2))

    def predict(auroc):
        return float(np.exp(np.polyval(coef, norm.ppf(auroc))))

    # ── the reference cell: the Sweep C cell whose operating point matches ───
    ref = min(sc, key=lambda c: abs(c["achieved_leaky_auroc"] - 0.9413))

    observations = [
        {"name": "synthetic Sweep C (code/47), AUROC_0 target 0.95",
         "harness": "synthetic isotropic, checkpoint argmax only",
         "achieved_operating_point": ref["achieved_leaky_auroc"], "gap": ref["gap"]},
        {"name": "real-feature checkpoint-only harness (code/43), capacity 128",
         "harness": "real Mistral-7B/HaluEval features, checkpoint argmax only",
         "achieved_operating_point": float(np.mean(rf["aucs"]["leaky"])),
         "gap": rf["leaky_minus_clean_matched"]["mean"]},
        {"name": "fidelity extension (code/49), capacity 128",
         "harness": "real features, argmax + LR scheduler + early stopping on the same fold",
         "achieved_operating_point": fx["means"]["leaky_plus_lrsched"],
         "gap": fx["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]["gap_mean"]},
    ]
    for o in observations:
        o["ratio_vs_sweep_C_matched_cell"] = o["gap"] / ref["gap"]
        o["sweep_C_interpolated_prediction_at_this_operating_point"] = predict(
            o["achieved_operating_point"])
        o["ratio_vs_sweep_C_interpolated"] = (
            o["gap"] / o["sweep_C_interpolated_prediction_at_this_operating_point"])

    ops = [o["achieved_operating_point"] for o in observations]
    gaps = [o["gap"] for o in observations]

    out = {
        "verdict": "HARNESS_SPECIFIC_NOT_A_TRANSPORTABLE_LAW",
        "summary": (
            "Three measurements of the same LEAKY-minus-CLEAN_MATCHED contrast at achieved "
            f"operating points spanning only {max(ops) - min(ops):.4f} AUROC differ in gap by "
            f"{max(gaps) / min(gaps):.1f}x. The {decline:.1f}x operating-point decline measured "
            "in code/47's Sweep C is therefore a property of that synthetic harness and its "
            "single fixed mechanism, not a severity law that transports across harnesses or "
            "mechanisms. The operating point remains a strong severity MODIFIER within a "
            "harness and must be matched before two severity numbers are compared; it is not a "
            "predictor of severity across them."),
        "sweep_C_operating_point_relationship": {
            "cells": sc,
            "decline_factor_0.70_to_0.985": decline,
            "probit_loglinear_interpolation": {
                "form": "log gap = m * Phi^-1(achieved AUROC) + b",
                "slope_m": float(coef[0]), "intercept_b": float(coef[1]),
                "r_squared_in_log_space": interp_r2,
                "note": ("Fit on ACHIEVED operating points, not nominal targets, because that "
                         "is the coordinate on which the real harnesses are matched."),
            },
        },
        "matched_operating_point_comparison": {
            "reference_cell": ref,
            "observations": observations,
            "operating_point_spread": float(max(ops) - min(ops)),
            "gap_spread_ratio": float(max(gaps) / min(gaps)),
        },
        "what_this_does_not_show": (
            "This does not falsify the operating-point relationship. Within code/47's synthetic "
            "harness it is monotone and strongly fit, and it is reported as such. What fails is "
            "TRANSPORT: the relationship's slope and level are harness-specific. Two of the "
            "three measurements here also differ in leakage MECHANIC (argmax only versus argmax "
            "plus scheduler plus early stopping), so mechanism and harness are not separated by "
            "this comparison -- which is itself the point, since the paper had claimed the "
            "operating point moves severity more than mechanism differences do."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("Sweep C (synthetic), operating point vs gap:")
    for c in sc:
        print(f"  target {c['target_auroc']:<6} achieved {c['achieved_leaky_auroc']:.4f}  "
              f"gap {c['gap']:+.5f}")
    print(f"  decline 0.70 -> 0.985: {decline:.1f}x")
    print(f"  probit log-linear interpolation R^2 = {interp_r2:.4f}\n")
    print(f"Matched-operating-point comparison (reference = Sweep C cell at "
          f"{ref['achieved_leaky_auroc']:.4f}, gap {ref['gap']:+.5f}):")
    for o in observations:
        print(f"  {o['name']}")
        print(f"    achieved {o['achieved_operating_point']:.4f}  gap {o['gap']:+.5f}  "
              f"= {o['ratio_vs_sweep_C_matched_cell']:.1f}x the matched Sweep C cell  "
              f"({o['ratio_vs_sweep_C_interpolated']:.1f}x the interpolated prediction)")
    print(f"\nOperating points span {max(ops) - min(ops):.4f} AUROC; gaps span "
          f"{max(gaps) / min(gaps):.1f}x.")
    print(f"VERDICT: {out['verdict']}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
