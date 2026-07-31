"""
Paper 2 -- aggregate the 2x2 ablation behind the Mechanism-3 fidelity
extension's reported number (Appendix A, issue 10).

Two independent corrections landed on `code/49` at the same time:
  (a) the early-stopping patience, which had been set to the LR scheduler's
      3 instead of the audited repo's own `config.patience = 15`; and
  (b) the calibration transform, replaced with the fully label-free
      axis-noising version (code/43) after the train-only version was found
      to still condition the shrinkage on each point's own label.
Changing two things at once and reporting the joint result would be exactly
the kind of unisolated comparison this paper criticizes elsewhere, so both
were varied factorially, n=100 seeds per cell, everything else identical:

    ES_PATIENCE_OVERRIDE={3,15} x USE_SUPERSEDED_CALIBRATION={1,0}

Run (about 40 min/cell on one core):
  for ES in 3 15; do
    USE_SUPERSEDED_CALIBRATION=1 ALPHA_OVERRIDE=0.1328 ES_PATIENCE_OVERRIDE=$ES \
      OUT_NAME=abl_oldcal_es$ES.json python3 code/49_mechanism3_fidelity_extension.py
  done
  ES_PATIENCE_OVERRIDE=3 OUT_NAME=mechanism3_fidelity_extension_espatience3_ablation.json \
    python3 code/49_mechanism3_fidelity_extension.py
  python3 code/49_mechanism3_fidelity_extension.py     # the reported cell

then this script to summarize.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "fidelity_extension_2x2_ablation.json"

CELLS = [
    ("superseded_label_conditional", 3, "abl_oldcal_es3.json"),
    ("superseded_label_conditional", 15, "abl_oldcal_es15.json"),
    ("label_free_axis_noising", 3, "mechanism3_fidelity_extension_espatience3_ablation.json"),
    ("label_free_axis_noising", 15, "mechanism3_fidelity_extension.json"),
]


def main():
    out = {"cells": [], "n_seeds_per_cell": 100}
    print(f"{'calibration':32s} {'ES':>3s} {'gap':>9s} {'BCa 95% CI':>22s} "
          f"{'wilcoxon p':>11s} {'LEAKY mean':>11s}")
    for cal, es, fname in CELLS:
        d = json.load(open(R / fname))
        g = d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]
        cell = {
            "calibration": cal,
            "es_patience": es,
            "source_file": f"results/{fname}",
            "alpha": d["alpha_train_only"],
            "gap_mean": g["gap_mean"],
            "bca_ci_95": g["bca_ci_95"],
            "wilcoxon_p": g["wilcoxon_p"],
            "paired_permutation_p": g.get("paired_permutation_p"),
            "leaky_mean_auroc": d["means"]["leaky_plus_lrsched"],
            "is_reported_cell": cal == "label_free_axis_noising" and es == 15,
        }
        out["cells"].append(cell)
        print(f"{cal:32s} {es:3d} {g['gap_mean']:+9.4f} "
              f"[{g['bca_ci_95'][0]:+.4f},{g['bca_ci_95'][1]:+.4f}] "
              f"{g['wilcoxon_p']:11.2g} {d['means']['leaky_plus_lrsched']:11.4f}")

    by = {(c["calibration"], c["es_patience"]): c for c in out["cells"]}
    old3 = by[("superseded_label_conditional", 3)]["gap_mean"]
    old15 = by[("superseded_label_conditional", 15)]["gap_mean"]
    new3 = by[("label_free_axis_noising", 3)]["gap_mean"]
    new15 = by[("label_free_axis_noising", 15)]["gap_mean"]

    out["patience_effect_at_superseded_calibration"] = old15 - old3
    out["patience_effect_at_label_free_calibration"] = new15 - new3
    out["sign_of_patience_effect_flips"] = (old15 - old3) * (new15 - new3) < 0
    out["previously_reported_number"] = old3
    out["currently_reported_number"] = new15

    print(f"\nEffect of the ES-patience correction (3 -> 15):")
    print(f"  at the superseded label-conditional calibration "
          f"(LEAKY operating point ~{by[('superseded_label_conditional', 3)]['leaky_mean_auroc']:.3f}): "
          f"{old15 - old3:+.4f}  (gap SHRINKS)")
    print(f"  at the label-free calibration "
          f"(LEAKY operating point ~{by[('label_free_axis_noising', 3)]['leaky_mean_auroc']:.3f}): "
          f"{new15 - new3:+.4f}  (gap GROWS)")
    print(f"  sign flips: {out['sign_of_patience_effect_flips']}")
    print("\nInterpretation: the patience correction's sign depends on the operating "
          "point, which is itself the paper's central severity relationship (SS5). "
          "Neither correction can be credited with the whole change from "
          f"{old3:+.4f} to {new15:+.4f}.")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
