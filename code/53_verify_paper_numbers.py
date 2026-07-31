"""
Paper 2 -- trace every headline number in the abstract, §5 and the conclusion
back to the result JSON that produces it, and fail loudly on any mismatch.

This exists because the single most common failure mode in this project's
history has been a number surviving in prose after the run behind it was
corrected. Run this before every commit that touches main.tex.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = (ROOT / "draft" / "latex" / "main.tex").read_text()

failures = []
checks = 0


def load(name):
    return json.load(open(R / name))


def check(label, value, fmt="{:+.4f}", must_appear=True):
    """Assert the formatted value literally appears in main.tex."""
    global checks
    checks += 1
    s = fmt.format(value)
    present = s in TEX or s.lstrip("+") in TEX
    ok = present if must_appear else not present
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {s}")
    if not ok:
        failures.append(f"{label} ({s}) not found in main.tex")


def check_absent(label, s):
    """Fail if `s` appears in main.tex as an assertion. Occurrences immediately
    preceded by a LaTeX open-quote (``) are allowed: those are this paper
    quoting its own retracted wording in order to correct it, which is the
    behaviour we want, not the behaviour we are guarding against."""
    global checks
    checks += 1
    asserted = 0
    i = TEX.find(s)
    while i != -1:
        if TEX[max(0, i - 2):i] != "``":
            asserted += 1
        i = TEX.find(s, i + 1)
    ok = asserted == 0
    print(f"  {'OK  ' if ok else 'FAIL'}  retracted string absent (or quoted-for-correction only): "
          f"{label!r}")
    if not ok:
        failures.append(f"retracted string {s!r} asserted {asserted}x in main.tex ({label})")


print("== Mechanism 2 (GUARDIAN, code/48) ==")
d = load("case_study_2_layer_decomposition.json")["randomized_stratified_splits"]
check("selection-specific component mean", d["selection_specific_component_mean"], "{:.3f}")
check("selection-specific component SD", d["selection_specific_component_sd"], "{:.3f}")

print("\n== Mechanism 3 isotropic sweep (code/02d via code/44) ==")
d = load("statistical_rigor_retrofit.json")["isotropic_sweep"]["by_capacity"]
for cap in ["16", "48", "128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])

print("\n== Mechanism 3 fidelity extension (code/49) ==")
d = load("mechanism3_fidelity_extension.json")
g = d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]
assert d["es_patience"] == 15, "reported run must use the repo-faithful ES patience of 15"
assert d["lr_patience"] == 3, "scheduler patience must remain 3"
assert d["calibration_method"] == "label_free_axis_noising"
check("fidelity-extension gap", g["gap_mean"])

print("\n== Mechanism 3 real-feature harness (code/43 via code/44) ==")
d = load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]["by_capacity"]
for cap in ["128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])
    check(f"cap {cap} tie count", d[cap]["leaky_minus_clean_matched"]["n_tied_abs_diffs"], "{:d}")

print("\n== Mechanism 4 (code/45) ==")
d = load("case_study_4_winners_curse.json")
check("all-cells mean", d["summary_all_cells"]["mean"])
check("all-cells CI low", d["summary_all_cells"]["bca_ci_95"][0])
check("all-cells CI high", d["summary_all_cells"]["bca_ci_95"][1])
check("non-saturated mean", d["summary_non_saturated"]["mean"])
check("ceiling-saturated mean", d["summary_ceiling_saturated"]["mean"])

print("\n== Severity relationship: operating point (code/47 Sweep C) ==")
d = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]
check("AUROC0=0.70 gap", d["0.7"]["gap_mean"])
check("AUROC0=0.985 gap", d["0.985"]["gap_mean"])
ratio = d["0.7"]["gap_mean"] / d["0.985"]["gap_mean"]
check("operating-point decline factor", ratio, "{:.1f}")

print("\n== Severity relationship: K scaling refit (code/50) ==")
d = load("evt_scaling_refit.json")["fits"]
check("EVT per-cell-sigma R^2", d["evt_per_cell_sigma"]["r_squared"], "{:.3f}")
check("log two-parameter R^2", d["log_two_parameter"]["r_squared"], "{:.3f}")

print("\n== Adaptive-selection control power (code/22) ==")
import numpy as np
d = load("epoch_forcing_confound_control.json")["by_capacity"]
for cap, want in [("128", 37.3), ("384", 25.7)]:
    m = {k: float(np.mean(v)) for k, v in d[cap]["aucs"].items()}
    ret = 100 * (m["clean_matched_adaptive"] - m["placebo"]) / (m["clean_matched"] - m["placebo"])
    checks += 1
    ok = abs(ret - want) < 0.15 and f"{ret:.1f}" in TEX
    print(f"  {'OK  ' if ok else 'FAIL'}  cap {cap} placebo-relative retention: {ret:.1f}%")
    if not ok:
        failures.append(f"retention cap {cap}: computed {ret:.1f}%, wanted ~{want}% and present in tex")

print("\n== Fidelity-extension 2x2 ablation (code/54) ==")
d = load("fidelity_extension_2x2_ablation.json")
for c in d["cells"]:
    check(f"cell {c['calibration'][:14]}/ES{c['es_patience']}", c["gap_mean"])
checks += 1
_ok = d["sign_of_patience_effect_flips"]
print(f"  {'OK  ' if _ok else 'FAIL'}  patience-correction sign flips across calibrations: {_ok}")
if not _ok:
    failures.append("2x2 ablation no longer shows the sign flip the appendix describes")
# like-for-like ratio quoted in Appendix A issue 12
_rf = load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]
_ratio = (d["currently_reported_number"]
          / _rf["by_capacity"]["128"]["leaky_minus_clean_matched"]["gap_mean"])
check("like-for-like fidelity ratio at capacity 128", _ratio, "{:.1f}")

print("\n== Case Study 2 derived artifact must exist and replay ==")
checks += 1
_art = R / "case_study_2_probe_scores.npz"
_ok = _art.exists() and _art.stat().st_size < 5e6
print(f"  {'OK  ' if _ok else 'FAIL'}  shipped derived artifact present and small: "
      f"{_art.name} ({_art.stat().st_size / 1e6:.2f} MB)" if _art.exists() else "  FAIL  missing")
if not _ok:
    failures.append("case_study_2_probe_scores.npz missing or unexpectedly large")

print("\n== Retracted claims must be gone ==")
check_absent("5.2x asserted as the current like-for-like ratio", "ratio of \\textbf{$5.2\\times$")
check_absent("severity differs sharply", "Severity differs sharply by mechanism")
check_absent("EVT described as confirmed", "the $K$-scaling prediction is now confirmed")
check_absent("Mechanisms 4/5 mapped to L1.1", "instances of L1.1")
check_absent("resists simple pattern-matching (unqualified)",
             "This class of leakage resists simple pattern-matching")
check_absent("verify all four case studies end to end",
             "independently verify all four case studies end to end")
check_absent("old alpha 0.1328 as the reported calibration",
             "converges to $\\alpha=0.1328$")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
