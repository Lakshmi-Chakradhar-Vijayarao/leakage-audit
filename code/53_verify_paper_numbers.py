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
_cs2 = load("case_study_2_layer_decomposition.json")
d = _cs2["randomized_stratified_splits"]
check("selection-specific component mean", d["selection_specific_component_mean"], "{:.4f}")
check("selection-specific component SD", d["selection_specific_component_sd"], "{:.4f}")
check("general gap mean", d["mean_gap_all_32_layers_mean"], "{:.4f}")
check("general gap SD", d["mean_gap_all_32_layers_sd"], "{:.4f}")
check("n_reps", d["n_reps"], "{:d}")
assert d["n_reps"] >= 50, "N_REPS must not silently shrink back"
assert "independence_caveat" in d, "the shared-dataset caveat must ship in the JSON"
assert "mechanical_positivity_caveat" in d
# The probe's convergence claim in SS4.2 must be measured, not assumed.
_cv = d["convergence"]
check("primary-probe max n_iter_", _cv["max_n_iter"], "{:d}")
assert _cv["n_convergence_warnings"] == 0 and _cv["max_n_iter"] < 1000
checks += 1
_rr = _cs2["regularization_robustness"]["by_C"]
_sel = [_rr[k]["selection_specific_component_mean"] for k in _rr]
_gen = [_rr[k]["general_gap_mean"] for k in _rr]
_ok = all(x > 0 for x in _sel) and all(abs(x) < 0.02 for x in _gen)
print(f"  {'OK  ' if _ok else 'FAIL'}  regularization robustness: sel-specific stays positive "
      f"({min(_sel):+.4f} to {max(_sel):+.4f}), general gap stays ~0 "
      f"({min(_gen):+.4f} to {max(_gen):+.4f})")
if not _ok:
    failures.append("regularization-robustness sweep no longer supports SS4.2's reading")
for k in ["0.01", "100.0"]:
    check(f"regularization sweep C={k} sel-specific", _rr[k]["selection_specific_component_mean"], "{:.4f}")

print("\n== Mechanism 3 isotropic sweep (code/02d via code/44) ==")
d = load("statistical_rigor_retrofit.json")["isotropic_sweep"]["by_capacity"]
for cap in ["16", "48", "128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])

print("\n== Mechanism 3 fidelity extension (code/49) ==")
d = load("mechanism3_fidelity_extension.json")
g = d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]
assert d["es_patience"] == 15, "reported run must use the repo-faithful ES patience of 15"
assert d["lr_patience"] == 3, "scheduler patience must remain 3"
assert d["calibration_method"] == "apply_calibration_label_free"
# Issue 14: the ported optimizer/data-pipeline settings must not silently revert.
import re as _re
_c49 = (ROOT / "code" / "49_mechanism3_fidelity_extension.py").read_text()
for _name, _want in [("LEARNING_RATE", "2e-4"), ("BATCH_SIZE", "28"),
                     ("WARMUP_EPOCHS", "5"), ("GRAD_CLIP", "0.5"),
                     ("MIN_LR", "1e-7"), ("MAX_EPOCHS", "45")]:
    checks += 1
    _ok = _re.search(rf"^{_name} = {_re.escape(_want)}\b", _c49, _re.M) is not None
    print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 ports {_name} = {_want}")
    if not _ok:
        failures.append(f"code/49 no longer sets {_name} = {_want} (Appendix A issue 14)")
for _tok in ["AdamW", "RobustScaler", "clip_grad_norm_"]:
    checks += 1
    _ok = _tok in _c49
    print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 uses {_tok}")
    if not _ok:
        failures.append(f"code/49 no longer uses {_tok} (Appendix A issue 14)")
check("fidelity-extension gap", g["gap_mean"])
check("fidelity-extension LEAKY operating point", d["means"]["leaky_plus_lrsched"], "{:.4f}")

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
# RETRACTED (Appendix A issues 12 and 14): under the ported training loop the
# patience correction no longer flips sign. This check now guards the retraction
# -- if the flip ever reappears, the appendix text must be revisited, not silently
# left stale in the other direction.
_ok = not d["sign_of_patience_effect_flips"]
print(f"  {'OK  ' if _ok else 'FAIL'}  patience-correction sign flip is ABSENT (retracted claim): "
      f"flips={d['sign_of_patience_effect_flips']}")
if not _ok:
    failures.append("2x2 ablation shows a sign flip again; Appendix A item 12 retracts it")
checks += 1
_cal = abs(d["calibration_effect_at_patience_15"])
_pat = abs(d["patience_effect_at_label_free_calibration"])
_ok = _cal > 5 * _pat
print(f"  {'OK  ' if _ok else 'FAIL'}  calibration effect dominates patience effect: "
      f"{_cal:+.4f} vs {_pat:+.4f}")
if not _ok:
    failures.append("Appendix A item 12's decomposition claim no longer holds")
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

print("\n== n_val isolation (code/47 Sweep D) and Sweep B relabelling ==")
d = load("selection_multiplicity_sweep.json")
assert "sweep_B_sample_size" in d and "sweep_B_n_val" not in d, \
    "Sweep B must be relabelled a sample-size sweep, not an n_val sweep"
assert "sweep_B_confound_note" in d
for k in ["2", "3", "5", "10"]:
    check(f"Sweep D K_CV={k} gap", d["sweep_D_n_val_isolated"][k]["gap_mean"])
checks += 1
_g = [d["sweep_D_n_val_isolated"][k]["gap_mean"] for k in ["2", "3", "5", "10"]]
# Non-increasing to within 1e-4: the K_CV=3 and K_CV=5 cells are tied at +0.0036
# (they differ by 2.7e-5), which SS5 and Appendix A both state rather than
# describing the sequence as strictly monotone.
_ok = all(_g[i] >= _g[i + 1] - 1e-4 for i in range(len(_g) - 1)) and _g[0] > _g[-1]
print(f"  {'OK  ' if _ok else 'FAIL'}  Sweep D gap is non-increasing as n_val falls "
      f"(opposite to the 1/sqrt(n_val) prediction): {[round(x, 4) for x in _g]}")
if not _ok:
    failures.append("Sweep D no longer shows the pattern SS5 and Appendix A describe")

print("\n== OOF-averaging alternative explanation (code/55) ==")
d = load("oof_averaging_control.json")["by_capacity"]
for cap in ["16", "48", "128", "384"]:
    check(f"cap {cap} un-averaged gap", d[cap]["single_model"]["leaky_minus_clean_matched"]["mean"])
    checks += 1
    _a = d[cap]["averaged"]["leaky_minus_clean_matched"]["mean"]
    _s = d[cap]["single_model"]["leaky_minus_clean_matched"]["mean"]
    _ok = _s > _a
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap}: un-averaged ({_s:+.4f}) exceeds averaged "
          f"({_a:+.4f}), ratio {_s / _a:.1f}x")
    if not _ok:
        failures.append(f"cap {cap}: averaging no longer attenuates the gap; SS4.3 says it does")
    # the averaged readout must still reproduce code/02d exactly
    checks += 1
    _ref = load("statistical_rigor_retrofit.json")["isotropic_sweep"]["by_capacity"][cap][
        "leaky_minus_clean_matched"]["gap_mean"]
    _ok = abs(_a - _ref) < 1e-9
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap}: averaged readout reproduces code/02d exactly")
    if not _ok:
        failures.append(f"cap {cap}: code/55's averaged readout diverged from code/02d")

print("\n== Fixed-d eigenspectrum sweep (code/56) ==")
d = load("eigenspectrum_sweep_fixed_dim.json")
assert d["feat_dim"] == 64, "the whole point is that d is held at the isotropic sweep's 64"
for name in ["powerlaw_beta0.0", "powerlaw_beta2.0", "real_top64"]:
    check(f"cap128 {name} gap", d["by_capacity"]["128"][name]["leaky_minus_clean_matched"]["mean"])
checks += 1
_c = d["by_capacity"]["128"]
_ok = all(abs(_c[n]["j_realized_population"] - d["j_target"]) < 1e-9 for n in _c)
print(f"  {'OK  ' if _ok else 'FAIL'}  every eigenspectrum cell is calibrated to the same "
      f"Mahalanobis J={d['j_target']:.4f}")
if not _ok:
    failures.append("code/56 cells are not all calibrated to the same J")

print("\n== Guardrail's identity-covariance assumption is documented ==")
checks += 1
_sc = (ROOT / "code" / "sanity_checks.py").read_text()
_ok = "Sigma = I" in _sc and "trace(Sigma)" in _sc and "NOT DIRECTLY APPLICABLE" in _sc
print(f"  {'OK  ' if _ok else 'FAIL'}  sanity_checks.py states the identity-covariance assumption")
if not _ok:
    failures.append("sanity_checks.py no longer documents its Sigma=I assumption")

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
check_absent("2x2 sign flip asserted as a current finding",
             "The patience correction's sign flips with the operating")
check_absent("Sweep B still called an n_val sweep",
             "Sweeping $n_{\\text{val}}$ (via")
check_absent("stale fidelity-extension number", "$+0.0221$ (BCa 95\\% CI")
check_absent("stale like-for-like ratio", "is a ratio of $\\mathbf{2.4\\times}$")
check_absent("~30 layers", "~30 layers")
check_absent("stale 8-rep GUARDIAN count", "under $8$ randomized stratified")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
