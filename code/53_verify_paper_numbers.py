"""
Paper 2 -- trace every headline number in the abstract, §5 and the conclusion
back to the result JSON that produces it, and fail loudly on any mismatch.

This exists because the single most common failure mode in this project's
history has been a number surviving in prose after the run behind it was
corrected. Run this before every commit that touches main.tex.
"""
import json
import re
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


def _quoted_spans(text):
    """Character spans of LaTeX ``...'' quotations."""
    spans, i = [], text.find("``")
    while i != -1:
        j = text.find("''", i + 2)
        if j == -1:
            break
        spans.append((i, j + 2))
        i = text.find("``", j + 2)
    return spans


_QUOTED = _quoted_spans(TEX)


def check_absent(label, s):
    """Fail if `s` appears in main.tex as an assertion.

    Occurrences INSIDE a LaTeX ``...'' quotation are allowed: those are this
    paper quoting its own retracted wording in order to correct it, which is
    the behaviour we want, not the behaviour we are guarding against. (This
    previously only exempted strings whose opening `` was immediately
    adjacent, which failed as soon as a retraction quoted a phrase from the
    middle of the retracted sentence -- e.g. ``the two axes interact
    multiplicatively rather than additively''.)"""
    global checks
    checks += 1
    asserted = 0
    i = TEX.find(s)
    while i != -1:
        inside_quote = any(a <= i and i + len(s) <= b for a, b in _QUOTED)
        if not inside_quote:
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
# The seed-decoupling retrofit must not silently revert to the confounded scheme.
checks += 1
_02d = load("corrected_capacity_placebo_sweep.json")
_ok = _02d["config"].get("seed_scheme") == "decoupled"
print(f"  {'OK  ' if _ok else 'FAIL'}  code/02d ran with decoupled data/split/fold/init seeds")
if not _ok:
    failures.append("code/02d reverted to the coupled single-seed scheme (SS4.3 says it is fixed)")
checks += 1
# ...and the legacy coupled run must still ship, so the comparison SS4.3 makes is checkable.
_ok = (R / "corrected_capacity_placebo_sweep_coupled_seed_legacy.json").exists()
print(f"  {'OK  ' if _ok else 'FAIL'}  the superseded coupled-seed run ships for comparison")
if not _ok:
    failures.append("coupled-seed legacy JSON missing; SS4.3's before/after comparison is unverifiable")
checks += 1
# code/47's default cell must reproduce code/02d's cap-128 cell. This is a
# DETERMINISM check, not independent confirmation: code/47's docstring states
# its harness is "an exact port of code/02d", so the two share an
# implementation. It catches drift between them; it does not corroborate the
# result via a second implementation. (An adversarial review found the previous
# comment here, and the matching sentence in SS4.3, calling code/47's cell
# "independently-written". Both are withdrawn.)
_c47 = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.8"]["gap_mean"]
_c02d = d["128"]["leaky_minus_clean_matched"]["gap_mean"]
_ok = abs(_c47 - _c02d) < 5e-5
print(f"  {'OK  ' if _ok else 'FAIL'}  code/47 default cell reproduces code/02d cap-128 "
      f"(determinism check, shared implementation) ({_c47:+.5f} vs {_c02d:+.5f})")
if not _ok:
    failures.append("code/47 and code/02d no longer agree on the shared configuration")

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
# Appendix A item 10 quoted a stale CLEAN_MATCHED-vs-PLACEBO gap that did not
# survive the issue-14 fidelity port. Pin it to the shipped array.
check("fidelity-extension control-vs-placebo gap",
      d["clean_matched_plus_lrsched_minus_placebo_plus_lrsched"]["gap_mean"])
# Training-depth confound (5th correction) must be tracked, not silently dropped.
checks += 1
_be = d.get("best_epoch_stats")
_ok = _be is not None and "mean_best_epoch" in _be
print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 tracks the training-depth confound (best_epoch_stats)")
if not _ok:
    failures.append("mechanism3_fidelity_extension.json has no best_epoch_stats (SS4.3 reports it)")
else:
    check("LEAKY mean kept-checkpoint epoch", _be["mean_best_epoch"]["leaky_plus_lrsched"], "{:.2f}")
    check("control mean kept-checkpoint epoch",
          _be["mean_best_epoch"]["clean_matched_plus_lrsched"], "{:.2f}")
    check("training-depth relative difference", _be["relative_difference_pct"], "{:.1f}")

print("\n== Mechanism 3 real-feature harness (code/43 via code/44) ==")
d = load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]["by_capacity"]
for cap in ["128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])
    check(f"cap {cap} tie count", d[cap]["leaky_minus_clean_matched"]["n_tied_abs_diffs"], "{:d}")

print("\n== Mechanism 4 (code/45) ==")
d = load("case_study_4_winners_curse.json")
# HEADLINE is the non-degenerate subset. The rotation estimator is
# algebraically 0 whenever all 3 rotations pick the same layer, so those cells
# measure nothing and must not be averaged in.
_nd = d["summary_non_degenerate"]
check("non-degenerate headline mean", _nd["mean"])
check("non-degenerate CI low", _nd["bca_ci_95"][0])
check("non-degenerate CI high", _nd["bca_ci_95"][1])
check("non-degenerate n_cells", _nd["n_cells"], "{:d}")
check("all-cells mean (contaminated, retained for continuity)", d["summary_all_cells"]["mean"])
check("all-cells CI low", d["summary_all_cells"]["bca_ci_95"][0])
check("all-cells CI high", d["summary_all_cells"]["bca_ci_95"][1])
check("non-saturated mean (contaminated)", d["summary_non_saturated"]["mean"])
check("non-saturated non-degenerate mean", d["summary_non_saturated_non_degenerate"]["mean"])
check("ceiling-saturated mean", d["summary_ceiling_saturated"]["mean"])
_da = d["degeneracy_audit"]
check("number of degenerate cells", _da["n_degenerate"], "{:d}")
check("degenerate cells' permutation-null mean",
      _da["permutation_null_mean_over_degenerate_cells"])
checks += 1
# Every degenerate cell must be EXACTLY zero, not a float residue: scipy's
# Wilcoxon drops exact zeros but ranks +-3.7e-17 as real observations, which
# is what moved the non-saturated subgroup's p from 0.047 to 0.0625.
_ok = all(v["winners_curse_estimate"] == 0.0 for v in d["per_cell"].values()
          if v["estimator_degenerate"])
print(f"  {'OK  ' if _ok else 'FAIL'}  degenerate cells are snapped to EXACT zero "
      f"(no float residue reaching the Wilcoxon)")
if not _ok:
    failures.append("code/45 is letting float residues through as signed observations again")
checks += 1
# The degeneracy must be DETECTED structurally, not inferred from the value.
_ok = all(("estimator_degenerate" in v and "selected_layers_across_rotations" in v)
          for v in d["per_cell"].values())
print(f"  {'OK  ' if _ok else 'FAIL'}  every cell records its per-rotation selected layers "
      f"and a degeneracy flag")
if not _ok:
    failures.append("code/45 no longer records the degeneracy diagnostic per cell")
checks += 1
# code/45's permutation null and bootstrap must be REPRODUCIBLE. An earlier
# version seeded them from Python's builtin hash() of the cell name, which is
# salted per interpreter process (PYTHONHASHSEED), so the shipped numbers could
# not be re-derived from the shipped code -- the exact failure mode this paper
# is about. It must use a stable hash.
_c45 = (ROOT / "code" / "45_case_study_4_winners_curse.py").read_text()
_ok = "zlib.crc32" in _c45 and "default_rng(abs(hash(" not in _c45
print(f"  {'OK  ' if _ok else 'FAIL'}  code/45 seeds its permutation null from a STABLE hash "
      f"(not Python's salted hash())")
if not _ok:
    failures.append("code/45's permutation null is seeded from a per-process-salted hash; "
                    "its reported null and bootstrap values are not reproducible")
checks += 1
# Per-model probed-layer counts: the paper said a flat "33 layers" for all 24
# cells; qwen2.5-7b probes 29.
_lc = d["per_model_layer_counts"]
_ok = _lc == {"llama3.1-8b": 33, "mistral-7b": 33, "qwen2.5-7b": 29} and "29" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  per-model probed-layer counts shipped and stated: {_lc}")
if not _ok:
    failures.append("per-model layer counts missing from the JSON or from main.tex")

print("\n== Operating-point transport check (code/58) ==")
d = load("operating_point_transport_check.json")
_obs = d["matched_operating_point_comparison"]["observations"]
checks += 1
_ok = d["verdict"] == "HARNESS_SPECIFIC_NOT_A_TRANSPORTABLE_LAW"
print(f"  {'OK  ' if _ok else 'FAIL'}  verdict: {d['verdict']}")
if not _ok:
    failures.append("code/58's verdict changed; SS5's transport paragraph depends on it")
for o in _obs[1:]:
    check(f"ratio vs matched Sweep C cell ({o['harness'][:28]})",
          o["ratio_vs_sweep_C_matched_cell"], "{:.1f}")
    check(f"achieved operating point ({o['harness'][:28]})",
          o["achieved_operating_point"], "{:.4f}")
checks += 1
# The whole point is that these are matched on operating point.
_ok = d["matched_operating_point_comparison"]["operating_point_spread"] < 0.005
print(f"  {'OK  ' if _ok else 'FAIL'}  the three harnesses are matched on operating point to "
      f"{d['matched_operating_point_comparison']['operating_point_spread']:.4f} AUROC")
if not _ok:
    failures.append("code/58's three harnesses are no longer operating-point matched")

print("\n== Severity relationship: operating point (code/47 Sweep C) ==")
d = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]
check("AUROC0=0.70 gap", d["0.7"]["gap_mean"])
check("AUROC0=0.985 gap", d["0.985"]["gap_mean"])
ratio = d["0.7"]["gap_mean"] / d["0.985"]["gap_mean"]
check("operating-point decline factor", ratio, "{:.1f}")

print("\n== Severity relationship: K scaling refit (code/50) ==")
_evt = load("evt_scaling_refit.json")
d = _evt["fits"]
check("EVT per-cell-sigma R^2", d["evt_per_cell_sigma"]["r_squared"], "{:.3f}")
check("log two-parameter R^2", d["log_two_parameter"]["r_squared"], "{:.3f}")
checks += 1
# The EVT test is UNTESTABLE here, not falsified: sigma is a single training
# trajectory's per-epoch dispersion, not the sampling-noise SD EVT requires.
_ok = (d["evt_per_cell_sigma"].get("verdict") == "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED"
       and _evt["sigma_misspecification"]["verdict"] ==
       "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED")
print(f"  {'OK  ' if _ok else 'FAIL'}  the EVT fit ships labelled UNTESTABLE, not falsified")
if not _ok:
    failures.append("code/50 no longer records the sigma misspecification; SS5, Limitations "
                    "(2a)/(3) and Appendix A all depend on the downgraded verdict")
# M4: the K law must survive restriction to the flat-operating-point range.
_r = d["log_two_parameter_K_ge_10"]
check("K-law restricted to K>=10: b", _r["params"]["b"], "{:.5f}")
check("K-law restricted to K>=10: R^2", _r["r_squared"], "{:.3f}")
checks += 1
_full_b = d["log_two_parameter"]["params"]["b"]
_ok = abs(_r["params"]["b"] - _full_b) / abs(_full_b) < 0.10
print(f"  {'OK  ' if _ok else 'FAIL'}  K law survives restricting to the flat-operating-point "
      f"range (b {_full_b:.5f} -> {_r['params']['b']:.5f})")
if not _ok:
    failures.append("the K law no longer survives the K>=10 restriction; SS5 says it does")
checks += 1
# ...and Sweep A's achieved operating-point drift must be disclosed, not just its target.
_lo, _hi = _evt["achieved_operating_point_full_range"]
_ok = f"{_lo:.4f}" in TEX and f"{_hi:.4f}" in TEX or f"{_lo:.4f}" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  Sweep A's achieved operating-point drift is disclosed "
      f"({_lo:.4f} to {_hi:.4f})")
if not _ok:
    failures.append("Sweep A's achieved operating-point drift is not disclosed in main.tex")

print("\n== K-law degeneracy gate (code/47 Sweep A -> code/50) ==")
_sw = load("selection_multiplicity_sweep.json")
checks += 1
# P0-3: the gate must actually be ENABLED at code/47's own call sites, not just
# defined. Every sweep must ship a per-cell degeneracy record.
_missing = [f"{name}[{k}]" for name in ["sweep_A_K", "sweep_B_sample_size",
                                        "sweep_C_operating_point", "sweep_D_n_val_isolated"]
            for k, v in _sw[name].items() if "degeneracy" not in v]
_ok = not _missing
print(f"  {'OK  ' if _ok else 'FAIL'}  every cell of Sweeps A/B/C/D carries a degeneracy record"
      + ("" if _ok else f" (missing: {_missing[:5]})"))
if not _ok:
    failures.append("code/47 no longer runs with degeneracy_check=True at every call site; "
                    "Appendix A.11 Correction 4 and SS5.3's K-law both depend on the records")
_ev = load("evt_scaling_refit.json")
_nd = _ev["fits"]["log_two_parameter_non_degenerate"]
_con = _ev["fits"]["log_two_parameter"]
check("corrected K-law slope b", _nd["params"]["b"], "{:.5f}")
check("corrected K-law R^2", _nd["r_squared"], "{:.3f}")
check("contaminated K-law R^2 (quoted as superseded)", _con["r_squared"], "{:.3f}")
checks += 1
# The threshold must be READ from code/57, not re-picked here after seeing which
# cells it removes. And no cell may sit near it.
_ok = (_nd["degeneracy_threshold_identical_state_dict_frac"] == 0.50
       and _nd["identical_frac_retained_max"] < 0.50 < _nd["identical_frac_excluded_min"]
       and "code/57" in _nd["threshold_source"])
print(f"  {'OK  ' if _ok else 'FAIL'}  gate threshold 0.50 read from code/57; retained cells "
      f"<= {_nd['identical_frac_retained_max']:.3f}, excluded >= "
      f"{_nd['identical_frac_excluded_min']:.3f} (no cell near the boundary)")
if not _ok:
    failures.append("the K-law degeneracy threshold is no longer inherited from code/57, or a "
                    "cell now sits near it; Appendix A.11 claims both")
checks += 1
# The correction's honest content: the slope survives, the dynamic range does not.
_slope_moved = abs(_nd["params"]["b"] - _con["params"]["b"]) / abs(_con["params"]["b"])
_ok = (_slope_moved < 0.10
       and _nd["dynamic_range_max_over_min"] < 0.5 * _con["dynamic_range_max_over_min"]
       and f"{_nd['dynamic_range_max_over_min']:.2f}" in TEX
       and f"{_con['dynamic_range_max_over_min']:.2f}" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  slope survives the gate ({_slope_moved * 100:.0f}% move) "
      f"but dynamic range falls {_con['dynamic_range_max_over_min']:.2f}x -> "
      f"{_nd['dynamic_range_max_over_min']:.2f}x, and both are in the tex")
if not _ok:
    failures.append("SS5.3 and A.11 state that the K-law's slope survives the degeneracy gate "
                    "while its dynamic range does not; that no longer holds or is not quoted")
checks += 1
_ok = _nd["K_values_excluded_as_degenerate"] == [3, 5, 10] and _nd["K_values_fit"] == [
    15, 25, 45, 75, 135, 225]
print(f"  {'OK  ' if _ok else 'FAIL'}  gate excludes K={_nd['K_values_excluded_as_degenerate']} "
      f"and fits K={_nd['K_values_fit']}")
if not _ok:
    failures.append("the set of degenerate K cells changed; SS5.3 names 1/3/5/10 explicitly")

print("\n== Adaptive-selection control power (code/22) ==")
import numpy as np
d = load("epoch_forcing_confound_control.json")["by_capacity"]
# P0-1 fix: the CORRECTED arm trains on tr2_idx, disjoint from its es_idx
# selection set (mirroring code/43). The superseded arm trained on the full
# tr_idx while selecting on a subset of it. Both retentions are checked, and
# both must appear in the tex -- the corrected one as the reported number, the
# contaminated one as the disclosed artifact.
for cap, want_fixed, want_contam in [("128", 94.2, 37.3), ("384", 79.9, 25.7)]:
    r = d[cap]["placebo_relative_retention_pct"]
    for arm, want in [("clean_matched_adaptive", want_fixed),
                      ("clean_matched_adaptive_contaminated", want_contam)]:
        checks += 1
        got = r[arm]
        ok = abs(got - want) < 0.15 and f"{got:.1f}" in TEX
        print(f"  {'OK  ' if ok else 'FAIL'}  cap {cap} placebo-relative retention "
              f"[{arm[:34]:34s}]: {got:.1f}%")
        if not ok:
            failures.append(f"retention cap {cap} {arm}: computed {got:.1f}%, "
                            f"wanted ~{want}% and present in tex")
    checks += 1
    # The diagnostic that exposes the defect: an arm selecting in-sample runs to
    # the end of the budget, an honest one does not.
    be = d[cap]["best_epoch_stats"]["mean_best_epoch"]
    _ok = (be["clean_matched_adaptive_contaminated"] > be["clean_matched_adaptive"] + 5
           and abs(be["clean_matched_adaptive"] - be["clean_matched"]) < 1.0)
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap} the contaminated arm runs deeper "
          f"({be['clean_matched_adaptive_contaminated']:.1f}) than the corrected arm "
          f"({be['clean_matched_adaptive']:.1f}), which tracks the honest arms "
          f"({be['clean_matched']:.1f})")
    if not _ok:
        failures.append(f"code/22 cap {cap}: the training-depth signature of the "
                        f"superseded control no longer reproduces; A.1 item 16 cites it")
checks += 1
# The corrected control must still be BEATEN by LEAKY -- that is the finding.
_ok = all(d[c]["gaps"]["leaky_minus_clean_matched_adaptive"]["mean"] > 0
          and d[c]["gaps"]["leaky_minus_clean_matched_adaptive"]["wilcoxon_p"] < 0.01
          for c in ["128", "384"])
print(f"  {'OK  ' if _ok else 'FAIL'}  LEAKY still beats the CORRECTED adaptive control at "
      f"both capacities (p<0.01): "
      + ", ".join(f"cap {c}: {d[c]['gaps']['leaky_minus_clean_matched_adaptive']['mean']:+.4f} "
                  f"(p={d[c]['gaps']['leaky_minus_clean_matched_adaptive']['wilcoxon_p']:.4f})"
                  for c in ["128", "384"]))
if not _ok:
    failures.append("code/22's corrected control no longer supports SS4.3's fold-reuse finding")

print("\n== Fidelity extension: budget matching and coupling isolation (code/49) ==")
d = load("mechanism3_fidelity_extension.json")
check("gap vs BUDGET-MATCHED control",
      d["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"])
check("gap vs superseded es-fed control",
      d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]["gap_mean"])
check("coupling continuity, isolated within one harness",
      d["coupling_continuity_isolation"]["gap_mean"])
checks += 1
# P0-2(d): the isolating arm must show that continuous coupling does NOT add
# severity over a one-shot argmax. The paper retracts the opposite claim.
_ok = d["coupling_continuity_isolation"]["gap_mean"] < 0
print(f"  {'OK  ' if _ok else 'FAIL'}  continuous coupling does NOT beat a one-shot argmax "
      f"({d['coupling_continuity_isolation']['gap_mean']:+.4f}); SS4.3 and the checklist "
      f"table retract the claim that it does")
if not _ok:
    failures.append("code/49's isolating arm no longer refutes the coupling-continuity claim, "
                    "which SS4.3 and Table checklist-scope both now state as retracted")
checks += 1
_r = d["sanity_ratios"]
_ok = _r["budget_matched_ratio_inside_comparator_band"] and not (
    _r["comparator_band"][0] <= _r["this_harness_vs_es_fed_control_superseded"]
    <= _r["comparator_band"][1])
print(f"  {'OK  ' if _ok else 'FAIL'}  (LEAKY-CM)/(CM-PLACEBO): budget-matched "
      f"{_r['this_harness_vs_budget_matched_control']:.2f} is INSIDE the comparator band "
      f"{[round(x, 2) for x in _r['comparator_band']]}, superseded "
      f"{_r['this_harness_vs_es_fed_control_superseded']:.2f} is outside")
if not _ok:
    failures.append("code/49's sanity-ratio story changed; SS4.3 states the budget-matched "
                    "ratio is inside the band and the superseded one outside")
checks += 1
_ratio = (d["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"]
          / load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]
          ["by_capacity"]["128"]["leaky_minus_clean_matched"]["gap_mean"])
_ok = f"{_ratio:.1f}" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  corrected like-for-like ratio {_ratio:.1f}x present in tex")
if not _ok:
    failures.append(f"corrected like-for-like ratio {_ratio:.1f}x missing from main.tex")

print("\n== Case Study 4 permutation null, reported in BOTH directions (code/45) ==")
d = load("case_study_4_winners_curse.json")["permutation_significance_audit"]
checks += 1
_ok = d["n_cells_p_below_05"] == 0 and "0 of the 24" in TEX.replace("$", "")
print(f"  {'OK  ' if _ok else 'FAIL'}  {d['n_cells_p_below_05']}/{d['n_cells_total']} cells reach "
      f"p<0.05 against their own permutation null, and the tex says so")
if not _ok:
    failures.append("main.tex must state plainly that 0 of the 24 Case Study 4 cells reach "
                    "p<0.05 against their own permutation null")
checks += 1
# The unfavourable direction: the non-degenerate cells' own null sits ABOVE
# their observed mean. Previously only the favourable direction was reported.
_ok = d["non_degenerate_observed_minus_null"] < 0 and \
    f"{d['non_degenerate_null_mean']:+.4f}".replace("+", "") in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  the non-degenerate cells' null ({d['non_degenerate_null_mean']:+.4f}) "
      f"sits above their observed mean ({d['non_degenerate_observed_mean']:+.4f}), and the tex "
      f"discloses it")
if not _ok:
    failures.append("main.tex must disclose that the non-degenerate cells' own permutation null "
                    "sits above their observed mean, not only the degenerate cells' favourable null")

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

print("\n== Mechanism 5 sample-size sensitivity (code/46, SWEEP=N) ==")
d = load("mechanism5_threshold_selection.json")
checks += 1
_ss = d.get("sample_size_sensitivity")
_ok = _ss is not None
print(f"  {'OK  ' if _ok else 'FAIL'}  Mechanism 5's n_test/n_val dependence is measured, not assumed")
if not _ok:
    failures.append("mechanism5_threshold_selection.json has no sample_size_sensitivity block")
else:
    for n in ["700", "1750", "3500", "10000"]:
        check(f"n={n} F1 gap", _ss["by_n_samples"][n]["f1_gap_mean"])
    check("F1-gap shrinkage factor 140->2000", _ss["shrinkage_factor_700_to_10000"], "{:.1f}")

print("\n== Joint severity surface (code/57) ==")
try:
    d = load("joint_severity_surface.json")
    checks += 1
    _ok = max(v["abs_diff"] for v in d["internal_consistency_vs_sweep_C"].values()) < 5e-5
    print(f"  {'OK  ' if _ok else 'FAIL'}  the grid's K=45 column reproduces code/47's Sweep C")
    if not _ok:
        failures.append("code/57's K=45 column diverged from code/47's Sweep C; harness drifted")
    for name in ["M1", "M3", "M4"]:
        check(f"{name} LOO R^2", d["fits"][name]["loo_r_squared"], "{:.3f}")
    check("interaction F-test p", d["interaction_f_test"]["p"], "{:.3f}")
    checks += 1
    _ok = "not a bound" in d["framing"]
    print(f"  {'OK  ' if _ok else 'FAIL'}  the joint fit ships labelled as an empirical fit, not a bound")
    if not _ok:
        failures.append("code/57's framing no longer disclaims being a bound")
    # ── the degeneracy gate: K=3 produced the old interaction on its own ──
    checks += 1
    _g = d["degeneracy_gate"]
    _ok = _g["passed"] and not _g["cells_exceeding"]
    print(f"  {'OK  ' if _ok else 'FAIL'}  degeneracy gate passed; worst fitted cell has "
          f"{_g['max_identical_frac_in_grid']:.2f} bitwise-identical LEAKY/CLEAN_MATCHED folds")
    if not _ok:
        failures.append(f"code/57's grid contains degenerate cells: {_g['cells_exceeding']}")
    checks += 1
    # K=3 must NOT be in the fitted grid, and its degeneracy must ship as evidence.
    _ok = (3 not in d["grid"]["K_values"] and _g["dropped_K3_column_identical_frac"]
           and min(_g["dropped_K3_column_identical_frac"].values()) > 0.5)
    print(f"  {'OK  ' if _ok else 'FAIL'}  K=3 is excluded from the fit and its degeneracy "
          f"ships as evidence "
          f"({min(_g['dropped_K3_column_identical_frac'].values()):.2f}-"
          f"{max(_g['dropped_K3_column_identical_frac'].values()):.2f} identical)")
    if not _ok:
        failures.append("code/57 refit K=3 or stopped shipping the evidence for dropping it")
    checks += 1
    # Every fitted cell must carry its own degeneracy record.
    _ok = all("degeneracy" in c for c in d["cells"].values())
    print(f"  {'OK  ' if _ok else 'FAIL'}  every fitted cell carries a per-cell degeneracy record")
    if not _ok:
        failures.append("code/57 cells no longer carry per-cell degeneracy records")
    checks += 1
    # The old, degenerate grid's F-test must ship for the record, and must NOT
    # be the number the paper reports.
    _old = d.get("interaction_f_test_on_old_degenerate_grid")
    _ok = _old is not None and abs(_old["p"] - 0.027) < 0.002
    print(f"  {'OK  ' if _ok else 'FAIL'}  the superseded degenerate-grid F-test ships for the "
          f"record (p={_old['p']:.4f} if present)" if _old else "  FAIL  missing")
    if not _ok:
        failures.append("code/57 no longer ships the old degenerate-grid F-test for comparison")
    # NOTE: F(1,16)=5.92 still APPEARS in SS5, but only inside the paragraph that
    # retracts it. It is deliberately not check_absent'd -- the retraction has to
    # be able to quote the number it is retracting. What is guarded instead is
    # the CLAIM the number was used to support (see the check_absent block below
    # for "interact multiplicatively rather than additively").
except FileNotFoundError:
    checks += 1
    print("  FAIL  joint_severity_surface.json missing (SS5 reports the joint fit)")
    failures.append("joint_severity_surface.json missing")

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
check_absent("stale pre-fidelity-port control-vs-placebo gap", "$+0.0498$, $p=3.7")
check_absent("stale coupled-seed Holm verdict (48 and 128)", "at 48 and 128 units")
check_absent("stale coupled-seed CM-placebo range", "gaps $+0.016$ to $+0.033$")
check_absent("stale coupled-seed isotropic range in prose",
             "$+0.0009$ to $+0.0034$ AUROC on the synthetic sweep")
check_absent("pooled-family claim that nothing survives",
             "would leave nothing significant at")
# ── retractions from the fourth adversarial review ──────────────────────────
check_absent("operating point asserted to dominate mechanism differences",
             "That single relationship moves severity more than any")
check_absent("operating-point decline asserted as larger than any mechanism difference",
             "which is larger than any difference we\nmeasure \\emph{between} mechanisms")
check_absent("EVT asserted as falsified (downgraded to untestable)",
             "is falsified by this data once fit as stated")
check_absent("EVT falsification asserted in Limitations",
             "derivation would most naturally rest on is falsified here once fit as")
check_absent("EVT functional form asserted not to fit",
             "The extreme-value functional form, tested as stated,")
check_absent("interaction asserted as a finding in the contributions list",
             "interact multiplicatively rather than additively")
check_absent("code/47's cell called independently written",
             "independently-written default cell")
check_absent("flat 33-layer count for all 24 Case Study 4 cells",
             "argmax, over $33$ layers")
check_absent("Case Study 4's contaminated mean used as the headline CI",
             "$[+0.0032,+0.0100]$")
check_absent("stale non-saturated-vs-saturated contrast",
             "$+0.0042$ versus $+0.0065$ on the non-saturated")
check_absent("replay script claiming every Case Study 2 number",
             "every Case Study 2 number in the paper is")

print("\n== Reverse guard: every AUROC-shaped literal in main.tex traces to a JSON ==")
# WHY THIS EXISTS. Every check above runs in ONE direction: it takes a number
# from a result JSON and asserts it appears in main.tex. That cannot catch a
# number that is in main.tex and in NO json -- e.g. a stale value, or the same
# cell rounded two different ways in two paragraphs. An adversarial review
# found exactly that: the (K=45, AUROC_0=0.80) cell (0.003649) was rendered
# "+0.0037" in one place and "+0.0036"/"+0.00365" elsewhere. This guard closes
# the loop for the class of literal that bug lived in: 4-and-5-decimal
# AUROC/gap-shaped numbers.
import glob as _glob

_json_floats = set()


def _collect(o):
    if isinstance(o, dict):
        for v in o.values():
            _collect(v)
    elif isinstance(o, list):
        for v in o:
            _collect(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        _json_floats.add(abs(float(o)))


for _f in _glob.glob(str(R / "*.json")):
    _collect(json.load(open(_f)))
_reprs = {f"{v:.{p}f}" for v in _json_floats for p in (4, 5)}

# Numbers that legitimately do NOT come from any shipped JSON. Each is here for
# a stated reason, not to silence the check.
ALLOWED_NON_JSON_LITERALS = {
    # Case Study 1 (HaRP): reported from prior work, explicitly NOT
    # re-verifiable from this artifact, and excluded from every comparison (SS4.1).
    "0.1906", "0.9620", "0.7583",
    # Numbers this paper quotes in order to RETRACT or supersede them. The
    # runs behind them are superseded and their JSONs no longer ship.
    "0.0498",   # pre-fidelity-port control-vs-placebo gap (Appendix A)
    "0.00064",  # Wilcoxon p under the superseded label-conditional calibration
    "0.9176",   # operating point of the superseded fidelity-extension run
    "0.00198", "0.00154",  # per-seed MDEs under the superseded label-conditional
                           # calibration, quoted in Appendix A's power check
    # Analytic, not measured: the Bayes-optimal AUROC the calibration bug
    # actually realized for a cell labelled 0.985 (Appendix A, issue 11).
    "0.99994",
}

# Quantiles of Case Study 2's per-rep Delta_sel are DERIVED from a shipped array
# rather than stored as scalars, so the substring trace above cannot see them.
# Recomputing them here is a stronger guarantee than allowlisting: if the array
# changes, this fails rather than silently passing.
_ssc = np.array(load("case_study_2_layer_decomposition.json")
                ["randomized_stratified_splits"]["selection_specific_component_per_rep"])
_q = np.percentile(_ssc, [0, 25, 50, 75, 100])
for _v in _q:
    ALLOWED_NON_JSON_LITERALS.add(f"{abs(_v):.4f}")
checks += 1
_ok = (all(f"{abs(v):.4f}" in TEX for v in _q)
       and f"{int((_ssc > 0).sum())} of {len(_ssc)}" in TEX.replace("$", ""))
print(f"  {'OK  ' if _ok else 'FAIL'}  Case Study 2's per-rep Delta_sel quantiles "
      f"({', '.join(f'{v:+.4f}' for v in _q)}) and its "
      f"{int((_ssc > 0).sum())}/{len(_ssc)} positive-rep count are in main.tex")
if not _ok:
    failures.append("Appendix B.2's per-rep Delta_sel disaggregation does not match the "
                    "shipped per-rep array")

_lits = set(re.findall(r"(?<![\d.])[+-]?0\.\d{4,5}(?![\d])", TEX))
_untraced = sorted(L for L in _lits
                   if L.lstrip("+-") not in _reprs
                   and L.lstrip("+-") not in ALLOWED_NON_JSON_LITERALS)
checks += 1
_ok = not _untraced
print(f"  {'OK  ' if _ok else 'FAIL'}  {len(_lits)} AUROC-shaped literals in main.tex; "
      f"{len(_untraced)} trace to no shipped JSON and are not allowlisted")
for L in _untraced:
    print(f"        untraceable: {L}")
if not _ok:
    failures.append(f"main.tex contains AUROC-shaped literals with no JSON source: {_untraced}")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
