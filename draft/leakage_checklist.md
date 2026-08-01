# A checklist for label-information leakage in hidden-state hallucination probing

Derived from four case studies in this paper -- two internal (HaRP's nested-
CV feature leakage; GUARDIAN's CV-based layer-selection optimism) and two
external (MultiHaluDet's per-fold checkpoint-selection leakage; the
quantized-LLM paper's test-set-driven best-layer selection) -- plus a fifth,
structurally distinct mechanism (MultiHaluDet's test-set-driven
decision-threshold selection) found while auditing the third case study's
own repository. All five are instances of the same root hazard -- **using
the evaluation labels, even indirectly, to make a choice, then reporting
the resulting metric as if that choice were free** -- surfacing through
five structurally different mechanisms. This is the paper's reusable,
actionable contribution: a set of concrete questions a researcher (or
reviewer) can ask of any hidden-state hallucination-detection pipeline to
identify which, if any, of these five mechanisms it is vulnerable to.

## When this checklist applies, and when it does not

Read this table before working through the five questions. Every entry is
grounded in a specific finding in the paper — the candidate-count and
operating-point relationship (main.tex §5), the five mechanisms (§4), and the
transport failure (§5.4) — rather than in a general view of good practice.

### Use it

| Your situation | What this checklist gives you, and on what evidence |
|---|---|
| You report AUROC (or any metric) from a pipeline that selects a checkpoint, layer, threshold or hyperparameter using a statistic computed on — or reused from — the data you then report on. | Q1–Q5 name which of the five mechanisms you have and where in the code to look. Three of the five are verified against pinned commits of published third-party code (§4), so this is a documented failure mode, not a hypothetical one. |
| A validation-fold decision is coupled to something downstream: early stopping, LR scheduling on a reused fold, best-epoch or best-layer selection. | Q2 and Q4. §4.3 shows the *continuity* of the coupling matters: porting a scheduler and epoch-by-epoch early stopping onto a checkpoint-selection harness raises the measured gap by **3.6x** over a single end-of-training argmax, on the same features at a matched calibration. |
| Your headline threshold-dependent numbers (F1, accuracy, MCC, Cohen's kappa, or an ECE derived from thresholded predictions) come from a threshold chosen on the evaluation labels. | Q5, which nothing upstream will catch: AUROC is unaffected by construction, so an AUROC-only audit clears the pipeline while F1 and accuracy are still inflated (+0.021 to +0.034 at n_test = 140; §4.5). |
| You want to know, before publishing a severity estimate, whether it is confounded with the operating point or with how many candidates were tried. | §5 names the two quantities to report alongside it. Within one harness the gap grows as `a + b ln K` (R^2 = 0.970 at a fixed operating point) and falls **48.6x** from AUROC_0 = 0.70 to 0.985; two severity numbers measured at different operating points are not comparable until it is matched. |

### Does not directly apply, or needs adaptation

| Your situation | Why the checklist adds little here |
|---|---|
| Your pipeline has no data-dependent selection step at all — fixed layer, fixed threshold, fixed hyperparameters, all committed before any evaluation data was seen. | None of the five mechanisms can fire and the checklist returns five no's. It is a selection-leakage instrument, not a general evaluation-quality review; clearing it says nothing about the other ways an evaluation can be wrong. |
| Every reported number comes from a held-out set touched exactly once, with no proxy metric reused for selection. | Q1–Q5 are satisfied by construction. What the checklist does *not* check is whether that set is a genuine random partition: Case Study 2's original 18.8-point gap turned out to be an artifact of a fixed sequential `H[:400]`/`H[400:]` slice whose two halves are separable at AUROC 0.734–0.776 (§4.2). |
| Your task runs near ceiling (reported AUROC >= ~0.98). | Still worth asking, but expect small answers and read them carefully. The same mechanism gives +0.0002 at AUROC_0 = 0.985 against +0.0093 at 0.70, and Case Study 4's ceiling-saturated cells give +0.0042 against +0.0157 on the non-saturated ones. A near-zero severity estimate here is weak evidence that a pipeline is clean. The converse is where this checklist earns most: far from ceiling, and at small n_test, is where every effect we measure is largest. |
| You want a severity *number* for your own pipeline, not a yes/no. | The checklist does not supply one, and §5's surface should not be used as a calculator: §5.4 finds three measurements at achieved operating points within 0.003 AUROC of each other exceeding its prediction by 14x and 52x. Measure it in your own harness against a matched control. |
| You want this run automatically over a corpus. | Not available from this paper. §6's regex scanner returned 0 true positives over 7 repositories and missed both known bugs, for two causes the paper diagnoses specifically. This is a manual code-reading protocol. |

## The five mechanisms, as questions to ask of a pipeline

### 1. Full-dataset fit-then-score leakage (severity: SEVERE — Case Study 1)
**Ask:** Is there any feature (a probe's predicted probability, a class
centroid, a fitted classifier's margin/embedding) that was computed by
fitting *something* on the full dataset's labels, before an outer
cross-validation loop later "holds out" a fold of that same dataset?
**Red flag pattern:** `feature_column = model.fit(X_all, y_all).predict(X_all)`,
followed later by `cross_validate(other_model, features_including_that_column, y_all, cv=...)`.
**Fix:** recompute the feature out-of-fold -- refit the inner
model/centroid using only the current fold's training indices, and only
score it on that fold's held-out indices.
**Reported severity (this paper, Case Study 1):** +0.19 AUROC inflation
(0.962 leaked -> 0.771 corrected), on real Qwen 2.5 3B hidden-state data.
**Evidentiary caveat:** this figure is reported from prior work and is
**not** independently re-verifiable from the artifact submitted with this
paper — no supporting script, log, or data file is included. It is
excluded from the paper's abstract severity range and from every
cross-mechanism comparison. Every other severity number in this checklist
is code-verified against shipped scripts and result JSONs.

### 2. Per-fold checkpoint-selection leakage (severity: MECHANISM REAL AND CODE-VERIFIED; SMALL, CAPACITY-INCONSISTENT RESIDUAL ONCE BUDGET- AND SEED-MATCHED — Case Study 3)
**Ask:** Inside a CV fold, is a model trained across multiple epochs/steps,
with the "best" checkpoint chosen by evaluating on the *same* fold's
held-out indices whose features/predictions later get used downstream as
if they were clean out-of-fold outputs?
**Red flag pattern:** `for epoch in range(N): ... if auc(val_idx) > best: keep_checkpoint()`,
followed by `oof_features[val_idx] = extract(kept_checkpoint, val_idx)`.
**Fix:** carve out a *third*, disjoint split from the training data
specifically for checkpoint/epoch selection, so the fold whose features
get reused downstream never influences which checkpoint is kept. **A
second, easy-to-miss fix requirement, learned the hard way in this
paper's own reconstruction (see lesson 5 below): if your "clean" control
condition also trains on less data than the leaky one because of the
carve-out, budget-match it (retrain on the full data for the epoch count
chosen from the carve-out) before comparing leaky vs. clean, or the
comparison is confounded by training-data budget, not leakage.**
**Measured severity (this paper, Case Study 3, controlled synthetic
reconstruction, corrected calibration and budget-matched control, n=100
seeds, capacities 16/48/128/384):** the original (budget-confounded)
LEAKY-CLEAN gap is statistically significant at all four capacities
tested (+0.0031 to +0.0062 AUROC, p=0.00003-0.0496) but shows no clear
capacity trend. Once training-data budget \emph{and} random seed are
both matched (LEAKY vs. CLEAN_MATCHED -- an earlier draft matched budget
but not seed, and wrongly found no significant residual anywhere), the
gap is individually significant at 2 of 4 capacities (16 and 128 units,
p=0.019 and p=0.0012, of which only capacity 128 survives Holm-Bonferroni
within that family), and underpowered rather than null at 48 and 384
(p=0.171 at both, against MDEs of 0.0023 and 0.0039). Three earlier drafts of this checklist entry (one
claiming "not significant even pooled," one claiming "significant,
confirmed real, capacity-growing," one claiming "not significant
anywhere once budget-matched") were each wrong in a different way, fixed
only by adding the missing control at each step -- including, in the
final round, catching that the budget-matching fix itself had introduced
an unmatched random seed. The mechanism itself remains real and
code-verified in the audited pipeline; the final, best-supported estimate
of its severity in this reconstruction is a small, real, but
capacity-inconsistent residual, not a null and not a clean capacity-growing
trend.

**A further fidelity extension: this may still understate the real
mechanism.** The checkpoint-selection-only harness above ports only the
final "keep the best-val-AUC checkpoint" decision. MultiHaluDet's actual
trainer also steps a `ReduceLROnPlateau` scheduler and ties early stopping
to the same validation fold's AUC *every epoch*, not just once at the end
-- a continuous, epoch-by-epoch reactive coupling to validation feedback.
Porting this mechanic in (`code/49_mechanism3_fidelity_extension.py`) took
two further rounds of correction to isolate properly: an early version
selected the control's retrain epoch count from the leaky run's own
training loop (which can never differ from it by construction); the fix
for that then discarded the correctly-trained control model and retrained
from scratch with no scheduler at all, silently confounding "reuses the
leaky fold" with "has an adaptive LR/early-stopping mechanism at all"
since the leaky and placebo conditions both kept their scheduler. The
final, corrected control keeps the model already trained with a real
scheduler on a genuinely disjoint carve-out, rather than discarding it.

**Two further corrections landed on this harness afterwards** (Appendix A,
issues 12-13). First, its early-stopping break used the LR scheduler's
`patience=3` rather than the audited repo's own `config.patience = 15`
(`src/config.py:19`) -- `patience=3` governs only `ReduceLROnPlateau`'s LR
reduction (`src/training/trainer.py:76`). Second, the calibration transform
it ran on was replaced with a fully label-free one after the previous
version was found to still condition the shrinkage on each point's own
label. Because both landed together, they were varied factorially rather
than reported jointly (`code/54`, n=100 seeds per cell):

A third correction landed later still (main.tex Appendix A, issue 14): the
harness was not actually faithful to the audited trainer's *optimizer or
data pipeline* — it ran Adam at a 10x learning rate, full-batch (one
gradient step per epoch), with the wrong scaler, no warmup and no gradient
clipping. Those are now ported, and the table below is the ablation rerun
under that port, so all four cells are on the same footing:

| Calibration (LEAKY operating point) | ES patience 3 | ES patience 15 |
|---|---|---|
| Superseded label-conditional (0.979 / 0.981) | +0.0081 | +0.0052 |
| Label-free axis-noising (0.920 / 0.942) | +0.0342 | **+0.0338** |

**A claim this document previously made is retracted here.** Before the
fidelity port, the patience correction shrank the gap at one calibration and
grew it at the other, and that sign flip was presented as the
operating-point relationship appearing inside a second harness. Under the
ported training loop the sign does not flip: the patience correction shrinks
the gap at both calibrations (-0.0029 and a negligible -0.0004). The earlier
flip was an artifact of the full-batch, 10x-learning-rate loop. What the
table does establish is that the *calibration* fix, not the patience fix,
moves this number: +0.0081 -> +0.0342 at patience 3 and +0.0052 -> +0.0338
at patience 15.

The reported cell is the bottom right: **+0.0338** (BCa 95% CI
[+0.0279, +0.0405], Wilcoxon p=6.2e-15, paired permutation p<0.0001, 28
tied absolute differences of 100 pairs). Against the
checkpoint-selection-only harness run on the same features under the same
label-free calibration (+0.0093 at capacity 128), that is a genuinely
like-for-like ratio of **3.6x**. The earlier "roughly 3-4x larger than
the checkpoint-selection-only estimate" claim in this document, the
intermediate "5.2x", and the "2.4x" that preceded the fidelity port are all
**superseded**: the first two compared across
differently-calibrated harnesses at different operating points, which
this paper's own Appendix A says is invalid. The residual non-equivalence
this document used to disclose — the two harnesses landing at LEAKY
operating points of ~0.918 and ~0.940 — has largely closed under the
fidelity port (now 0.9424 vs 0.9403), so 3.6x is no longer reported as an
upper bound on that account. **What "fidelity" does not cover, stated so it
is not over-read:** the audited trainer's EMA, its composite
BCE/focal/asymmetric/contrastive objective, class rebalancing, label
smoothing, mixup, cutmix and its 6-layer transformer are all *not* ported.
This measures fidelity of the validation-signal coupling and the optimizer
schedule only. The scheduler-bearing
control itself still beats PLACEBO by +0.0211 (p=3.6e-7) under the ported
training loop, so it is a strongly informative, non-degenerate honest
baseline rather than a straw one. **Lesson: when a real
pipeline reacts to validation feedback continuously (a scheduler, early
stopping), a "clean" control that reacts to it not at all is a different,
easy-to-underestimate confound in its own right -- match the mechanism's
presence, not just its final output, between leaky and clean conditions.**

### 3. Test-set-driven multiple-hypothesis selection (severity: small, mechanism confirmed — Case Study 4)
**Ask:** Is a "best" layer, probe type, or model configuration chosen by
taking the argmax of a metric computed on the *same test set* that gets
reported as the paper's headline number, across many (layers x seeds x
configs) candidates, with no correction for the number of things tried?
**Red flag pattern:** `for layer in all_layers: results[layer] = eval_on_test(...)`,
then `best = max(results, key=results.get)` and `report(results[best])`.
**Fix:** either select the best configuration using a validation split
disjoint from the test set, or apply a multiple-comparisons-aware
correction (e.g., report the test AUROC of the layer chosen by a
validation-only criterion, not the test-set argmax).
**Measured severity:** initially believed to require re-extracting hidden
states from the audited 7B-scale models; a fresh review found this
unnecessary, since the audited repository ships per-layer, per-seed
AUROC arrays for all 24 model/dataset/probe-type combinations tested.
Selecting the best layer on two of three seeds and evaluating on the
third held-out seed gives a directly-measured winner's-curse estimate:
mean $+0.0076$ AUROC across the **17 of 24 combinations on which the
estimator is non-degenerate** (BCa 95% CI $[+0.0048,+0.0134]$, Wilcoxon
$p=1.5\times10^{-5}$; max $+0.0347$), small and consistent in direction with
Mechanism 3's.

> **Two corrections are folded into that number, and the second matters more
> than the first.** (1) An earlier estimator took its naive baseline from a
> max over *all three* seeds including the held-out one. (2) Rebuilding it as
> a leave-one-seed-out rotation removed that stated cause but left the defect
> intact: the rotation estimator is *algebraically* zero whenever all three
> rotations pick the same layer (it reduces to `(3/2)*(abar - a_r)` averaged
> over `r`, which is identically 0), which is true in 7 of 24 cells. All 7
> fall in the non-saturated subgroup, so that subgroup's previously-quoted
> $+0.0065$ was 5 real measurements averaging $+0.0157$ diluted by 7
> algebraic zeros. The headline above therefore excludes the 7 degenerate
> cells rather than averaging them in; their permutation null averages
> $+0.0179$, so those exact zeros are the estimator failing, not the effect
> being absent. `main.tex` §4.4 and `code/45` document this in full.

Note also that 12 of the 24 cells are ceiling-saturated. The honest
saturated-versus-non-saturated contrast is $+0.0042$ versus $+0.0157$ (a 3.7x
difference, comparing against the 5 non-saturated cells the estimator can
actually see), **not** the $+0.0042$ versus $+0.0065$ an earlier version of
this line quoted.

### 4. Selection-based optimism in cross-validated hyperparameter choice (severity: real, but selection-specific component small once measured with a proper random split — Case Study 2)
**Ask:** Was a hyperparameter (a layer, a regularization strength, a
threshold) chosen by maximizing a cross-validated metric, and is that same
CV number then reported as the model's performance, without a further,
genuinely held-out evaluation at the chosen setting?
**Red flag pattern:** `best_layer = argmax(cv_auroc_per_layer)`, then
`report(cv_auroc[best_layer])` with no separate held-out check.
**Fix:** after selecting via CV, evaluate exactly once more on a disjoint
held-out set that played no role in the selection, and report that number
as the headline result. **A second, easy-to-miss fix requirement, learned
the hard way in this paper's own GUARDIAN case study: make sure that
disjoint held-out set is an actual random partition, not a fixed slice of
whatever order the data happens to be stored in** -- see below.
**Measured severity (this paper, Case Study 2):** 18.8 AUROC points
(CV-selection number 0.804 vs. true held-out 0.616) at N=700, GUARDIAN's
own Mistral-7B pipeline, at the specific layer (L11) GUARDIAN's own
argmax selected.
**Correction, found by a subsequent independent review:** GUARDIAN's own
held-out split is `H[:400]`/`H[400:]`, a fixed sequential slice, not a
random partition -- the two halves are separable by a hidden-state probe
at AUROC 0.734-0.776, i.e. they are systematically different
populations. The *selection-specific* component (gap at the layer that
split's own argmax actually selects, minus the mean gap across all
layers) is $-0.004$ under GUARDIAN's own sequential split (where L11 is
selected) -- indistinguishable from zero. **A second review caught a bug
in this correction itself**: an earlier version evaluated every
randomized-split rep at a hardcoded L11, rather than at that rep's own
selected layer -- which does not measure selection optimism, since a
different split can select a different layer. Fixed: every rep now uses
its own argmax layer. Reversing which half selects picks a different
layer entirely (L17, not L11) and gives a component of $+0.074$ -- a
large swing in both which layer is picked and what the component
measures, itself evidence the sequential-split number tracks split
order, not a stable selection-optimism estimate. Recomputing under 50
randomized stratified splits (raised from 8 after a review noted that a
small, unexplained rep count cannot be distinguished from one chosen after
inspection; the stopping point is set by the size of the shipped replay
artifact, ~100 KB/rep, not by runtime), each evaluated at its own selected
layer, gives a mean selection-specific component of $+0.0255$ (SD $0.0186$;
$t=9.71$; Wilcoxon $p=7.7\times10^{-12}$) -- small but, for the first
time, correctly measured and consistently positive, about 7x
smaller than the sequential split's apparent effect. Separately, the
*general* CV-vs-held-out gap averaged across all 32 layers is $+0.192$
under the sequential split but only $-0.0023$ (SD $0.0541$) under the same
randomized splits -- indistinguishable from zero.

**Two caveats on that number, both stated in main.tex §4.2 and repeated here
because this document is read on its own.** (i) All 50 reps resample splits
of the *same* 700 samples, so the SD is split-to-split variability under one
dataset, not sampling variability across independent draws; the t-test's
i.i.d. assumption is violated and its p-value is a within-dataset robustness
statistic, not a population-generalizing significance test. (ii) The
selection-specific component is positive *in expectation even under a
pure-noise null*, because the selected layer is the argmax of CV AUROC and
the gap contains that same CV term. That is exactly the winner's curse the
metric is built to measure, but it means its significance is not evidence of
anything beyond argmax-over-a-noisy-statistic. **The corrected lesson
reverses, not just narrows, the original 18.8-point headline:** the
"large, real, ~19-point gap present at essentially every layer" was
itself entirely an artifact of the sequential split's
population-difference confound; under proper randomization there is no
general CV-vs-held-out optimism in this setup at all. What survives is
narrower but genuine: a small, statistically significant,
selection-specific penalty from CV-based argmax-over-layers selection,
about 7x smaller than GUARDIAN's original headline number. **Lesson:
"held out" must mean "randomly partitioned," not merely "a different
index range" -- and when computing a *selection-specific* quantity
across multiple resamples, evaluate each resample at its own selected
layer, not a layer hardcoded from one particular (e.g. the original)
split.**

### 5. Test-set-driven decision-threshold selection (severity: leaves AUROC exactly unaffected by construction, but materially inflates every threshold-dependent metric — Mechanism 5, found while auditing Case Study 3's own repository)
**Ask:** Is a decision threshold (or other operating point: a Youden-J
cutoff, an F1-maximizing cutoff) chosen by optimizing a metric on the
*test* labels themselves, with that same test set then used to report
every threshold-dependent metric (F1, accuracy, MCC, Cohen's kappa,
balanced accuracy, or an ECE computed from those threshold-derived
predictions) at that selected threshold?
**Red flag pattern:** `best_thresh = find_best_thresholds(probs_test, y_test)`,
then `f1_score(y_test, (probs_test >= best_thresh).astype(int))` reported
as the headline metric.
**Fix:** select the threshold on an independent validation split that
never touches the test labels, then apply it as-is to the test set.
**Measured severity (this paper, isotropic-Gaussian calibrated synthetic
reconstruction, `code/46_mechanism5_threshold_selection.py`, $n=200$
seeds):** F1 gaps (leaky-selected threshold vs. honestly-selected
threshold) ranging from $+0.0212$ (AUROC target $0.985$, MultiHaluDet's
own reported regime) to $+0.0338$ (AUROC target $0.80$), all significant
at $p<10^{-31}$; accuracy gaps of comparable magnitude ($+0.0215$ to
$+0.0457$).

> **⚠ READ THE RANGE ABOVE WITH ITS TEST-SET SIZE, OR YOU WILL MISAPPLY IT.**
> Every number in that range was measured at **$n_{\text{test}}=140$**, and
> this mechanism's severity depends strongly on $n_{\text{test}}$ — more
> strongly than on anything else measured about it. It is a winner's curse
> over a *finite-sample* criterion: the leaky threshold is chosen by argmax
> over a grid scored on the test set, and the noisier that scoring is, the
> more the argmax overfits it. Holding the operating point and every other
> setting fixed and varying only the sample size, the F1 gap falls
> monotonically and by a large factor:
>
> | $n_{\text{test}}$ | 140 | 350 | 700 | 2000 |
> |---|---|---|---|---|
> | F1 gap | $+0.0212$ | $+0.0111$ | $+0.0067$ | $+0.0034$ |
> | accuracy gap | $+0.0215$ | $+0.0120$ | $+0.0066$ | $+0.0036$ |
>
> That is a **$6.2\times$ shrinkage from $n_{\text{test}}=140$ to $2000$**
> (`code/46`, `SWEEP=N`; main.tex §4.5). Practical consequence for using this
> checklist: **do not quote the $+0.021$–$+0.034$ range for a pipeline whose
> test set is not about $140$ samples.** Scale your expectation down for a
> larger test set. The mechanism is still real and still worth fixing at any
> $n_{\text{test}}$ — the fix costs nothing — but its *magnitude* on a
> few-thousand-sample test set is a few tenths of a point of F1, not a few
> points. The direction of the dependence also means the reverse is true: on
> a test set smaller than $140$, expect *more* than $+0.034$.

**Important caveat, found in this paper's own first attempt
at reporting this result:** an earlier draft's severity numbers were
themselves computed from a synthetic generator with the exact calibration
bug this checklist's own lesson-5a warns about (class-mean offset off by
a factor of 2, quadrupling the realized effect size) -- fixed here, and
now guarded against by a
non-tautological calibration check (`code/sanity_checks.py`). **That check
is not universal and its assumption is load-bearing:** its bias correction
assumes identity within-class covariance, so it is not directly applicable
to this project's anisotropic generators (`code/27`) without modification --
not merely un-retrofitted to them. See main.tex Appendix A, issue 11. **A second
caveat: this mechanism's severity is not directly comparable to Mechanism
2's AUROC-scale severity above.** AUROC is threshold-free by
construction, so test-set threshold selection has *exactly zero* effect
on it; the F1 gap is also structurally guaranteed non-negative, since the
leaky threshold is chosen by an argmax over the same grid the test F1 is
then scored on. An earlier draft of this paper described Mechanism 5 as
"an order of magnitude more severe" than Mechanism 2 by comparing these
F1/accuracy gaps directly to Mechanism 2's AUROC gaps -- a category
error, since the two measure different things, now corrected. The honest
statement: this mechanism leaves AUROC entirely unaffected, but
materially inflates every other metric the same pipeline reports
alongside it -- a distinct, real leak in its own right, not a directly
rankable "more or less severe."

## Using this checklist

For any hidden-state hallucination/factuality-probing paper (your own, or
one you are reviewing), work through the five questions above in order.
None of them require re-running the paper's experiments to *ask* -- each
is answerable by reading the training/evaluation code directly, the way
all four case studies (plus Mechanism 5, found while auditing one of
them) in this paper were identified. Quantifying the resulting inflation,
if you choose to, requires either the paper's released code+data (Case
Studies 1-3) or a controlled, difficulty-calibrated synthetic
reconstruction of the specific mechanism when the original compute scale
isn't reproducible (Case Study 3's and Mechanism 5's approach,
generalizable to any of the five patterns).

## An additional, cross-cutting lesson: quantifying a mechanism needs a placebo, not just a clean/leaky pair

Case Study 3's own reconstruction taught us this the hard way, via
independent adversarial review. A first attempt at quantifying mechanism
2 (Section 4.2) reported a capacity-dependent gap between a LEAKY
condition (checkpoint selected on the reused fold) and a CLEAN condition
(checkpoint selected on a disjoint carve-out) that grew with model
capacity but was individually non-significant at every tested capacity
(n=10 seeds, all p >= 0.105). Independent review correctly flagged two
distinct unresolved questions this pattern could not distinguish: (a) is
the effect simply underpowered, or genuinely absent, and (b) is a growing
gap evidence of genuine label-peeking, or just generic capacity-driven
*variance* in any checkpoint-selection process (even a clean one) as
models get larger? A CLEAN-only control cannot answer (b), because CLEAN
still uses real, non-permuted labels -- it is a *different, non-leaky*
mechanism, not a *zero-signal* baseline.

We resolved both by adding a third condition, PLACEBO: checkpoint
selection using the same held-out fold and capacity as LEAKY, but with
the selection labels randomly permuted -- identical variance profile,
**zero *selection*-signal, not zero signal overall: PLACEBO still trains
via ordinary gradient descent on the real, unpermuted training labels
(`y_train[tr_idx]`); only the checkpoint-selection criterion sees
permuted labels (`y_val_permuted`), which is why its mean AUROC sits at
0.727-0.738 across the four tested capacities (`results/corrected_capacity_placebo_sweep.json`),
not 0.5.** Re-running at 10x the seed count (n=100) at the
flagship capacity found the LEAKY-CLEAN gap statistically significant
(p=0.0385), and we initially reported this as confirming a genuine,
modest leak.

**A third round of review found this was still wrong, in two further
ways -- both worth stating as their own lessons.**

**Lesson 5a: verify your calibration formula's inversion, not just its
target value.** The synthetic task's difficulty was calibrated via
AUROC = Phi(sqrt(J)/2) -- inverted to solve for a target J from a desired
AUROC of 0.80. That formula is wrong: it is the equal-prior Bayes-
*accuracy* identity, not the binormal AUROC identity, which is
Phi(sqrt(J/2)). The two coincide only if 2 = sqrt(2), which they do not.
Using the wrong inversion, our "0.80-AUROC task" was actually an
0.883-AUROC task -- consistent with the ~0.87-0.88 AUROCs we observed and
did not think to double-check against the intended 0.80. **If your
severity estimate depends on calibrating a synthetic task to a target
difficulty via a closed-form identity, verify the achieved AUROC matches
the intended one empirically, not just that the formula "ran."**

**Lesson 5b: a clean/leaky comparison must also match training-data
budget, or the comparison measures data quantity, not leakage.** Our
CLEAN condition selected its checkpoint using a disjoint carve-out of the
training fold -- which meant CLEAN trained on ~15% less data than LEAKY
at every capacity. Adding a budget-matched control (CLEAN_MATCHED: same
disjoint-fold epoch selection, but retrained from scratch on the same
full data budget as LEAKY) and rerunning the full capacity sweep at
n=100 seeds throughout, under the corrected calibration, initially found
the budget-matched gap was **not significant at any capacity**
(p=0.07-0.36) -- an apparent resolution.

**Lesson 5c: a "fixed" control must itself be checked for new confounds
it may have introduced.** A fourth round of review caught that
CLEAN_MATCHED's from-scratch retraining used a *different random seed*
than LEAKY and CLEAN, adding an independent source of variance that
diluted the budget-matched comparison's own power -- the "no significant
residual anywhere" conclusion was itself partly an artifact of the fix's
implementation, not purely of the training-budget correction. Matching
the seed (so CLEAN_MATCHED differs from LEAKY only in epoch-count
provenance, not also in weight initialization) and rerunning gives the
actual result: the budget-matched gap is individually significant at 2 of 4
capacities (16 and 128 units), of which only capacity 128 survives
Holm-Bonferroni across the four-capacity family, and underpowered rather
than confidently null at 48 and 384 units. (These are the numbers after a
later closure review also decoupled the data/split/fold/init seeds in
`code/02d`, which left the magnitudes essentially fixed but moved which two
capacities clear significance -- itself a caution against reading per-cell
significance labels as stable.) **A fix for one confound is not
exempt from introducing another; re-derive or spot-check every random
seed, split, or initialization that changes between your "leaky" and
"clean" conditions, not only the one variable you set out to control.**
What looked, after lesson 5's placebo, like a confirmed genuine leak, and
then (before lesson 5c) like entirely a training-budget artifact, turned
out to be a small, real, but capacity-inconsistent residual -- neither
of the two cleaner-sounding intermediate conclusions.

**Recommendation for anyone quantifying a suspected leakage mechanism via
synthetic reconstruction:** report power (or run enough seeds to reach
it) before drawing a trend conclusion from individually non-significant
numbers; verify your calibration formula's inversion empirically, not
just algebraically; match training-data budget between clean and leaky
conditions before attributing any gap to leakage; check that the fix
introduced to match one variable (e.g. training budget) did not
introduce a new one (e.g. a different random seed); and include a
permuted-label placebo alongside any clean/leaky pair to separate genuine
peeking from generic capacity-driven variance. Each of these five
controls was independently necessary in this paper's own reconstruction
-- missing any single one produced a confident-sounding but wrong
conclusion, and it took four separate rounds of independent adversarial
review to accumulate all five. Also check your own pre-registered
decision rule against the final data rather than narrating past it: ours
returns \texttt{MIXED}, and we report that rather than a cleaner-sounding
"confirmed" or "refuted" story. This is the checklist's own recursive
lesson, now demonstrated rather than merely asserted: an audit of leakage
severity is itself vulnerable to under-powered, miscalibrated,
budget-confounded, seed-confounded, or un-controlled claims if it isn't
held to the same standard it asks of the papers it audits.
