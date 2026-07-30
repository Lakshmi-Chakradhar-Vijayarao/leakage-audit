# Not All Leaks Are Equal: A Taxonomy and Audit of Label-Information Leakage in Hidden-State Hallucination Detection

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

Linear and shallow-MLP probes on LLM hidden states are a standard tool for
hallucination detection, with reported AUROCs routinely exceeding 0.90. We
identify four distinct mechanisms by which evaluation labels leak into
these numbers, each requiring a different fix: (1) full-dataset
fit-then-score feature leakage feeding a nested cross-validation loop
(+0.19 AUROC inflation, verified in our own prior pipeline, HaRP); (2)
cross-validation-based hyperparameter selection reported as held-out
performance (an 18.8-point gap between the CV-selected and true held-out
AUROC, in a second pipeline of ours, GUARDIAN); (3) per-fold checkpoint
selection using the same fold later reused as "out-of-fold" features,
verified directly in a published multilingual hallucination-detection
pipeline's (MultiHaluDet) released code; (4) test-set-driven
best-of-many-layers selection with no multiplicity correction, verified in
a second published paper's code.

Mechanisms (1)-(2) are demonstrated in our own previously-unpublished
pipelines; we release the training code and logs for independent
verification, though none has been done yet. Mechanisms (3)-(4) are
verified against pinned commits of externally-authored code -- a stronger
evidentiary tier.

Severity differs sharply and is not comparable across mechanisms.
Mechanism (1) is catastrophic regardless of scale. Mechanism (3)'s
severity, quantified via a calibrated synthetic reconstruction ($n=100$
seeds, four capacities, budget- and seed-matched), is small: $+0.0009$ to
$+0.0034$ AUROC, significant at two of four capacities after
Holm-Bonferroni correction. A fifth control condition, isolating fold
reuse from "having adaptive checkpoint selection at all," finds the gap
under this stricter test larger, not smaller ($+0.0143$ to $+0.0198$,
$p<0.0001$ at both tested capacities) -- evidence against, though not a
definitive exclusion of, this alternative explanation, since the control
condition's own behavior is itself not fully explained. A second
generative process, replacing the isotropic-Gaussian calibration with a
real, anisotropic covariance structure fit from real hidden-state
features, replicates the severity gap at most tested capacities and is
if anything larger at capacity 128 -- evidence against the isotropic
assumption alone being responsible for the estimate. A real-Mistral-7B-feature
test of the same mechanism, corrected for an architecture mismatch
(Appendix A), could not ultimately be brought onto the synthetic sweep's
operating point: the calibration method that rescales only the
between-class mean while preserving real within-class structure exactly
turned out to be incapable of controlling the downstream pipeline's
achieved discriminability for these features (still $\geq0.93$ AUROC even
at zero mean separation, since most of the real signal lives in
higher-order structure the method never touches). This test is retired
as uninformative about transfer rather than reported as evidence for or
against it (Appendix A); this paper's severity estimate for Mechanism 3
rests on the two synthetic sweeps alone.

We provide a checklist covering all four mechanisms and test whether it
can be automated: a regex-based scanner run over 7 repositories (the 2
already-confirmed-leaky case studies, to test false negatives, plus 5
additional, not-previously-examined repositories, to test false
positives) produces zero true positives and misses both known bugs --
this class of leakage resists simple pattern-matching in both
directions.

## 1. Introduction

Detecting hallucination by probing an LLM's internal hidden states is one
of the most active subfields in LLM reliability research: a linear probe
is cheap to train and needs no additional model calls, and reported
AUROCs are often striking -- 0.90 or higher, sometimes a perfect 1.000.
This is also a setting with few samples and many dimensions (hundreds of
samples, thousands of hidden dimensions), where high reported numbers
should invite scrutiny of the evaluation protocol that produced them.

This paper began by accident. While building our own hidden-state
hallucination detector (HaRP, targeting Qwen2.5-3B), we found and fixed a
nested-cross-validation leakage bug that had inflated our own reported
AUROC by 0.19 points. That raised a question: is this an isolated
mistake, or a structural hazard the broader literature shares? We find
four *distinct* mechanisms by which evaluation labels can leak into a
reported metric, verify two of them in externally published, code-available
pipelines, and show via controlled reconstruction that these mechanisms
differ substantially in how much inflation they actually cause -- a
distinction the field currently has no vocabulary for.

**Contributions:**
1. A verified, quantified nested-CV feature-leakage bug in our own
   pipeline (HaRP), with an independent cross-check confirming the fix.
2. A verified, quantified CV-based hyperparameter-selection optimism bias
   in a second pipeline of ours (GUARDIAN): an 18.8-point gap between the
   CV-selection number and genuine held-out performance.
3. A code-verified identification of a third, distinct mechanism
   (per-fold checkpoint selection) in an externally published pipeline's
   released code, plus a controlled synthetic reconstruction -- calibrated
   via a closed-form Fisher-ratio/AUROC identity -- that quantifies this
   mechanism's severity and tests, via an explicit control condition,
   whether an alternative confound could explain it instead.
4. A code-verified identification of a fourth, distinct mechanism
   (test-set-driven best-of-many-layers selection) in a second externally
   published paper's released pipeline.
5. A checklist synthesizing all four mechanisms into actionable questions,
   and an attempt to automate it that is itself informative about how hard
   this class of bug is to detect mechanically.

## 2. Related Work

**General methodological leakage.** Leakage -- evaluation-target
information influencing a reported metric -- is well documented in data
mining (Kaufman et al. 2012) and in model-selection statistics:
cross-validation-based hyperparameter or feature selection is optimistic
unless nested inside the outer evaluation loop (Varma & Simon 2006). Case
Studies 2 and 4 are direct instances of this general phenomenon in
hidden-state probing; Case Studies 1 and 3 are more specific to the
two-stage feature-extraction-then-classification structure common in this
literature. To our knowledge, no prior work names this as a distinct
pattern here.

**Calibration and benchmark-construction critiques.** A large, separate
literature asks whether LLM confidence is calibrated to correctness
(surveys, hidden-state-based confidence estimation, conformal-prediction
guarantees); this work treats the evaluation protocol as given, whereas
our contribution sits upstream of it -- an inflated AUROC corrupts any
downstream calibration analysis built on it. Closer to our concern,
PARALLAX (arXiv 2605.17028) audits 22 detection methods across 12 models
and 6 corpora and finds four of six corpora leak the ground-truth answer
directly into the input prompt; once controlled for, most baselines fall
to near chance. This is independent, larger-scale evidence that the
field's reported AUROCs are frequently protocol or construction
artifacts, though PARALLAX's mechanism (benchmark construction) is
distinct from all four mechanisms documented here (each of which concerns
a downstream selection step, not the benchmark's own construction).
Separately, surface-form correctness metrics (ROUGE-L, exact match) are
known to diverge from LLM-judge or human factual-correctness judgments --
a related but distinct label-quality concern this paper does not address.
Fixing the leakage patterns we identify does not, by itself, address
label-quality or benchmark-construction problems in the underlying
dataset.

**Pretraining-data contamination.** A separate 2023-2026 thread asks
whether a benchmark's answers appeared in an LLM's pretraining corpus
(Shi et al. 2024; Golchin & Surdeanu 2024; Sainz et al. 2023) -- a
different failure mode from ours: contamination concerns what the model
has already seen, not whether a downstream selection step used test-set
signal after the model was already fixed. The two are complementary and,
in principle, additive; we do not audit our four case studies' benchmarks
for contamination.

To our knowledge, no prior paper both (a) names and distinguishes
multiple, structurally different label-leakage mechanisms specific to
hidden-state probing pipelines, and (b) quantifies their relative
severity via controlled reconstruction rather than assuming all leakage
is equally damaging.

## 3. Method: How These Case Studies Were Selected and Verified

Two case studies (§4.1-4.2) come from our own prior work, where we had
full access to code, data, and commit history. The external case studies
(§4.3-4.4) required three things: a real, runnable public code
repository, verified directly via the repository host, not taken on
faith from the paper's text; a reported AUROC in the range that warrants
scrutiny (0.90 or higher); and an evaluation protocol whose leakage risk
we could assess by reading the actual training/evaluation code, not just
a methodology section. We did not limit the search to K-fold setups
specifically, since one candidate paper's bug (§4.4) lives in a single
train/val/test split with no K-fold structure at all -- "avoid K-fold" is
not itself a fix for this family of hazards.

## 4. Four Case Studies

**Evidentiary status.** These four case studies are not equally
verifiable by a reader of this repository. Case Studies 3 (MultiHaluDet)
and 4 (quantized-LLM paper) are external, published, code-available
systems: their claimed leaks are demonstrated against pinned commits of
publicly released third-party code, and this paper's own audit scripts
and result JSONs are included in full. Case Studies 1 (HaRP) and 2
(GUARDIAN) are this paper's own prior, unpublished pipelines -- built in
the course of other work and only retroactively recognized as carrying
these patterns (§1). Their severity numbers ($+0.19$ AUROC; 18.8 AUROC
points) are traceable to the exact training scripts and logs that
produced them, released as two public repositories (§6), so a reader can
independently verify all four case studies end to end. Neither HaRP nor
GUARDIAN was run through the automated scanner in §5, since both were
built and their leaks discovered before that scanner existed.

### 4.1 Case Study 1 (severe): Full-dataset fit-then-score leakage — HaRP

[See `draft/worked_examples.md`, Case Study 1, for full detail.]

HaRP's Group-C representation feature `probe_conf` came from a
logistic-regression probe fit on all 700 samples, which then scored those
same 700 samples; only afterward did this feature enter a separate outer
5-fold cross-validation loop for the downstream failure estimator. This is
nested-CV leakage in its cleanest form: the "held-out" fold's feature was
manufactured by a model that had already seen every label, including that
fold's own.

**Quantified effect.** A+B+C AUROC under 5-fold CV: 0.9620 (leaked) versus
0.7714 (fixed, out-of-fold) -- a **+0.1906** inflation. `probe_conf`
alone: in-sample AUROC 1.0000 versus proper out-of-fold AUROC 0.7573,
independently cross-checked against a separately-obtained cross-validated
AUROC of 0.7583 (agreement within 0.001).

**Fix:** refit both the probe and the class centroids per fold, using
only that fold's training indices, scoring only the held-out indices.

### 4.2 Case Study 2 (severity: 18.8 AUROC points): CV-based layer-selection optimism — GUARDIAN

[See `draft/worked_examples.md`, Case Study 2, for full detail.]

GUARDIAN selects its detection layer -- L11 of 32, on Mistral-7B -- by
argmax of a 5-fold cross-validated AUROC across all layers, then reports
that same CV number as the model's performance.

**Quantified effect.** CV-selection AUROC at L11: 0.804. True held-out
AUROC at L11 (trained on the first 400 samples, tested on a separate,
never-selection-touched 300): **0.616** -- an 18.8-point gap. Which layer
counts as "optimal" also depends on the selection rule itself: argmax
train-split CV picks L11; argmax all-data AUROC picks L19 instead.

**Fix:** after CV-based selection, evaluate exactly once more on a
disjoint held-out set that played no role in the selection.

### 4.3 Case Study 3 (severity: real but modest at tested scale): Per-fold checkpoint-selection leakage — MultiHaluDet

[See `draft/case_study_multihaludet.md` for full code excerpts. Appendix A
documents the calibration/confound-detection process and additional
robustness checks behind this section's numbers.]

MultiHaluDet (arXiv 2605.24919; repo `github.com/alvi-uiu/MultiHaluDet`,
pinned at commit `c7597518`) reports up to 98.55% AUROC detecting
hallucination from Mistral-7B/LLaMA-2-7B hidden states. Its training code
(`run_pipeline.py::stage_2_3_train_oof`,
`src/training/trainer.py::train_deep_model_fold`) selects each inner
fold's kept checkpoint by argmax validation AUROC on that same fold, then
stores that fold's features as "out-of-fold" for a downstream
meta-learner. The checkpoint choice is a function of the validation
fold's own labels; unlike Case Study 1, only the *checkpoint selection*,
not gradient descent itself, sees the fold's labels.

**Quantifying it.** Re-running MultiHaluDet's actual 7B-scale pipeline was
outside this paper's compute budget, so we built a synthetic
reconstruction mirroring its training loop exactly (5-fold structure,
best-val-AUC checkpoint logic, OOF-feature downstream classifier),
calibrated to a Bayes-optimal-linear ceiling of AUROC$=0.80$ via
$\text{AUROC}=\Phi(\sqrt{J/2})$ (Simpson \& Fitter 1973; $J=1.417$). Four
conditions, $n=100$ seeds, four capacities (16/48/128/384 hidden units):
**LEAKY** (checkpoint chosen on the reused fold), **CLEAN\_MATCHED**
(checkpoint chosen on a disjoint carve-out, but retrained from scratch on
the full training budget for the resulting epoch count, using the same
seed as LEAKY -- isolating fold reuse from data budget), and **PLACEBO**
(checkpoint chosen on permuted labels -- zero *selection*-signal, but real
training signal, hence AUROC 0.73-0.74, not 0.5).

| Hidden | LEAKY | CLEAN\_MATCHED | PLACEBO | LEAKY$-$CLEAN\_MATCHED (p) |
|---|---|---|---|---|
| 16  | 0.7611 | 0.7602 | 0.7272 | +0.0009 (0.309) |
| 48  | 0.7647 | 0.7631 | 0.7375 | **+0.0017 (0.015)** |
| 128 | 0.7604 | 0.7571 | 0.7309 | **+0.0034 (0.008)** |
| 384 | 0.7533 | 0.7506 | 0.7350 | +0.0027 (0.077) |

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/capacity-sweep.pdf}
\caption{Mean AUROC for LEAKY, CLEAN, CLEAN\_MATCHED, and PLACEBO across all four tested capacities. LEAKY, CLEAN, and CLEAN\_MATCHED cluster tightly together at every capacity, all clearly above PLACEBO; the gap that constitutes actual leakage severity (LEAKY vs.\ CLEAN\_MATCHED) is visually small relative to this overall separation.}
\label{fig:capacity-sweep}
\end{figure}

CLEAN\_MATCHED beats PLACEBO decisively at every capacity ($p<0.0001$,
gaps $+0.016$ to $+0.033$), confirming honest checkpoint selection is
itself strongly informative. The LEAKY$-$CLEAN\_MATCHED residual -- the
actual leakage estimate -- survives Holm-Bonferroni correction across the
four capacities at 48 and 128 units ($p=0.015$, $p=0.008$); it is
near-zero at 16 units and borderline at 384 (exact MDE $\approx 0.0038$
against an observed gap of $0.0027$: underpowered, not null). This
correction is applied within this four-capacity family only, not jointly
across every p-value reported in this section.

**Testing an alternative explanation directly.** An independent review
proposed that LEAKY's advantage might reflect "having adaptive checkpoint
selection at all" during a full-budget run, not fold reuse specifically,
since CLEAN\_MATCHED's full-budget retrain is a blind, fixed-epoch run
rather than an adaptively-selected one. We tested this with a fifth
condition, CLEAN\_MATCHED\_ADAPTIVE: full training budget and *ongoing*
adaptive checkpoint selection during that same run -- the identical
mechanism LEAKY uses -- but selecting against the disjoint, honestly
held-out carve-out instead of the reused fold
(`code/22_epoch_forcing_confound_control.py`). If "having adaptive
selection" were the true driver, CLEAN\_MATCHED\_ADAPTIVE should approach
LEAKY. It does not: at capacity 128, LEAKY beats CLEAN\_MATCHED\_ADAPTIVE
by $+0.0198$ ($p<0.0001$) -- larger than the original LEAKY$-$CLEAN\_MATCHED
gap, not smaller -- and CLEAN\_MATCHED\_ADAPTIVE itself underperforms the
blind CLEAN\_MATCHED control ($-0.0164$, $p<0.0001$). At capacity 384 the
same pattern holds and is significant ($+0.0143$, $p<0.0001$), where the
original comparison was only borderline. This alternative explanation is
not supported by this test: LEAKY beats CLEAN\_MATCHED\_ADAPTIVE by more,
not less, than it beats CLEAN\_MATCHED, which is inconsistent with "any
adaptive selection helps" as the driver.

CLEAN\_MATCHED\_ADAPTIVE's own underperformance relative to the blind
CLEAN\_MATCHED control is itself unexpected and not fully explained --
selecting against genuine, honestly held-out signal should not do worse
than blindly reusing an epoch count learned from a separate, smaller-budget
run. The most likely account, by analogy with the epoch-count mechanism
identified below for the CLEAN\_MATCHED-vs-PLACEBO anomaly, is that
adaptive selection against a small (15\%) carve-out is itself a noisy
selection signal, a different phenomenon from the fold-reuse question
this control was built to isolate. **This carve-out is also a genuine,
disclosed confound with LEAKY vs.\ CLEAN\_MATCHED\_ADAPTIVE specifically,
not just with CLEAN\_MATCHED\_ADAPTIVE's own baseline behavior**: LEAKY
selects against its full 112-sample validation fold ($560/5$), while
CLEAN\_MATCHED\_ADAPTIVE selects against only a 67-sample carve-out
($15\%$ of the $448$-sample training split) -- a $1.67\times$ difference
in selection-set size. If a smaller selection set makes
CLEAN\_MATCHED\_ADAPTIVE's own checkpoint choice noisier (which its
underperformance against blind CLEAN\_MATCHED is itself evidence for),
that same noise would inflate, not just accompany, the LEAKY-vs-CLEAN\_MATCHED\_ADAPTIVE
gap this control reports as its headline result -- so the comparison
this control is built to settle is not fully isolated from the
confound it separately flags. A cleaner version of this control would
size the carve-out to match the fold ($25\%$ of $448=112$) rather than
the $15\%$ used here; we did not rerun it at that setting, and flag this
as an open robustness gap rather than treat the current result as fully
conclusive on its own.

**A second, structurally different generative process -- with two
disclosed confounds of its own.** The sweep above draws each class from
an isotropic Gaussian (identity covariance, classes differing only in a
constant mean offset) -- a reasonable calibration device, but a
toy-model artifact is a live alternative explanation if the severity
estimate depends on that assumption rather than being a property of the
underlying leakage mechanism. We built a second generative process that
keeps the controlled, repeatable sweep design but replaces the
covariance structure: fit the real pooled within-class covariance and
mean-difference direction from the real Mistral-7B/HaluEval features
already used below, then rescale the mean difference (keeping the real,
anisotropic, correlated covariance shape exactly as observed) to hit the
same AUROC$=0.80$ calibration target via the binormal identity
(`code/27_anisotropic_covariance_capacity_sweep.py`, $n=100$ seeds, same
four capacities). Two confounds with the isotropic sweep, caught by a
fresh review, limit how cleanly this isolates covariance shape alone:
the feature dimensionality changes from 64 (isotropic) to 414 (real
features' native dimension), a $6.5\times$ change in the
dimension-to-sample ratio that itself affects checkpoint-selection
dynamics; and the realized empirical AUROC under the trained MLP is
$0.672$-$0.681$ across capacities, not the $0.753$-$0.765$ the isotropic
sweep actually achieves despite both targeting the same Bayes-optimal
$0.80$ -- the two sweeps are calibrated to the same ceiling but do not
land at the same empirical difficulty, so some of the gap difference
between them is not attributable to covariance shape alone.

With those two confounds disclosed, the pattern itself: LEAKY vs.\
CLEAN\_MATCHED is $+0.0082$ ($p=0.007$, capacity 128, vs.\ $+0.0034$
under the isotropic sweep) and $+0.0099$ ($p=0.009$, capacity 384,
vs.\ $+0.0027$, not significant, under the isotropic sweep) -- both
significant where capacity 384 previously was not. But only one of the
four capacities (128) is significance-concordant between the two
sweeps; capacity 48 goes from significant (isotropic, $p=0.015$) to not
(anisotropic, $p=0.22$), and capacity 384 goes the other way. At the two
lower capacities, LEAKY vs.\ CLEAN\_MATCHED weakens (capacity 16:
$+0.0000$, $p=0.85$; capacity 48: $+0.0016$, $p=0.22$), while
CLEAN\_MATCHED vs.\ PLACEBO becomes significant instead ($+0.0084$,
$p=0.015$; $+0.0080$, $p=0.010$) -- some real gap is present at every
capacity under one comparison or the other, not concentrated only where
the isotropic sweep found it, but the specific capacity-by-capacity
pattern reshuffles rather than replicating cleanly. We also note that
pooling all eight synthetic LEAKY-vs-CLEAN\_MATCHED tests (both sweeps)
into one Holm-Bonferroni family, rather than treating the anisotropic
sweep as a separate robustness check the way this paper treats the
adaptive-selection control elsewhere, would leave nothing significant at
the family-wise $0.05$ level (smallest $p=0.007\times8=0.056$); we report
the anisotropic sweep's $p$-values uncorrected, in that same
robustness-check role, not as a second independent confirmation carrying
equal inferential weight to the pre-registered isotropic family. Taken
together this is evidence against the isotropic-covariance assumption
alone being responsible for the severity estimate, and does not shrink
or eliminate the gap -- but it is weaker, less clean evidence than a
simple "replicates and is larger" summary would suggest.

**Real-feature validation.** Feeding real Mistral-7B/HaluEval features
($n=400$, MultiHaluDet's own unmodified feature-extraction code, a 4-bit
AWQ checkpoint pinned to a fixed revision) through the same 5-fold
out-of-fold-plus-meta-learner LEAKY/CLEAN\_MATCHED/PLACEBO architecture
used in the synthetic sweep above -- not the single-fold variant an
earlier version of this test substituted, caught during review (Appendix
A) -- gives LEAKY beating CLEAN\_MATCHED by only $+0.0002$ AUROC
($p=0.56$, $n=100$ seeds) at capacity 128 and $+0.0001$ ($p=0.96$) at
capacity 384. **This comparison is not actually usable as reported: every
condition operates at AUROC $\approx0.985$ on these real features** --
the same ceiling problem Appendix A's own first correction-history entry
identifies for this paper's earlier calibration bug ("a task with no room
to fail leaves no room for a leakage bug to show inflation either"). An
absolute-AUROC-point MDE computed at a $0.80$ operating point (the
synthetic sweep's calibration target) is not comparable to an effect
measured at a $0.985$ operating point; a fresh review caught this and it
is a real, separate error from the architecture mismatch, not a
restatement of it.

**Attempted correction: calibrating the real features to the same
operating point as the synthetic sweep -- and a second, deeper problem
this attempt surfaced.** We rescaled only the between-class mean
separation of the real features (keeping every sample's own real,
unmodified within-class deviation exactly as observed -- no Gaussian
resampling, unlike the anisotropic sweep below) by the binormal-identity
factor used throughout this paper, targeting the synthetic sweep's
AUROC$=0.80$ calibration (`code/31_real_feature_test_calibrated.py`). A
further, independent review found this factor was itself computed from a
biased estimator: a permutation test (shuffling labels and recomputing
the identical plug-in formula) gives an "achieved" separation
($J\approx2.3$) that already exceeds the value this paper defines as
corresponding to AUROC$=0.80$ ($J=1.42$) -- pure label noise registers as
more separated than the calibration target itself, because the plug-in
Mahalanobis distance is unstable once feature dimensionality (414) is
comparable to sample size (400). We replaced this with an empirical
calibration (`code/33_real_feature_test_cv_calibrated.py`): bisection
search over the rescaling factor, using 5-fold cross-validated,
regularized logistic regression (confirmed unbiased under permutation:
$0.504\pm0.043$ CV AUROC on shuffled labels, vs.\ the plug-in formula's
$0.859$) to measure, rather than analytically predict, the achieved
AUROC at each candidate value. This gives a rescaling factor that
genuinely, verifiably hits AUROC$=0.80$ by the measure used to calibrate
it ($0.795\pm0.013$, 10-seed re-check).

But checking what this calibrated data actually produces under this
paper's own downstream evaluation (the same 5-fold OOF-plus-meta-learner
LEAKY/CLEAN/CLEAN\_MATCHED/PLACEBO pipeline used everywhere else in this
case study) shows the calibration does not transfer: every condition
still operates at AUROC $0.93$-$0.955$, not $0.80$. We traced this to its
source rather than reporting a partially-improved number and moving on:
at the rescaling factor that collapses both classes to an *identical*
mean -- zero linear separation between classes, by construction -- the
downstream pipeline's CLEAN\_MATCHED condition alone still achieves
$0.9535$ AUROC. The real Mistral-7B/HaluEval features' actual
class-discriminative content lives almost entirely in within-class
residual/higher-order structure (variance, correlation, and shape
differences between the two classes' distributions), not in a simple
linear mean shift, and this calibration method -- by explicit design, to
preserve real noise exactly rather than resample it synthetically --
only ever touches the mean. No value of the rescaling factor, from $0$
to $1$, can bring this real-feature test to the synthetic sweep's
operating point: the residual structure alone imposes a floor near
AUROC $0.95$, independent of between-class separation.

**We are retiring the real-feature calibrated severity test as
structurally uninformative for this dataset, rather than reporting
either its biased or its bias-corrected version as evidence for or
against transfer.** The mean-only-rescaling family of calibration (used
both here and, in a different, appropriately-generative form, in the
anisotropic sweep below) cannot be forced onto real Mistral-7B/HaluEval
features at $n=400$, $d=414$: the analytic version was additionally
biased by covariance-estimation noise at this near-degenerate
dimensionality, and even its bias-corrected replacement cannot control
the quantity that matters, because most of this feature set's real
discriminability sits outside what mean-rescaling touches at all. This
paper's severity estimate for Mechanism 3 therefore rests on the two
synthetic sweeps alone (isotropic and anisotropic, above), both of which
generate fresh samples from a fully-specified, controlled generative
process rather than rescaling a small, fixed real sample -- avoiding
both problems found here. A separate FP16-vs-AWQ control on the same 50
samples finds identical AUROC (0.9600 both) at a single point estimate;
a resampled version (Appendix A) gives a 95\% CI of $[-0.061, +0.080]$,
consistent with no quantization confound rather than a demonstration
that rules one out at the resolution this severity estimate needs.
Appendix A documents the full progression of this test -- the original
architecture mismatch, the ceiling-effect error the "corrected" version
still had, the biased calibration factor a further review caught, and
the structural floor that led to this test's retirement -- along with a
further robustness check (a resampled-effect-size reanalysis of the
FP16-vs-AWQ control).

**A pre-registered decision rule whose verdict depends on which version
of the test is asked.** Before running the sweep, we pre-registered a
rule classifying each capacity as `GENUINE_LEAK_CONFIRMED`,
`CONFOUND_CONFIRMED_NO_REAL_LEAK`, or `MIXED` from the placebo-relative
gaps. We initially reported this branch as structurally unreachable, on
the basis that leaky\_minus\_placebo is significant at $p<10^{-8}$ "in
every version of this experiment" -- this was wrong, caught by the same
review that caught the ceiling-effect error above. Applying the rule to
every version actually run: the isotropic synthetic sweep returns
`MIXED` at all four capacities; the anisotropic sweep returns `MIXED` at
three and `GENUINE_LEAK_CONFIRMED` at one (capacity 128); the
ceiling-confounded real-feature test returns `CONFOUND_CONFIRMED_NO_REAL_LEAK`
at both tested capacities; both the biased-calibration and the
bias-corrected real-feature test return `MIXED` at both (though, per
above, we no longer treat either real-feature version's severity
estimate as informative -- these verdicts are retained here only as
further illustration of rule instability, not as a claim about real
severity). The rule is therefore reachable in every
direction -- what it is not, is *stable*: the same underlying mechanism
produces three different verdicts depending on generative process,
calibration, and capacity. We read this as reinforcing, not undermining,
this paper's central methodological point: a single pre-registered
decision rule, however principled, is not a substitute for reporting the
full distribution of outcomes across every control run, and a reader
should not take "pre-registered" as license to trust one capacity's
classification without checking the others; Appendix A documents the
full analysis.

**Bottom line.** The mechanism itself -- checkpoint choice as a function
of the reused fold's own labels -- is unambiguous and code-verified. Its
quantified severity, once budget- and seed-matched, is small ($+0.0009$
to $+0.0034$ AUROC on the synthetic sweep) and significant at most but not
all tested capacities, and this significance strengthens under the
stricter adaptive-selection control above, and again (if anything, more
strongly, though with a reshuffled rather than uniformly stronger
capacity pattern -- see Appendix A) under a second generative process
replacing the isotropic calibration with a real, anisotropic covariance
structure. A real-feature validation attempt, going through two further
rounds of correction (an architecture-mismatch fix, then a
biased-calibration-factor fix), ultimately could not be brought onto the
synthetic sweep's operating point at all: most of the real feature set's
discriminative content lives in structure the mean-only-rescaling method
cannot touch, imposing a floor near AUROC $0.95$ regardless of the
rescaling factor chosen. This is not a partial confirmation of transfer
-- it is a retirement of this specific test once its deeper structural
limitation was found (Appendix A), and this paper's severity estimate
for Mechanism 3 rests on the two synthetic sweeps alone. This also does
not settle MultiHaluDet's own 98.55\% AUROC's exact inflation: their real
architecture (6 transformer layers, multi-scale, heavy augmentation) is
considerably more expressive than this reconstruction. Getting this
diagnosis right required tracing each correction to its actual source
rather than accepting a partially-improved number at face value;
Appendix A documents that process in full.

### 4.4 Case Study 4 (secondary; severity unquantified, mechanism confirmed): Test-set-driven best-layer selection — quantized-LLM paper

[See `draft/case_study_quantized_llm_paper.md` for full detail, including
literal code excerpts.]

"Hallucination Is Linearly Decodable from Mid-Layer Hidden States in
Quantized LLMs" (arXiv 2606.02628; repo
`github.com/Ezharjan/HallucinationPatternDetection`, pinned at commit
`ea0b9678`) reports AUROC up to 1.000. Its per-layer probe training
(`src/detection/probes.py::train_probe`) uses a clean single 70/10/20
train/val/test split, genuinely free of Case Studies 1-3's exact
patterns. But `src/detection/saplma.py::saplma_probe_per_layer` selects a
`best_layer` by the argmax, over ~30 layers, of each layer's **test-set**
AUROC, then reports that same test-set number as the headline result,
with no correction for having tried ~30 hypotheses against the one test
set. This is the classical "winner's curse" of multiple-hypothesis
selection using the exact metric later reported.

We did not re-quantify this mechanism's severity -- their public repo
ships only summary result tables, not the underlying per-layer
hidden-state arrays `saplma_probe_per_layer` consumes, so re-quantification
would need their exact quantized checkpoint and dataset re-run from
scratch, outside this paper's compute budget. We flag it as a fourth,
structurally distinct pattern worth watching for independently of the
other three.

## 5. A Checklist for This Literature

[Full version in `draft/leakage_checklist.md`.] Four questions correspond
to the four mechanisms above:

1. Is any feature computed by fitting something on the full dataset's
   labels before an outer CV loop treats part of that dataset as held out?
2. Inside a CV fold, is a "best" checkpoint/epoch chosen using the same
   fold whose features/predictions get reused downstream as clean OOF
   output?
3. Is a "best" layer/config selected by the argmax of a metric computed
   on the same test set later reported as the headline number, across
   many candidates, with no correction?
4. Was a hyperparameter chosen by maximizing a cross-validated metric,
   with that same CV number then reported as performance, with no
   further genuinely held-out check?

**We tried to automate this checklist; the attempt itself is informative.**
We built a regex-based scanner for the four patterns and ran it against 7
repositories total: the two already-audited case studies (MultiHaluDet,
HallucinationPatternDetection), to test whether it catches known bugs,
plus 5 additional, not-previously-examined, externally-found
hidden-state-probing repositories, to test its false-positive rate on
presumed-clean pipelines (`code/04_leakage_linter.py`,
`results/leakage_linter_report.json`). This is a heuristic line-matcher,
not a validated static analyzer; every flag was manually read and
classified before being treated as a finding.

The scanner produced 7 raw hits, concentrated in 2 repos, and manual
reading showed all 7 are false positives: 3 flag
`HallucinationPatternDetection`'s `embed_viz.py` for "fit-then-score" --
these are t-SNE/UMAP/PCA calls used only for plotting; 4 flag
`haloscope`'s layer/threshold selection -- reading the code shows
selection correctly uses a separate validation split, the opposite of
leakage, and a positive data point for that NeurIPS 2024 pipeline. More
tellingly, the scanner missed both known true positives in this corpus:
Case Study 4's bug is a plain loop with neither an "argmax" token nor a
"test"-named variable at the selection site; Case Study 3's bug selects
on `best_auc`/`best_model`, not the `checkpoint`/`best_epoch` keywords the
scanner looks for. Zero true positives among the 5 newly-scanned repos,
two false negatives on the two known positives, seven-for-seven false
positives on its raw hits: at this level of tooling, an automated scanner
is not a substitute for manual, one-repo-at-a-time reading.

This is a prevalence statement about 2 of 7 repos, not a representative
sample -- all 7 were found via targeted search for exactly this failure
class, not a random or exhaustive sample of the field (HaRP and GUARDIAN,
this paper's own pipelines and the source of Case Studies 1-2, were never
run through the scanner at all -- they sit entirely outside this count).
The true population-level prevalence of any
of these four mechanisms remains unknown and is not estimated by this
count. We report the negative automation result because the same
variable-naming and control-flow subtlety that makes this leakage easy to
introduce by accident also makes it resist simple pattern-matching, in
both directions.

## 6. Discussion

**Data and code availability.** All code, cached result JSONs, and the
paper source are included in the anonymized supplementary material for
double-blind review, and will be released as public GitHub repositories
under the authors' names upon publication (three repositories: one for
the main audit and checklist, and one each for Case Studies 1 and 2,
each containing only the subset of the original project relevant to the
specific finding reported, not the full original codebase). Case
Studies 3 and 4 are fully reproducible from what ships with this paper:
pinned third-party commits, our audit scripts, and the resulting JSON
logs. The regex
linter in §5 is included in full and is, per its own results, a confirmed
non-detector on this corpus (0 true positives, 2 false negatives on the
corpus's own known bugs) -- included for transparency, not as a working
tool a reader should rely on today.

**Scope note.** This paper makes no inference-economy claim. It is an
evaluation-protocol audit, not an inference-time method; fixing any of
the four leakage mechanisms makes no claimed difference to deployment
cost. The paper's value is entirely in correcting reported detection
metrics.

**Severity numbers matter operationally, even though we do not quantify
deployment cost here.** A detector deployed at a reported AUROC several
points above its true value (Case Study 2's 18.8-point gap; Case Study
1's $+0.19$ inflation) will, at a fixed threshold, pass more
hallucinations to production and flag more correct outputs for
unnecessary review than promised. We lack deployment-volume and
review-cost data for any audited system, so we do not attempt a dollar
estimate; any severity number this paper reports translates directly into
a cost multiplier at whatever volume and review cost a practitioner
actually has.

**This audit's central lesson recurs in related, independently-conducted
work** examining hidden-state signals and benchmark severity estimates
in adjacent areas of LLM evaluation, none sharing a leakage mechanism
with the four case studies here: an initially plausible signal or
benchmark result substantially weakens or changes character once a
specific confound is directly tested, rather than merely acknowledged as
possible. This paper shares that discipline: a claimed severity or
signal is not established until the confounds that could produce it for
free have been named and tested, not merely acknowledged as possible.

**Limitations.** We could not re-run either external paper's full
7B-scale pipeline within this paper's compute budget, so Case Studies 3
and 4's severity for the *actual published numbers* remain open
questions; our reconstruction of Case Study 3 supports a small, positive
estimate at 2 of 4 tested capacities, and MultiHaluDet's actual
architecture (6 transformer layers, multi-scale, heavy augmentation) is
more expressive still, so this should not be over-read in either
direction for their pipeline. The real-feature test (\S4.3) was also single-dataset, single-language
($n=400$ HaluEval English `qa_samples` only), despite MultiHaluDet's own
contribution being explicitly multilingual; but more fundamentally, this
test was retired (Appendix A) once we found no calibration of this
feature set's between-class mean, at any strength, could bring it onto
the synthetic sweep's operating point, because most of its
discriminability lives in within-class structure the calibration method
cannot touch. Whether the synthetic severity estimate transfers to real
hidden states at all -- on this dataset, non-English datasets, or larger
ones -- remains untested rather than confirmed or excluded, and would
need a different validation strategy than mean-only rescaling to test at
this sample size and dimensionality.

## 7. Conclusion

Four distinct label-information-leakage mechanisms recur across
hidden-state hallucination-detection pipelines, including our own, and
differ substantially in severity. Case Study 1 is catastrophic ($+0.19$
AUROC) regardless of scale. Case Study 3's checkpoint-selection mechanism
is a small residual gap, significant at 2 of 4 tested capacities, that
strengthens rather than weakens under a stricter control isolating fold
reuse from adaptive selection in general (Appendix A documents why a
defensible estimate here requires several controls to hold
simultaneously). That difficulty -- needing calibration, budget-matching,
seed-matching, and an alternative-confound test all to hold at once
before trusting the number -- is the strongest argument in this paper for
the checklist we provide, more so than any single mechanism in isolation.

The checklist lets researchers and reviewers in this fast-growing
subfield ask precise, mechanism-specific questions of a pipeline's
evaluation protocol, replacing "did you use cross-validation correctly"
as a single, undifferentiated concern.

The automated-scanner exercise (§5) covers seven repositories in total:
two already-confirmed positive case studies (§4.3-4.4, MultiHaluDet and
HallucinationPatternDetection) and five additional, not-previously-examined
repositories, which surfaced no new confirmed positives -- too small a
sample to support any claim about how common these four mechanisms are
across the wider literature, and we do not make one. (Case Studies 1-2,
HaRP and GUARDIAN, were manually audited and are not part of this count
at all.) A pre-registered audit applying
this checklist to a fixed,
defined sample of 15-20 externally published probing papers, selected
before any of them are examined, is planned as follow-on work; that
protocol is already drafted.

## Appendix A: Correction History and Robustness Checks for Case Study 3 (MultiHaluDet)

**This appendix documents how the Case Study 3 result in §4.3 was
reached, and additional robustness checks omitted from the main text for
length.** §4.3 states the result directly; nothing below is required to
verify it.

**Getting the severity estimate right took three earlier, each wrong in a
different way.** An initial, uncalibrated pass saturated at AUROC$=1.0000$
for both LEAKY and CLEAN -- a task with no room to fail leaves no room for
a leakage bug to show inflation either. Recalibrating via the identity
$\text{AUROC}=\Phi(\sqrt{J/2})$ (Simpson \& Fitter 1973) surfaced a second
issue: the calibration script had inverted the wrong formula,
$\Phi(\sqrt{J}/2)$ (the equal-prior Bayes-*accuracy* identity, not AUROC,
since $2\neq\sqrt2$), giving a task calibrated to AUROC$=0.883$, not the
intended $0.80$ -- consistent with the $\approx0.87$-$0.88$ LEAKY/CLEAN
AUROCs actually observed under the miscalibrated version. The corrected
inversion, $J=2\Phi^{-1}(0.80)^2=1.417$, gives a verified
$\Phi(\sqrt{J/2})=0.800$. Fixing calibration alone still left a third
issue: LEAKY and CLEAN were never matched on training-data budget (CLEAN's
15\% early-stopping carve-out meant $\approx$15\% less training data than
LEAKY), so a naive fix (adding CLEAN\_MATCHED, retrained on the full
budget) initially used a different random seed for that retrain than
LEAKY/CLEAN used, diluting the budget-matched comparison's power and
making an early "not significant at any capacity" reading partly an
artifact of the fix's own procedure. Matching the seed as well gives the
result now stated in §4.3.

**A fourth issue, on the real-feature test specifically, caught during
review: an architecture mismatch.** `code/03_real_feature_leakage_test.py`'s
docstring claimed its LEAKY/CLEAN\_MATCHED/PLACEBO test used "identical
architecture ... to `code/02d_corrected_capacity_placebo_sweep.py`." It
did not: 02d performs a genuine 5-fold cross-validation loop, pools
out-of-fold (OOF) features from all five folds, and fits a downstream
logistic-regression meta-learner on the pooled OOF features before
evaluating on the held-out test set; `03` (and the later diagnostic
rerun, `code/19_real_feature_leakage_diagnostics.py`, which reused the
same architecture to investigate the anomaly below) instead took only the
first of five folds, trained a single MLP on it, and evaluated that one
model's raw output directly on the test set -- a materially different,
higher-variance pipeline that happened to produce a significant
$+0.0026$ AUROC gap ($p=0.0012$) on real features. Rerunning with the
actual 02d architecture
(`code/25_real_feature_leakage_test_corrected_architecture.py`, same
cached real Mistral-7B features, no new inference) gives $+0.0002$ AUROC
($p=0.56$, capacity 128) and $+0.0001$ ($p=0.96$, capacity 384) -- not
significant at either capacity. This is a fourth
instance of the same lesson this appendix documents: a single
confirmatory-looking number is not sufficient without checking that the
procedure producing it actually matches the one it is claimed to match.
It is not, however, the end of the story: the "corrected" numbers above
have a second problem, documented next.

**A fifth issue: the corrected architecture still ran the test at a
ceiling operating point, and a fresh review caught this too.** All four
conditions in the "corrected architecture" real-feature test operate at
AUROC $\approx0.985$ -- essentially the same saturation problem this
appendix's very first entry describes for the original miscalibrated
synthetic sweep ("a task with no room to fail leaves no room for a
leakage bug to show inflation either"). We had not applied that same
discipline to the real features. Comparing an $0.00087$ MDE computed at
capacity 128's $\approx0.757$ operating point to a $+0.0002$ effect
measured at a $\approx0.985$ operating point is not a valid
apples-to-apples power argument, and the "adequately powered null" claim
in an earlier version of this section was wrong on that basis, not just
imprecise. Recomputed on a scale-invariant basis (relative error
reduction against the synthetic sweep's own capacity-128 result, or the
Fisher-$J$ implied by the binormal identity used throughout this paper),
the raw $+0.0002$ effect at $0.985$ headroom is not smaller than what
the synthetic estimate would predict at that headroom -- if anything
slightly larger. We fixed this properly rather than only computing a
rescaled comparison: `code/31_real_feature_test_calibrated.py` rescales
*only* the real features' between-class mean separation (by the same
binormal-identity factor as the anisotropic sweep) while leaving every
sample's own real, unmodified within-class deviation untouched -- no
Gaussian resampling, unlike the anisotropic sweep -- to bring the real
features to the same AUROC$=0.80$ target the synthetic sweep uses. At
this comparable operating point: LEAKY beats CLEAN\_MATCHED by $+0.0014$
($p=0.12$, capacity 128) and $+0.0010$ ($p=0.21$, capacity 384) -- both
inside the synthetic sweep's own $+0.0009$ to $+0.0034$ range, though
per-seed variance is larger at this operating point ($SD=0.0084$-$0.0094$
vs.\ $0.0031$-$0.0032$ uncalibrated -- moving off ceiling makes
seed-to-seed AUROC genuinely noisier, not a sign of a bug), giving an
80\%-power MDE ($0.0023$-$0.0026$) this specific test does not
independently clear at the observed magnitude. The reading at the time
was "consistent with transfer, not independently confirmatory" -- a real
correction to the previous version's "evidence against transferring"
claim. Two further issues, described next, superseded this reading.

**A sixth issue: the rescaling factor above was itself computed from a
biased estimator, caught by a further independent review.** The factor
$\alpha=\sqrt{J_{\text{target}}/J_{\text{real}}}$ used above depends on
$J_{\text{real}}=\Delta\mu^\top\Sigma^{-1}\Delta\mu$, the real features'
plug-in Mahalanobis separation, computed from $n=400$ samples in
$d=414$ dimensions -- a regime where $d\approx n$ makes the sample
covariance matrix's inverse unstable regardless of the $0.05\times
\mathrm{trace}(\Sigma)/d$ ridge term already applied. A direct
permutation test exposes this: shuffling the labels and recomputing the
identical formula 30 times gives a mean $J_{\text{perm}}=2.31$ ($SD=0.29$)
-- already larger than $J_{\text{target}}=1.42$, the value this paper
defines as corresponding to AUROC$=0.80$. Pure label noise, with the
true label-feature relationship completely destroyed, registers as
*more* separated by this estimator than the calibration target itself;
the original $\alpha=0.1455$ was computed from a $J_{\text{real}}=66.93$
that is contaminated by exactly this same noise-driven inflation and was
never a trustworthy calibration handle.

We fixed this with an empirical, cross-validated calibration
(`code/33_real_feature_test_cv_calibrated.py`) rather than a corrected
analytic formula: for a candidate $\alpha$, apply the identical
affine class-mean rescaling, then measure the \emph{actually achieved}
AUROC via 5-fold cross-validated, $L_2$-regularized ($C=0.01$) logistic
regression -- out-of-fold, so a sample's held-out prediction never
depends on that same sample's contribution to the fitted boundary --
and bisection-search $\alpha$ until the measured (not theoretical) CV
AUROC hits $0.80$. Verified this approach is not subject to the same
bias: under label permutation, this CV-AUROC estimator gives
$0.504\pm0.043$ (10 permutations) -- correctly centered at chance,
against the plug-in formula's $0.859$ (equivalent to $J_{\text{perm}}=2.31$).
Bisection converges to $\alpha=0.2031$, with a 10-seed re-check CV AUROC
of $0.7948\pm0.0132$ -- genuinely, verifiably at the target this time.

**A seventh issue, found while verifying the sixth fix actually worked
end to end: the corrected $\alpha$ still does not control what matters.**
Re-running this paper's own downstream 5-fold OOF-plus-meta-learner
severity pipeline (the same one used for every other number in this
case study, not the calibration probe above) on data calibrated with
$\alpha=0.2031$ gives LEAKY$=0.9519$, CLEAN\_MATCHED$=0.9487$,
PLACEBO$=0.9350$ at capacity 128 (and $0.9549$/$0.9522$/$0.9450$ at
capacity 384) -- not the intended $0.80$ operating point at all. Rather
than report this discrepancy as an unexplained residual gap, we traced
it directly: evaluating the CLEAN\_MATCHED condition alone at
$\alpha=0$ -- the rescaling factor that collapses both classes to an
\emph{identical} mean, i.e., zero linear class separation by
construction -- still gives $0.9535\pm0.0193$ AUROC (8 seeds). Since
$\alpha$ only ever rescales the between-class mean and leaves each
sample's own real within-class deviation completely untouched (by this
method's explicit, and otherwise correct, design goal of never resampling
real noise), this result means the real Mistral-7B/HaluEval features'
actual class-discriminative content lives almost entirely in
within-class residual/higher-order structure that this calibration
family cannot reach at any $\alpha\in[0,1]$. The floor this structure
imposes ($\approx0.95$) sits well above the synthetic sweep's $0.80$
target regardless of how the mean-difference term is tuned.

**Resolution: retirement, not a third corrected number.** Given that no
value of $\alpha$ can bring this specific real-feature test to a fair
comparison with the synthetic sweep, we are retiring it as structurally
uninformative for this dataset rather than reporting a fourth version of
"the real-feature severity number." §4.3's main text and abstract have
been updated accordingly: this paper's severity estimate for Mechanism 3
now rests on the two synthetic sweeps (isotropic and anisotropic) alone,
neither of which shares this problem, since both generate fresh samples
from a fully-specified generative process rather than rescaling a small,
fixed real sample. The exact-MDE and epoch-count diagnostics below,
computed on the now-retired real-feature tests, are retained as
historical documentation of how this progression of corrections was
reached, not as claims this paper still relies on.

**The lesson this demonstrates about itself, not just argues for.** A
single confirmatory-looking or null-looking number is not sufficient to
close a leakage-severity question, even at adequate seed count and with a
placebo control. Eight controls must hold *simultaneously*: a
correctly-inverted calibration formula; a budget-matched control checked
for confounds introduced by the matching procedure itself; adequate
power; a permuted-label placebo; consulting one's own pre-registered
decision rule rather than narrating past it; matching the operating
point (not just the architecture) whenever a result is compared across
data sources; verifying a calibration estimator is itself unbiased
(ideally by permutation test) rather than trusting a closed-form
identity at face value; and confirming that whatever a calibration
targets is the same quantity the downstream evaluation pipeline actually
measures, not a linear proxy for it. Get any one wrong, in either
direction, and a confident-looking conclusion can be wrong -- including,
as this appendix now documents about itself twice over, a conclusion
arrived at while explicitly trying to fix a previous mistake.

**The pre-registered decision rule's verdict depends on which version of
the test is asked.** A rule classifying each capacity as
`GENUINE_LEAK_CONFIRMED`, `CONFOUND_CONFIRMED_NO_REAL_LEAK`, or `MIXED`
from the placebo-relative gaps (`code/02d_corrected_capacity_placebo_sweep.py`)
gives, across every version of this experiment actually run: `MIXED` at
all four capacities on the isotropic synthetic sweep; `MIXED` at three
capacities and `GENUINE_LEAK_CONFIRMED` at one (capacity 128) on the
anisotropic sweep; `CONFOUND_CONFIRMED_NO_REAL_LEAK` at both tested
capacities on the ceiling-confounded (uncalibrated) real-feature test;
and `MIXED` at both capacities on both the biased-calibration and the
bias-corrected (`code/33`) real-feature tests. An earlier version of
this section claimed the `CONFOUND_CONFIRMED_NO_REAL_LEAK` branch was
structurally unreachable "in every version of this experiment" -- this
was wrong, and the ceiling-confounded real-feature test above is the
direct counterexample, caught by the same review that caught the
ceiling-effect error itself. The honest characterization is not
"unreachable" but "unstable": the same underlying mechanism produces
three different verdicts across five tested conditions, depending on
generative process, calibration, and capacity. `MIXED` is the modal, but
not universal, honest label for what the data show. (As above, we no
longer treat either real-feature version's severity number as
informative about transfer; these verdicts illustrate rule instability,
not real severity.)

**Exact, not back-of-envelope, minimum detectable effect, for both the
uncalibrated and calibrated real-feature tests (retained as historical
documentation of a now-retired test; see above).** The uncalibrated
corrected-architecture rerun
(`code/25_real_feature_leakage_test_corrected_architecture.py`) saves
full per-seed gap arrays. Exact per-seed gap SD for
LEAKY-vs-CLEAN\_MATCHED is $0.003095$ (capacity 128), giving an
80\%-power MDE of $0.00087$ -- but this MDE is itself measured at the
same $\approx0.985$ ceiling operating point as the effect it is compared
to, so the earlier claim that this makes the null "adequately powered"
does not hold once the operating-point mismatch above is accounted for;
it is a valid MDE for *that* operating point, not for the synthetic
sweep's $0.80$ operating point. The properly calibrated version
(`code/31_real_feature_test_calibrated.py`) gives exact per-seed gap SD
of $0.00940$ (capacity 128) and $0.00834$ (capacity 384), 80\%-power MDE
of $0.00263$ and $0.00234$ respectively -- both larger than the observed
gaps ($+0.0014$, $+0.0010$), meaning this specific test, honestly
calibrated, is underpowered to independently confirm an effect of the
size it observes, though the observed gaps still sit inside the
synthetic estimate's own range. (The original, architecturally-mismatched
version of this test had a larger per-seed SD still -- $0.008294$ and
$0.009420$ -- consistent with the single-fold pipeline's higher
variance; that version's significant gap is an artifact of the
architecture mismatch, not a property of the underlying mechanism.)

**A single point estimate is weaker rigor than every other comparison in
this paper.** The FP16-vs-AWQ quantization control (§4.3) was originally a
single 5-fold-CV logistic-regression point estimate at $n=50$. We reran
the identical comparison through this paper's own capacity-128
SweepMLP architecture -- the same one the headline severity claim uses --
with 100 seed-resampled splits instead of one point estimate (reusing
cached features, no new model inference). Result: mean AUROC $0.946$
(FP16) vs.\ $0.939$ (AWQ), gap $+0.0068$, 95\% CI $[-0.061, +0.080]$
(includes zero), Wilcoxon $p=0.148$ -- not significant, though a CI this
wide (roughly 78$\times$ the magnitude of the severity estimate itself)
means this control is consistent with no quantization confound rather
than a demonstration that rules one out; §4.3's main-text wording has
been softened to match.
(`results/fp16_vs_awq_control_matched_architecture.json`).

**The epoch-count mechanism below described the single-fold architecture's
anomaly specifically, and applies to neither corrected version (also
retained as historical documentation of the now-retired real-feature
tests).** Under
the original, single-fold architecture, CLEAN\_MATCHED underperformed
PLACEBO by $-0.0025$ ($p=0.0115$), which we traced to an epoch-count
selection difference: instrumented epoch counts across 100 seeds
(`code/19_real_feature_leakage_diagnostics.py`) showed CLEAN's
early-stopped checkpoint averaging epoch $12.18$, while PLACEBO's
checkpoint-selection -- even against permuted, meaningless validation
labels -- averaged epoch $17.7$ ($p=0.012$, per-seed epoch gap predicting
the AUROC gap directly at $r=-0.594$, $p=7.5\times10^{-11}$). Under the
uncalibrated corrected (5-fold-OOF) architecture, this gap shrinks to
$-0.0006$ ($p=0.39$, capacity 128) and is no longer significant. Under
the properly calibrated real-feature test, the comparison reverses sign
entirely: CLEAN\_MATCHED substantially *outperforms* PLACEBO
($+0.0189$, $p=5.8\times10^{-8}$, capacity 128; $+0.0102$,
$p=2.1\times10^{-7}$, capacity 384) -- the expected direction (real,
informative selection labels should do at least as well as permuted
ones), not an anomaly at all. The anomaly this mechanism was built to
explain was specific to the single-fold architecture's higher variance
and does not recur in either corrected version. We retain this
diagnostic here, clearly marked as describing the superseded
architecture, because it demonstrates the same point as the rest of this
appendix: an unexplained sanity-check anomaly should be run down before
being narrated past, and in this instance running it down further --
via the architecture-mismatch check above -- also resolved the anomaly
itself.

## References

Full citations below, compiled from exactly the bibliographic detail
already verified in-text; HaRP and GUARDIAN (Case Studies 1-2) are this
paper's own pipelines, not external publications, and are not separately
listed here.

PARALLAX (2026). Separating Genuine Hallucination Detection from
Benchmark Construction Artifacts. arXiv 2605.17028.

Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in Data Mining:
Formulation, Detection, and Avoidance. *Proceedings of the 18th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining
(KDD 2012)*.

Varma, S., & Simon, R. (2006). Bias in Error Estimation When Using
Cross-Validation for Model Selection. *BMC Bioinformatics*, 7(1), 91.

Shi, W., et al. (2024). Detecting Pretraining Data from Large Language
Models. *ICLR 2024*. arXiv 2310.16789.

Golchin, S., & Surdeanu, M. (2024). Time Travel in LLMs: Tracing Data
Contamination in Large Language Models. *ICLR 2024*. arXiv 2308.08493.

Sainz, O., et al. (2023). NLP Evaluation in Trouble: On the Need to
Measure LLM Data Contamination for Each Benchmark. *EMNLP Findings
2023*. arXiv 2310.18018.

MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State
Probing. arXiv 2605.24919.

Aiersilan, A. (2026). Hallucination Is Linearly Decodable from
Mid-Layer Hidden States in Quantized LLMs. arXiv 2606.02628.
