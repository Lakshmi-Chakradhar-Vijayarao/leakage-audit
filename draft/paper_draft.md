# Not All Leaks Are Equal: A Taxonomy and Audit of Label-Information Leakage in Hidden-State Hallucination Detection

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

Linear and shallow-MLP probes on LLM hidden states are a standard
hallucination-detection tool, with reported AUROCs routinely exceeding
0.90. We identify five distinct mechanisms by which evaluation labels
leak into these numbers: (1) full-dataset fit-then-score feature leakage
(+0.19 AUROC, our own pipeline, HaRP); (2) cross-validation-based layer
selection reported as held-out performance (an 18.8-point gap, our own
pipeline, GUARDIAN -- a full 32-layer sweep shows a comparably large
CV-vs-held-out gap at nearly every layer, though a corrected,
randomization-based estimate finds the additional penalty specifically
attributable to argmax-over-layers selection is small, +0.007 (SD
0.030), once the dataset's own non-random layer-selection split is
replaced with proper random resampling); (3) per-fold checkpoint
selection that reuses the same fold as "out-of-fold" features, verified
in a published pipeline (MultiHaluDet); (4) test-set-driven layer
selection with no multiplicity correction, verified in a second
published paper; (5) test-set-driven decision-threshold selection, found
in the same pipeline audited for Mechanism 3 -- leaves AUROC exactly
unaffected by construction but materially inflates every
threshold-dependent metric (F1, accuracy, MCC) the same pipeline reports
alongside it.

Severity differs sharply by mechanism, not by pipeline. Mechanism 1 is
catastrophic regardless of scale. Mechanism 3's severity is small under
a checkpoint-selection-only harness (+0.0009 to +0.0034 AUROC) but
roughly 3-4x larger once we model the pipeline's actual training loop (an
LR scheduler and early stopping both reacting to the same validation
fold). Mechanism 5's F1/accuracy gaps (+0.021 to +0.034) are not directly
comparable to Mechanism 3's AUROC gaps -- different metrics measuring
different things, a distinction an earlier draft of this paper
conflated -- but are themselves large and highly significant (p<10^-31).
While deriving Mechanism 3's
severity estimate, we found that our own measurement instrument had
committed exactly the leakage pattern this paper audits for -- a
full-dataset calibration transform evaluated across the split it was fit
on -- and fixed it: the paper's own thesis, surfacing in its own
tooling.

We provide a five-question checklist and test whether it can be
automated: a regex scanner run over 7 repositories (2 known-leaky, 5
previously unexamined) produces zero true positives and misses both
known bugs. This class of leakage resists simple pattern-matching.

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
five *distinct* mechanisms by which evaluation labels can leak into a
reported metric, verify three of them (Mechanisms 3-5) in two externally
published, code-available pipelines, and show via controlled
reconstruction that these mechanisms
differ substantially in how much inflation they actually cause -- a
distinction the field currently has no vocabulary for.

**Contributions:**
1. A verified, quantified nested-CV feature-leakage bug in our own
   pipeline (HaRP), with an independent cross-check confirming the fix.
2. A verified, quantified CV-based hyperparameter-selection optimism bias
   in a second pipeline of ours (GUARDIAN): an 18.8-point gap between the
   CV-selection number and genuine held-out performance at the selected
   layer, plus a full 32-layer decomposition and a randomization-based
   correction isolating how much of that gap is specifically attributable
   to layer selection (small, +0.007 SD 0.030) versus a general
   CV-vs-held-out optimism present at nearly every layer.
3. A code-verified identification of a third, distinct mechanism
   (per-fold checkpoint selection) in an externally published pipeline's
   released code, plus a controlled synthetic reconstruction -- calibrated
   via a closed-form Fisher-ratio/AUROC identity -- that quantifies this
   mechanism's severity and tests, via an explicit control condition,
   whether an alternative confound could explain it instead.
4. A code-verified identification of a fourth, distinct mechanism
   (test-set-driven best-of-many-layers selection) in a second externally
   published paper's released pipeline.
5. A code-verified identification of a fifth, distinct mechanism
   (test-set-driven decision-threshold selection) in the same pipeline
   audited for Mechanism 3, quantified on the same calibrated synthetic
   harness and found to be substantially more severe.
6. A checklist synthesizing all five mechanisms into actionable questions,
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

**Kapoor \& Narayanan's cross-field leakage taxonomy.** The closest prior
art to this paper's central move is Kapoor \& Narayanan (2023), who
survey leakage across 17 scientific fields and 294 papers and propose an
8-type taxonomy: L1.1 (no test set), L1.2 (pre-processing computed on
train+test combined), L1.3 (feature selection on train+test), L1.4
(duplicates across the split), L2 (illegitimate/proxy features), L3.1
(temporal leakage), L3.2 (non-independence between train and test
samples), and L3.3 (test distribution not matching the distribution of
scientific interest). Mapping this paper's five mechanisms onto that
taxonomy: Mechanism 4 (test-set-driven best-layer selection) and
Mechanism 5 (test-set-driven threshold selection, §5) are both direct
instances of L1.1 -- the test set is reused for a modeling decision
(which layer, which threshold) and then for the reported metric,
leaving no genuinely held-out data at all. Mechanism 1 (full-dataset
fit-then-score feature leakage) is an instance of L1.2, generalized from
a preprocessing statistic to an entire fitted feature-extraction model.
Mechanism 3 (per-fold checkpoint-selection leakage) is a narrower-bandwidth
variant of the same L1.2 pattern, restricted to which training iterate is
kept rather than which preprocessing statistic is used. Mechanism 2
(CV-based layer/hyperparameter selection optimism, i.e.\ Varma \& Simon's
problem) does not map cleanly onto any single one of Kapoor \& Narayanan's
eight leaf types -- it is closest in spirit to L1.2/L1.3 but is really a
distinct, model-selection-specific pattern their taxonomy does not name
explicitly, even though they cite the general model-selection-optimism
literature elsewhere. This is itself informative: three of this paper's
five mechanisms are genuine instances of an existing, general taxonomy
built for a wider scientific audience, while the CV-based
architecture/hyperparameter-selection pattern this paper's Case Study 2
documents is not explicitly one of the eight named types anywhere in
that taxonomy -- consistent with this paper's claim that hidden-state
hallucination probing's specific two-stage structure surfaces at least
one leakage pattern general cross-field taxonomies do not yet enumerate
by name.

**Benchmarking against REFORMS and the ML Reproducibility Checklist.** Two
further external standards, both broader than a leakage-specific
taxonomy, let us check whether this paper's own checklist (\S5) covers
ground those standards separately consider load-bearing. REFORMS (Kapoor
et al. 2024, *Science Advances*) is a 32-item, 8-module consensus
checklist for ML-based science: study goals (3 items), computational
reproducibility (5), data quality (7), data preprocessing (3), modeling
(6), data leakage (3), metrics and uncertainty quantification (3), and
generalizability and limitations (2). Its dedicated data-leakage module
covers exactly three questions -- train/test separation, dependencies
between train and test instances, and feature legitimacy -- all three of
which are subsumed by this paper's five-mechanism taxonomy, and Mechanism
2 (CV-based layer/hyperparameter-selection optimism) again falls outside
this module's scope for the same reason it falls outside Kapoor \&
Narayanan's eight leaf types: REFORMS' data-leakage module, like the
cross-field taxonomy, treats train/test separation and feature
legitimacy as the leakage surface, not iterative model/architecture
selection against a validation signal. REFORMS' other seven modules,
however, cover reporting practices this paper does not itself audit
(study-goal framing, uncertainty quantification beyond the specific
severity estimates reported here, external-validity discussion) --
useful context for readers applying both instruments together, since
neither is a superset of the other. The NeurIPS Machine Learning
Reproducibility Checklist (Pineau et al. 2021, *JMLR*) is reporting-
practice-oriented rather than leakage-specific: its four sections (all
models/algorithms presented; any theoretical claims; all datasets used;
all experiments) ask whether code, hyperparameters, dataset statistics,
compute budgets, and multi-seed variance are disclosed, none of which
individually detects a leakage bug the way REFORMS' or this paper's own
checklists attempt to. This paper's own reproducibility practice (full
per-seed result arrays, vendored external repositories, BCa
bootstrap/permutation tests on every severity gap) satisfies the bulk of
the Pineau checklist's disclosure-oriented items as a byproduct of the
statistical rigor already required for the severity claims themselves,
but that checklist would not, on its own, have caught any of the nine
issues this paper's own Appendix A documents finding and fixing in its
own instrument -- consistent with this paper's overall argument that
reporting-practice checklists and leakage-detection checklists are
complementary, not interchangeable, and that a paper can satisfy the
former while still committing the latter.

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
distinct from all five mechanisms documented here (each of which concerns
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

**Full 32-layer decomposition.** An earlier version of this section
disclosed this as unavailable ("no per-layer data exists to decompose
where the gap concentrates"). That was incorrect: GUARDIAN's own raw
hidden-state cache contains the complete (700, 32, 4096) Mistral-7B
representations -- all 700 samples (the 400-sample CV pool and the
300-sample held-out set) at full dimensionality, every layer, not just
L11. `code/48_case_study_2_layer_decomposition.py` repeats the identical
CV-vs-held-out comparison at all 32 layers under GUARDIAN's own
sequential `H[:400]`/`H[400:]` split (sanity-checked to reproduce
the published L11 numbers exactly: CV$=0.804$, held-out$=0.616$). Under
this split, the mean optimism gap across all 32 layers is $+0.192$ (SD
$0.053$), statistically indistinguishable from L11's own $+0.188$, and
$18$ of the other $31$ layers show a *larger* gap than L11.

**Correction: the sequential split confounds selection optimism with a
population-difference artifact.** A subsequent adversarial review of
this very analysis found that GUARDIAN's `H[:400]`/`H[400:]` split is a
fixed slice of whatever order the dataset happens to store samples in,
not a random partition -- and the two halves are separable by a
hidden-state probe at AUROC $0.734$-$0.776$, i.e. they are systematically
different populations, not exchangeable draws. The quantity that
isolates selection optimism specifically is the *selection-specific
component*, gap-at-selected-layer minus the mean gap across all 32
layers: under GUARDIAN's own sequential split this component is
$-0.004$ -- indistinguishable from zero, meaning the 18.8-point gap is
not disproportionately concentrated at the selected layer at all once
the population-difference confound is present in every layer equally.
Reversing which half is used for selection (`H[400:]` selects, `H[:400]`
held out) flips the sign of L11's own gap entirely (to $-0.014$,
selection-specific component $-0.083$), which is itself direct evidence
that the sequential-split number was tracking which half happens to be
easier, not a stable property of the selection procedure. Recomputing
the selection-specific component under $8$ randomized stratified
$400/300$ splits (never touching the sequential order) gives a mean of
$+0.007$ (SD $0.030$, range across reps $-0.033$ to $+0.058$) -- small,
sign-unstable across resamples, and about $11.5\times$ smaller than the
sequential split's apparent $+0.188$. **The corrected conclusion is
narrower than originally stated**: CV-based layer selection on this
dataset does exhibit a large, real CV-vs-held-out optimism gap
($\sim$19 points, present at essentially every layer), but the
*additional* penalty specifically attributable to argmax-over-layers
selection, once a proper random split is used, is small and not reliably
distinguishable from zero at this sample size -- not the sharply
concentrated 18.8-point selection-specific effect the original
sequential-split analysis appeared to show. Full results, all three
splits: `results/case_study_2_layer_decomposition.json`.

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

BCa bootstrap 95\% CIs on the LEAKY$-$CLEAN\_MATCHED gap
(`code/44_statistical_rigor_retrofit.py`, $10{,}000$ resamples,
paired permutation $p$ alongside Wilcoxon): $16$: $[-0.0009,+0.0027]$,
perm.\ $p=0.359$; $48$: $[+0.0002,+0.0030]$, perm.\ $p=0.021$; $128$:
$[+0.0014,+0.0055]$, perm.\ $p=0.003$; $384$: $[+0.0002,+0.0054]$,
perm.\ $p=0.051$ -- concordant with the Wilcoxon verdicts above.

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
CLEAN\_MATCHED vs.\ PLACEBO becomes significant instead at those same two
capacities ($+0.0084$, $p=0.015$; $+0.0080$, $p=0.010$). Reporting all
four capacities for this comparison, not only the two where it is
significant: capacity 128 is $+0.0024$ ($p=0.45$, not significant) and
capacity 384 \emph{reverses sign} to $-0.0037$ ($p=0.11$, also not
significant) -- permuted-label checkpoint selection nominally
outperforming honest checkpoint selection at capacity 384, the same
sign anomaly the epoch-count diagnostic (Appendix A) was built to
explain and that this appendix confirms does not recur under either
corrected real-feature calibration. Some real gap is present at every
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
regularized logistic regression to measure, rather than analytically
predict, the achieved AUROC at each candidate value. This gives a
rescaling factor that hits AUROC$=0.80$ by the measure used to calibrate
it ($0.795\pm0.013$, 10-seed re-check).

**A further, deeper problem: the calibration itself leaked label
information across the split it was later evaluated on.** A further
independent review found this rescaling factor -- and the diagnosis
built on it in an earlier version of this section -- rested on a
transform that centers class means on the *full* labeled sample before
any train/test split exists. Because each class's per-sample deviations
sum to exactly zero over the sample used to estimate the class means,
any subsequent split mechanically forces the two halves' leftover
class-mean directions into an algebraic identity
($n_{\text{tr}}\cdot\overline{\Delta}_{\text{tr}} =
-n_{\text{te}}\cdot\overline{\Delta}_{\text{te}}$), which we verified
directly at $\alpha=0$: $\cos(\Delta\mu_{\text{train}},
\Delta\mu_{\text{test}})=-0.9999$, and a linear model fit on train and
scored on test gives AUROC $=0.06$ -- not "zero separation," but
near-perfect, mechanically-induced anti-correlation. This is the exact
leakage pattern this paper's own checklist names (§5): a full-dataset,
label-dependent transform, applied before the split it is later
evaluated on -- committed by this paper's own severity-measurement
instrument.

**Fix and re-derivation.** We re-centered on train indices only, applying
the resulting affine map to both splits (`code/43_calibration_leakage_diagnostic.py`).
This removes the artifact: at $\alpha=0$, $\cos(\Delta\mu_{\text{train}},
\Delta\mu_{\text{test}})=0.0000\pm0.0007$ and train-fit$\to$test-eval
AUROC $=0.58\pm0.06$ -- a plausible, non-inverted value, not the prior
draft's mechanically-guaranteed $-0.9999$/$0.06$. Re-running the alpha
search with this corrected, nested (fold-local) centering converges to
$\alpha=0.1328$. But checking what this properly-calibrated data produces
under the paper's own downstream 5-fold OOF-plus-meta-learner severity
pipeline shows the operating-point problem persists even without the
leak: LEAKY/CLEAN\_MATCHED/PLACEBO still reach $0.966$/$0.964$/$0.949$
(capacity 128) and $0.968$/$0.966$/$0.959$ (capacity 384) -- not the
intended $0.80$. Unlike the earlier, leak-contaminated diagnosis, this
is now a properly verified finding rather than an artifact: the real
Mistral-7B/HaluEval features' within-class residual structure genuinely
carries more discriminative content than mean-only rescaling can
suppress, confirmed without the confound that previously made this
claim untrustworthy.

Because the achieved operating point ($\approx0.96$-$0.97$) still differs
from the synthetic sweep's $0.80$ calibration target, a direct
severity-magnitude comparison between the two remains invalid by this
paper's own stated principle (the same principle that flagged the
original $0.985$-ceiling error above). The real-feature test's own
internal comparison, however, is now fully trustworthy at its own
achieved operating point: LEAKY beats CLEAN\_MATCHED by $+0.0021$
(BCa 95\% CI $[+0.0007, +0.0035]$, capacity 128) and $+0.0022$
(CI $[+0.0012, +0.0034]$, capacity 384) -- both significant
($p=0.0006$, $p=0.0002$), small, and directionally consistent with,
though not numerically comparable to, the synthetic sweep's severity
range. CLEAN\_MATCHED beats PLACEBO decisively at both capacities
($+0.0157$, $p<10^{-8}$; $+0.0072$, $p<10^{-3}$), again confirming
substantial real, non-leakage-driven signal in adaptive checkpoint
selection on real features. This paper's primary severity estimate for
Mechanism 3 still rests on the two synthetic sweeps, which generate
fresh samples from a fully-specified generative process each seed rather
than rescaling a small, fixed real sample, and so share neither the
analytic-bias nor the split-leakage problem found here.

A separate FP16-vs-AWQ control on the same 50
samples finds identical AUROC (0.9600 both) at a single point estimate;
a resampled version (Appendix A) gives a 95\% CI of $[-0.061, +0.080]$,
consistent with no quantization confound rather than a demonstration
that rules one out at the resolution this severity estimate needs.
Appendix A documents the full progression of this test -- the original
architecture mismatch, the ceiling-effect error the "corrected" version
still had, the biased calibration factor a further review caught, the
split-crossing leakage that same correction's own instrument turned out
to commit, and the properly-verified operating-point mismatch that
survives the fix -- along with a further robustness check (a
resampled-effect-size reanalysis of the FP16-vs-AWQ control).

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
at both tested capacities; the biased-calibration, the
leak-contaminated bias-corrected, and the properly
train-only-centered real-feature tests all return `MIXED` at both
capacities (the first two of these three real-feature severity
estimates are not treated as informative for the reasons given above;
the train-only-centered version is). The rule is therefore reachable in every
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
all tested capacities. The LEAKY-vs-CLEAN_MATCHED_ADAPTIVE gap is larger
and more significant still, but -- as disclosed above -- that widened gap
is at least partly confounded by CLEAN_MATCHED_ADAPTIVE's own
underperformance relative to blind CLEAN_MATCHED, itself most plausibly a
symptom of the smaller ($15\%$) carve-out being a noisier selection
signal rather than confirmation that fold-reuse leakage is larger than
the primary estimate; we do not treat this control as straightforwardly
strengthening the headline severity number, only as failing to support
the alternative "any adaptive selection helps" explanation it was built
to rule out. Severity is again small (if anything, more
strongly significant, though with a reshuffled rather than uniformly stronger
capacity pattern -- see Appendix A) under a second generative process
replacing the isotropic calibration with a real, anisotropic covariance
structure. A real-feature validation attempt went through three further
rounds of correction (an architecture-mismatch fix, a biased-calibration-factor
fix, and then a fix to a split-crossing leakage bug in that very
correction's own centering step) before reaching a trustworthy account:
even leak-free, it cannot be brought onto the synthetic sweep's $0.80$
operating point (it saturates near $0.96$-$0.97$), because the real
feature set's discriminative content genuinely includes structure the
mean-only-rescaling method cannot touch -- but at its own achieved
operating point, the test now gives a small, significant,
BCa-confidence-bounded severity estimate ($+0.0021$ to $+0.0022$ AUROC)
directionally consistent with the synthetic sweeps. This paper's primary
severity estimate for Mechanism 3 still rests on the two synthetic
sweeps, which share neither the operating-point nor the leakage problem
found here. This also does
not settle MultiHaluDet's own 98.55\% AUROC's exact inflation: their real
architecture (6 transformer layers, multi-scale, heavy augmentation) is
considerably more expressive than this reconstruction. Getting this
diagnosis right required tracing each correction to its actual source
rather than accepting a partially-improved number at face value --
including, in the end, tracing a bug in the correction script itself;
Appendix A documents that process in full.

### 4.4 Case Study 4 (secondary; severity small, mechanism confirmed): Test-set-driven best-layer selection — quantized-LLM paper

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

We initially believed re-quantifying this mechanism's severity would
require their exact quantized checkpoint and dataset re-run from
scratch; a fresh review found this unnecessary. Their repo ships
per-layer, per-seed AUROC arrays for all 24 model/dataset/probe-type
combinations tested (`code/external/HallucinationPatternDetection/results/probes/*.json`,
33 layers, 3 seeds each). For each combination, selecting the best layer
using two of the three seeds and evaluating on the third held-out seed
(`code/45_case_study_4_winners_curse.py`) gives a directly-measured
winner's-curse estimate: mean $+0.0047$ AUROC across all 24
combinations (max $+0.0288$), against every layer scoring $\geq0.975$
on the two HaluEval-QA combinations most comparable to Case Study 3's
task, so the ceiling-effect concern from §4.3 recurs here too. This
mechanism's severity is small and consistent in direction with
Mechanism 3's, not unquantified.

### 4.5 Mechanism 5 (found while auditing Case Study 3's own repository): Test-set-driven operating-point selection — MultiHaluDet

A fresh, independent review found a fifth, structurally distinct
leakage mechanism sitting in the exact file this paper already audits
for Mechanism 3. `MultiHaluDet/run_pipeline.py::stage_4_ensemble` calls
`find_best_thresholds(probs, y_test)` (`src/utils/metrics.py`), which
sweeps 81 candidate F1 thresholds and the ROC-optimal (Youden) threshold
*directly against the test labels*, then reports every threshold-dependent
metric -- F1, accuracy, Matthews correlation, Cohen's $\kappa$, balanced
accuracy, and the ECE computed from those predictions -- at that
test-label-selected threshold. AUROC is threshold-free and unaffected by
this specific bug, but every other headline number MultiHaluDet reports
is test-set-optimized operating-point selection: the choice of *where to
draw the decision boundary* uses the same labels the boundary is then
scored against. This is structurally distinct from Mechanisms 1-4, which
all concern *which feature, checkpoint, or layer* to use -- this one
concerns *which threshold to report at*, given a fixed, already-chosen
model.

We quantified this severity using the same isotropic-Gaussian calibration
this paper's other severity estimates already rely on (binormal identity,
Bayes-optimal AUROC target), with a simple logistic-regression classifier
in place of the full SweepMLP-plus-meta-learner architecture (threshold-selection
leakage does not depend on which classifier produced the probabilities,
so a simpler classifier keeps this demonstration self-contained;
`code/46_mechanism5_threshold_selection.py`). **Correction:** an earlier
version of this synthetic generator itself committed the exact
calibration-inversion class of error Lesson 5a (Appendix A) already
documents finding once elsewhere in this paper's own tooling -- the
class-mean offset used the full separation rather than half of it on
each side, silently quadrupling the realized Fisher $J$ relative to the
labeled operating point. This was caught by a subsequent independent
review and fixed (verified via a dedicated, non-tautological calibration
guardrail, `code/sanity_checks.py`, now mandatory before any synthetic
generator in this project returns data). Comparing a threshold
selected on the test labels (`find_best_thresholds`, verbatim ported from
MultiHaluDet's own code) against a threshold selected on an independent
validation split never touching the test labels, across five operating
points spanning this paper's other severity estimates, under the
corrected calibration: F1 gaps ranging from $+0.0212$ ($0.985$,
MultiHaluDet's own reported regime) to $+0.0338$ (AUROC target $0.80$,
the largest of the five points, with no clean monotonic trend across
operating points: $0.70{\to}0.0228$, $0.80{\to}0.0338$, $0.90{\to}0.0311$,
$0.95{\to}0.0233$, $0.985{\to}0.0212$), all significant at $p<10^{-31}$
over $200$ seeds (BCa 95\% CIs excluding zero throughout); accuracy gaps
of comparable magnitude ($+0.0215$ to $+0.0457$).

**A necessary caveat on comparing this to Mechanism 3: these are not the
same metric, and the comparison is not like-for-like.** AUROC is
threshold-free by construction, so test-set threshold selection has
*exactly zero* effect on it -- Mechanism 5's severity on the one metric
it shares with Mechanism 3 is not small but precisely nil. The F1 gap
reported above is also structurally guaranteed non-negative: the LEAKY
threshold is chosen by an argmax over the very same grid the test F1 is
then scored on, so it cannot do worse than any other threshold drawn from
that grid, including HONEST's. An earlier draft of this section compared
these F1/accuracy gaps directly against Mechanism 3's AUROC gaps and
described Mechanism 5 as "an order of magnitude more severe" -- that
comparison conflated two different metrics measuring two different
things and has been removed. The honest statement is narrower: Mechanism
5 leaves the pipeline's reported AUROC entirely unaffected, but
materially inflates every threshold-dependent metric (F1, accuracy, MCC,
Cohen's $\kappa$, and any ECE computed from those threshold-derived
predictions) that MultiHaluDet also reports at that same selected
threshold -- a distinct, real leak in its own right, not a directly
rankable "more or less severe than Mechanism 3."

## 5. A Checklist for This Literature

[Full version in `draft/leakage_checklist.md`.] Five questions correspond
to the five mechanisms above:

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
5. Was a decision threshold or other operating point chosen by
   optimizing a metric on the test labels themselves, with that same
   test set then used to report the metric at that threshold?

**We tried to automate this checklist; the attempt itself is informative.**
We built a regex-based scanner for the original four patterns (Mechanism
5 was found after this scanner was built and is not yet covered by it)
and ran it against 7
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
of these five mechanisms remains unknown and is not estimated by this
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
the five leakage mechanisms makes no claimed difference to deployment
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
direction for their pipeline. The real-feature test (\S4.3) is also
single-dataset, single-language ($n=400$ HaluEval English `qa_samples`
only), despite MultiHaluDet's own contribution being explicitly
multilingual; and even leak-free, no calibration of this feature set's
between-class mean, at any strength, brings it onto the synthetic
sweep's exact $0.80$ operating point, because a real portion of its
discriminability lives in within-class structure the calibration method
cannot touch -- so its severity estimate, while now trustworthy, is
reported at its own ($\approx0.96$-$0.97$) operating point rather than
the synthetic sweep's, and the two are not numerically comparable.
Whether the synthetic severity estimate transfers to real hidden states
at the *same* operating point -- on this dataset, non-English datasets,
or larger ones -- remains untested rather than confirmed or excluded, and
would need a different validation strategy than mean-only rescaling to
test at this sample size and dimensionality.

## 7. Conclusion

Five distinct label-information-leakage mechanisms recur across
hidden-state hallucination-detection pipelines, including our own, and
differ substantially in severity. Case Study 1 is catastrophic ($+0.19$
AUROC) regardless of scale. Case Study 3's checkpoint-selection mechanism
is a small residual gap, significant at 2 of 4 tested capacities. A
stricter control isolating fold reuse from adaptive selection in general
finds an even larger, more significant gap, but that widened gap is
itself at least partly confounded by the control's own noisier,
smaller-carve-out selection signal (§4.3, Appendix A), so we do not read
it as simply confirming a larger severity than the primary estimate --
only as failing to support the alternative "any adaptive selection
helps" explanation. A K/selection-set-size sweep (Appendix A) confirms
this severity scales with the number of candidates and the selection-set
size, as extreme-value theory predicts, rather than with model capacity
as the original sweep varied. Mechanism 5, the test-set threshold
selection found in this same pipeline, leaves AUROC exactly unaffected by
construction but produces F1/accuracy gaps ($+0.021$ to $+0.034$) that
are large and highly significant on their own terms -- not directly
comparable in magnitude to Mechanism 3's AUROC-scale severity, since the
two measure different metrics, but a clear case that a pipeline can be
simultaneously clean on one reported metric and leaking on another. That
difficulty --
needing calibration, budget-matching, seed-matching, and an
alternative-confound test all to hold at once before trusting a single
number -- is the strongest argument in this paper for the checklist we
provide, more so than any single mechanism in isolation.

The checklist lets researchers and reviewers in this fast-growing
subfield ask precise, mechanism-specific questions of a pipeline's
evaluation protocol, replacing "did you use cross-validation correctly"
as a single, undifferentiated concern.

The automated-scanner exercise (§5) covers seven repositories in total:
two already-confirmed positive case studies (§4.3-4.4, MultiHaluDet and
HallucinationPatternDetection) and five additional, not-previously-examined
repositories, which surfaced no new confirmed positives -- too small a
sample to support any claim about how common these five mechanisms are
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

**TL;DR.** Eleven issues were found and fixed via iterative review while
deriving this case study's severity estimate (full table below). In
order: (1) an uncalibrated
synthetic sweep saturated at AUROC=1.0; (2) the calibration formula
itself was inverted; (3) LEAKY/CLEAN were not budget- or seed-matched;
(4) the real-feature test's architecture did not match the synthetic
sweep's (single-fold MLP vs. the intended 5-fold OOF+meta-learner);
(5) the "corrected architecture" version still ran at a saturated
operating point (AUROC$\approx0.985$); (6) the rescaling factor used to
fix (5) was itself computed from a biased, noise-inflated estimator;
(7) a permutation-null number was misattributed to the wrong check;
(8) **the calibration transform itself leaked label information across
the split it was later evaluated on** (full-sample class-mean centering
before the split) -- this is the load-bearing finding of this appendix,
and the reason this paper's own severity-measurement instrument is used
as a worked example of Mechanism 1 in \S5; (9) fixing (8) left the
operating-point mismatch from (5) unresolved, so the corrected test is
reported at its own operating point rather than retired. **The numbers
that survive all nine and are the ones §4.3 actually reports:**
LEAKY beats CLEAN\_MATCHED by $+0.0021$ (capacity 128, $p=0.0025$) and
$+0.0022$ (capacity 384, $p=0.0003$) on the train-only-calibrated
real-feature test (issue 8's fix); the two synthetic sweeps (isotropic,
anisotropic) remain the paper's primary severity estimate since they
share neither the architecture-mismatch (4) nor the split-leakage (8)
problem the real-feature test needed fixing for. A tenth, later addition
-- the fidelity extension modeling MultiHaluDet's actual LR-scheduler and
early-stopping-tied-to-validation behavior, not just checkpoint
selection -- went through two further, sequential control-construction
bugs of its own (an earlier version selected the retrain epoch count from
the leaky run itself, which cannot differ from it by construction; the
fix for that then discarded its own correctly-trained control model and
retrained blind with no scheduler at all, reintroducing a different
confound) before converging on a genuine, scheduler-bearing,
disjoint-carve-out control, which found a substantially *larger* gap
($+0.0111$, $p=4.3\times10^{-9}$) than the checkpoint-selection-only
harness above, described later in this appendix.

**Correction table.** Each row is one issue found by iterative review while
deriving §4.3's severity estimate; full derivations live in the cited
script/result files, not below.

| # | Issue | Fix | Resulting number |
|---|---|---|---|
| 1 | Uncalibrated sweep saturated at AUROC$=1.0$ | Calibrate via $\text{AUROC}=\Phi(\sqrt{J/2})$ | -- |
| 2 | Calibration formula was inverted ($\Phi(\sqrt J/2)$, the accuracy identity, not AUROC) | Corrected inversion, $J=2\Phi^{-1}(0.80)^2=1.417$ | Verified $\Phi(\sqrt{J/2})=0.800$ |
| 3 | LEAKY/CLEAN not budget- or seed-matched (CLEAN's ES carve-out $\approx$15\% less data; matched retrain used a different seed) | Match both budget and seed | Result now stated in §4.3 |
| 4 | Real-feature test's architecture didn't match the synthetic sweep's (single-fold MLP vs. intended 5-fold OOF+meta-learner) | Rerun with the actual architecture (`code/25`) | $+0.0002$ ($p=0.56$, cap. 128), not significant |
| 5 | "Corrected architecture" still ran at a ceiling operating point (AUROC$\approx0.985$), making its MDE incomparable to the synthetic sweep's | Rescale real features to the same $0.80$ target (`code/31`) | $+0.0014$/$+0.0010$ (cap. 128/384), inside the synthetic range but individually underpowered |
| 6 | Rescaling factor $\alpha$ itself came from a biased estimator ($J_{\text{real}}$ unstable at $d\approx n$; permutation test gives $J_{\text{perm}}=2.31 > J_{\text{target}}=1.42$) | Empirical, CV-calibrated bisection on *achieved* AUROC (`code/33`) | Converges to $\alpha=0.2031$ |
| 7 | A permutation-null number ($0.504\pm0.043$) was misattributed to this calibration check | Reconciled: $0.504$ is a different, unrelated check; this check's real value is $0.1202\pm0.0170$ | Both numbers now correctly labeled |
| 8 | **The calibration transform itself leaked label information across the split it was evaluated on** -- full-sample class-mean centering before the split forces $\cos(\Delta\mu_{\text{tr}},\Delta\mu_{\text{te}})\approx-1$ by construction (verified: $-0.9999\pm0.0000$, train$\to$test AUROC$=0.062$) | Re-center on train indices only (`code/43`, `apply_calibration_train_only`); re-run alpha search per-fold | $\alpha=0.1328$; $\cos=0.0000\pm0.0007$, AUROC$=0.584\pm0.059$ (no longer inverted) |
| 9 | Operating-point mismatch (issue 5) survives the leakage fix: train-only-calibrated pipeline still runs at AUROC$\approx0.96$-$0.97$, not $0.80$ | Report at the test's own operating point rather than retire it | LEAKY beats CLEAN\_MATCHED by $+0.0021$ (cap. 128, $p=0.0025$) / $+0.0022$ (cap. 384, $p=0.0003$) -- **the number §4.3 reports** |
| 10 | Checkpoint-selection-only harness may understate MultiHaluDet's actual leak: the real trainer couples an LR scheduler and early stopping to the same validation fold every epoch, not just a final checkpoint pick | Port the scheduler+shared-patience mechanic into the harness (`code/49`). Two sequential bugs found along the way: (a) an early version selected the retrain epoch count from the leaky run's own loop, which can never differ from it by construction; (b) the fix for (a) then discarded the correctly-trained control model and retrained blind with no scheduler at all, confounding "reuses the leaky fold" with "has an adaptive LR mechanism at all." Final fix: keep the model already trained with a real scheduler on a disjoint carve-out, rather than discarding it | LEAKY\_PLUS\_LRSCHED beats CLEAN\_MATCHED\_PLUS\_LRSCHED by $+0.0111$ ($p=4.3\times10^{-9}$) -- **roughly $3$-$4\times$ larger** than the checkpoint-only estimate (CLEAN\_MATCHED\_PLUS\_LRSCHED itself beats PLACEBO\_PLUS\_LRSCHED decisively, $+0.0498$, $p=3.7\times10^{-16}$) |
| 11 | Two of this paper's own synthetic-data generators (`code/46`, Mechanism 5; `code/47`, the K/selection-set-size sweep) independently committed the same calibration-inversion error class as issue 2, in a new form: the per-class mean offset used the full separation ($\pm\text{class\_sep}$) rather than half ($\pm\text{class\_sep}/2$), silently realizing $4\times$ the intended Fisher $J$ | Corrected both generators to the $\pm\text{class\_sep}/2$ convention already used elsewhere (`code/02d`), and added a new, mandatory, non-tautological guardrail (`code/sanity_checks.py`) every synthetic generator in this project must now call before returning data. The guardrail checks the bias-corrected realized Fisher $J$ ratio directly (not a fitted classifier's CV AUROC, which undershoots by regularization, a separate real effect -- see issue 5) | Ratio cleanly separates correct generators (0.33-1.83 across 2000 seeds, five operating points) from the $4\times$ bug (never below 2.77) |

**Issue 8 is the load-bearing finding of this appendix**, and the reason
this paper's own severity-measurement instrument is used as a worked
example of Mechanism 1 in §5: a full-dataset, label-dependent transform
applied before the split it is later evaluated on -- the exact pattern
this paper's checklist tells readers to look for in other people's code,
found instead in ours. The synthetic sweeps (isotropic, anisotropic)
remain this paper's primary severity estimate for Mechanism 3, since
they share neither the architecture-mismatch (4) nor the split-leakage
(8) problem the real-feature test needed fixing for.

**What this correction history demonstrates about itself.** A single
confirmatory- or null-looking number was not enough at any step above --
a correctly-inverted formula, a budget/seed-matched control, adequate
power, an unbiased calibration estimator, and a train-only split all had
to hold *simultaneously*, and each in turn was checked, found wanting,
and fixed by a fresh round of review. A pre-registered decision rule
(`GENUINE_LEAK_CONFIRMED` / `CONFOUND_CONFIRMED_NO_REAL_LEAK` / `MIXED`,
from `code/02d`) returns `MIXED` on most tested versions of this
experiment and `CONFOUND_CONFIRMED_NO_REAL_LEAK` on the one
ceiling-confounded version -- an earlier claim that the confound branch
was structurally unreachable was itself wrong, and is corrected here;
`MIXED` is the modal but not universal honest label for what the data
show across six tested configurations.

**Two smaller, fully-superseded checks, retained for completeness.**
Exact per-seed gap SDs (not back-of-envelope estimates) confirm the
train-only-calibrated test (issue 8's fix) is adequately powered for the
gap it reports (MDE $0.00198$/$0.00154$ at capacities 128/384, both at
or below the observed $+0.0021$/$+0.0022$); earlier, architecturally-
mismatched versions were not. A single-point-estimate FP16-vs-AWQ
quantization control was re-run with 100 seed-resampled splits through
this paper's own severity architecture: gap $+0.0068$, 95\% CI
$[-0.061,+0.080]$ (includes zero) -- consistent with no confound, not a
demonstration ruling one out. An epoch-count anomaly specific to the
now-superseded single-fold architecture (CLEAN\_MATCHED underperforming
PLACEBO by $-0.0025$) reverses sign under every corrected version
(CLEAN\_MATCHED outperforms PLACEBO by $+0.0157$ under the final,
train-only-calibrated test) and does not recur; full detail in
`results/real_feature_leakage_diagnostics.json`.

**A ninth check: does severity scale the way extreme-value theory
predicts?** The reasoning behind Mechanism 3 is a winner's-curse
argument: selecting the best of $K$ noisy validation estimates inflates
the reported score by $\text{gap}\approx c\cdot\sigma_{\text{val}}
\sqrt{2\ln K}$, shrinking with validation-set size $n_{\text{val}}$.
`code/47_selection_multiplicity_sweep.py` tests this directly (decoupled
`data_seed`/`split_seed`/`fold_seed_base`/`init_seed_base`, so no single
draw can produce all three patterns by coincidence). **Correction:** an
earlier version of this sweep's synthetic generator had the same
calibration bug as Mechanism 5 (Appendix A, issue 11) -- corrected here
using the identical fix and the same non-tautological
`sanity_checks.py` guardrail. Sweeping
$K\in\{1,3,5,10,15,25,45,75,135,225\}$ at fixed capacity now gives a gap
that rises essentially monotonically with $K$ ($+0.0000$ at $K{=}1$ to
$+0.0052$ at $K{=}225$, with only a brief plateau between $K{=}10$ and
$K{=}15$), fit by $c=0.0417$, $R^2=0.756$ -- a substantially tighter fit
than the uncorrected sweep previously reported ($R^2=0.54$), and every
cell from $K{=}3$ upward is individually significant ($p<0.0002$
throughout). Sweeping $n_{\text{val}}$ (via
$N_{\text{SAMPLES}}\in\{350,700,2800\}$) does not shrink cleanly toward
zero as the simplest reading of the theory predicts: the gap is smallest
and significant at the largest $n_{\text{val}}$ ($+0.0005$,
$n_{\text{val}}{\approx}448$, $p=0.047$), but is *larger*, not smaller,
at $n_{\text{val}}{\approx}112$ ($+0.0036$, $p=0.0012$) than at
$n_{\text{val}}{\approx}56$ ($+0.0026$, not significant, $p=0.33$, CI
crossing zero) -- consistent with the theory's qualitative claim that
very small validation sets are noisy enough to leave the winner's-curse
signal underpowered, but not with a clean monotone $1/\sqrt{n_{\text{val}}}$
decline across all three points tested. Sweeping the operating point
($\text{TARGET\_AUROC}\in\{0.70,\ldots,0.985\}$) gives a cleanly monotone
*point-estimate* decline toward the ceiling ($+0.0093\to+0.0002$), but
significance is not monotone: $0.70$, $0.80$, and $0.95$ are each
significant ($p=0.0001$, $p=0.0012$, $p=0.037$), while $0.90$ and $0.985$
-- MultiHaluDet's own operating point -- are not ($p=0.067$, $p=0.219$),
the latter consistent with the ceiling-effect ranking noise this paper's
own severity harness already documents (Appendix A, issue 5) rather than
with the mechanism vanishing at high AUROC specifically. Taken together:
the $K$-scaling prediction is now confirmed with a materially tighter
fit than previously reported; the $n_{\text{val}}$- and operating-point
dependencies point in the theory's qualitative direction but are neither
perfectly monotone nor individually significant at every cell, so we read
this as supportive of, not a precise quantitative confirmation of, the
extreme-value-theory account. Full per-cell results:
`results/selection_multiplicity_sweep.json`.

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

Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility
Crisis in ML-based Science. *Patterns*, 4(9). arXiv 2207.07048.

Kapoor, S., et al. (2024). REFORMS: Consensus-based Recommendations for
Machine-learning-based Science. *Science Advances*, 10(18). arXiv 2308.07832.

Pineau, J., et al. (2021). Improving Reproducibility in Machine Learning
Research (A Report from the NeurIPS 2019 Reproducibility Program).
*Journal of Machine Learning Research*, 22(164), 1-20.

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
