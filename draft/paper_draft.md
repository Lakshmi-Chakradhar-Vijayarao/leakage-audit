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
$p<0.0001$ at both tested capacities) -- ruling out the most plausible
alternative explanation for the effect. A real-Mistral-7B-feature
replication lands inside the synthetic estimate's range.

We provide a checklist covering all four mechanisms and test whether it
can be automated: a regex-based scanner over 9 repositories (2 confirmed
leaky, 7 externally sourced) produces zero true positives and misses both
known bugs -- this class of leakage resists simple pattern-matching in
both directions.

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
not supported: the residual tracks fold reuse specifically, not adaptive
selection in general.

**Real-feature validation.** Feeding real Mistral-7B/HaluEval features
($n=400$, MultiHaluDet's own unmodified feature-extraction code, a 4-bit
AWQ checkpoint pinned to a fixed revision) through the identical
LEAKY/CLEAN\_MATCHED/PLACEBO test gives LEAKY beating CLEAN\_MATCHED by
$+0.0026$ AUROC ($p=0.0012$, $n=100$ seeds), inside the synthetic sweep's
estimated range, and replicating at the architecturally-matched capacity
384 ($+0.0021$, $p=0.0010$). One sanity check on this real-feature run
does fail: CLEAN\_MATCHED underperforms PLACEBO by $-0.0025$
($p=0.0115$), sharpening rather than weakening with more seeds. This
traces to an epoch-count mismatch, not a challenge to the headline
result: CLEAN\_MATCHED's epoch count is fixed from an earlier,
smaller-budget run, while PLACEBO adaptively drifts to a later,
better-fit epoch even against meaningless permuted labels (mean epoch
17.7 vs.\ 12.18, $p=0.012$); the per-seed epoch gap predicts the AUROC
gap directly ($r=-0.594$, $p=7.5\times10^{-11}$). This account is
correlational, not an intervention-confirmed causal mechanism. A separate
FP16-vs-AWQ control on the same 50 samples finds identical AUROC (0.9600
both), ruling out quantization as the source of either result. Appendix
A documents this diagnostic and two further robustness checks (an exact,
not back-of-envelope, minimum-detectable-effect calculation, and a
resampled-effect-size reanalysis of the FP16-vs-AWQ control) in full.

**Bottom line.** The mechanism itself -- checkpoint choice as a function
of the reused fold's own labels -- is unambiguous and code-verified. Its
quantified severity, once budget- and seed-matched, is small ($+0.0009$
to $+0.0034$ AUROC on the synthetic sweep, $+0.0021$ to $+0.0034$ on real
features) and significant at most but not all tested capacities, and this
significance strengthens under the stricter adaptive-selection control
above. This does not settle MultiHaluDet's own 98.55\% AUROC's exact
inflation: their real architecture (6 transformer layers, multi-scale,
heavy augmentation) is considerably more expressive than this
reconstruction. Getting this number right required simultaneously correct
calibration, budget-matching, and seed-matching; Appendix A documents
that process in full.

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
externally-found hidden-state-probing repositories plus the two
already-audited case studies (MultiHaluDet, HallucinationPatternDetection)
-- 9 total (`code/04_leakage_linter.py`,
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
scanner looks for. Zero true positives among 5 newly-scanned repos, two
false negatives on the two known positives, seven-for-seven false
positives on its raw hits: at this level of tooling, an automated scanner
is not a substitute for manual, one-repo-at-a-time reading.

This is a prevalence statement about 2 of 9 repos, not a representative
sample -- 7 of 9 were found via targeted search for exactly this failure
class (HaRP and GUARDIAN, this paper's own pipelines, were never run
through the scanner at all). The true population-level prevalence of any
of these four mechanisms remains unknown and is not estimated by this
count. We report the negative automation result because the same
variable-naming and control-flow subtlety that makes this leakage easy to
introduce by accident also makes it resist simple pattern-matching, in
both directions.

## 6. Discussion

**Data and code availability.** All code, cached result JSONs, and the
paper source are public at
`https://github.com/Lakshmi-Chakradhar-Vijayarao/leakage-audit`. Case
Studies 3 and 4 are fully reproducible from what ships with this paper:
pinned third-party commits, our audit scripts, and the resulting JSON
logs. Case Studies 1 and 2 are independently reproducible via two
additional public repositories,
`https://github.com/Lakshmi-Chakradhar-Vijayarao/harp-leakage-case-study`
and
`https://github.com/Lakshmi-Chakradhar-Vijayarao/guardian-leakage-case-study`,
each containing only the subset of the original project relevant to the
specific finding reported, not the full original codebase. The regex
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

**This audit's central lesson recurs across three companion studies from
the same project**, none sharing a leakage mechanism with the four case
studies here: a geometric-certificate paper finds a near-saturated
hallucination benchmark is mostly explained by answer length, not
truthfulness; an FFN-mechanism paper finds a cross-architecture component
comparison reverses once an instruction-tuned model is queried with its
correct chat template; an agentic-failure taxonomy finds a hidden-state
"early-warning" signal is mostly a difficulty readout once tested
directly. All four studies share one discipline: a claimed severity or
signal is not established until the confounds that could produce it for
free have been named and tested, not merely acknowledged as possible.

**Limitations.** We could not re-run either external paper's full
7B-scale pipeline within this paper's compute budget, so Case Studies 3
and 4's severity for the *actual published numbers* remain open
questions; our reconstruction of Case Study 3 supports a small, positive
estimate at 2 of 4 tested capacities, and MultiHaluDet's actual
architecture (6 transformer layers, multi-scale, heavy augmentation) is
more expressive still, so this should not be over-read in either
direction for their pipeline. The real-feature validation is also
single-dataset, single-language ($n=400$ HaluEval English `qa_samples`
only), despite MultiHaluDet's own contribution being explicitly
multilingual; whether this severity estimate holds in non-English
configurations is untested.

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

This paper audits seven external repositories, of which two are
confirmed positive -- too small a sample to support any claim about how
common these four mechanisms are across the wider literature, and we do
not make one. A pre-registered audit applying this checklist to a fixed,
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

**The lesson this demonstrates about itself, not just argues for.** A
single confirmatory-looking or null-looking number is not sufficient to
close a leakage-severity question, even at adequate seed count and with a
placebo control. Five controls must hold *simultaneously*: a
correctly-inverted calibration formula; a budget-matched control checked
for confounds introduced by the matching procedure itself; adequate
power; a permuted-label placebo; and consulting one's own pre-registered
decision rule rather than narrating past it. Get any one wrong, in either
direction, and a confident-looking conclusion can be wrong.

**The pre-registered decision rule is itself flawed.** A rule classifying
each capacity as `GENUINE_LEAK_CONFIRMED`, `CONFOUND_CONFIRMED_NO_REAL_LEAK`,
or `MIXED` from the placebo-relative gaps (`code/02d_corrected_capacity_placebo_sweep.py`)
returns `MIXED` at all four capacities. Its
`CONFOUND_CONFIRMED_NO_REAL_LEAK` branch requires
$\text{leaky\_minus\_placebo}\ p>0.05$, but that comparison is significant
at $p<10^{-8}$ at every capacity in every version of this experiment --
the branch is structurally unreachable, so the rule can only ever return
`GENUINE_LEAK_CONFIRMED` or `MIXED`, never a clean "no leak" verdict. This
is a flawed pre-registration, not a rigor credential; `MIXED` is the
honest label for what the data show.

**Exact, not back-of-envelope, minimum detectable effect.** A diagnostic
rerun (`code/19_real_feature_leakage_diagnostics.py`) saved full per-seed
gap arrays for the real-feature validation. Exact per-seed gap SD for
LEAKY-vs-CLEAN\_MATCHED is $0.008294$, giving an 80\%-power MDE (two-sided
$\alpha=0.05$) of $0.00232$ -- essentially identical to the
back-of-envelope estimate used in §4.3 ($\approx0.0023$). The observed gap
($0.00261$) sits comfortably above this exact MDE. For
CLEAN\_MATCHED-vs-PLACEBO, exact SD $=0.009420$, MDE $=0.00264$: the
observed gap ($-0.00248$) sits just under this MDE, consistent with the
sanity-check anomaly being real but modest rather than certain to be
detected at this $n$.

**A single point estimate is weaker rigor than every other comparison in
this paper.** The FP16-vs-AWQ quantization control (§4.3) was originally a
single 5-fold-CV logistic-regression point estimate at $n=50$. We reran
the identical comparison through this paper's own capacity-128
SweepMLP architecture -- the same one the headline severity claim uses --
with 100 seed-resampled splits instead of one point estimate (reusing
cached features, no new model inference). Result: mean AUROC $0.946$
(FP16) vs.\ $0.939$ (AWQ), gap $+0.0068$, 95\% CI $[-0.061, +0.080]$
(includes zero), Wilcoxon $p=0.148$ -- not significant, and consistent
with the original point estimate's conclusion, now with an effect-size
distribution rather than a single number
(`results/fp16_vs_awq_control_matched_architecture.json`).

**Epoch-count mechanism behind the CLEAN\_MATCHED-vs-PLACEBO sanity-check
anomaly (§4.3).** Instrumented epoch counts across 100 seeds
(`code/19_real_feature_leakage_diagnostics.py`) show CLEAN's
early-stopped checkpoint averages epoch $12.18$, while PLACEBO's
checkpoint-selection -- even against permuted, meaningless validation
labels -- averages epoch $17.7$ ($p=0.012$): with no real stopping signal
to act on, PLACEBO's selection drifts toward later, better-fit-to-$X$
epochs, while CLEAN's genuine early-stopping on a smaller, noisier
held-out slice stops sooner. The per-seed epoch gap predicts the AUROC
gap directly ($r=-0.594$, $p=7.5\times10^{-11}$): a structural consequence
of CLEAN\_MATCHED's fixed epoch count under-training relative to where
PLACEBO's own selection lands, not a challenge to the LEAKY-vs-CLEAN\_MATCHED
severity estimate. This account is correlational, not an
intervention-confirmed causal mechanism (we did not force a range of
fixed epoch counts on CLEAN\_MATCHED and observe AUROC track it
directly); LEAKY, CLEAN\_MATCHED, and PLACEBO share an identical random
seed and identical training data, differing only in which epoch along
one shared training trajectory gets selected -- a deliberate,
confound-minimizing design that also means the epoch-gap correlation is
partly structural to that shared-trajectory comparison, not from two
independently-trained models.

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
