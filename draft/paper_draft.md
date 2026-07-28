# Not All Leaks Are Equal: A Taxonomy and Audit of Label-Information Leakage in Hidden-State Hallucination Detection

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

Linear and shallow-MLP probes on LLM hidden states are now a standard tool
for detecting hallucination and factual error. Reported AUROCs routinely
exceed 0.90, some reaching a perfect 1.000. We identify a recurring family
of methodological errors behind these numbers: using evaluation labels,
even indirectly, to make a choice, then reporting the resulting metric as
if that choice were free.

We identify and verify four distinct mechanisms in this family, at two
different evidentiary tiers. Two are demonstrated in our own prior,
previously-unpublished pipelines; we release the exact training code and
result logs that produced their severity numbers, so an independent
party can check them, though none has done so yet. Two are demonstrated
directly in the released code of externally published, third-party
papers, verified against pinned commits -- a stronger tier, since the
audited code was authored independently of this paper. (1) Full-dataset fit-then-score feature leakage feeding a nested
cross-validation loop: +0.19 AUROC inflation, verified before and after on
real Qwen2.5-3B hidden states. (2) Cross-validation-based
hyperparameter/layer selection reported as a held-out estimate: an
18.8-point AUROC gap between the CV-selection number and true held-out
performance, on Mistral-7B hidden states. (3) Per-fold deep-model
checkpoint selection using the same fold whose features are later reused
downstream as "out-of-fold," verified directly in a published
multilingual hallucination-detection pipeline's (MultiHaluDet) released
code and quantified via a controlled, difficulty-calibrated synthetic
reconstruction. (4) Test-set-driven best-of-many-hypotheses layer
selection with no correction for the number of hypotheses tried, verified
in a second published paper's released code.

We treat these as four mechanisms rather than one, because they occur at
four different, independently-fixable pipeline stages -- feature
construction, hyperparameter selection, checkpoint selection, and final
architecture selection -- and detecting or preventing one does nothing
to detect or prevent the others; a single generic "avoid data leakage"
check does not tell a practitioner which of these four places in their
own pipeline to inspect. The shared root logical structure (evaluation
labels influencing a choice, then that choice reported as free)
unifies them as one family; the four-way split is what makes that
family actionable, since each of the four stages calls for a different,
specific fix that the other three do not address.

These four mechanisms differ sharply in severity, and even though all
four are now independently verifiable, a reader should still weight
mechanisms (1) and (3)'s severity numbers -- the two quantified below
in detail -- against each other carefully, not at face value. Mechanisms (1) and (2) are
the authors' own prior, unpublished pipelines (HaRP, GUARDIAN),
retroactively recognized as carrying these patterns; their severity
numbers (+0.19 AUROC; 18.8 AUROC points) are traceable to the exact
training scripts and result logs that produced them, now released
alongside this paper (see Data and Code Availability) rather than
resting on narrative report alone. Mechanisms (3) and (4) are
externally published, code-available systems verified directly against
pinned third-party commits, and this paper's audit scripts and result
logs for them are included in full. The two severity
numbers below are also not on the same scale even within this stronger
evidentiary tier: mechanism (1)'s +0.19 is an effect size on real hidden
states, while mechanism (3)'s severity is measured on a linear-Gaussian
synthetic reconstruction, not MultiHaluDet's own real, non-Gaussian
representational geometry. A properly calibrated,
budget-matched, seed-matched reconstruction of mechanism (3) ($n=100$
seeds, four capacities, permuted-label placebo) finds a real residual
leak, significant at two of four capacities after Holm-Bonferroni
correction (48 units, $p=0.015$; 128 units, $p=0.008$) and not significant
at the other two (16 units, indistinguishable from zero; 384 units,
underpowered at $p=0.077$). Getting this number right required
simultaneously correct handling of statistical power, calibration,
training-budget matching, and seed matching; Appendix A documents this
process in full, as supporting evidence for the checklist we
provide. A defensible severity estimate needs a correctly-inverted
calibration formula, a budget-matched control checked for its own
confounds, adequate power, a permuted-label placebo, and a
pre-registered decision rule -- which is itself flawed:
its \texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK} branch is structurally
unreachable given how large the LEAKY-vs-PLACEBO gap always is.

The checkpoint-selection channel is also confirmed on real hidden-state
geometry: extracting features via MultiHaluDet's own unmodified code
(Mistral-7B, 4-bit AWQ) on $n=400$ real HaluEval samples and re-running
the identical LEAKY/CLEAN\_MATCHED/PLACEBO test at $n=100$ seeds gives a
real-feature severity estimate ($+0.0026$ AUROC, $p=0.0012$) inside the
synthetic sweep's own range ($+0.0009$ to $+0.0034$). A separate control
rules out AWQ quantization itself as a confound: full-precision FP16
features on the same 50 samples give an identical AUROC (0.96 vs.\ 0.96,
diff$=0$). This real-feature run's own sanity check shows a small anomaly:
CLEAN\_MATCHED underperforms a permuted-label PLACEBO by $-0.0025$
AUROC ($p=0.0115$), with significance sharpening rather than weakening at $n=100$ --
inconsistent with sampling noise. The explanation is
correlational rather than intervention-confirmed: CLEAN\_MATCHED's
fixed epoch count under-trains relative to where PLACEBO's own
checkpoint selection lands, and the per-seed epoch gap predicts the
AUROC gap directly ($r=-0.594$, $p=7.5\times10^{-11}$) -- a construction
artifact of one control, not a challenge to the checkpoint-reuse severity
estimate itself, which is adequately powered (exact, not
back-of-envelope, MDE) and holds.

We provide a checklist for identifying which of these four patterns a
given hidden-state probing pipeline is vulnerable to, built directly from
this paper's own four-round history of getting one number wrong in both
directions. We also tried to automate this checklist with a regex-based
scanner across 7 real external repos, and the attempt itself proved more
instructive than any prevalence count: zero true positives among the 5
newly-scanned repos, two demonstrated false negatives on the two known
true positives in this corpus, and all 7 raw regex hits (across 2 flagged
repos) confirmed as false positives -- this kind of leakage resists
simple automated detection in both directions.

## 1. Introduction

Detecting hallucination by probing an LLM's internal hidden states is now
one of the most active subfields in LLM reliability research. This
approach differs from output-level signals like token entropy or
self-consistency. It has grown fast over the past two years. The appeal
is direct: a linear probe is cheap to train and requires no additional
model calls. Reported AUROCs are often striking -- 0.90 or higher,
sometimes reaching a perfect 1.000 in at least one published
configuration we examine below.

This field's samples are few and its dimensions are many: hundreds of
samples, thousands of hidden dimensions. High reported numbers in that
setting should invite scrutiny of the evaluation protocol producing
them. This paper began as an accident, not a planned survey. We were
building our own hidden-state hallucination detector -- a project we
call HaRP, targeting Qwen 2.5 3B. While building it, we discovered and
fixed a nested-cross-validation leakage bug. That bug had inflated our
own reported AUROC by 0.19 points. Catching a genuine, quantifiable,
easy-to-miss methodological error in our own pipeline raised a question:
is this an isolated mistake, or a structural hazard the broader
literature shares? We show it is the latter, but not uniformly. We find
at least four *distinct* mechanisms by which evaluation labels can leak
into a reported metric. We verify two of them in externally published,
code-available pipelines. A controlled synthetic experiment shows these
mechanisms differ substantially in how much inflation they actually
cause -- a distinction the field currently has no vocabulary for.

**Contributions:**
1. A verified, quantified account of a nested-CV feature-leakage bug in
   our own pipeline (HaRP). We report the exact before/after numbers and
   an independent cross-check confirming the fix is correct.
2. A verified, quantified account of a cross-validation-based
   hyperparameter-selection optimism bias in a second pipeline of ours
   (GUARDIAN). We show an 18.8-point AUROC gap between the CV-selection
   number and genuine held-out performance.
3. A code-verified identification of a third, distinct leakage mechanism
   (per-fold checkpoint selection) in an externally published,
   MeLLM-workshop paper's released pipeline, plus a controlled synthetic
   reconstruction -- calibrated to a realistic task difficulty via a
   closed-form Fisher-ratio/AUROC identity -- that quantifies this
   mechanism's severity in isolation and finds it markedly milder than
   mechanism (1) at matched scale.
4. A code-verified identification of a fourth, distinct mechanism
   (test-set-driven best-of-many-layers selection) in a second externally
   published paper's released pipeline.
5. A checklist synthesizing all four mechanisms into actionable questions
   a researcher or reviewer can ask of any hidden-state probing pipeline.

## 2. Related Work

**General methodological leakage in ML evaluation.** Leakage, broadly
defined, means information from the evaluation target influences a
reported metric. This is a well-documented hazard in data mining
(Kaufman et al. 2012). It is also a classical result in model-selection
statistics: cross-validation-based hyperparameter or feature selection
produces optimistic performance estimates unless the selection step is
nested inside the outer evaluation loop (Varma & Simon 2006). Case
Studies 2 and 4 are direct, LLM-hidden-state-probing instances of exactly
this general phenomenon. Case Studies 1 and 3 are more specific: they
arise from the two-stage, feature-extraction-then-classification
structure common in modern hidden-state probing pipelines. To our
knowledge, no prior work names this as a distinct pattern in this
literature.

**Uncertainty and calibration in LLMs.** A fast-growing body of work
studies whether LLM confidence, however measured, is calibrated to actual
correctness. This spans three strands: broad survey work on the field,
linear-probe-based confidence estimation from hidden states, and
conformal-prediction approaches that give a formal coverage guarantee on
generated content. One example of the third strand: factuality-guarantee
conformal frameworks that decompose generations into sub-claims and
calibrate acceptance thresholds against a held-out calibration set. These
lines of work treat the *evaluation protocol itself* as given; they focus
on improving the calibration method. Our contribution sits upstream of
that question -- if the reported AUROC or calibration metric a paper
builds on is itself inflated by one of our four leakage mechanisms, any
downstream calibration analysis inherits that inflation.

**Benchmark-construction and label-quality critiques.** A separate,
also-recent thread of methodological critique targets two adjacent
problems, not evaluation-protocol leakage. First: benchmark-construction
artifacts, where question/answer pool overlap or construction choices
bias apparent detector performance, independent of any modeling choice.
Most directly, PARALLAX (arXiv 2605.17028) runs a large-scale audit --
22 detection methods, 12 open-source models across 6 architectural
families, 6 corpora -- and finds that four of six corpora embed the
ground-truth answer directly in the input prompt, letting a naive
text-similarity baseline reach near-perfect detection without touching
model internals at all; once this and related artifacts are controlled
for, most established baselines fall to near chance, with supervised
upper-layer hidden-state probes among the few consistent exceptions.
This is independent, larger-scale, external evidence that the general
concern this paper raises about the field's reported AUROCs -- that
apparent progress is frequently a protocol or construction artifact
rather than a genuine detection capability -- generalizes well beyond
the four case studies audited here, even though PARALLAX's specific
mechanism (ground-truth-in-prompt benchmark construction) is distinct
from every leakage mechanism this paper documents (all of which concern
a downstream selection step, not the benchmark's own construction).
Second: label-quality issues, where commonly-used surface-form
correctness metrics (ROUGE-L, exact match) diverge sharply from human or
LLM-judge assessments of actual factual correctness. Both are real and
complementary to this paper's concern. A hallucination detector's
reported AUROC can be inflated simultaneously by evaluation-protocol
leakage (this paper's subject), benchmark-construction artifacts
(PARALLAX's subject), and label noise (a separate, already-documented
concern). Fixing the leakage patterns we
identify does not, by itself, address label-quality or
benchmark-construction problems in the underlying dataset.

**Pretraining-data contamination of evaluation benchmarks.**
A distinct, fast-growing 2023-2026 thread studies a
different question. It asks whether an LLM's *pretraining corpus*
overlapped with a benchmark's test questions or answers. This overlap
inflates apparent capability or accuracy, independent of any
evaluation-protocol choice. Three examples: membership-inference-style
detectors that compare a model's token-level likelihood on
suspected-contaminated text against a reference distribution (Shi et al.,
"Detecting Pretraining Data from Large Language Models," ICLR 2024,
arXiv 2310.16789, the Min-K\%~Prob method); guided-instruction probes
that elicit verbatim or near-verbatim benchmark content from a model to
demonstrate exposure (Golchin \& Surdeanu, "Time Travel in LLMs: Tracing
Data Contamination in Large Language Models," ICLR 2024, arXiv
2308.08493); and calls to measure and report per-benchmark contamination
risk as standard practice (Sainz et al., "NLP Evaluation in Trouble: On
the Need to Measure LLM Data Contamination for Each Benchmark," EMNLP
Findings 2023, arXiv 2310.18018). This is a different failure mode
from every leakage pattern in this paper. Contamination concerns whether
the model has *already seen* the test answer during pretraining. Our
four case studies concern something else: whether a *downstream
selection step* (fold reuse, checkpoint choice, hyperparameter search)
was itself informed by test-set signal after the model was already
fixed. The two are complementary and, in principle, additive -- a
hidden-state probe evaluated on a contaminated benchmark, using a leaky
checkpoint-selection procedure, could have its reported AUROC inflated by
both mechanisms simultaneously, and neither literature currently accounts
for the other. We do not audit any of our four case studies' underlying
benchmarks for pretraining contamination; that remains a separate,
unaddressed question, orthogonal to the leakage mechanisms this paper
documents.

To our knowledge, no prior paper does both of the following: (a) name and
distinguish multiple, structurally different label-leakage mechanisms
specific to hidden-state probing pipelines; (b) quantify their relative
severity via controlled reconstruction, rather than assuming all leakage
is equally damaging.

## 3. Method: How These Case Studies Were Selected and Verified

This section states our selection criteria for all four case studies.
Two case studies (Sections 4.1-4.2) come from our own prior work, where
we had full access to code, data, and git/commit history. The external
case studies (Sections 4.3-4.4) required three things. First, a real,
runnable public code repository -- not merely "available upon request"
-- verified directly via the repository host, not taken on faith from the
paper's text. Second, a reported AUROC in the range that warrants
scrutiny: 0.90 or higher. Third, an evaluation protocol whose leakage
risk we could assess by reading the actual training/evaluation code, not
just the paper's prose methodology section. We deliberately did not
limit the search to K-fold cross-validation setups specifically, since
one candidate paper's bug (Section 4.4) lives in a single train/val/test
split with no K-fold structure at all -- a useful reminder that "avoid
K-fold" is not itself a fix for this family of hazards.

## 4. Four Case Studies

**Evidentiary status.** These four case
studies are not equally verifiable by a reader of this
repository. Case
Studies 3 (MultiHaluDet) and 4 (quantized-LLM paper) are external,
published, code-available systems: their claimed leaks are demonstrated
against pinned commits of publicly released third-party code, and this
paper's own audit scripts and results JSON are included here in full.
Case Studies 1 (HaRP) and 2 (GUARDIAN) are this paper's own prior,
unpublished pipelines -- built by the authors in the course of other
work and only retroactively recognized as carrying these leakage
patterns (\S1). Their severity numbers (+0.19 AUROC; 18.8 AUROC points)
are traceable to the exact
training scripts and logs that produced them
(\texttt{experiments/09\_train\_estimator\_v2.py},
\texttt{src/authorization/failure\_estimator.py}, and related files, for
HaRP; \texttt{experiments/02\_layer\_sweep.py} and related files for
GUARDIAN), released as
two public repositories (see Data and Code Availability,
\S6), so a reader can independently verify all four case studies
end to end from what is shipped, not just Case Studies 3-4. Neither
HaRP nor GUARDIAN was run through the automated scanner in \S5, since
both were built and their leaks discovered before that scanner existed
-- this is a scope note, not an evidentiary gap.

### 4.1 Case Study 1 (severe): Full-dataset fit-then-score leakage — HaRP

[See `draft/worked_examples.md`, Case Study 1, for full detail.]

HaRP's Group-C representation feature `probe_conf` came from a
logistic-regression probe fit on all 700 samples. That probe then scored
those same 700 samples. Only afterward did this feature enter a separate
outer 5-fold cross-validation loop for the downstream failure estimator.
The result is nested-CV leakage in its cleanest form: the "held-out"
fold's feature was manufactured by a model that had already seen every
label, including that fold's own.

**Quantified effect.** This measures the gap between the leaked and
fixed evaluation protocols on the same feature set. A+B+C AUROC under
5-fold CV: 0.9620 (leaked) versus 0.7714 (fixed, out-of-fold) -- a
**+0.1906** inflation. `probe_conf` alone shows the same pattern:
in-sample AUROC 1.0000 versus proper out-of-fold AUROC 0.7573. We
independently cross-checked this against a separately-obtained
cross-validated probe AUROC of 0.7583. The two agree to within 0.001 --
two different measurement routes for the same underlying quantity
converge, strong evidence the fix is correct.

**Fix:** refit both the probe and the class centroids per fold. Use only
that fold's training indices. Score only on the held-out indices.

### 4.2 Case Study 2 (severity: 18.8 AUROC points): CV-based layer-selection optimism — GUARDIAN

[See `draft/worked_examples.md`, Case Study 2, for full detail.]

GUARDIAN selects its detection layer -- L11 of 32, on Mistral 7B -- by
taking the argmax of a 5-fold cross-validated AUROC across all layers.
It then reports that same CV number as the model's performance.

**Quantified effect.** This compares the reported CV-selection number
against a genuinely held-out estimate at the same layer. CV-selection
AUROC at L11: 0.804. True held-out AUROC at L11: **0.616** -- an
18.8-point gap. We trained the held-out probe on the first 400 samples
and tested it on a separate, never-selection-touched 300. A second,
independent test shows the same fragility: which layer counts as
"optimal" depends on the selection rule. Argmax train-split CV picks
L11; argmax all-data AUROC picks L19 instead.

**Fix:** after CV-based selection, evaluate exactly once more. Use a
disjoint held-out set that played no role in the selection. Report that
number.

### 4.3 Case Study 3 (severity: structurally real, empirically modest at tested scale): Per-fold checkpoint-selection leakage — MultiHaluDet

[See `draft/case_study_multihaludet.md` for full detail, including
literal code excerpts from the public repository. **This subsection
states the methodology and result directly. Appendix A documents the
calibration and confound-detection process for this case study in
detail, including the specific ways each control (below) can fail
independently -- supporting evidence for the checklist in \S5. It is
not required to verify the result below.]**

MultiHaluDet (arXiv 2605.24919, verified title "MultiHaluDet: Multilingual
Hallucination Detection via LLM Hidden State Probing"; repo:
github.com/alvi-uiu/MultiHaluDet, pinned at commit
`c7597518`, 2026-07-02) reports up to 98.55% AUROC detecting
hallucination from Mistral-7B/LLaMA-2-7B hidden states. We read its
released training code directly: `run_pipeline.py::stage_2_3_train_oof`
and `src/training/trainer.py::train_deep_model_fold`. Inside each of 5
inner CV folds, a deep model trains via gradient descent on the fold's
training split. The checkpoint kept, across ~45 epochs, is whichever one
maximizes AUROC on that fold's *validation* split. That same split's
features are then stored as "out-of-fold" and fed, unmodified, to a
downstream meta-learner ensemble.

This is a genuine leakage channel. The checkpoint choice is a function of
the validation fold's own labels. It is structurally distinct from Case
Study 1, though: the model's *weights* are never trained via gradient
descent on the validation fold. Only the *choice of which epoch's weights
to keep* depends on it.

**Quantifying it, final methodology.** Re-running MultiHaluDet's actual
7B-scale pipeline was outside this paper's compute budget. Instead we
built a controlled synthetic reconstruction. It mirrors their training
loop exactly: same 5-fold structure, same best-val-AUC-checkpoint logic,
same downstream-classifier-on-OOF-features structure. We calibrate this
reconstruction's task difficulty to a realistic, non-saturated ceiling --
a Bayes-optimal-linear AUROC of $0.80$. The following identity converts a
signal-detection separation statistic into that AUROC ceiling:
$\text{AUROC}=\Phi(\sqrt{J/2})$, where $\Phi$ is the standard normal CDF
and $J$ is the Mahalanobis-style separation ($J=1.417$; a classical
signal-detection-theory result, Simpson \& Fitter 1973). The realized
classifier AUROC sits below this ceiling by construction, since a finite,
regularized MLP on OOF features does not reach the Bayes-optimal bound.
Observed LEAKY/CLEAN means fall in the range 0.7474-0.7647. So
"calibrated to 0.80" means the task's ceiling is 0.80 -- not that the
classifiers solve an 0.80-difficulty task; realized difficulty is closer
to $\approx$0.75.

We run four conditions at $n=100$ seeds, across four capacities
(16/48/128/384 hidden units). **LEAKY**: checkpoint selected on the
reused validation fold. **CLEAN**: checkpoint selected on a disjoint
carve-out, but trained on $\approx$15\% less data -- a budget confound.
**CLEAN\_MATCHED**: same disjoint-carve-out selection, but retrained from
scratch on the full training budget, for exactly that many epochs, using
the same random seed as LEAKY -- this isolates "does the epoch count
come from a peeked-at fold" from "does the model see less data."
**PLACEBO**: checkpoint selected on permuted labels -- this gives zero
*selection*-signal, not zero signal overall, since training itself still
uses the real labels, which is why placebo AUROC sits at 0.7272-0.7375,
not 0.5.

**Result: full capacity sweep, corrected calibration, all four
conditions.** The table below reports mean AUROC per condition at each
capacity, plus the two pairwise gaps and their p-values.

| Hidden | LEAKY | CLEAN | CLEAN_MATCHED | PLACEBO | LEAKY$-$CLEAN (p) [retracted comparison, confounded] | LEAKY$-$CLEAN_MATCHED (p) |
|---|---|---|---|---|---|---|
| 16  | 0.7611 | 0.7567 | 0.7602 | 0.7272 | +0.0044 (0.0019) | +0.0009 (0.309) |
| 48  | 0.7647 | 0.7617 | 0.7631 | 0.7375 | +0.0030 (0.0058) | **+0.0017 (0.015)** |
| 128 | 0.7604 | 0.7555 | 0.7571 | 0.7309 | +0.0049 (0.0008) | **+0.0034 (0.008)** |
| 384 | 0.7533 | 0.7474 | 0.7506 | 0.7350 | +0.0059 (0.0012) | +0.0027 (0.077) |

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/capacity-sweep.pdf}
\caption{Mean AUROC for the four conditions (LEAKY, CLEAN, CLEAN\_MATCHED, PLACEBO) across all four tested capacities, visualizing the table above. LEAKY, CLEAN, and CLEAN\_MATCHED cluster tightly together at every capacity, all clearly above PLACEBO -- confirming that honest checkpoint selection is itself strongly informative -- while the gap that constitutes the actual leakage severity (LEAKY vs.\ CLEAN\_MATCHED) is visually small relative to this overall separation, consistent with the severity estimate of +0.0009 to +0.0034 AUROC reported in the text.}
\label{fig:capacity-sweep}
\end{figure}

(AUROC columns are means across 100 seeds. Both CLEAN and CLEAN_MATCHED
beat PLACEBO decisively at every capacity, p$<$0.0001 throughout --
CLEAN_MATCHED$-$PLACEBO gaps range +0.0156 to +0.0330. This confirms
honest label-based checkpoint selection is itself strongly informative.
The LEAKY$-$CLEAN column is the original, budget-confounded comparison;
we retain it only for transparency and do not treat its significance as
evidence of anything beyond "a budget-confounded gap exists," per the
correction below.)

**Significance, corrected for multiple comparisons and power.** This
tests whether the budget-matched LEAKY$-$CLEAN\_MATCHED gap survives
correction for testing four capacities at once. The gap is significant
at 2 of 4 capacities: 48 units, $p=0.015$; 128 units, $p=0.008$. It is
not significant at 16 units ($p=0.309$, gap $+0.0009$, near zero). It is
borderline at 384 units ($p=0.077$, gap $+0.0027$).

We apply Holm-Bonferroni correction across the four p-values, to control
the false-positive rate from testing four capacities. Sorted ascending,
the four p-values are 0.0075, 0.0151, 0.0772, 0.3089, tested against
thresholds $\alpha/4, \alpha/3, \alpha/2, \alpha$ in turn. 128 units and
48 units both survive; 384 and 16 units do not.

**Scope of this correction, stated plainly.** This Holm-Bonferroni
correction is applied within the four-capacity sweep only. It is not
applied jointly across this sweep, the separate real-feature check
(\S4.3), the FP16-vs-AWQ quantization control, and the CLEAN\_MATCHED-
vs-PLACEBO anomaly test elsewhere in this paper -- a reader applying a
single family-wise correction across all of this paper's reported
p-values, rather than treating each analysis as its own family, would
be applying a more conservative standard than we do here. We do not
claim our within-analysis correction choices are the only defensible
ones.

We also compute a minimum detectable effect (MDE): the smallest true gap
this design could reliably detect at 80\% power and two-sided
$\alpha=0.05$. This requires the per-seed gap standard deviation for the
budget-matched LEAKY$-$CLEAN\_MATCHED comparison. These stds are
$0.0071$-$0.0135$ across capacities (distinct from the larger,
budget-confounded LEAKY$-$CLEAN column's stds, which reach $0.017$ at
384 units). The resulting MDE is roughly
$0.002$-$0.004$ AUROC. The 384-unit borderline result sits almost
exactly at this capacity's MDE ($\approx 0.0038$); we read it as
underpowered rather than null. The 16-unit null (gap $+0.0009$, MDE
$\approx 0.0026$) is genuinely smaller than this design could detect,
supporting a real null there.

This result is neither "confirmed real and capacity-growing" nor
"possibly entirely a training-budget artifact." It is a small,
capacity-dependent residual effect. It is significant, surviving
multiplicity correction, at two of four capacities. One of the remaining
two is better described as underpowered than null.

This decomposes the budget-confounded LEAKY$-$CLEAN gap into a
budget-mismatch component and a residual. At 16 units, budget-matching
removes 80\% of the point estimate (residual $+0.0009$ of $+0.0044$). At
48, 128, and 384 units, it removes only 43\%, 31\%, and 54\%
respectively. That leaves a majority-or-near-majority residual at three
of four capacities, significant at two of those three. Budget mismatch
explains all of the apparent effect at the smallest capacity tested; it
explains only a minority of it at the other three. The best-supported
reading is a real, small, inconsistently-detectable residual leak, not
its absence.

**Our own pre-registered decision rule** classifies each capacity as
`GENUINE_LEAK_CONFIRMED`, `CONFOUND_CONFIRMED_NO_REAL_LEAK`, or `MIXED`,
from the placebo-relative gaps (`02d_....py`). It returns \texttt{MIXED}
at all four capacities. This rule was itself a flawed pre-registration,
not a rigor credential: its \texttt{CONFOUND\_CONFIRMED\_NO\_REAL\_LEAK}
branch requires $\text{leaky\_minus\_placebo}\,p>0.05$, but that
comparison is significant at $p<10^{-8}$ at every capacity, in every
version of this experiment. The branch is structurally unreachable, so
the rule can only ever return \texttt{GENUINE\_LEAK\_CONFIRMED} or
\texttt{MIXED} -- never a clean "no leak" verdict. \texttt{MIXED} is the
honest label for what the data show: significant at 2/4 capacities.

**Bottom line.** The checkpoint-selection-leakage mechanism itself
remains real and code-verified in MultiHaluDet's released pipeline: the
checkpoint choice is unambiguously a function of the reused fold's own
labels. Our synthetic reconstruction's quantitative severity estimate,
once genuinely budget-matched, is small ($+0.0009$ to $+0.0034$ AUROC)
and inconsistently significant across capacity -- real evidence of a
modest effect at two of four capacities, not a clean confirmation or a
clean null. We still cannot claim this settles MultiHaluDet's own
reported 98.55\% AUROC's exact inflation: their real architecture (6
transformer layers, multi-scale branches, heavy augmentation, longer
training) is considerably more expressive than even our 384-unit MLP
reconstruction, so no severity estimate here should be read as a settled
answer for their pipeline. **Getting this number right required
simultaneously correct handling of statistical power, calibration,
training-budget matching, and seed matching (full detail in Appendix
A); this is itself the paper's strongest argument for
the checklist in \S5.**

**A real-hidden-state validation of the
synthetic severity estimate.** This closes an open question rather than
leaving it open: does the synthetic estimate hold on real geometry, not
just a Gaussian toy? We extracted real features from MultiHaluDet's own,
publicly-released feature-extraction code (`extract_features`,
`src/data/feature_extractor.py`). We copied this code with only cosmetic
changes: a bare `except:` narrowed to `except Exception:`, and stripped
type hints and formatting. No logic was altered; the code
is already model-agnostic, so no functional adaptation was needed. We
used their own default model (`mistral-7b`) and their own data loader's
exact HuggingFace fallback (`pminervini/HaluEval`, config `qa_samples`),
on $n=400$ real samples. For Kaggle GPU tractability, we used
TheBloke/Mistral-7B-Instruct-v0.2-AWQ, a 4-bit quantized checkpoint.

Two substitutions here are disclosed tractability compromises. Both the dataset and the model checkpoint are
pinned to a specific HuggingFace revision hash in our code
(`code/03_real_feature_leakage_test.py`: model revision
`f970a2bb8`, dataset revision `12a856119`). The AWQ checkpoint remains a
third-party requantization of MultiHaluDet's actual default
(`mistralai/Mistral-7B-Instruct-v0.2`), not the original full-precision
weights -- that substitution is disclosed, not fixed by pinning.

We did not attempt to reproduce MultiHaluDet's full 6-layer multi-scale
transformer classifier, with its mixup/cutmix/EMA/SWA/contrastive/focal-loss
training. Faithfully replicating an unfamiliar, complex training loop is
a substantially larger undertaking, with real risk of subtle integration
bugs; it remains future work. Instead, we fed these real extracted
features -- a $400\times414$ matrix: 32 sampled layers $\times$ 12
per-layer statistics, flattened, plus 30 global logit/norm-trajectory
features, exactly as their code produces -- into this paper's own
already-validated LEAKY/CLEAN/CLEAN\_MATCHED/PLACEBO test. This test uses
identical architecture and procedure to the synthetic sweep above, at
capacity 128. Only the
synthetic-Gaussian data-generation step changed
(`code/03_real_feature_leakage_test.py`). We re-run at
$n=100$ seeds, matching the synthetic
sweep's own seed count, per the power recommendation below -- both HuggingFace sources (model
`TheBloke/Mistral-7B-Instruct-v0.2-AWQ`, dataset `pminervini/HaluEval`)
are pinned to a specific revision hash in code. All numbers below are
from this $n=100$ run.

**Result: the real-feature severity estimate lands almost exactly inside
the synthetic reconstruction's previously-estimated range.** On real
features, LEAKY (mean AUROC $0.9832$) beats CLEAN\_MATCHED (mean AUROC
$0.9806$) by $+0.0026$ AUROC. This is significant at $p=0.0012$
(Wilcoxon signed-rank on the 100 paired seed-level gaps) -- if anything
more significant than the $n=50$ run's $p=0.0175$, at an essentially
identical effect size ($+0.0026$ vs.\ the earlier $+0.0028$). It falls
inside the $+0.0009$-to-$+0.0034$ range the synthetic sweep estimated
across capacities.

**Capacity 128 is not this paper's flagship configuration.** \S4.3's
own synthetic sweep (\texttt{code/02c\_placebo\_and\_power\_check.py},
\texttt{02d\_corrected\_capacity\_placebo\_sweep.py}) designates capacity
384 as the flagship, since 384 is the one capacity matching
MultiHaluDet's actual \texttt{hidden\_dim=384}, and it is also the one
synthetic capacity that does not survive Holm-Bonferroni correction
($p=0.077$, Table above). Real-feature validation at capacity 128 was
therefore run at a capacity that already survived correction in the
synthetic sweep, not at the architecturally-matched one -- a legitimate
capacity-shopping concern, addressed by re-running the identical
real-feature test at capacity 384 (same cached Mistral-7B features, no
new model inference, $n=100$ seeds:
\texttt{results/real\_feature\_leakage\_test\_result\_capacity384.json}).
\textbf{The result replicates: LEAKY beats
CLEAN\_MATCHED by $+0.0021$ AUROC at capacity 384 ($p=0.0010$), against
$+0.0026$ ($p=0.0012$) at capacity 128} -- same direction, same order of
magnitude, both significant. The CLEAN\_MATCHED-vs-PLACEBO sanity-check
anomaly (\S4.3 below) also replicates at capacity 384
(CLEAN\_MATCHED underperforms PLACEBO by $-0.0015$, $p=0.045$, versus
$-0.0025$, $p=0.0115$ at capacity 128). Capacity 128, not the architecturally-matched
flagship capacity 384, was used for the initial real-feature run;
re-running at capacity 384 does not change this paper's substantive
conclusion in either direction.

This is a genuinely different task regime than the calibrated synthetic
one, though, and the difference is itself informative. Real Mistral-7B
HaluEval features are separable enough that even the \texttt{PLACEBO}
condition -- checkpoint selection on \emph{permuted} labels -- reaches
mean AUROC $0.9831$, statistically indistinguishable from LEAKY
($+0.0001$, $p=0.918$). On real representational geometry this
saturated, \emph{almost any} reasonable checkpoint-selection rule, even a
random one, lands on a good-performing model, so the LEAKY-vs-PLACEBO
comparison that is decisive on our calibrated-to-0.80 synthetic task is
uninformative here.

The LEAKY-vs-CLEAN\_MATCHED comparison remains the correct, decisive
test: it isolates whether the checkpoint is chosen from the reused fold
specifically, independent of overall task difficulty. It is where the
small, real, statistically significant residual shows up.

**The real-feature run's own sanity check fails.**
This checks a foundational assumption before
trusting the headline gap: does honest checkpoint selection actually
beat permuted-label selection on this data? The synthetic sweep's
entire validity argument rests on CLEAN and CLEAN\_MATCHED (honest,
label-based checkpoint selection) clearly beating PLACEBO (checkpoint
selection on permuted labels) -- that gap is what establishes "the
checkpoint chosen matters" as a real, measurable effect on this task,
before asking whether reusing the same fold on top of it adds anything.

On the real 400-sample Mistral features, that foundational gap does not
hold, at either seed count, and is not explained by sampling noise. At $n=50$, CLEAN\_MATCHED underperformed
PLACEBO by $-0.0027$ AUROC ($p=0.0497$), and plain CLEAN
underperformed PLACEBO by a nearly identical $-0.0027$ ($p=0.051$). At $n=100$, the same anomaly
persists at essentially the same effect size -- CLEAN\_MATCHED still
underperforms PLACEBO, now by $-0.0025$ AUROC, with significance
sharpening rather than resolving: $p=0.0115$ for CLEAN\_MATCHED
vs.\ PLACEBO, $p=0.0064$ for CLEAN vs.\ PLACEBO -- both
comfortably below $\alpha=0.05$. Doubling the seed
count rules out $n=50$ sampling noise as the explanation.

We still do not think this means checkpoint-selection leakage is not
real on real data -- the LEAKY-vs-CLEAN\_MATCHED gap (below) is a
different comparison and survives independently.

**The anomaly has a confirmed mechanistic
account: an epoch-count mismatch.** CLEAN\_MATCHED retrains on the *full*
training fold (identical size to PLACEBO and LEAKY) for a *fixed* epoch
count inherited from CLEAN's early-stopping run on the smaller,
disjoint fold. We instrumented both conditions' internally
selected epoch counts across all 100 seeds
(`code/19_real_feature_leakage_diagnostics.py`,
`results/real_feature_leakage_diagnostics.json`): CLEAN's early-stopped
checkpoint averages epoch $12.18$; PLACEBO's checkpoint-selection --
even against permuted, meaningless validation labels -- averages epoch
$17.7$, a significant $5.5$-epoch gap (Wilcoxon $p=0.012$). This makes
sense mechanistically: "best checkpoint against permuted labels" has no
real stopping signal to act on, so it drifts toward later epochs (more
fitting to $X$'s real structure, independent of the permutation),
while CLEAN's genuine early-stopping, on a smaller and noisier held-out
slice, tends to stop sooner. Critically, this epoch gap predicts the
AUROC anomaly directly: the per-seed correlation between
(PLACEBO epoch $-$ CLEAN epoch) and (CLEAN\_MATCHED AUROC $-$ PLACEBO
AUROC) is $r=-0.594$ ($p=7.5\times10^{-11}$) -- the more epochs
CLEAN\_MATCHED's fixed count under-trains relative to what PLACEBO's
own selection lands on, the more CLEAN\_MATCHED underperforms. The
sanity-check anomaly is an epoch-count-matching artifact of the
CLEAN\_MATCHED construction, not evidence against the checkpoint-reuse
severity estimate itself.

**Power check, exact rather than back-calculated.** The diagnostic
rerun above also saved the full 100-seed per-seed gap arrays.
Exact per-seed gap SD for LEAKY-vs-CLEAN\_MATCHED is $0.008294$,
giving an 80\%-power MDE (two-sided $\alpha=0.05$) of $0.00232$ --
essentially identical to the back-of-envelope estimate
($\approx0.0023$) used earlier in this section. The observed gap
($0.00261$) sits comfortably above this exact MDE. For
CLEAN\_MATCHED-vs-PLACEBO, exact SD $=0.009420$, MDE $=0.00264$: the
observed gap ($-0.00248$) sits just under this MDE, consistent with the
anomaly being a real, if modest, effect rather than one certain to be
detected at this $n$ -- fully consistent with the epoch-count mechanism
identified above, which does not require the anomaly to be large, only
systematic.

**The LEAKY-vs-CLEAN\_MATCHED
result is not weak or inconclusive, and the sanity-check anomaly is not
an open, unresolved puzzle: it has a confirmed epoch-count
mechanism (above).** The
$n=100$ replication directly tests, and rejects, the hypothesis that $n=50$ was
underpowered for the LEAKY-vs-CLEAN\_MATCHED comparison ($p=0.0012$,
above the exact MDE). The separate CLEAN/CLEAN\_MATCHED-
vs-PLACEBO anomaly is fully explained: it is a direct consequence
of CLEAN\_MATCHED's fixed epoch count under-training relative to what
PLACEBO's own checkpoint-selection lands on, confirmed by a strong,
highly significant correlation ($r=-0.594$, $p=7.5\times10^{-11}$)
between the epoch gap and the AUROC gap, per seed. This is a
construction artifact of the CLEAN\_MATCHED control specifically, not
a challenge to the checkpoint-reuse severity estimate LEAKY-vs-
CLEAN\_MATCHED itself measures.

**A per-seed correlation does not establish a
confirmed causal mechanism.** $r=-0.594$ ($p=7.5\times10^{-11}$)
is a strong association across 100 seeds, not an intervention. We did
not force a range of fixed epoch counts on CLEAN\_MATCHED and observe
AUROC track it directly -- the one manipulation that would license the
word "confirmed" in a causal sense. We also note a relevant structural
detail: LEAKY, CLEAN\_MATCHED, and PLACEBO all call
\texttt{torch.manual\_seed(fold\_seed)} with the identical \texttt{fold\_seed}
and train on identical data (\texttt{X\_train[tr\_idx]}) with full-batch
gradient descent, so LEAKY and PLACEBO's underlying weight
trajectories are checkpoint-for-checkpoint identical -- the two
conditions differ only in \emph{which} epoch along that shared
trajectory gets selected (real vs.\ permuted validation labels), not in
what was trained. This is a deliberate, confound-minimizing design, not
a bug, but it means the epoch-gap-predicts-AUROC-gap correlation is
partly a structural consequence of comparing two selection rules
applied to one shared training curve, not two independently-trained
models. The epoch-count account is
strongly evidenced and consistent with every diagnostic we ran, but it
is a correlational account, not a confirmed causal mechanism in the
interventional sense; we did not run the epoch-forcing experiment
that would establish one.

**The AWQ-vs-full-precision substitution is bounded.** We extracted the
identical features on the same first 50 HaluEval samples using the full-
precision checkpoint MultiHaluDet's own pipeline actually defaults to
(\texttt{mistralai/Mistral-7B-Instruct-v0.2}, FP16) and compared 5-fold-CV
logistic-regression AUROC against the corresponding 50 rows of the
cached AWQ-extracted features
(`kaggle\_kernels/paper2-fp16-vs-awq-control/run\_fp16\_vs\_awq\_control.py`,
`results/fp16\_vs\_awq\_control.json`). AUROC is identical to four
decimal places: $0.9600$ both, a difference of exactly $0.0$. At $n=50$
this is not a claim that quantization has zero effect at every possible
$n$ or capacity, but it directly rules out 4-bit quantization as a
driver of the near-saturated regime this section's sanity check
operates in -- the AWQ substitution is not the source of the anomaly
either. It does
not settle MultiHaluDet's exact 98.55\% AUROC inflation: their full
classifier, 10$\times$ our sample size, and their exact training regime
remain untested.

**A single 5-fold-CV logistic-regression
point estimate at $n=50$ is weaker rigor than every other comparison in
this paper.** We reran the identical FP16-vs-AWQ comparison through this
paper's own capacity-128 SweepMLP/\texttt{train\_to\_best\_checkpoint}
architecture -- the same one the headline severity claim uses -- with
100 seed-resampled train/validation/test splits rather than one 5-fold
point estimate (reusing both already-cached feature sets, no new model
inference). The result: mean AUROC $0.946$ (FP16) vs.\ $0.939$ (AWQ),
gap $+0.0068$, 95\% CI $[-0.061, +0.080]$ (includes zero), Wilcoxon
$p=0.148$ -- not significant, and consistent with the original $n=50$
point estimate's conclusion, now with an effect-size distribution and a
significance test rather than a single number
(`results/fp16\_vs\_awq\_control\_matched\_architecture.json`).

### 4.4 Case Study 4 (secondary; severity unquantified, mechanism confirmed): Test-set-driven best-layer selection — quantized-LLM paper

[See `draft/case_study_quantized_llm_paper.md` for full detail, including
literal code excerpts.]

"Hallucination Is Linearly Decodable from Mid-Layer Hidden States in
Quantized LLMs" (arXiv 2606.02628, verified title matches, author
Aizierjiang Aiersilan; repo:
github.com/Ezharjan/HallucinationPatternDetection, pinned at commit
`ea0b9678`, 2026-06-12) reports AUROC up to 1.000. Its per-layer probe
training (`src/detection/probes.py::train_probe`) uses a clean single
70/10/20 train/val/test split -- genuinely free of Case Studies 1-3's
exact nested-CV patterns. But `src/detection/saplma.py::saplma_probe_per_layer`
selects a `best_layer` differently: it takes the argmax, over ~30
layers, of each layer's **test-set** AUROC (`ProbeMetrics.auroc`,
computed on `X_te`/`y_te`). It then reports that same test-set number as
the headline result, with no correction for having tried ~30 hypotheses
against the one test set.

This is not nested-CV leakage. It is the classical "winner's curse" of
multiple-hypothesis selection using the exact metric you then report. We
did not re-quantify this mechanism's severity -- doing so requires
re-extracting hidden states from 7B-scale models, outside this paper's
budget -- but flag it as a fourth, structurally distinct pattern worth
watching for in this literature independently of the other three.
**We confirmed this is not a data
availability issue we could have closed cheaply.** Their public repo
ships only summary result tables (\texttt{results/tables/*.csv}), not
the underlying per-layer hidden-state arrays \texttt{saplma\_probe\_per\_layer}
consumes -- a severity re-quantification would need their exact
quantized checkpoint and dataset re-run from scratch, not a quick
reuse of already-extracted embeddings, confirming the budget
constraint above rather than a gap we simply had not checked.

## 5. A Checklist for This Literature

[Full version in `draft/leakage_checklist.md`.] These four questions
correspond to the four mechanisms above:

1. Is any feature computed by fitting something on the full dataset's
   labels before an outer CV loop treats part of that dataset as held out?
2. Inside a CV fold, is a "best" checkpoint/epoch chosen using the same
   fold whose features/predictions get reused downstream as clean OOF
   output?
3. Is a "best" layer/config selected by the argmax of a metric computed
   on the same test set that gets reported as the headline number, across
   many candidates, with no correction?
4. Was a hyperparameter chosen by maximizing a cross-validated metric,
   with that same CV number then reported as performance, with no further
   genuinely held-out check?

**We tried to automate this checklist, and the
attempt itself is the useful result.** We built a regex-based scanner
for the four patterns above and ran it against 7 real, external
hidden-state-probing repositories found via targeted search
(\texttt{haloscope}, \texttt{semantic-entropy-probes},
\texttt{hallucination\_probes}, \texttt{DetectLLMHallucination},
\texttt{halluscope}, plus the two case-study repos already audited by
hand, MultiHaluDet and HallucinationPatternDetection) --
\texttt{code/04\_leakage\_linter.py},
\texttt{results/leakage\_linter\_report.json}. We state the scope
honestly before any result: this is a heuristic line-matching tool, not
a validated static analyzer, and every flag was manually read and
classified by us before being treated as a finding -- we do not report
the scanner's own precision/recall as if it were established.

\textbf{The scanner produced 7 raw regex hits across the 7 repos,
concentrated in exactly 2 repos (both outside the already-known case
studies), and manual reading confirmed every one of the 7 is a false
positive.} 3 hits flag \texttt{HallucinationPatternDetection}'s
\texttt{embed\_viz.py} for a "fit-then-score, no split" pattern --
reading the code shows all three are t-SNE/UMAP/PCA calls used purely
for visualization plotting, not a detection-metric-relevant model fit
at all. The other 4 hits flag \texttt{haloscope}'s
\texttt{hal\_det\_llama.py} and its near-duplicate
\texttt{hal\_det\_opt.py} (2 hits each, one per model variant) for
"test-set-driven argmax" -- reading the code (and its own inline
comment, "get the best hyper-parameters on validation set") shows layer/
threshold selection is correctly performed on a separate eval/validation
split (\texttt{embed\_generated\_eval}, \texttt{gt\_label\_val}), with
the selected layer only \emph{applied} to \texttt{embed\_generated\_test}
afterward for genuine held-out scoring -- the opposite of leakage, and,
incidentally, a positive data point that a real, peer-reviewed
NeurIPS 2024 pipeline gets this specific split discipline right.

\textbf{More tellingly, the same scanner MISSED both instances we know
are true positives in this exact corpus.} Case Study 4's actual bug
(\S4.4, \texttt{saplma.py}'s \texttt{best\_layer} selection) is a plain
\texttt{for} loop with a manual \texttt{if v > best\_auc} update, using
neither the literal token "argmax" nor a variable name containing
"test" at the point of selection -- the leakage is one function call
removed, inside \texttt{train\_layerwise\_probes}, invisible to a
line-local heuristic. Case Study 3's actual bug (\S4.3, MultiHaluDet's
per-fold checkpoint reuse) is missed for the identical structural
reason from the opposite direction: the P3 detector's keyword list
(\texttt{best\_epoch}/\texttt{checkpoint}) never matches, because the
real code selects on \texttt{best\_auc}/\texttt{best\_model} instead
-- confirmed by grepping the pinned MultiHaluDet commit directly for
\texttt{checkpoint}/\texttt{best\_epoch} (one unrelated hit, in
\texttt{.gitignore}) and by \texttt{results/leakage\_linter\_report.json}
itself, whose \texttt{MultiHaluDet} entry shows \texttt{any\_flag:
false} despite this being a repo with a confirmed, manually-audited
leak. \textbf{Zero true positives found in the 5 newly-scanned external
repos, two demonstrated false negatives on the two known true
positives in this corpus, and all 7 raw hits (across 2 flagged repos)
confirmed as false positives}: at this
level of tooling, an automated scanner is not a reliable substitute for
the manual, one-repo-at-a-time reading this paper's four case studies
required, and we do not claim otherwise. \textbf{This is a prevalence
statement about 2 of 9 repos examined in this paper, not a random or
representative sample of hidden-state-probing code in general.}
\textbf{Not all 9 were found the same way.} Two of the four
case-study pipelines (MultiHaluDet, HallucinationPatternDetection) plus
the 5 additionally scanned here -- 7 of 9 -- were found via targeted
search for exactly this failure class. The other two case studies
(HaRP, GUARDIAN) are this paper's own pipelines, built by the authors
in the course of other work and only retroactively recognized as
carrying these leakage patterns (see \S1's origin story); they were not
independently discovered by search and, unlike the other 7, were never
run through the automated scanner in \S5 at all. The targeted-search
framing applies to 7 of the 9 repos this paper examines, not all of
them, which selects for repos already suspected or confirmed to be
relevant -- the true population-level prevalence of any of the four leakage
patterns remains unknown and is not estimated by this count. We report the attempt because
a negative result about \emph{how hard this is to automate} is itself
informative for anyone tempted to build a "leakage linter" as a
lighter-weight alternative to manual audit: the same variable-naming
and control-flow subtlety that makes leakage easy to introduce
by accident (a checkpoint selected on the "eval" split by name, applied
correctly to "test") also makes it resist simple pattern-matching
detection, in both directions -- our tool flagged the innocent cases
and missed the guilty one.

## 6. Discussion

**Data and code availability.** All code,
cached result JSONs, and the paper source are publicly available at
`https://github.com/Lakshmi-Chakradhar-Vijayarao/leakage-audit`.
Case Studies 3
(MultiHaluDet) and 4 (quantized-LLM paper) are fully reproducible from
what ships with this paper: pinned third-party commits, our audit
scripts, and the resulting JSON logs are all included. Case Studies 1
(HaRP) and 2 (GUARDIAN) are now also independently reproducible: the
relevant training scripts and result logs behind their severity numbers
(+0.19 AUROC; 18.8 AUROC points) are released as two additional public
repositories,
`https://github.com/Lakshmi-Chakradhar-Vijayarao/harp-leakage-case-study`
and
`https://github.com/Lakshmi-Chakradhar-Vijayarao/guardian-leakage-case-study`
respectively -- each containing only the subset of the original project
relevant to the specific leakage finding this paper reports, not the
full original codebase. Neither pipeline was ever run through the §5
scanner, since both were built and their leaks discovered before that
scanner existed (see §4's evidentiary-status note). The regex linter referenced in §5 is included
in full and is reproducible against the shipped case studies; it is
also, per §5's own results, a confirmed non-detector on this corpus
(0 true positives, 2 false negatives on the corpus's own known bugs),
so its inclusion is for transparency and future improvement, not as a
working tool a reader should rely on today.

**Scope note: this paper makes no inference-economy claim.** It is an
evaluation-protocol audit, not an inference-time method. There is no
generation loop, no routing or early-exit mechanism. Fixing any of the
four leakage mechanisms makes no claimed difference to deployment cost.
The paper's value is entirely in correcting reported detection metrics,
not in reducing compute.

**An inflated AUROC is not a purely academic
problem -- it is a downstream compute-and-labor cost, even though we do
not quantify that cost here.** A hallucination detector deployed at a
reported AUROC that is actually several points lower (Case Study 2's
18.8-point CV-vs-held-out gap; Case Study 1's +0.19 full-dataset-leak
inflation) will, at a fixed operating threshold, pass more hallucinated
outputs through to production and flag more correct outputs for
unnecessary human review than its reported number promises. Both
failure directions cost real resources downstream: unflagged
hallucinations that reach an end user, and flagged-but-correct outputs
that consume human-review time or a second, more expensive verification
pass. We do not have deployment-volume, review-cost, or incident-cost
data for any of the four audited systems, so we do not attempt a dollar
estimate here -- doing so honestly would require production telemetry
we do not have access to. The scoped claim is narrower and does not
need that data: any severity number this paper (or a similarly-audited
system) reports translates directly into a compute-and-labor-cost
multiplier at whatever deployment volume and review cost a practitioner
actually has, which is exactly the translation a deployment decision
requires and a purely academic AUROC comparison does not surface on its
own.

**This audit's central lesson recurs, independently, across three
companion studies from the same project.** A geometric-certificate paper
finds a near-saturated forced-choice hallucination benchmark is
substantially explained by trivial answer-length features, not
truthfulness. An FFN-mechanism paper finds a cross-architecture component
comparison reverses entirely once an instruction-tuned model is queried
with its correct chat template instead of a bare prompt. An
agentic-failure taxonomy finds a hidden-state "early-warning" signal is
mostly-to-entirely a difficulty readout, once the confound is directly
tested rather than assumed absent. None of these three studies shares a
leakage mechanism with any of the four case studies here. But all four
studies share the same discipline this paper argues for: a claimed
severity or signal is not established until the specific confounds that
could produce it for free have been named and tested, not merely
acknowledged as theoretically possible.

**Leakage severity is not uniform, is not always a fixed property of a
mechanism, and the field currently has no vocabulary to distinguish
either.** Case Study 1 is catastrophic -- +0.19 AUROC -- regardless of
scale. It is a full-dataset labeling leak, not something a bigger or
smaller model changes. Case Study 3's mechanism is different in kind:
the properly budget-\emph{and}-seed-matched result is a small,
real residual gap that reaches significance at 2 of the 4 capacities tested
(48 and 128 units), and does not reach significance at the other 2 (16
units, indistinguishable from zero; 384 units, borderline).

The lesson this mechanism teaches is narrower but still real. A
severity estimate from a synthetic reconstruction depends on getting
three things correct simultaneously, before asking whether a placebo
control or a pre-registered decision rule confirms or refutes peeking:
a correctly-inverted calibration formula; the training-budget matching;
and verification that the matching procedure itself introduces no new
confound (in particular, an unmatched random seed introduced by the
matching procedure itself can dilute power and change which capacities
reach significance). Get any
one wrong, in either direction, and a confident-looking conclusion can
be wrong. Appendix A documents the calibration and confound-detection
process for this case study in full, as supporting evidence for the
checklist in \S5.

The real-feature validation of this mechanism (\S4.3)
shows a small anomaly: CLEAN\_MATCHED underperforms a permuted-label
PLACEBO on the real, smaller, noisier feature set, at both $n=50$ and
$n=100$ seeds, with significance sharpening rather than weakening at the
larger $n$ ($p=0.0012$
at $n=100$, versus $p=0.0175$ at $n=50$, a near-identical effect size
both times) -- inconsistent with the anomaly being sampling noise. The headline
LEAKY-vs-CLEAN\_MATCHED gap is adequately powered and real across both
sample sizes. The anomaly has an
account that is correlational rather than intervention-confirmed (\S4.3):
CLEAN\_MATCHED's fixed epoch count, inherited from
CLEAN's early-stopping, under-trains relative to where PLACEBO's own
checkpoint-selection lands, and the per-seed epoch gap predicts the
AUROC gap directly ($r=-0.594$, $p=7.5\times10^{-11}$) -- a
construction artifact of one control, not a challenge to the
checkpoint-reuse severity estimate itself, which remains adequately
powered and holds.

**Limitations.** We could not re-run either external paper's full
7B-scale pipeline within this paper's compute budget, so Case Studies 3
and 4's severity estimates for the *actual published numbers* remain
open questions. Our budget-and-seed-matched, capacity-swept
reconstruction of Case Study 3 supports a small, positive severity
estimate at 2 of 4 capacities tested (16-384 units), with no
distinguishable effect at the other 2. MultiHaluDet's actual
architecture (6 transformer layers, multi-scale branches, heavy
augmentation) is more expressive still, so this mixed result should not
be over-read in either direction for their actual pipeline -- it is what
our reconstruction shows, not a settled answer for theirs. We see a
full-scale replication as a natural next step for follow-up work with
access to appropriate compute, not a gap this paper resolves. The
real-feature validation itself is also single-dataset and single-language:
$n=400$ HaluEval English \texttt{qa\_samples} only, despite MultiHaluDet's
own stated contribution being explicitly multilingual detection; whether
this severity estimate holds in the non-English configurations that
system's name centers on is untested and would need those language-specific
data splits, which are not part of MultiHaluDet's public
convenience-default loader path.

## 7. Conclusion

Four distinct label-information-leakage mechanisms recur across hidden-
state hallucination-detection pipelines, including our own. They differ
substantially in severity. Case Study 1 is a catastrophic +0.19 AUROC
inflation that holds regardless of model size. Case Study 3 is a related
but distinct checkpoint-selection mechanism; its true severity is a small
residual gap, significant at 2 of
4 tested capacities and not the other 2 -- neither a "confirmed,
significant, capacity-growing" effect nor a "not
significant anywhere, possibly entirely a budget artifact" one (Appendix A
documents why a defensible estimate here requires several controls to hold
simultaneously).

That two-step reversal-and-partial-reversal is itself a data point. A
defensible severity estimate for this mechanism requires simultaneously
correct handling of the calibration formula, training-budget matching,
and seed matching (Appendix A documents how each can independently
fail), checked against a pre-registered decision rule that itself
returns \texttt{MIXED}, not a cleaner verdict. That difficulty
is the strongest argument in the paper for the checklist we
provide, more so than any of the four mechanisms in isolation.

The checklist lets researchers and reviewers in this fast-growing
subfield ask precise, mechanism-specific questions of a pipeline's
evaluation protocol. Was a claimed severity number computed with a
correctly-inverted calibration formula? Was the budget-matched control
itself checked for new confounds? Was a pre-registered decision rule
actually consulted, rather than narrated past? These questions replace
"did you use cross-validation correctly" as a single, undifferentiated
concern.

This paper audits seven external repositories, of which two are
confirmed positive -- too small a sample to support any claim about how
common these four mechanisms are across the wider hidden-state probing
literature, and we do not make one. A natural next step, planned as
follow-on work rather than claimed here, is a pre-registered audit
applying this same checklist to a fixed, defined sample of 15-20
externally published probing papers selected before any of them are
examined, reporting outcomes for every included paper regardless of
result. That protocol -- sampling frame, inclusion criteria, and a
frozen version of the checklist above -- is already drafted; running it
is future work.

## Appendix A: Full Correction History for Case Study 3 (MultiHaluDet)

**This appendix documents how the Case Study 3 result in \S4.3 was
reached.** \S4.3 states that result directly; nothing below is required to
verify it. This appendix instead records the process of getting it
right -- three earlier approximations, each wrong in a different way, and
what was wrong with each -- as supporting evidence for the checklist in
\S5.

**Round 1 (initial pass): saturation.** Our first, uncalibrated attempt
at the synthetic reconstruction used an arbitrary class-separation
constant. It saturated at AUROC$=1.0000$ for both LEAKY and CLEAN
conditions. This is a real methodological trap worth naming explicitly:
a task with no room to fail leaves no room for a leakage bug to show any
inflation either.

We recalibrated task difficulty to a target, non-saturated AUROC of
$0.80$. We used the closed-form binormal identity
$\text{AUROC}=\Phi(\sqrt{J/2})$ (Simpson \& Fitter 1973), where $\Phi$ is
the standard normal CDF and $J$ the target separation. At 20 seeds,
$N=700$, 5-fold, hidden$=48$, we reported LEAKY mean AUROC $0.8708$
versus CLEAN mean AUROC $0.8689$ -- a gap of $+0.0019$, not significant
($p=0.170$). This first pass was underpowered ($n=10$ seeds in an
earlier iteration). It wrongly suggested "not significant, but
trending."

**Round 2: the calibration itself was wrong.**
This identifies why round 1's numbers looked plausible but were not. The
scripts underlying the round-1 results inverted $\Phi(\sqrt{J}/2)$ to
pick the target $J$. That is the equal-prior Bayes-\emph{accuracy}
formula, not the binormal AUROC identity. A companion paper's
Proposition 1 independently found and fixed the same formula bug
($\Phi(\sqrt{J}/2) \neq \Phi(\sqrt{J/2})$, since $2 \neq \sqrt{2}$).

Concretely: $J=2.833$ was intended to calibrate a 0.80-AUROC task. Under
the correct identity, $J=2.833$ actually gives a Bayes-optimal AUROC of
$0.883$ -- matching, almost exactly, the $\approx$0.87-0.88 LEAKY/CLEAN
AUROCs actually observed in round 1 and originally reported without
comment. The task was closer to saturation than intended, throughout
every round-1 result.

The corrected inversion is $J = 2\Phi^{-1}(0.80)^2 = 1.417$, which gives
a verified $\Phi(\sqrt{J/2})=0.800$ exactly. A properly-powered $n=100$
test at this corrected calibration, at one capacity, wrongly concluded
"significant, genuine peeking." It missed a second problem, described
below.

**Round 3: a training-budget confound.** LEAKY and
CLEAN were never matched on training-data budget. CLEAN's
checkpoint-selection carve-out (15\% of the training fold, held out for
early-stopping) meant CLEAN trained on $\approx$15\% less data than
LEAKY, at every capacity. So any LEAKY-CLEAN gap could reflect a
training-budget artifact rather than label-peeking.

We added a third condition, CLEAN\_MATCHED. The best epoch count is
still chosen from the disjoint carve-out, using the same selection
procedure as CLEAN. But the model is then retrained from scratch on the
\emph{full} training fold -- the same data budget as LEAKY -- for
exactly that many epochs. Fixing this, at first pass, overcorrected into
"not significant at any capacity." The fix had introduced its own
confound, described below.

**Round 4: an unmatched
random seed.** `CLEAN_MATCHED`'s retraining used a different random seed
(`fold_seed + 777`) than `LEAKY`/`CLEAN` (`fold_seed`). This introduced
an independent source of variance that diluted the budget-matched
comparison's power. It made the round-3 "not significant at any
capacity" conclusion partly an artifact of the fix's own procedure, not
just of the training-budget correction.

We matched the seed, so `CLEAN_MATCHED` differs from `LEAKY` only in
epoch-count provenance, not also in weight initialization. Rerunning all
four capacities at $n=100$ gives the result now stated in \S4.3: a
small, capacity-dependent residual effect. It is significant, surviving
Holm-Bonferroni multiplicity correction, at two of four capacities. One
of the remaining two is better described as underpowered than null. This
matches neither of the two previous "resolved" conclusions -- not round
2's "confirmed, capacity-growing," nor round 3's "possibly entirely a
training-budget artifact."

**The methodological lesson this section demonstrates about itself, not
just argues for.** A single confirmatory-looking or null-looking number
is not sufficient to close a leakage-severity question. This holds even
at adequate seed count and with a placebo control. Five specific
controls must all be present \emph{simultaneously}: a correctly-inverted
calibration formula; a training-budget-matched control checked for new
confounds \emph{introduced by the matching procedure itself}; adequate
power; a permuted-label placebo; and consulting one's own pre-registered
decision rule rather than narrating past it. Appendix A documents the full
calibration and confound-detection process for Case Study 3, including
the specific ways each control can fail independently -- supporting
evidence for the checklist in \S5.

## References

Full citations below, compiled
from exactly the bibliographic detail already verified in-text; HaRP and
GUARDIAN (Case Studies 1-2) are this paper's own pipelines, not external
publications, and are not separately listed here.

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
