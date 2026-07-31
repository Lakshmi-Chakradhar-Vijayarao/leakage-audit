# Paper 2 — Worked Examples (own-project case studies)

These are the two internal case studies that anchor the audit paper before
we bring in external targets. Both come from projects in this portfolio.

> **Evidentiary status (must be read alongside every number below).**
> These two case studies are **not** equally verifiable by a reader of the
> submitted artifact, and the paper now grades them explicitly:
>
> - **Case Study 2 (GUARDIAN)** is reproducible from what ships with the
>   paper. Its original 171 MB hidden-state cache is too large to include,
>   so `code/48` emits a ~1 MB derived artifact (per-layer, per-sample CV
>   fold assignments and probe scores for all 32 layers, for the
>   sequential split, the reversed split and all 8 randomized reps) and
>   `code/51` recomputes and asserts every reported number from it.
> - **Case Study 1 (HaRP)** is **not** independently re-verifiable from
>   the submitted artifact. No script, log, or data file supporting its
>   `+0.19` figure is included. The mechanism is real and clearly
>   described; the number is reported from prior work, is excluded from
>   the paper's abstract severity range, and is excluded from every
>   cross-mechanism comparison. Treat the table below as a narrative
>   account, not a measurement a reader can check here.

---

## Case Study 1 — Nested-CV target leakage (HaRP, hidden-state hallucination detection)

### The bug, precisely

HaRP's "Group C" representation-signal feature set includes `probe_conf`:
the output of a logistic-regression probe trained on Qwen 2.5 3B's L32
last-token hidden state, predicting P(correct). The leaked version
(`experiments/09_train_estimator_v2.py`) fit this probe on **all 700**
labeled samples, then called `predict_proba` on those same 700 samples to
populate the feature column in `features_combined.csv`. That column then
fed into the failure estimator's outer 5-fold CV
(`src/authorization/failure_estimator.py`).

This is nested-CV target leakage in its cleanest form: the outer CV correctly
holds out test folds *for the failure estimator*, but the `probe_conf`
feature itself was manufactured by a model that had already seen every
label in the dataset, including the labels of whatever fold the outer loop
later treats as "held out." The outer split cannot retroactively un-leak an
inner model that was fit on the full label set before the split ever ran.

### Quantified effect (before → after)

| Metric | Leaked (Exp 23) | Fixed, OOF (Exp 24) | Δ |
|---|---|---|---|
| A+B+C AUROC (5-fold CV) | 0.9620 | 0.7714 | **+0.1906** |
| SOTA AUROC (5-fold CV) | 0.9624 | 0.7753 | +0.1871 |
| Test AUROC | 0.9521 | 0.7871 | +0.1650 |
| Hallucination rate on ACCEPT | 0.0000 | 0.1667 | −0.1667 |
| Accept rate | 31.7% | 4.3% | −27.4pp |

`probe_conf` in isolation: in-sample AUROC **1.0000** (pure memorization) →
proper out-of-fold AUROC **0.7573**. This independently cross-checks against
Exp 06's separately-obtained cross-validated probe AUROC of **0.7583** —
the two numbers agree to within 0.001, which is the strongest evidence the
fix is correct (an independent measurement of the same underlying quantity,
obtained a different way, lands in the same place).

### The fix

`experiments/09b_oof_group_c.py` / `src/signals/representation_signals.compute_oof_group_c`:
for each of 5 `StratifiedKFold` splits, refit BOTH the logistic-regression
probe AND the class centroids (μ_correct, μ_hallucinated) using **training-fold
data only**, then compute `probe_conf`, `cos_correct`, `cos_hallucinated` on
the **held-out fold only**. The fourth Group-C feature, `h_norm` (a label-free
L2 norm), needs no correction since it never touched labels.

### Why this generalizes beyond one project

The failure mode is generic to any pipeline where a "representation summary"
feature (a centroid, a fitted probe's confidence, a fitted classifier's
margin) is computed once over the full dataset and then treated as an input
feature to a separate downstream CV loop. This exact shape recurs across
the hidden-state-probing literature whenever papers report "we combine our
probe's output with other signals in an ensemble" — which is common. That's
the generalizable claim Paper 2 needs external evidence for (see the
external audit-candidates task).

---

## Case Study 2 — Cross-validation optimism bias in layer/hyperparameter selection (GUARDIAN, Mistral 7B)

### The bug, precisely

This is a *different* bias from Case Study 1 — no feature is leaked across
folds; instead, the **layer** used for detection is selected by maximizing
a 5-fold CV AUROC over all 32 layers, and that same CV number is then
reported as if it were the model's performance. This is the classical
"selection bias in performance estimation when CV is used for model
selection" problem (Varma & Simon, *BMC Bioinformatics* 2006) — well known
in general ML methodology, but not something this specific literature
(LLM hidden-state hallucination probing) has been shown to be vulnerable
to, concretely, with real numbers, until this project's own internal audit.

### Quantified effect

At GUARDIAN's selected layer (L11, chosen via argmax train-split CV AUROC):

| Evaluation protocol | AUROC |
|---|---|
| 5-fold CV, training split only (the number used to *select* L11) | 0.804 |
| 5-fold CV, all 700 samples | 0.776 |
| True held-out (probe trained on first 400, tested on the remaining, never-seen-during-selection 300) | **0.616** |

The gap between the CV-based selection number and the true held-out number
is **18.8 points of AUROC**. An earlier version of this document described
that gap as "entirely attributable" to L11 having been *chosen* because it
maximized CV AUROC on this specific sample. **That attribution is
retracted** — see "Correction: the held-out split above is not actually
random" below. Under a proper random partition, the general CV-vs-held-out
gap is indistinguishable from zero, and the component genuinely
attributable to argmax-over-layers selection is $+0.027$ (SD $0.021$,
$p=0.009$), about 7x smaller than the 18.8-point headline. The 18.8-point
number is retained here only as the original, uncorrected observation.

A second, independent demonstration of the same fragility: which layer
counts as "optimal" depends on the selection rule. Argmax over the
train-split CV AUROC selects **L11** (37.5% relative depth). Argmax over
all-data AUROC instead selects **L19** (62.5% relative depth). Two
equally-defensible selection rules, applied to the same underlying numbers,
disagree about which layer the paper's own headline claim should be about.

### Why this generalizes beyond one project

Any paper in this space that reports "we swept all layers and found layer
X is optimal, achieving AUROC Y" and uses the *same* cross-validation run
both to select X and to report Y is vulnerable to this exact gap. The
correctly-isolated size of that gap here, however, is small: $+0.027$
AUROC (SD $0.021$, $p=0.009$) under randomized splits, not the 18.8 points
the confounded sequential split suggested. It is a real and statistically
significant effect at the sample sizes (N≈700) common in this literature,
but a small one — consistent with every other code-verified severity
estimate in this paper, all of which fall between roughly $0.000$ and
$+0.027$ AUROC.

### Correction: the held-out split above is not actually random

A subsequent independent review found that the `H[:400]`/`H[400:]` split
used for "True held-out" in the table above is a fixed sequential slice
of the dataset's storage order, not a random partition — a hidden-state
probe can separate the two halves at AUROC 0.734-0.776, meaning they are
systematically different populations, not exchangeable draws. The gap
specifically attributable to *selecting* L11 via argmax (gap at the
selected layer minus the mean gap across all layers) is only $-0.004$
under this sequential split (where L11 is selected) — indistinguishable
from zero.

**A second review caught a bug in this correction itself:** an earlier
version evaluated every randomized-split rep at a hardcoded L11 rather
than at that rep's own selected layer, which does not measure selection
optimism — a different split can select a different layer entirely.
Fixed: every rep now uses its own argmax-selected layer. Reversing which
half selects picks L17 (not L11) and gives a component of $+0.074$ — a
large swing in both which layer is picked and what the component
measures, itself evidence the sequential-split number tracks split
order rather than a stable property of the selection procedure.
Recomputing this selection-specific component under 8 randomized
stratified splits, each evaluated at its own selected layer (L10, L11,
L12, L14, L16, L18, L19, L19 across the 8 reps), gives a mean of
$+0.027$ (SD $0.021$; $t=3.60$, $p=0.0087$; Wilcoxon $p=0.023$) — small
but, for the first time, correctly measured and statistically
significant, about 7x smaller than the sequential split's apparent
effect. Separately, the *general* CV-vs-held-out gap averaged across all
32 layers is $+0.192$ under the sequential split but only $-0.005$ (SD
$0.073$) under the same randomized splits — indistinguishable from zero.
**The corrected claim reverses, not just narrows, the original one**:
the large, real, ~19-point gap "present at essentially every layer" was
itself entirely an artifact of the sequential split's
population-difference confound — under a proper random partition there
is no general CV-vs-held-out optimism in this setup at all. What
survives is narrower but genuine: a small, statistically significant,
selection-specific penalty from CV-based argmax-over-layers selection,
about 7x smaller than GUARDIAN's original headline number. The broader
methodological lesson survives, and is arguably strengthened by the
second bug: "held out" must mean "randomly partitioned," not merely "a
different index range" — and a *selection-specific* quantity computed
across multiple resamples must evaluate each resample at its own
selected layer, not one hardcoded from a single split. This project's
own held-out split, and its own first attempt at correcting it, both
needed exactly the scrutiny this paper recommends applying to others.

---

## What Paper 2 needs next (external validation)

Both case studies above are internal — a skeptical reviewer's first
question will be "is this just something wrong with your own two hobby
projects, or does it generalize?" That's exactly why the paper's core new
work is auditing 2-3 *externally published* papers with public code,
re-deriving out-of-fold numbers where they report suspiciously high (0.90+)
cross-validated AUROC, and reporting honestly whether the same two bias
patterns (Case Study 1: nested feature leakage; Case Study 2: CV-based
selection optimism) are present, absent, or present in some different form.
See the separate candidate-scoping task for the current shortlist.
