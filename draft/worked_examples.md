# Paper 2 — Worked Examples (own-project case studies)

These are the two internal, fully-reproducible case studies that anchor the
audit paper before we bring in external targets. Both come from projects in
this portfolio and are already verified against raw logs (not re-derived
from memory) — see the earlier deep-dive reports for the primary evidence
trail. This document turns them into the clean, paper-ready form.

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

The gap the CV-based selection number and the true held-out number is
**18.8 points of AUROC** — entirely attributable to the fact that L11 was
*chosen* because it happened to maximize CV AUROC on this specific sample,
which is exactly the condition under which CV-based selection is known to
be optimistic.

A second, independent demonstration of the same fragility: which layer
counts as "optimal" depends on the selection rule. Argmax over the
train-split CV AUROC selects **L11** (37.5% relative depth). Argmax over
all-data AUROC instead selects **L19** (62.5% relative depth). Two
equally-defensible selection rules, applied to the same underlying numbers,
disagree about which layer the paper's own headline claim should be about.

### Why this generalizes beyond one project

Any paper in this space that reports "we swept all layers and found layer
X is optimal, achieving AUROC Y" and uses the *same* cross-validation run
both to select X and to report Y is vulnerable to this exact gap. The size
of the gap (18.8 points here) suggests this is not a negligible, ignorable
effect at the sample sizes (N≈700) common in this literature.

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
