# Case Study 4 (secondary) — Test-set-driven best-layer selection

## Target
"Hallucination Is Linearly Decodable from Mid-Layer Hidden States in
Quantized LLMs" (arXiv 2606.02628). Repo:
https://github.com/Ezharjan/HallucinationPatternDetection. Reports
AUROC 0.904-1.000 (yes, a perfect 1.000 in at least one configuration)
detecting hallucination from Llama-3.1-8B / Mistral-7B / Qwen2.5-7B hidden
states, 4-bit NF4 quantized.

## Why this is a DIFFERENT bug than Case Studies 1 and 3, verified directly in code

`src/detection/probes.py::train_probe` does a clean single stratified
70/10/20 train/val/test split per (model, dataset, layer) -- **not**
K-fold CV, so it structurally cannot have HaRP's or MultiHaluDet's exact
nested-CV leakage pattern. Early stopping correctly uses only the val
split (`best_val` tracked via `val_auroc`, `X_te`/`y_te` never touched
during training). On that axis, this pipeline is clean.

The bug is one level up, in `src/detection/saplma.py::saplma_probe_per_layer`:

```python
layerwise = train_layerwise_probes(hidden_states, labels, ...)  # trains one
                                                                  # probe PER LAYER
                                                                  # (33 layers for llama3.1-8b
                                                                  #  and mistral-7b, 29 for
                                                                  #  qwen2.5-7b, x n_seeds runs),
                                                                  # each with its OWN held-out test AUROC
best_layer = -1
best_auc = -np.inf
for li, m in layerwise["per_layer"].items():
    v = m.get("auroc_mean", float("nan"))     # <- this is the TEST-SET AUROC, not train/val
    if not np.isnan(v) and v > best_auc:
        best_auc = v
        best_layer = int(li)
```

`train_layerwise_probes` (same file, `probes.py`) calls `train_probe` once
per layer, and the metric it aggregates into `auroc_mean` is
`ProbeMetrics.auroc` -- computed on `X_te`/`y_te`, the held-out **test**
split, inside `train_probe` itself (`acc, f1, auroc, auprc = _metrics(y_te, prob)`).

So `best_layer` is selected by taking the argmax, over every probed layer
(and multiple seeds), of a metric computed on the test set -- and that same
test-set AUROC is then reported as the paper's headline number for that
layer. This is not nested-CV leakage; it is **test-set-driven multiple-
hypothesis selection with no correction** -- the classical "winner's
curse" / "if you try enough things and report the best one, the best one
looks better than it is" bias. Trying 33 layers for `llama3.1-8b` and
`mistral-7b`, or 29 for `qwen2.5-7b` (times however many
probe-type/model/dataset combinations get run) and keeping only the single
best test-AUROC number, without a separate selection split or a multiple-
comparisons correction, is exactly the setup under which a reported
"1.000 AUROC" becomes plausible even if no single layer is genuinely that
separable.

## Relationship to GUARDIAN's own internal finding (Case Study 2)

This is structurally the *same family* of bias as GUARDIAN's CV-optimism
gap (Case Study 2) -- both are "used the evaluation labels to make a
selection decision, then reported the resulting metric as if the
selection were free" -- but via a different mechanism (best-of-many-
hypotheses test-set selection here, vs. best-of-many-hypotheses CV-based
layer selection there). Seeing the same underlying hazard surface in two
independently-written, unrelated pipelines (one from this portfolio, one
external) is itself evidence this is a general pattern in the hidden-
state-probing literature, not a one-off mistake.

## What we did NOT do here (scope note for the paper)

We did not re-run this pipeline end-to-end with a corrected protocol
(e.g., select the layer using only the val split, then report that
layer's test AUROC once) -- doing so requires re-extracting Llama-3.1-8B/
Mistral-7B/Qwen2.5-7B hidden states, which needs GPU beyond what's
budgeted for this secondary case study.

**UPDATE: this case study IS quantified.** An earlier version of this
document said its contribution was "the code-verified identification of a
second, distinct bias family (not a quantified before/after number)."
That is no longer accurate. A fresh review found the winner's-curse
estimate is fully computable from files already vendored in this repo:
`code/external/HallucinationPatternDetection/results/probes/*.json` ships
per-layer, per-seed AUROC arrays for all 24 model x dataset x probe-type
combinations (33 probed layers for `llama3.1-8b` and `mistral-7b`, 29 for
`qwen2.5-7b`; 3 seeds each — an earlier version of this document said a flat
"33 layers" in four places, which was wrong for the eight `qwen2.5-7b` cells;
`code/45` now asserts the per-model counts against the shipped files).
`code/45_case_study_4_winners_curse.py` holds out each seed in turn,
selects the best layer using **only** the other two, takes the naive
baseline as the selection-set mean at that layer, and takes the honest
estimate as the held-out seed's AUROC at the same layer.

**Result (leave-one-seed-out, 3 rotations each). The headline is the 17 of
24 cells on which the estimator is non-degenerate — see the correction
history below for why the other 7 are excluded rather than averaged in:**

| Subset | n cells | Mean winner's curse | BCa 95% CI | Wilcoxon p |
|---|---|---|---|---|
| **Non-degenerate (headline)** | **17** | **+0.0076** | [+0.0048, +0.0134] | 1.5e-05 |
| Degenerate (estimate algebraically 0; disclosed, not averaged in) | 7 | +0.0000 | [0, 0] | — |
| Ceiling-saturated (selected layer AUROC >= 0.975; none degenerate) | 12 | +0.0042 | [+0.0029, +0.0057] | 0.0005 |
| Non-saturated, non-degenerate | 5 | +0.0157 | [+0.0074, +0.0276] | 0.0625 |
| *Non-saturated, all cells (contaminated: 7 of 12 are algebraic zeros)* | *12* | *+0.0065* | *[+0.0022, +0.0154]* | *0.0625* |
| *All 24 cells (contaminated by the same 7)* | *24* | *+0.0054* | *[+0.0031, +0.0101]* | *0.00029* |

The saturated-vs-non-saturated contrast is **+0.0042 vs +0.0157**, a 3.7x
difference — not the +0.0042 vs +0.0065 the contaminated rows suggest.

By dataset: `fever` +0.0124, `halueval_qa` +0.0052, `synthetic` +0.0033,
`truthfulqa` +0.0007 (the last is exactly 0.0000 in four of its six cells for
the degeneracy reason below, and is not a dataset effect). Max single cell:
+0.0347 (`qwen2.5-7b__fever__linear`).

**Correction 1, embedded in that number.** The first version of `code/45`
took the naive baseline from the repo's own reported `best_auroc`, which
is a max over **all three** seeds including the one then used as
"held-out." Whenever the leave-one-out argmax coincided with the
full-sample argmax (7 of 24 cells), the two sides of the subtraction were
arithmetically forced equal and the estimate was pinned at exactly
0.0000.

**Correction 2: Correction 1 did not work, and this is the important one.**
A second independent review found that rebuilding the estimator as a
leave-one-seed-out rotation removed the *stated cause* of that degeneracy
while leaving the degeneracy itself intact, in the same 7 of 24 cells. The
rotation estimator is *algebraically* pinned to zero whenever all three
rotations select the same layer. With `a_0, a_1, a_2` the three seeds' AUROC
at that common layer, `S = sum(a_r)` and `abar = S/3`:

    naive_r  = (S - a_r) / 2      (mean of the two selection seeds)
    honest_r = a_r
    wc_r     = naive_r - honest_r = (3/2) * (abar - a_r)
    mean_r wc_r = (3/2) * (abar - abar) = 0, identically, for ANY data.

The per-rotation values are 1.5x the held-out seed's deviation from the seed
mean — pure noise summing to zero by construction. Two of the seven cells
carried float residues of +-3.7e-17, which scipy's Wilcoxon was ranking as
real signed observations instead of dropping as exact zeros.

All 7 degenerate cells fall in the **non-saturated** subgroup (7 of its 12),
which is the subgroup main.tex §5 leans on. Its +0.0065 was 5 genuine
measurements averaging +0.0157 diluted by 7 algebraic zeros, and its Wilcoxon
p moves from 0.047 to 0.0625 once the residues are counted as zeros.

`code/45` now (i) detects and declares degeneracy per cell, recording each
rotation's selected layer and snapping |wc| < 1e-12 to exact zero;
(ii) reports the headline over the 17 non-degenerate cells; and
(iii) characterizes the estimator *including* its degeneracy with a
permutation null. A **global** seed relabelling leaves this estimator exactly
invariant — it already averages over all three choices of held-out seed — so
the shuffle is applied independently **within each layer**, which destroys the
layer x seed structure that makes an argmax stable while preserving each
layer's marginal values. That null is the regime of *maximal* winner's curse
(noise-driven argmax), so it answers what the degenerate cells cannot: the 7
degenerate cells' null averages **+0.0179**, above the non-degenerate cells'
observed +0.0076. Their exact zeros therefore reflect a stable argmax
defeating the estimator, not an absent selection effect.

**The same null cuts the other way too, and an earlier revision reported only
the favourable half.** Run on all 24 cells, **0 of the 24 reach p < 0.05**
against their own null (minimum p = 0.277, median p = 1.000), and the
non-degenerate cells' own null mean of **+0.0118** likewise sits *above* their
observed **+0.0076**. This does not disturb the degeneracy result — that is an
exact algebraic identity and depends on no null — nor the reading of the seven
exact zeros as artifacts. It does disturb any claim that +0.0076 is
statistically established: against this null it is not. The null is a hard
reference by design (shuffling within layers destroys the structure that makes
an argmax stable, putting the estimator in its maximal-winner's-curse regime),
so a real but modest effect on a stable argmax is expected to fall below it —
which is exactly what makes it diagnostic for the degenerate cells and a poor
significance test for the rest. With 3 seeds per cell no sharper null is
available from the shipped arrays. +0.0076 is therefore reported as a point
estimate with a BCa interval, resting on the uniform positivity across all 17
non-degenerate cells rather than on a p-value against this null. All 24
per-cell values and p-values are tabulated in `main.tex` Appendix B.6.

A structurally different second estimator — the bootstrap bias of a
max-of-means, which cannot telescope because its two sides use different seed
multisets — agrees in sign everywhere and averages +0.0027 over the
non-degenerate cells (smaller than the rotation estimate, as expected at three
seeds).

Half these cells are ceiling-saturated, which is itself the paper's
broader point: severity here is governed by where on the difficulty curve
the pipeline operates, not by which leakage mechanism is present. (main.tex
§5 now also bounds how far that relationship transports.)

## Two further sites in this same repository, found by an independent blind re-read

Added during a closure review. A second rater, given the vendored source
but not this paper, its checklist, or the location of any known bug, was
asked to read this repository for leakage sites. It re-derived the
`saplma.py` bug above independently, and surfaced two sites this paper had
not previously reported. Both were verified directly against the pinned
commit `ea0b9678`.

**(i) `scripts/06_analyze_attention.py:33-52` — best-layer argmax with no
split at all.** The loop computes `score_to_metrics(labels, per_sample[:, li])`
for every kept layer using `data["labels"]` — the *entire* labelled set,
with no train/val/test partition anywhere in the file — then takes
`best = max(per_layer, key=lambda k: per_layer[k]["auroc"])` and writes
`best_auroc` into the metrics JSON. `scripts/07_aggregate_results.py:74-86`
folds that value straight into the headline results table. This is
Mechanism 4 in a purer form than the `saplma.py` site: there, at least, a
held-out split exists and is merely selected on; here there is none.

**(ii) `src/analysis/metrics.py:57-65` (`score_to_metrics`) — Mechanism 5,
in this repository too.** The function computes
`fpr, tpr, thr = roc_curve(target, score)`, takes `best = int(np.argmax(j))`
with `j = tpr - fpr` (Youden), and then scores `accuracy_score(y_true, ...)`
and `f1_score(y_true, ...)` at that threshold — against the same labels the
threshold was chosen on. `scripts/07_aggregate_results.py:85-86` reports
both as `accuracy` and `f1`.

Site (ii) matters beyond this case study: Mechanism 5 was introduced in
`main.tex` §4.5 as something found inside MultiHaluDet's `run_pipeline.py`.
It occurs independently in *both* audited repositories. Two independently
published pipelines committing the identical test-label-tuned
operating-point selection is what makes the mechanism worth naming
separately; one instance would not have been.

Neither site changes any severity number in this paper — both are additional
instances of mechanisms already quantified, not new mechanisms — and neither
is counted among the regex scanner's 7 raw hits, which are the scanner's
output rather than a human (or second-rater) audit's.
