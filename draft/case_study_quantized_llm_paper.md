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
                                                                  # (33 layers x n_seeds runs),
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

So `best_layer` is selected by taking the argmax, over 33 layers (and
multiple seeds), of a metric computed on the test set -- and that same
test-set AUROC is then reported as the paper's headline number for that
layer. This is not nested-CV leakage; it is **test-set-driven multiple-
hypothesis selection with no correction** -- the classical "winner's
curse" / "if you try enough things and report the best one, the best one
looks better than it is" bias. Trying 33 layers (times however many
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
combinations (33 layers, 3 seeds each).
`code/45_case_study_4_winners_curse.py` holds out each seed in turn,
selects the best layer using **only** the other two, takes the naive
baseline as the selection-set mean at that layer, and takes the honest
estimate as the held-out seed's AUROC at the same layer.

**Result (24 combinations, leave-one-seed-out, 3 rotations each):**

| Subset | n cells | Mean winner's curse | BCa 95% CI | Wilcoxon p |
|---|---|---|---|---|
| All cells | 24 | **+0.0054** | [+0.0032, +0.0100] | 0.00017 |
| Ceiling-saturated (selected layer AUROC >= 0.975) | 12 | +0.0042 | [+0.0029, +0.0057] | 0.0005 |
| Non-saturated | 12 | +0.0065 | [+0.0023, +0.0152] | 0.047 |

By dataset: `fever` +0.0124, `halueval_qa` +0.0052, `synthetic` +0.0033,
`truthfulqa` +0.0007. Max single cell: +0.0347
(`qwen2.5-7b__fever__linear`).

**Correction embedded in that number.** The first version of `code/45`
took the naive baseline from the repo's own reported `best_auroc`, which
is a max over **all three** seeds including the one then used as
"held-out." Whenever the leave-one-out argmax coincided with the
full-sample argmax (7 of 24 cells), the two sides of the subtraction were
arithmetically forced equal and the estimate was pinned at exactly
0.0000. The corrected, genuinely leave-one-out protocol raises the mean
from +0.0047 to +0.0054 and the max from +0.0288 to +0.0347, and — for
the first time — attaches a CI and a significance test to this section.

Half these cells are ceiling-saturated, which is itself the paper's
broader point: severity here is governed by where on the difficulty curve
the pipeline operates, not by which leakage mechanism is present.

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
