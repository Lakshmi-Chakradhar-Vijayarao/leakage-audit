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
                                                                  # (~30 layers x n_seeds runs),
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

So `best_layer` is selected by taking the argmax, over ~30 layers (and
multiple seeds), of a metric computed on the test set -- and that same
test-set AUROC is then reported as the paper's headline number for that
layer. This is not nested-CV leakage; it is **test-set-driven multiple-
hypothesis selection with no correction** -- the classical "winner's
curse" / "if you try enough things and report the best one, the best one
looks better than it is" bias. Trying ~30 layers (times however many
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
budgeted for this secondary case study. This case study's contribution is
the code-verified identification of a second, distinct bias family (not a
quantified before/after number, which Case Study 3's synthetic ablation
already provides for the sibling checkpoint-selection bug). If Kaggle
budget allows after Paper 3's runs complete, a cheap partial check is
possible: re-run just `03_train_probes.py` on already-extracted layers
with a corrected val-based layer-selection rule and compare the single
selected layer's test AUROC against the paper's reported "best of 30"
number, without needing to re-extract any hidden states.
