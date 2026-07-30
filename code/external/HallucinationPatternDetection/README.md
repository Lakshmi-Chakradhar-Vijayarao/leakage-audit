# Hallucination Pattern Detection in Open-Source LLMs

An academic research framework for detecting, analyzing, and visualizing
hallucination patterns in open-source Large Language Models by probing
their internal hidden states.

The entire framework is **pure Python** and runs identically on
**Windows, macOS, and Linux** -- no shell, batch, or PowerShell scripts
are used or required.

---

## Table of Contents

- [Hallucination Pattern Detection in Open-Source LLMs](#hallucination-pattern-detection-in-open-source-llms)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Hardware target](#hardware-target)
  - [Quickstart (Windows, macOS, Linux -- identical commands)](#quickstart-windows-macos-linux----identical-commands)
  - [Environment setup (detailed)](#environment-setup-detailed)
    - [1. Create the environment](#1-create-the-environment)
    - [2. Activate `swarm`](#2-activate-swarm)
    - [3. Verify GPU + interpreter](#3-verify-gpu--interpreter)
  - [Common fix: torch is CPU-only OR built for the wrong GPU arch](#common-fix-torch-is-cpu-only-or-built-for-the-wrong-gpu-arch)
  - [HuggingFace authentication (optional)](#huggingface-authentication-optional)
  - [Models analyzed](#models-analyzed)
  - [Datasets](#datasets)
  - [Detection methods (high level)](#detection-methods-high-level)
  - [Methodology (detailed)](#methodology-detailed)
    - [1. Hypothesis](#1-hypothesis)
    - [2. Models (detail)](#2-models-detail)
    - [3. Datasets (detail)](#3-datasets-detail)
    - [4. Hidden-state extraction](#4-hidden-state-extraction)
    - [5. Detection methods (detail)](#5-detection-methods-detail)
      - [5.1 SAPLMA-style probes](#51-saplma-style-probes)
      - [5.2 INSIDE / Semantic entropy](#52-inside--semantic-entropy)
      - [5.3 Self-consistency](#53-self-consistency)
      - [5.4 Attention entropy](#54-attention-entropy)
    - [6. Metrics](#6-metrics)
    - [7. Layer geometry](#7-layer-geometry)
  - [Running the pipeline](#running-the-pipeline)
    - [Full pipeline (one command)](#full-pipeline-one-command)
    - [Individual stages](#individual-stages)
    - [Visualization-only mode (regenerate figures from existing results)](#visualization-only-mode-regenerate-figures-from-existing-results)
    - [Resuming after a failure](#resuming-after-a-failure)
    - [Useful flags for the runner](#useful-flags-for-the-runner)
  - [What you get](#what-you-get)
  - [Output organization](#output-organization)
  - [Reduce wall-clock time](#reduce-wall-clock-time)
  - [Add a new model](#add-a-new-model)
  - [Tests](#tests)
  - [Reproducibility](#reproducibility)
  - [Repository layout](#repository-layout)
  - [Citation](#citation)
  - [License](#license)

---

## Overview

This project investigates *whether and how a language model's own internal
representations encode the difference between truthful generations and
hallucinated ones*. We extract per-layer hidden states and attention
patterns from quantized 7B/8B models, train lightweight probes on these
representations, apply state-of-the-art uncertainty estimators
(INSIDE-style semantic entropy, SAPLMA, self-consistency), and visualize
the resulting hallucination geometry across layers, models, and datasets.

The framework is designed for reproducible academic experimentation on
consumer GPUs (RTX 5060, 8 GB VRAM) via 4-bit quantization.

## Hardware target

- GPU: NVIDIA RTX 5060 (8 GB VRAM) -- any 8 GB+ CUDA GPU works
- Disk: ~30 GB free for models, hidden states, and figures
- CUDA / cuDNN compatible drivers
- For RTX 50-series (Blackwell, sm_120) you MUST use the cu128 PyTorch
  wheels (see [Common fix](#common-fix-torch-is-cpu-only-or-built-for-the-wrong-gpu-arch) below). The cu124 wheels do not include
  SASS code for sm_120 and will silently fail.

The full pipeline on an RTX 5060 (8 GB) is estimated at roughly 4-8
hours: the bulk is INSIDE and self-consistency, which require K
stochastic generations per prompt. To reduce wall-clock time, see
[Reduce wall-clock time](#reduce-wall-clock-time).

## Quickstart (Windows, macOS, Linux -- identical commands)

> Every command below is plain Python. The framework deliberately
> avoids shell, batch, and PowerShell scripts so the same workflow
> runs on every OS. You only need (1) a working conda installation
> and (2) a CUDA-compatible driver for your GPU.

```text
# 1) Create the conda environment (once)
conda env create -f environment.yml

# 2) Activate it -- REQUIRED before running anything in this project
conda activate swarm

# 3) Verify everything (torch+CUDA, GPU arch match, bitsandbytes, ...)
python scripts/00_check_install.py

# 4) Run the entire pipeline end-to-end (pre-flight runs automatically)
python scripts/run_pipeline.py
```

`scripts/run_pipeline.py` invokes `scripts/00_check_install.py` first.
If your torch wheel is CPU-only, bitsandbytes is missing, your GPU
arch isn't in torch's compiled arch list, or any other dependency is
broken, the runner aborts before downloading the 16 GB of model
weights and prints the exact `pip` command to fix the problem.

## Environment setup (detailed)

### 1. Create the environment

```text
conda env create -f environment.yml
```

This creates an environment called **`swarm`** with PyTorch,
`transformers`, `bitsandbytes`, `datasets`, `umap-learn`, etc.

### 2. Activate `swarm`

```text
conda activate swarm
```

You must run this in every new terminal session before invoking any
Python script in this project. The pipeline runner verifies this for
you and prints a clear warning if you are in the wrong environment.

### 3. Verify GPU + interpreter

```text
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
python scripts/run_pipeline.py --check-env
```

If you do not see your GPU, install the appropriate PyTorch CUDA wheel
inside `swarm` (see the [Common fix](#common-fix-torch-is-cpu-only-or-built-for-the-wrong-gpu-arch) section).

## Common fix: torch is CPU-only OR built for the wrong GPU arch

If the pre-flight reports any of:

- `torch ... is CPU-only`
- `torch can't run kernels on your GPU`
- `compiled for [... sm_90]; your GPU needs sm_120` (RTX 50-series Blackwell)
- `GPU op failed` after `torch.cuda.is_available()` returned True

run **inside the `swarm` env**:

```text
pip uninstall -y torch torchvision torchaudio
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

The `cu128` wheels cover every NVIDIA architecture from Pascal (sm_60)
through Blackwell (sm_120), so they work on every supported card
including the RTX 50-series (5060, 5070, 5080, 5090).

Then re-run:

```text
python scripts/00_check_install.py
python scripts/run_pipeline.py
```

## HuggingFace authentication (optional)

The default model list is fully open and requires **no HF login**. If
you'd rather use the *gated* `meta-llama/Llama-3.1-8B-Instruct` repo,
swap the `hf_id` back in `config/config.yaml`, then authenticate once:

```text
huggingface-cli login
```

Mistral-7B-Instruct and Qwen2.5-7B-Instruct are open and require no
login.

The pipeline is resilient to per-model failures: if one model can't
be loaded (e.g. a gated repo without auth), the stage logs the error and
continues with the remaining models -- no single-model 401s killing
the whole run.

## Models analyzed

All models are loaded in 4-bit NF4 quantization via `bitsandbytes`
(`bnb_4bit_quant_type=nf4`, double quantization, compute dtype
`bfloat16`). This keeps every model under 6 GB VRAM, leaving room
for activations on an RTX 5060 (8 GB). The defaults are publicly
accessible (no HF login required):

| short name      | HF repo                                       | layers | hidden |
|-----------------|-----------------------------------------------|--------|--------|
| llama3.1-8b     | NousResearch/Meta-Llama-3.1-8B-Instruct       | 32     | 4096   |
| mistral-7b      | mistralai/Mistral-7B-Instruct-v0.3            | 32     | 4096   |
| qwen2.5-7b      | Qwen/Qwen2.5-7B-Instruct                      | 28     | 3584   |

The Llama-3.1-8B entry is the NousResearch ungated mirror, which is
byte-identical to `meta-llama/Llama-3.1-8B-Instruct`. If you already
have access to the gated meta-llama repo and have run
`huggingface-cli login`, you can swap the `hf_id` back in
`config/config.yaml`.

Per-model VRAM footprint with 4-bit NF4 is approximately 4.5 - 5.5 GB,
leaving headroom for activations. Eager attention implementation
(`attn_implementation="eager"`) is used because Flash-Attention does
not currently support `output_attentions=True`.

## Datasets

- **TruthfulQA** -- adversarial truthfulness questions (`truthful_qa`).
- **HaluEval** -- purpose-built hallucination benchmark covering QA,
  dialogue, and summarization (`pminervini/HaluEval`).
- **FEVER** -- fact-verification claims. The loader tries multiple HF
  mirrors (`pminervini/fever`, `fever/v1.0`, `mwong/fever-evidence-related`,
  `copenlu/fever_gold_evidence`) and probes multiple column-name and
  label-encoding variants (`label`, `label_id`, `gold_label`,
  `verifiable_label`; strings + integers). If a mirror yields 0 usable
  rows it falls through to the next one automatically.
- **Synthetic true/false statement pairs** -- controlled probes generated
  programmatically for clean class signal.

Each dataset is normalized to a list of
`PromptItem(prompt, answer, label, dataset, meta)`. Truthful pairs
receive label = 1, hallucinated pairs label = 0. See
[Methodology -- Datasets](#3-datasets-detail) for the per-dataset
construction details.

## Detection methods (high level)

1. **SAPLMA-style probes** -- linear and MLP classifiers trained on the
   hidden state of the final answer token at every transformer layer.
2. **INSIDE / Semantic entropy** -- eigenvalue-spectrum-based covariance
   measure over multiple sampled generations' hidden states.
3. **Self-consistency** -- exact-match and semantic-similarity agreement
   across stochastic samples.
4. **Attention-pattern analysis** -- per-head entropy of the final
   answer token's attention distribution over the input.

The mathematical formulation of each method is in
[Methodology -- Detection methods](#5-detection-methods-detail).

## Methodology (detailed)

### 1. Hypothesis

The central hypothesis is that an open-source LLM's internal hidden
representations carry a *linearly* (or near-linearly) decodable signal
distinguishing factually correct generations from hallucinated ones.
We test the hypothesis layer by layer, dataset by dataset, model by
model, and report the geometry of the resulting separation.

### 2. Models (detail)

All models are loaded in 4-bit NF4 quantization via `bitsandbytes`,
with double quantization and `bfloat16` compute dtype. The same
configuration is used for all three models; see the [Models analyzed](#models-analyzed)
table above for layer count / hidden size.

### 3. Datasets (detail)

- **TruthfulQA (`multiple_choice` config)** -- for each question we pair
  the correct choice with one randomly-sampled incorrect choice.
- **HaluEval (QA split)** -- each row supplies `right_answer` and
  `hallucinated_answer`; we keep both and label accordingly.
- **FEVER** -- claims labeled `SUPPORTS` (1) or `REFUTES` (0); NEI
  examples are dropped to preserve binary clarity.
- **Synthetic** -- paired true/false statements about world capitals,
  chemical symbols, literary authorship, and planetary orbits. Two
  items share the same prompt, differing only in `answer`, giving the
  cleanest possible probe signal.

### 4. Hidden-state extraction

For every `PromptItem` we tokenize the concatenation
`prompt + " " + answer`. A single forward pass with
`output_hidden_states=True` yields a tuple of `L + 1` tensors
(embedding layer plus each transformer block). For each layer we pool
across the *answer-span* positions and keep the result as a `[D]`
vector. The default pooling is `last_token` (the last answer token),
matching SAPLMA conventions; `mean` and `max` pooling are available.

When `capture_attention: true`, we additionally collect attention
matrices for a configurable subset of layers (typically first, middle,
last). We summarize each kept layer by computing the entropy of the
last-answer-token row's attention distribution per head -- a `[H]`
vector per sample per layer.

Outputs per (model, dataset) are persisted in a single PyTorch file
`<model>__<dataset>.pt` containing the stacked `[N, L+1, D]` hidden
states, labels, prompts, answers, and optional attention summaries.

### 5. Detection methods (detail)

#### 5.1 SAPLMA-style probes
Per layer, a linear or MLP probe is trained to predict the binary label
from the layer's hidden vector. We sweep all layers, run `n_seeds`
splits, and report mean AUROC +/- std. The peak layer and peak AUROC
provide a model's overall probing score.

#### 5.2 INSIDE / Semantic entropy
For each prompt we sample `K` stochastic continuations and capture the
last generated token's hidden state at the target layer for each
sample. The K x D matrix's *EigenScore* -- the log-determinant of its
regularized covariance -- is used as a hallucination score: high
EigenScore => high semantic dispersion => likely hallucination.

#### 5.3 Self-consistency
For each prompt we draw `K` stochastic completions and measure their
agreement via (a) exact-match plurality and (b) mean pairwise
sentence-embedding similarity (`all-MiniLM-L6-v2`). Hallucination
score = 1 - consistency.

#### 5.4 Attention entropy
The head-averaged entropy of the last answer token's attention
distribution serves as a complementary signal; we report AUROC per
kept layer.

### 6. Metrics

Continuous scores (INSIDE, self-consistency, attention entropy) are
evaluated with AUROC, AUPRC, and accuracy / F1 at the Youden-optimal
threshold. Probes are evaluated on a stratified test split (default
20 %); a 10 % validation split selects the best epoch's weights.

### 7. Layer geometry

Independent of any classifier we report three geometric quantities
per layer: the L2 distance between class centroids, the within-class
spread (mean L2 distance from centroid), and the *separation ratio*
(centroid distance / mean spread). This gives a parametric, model-free
view of how the truthful and hallucinated clouds drift apart through
the network depth.

## Running the pipeline

### Full pipeline (one command)

```text
python scripts/run_pipeline.py
```

This invokes every stage of the pipeline in order, using the same
Python interpreter you launched it with -- so the active `swarm`
environment is honoured automatically on every OS. Stage 08
(visualization) is always invoked as the final stage, so a fresh
end-to-end run regenerates every figure -- no extra command needed.

### Individual stages

Inside `swarm`, any stage can be invoked on its own. Every stage is
idempotent -- it skips outputs that already exist.

```text
python scripts/00_check_install.py
python scripts/01_download_data.py
python scripts/02_extract_hidden_states.py
python scripts/03_train_probes.py
python scripts/04_run_inside.py
python scripts/05_run_self_consistency.py
python scripts/06_analyze_attention.py
python scripts/07_aggregate_results.py
python scripts/08_generate_visualizations.py
```

### Visualization-only mode (regenerate figures from existing results)

Stage 08 (`scripts/08_generate_visualizations.py`) is fully decoupled
from the rest of the pipeline: it consumes the on-disk outputs of the
earlier stages (`results/hidden_states/*.pt`, `results/probes/*.json`,
`results/metrics/*.json`, `results/tables/all_results.csv`) and writes
every figure to `results/figures/`. This means you can regenerate the
entire figure set with a single command, without re-running any of the
heavy stages:

```text
# Run the visualization stage on its own (uses existing results)
python scripts/08_generate_visualizations.py
```

The same effect via the cross-platform runner (also a single command):

```text
python scripts/run_pipeline.py --only 08 --skip-preflight
```

By default every figure is written as zero-margin vector PDF. Useful
flags on stage 08:

```text
# Override the output format (default: pdf; from viz.figure_format in config)
python scripts/08_generate_visualizations.py --figure-format pdf
python scripts/08_generate_visualizations.py --figure-format png

# Write to a different directory (e.g. for an alternate figure set)
python scripts/08_generate_visualizations.py --figures-dir results/figures_v2

# Override the savefig DPI (only matters for raster formats like PNG)
python scripts/08_generate_visualizations.py --dpi 400
```

When you instead run the full pipeline (`python scripts/run_pipeline.py`),
stage 08 is invoked automatically as the final stage, so figures are
always regenerated end-to-end on a fresh run.

### Resuming after a failure

If the pipeline fails partway through, you don't need to re-run
everything from the beginning. Use `--from` to resume at the first
failed stage (all earlier stages' outputs are already on disk):

```text
# Resume from stage 05 onward (skips stages 01-04 that already succeeded)
python scripts/run_pipeline.py --from 05

# Re-run only the specific stages that failed
python scripts/run_pipeline.py --only 05 08
```

### Useful flags for the runner

```text
python scripts/run_pipeline.py --check-env       # only the pre-flight, then exit
python scripts/run_pipeline.py --only 03 08      # run only stages 03 and 08
python scripts/run_pipeline.py --skip 04 05      # skip stages 04 and 05
python scripts/run_pipeline.py --from 03         # start at stage 03 (resume)
python scripts/run_pipeline.py --dry-run         # print the plan, run nothing
python scripts/run_pipeline.py --stop-on-error   # halt on first failed stage
python scripts/run_pipeline.py --skip-preflight  # skip pre-flight (not advised)
```

The runner uses `sys.executable` to invoke every stage, so whatever
Python interpreter you launched it with is the interpreter every stage
uses -- guaranteeing the `swarm` environment is honoured on every OS.

## What you get

After the pipeline finishes:

- `results/hidden_states/*.pt` -- per-(model, dataset) extracted hidden
  states and labels.
- `results/probes/*.json` -- trained probes' per-layer metrics
  (mean +/- std across seeds).
- `results/metrics/*.json` -- AUROC, F1, accuracy per (model, layer,
  dataset, method) for INSIDE / self-consistency / attention.
- `results/figures/*.pdf` -- t-SNE/PCA projections, layer-wise AUROC
  curves and heatmaps, class-separation plots, attention-entropy
  histograms, score distributions, and per-dataset method-comparison
  charts. Every figure is saved as zero-margin vector PDF. The format
  is configurable via `viz.figure_format` in `config/config.yaml` or
  the `--figure-format` flag on stage 08 (see
  [Visualization-only mode](#visualization-only-mode-regenerate-figures-from-existing-results)).
- `results/tables/all_results.csv` -- every (model, dataset, method,
  layer) row in a single tidy table.
- `results/tables/best_per_method.csv` -- best layer per method.
- `results/tables/summary.md` -- markdown executive summary.

## Output organization

```
results/
├── hidden_states/   # *.pt, [N, L+1, D] hidden tensors per (model, dataset)
├── probes/          # *.json, layer-wise probe metrics
├── metrics/         # *.json, INSIDE / self-consistency / attention metrics
├── tables/          # *.csv, *.md, aggregated results tables
├── figures/         # *.pdf (vector, zero-margin), all visualizations
└── logs/            # *.log, per-script run logs
```

## Reduce wall-clock time

Open `config/config.yaml` and adjust any of the following:

- Lower `datasets.<name>.n_samples` (e.g. 100) for a faster smoke test.
- Lower `inside.n_samples` and `self_consistency.n_samples` to 3.
- Disable attention capture (`extraction.capture_attention: false`).
- Set `models` to a single entry to test the pipeline end-to-end.

## Add a new model

Append a block under `models:` in `config/config.yaml`:

```yaml
- short_name: phi-3.5-mini
  hf_id: microsoft/Phi-3.5-mini-instruct
  family: phi
  chat_template: llama3
```

The loader, extractor, and downstream pipeline will pick it up on the
next run.

## Tests

Lightweight unit tests live in `tests/`:

```text
python -m pytest -q
```

These cover the synthetic data generator, INSIDE math, and probe metric
plumbing -- they do not require a GPU.

## Reproducibility

- All random seeds are set deterministically in `config/config.yaml`
  and applied via `src.utils.set_seed` at the top of every script.
- Model loading, tokenization, sampling temperatures, batch sizes, and
  sample counts are versioned in the same config.
- Datasets are cached in `data/cache/`; processed JSONL items live in
  `data/processed/`.
- Hidden states and probes are versioned by file name
  (`<model>__<dataset>__<probe>.json`), so partial reruns are safe.
- The pipeline is **idempotent** -- re-running it skips stages whose
  outputs already exist. Empty / corrupted output files are detected
  and re-attempted on the next run.

## Repository layout

```
HallucinationPatternDetection/
├── README.md                         # this file (single source of truth)
├── requirements.txt
├── environment.yml
├── config/
│   └── config.yaml
├── src/
│   ├── data/                         # dataset loaders + synthetic prompt generator
│   ├── models/                       # quantized model loading + hidden-state extraction
│   ├── detection/                    # probes, INSIDE, SAPLMA, self-consistency
│   ├── analysis/                     # cross-model metrics, layer analysis
│   ├── visualization/                # t-SNE/UMAP, heatmaps, attention plots
│   │   ├── style.py                  # shared figure style + zero-margin savefig helper
│   │   ├── embed_viz.py              # t-SNE / UMAP / PCA scatter
│   │   ├── layer_viz.py              # layer-wise AUROC curves + heatmaps + separation
│   │   ├── attention_viz.py          # attention entropy histograms + heatmaps
│   │   └── score_viz.py              # score distributions, ROC, method comparison
│   └── utils/
├── scripts/
│   ├── 00_check_install.py           # pre-flight environment check
│   ├── 01_download_data.py
│   ├── 02_extract_hidden_states.py
│   ├── 03_train_probes.py
│   ├── 04_run_inside.py
│   ├── 05_run_self_consistency.py
│   ├── 06_analyze_attention.py
│   ├── 07_aggregate_results.py
│   ├── 08_generate_visualizations.py # standalone-runnable, PDF-by-default
│   └── run_pipeline.py               # cross-platform Python runner (auto-invokes stage 08)
├── data/
├── results/
│   ├── figures/                      # *.pdf, vector, zero-margin
│   ├── metrics/
│   ├── tables/
│   ├── hidden_states/
│   ├── probes/
│   └── logs/
└── tests/
```


## Citation


If you find this work useful in your research, please cite it:

```bibtex
@article{aiersilan2026hallucination,
  title={Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs},
  author={Aiersilan, Aizierjiang},
  journal={arXiv preprint arXiv:2606.02628},
  year={2026}
}

```



## License

This project is released under the [MIT License](LICENSE).
