# Leakage Audit

Code, results, and paper source for "Not All Leaks Are Equal: A Taxonomy and
Audit of Label-Information Leakage in Hidden-State Hallucination Detection."

## What's included
Everything. Unlike the companion repos in this project, no files were excluded
here — this paper's own results never required large raw caches.
- `code/` — all audit scripts, including the regex-based leakage linter and the
  capacity-128/384 real-feature reruns.
- `results/` — every result JSON reported in the paper, including the full
  four-round correction history for Case Study 3 (see the paper's Appendix A).
- `draft/` — the paper source.
- `kaggle_kernels/`, `related_work/`, `requirements.txt`.

## Note on external case studies
Case Studies 3 and 4 audit two externally published, third-party repositories
at pinned commits (linked directly in the paper). This repository contains only
this project's own audit scripts and results, not copies of those third-party
codebases — see the paper's §4 and §6 for the exact pinned-commit references.
