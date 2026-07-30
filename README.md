# Leakage Audit

Code, results, and paper source for "Not All Leaks Are Equal: A Taxonomy and
Audit of Label-Information Leakage in Hidden-State Hallucination Detection."

## What's included
Everything, including the vendored third-party repositories the paper audits.
- `code/` — all audit scripts, including the regex-based leakage linter, the
  capacity-128/384 real-feature reruns, the calibration-leakage
  diagnostic that found and fixed a split-crossing leakage bug in this
  paper's own severity-measurement instrument (`code/43`), a statistical-rigor
  retrofit adding BCa bootstrap CIs and paired permutation tests to every
  reported severity gap (`code/44`), a winner's-curse quantification for
  Case Study 4 (`code/45`), a quantification of a fifth leakage mechanism
  found while auditing Case Study 3's own repository -- test-set-driven
  decision-threshold selection (`code/46`) -- and a K/validation-set-size/
  operating-point sweep testing whether severity scales the way
  extreme-value theory predicts (`code/47`).
- `code/external/` — the two externally published, third-party repositories
  Case Studies 3 and 4 audit (MultiHaluDet, HallucinationPatternDetection),
  vendored at the pinned commits cited in the paper (§4, §6), plus five
  additional repositories scanned by the leakage linter (§5). Each retains
  its own license; see the paper for exact commit hashes and links to the
  original repositories.
- `results/` — every result JSON reported in the paper, including the full
  correction history for Case Study 3 (see the paper's Appendix A), the
  Mechanism 5 and K/n_val/operating-point sweep results, and the
  real Mistral-7B/HaluEval feature cache (`real_features_mistral7b_halueval.npz`)
  needed to re-run `code/27`, `code/31`, `code/33`, and `code/43`.
- `draft/` — the paper source.
- `kaggle_kernels/`, `related_work/`, `requirements.txt`.
