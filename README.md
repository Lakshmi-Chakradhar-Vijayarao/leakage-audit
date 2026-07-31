# Leakage Audit

Code, results, and paper source for "Selection-Induced Optimism in
Hidden-State Hallucination Detection: A Code-Verified Taxonomy, and Why
Severity Tracks Candidate Count and Operating Point Rather Than Mechanism."

## Reproducibility status, up front

| Case study | Reproducible from this repo? |
|---|---|
| 3 (MultiHaluDet) | Yes — pinned third-party commit vendored, all scripts and result JSONs included |
| 4 (quantized-LLM paper) | Yes — pinned commit vendored; `code/45` recomputes the severity estimate from shipped per-layer result files |
| 2 (GUARDIAN) | Yes, from a derived artifact — the original 171 MB hidden-state cache is too large to ship, so `code/48` emits `results/case_study_2_probe_scores.npz` (~1 MB: per-layer, per-sample CV fold assignments and probe scores) and `code/51` recomputes and asserts every reported number from it |
| 1 (HaRP) | **No.** No script, log, or data file supporting its `+0.19` figure is included. The paper states this explicitly, excludes the number from its abstract severity range, and excludes it from every cross-mechanism comparison. |

## What's included

- `draft/` — the paper source: `paper_draft.md` (canonical, markdown mirror
  of the submission), plus supporting write-ups referenced from the paper:
  `worked_examples.md` (Case Studies 1-2, this paper's own pipelines),
  `case_study_multihaludet.md` and `case_study_quantized_llm_paper.md`
  (Case Studies 3-4, full code excerpts from the external targets), and
  `leakage_checklist.md` (the standalone five-question checklist, §5).
- `code/` — every audit script, numbered roughly in the chronological order
  they were written. **The numbering has gaps** (e.g. `02`→`19`, `33`→`43`):
  earlier-numbered scripts were superseded by later, corrected versions
  during the iterative-review process this paper's Appendix A documents in
  full; the gaps are not withheld files. Notable scripts: the regex-based
  leakage linter (`04`), the calibration-leakage diagnostic that found and
  fixed a split-crossing leakage bug in this paper's own severity-
  measurement instrument (`43`), a statistical-rigor retrofit adding BCa
  bootstrap CIs and paired permutation tests to every reported severity gap
  (`44`), a winner's-curse quantification for Case Study 4 (`45`), a
  quantification of a fifth leakage mechanism found while auditing Case
  Study 3's own repository (`46`), a K/validation-set-size/operating-point
  sweep testing whether severity scales the way extreme-value theory
  predicts (`47`), a full 32-layer decomposition of Case Study 2's
  selection-optimism gap (`48`), a fidelity extension modeling Case
  Study 3's actual LR-scheduler/early-stopping training loop rather than
  just its checkpoint selection (`49`), an honest re-fit of the K-scaling
  law that shows the extreme-value functional form does *not* fit once
  each cell's own measured sigma is used (`50`), a replay script that
  re-derives every Case Study 2 number from the small shipped derived
  artifact with no access to the 171 MB raw cache (`51`), and the
  LaTeX-to-markdown sync script that makes `draft/paper_draft.md` a
  mechanical function of `draft/latex/main.tex` so the two cannot drift
  (`52`).
- `code/external/` — the seven third-party repositories this paper audits
  or scans: `MultiHaluDet` and `HallucinationPatternDetection` (Case
  Studies 3-4, vendored at the pinned commits cited in the paper), plus
  `DetectLLMHallucination`, `hallucination_probes`, `halluscope`,
  `haloscope`, and `semantic-entropy-probes` (scanned by the leakage
  linter, §5). Each retains its own license; see the paper for exact
  commit hashes and links to the original repositories.
- `results/` — every result JSON reported in the paper, including the full
  correction history for Case Study 3 (paper's Appendix A), the Mechanism 5
  and K/n_val/operating-point sweep results, and the real Mistral-7B/
  HaluEval feature cache (`real_features_mistral7b_halueval.npz`) needed to
  re-run `code/27`, `code/31`, `code/33`, and `code/43`.
- `kaggle_kernels/` — the one script in this paper that needed GPU
  (`paper2-real-feature-v2`, AWQ-quantized Mistral-7B feature extraction),
  plus its Kaggle run output: the result JSON and a cleaned, human-readable
  transcript of the run (`.txt` — the raw Kaggle log is a JSON event stream
  with tqdm/pip noise; this is the same content with that noise stripped).
- `requirements.txt` — exact library versions used to produce this paper's
  results (see the file's header comment for what's locally-pinned vs.
  Kaggle-only).
