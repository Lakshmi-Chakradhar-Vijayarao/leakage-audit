# Case Study 3 — External audit target: MultiHaluDet (arXiv 2605.24919)

## What we checked, and how

MultiHaluDet ("Multilingual Hallucination Detection via LLM Hidden State
Probing," MeLLM Workshop @ ACL 2026) has a fully public, re-runnable
pipeline: https://github.com/alvi-uiu/MultiHaluDet. It reports up to
98.55% AUROC detecting hallucination from Mistral-7B / LLaMA-2-7B hidden
states on HaluEval, via a 4-stage pipeline (feature extraction -> per-fold
deep-model training -> out-of-fold feature generation -> log-odds ensemble
stacking).

We read the released training code directly (`run_pipeline.py::stage_2_3_train_oof`,
`src/training/trainer.py::train_deep_model_fold`) rather than relying on
the paper's prose description of its methodology.

## The mechanism, verified in code

Inside each of 5 inner CV folds:

```python
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_seq_train, y_train)):
    model, _ = train_deep_model_fold(
        X_seq_train[tr_idx], ..., y_train[tr_idx],       # gradient-descent training data
        X_seq_train[val_idx], ..., y_train[val_idx],     # <- also the checkpoint-selection split
        config,
    )
    feats_val = extract_features_batch(model, X_seq_train[val_idx], ...)
    oof_features[val_idx] = feats_val   # <- these become the "held-out" fold's features
```

and inside `train_deep_model_fold`, every epoch is scored on that same
`val_idx` split, and whichever epoch's checkpoint maximizes AUROC there is
the one kept (`best_model = copy.deepcopy(model.state_dict())` when
`auc_score > best_auc`).

This means the "out-of-fold" features stored for `val_idx` come from a
model that was explicitly selected because it scores well on `val_idx`'s
own labels. The model's *weights* never see `val_idx` via gradient descent
-- so this is not the classic "trained on the test set" bug -- but the
*checkpoint choice* is a function of `val_idx`'s labels, and that choice
directly determines the features later fed, unmodified, to the downstream
meta-learner as if they were clean OOF features. This is a distinct
sibling of HaRP's bug (Case Study 1): not full-dataset leakage, but
**per-fold checkpoint-selection leakage**.

## Quantifying it: controlled synthetic reconstruction

Re-running MultiHaluDet's actual pipeline requires a 7B-parameter model on
real HaluEval/TriviaQA data -- not something this project's compute budget
(Kaggle GPU, already committed to Paper 3) can spend on a secondary
replication. Instead we isolated the *mechanism* in a controlled synthetic
setting that mirrors their training loop exactly (same 5-fold structure,
same "keep the best-val-AUC checkpoint" logic, same downstream-classifier-
on-OOF-features structure), with one addition: a `CLEAN` control condition
where checkpoint selection uses a separate carve-out from the training
fold, never touching `val_idx`'s labels at all.

Task difficulty was calibrated using **GEOM-PROOF's own closed-form result**
from earlier in this project's arc -- the binormal AUROC identity
AUROC = Phi(sqrt(J/2)), where J is the Fisher discriminant ratio -- inverted
to target a realistic, non-saturated AUROC of 0.80 (matching the
0.775-0.804 range seen in HaRP and GUARDIAN), rather than guessing a
class-separation constant. **Correction, added after third-round review:
the scripts originally used to produce the numbers below inverted
Phi(sqrt(J)/2) instead -- the equal-prior Bayes-accuracy formula, not
this AUROC identity -- so the "calibrated-to-0.80" task was actually an
0.883-AUROC task; this is fixed in `02d_corrected_capacity_placebo_sweep.py`
and all numbers below use the corrected inversion, J = 2*Phi^-1(0.80)^2.**
Our first attempt at this synthetic task used an arbitrary separation and
produced a saturated AUROC=1.0000 for both conditions -- a real
methodological trap: a task with no room to fail leaves no room for a
leakage bug to show any inflation either. Recalibrating via the Fisher/AUROC
identity fixed this.

**Result (20 seeds, N=700, 5-fold, hidden=48):**

| Condition | Mean AUROC | Std |
|---|---|---|
| LEAKY (checkpoint selected on val_idx) | 0.8708 | 0.0298 |
| CLEAN (checkpoint selected on a disjoint carve-out) | 0.8689 | 0.0308 |
| Gap (leaky − clean) | **+0.0019** | 0.0057 |

Wilcoxon signed-rank test on the paired per-seed gap: p=0.170 (not
significant at the standard 0.05 threshold). Positive gap in 12/20 seeds.

**Capacity-sweep follow-up, corrected (four rounds of review,
`02d_corrected_capacity_placebo_sweep.py`):** the obvious objection to
reporting a single number from an arbitrarily-sized 48-unit MLP is that
MultiHaluDet's actual model uses `hidden_dim=384` -- 8x larger. An
earlier version of this case study reran at 16/48/128/384 units with 10
seeds and reported a growing, individually-non-significant gap; a
follow-up n=100 single-capacity rerun then reported the flagship-capacity
gap as significant and, via a permuted-label placebo, "genuine peeking, not
capacity variance." A third round of review found two further problems:
the task-difficulty calibration inverted the wrong AUROC identity
(actual task AUROC ~0.883, not the intended 0.80), and LEAKY trained on
~15% more data than CLEAN (an unmatched training budget, not just a
different selection rule). Correcting both and adding a budget-matched
`CLEAN_MATCHED` condition initially showed no significant residual at any
capacity -- **but a fourth round of review found that fix had itself
introduced a new confound: `CLEAN_MATCHED` was retrained with a
different random seed than `LEAKY`, adding independent noise that
diluted the comparison's power.** Matching the seed and rerunning the
full capacity sweep at n=100 seeds throughout gives the final result:

| Hidden | LEAKY | CLEAN | CLEAN_MATCHED | PLACEBO | Gap LEAKY−CLEAN (p) [confounded, retracted] | Gap LEAKY−CLEAN_MATCHED (p) |
|---|---|---|---|---|---|---|
| 16  | 0.7611 | 0.7567 | 0.7602 | 0.7272 | +0.0044 (0.0019) | +0.0009 (0.309) |
| 48  | 0.7647 | 0.7617 | 0.7631 | 0.7375 | +0.0030 (0.0058) | **+0.0017 (0.015)** |
| 128 | 0.7604 | 0.7555 | 0.7571 | 0.7309 | +0.0049 (0.0008) | **+0.0034 (0.008)** |
| **384 (matches MultiHaluDet's config)** | 0.7533 | 0.7474 | 0.7506 | 0.7350 | +0.0059 (0.0012) | +0.0027 (0.077) |

**The budget-confounded gap (LEAKY-CLEAN) is significant at all four
capacities but shows no clear capacity trend** (it dips at 48, then rises
-- not a monotonic growth curve); we retain this column only for
transparency and do not treat it as evidence of anything beyond "a
budget-confounded gap exists." **The properly seed-matched,
budget-matched gap (LEAKY-CLEAN_MATCHED) is significant, surviving
Holm-Bonferroni correction across the four capacities, at 2 of 4
capacities (48 and 128 units), not significant at 16 units (gap
$\approx$0, $p$=0.309, below this capacity's own minimum detectable
effect), and underpowered rather than null at 384 units ($p$=0.077, at
the edge of that capacity's minimum detectable effect).**
Decomposing the confounded gap: budget mismatch explains 80% of the
apparent effect at 16 units but only 31-54% at the other three
capacities, leaving a majority-share residual that reaches significance
at two of them. Our own pre-registered decision rule returns
\texttt{MIXED} at all four capacities (its \texttt{CONFOUND\_CONFIRMED}
branch is structurally unreachable, since LEAKY-vs-PLACEBO is always
significant at $p<10^{-8}$) -- we report that verdict rather than
translating it into a cleaner narrative.

## Why this matters for the paper's framing

This is a genuinely important, non-obvious finding, but not the one any
of our first three passes concluded. **We withdraw both the original
claim that "this mechanism's severity scales with model capacity" and
the intermediate, seed-confounded claim that the effect was "possibly
entirely a training-budget artifact."** The final data show a small,
real, inconsistently-significant-across-capacity residual leak -- not a
capacity trend, and not a null. HaRP's own +0.19 AUROC
catastrophic inflation from full-dataset fit-then-score leakage (Case
Study 1) remains severe regardless of model size and is unaffected by any
of this section's corrections -- it is a structurally different mechanism
(the model literally trains on the labels it is later scored against),
not a checkpoint-selection subtlety.

This still does **not** mean MultiHaluDet's reported 98.55% AUROC's exact
inflation is now known -- even our largest-capacity reconstruction (384
units) is a single MLP, far simpler than their actual architecture (a
6-layer, 384-hidden-dim, multi-scale transformer with mixup/cutmix/EMA/SWA
and heavy augmentation). Our small, capacity-inconsistent residual is a
lower bound at best on what a more expressive, longer-trained real
pipeline might show; we flag this honestly as an open question rather
than resolving it without access to their exact compute budget.

**The paper's actual contribution here is the taxonomy and the checklist,
not a single severity verdict for Case Study 3**: leakage in this
literature is not one uniform failure mode, and quantifying any one
mechanism's severity via synthetic reconstruction is itself failure-prone
in ways this paper's own four-round correction history now documents
directly. **Stated explicitly (round-5 review): the +0.19 (Case Study 1)
and +0.0009-to-+0.0034 (Case Study 3) numbers below are not measurements
on the same scale and must not be read as "mechanism 3 is roughly
20-200x milder than mechanism 1." The first is a real-hidden-state
effect size; the second is a linear-Gaussian synthetic-proxy effect
size. The only claim the comparison licenses is that mechanism (1)'s
severity does not depend on the measurement substrate being real vs.
synthetic (it is catastrophic either way, by construction: the model
trains on its own scoring labels), while mechanism (3)'s severity on
real MultiHaluDet geometry is simply unmeasured.** There is a real
severity spectrum for the mechanisms
themselves --

1. **Severe, scale-independent** (Case Study 1, HaRP): a feature-generating
   model fit on *all* labels, scored on those same labels, feeding a
   downstream CV loop -- +0.19 AUROC, unambiguous, large, and not a
   function of model capacity.
2. **Structurally real, code-verified, small and capacity-inconsistent**
   (Case Study 3, MultiHaluDet): checkpoint selection using the same fold
   whose features get reused downstream -- a genuine, code-verified leak
   channel whose magnitude, in our synthetic reconstruction, is small
   ($+0.0009$ to $+0.0034$ AUROC) and reaches significance at only 2 of 4
   capacities tested, once training-budget and random-seed confounds are
   both properly controlled for.
3. **A related but distinct bias family entirely** (test-set-driven
   best-layer selection with no correction for having tried ~30
   hypotheses -- see the quantized-LLM paper secondary case study) --
   not "nested CV leakage" in the classical sense at all, but the same
   broad hazard (using labels to make a choice, then reporting performance
   as if that choice were free).

This taxonomy -- not a blanket "everyone's numbers are inflated" claim --
is the paper's real methodological contribution: a checklist for
recognizing which of several distinct label-information-leak patterns a
given pipeline is vulnerable to, and honest evidence about which ones are
likely to matter most in practice.
