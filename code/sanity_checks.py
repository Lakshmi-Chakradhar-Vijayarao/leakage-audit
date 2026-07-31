"""
Shared guardrail, added after an independent adversarial review found that
code/46 and code/47 silently quadrupled the realized Fisher J relative to
their labeled target_auroc (the exact class of calibration bug Appendix A
already documents finding and fixing once in code/02d's inverted formula).
code/02d had a per-run "calibration_sanity_check_auroc" field, but on
inspection it turned out to be `norm.cdf(sqrt(FISHER_J/2))` recomputed from
the *same* FISHER_J used to build the data -- a tautological restatement of
the analytic target, not an empirical measurement. It would not have caught
this bug class either. This module is the real, non-tautological check:
every synthetic generator in this project must call assert_calibration()
before returning data.

Deliberately NOT implemented as "fit a classifier and score its CV AUROC":
a standard LogisticRegression's L2 regularization systematically shrinks
the achieved separation relative to the isotropic-Gaussian Bayes-optimal
rate at these dimensions/sample sizes (verified: ~0.04-0.05 AUROC undershoot
at d=64, n=700, consistent across three target AUROCs, far exceeding
seed-to-seed noise). That gap is a real, already-documented property of
this paper's own severity harness (Appendix A's capacity-128 vs.
0.80-ceiling disclosure is the same phenomenon), not a bug -- gating on it
would reject correct generators.

Also deliberately NOT gating in AUROC-space on the realized Fisher J
(tried first): AUROC = Phi(sqrt(J/2)) compresses near the ceiling, so a
single AUROC-space tolerance can't simultaneously (a) tolerate normal
sampling noise in the realized J estimate at low target_auroc and (b)
still catch the bug at high target_auroc, where both buggy and correct J
map to AUROC very close to 1. Verified numerically: at target=0.985 the
bug's realized AUROC deviates from target by only ~0.015, smaller than the
~0.03-0.07 sampling noise band needed to avoid false positives at
target=0.70.

Instead this checks the ratio realized_J / target_J directly, which is
scale-invariant: the bug is a fixed, structural 4x multiplier on J (using
class_sep instead of class_sep/2 doubles the per-dimension mean gap, and
squaring doubles again), independent of target_auroc. Verified numerically
across target_auroc in {0.70, 0.80, 0.90, 0.95, 0.985}, 2000 seeds each,
FEAT_DIM=64, N_SAMPLES=700: correct generators realize a ratio in
[0.33, 1.83]; the bug never drops below 2.77 -- clean separation.

Known limitation: at smaller n_samples (code/47's Sweep B goes down to
n_samples=350, i.e. n_pos=n_neg=175), per-dimension sampling noise grows
relative to a low target_auroc's small true J, and the fixed-vs-buggy
ratio distributions start to overlap (verified: at n_samples=350,
target_auroc=0.70, fixed-data ratio can reach 2.32 while buggy-data ratio
can be as low as 2.03 -- no threshold cleanly separates both). Bounds are
set to [0.2, 2.6] to prioritize never rejecting correct data (verified
safe up to n_samples=350) at the cost of reduced sensitivity to a 4x-style
bug specifically in that small-n_val, low-target_auroc regime -- an
inherent SNR limit, not a defect in this check's logic. It still reliably
catches the bug at n_samples>=700, which covers every cell except Sweep B.
"""
import numpy as np
from scipy.stats import norm


def assert_calibration(X, y, target_auroc, ratio_bounds=(0.2, 2.6)):
    """Raise if the realized Fisher J of freshly-generated synthetic data
    (measured directly from the empirical class-mean difference, not via a
    fitted classifier, and bias-corrected for finite-sample noise) falls
    outside ratio_bounds of the J implied by target_auroc. Deliberately a
    coarse, fast check meant to catch gross errors like a missing/extra
    factor of 2 in a class-mean offset, not to replace proper calibration
    search.

    The raw sum-of-squared-differences ||mean_pos - mean_neg||^2 is
    upward-biased at finite n: each of FEAT_DIM per-dimension mean
    estimates carries sampling noise of variance (1/n_pos + 1/n_neg), and
    squaring a noisy estimate inflates its expectation by that noise
    variance, summed over all dimensions. Subtracting this known bias term
    recovers an unbiased Fisher J estimate."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    mean_pos = X[y == 1].mean(axis=0)
    mean_neg = X[y == 0].mean(axis=0)
    feat_dim = X.shape[1]
    raw_j = float(np.sum((mean_pos - mean_neg) ** 2))
    bias = feat_dim * (1.0 / n_pos + 1.0 / n_neg)
    realized_j = max(raw_j - bias, 0.0)
    target_j = 2 * (norm.ppf(target_auroc)) ** 2
    ratio = realized_j / target_j if target_j > 0 else float("inf")
    lo, hi = ratio_bounds
    if not (lo <= ratio <= hi):
        realized_auroc = float(norm.cdf(np.sqrt(realized_j / 2)))
        raise AssertionError(
            f"Calibration check FAILED: generated data has realized Fisher J "
            f"= {realized_j:.4f} vs. target J = {target_j:.4f} (ratio "
            f"{ratio:.3f}, allowed [{lo},{hi}]) -- implied Bayes-optimal "
            f"AUROC {realized_auroc:.4f} vs. target {target_auroc:.4f}. "
            f"This generator's class-mean offset formula is very likely "
            f"wrong (check for a missing /2, a squared vs. non-squared "
            f"term, etc.)."
        )
    return float(norm.cdf(np.sqrt(realized_j / 2)))
