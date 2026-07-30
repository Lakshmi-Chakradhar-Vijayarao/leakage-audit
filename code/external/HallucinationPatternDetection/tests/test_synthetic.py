"""Sanity tests that don't require a GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.prompt_generator import generate_synthetic_dataset
from src.detection.inside_method import compute_inside_score, inside_batch
from src.detection.probes import train_probe
from src.analysis.metrics import binary_metrics, score_to_metrics


def test_synthetic_dataset_balanced():
    items = generate_synthetic_dataset(n_samples=100, seed=0)
    assert len(items) == 100
    n_pos = sum(1 for it in items if it.label == 1)
    n_neg = sum(1 for it in items if it.label == 0)
    # close to balanced
    assert abs(n_pos - n_neg) <= 10
    # prompts and answers are non-empty
    for it in items:
        assert it.prompt and it.answer
        assert it.label in (0, 1)
        assert it.dataset == "synthetic"


def test_inside_score_is_higher_for_more_diverse_samples():
    rng = np.random.default_rng(0)
    tight = rng.normal(size=(10, 16)) * 0.01
    spread = rng.normal(size=(10, 16)) * 1.0
    s_tight = compute_inside_score(tight, alpha=1e-3, mode="eigen")
    s_spread = compute_inside_score(spread, alpha=1e-3, mode="eigen")
    assert s_spread > s_tight


def test_inside_batch_shape():
    rng = np.random.default_rng(0)
    batch = [rng.normal(size=(5, 8)) for _ in range(7)]
    res = inside_batch(batch)
    assert res.shape == (7,)


def test_probe_separates_synthetic_clusters():
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=+2, size=(80, 16))
    X_neg = rng.normal(loc=-2, size=(80, 16))
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * 80 + [0] * 80)
    _, m = train_probe(X, y, probe_type="linear", epochs=10, seed=0)
    # well-separated Gaussians should be near-perfectly probed
    assert m.auroc > 0.95


def test_binary_metrics_keys():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.6, 0.3])
    out = binary_metrics(y, p)
    assert {"accuracy", "f1", "auroc", "auprc"} <= set(out.keys())


def test_score_to_metrics_invertible():
    # constructed so that score is exactly the hallucination indicator
    y = np.array([1, 0, 1, 0, 1, 0])
    score = (1 - y).astype(float) + np.random.default_rng(0).normal(scale=0.01, size=y.shape)
    m = score_to_metrics(y, score)
    assert m["auroc"] > 0.95
