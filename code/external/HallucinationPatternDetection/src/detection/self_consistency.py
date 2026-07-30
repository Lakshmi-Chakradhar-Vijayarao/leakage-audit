"""Self-consistency hallucination scoring.

Idea: a confident, well-known fact yields highly consistent answers
across stochastic decoding samples; hallucinated answers diverge.
We sample K completions for each prompt and score the agreement among
them using

  (a) exact-match consistency, and
  (b) semantic-similarity consistency via sentence-transformers
      embeddings.

`run_self_consistency` is the high-level driver: given a ModelBundle
and a list of PromptItem, it generates K samples per prompt, computes
both scores, and returns a structured dict.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset_loader import PromptItem
from src.models.model_loader import ModelBundle


# ------------------------ generation helper ----------------------------
def _generate_samples(
    bundle: ModelBundle,
    prompt: str,
    n_samples: int,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    tok = bundle.tokenizer
    enc = tok(prompt, return_tensors="pt").to(bundle.device)
    out_texts: List[str] = []
    with torch.inference_mode():
        out = bundle.model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    for seq in out:
        gen = seq[prompt_len:]
        out_texts.append(tok.decode(gen, skip_special_tokens=True).strip())
    return out_texts


# ------------------------ scoring helpers ------------------------------
def _exact_match_consistency(samples: Sequence[str]) -> float:
    """Plurality fraction: largest equivalence class / K (normalized strings)."""
    if not samples:
        return 0.0
    norm = [s.strip().lower() for s in samples]
    counts: Dict[str, int] = {}
    for s in norm:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) / len(samples)


def _semantic_consistency(samples: Sequence[str], embedder) -> Tuple[float, float]:
    """Returns (mean_pairwise_similarity, fraction_above_threshold)."""
    if not samples:
        return 0.0, 0.0
    emb = embedder.encode(list(samples), convert_to_numpy=True, normalize_embeddings=True)
    # pairwise cosine = matrix product since rows are normalized
    sims = emb @ emb.T
    K = sims.shape[0]
    if K < 2:
        return 1.0, 1.0
    mask = ~np.eye(K, dtype=bool)
    mean = float(sims[mask].mean())
    return mean, float((sims[mask] > 0.7).mean())


# ------------------------ public API -----------------------------------
def self_consistency_score(
    samples: Sequence[str], embedder=None
) -> Dict[str, float]:
    """Compute scalar consistency scores for a single sample set."""
    exact = _exact_match_consistency(samples)
    sem_mean, sem_frac = (math.nan, math.nan)
    if embedder is not None:
        sem_mean, sem_frac = _semantic_consistency(samples, embedder)
    return {
        "exact_match": exact,
        "semantic_mean": sem_mean,
        "semantic_frac_high": sem_frac,
    }


def run_self_consistency(
    bundle: ModelBundle,
    items: Sequence[PromptItem],
    n_samples: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Generate K samples per prompt and compute consistency scores.

    A *low* consistency score is the hallucination signal:
      hallucination_score = 1 - consistency.
    """
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding_model)
    except Exception:
        embedder = None

    records: List[Dict[str, Any]] = []
    for it in tqdm(items, desc=f"self-consistency:{bundle.short_name}"):
        samples = _generate_samples(
            bundle, it.prompt,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
        scores = self_consistency_score(samples, embedder=embedder)
        records.append({
            "prompt": it.prompt,
            "answer": it.answer,
            "label": it.label,
            "dataset": it.dataset,
            "samples": samples,
            **scores,
            "hallucination_score_exact": 1.0 - scores["exact_match"],
            "hallucination_score_semantic": (
                1.0 - scores["semantic_mean"]
                if not (isinstance(scores["semantic_mean"], float) and math.isnan(scores["semantic_mean"]))
                else math.nan
            ),
        })

    return {
        "model_short_name": bundle.short_name,
        "n_samples_per_prompt": n_samples,
        "records": records,
    }
