"""Dataset loaders.

Each loader returns a list[PromptItem]. A PromptItem holds:
  - prompt       : the input string we will feed the LLM
  - answer       : the answer string to attach when extracting hidden
                   states for the *answer* token (used by SAPLMA-style
                   probes — we extract the hidden state of the *last
                   answer token*).
  - label        : 1 if the (prompt, answer) pair is TRUTHFUL,
                   0 if it is a HALLUCINATION / FALSE.
  - dataset      : name of the source dataset (str).
  - meta         : optional dict of source-specific metadata.

For every benchmark we construct *paired* truthful / hallucinated
examples for the same underlying question whenever possible. Balanced
classes make probing and AUROC meaningful.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import load_dataset


# ----------------------------- types -----------------------------------
@dataclass
class PromptItem:
    prompt: str
    answer: str
    label: int                  # 1 = truthful, 0 = hallucinated
    dataset: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- TruthfulQA -------------------------------
def load_truthfulqa(
    n_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[PromptItem]:
    """Load TruthfulQA in `multiple_choice` config."""
    rng = random.Random(seed)
    ds = load_dataset("truthful_qa", "multiple_choice", cache_dir=cache_dir)["validation"]
    items: List[PromptItem] = []
    for ex in ds:
        q = ex["question"]
        choices: List[str] = ex["mc1_targets"]["choices"]
        labels: List[int] = ex["mc1_targets"]["labels"]
        truth_idx = [i for i, l in enumerate(labels) if l == 1]
        false_idx = [i for i, l in enumerate(labels) if l == 0]
        if not truth_idx or not false_idx:
            continue
        t = choices[truth_idx[0]]
        f = choices[rng.choice(false_idx)]
        items.append(PromptItem(
            prompt=f"Question: {q}\nAnswer:",
            answer=t,
            label=1,
            dataset="truthfulqa",
            meta={"question": q, "kind": "truthful"},
        ))
        items.append(PromptItem(
            prompt=f"Question: {q}\nAnswer:",
            answer=f,
            label=0,
            dataset="truthfulqa",
            meta={"question": q, "kind": "hallucinated"},
        ))
    rng.shuffle(items)
    if n_samples is not None:
        items = items[:n_samples]
    return items


# ----------------------------- HaluEval --------------------------------
def load_halueval_qa(
    n_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[PromptItem]:
    """HaluEval QA split: pair right_answer (1) vs hallucinated_answer (0)."""
    rng = random.Random(seed)
    ds = load_dataset("pminervini/HaluEval", "qa", cache_dir=cache_dir)
    split_name = "data" if "data" in ds else list(ds.keys())[0]
    rows = ds[split_name]
    items: List[PromptItem] = []
    for ex in rows:
        q = ex.get("question") or ex.get("user_query") or ""
        k = ex.get("knowledge", "")
        right = ex.get("right_answer") or ex.get("answer", "")
        wrong = ex.get("hallucinated_answer", "")
        if not (q and right and wrong):
            continue
        prompt = (
            f"Knowledge: {k}\nQuestion: {q}\nAnswer:" if k else
            f"Question: {q}\nAnswer:"
        )
        items.append(PromptItem(
            prompt=prompt, answer=right, label=1,
            dataset="halueval_qa",
            meta={"question": q, "knowledge": k, "kind": "truthful"},
        ))
        items.append(PromptItem(
            prompt=prompt, answer=wrong, label=0,
            dataset="halueval_qa",
            meta={"question": q, "knowledge": k, "kind": "hallucinated"},
        ))
    rng.shuffle(items)
    if n_samples is not None:
        items = items[:n_samples]
    return items


# ----------------------------- FEVER -----------------------------------
# FEVER labels on the Hub come in several encodings. We translate any
# of them into 1 (SUPPORTS = truthful) or 0 (REFUTES = hallucinated).
# Anything we can't classify is dropped (NEI, unknowns).
_FEVER_STR_TO_BIN: Dict[str, int] = {
    "SUPPORTS": 1, "SUPPORTED": 1, "SUPPORT": 1, "TRUE": 1, "T": 1,
    "REFUTES": 0, "REFUTED": 0, "REFUTE": 0, "FALSE": 0, "F": 0,
}
_FEVER_INT_TO_BIN: Dict[int, int] = {0: 1, 1: 0}   # 0=SUPPORTS, 1=REFUTES, 2=NEI


def _fever_normalize_label(raw: Any) -> Optional[int]:
    """Return 1 for SUPPORTS, 0 for REFUTES, None otherwise (skip)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return _FEVER_INT_TO_BIN.get(int(raw))
    s = str(raw).strip().upper()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return _FEVER_INT_TO_BIN.get(int(s))
    return _FEVER_STR_TO_BIN.get(s)


# Field names we'll probe in order to find the claim and label columns
# across the different FEVER mirrors on the Hub.
_FEVER_CLAIM_FIELDS = (
    "claim", "statement", "sentence", "text", "hypothesis",
)
_FEVER_LABEL_FIELDS = (
    "label", "label_id", "gold_label", "verifiable_label", "verifiable",
    "fever_label",
)


def _fever_extract_row(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Pull a (claim, normalized_label) out of any FEVER row shape."""
    claim = None
    for f in _FEVER_CLAIM_FIELDS:
        v = ex.get(f)
        if v:
            claim = v if isinstance(v, str) else str(v)
            break
    label = None
    for f in _FEVER_LABEL_FIELDS:
        if f in ex:
            label = _fever_normalize_label(ex[f])
            if label is not None:
                break
    return claim, label


def _items_from_split(rows, n_target: Optional[int]) -> List[PromptItem]:
    items: List[PromptItem] = []
    for ex in rows:
        # `datasets` Dataset rows behave like dicts; normalize to dict
        # via .get when available, else by item access.
        get = ex.get if hasattr(ex, "get") else (lambda k, d=None: ex[k] if k in ex else d)
        row_dict = {k: get(k) for k in _FEVER_CLAIM_FIELDS + _FEVER_LABEL_FIELDS}
        claim, bin_label = _fever_extract_row(row_dict)
        if not claim or bin_label is None:
            continue
        prompt = f"Claim: {claim}\nIs this statement true?"
        items.append(PromptItem(
            prompt=prompt, answer=claim, label=bin_label, dataset="fever",
            meta={"raw_label": row_dict},
        ))
        if n_target is not None and len(items) >= max(n_target * 4, n_target):
            break
    return items


def load_fever(
    n_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[PromptItem]:
    """FEVER fact-verification claims.

    Tries multiple HF mirrors and label/claim column variants. Falls
    through to the next mirror if the current one yields 0 usable rows.
    """
    rng = random.Random(seed)
    candidates: List[Tuple[str, Optional[str]]] = [
        ("pminervini/fever", "v1.0"),
        ("fever", "v1.0"),
        ("mwong/fever-evidence-related", None),
        ("copenlu/fever_gold_evidence", None),
    ]
    preferred_splits = ["paper_test", "validation", "valid", "test", "dev", "data", "train"]

    last_err: Optional[Exception] = None
    for repo, cfg in candidates:
        try:
            ds = (load_dataset(repo, cfg, cache_dir=cache_dir) if cfg
                  else load_dataset(repo, cache_dir=cache_dir))
        except Exception as e:
            last_err = e
            continue

        # try splits in order until we get usable items
        all_items: List[PromptItem] = []
        splits = [s for s in preferred_splits if s in ds] + [
            s for s in ds.keys() if s not in preferred_splits
        ]
        for split in splits:
            rows = ds[split]
            try:
                all_items = _items_from_split(rows, n_samples)
            except Exception:
                all_items = []
            if all_items:
                rng.shuffle(all_items)
                if n_samples is not None:
                    all_items = all_items[:n_samples]
                return all_items
        # else: this mirror yielded 0 items, try the next mirror

    raise RuntimeError(
        f"All FEVER mirrors yielded 0 usable rows. Last load error: {last_err}"
    )


# ----------------------------- registry --------------------------------
DATASET_LOADERS: Dict[str, Callable[..., List[PromptItem]]] = {
    "truthfulqa": load_truthfulqa,
    "halueval_qa": load_halueval_qa,
    "fever": load_fever,
}


# ----------------------------- IO helpers ------------------------------
def save_items_jsonl(items: List[PromptItem], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def load_items_jsonl(path: str | Path) -> List[PromptItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(PromptItem(**d))
    return items
