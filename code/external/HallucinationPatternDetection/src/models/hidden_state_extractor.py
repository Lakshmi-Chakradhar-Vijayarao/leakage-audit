"""Extract per-layer hidden states and (optionally) attention tensors.

For every (prompt, answer) PromptItem we tokenize the *concatenation*
`prompt + " " + answer`, forward through the model once, and read out
`outputs.hidden_states` (and optionally `outputs.attentions`).

The pooled representation per layer is the hidden vector at the
*last token of the answer span*; this matches SAPLMA. We additionally
provide `mean` and `max` pooling for robustness checks.

Outputs are saved as a single torch `.pt` file containing:
  {
    "hidden_states": Tensor[N, L+1, D]   # N items, L blocks + embed layer, D = hidden_size
    "labels": Tensor[N]                  # int64 (0/1)
    "prompts": List[str]
    "answers": List[str]
    "datasets": List[str]
    "model_short_name": str
    "pool": str
    "num_layers": int
    "hidden_size": int
    # only when capture_attention is enabled:
    "attention_entropy": Tensor[N, L_keep, H]   # per-head entropy of the
                                                # last answer token's row
    "attention_layers_kept": List[int]
  }
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.dataset_loader import PromptItem
from src.models.model_loader import ModelBundle


# ----------------------------- helpers ---------------------------------
def _resolve_layer_index(idx: Any, n_layers: int) -> int:
    """Resolve possibly-symbolic indices ("middle", -1) into ints."""
    if isinstance(idx, int):
        return idx if idx >= 0 else n_layers + idx
    if idx == "middle":
        return n_layers // 2
    return int(idx)


def _pool(hidden: torch.Tensor, answer_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool a [B, T, D] hidden tensor across the answer-span tokens.

    - last_token: returns the last True position in answer_mask
    - mean      : mean over answer_mask positions
    - max       : per-dim max over answer_mask positions
    """
    B, T, D = hidden.shape
    if mode == "last_token":
        last_idx = answer_mask.float().cumsum(dim=1).argmax(dim=1)   # [B]
        return hidden[torch.arange(B, device=hidden.device), last_idx]
    mask = answer_mask.unsqueeze(-1).to(hidden.dtype)   # [B, T, 1]
    if mode == "mean":
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom
    if mode == "max":
        neg_inf = torch.full_like(hidden, -1e9)
        masked = torch.where(mask > 0, hidden, neg_inf)
        return masked.max(dim=1).values
    raise ValueError(f"Unknown pool mode {mode}")


# ----------------------------- dataset ---------------------------------
class _PromptDataset(Dataset):
    def __init__(self, items: Sequence[PromptItem]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PromptItem:
        return self.items[idx]


def _collate(batch: List[PromptItem]) -> List[PromptItem]:
    return batch


# ----------------------------- extractor -------------------------------
@dataclass
class ExtractionConfig:
    pool: str = "last_token"
    capture_attention: bool = False
    attention_layers: Sequence[Any] = ()
    batch_size: int = 4
    max_input_length: int = 512


class HiddenStateExtractor:
    """Single-pass per-layer hidden-state extraction.

    Usage:
        ex = HiddenStateExtractor(bundle, ExtractionConfig(...))
        out = ex.extract(items)
        torch.save(out, "out.pt")
    """

    def __init__(self, bundle: ModelBundle, cfg: ExtractionConfig):
        self.bundle = bundle
        self.cfg = cfg

    # --------------- tokenization with answer-mask -----------------
    def _encode_batch(self, items: List[PromptItem]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        tok = self.bundle.tokenizer
        prompts = [it.prompt for it in items]
        answers = [it.answer for it in items]
        # tokenize prompt-only to learn its length, then prompt + answer
        full_texts = [p + " " + a for p, a in zip(prompts, answers)]
        enc = tok(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
            return_tensors="pt",
        )
        prompt_lens = [
            len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts
        ]
        # answer_mask = True for positions strictly after the prompt and
        # inside the (non-padding) attention region.
        T = enc["input_ids"].shape[1]
        am = torch.zeros((len(items), T), dtype=torch.bool)
        for i, plen in enumerate(prompt_lens):
            valid = enc["attention_mask"][i].bool()
            valid_len = int(valid.sum().item())
            start = min(plen, valid_len)
            am[i, start:valid_len] = True
            if not am[i].any():        # answer was truncated; fall back to last valid
                am[i, max(0, valid_len - 1)] = True
        return enc, am

    # --------------------- main extraction loop --------------------
    @torch.inference_mode()
    def extract(self, items: Sequence[PromptItem]) -> Dict[str, Any]:
        device = self.bundle.device
        loader = DataLoader(
            _PromptDataset(items),
            batch_size=self.cfg.batch_size,
            collate_fn=_collate,
            shuffle=False,
        )
        all_hidden: List[torch.Tensor] = []
        all_attn_summary: List[torch.Tensor] = []
        labels: List[int] = []
        prompts: List[str] = []
        answers: List[str] = []
        datasets: List[str] = []

        n_layers = self.bundle.num_hidden_layers
        keep_attn_layers = (
            [_resolve_layer_index(i, n_layers) for i in self.cfg.attention_layers]
            if self.cfg.capture_attention else []
        )

        for batch in tqdm(loader, desc=f"extract:{self.bundle.short_name}"):
            enc, answer_mask = self._encode_batch(batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            answer_mask = answer_mask.to(device)
            outputs = self.bundle.model(
                **enc,
                output_hidden_states=True,
                output_attentions=self.cfg.capture_attention,
                use_cache=False,
                return_dict=True,
            )
            # hidden_states: tuple of length (L+1) [embed, block_1, ..., block_L]
            pooled_layers = []
            for h in outputs.hidden_states:
                pooled = _pool(h.float(), answer_mask, self.cfg.pool)   # [B, D]
                pooled_layers.append(pooled.cpu())
            stacked = torch.stack(pooled_layers, dim=1)                # [B, L+1, D]
            all_hidden.append(stacked)

            if self.cfg.capture_attention and outputs.attentions is not None:
                # outputs.attentions: tuple of L tensors [B, H, T, T]
                # We summarize: per chosen layer, mean head, distribution
                # entropy of attention from the LAST answer-token over
                # the input -> a [B, L_keep, H] entropy vector.
                B = stacked.shape[0]
                last_idx = answer_mask.float().cumsum(dim=1).argmax(dim=1)   # [B]
                summaries = []
                for li in keep_attn_layers:
                    if li >= len(outputs.attentions):
                        continue
                    attn = outputs.attentions[li].float()                   # [B,H,T,T]
                    H = attn.shape[1]
                    row = attn[torch.arange(B, device=device), :, last_idx]  # [B,H,T]
                    eps = 1e-9
                    ent = -(row * (row + eps).log()).sum(dim=-1)             # [B,H]
                    summaries.append(ent.cpu())
                if summaries:
                    all_attn_summary.append(torch.stack(summaries, dim=1))    # [B,L_keep,H]

            for it in batch:
                labels.append(int(it.label))
                prompts.append(it.prompt)
                answers.append(it.answer)
                datasets.append(it.dataset)

        out: Dict[str, Any] = {
            "hidden_states": torch.cat(all_hidden, dim=0).contiguous(),   # [N,L+1,D]
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompts": prompts,
            "answers": answers,
            "datasets": datasets,
            "model_short_name": self.bundle.short_name,
            "pool": self.cfg.pool,
            "num_layers": n_layers,
            "hidden_size": self.bundle.hidden_size,
        }
        if all_attn_summary:
            out["attention_entropy"] = torch.cat(all_attn_summary, dim=0).contiguous()
            out["attention_layers_kept"] = keep_attn_layers
        return out


# ------------------- top-level convenience function --------------------
def extract_hidden_states_for_dataset(
    bundle: ModelBundle,
    items: Sequence[PromptItem],
    out_path: str | Path,
    cfg: ExtractionConfig,
) -> Dict[str, Any]:
    """Run extraction and save to `out_path` (a .pt file). Returns the dict."""
    ex = HiddenStateExtractor(bundle, cfg)
    res = ex.extract(items)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, out_path)
    return res
