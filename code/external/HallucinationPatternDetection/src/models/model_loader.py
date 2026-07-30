"""Load 4-bit NF4 quantized HuggingFace causal LMs onto an 8 GB GPU.

Returns a `ModelBundle` carrying the model, tokenizer, and metadata
needed downstream (number of hidden layers, hidden size, family name,
chat template formatter).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# -------------------- chat-template formatters -------------------------
def _format_llama3(messages: List[Dict[str, str]]) -> str:
    """Llama-3 instruct template (used when tokenizer lacks one)."""
    out = "<|begin_of_text|>"
    for m in messages:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out


def _format_mistral(messages: List[Dict[str, str]]) -> str:
    # Mistral-Instruct expects [INST] ... [/INST] turns.
    out = "<s>"
    for m in messages:
        if m["role"] == "user":
            out += f"[INST] {m['content']} [/INST]"
        elif m["role"] == "assistant":
            out += f" {m['content']}</s><s>"
        elif m["role"] == "system":
            out += f"[INST] <<SYS>>\n{m['content']}\n<</SYS>>\n\n[/INST]"
    return out


def _format_qwen(messages: List[Dict[str, str]]) -> str:
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out


CHAT_TEMPLATES: Dict[str, Callable[[List[Dict[str, str]]], str]] = {
    "llama3": _format_llama3,
    "mistral": _format_mistral,
    "qwen": _format_qwen,
}


# ------------------------------ bundle ---------------------------------
@dataclass
class ModelBundle:
    short_name: str
    hf_id: str
    family: str
    model: Any
    tokenizer: Any
    num_hidden_layers: int
    hidden_size: int
    chat_template: str
    device: str

    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's chat template.

        Prefers the tokenizer's built-in template; falls back to our
        manual formatter if the tokenizer lacks one.
        """
        try:
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        fn = CHAT_TEMPLATES.get(self.chat_template, _format_llama3)
        return fn(messages)


# ------------------------------ loader ---------------------------------
def load_quantized_model(
    cfg_model: Dict[str, Any],
    quant_cfg: Dict[str, Any],
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> ModelBundle:
    """Load `cfg_model` in 4-bit NF4 and return the assembled bundle."""
    hf_id = cfg_model["hf_id"]
    compute_dtype = getattr(torch, quant_cfg.get("compute_dtype", "bfloat16"))

    bnb = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("bits", 4) == 4,
        load_in_8bit=quant_cfg.get("bits", 4) == 8,
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(hf_id, cache_dir=cache_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        quantization_config=bnb if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # required for output_attentions=True
    )
    model.eval()

    return ModelBundle(
        short_name=cfg_model["short_name"],
        hf_id=hf_id,
        family=cfg_model.get("family", ""),
        model=model,
        tokenizer=tokenizer,
        num_hidden_layers=getattr(config, "num_hidden_layers", 32),
        hidden_size=getattr(config, "hidden_size", 4096),
        chat_template=cfg_model.get("chat_template", "llama3"),
        device=device,
    )
