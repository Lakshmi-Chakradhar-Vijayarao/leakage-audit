import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from utils.config import CFG

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    dtype = torch.float16 if CFG.model.torch_dtype == "float16" else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(
        CFG.model.model_name,
        trust_remote_code=CFG.model.trust_remote_code
    )
    # Phi-2 has no pad token by default
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        CFG.model.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=CFG.model.trust_remote_code,
        output_hidden_states=True,   # Required for internal state extraction
    )
    _model.eval()
    return _model, _tokenizer


def generate_responses(question: str) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Generate K responses for a question.
    Returns:
        responses  : list of decoded strings
        hidden_states_list : list of hidden state tensors, one per response
                             shape per item: (num_layers, seq_len, hidden_dim)
    """
    model, tok = load_model()
    sc = CFG.sampling

    prompt = f"Question: {question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    responses, hidden_states_list = [], []
    with torch.no_grad():
        for _ in range(sc.num_generations):
            out = model.generate(
                **inputs,
                max_new_tokens=sc.max_new_tokens,
                do_sample=True,
                temperature=sc.temperature,
                top_p=sc.top_p,
                top_k=sc.top_k,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Decode only the newly generated tokens
            gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
            text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            responses.append(text)

            # out.hidden_states: tuple of steps, each step is tuple of layers
            # Stack into (num_layers, total_new_tokens, hidden_dim)
            step_last_layers = [step[-1] for step in out.hidden_states]  # last layer each step
            # shape: (new_tokens, 1, hidden_dim) → squeeze
            all_layers = []
            num_layers = len(out.hidden_states[0])
            for layer_idx in range(num_layers):
                layer_tokens = torch.cat(
                    [step[layer_idx][0] for step in out.hidden_states], dim=0
                )  # (new_tokens, hidden_dim)
                all_layers.append(layer_tokens)
            # Stack: (num_layers, new_tokens, hidden_dim)
            hidden = torch.stack(all_layers, dim=0)
            hidden_states_list.append(hidden)

    return responses, hidden_states_list 