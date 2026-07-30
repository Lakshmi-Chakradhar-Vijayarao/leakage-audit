from datasets import load_dataset
from typing import List, Dict

def load_triviaqa(split="validation", max_samples=500) -> List[Dict]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    samples = []
    for item in ds.select(range(min(max_samples, len(ds)))):
        answers = item["answer"]["aliases"] if item["answer"]["aliases"] else [item["answer"]["value"]]
        samples.append({"question": item["question"], "answers": answers})
    return samples

def load_coqa(split="validation", max_samples=500) -> List[Dict]:
    ds = load_dataset("coqa", split=split)
    samples = []
    for story in ds.select(range(min(max_samples, len(ds)))):
        for q, a in zip(story["questions"], story["answers"]["input_text"]):
            samples.append({"question": q, "answers": [a]})
            if len(samples) >= max_samples:
                return samples
    return samples

def load_custom(path: str) -> List[Dict]:
    """Load from a JSON file: [{'question':..., 'answers':[...]}]"""
    import json
    with open(path) as f:
        return json.load(f)