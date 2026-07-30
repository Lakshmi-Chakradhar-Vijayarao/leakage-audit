from rouge_score import rouge_scorer
from typing import List

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l(prediction: str, references: List[str], threshold=0.5) -> bool:
    scores = [_scorer.score(ref, prediction)["rougeL"].fmeasure for ref in references]
    return max(scores) >= threshold

def is_correct(prediction: str, references: List[str]) -> bool:
    return rouge_l(prediction, references)