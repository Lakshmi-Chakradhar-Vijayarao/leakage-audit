import json, os
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from typing import List, Dict
from utils.config import CFG

def auroc(scores: List[float], labels: List[int]) -> float:
    """labels: 1 = correct (non-hallucination), 0 = hallucination"""
    try:
        return roc_auc_score(labels, [-s for s in scores])  # lower score → correct
    except Exception:
        return 0.5

def pcc(scores: List[float], correctness_scores: List[float]) -> float:
    try:
        r, _ = pearsonr([-s for s in scores], correctness_scores)
        return float(r)
    except Exception:
        return 0.0

def save_results(results: Dict, name: str):
    os.makedirs(CFG.results_dir, exist_ok=True)
    path = os.path.join(CFG.results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path