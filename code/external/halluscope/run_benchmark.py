"""
CLI entry point for running benchmarks without the Streamlit UI.
Usage:
    python run_benchmark.py --dataset triviaqa --samples 100 --no-clip
"""
import argparse
import json
from tqdm import tqdm

from core.model import generate_responses, load_model
from core.eigenscore import eigenscore
from core.spectral_entropy import spectral_entropy, combined_score
from eval.datasets import load_triviaqa, load_coqa
from eval.correctness import is_correct
from eval.benchmark import auroc, pcc, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["triviaqa", "coqa"], default="triviaqa")
    parser.add_argument("--samples",  type=int, default=100)
    parser.add_argument("--no-clip",  action="store_true")
    args = parser.parse_args()

    apply_clip = not args.no_clip
    print(f"Loading model...")
    load_model()

    loader = load_triviaqa if args.dataset == "triviaqa" else load_coqa
    samples = loader(max_samples=args.samples)
    print(f"Running on {len(samples)} samples from {args.dataset}...")

    records, e_scores, s_scores, c_scores, labels = [], [], [], [], []

    for sample in tqdm(samples):
        q, refs = sample["question"], sample["answers"]
        responses, hidden_states = generate_responses(q)
        e_res = eigenscore(hidden_states, apply_clip=apply_clip)
        s_res = spectral_entropy(hidden_states, apply_clip=apply_clip)
        c_sc  = combined_score(e_res, s_res)
        correct = is_correct(responses[0], refs)

        e_scores.append(e_res["score"])
        s_scores.append(s_res["normalized_score"])
        c_scores.append(c_sc)
        labels.append(int(correct))
        records.append({
            "question": q, "prediction": responses[0],
            "correct": int(correct),
            "eigenscore": e_res["score"],
            "spectral_norm": s_res["normalized_score"],
            "combined": c_sc,
        })

    results = {
        "dataset": args.dataset,
        "n_samples": len(samples),
        "feature_clip": apply_clip,
        "auroc_eigen":    auroc(e_scores, labels),
        "auroc_spectral": auroc(s_scores, labels),
        "auroc_combined": auroc(c_scores, labels),
        "rows": records,
    }

    path = save_results(results, f"{args.dataset}_{len(samples)}")
    print(f"\n{'='*50}")
    print(f"AUROC EigenScore:      {results['auroc_eigen']:.4f}")
    print(f"AUROC SpectralEntropy: {results['auroc_spectral']:.4f}")
    print(f"AUROC Combined:        {results['auroc_combined']:.4f}")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()