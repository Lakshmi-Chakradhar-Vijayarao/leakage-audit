"""
Paper 2 -- quantifies Case Study 4's winner's-curse severity directly
from data already shipped in code/external/HallucinationPatternDetection/
results/probes/*.json (33 layers x 3 seeds each, auroc per layer per
seed). The paper previously claimed this required re-extracting hidden
states from the quantized checkpoint from scratch; a fresh review found
the winner's-curse estimate is fully computable from files already in
this repo.

Protocol: for each of the 3 seeds, use it as the held-out evaluation
seed and select the best layer using the MEAN of the other 2 seeds
(honest, held-out selection). Compare the resulting held-out AUROC to
the naive reported protocol (best_auroc = mean of all 3 seeds at the
layer that is the argmax of that same mean -- i.e., no held-out split
at all). The difference, averaged over the 3 leave-one-seed-out
rotations, is the winner's-curse estimate for that model/dataset/probe
combination.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "code" / "external" / "HallucinationPatternDetection" / "results" / "probes"
OUT_PATH = ROOT / "results" / "case_study_4_winners_curse.json"


def quantify_file(path):
    d = json.load(open(path))
    layers = d["layers"]
    per_layer = d["per_layer"]
    n_seeds = len(per_layer[str(layers[0])]["seed_values"]["auroc"])

    naive_best_auroc = d["best_auroc"]
    naive_best_layer = d["best_layer"]

    honest_aurocs = []
    for held_out_seed in range(n_seeds):
        other_seeds = [s for s in range(n_seeds) if s != held_out_seed]
        # select layer using mean of the OTHER seeds only
        mean_over_others = {
            layer: np.mean([per_layer[str(layer)]["seed_values"]["auroc"][s] for s in other_seeds])
            for layer in layers
        }
        selected_layer = max(mean_over_others, key=mean_over_others.get)
        held_out_auroc = per_layer[str(selected_layer)]["seed_values"]["auroc"][held_out_seed]
        honest_aurocs.append(held_out_auroc)

    honest_mean = float(np.mean(honest_aurocs))
    winners_curse = float(naive_best_auroc - honest_mean)

    all_layer_means = [per_layer[str(layer)]["auroc_mean"] for layer in layers]
    worst_layer_auroc = float(min(all_layer_means))
    mean_over_all_layers = float(np.mean(all_layer_means))

    return {
        "naive_best_layer": naive_best_layer,
        "naive_best_auroc": naive_best_auroc,
        "honest_held_out_auroc": honest_mean,
        "winners_curse_estimate": winners_curse,
        "mean_over_all_layers": mean_over_all_layers,
        "worst_layer_auroc": worst_layer_auroc,
        "spread_best_minus_worst": float(naive_best_auroc - worst_layer_auroc),
    }


def main():
    out = {}
    files = sorted(PROBES_DIR.glob("*.json"))
    print(f"Found {len(files)} probe result files", flush=True)
    for f in files:
        key = f.stem
        try:
            out[key] = quantify_file(f)
            print(f"  {key}: winners_curse={out[key]['winners_curse_estimate']:+.4f}  "
                  f"naive_best={out[key]['naive_best_auroc']:.4f}  "
                  f"honest={out[key]['honest_held_out_auroc']:.4f}  "
                  f"mean_all_layers={out[key]['mean_over_all_layers']:.4f}  "
                  f"worst={out[key]['worst_layer_auroc']:.4f}", flush=True)
        except Exception as e:
            print(f"  {key}: SKIPPED ({e})", flush=True)

    curses = [v["winners_curse_estimate"] for v in out.values()]
    print(f"\nAcross {len(curses)} model/dataset/probe combinations:")
    print(f"  mean winners-curse estimate: {np.mean(curses):+.4f}")
    print(f"  max winners-curse estimate:  {np.max(curses):+.4f}")
    print(f"  min winners-curse estimate:  {np.min(curses):+.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
