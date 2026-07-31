"""
Generates figures/capacity-sweep.pdf for the paper.

This script performs no new computation, only visualization.

FIXED (found during a closure review): the four condition-mean arrays used to
be hardcoded copies of the SS4.3 table, so a rerun of code/02d would silently
leave the figure showing the previous run's numbers -- the same
number-survives-after-the-run-was-corrected failure this project's own
code/53 exists to catch, sitting inside the figure pipeline. They are now read
from results/corrected_capacity_placebo_sweep.json, so the figure cannot drift
from the table.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = json.load(open(ROOT / "results" / "corrected_capacity_placebo_sweep.json"))
assert D["config"].get("seed_scheme") == "decoupled", \
    "figure must be built from the decoupled-seed run reported in SS4.3"

capacities = [str(c) for c in D["config"]["capacities"]]
means = {k: [float(np.mean(D["by_capacity"][c]["aucs"][k])) for c in capacities]
         for k in ["leaky", "clean", "clean_matched", "placebo"]}

x = np.arange(len(capacities))
width = 0.2

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - 1.5 * width, means["leaky"], width, label="LEAKY", color="#C44E52")
ax.bar(x - 0.5 * width, means["clean"], width, label="CLEAN", color="#8172B2")
ax.bar(x + 0.5 * width, means["clean_matched"], width, label="CLEAN_MATCHED", color="#4C72B0")
ax.bar(x + 1.5 * width, means["placebo"], width, label="PLACEBO", color="#55A868")
ax.set_ylabel("AUROC")
lo = min(min(v) for v in means.values())
hi = max(max(v) for v in means.values())
pad = 0.35 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
ax.set_xlabel("Hidden units (capacity)")
ax.set_xticks(x)
ax.set_xticklabels(capacities)
ax.legend(loc="lower center", fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.32))
ax.set_title("Case Study 3 capacity sweep: LEAKY, CLEAN, and CLEAN_MATCHED\n"
             "cluster tightly together, all clearly above PLACEBO, at every capacity",
             fontsize=9.5)
fig.tight_layout()
out = ROOT / "draft" / "latex" / "figures" / "capacity-sweep.pdf"
fig.savefig(out)
print(f"Saved: {out}")
for k, v in means.items():
    print(f"  {k:15s} {[round(z, 4) for z in v]}")
