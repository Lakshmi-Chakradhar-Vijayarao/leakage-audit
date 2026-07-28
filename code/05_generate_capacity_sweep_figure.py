"""
Generates figures/capacity-sweep.pdf for the paper.
All numbers are copied directly from the capacity-sweep table in Section
4.3 (see results/ for the underlying per-seed data) -- this script
performs no new computation, only visualization of already-reported
numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

capacities = ["16", "48", "128", "384"]
leaky = [0.7611, 0.7647, 0.7604, 0.7533]
clean = [0.7567, 0.7617, 0.7555, 0.7474]
clean_matched = [0.7602, 0.7631, 0.7571, 0.7506]
placebo = [0.7272, 0.7375, 0.7309, 0.7350]

x = np.arange(len(capacities))
width = 0.2

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - 1.5*width, leaky, width, label="LEAKY", color="#C44E52")
ax.bar(x - 0.5*width, clean, width, label="CLEAN", color="#8172B2")
ax.bar(x + 0.5*width, clean_matched, width, label="CLEAN_MATCHED", color="#4C72B0")
ax.bar(x + 1.5*width, placebo, width, label="PLACEBO", color="#55A868")
ax.set_ylabel("AUROC")
ax.set_ylim(0.70, 0.78)
ax.set_xlabel("Hidden units (capacity)")
ax.set_xticks(x)
ax.set_xticklabels(capacities)
ax.legend(loc="lower center", fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.32))
ax.set_title("Case Study 3 capacity sweep: LEAKY, CLEAN, and CLEAN_MATCHED\ncluster tightly together, all clearly above PLACEBO, at every capacity", fontsize=9.5)
fig.tight_layout()
fig.savefig("../draft/latex/figures/capacity-sweep.pdf")
print("Saved.")
