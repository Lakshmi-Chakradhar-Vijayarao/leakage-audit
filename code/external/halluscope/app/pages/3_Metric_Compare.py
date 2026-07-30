# ── app/pages/3_Metric_Compare.py ─────────────────────────────────────────────
import streamlit as st
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from utils.visualize import plot_metric_comparison

st.set_page_config(page_title="Metric Comparison", layout="wide")
st.title("📊 Metric Comparison")

result_files = glob.glob("results/*.json")
if not result_files:
    st.warning("No results found. Run Batch Evaluation first.")
    st.stop()

selected = st.multiselect("Select result files", result_files, default=result_files[:1])
if not selected:
    st.stop()

rows = []
for f in selected:
    with open(f) as fp:
        d = json.load(fp)
    rows.append({
        "Run": os.path.basename(f).replace(".json", ""),
        "AUROC EigenScore": d.get("auroc_eigen", 0),
        "AUROC SpectralEntropy": d.get("auroc_spectral", 0),
        "AUROC Combined": d.get("auroc_combined", 0),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

for _, row in df.iterrows():
    fig = plot_metric_comparison(
        methods=["EigenScore", "SpectralEntropy", "Combined"],
        auroc_scores=[row["AUROC EigenScore"], row["AUROC SpectralEntropy"], row["AUROC Combined"]],
        pcc_scores=[0, 0, 0],  # Fill in if PCC computed
    )
    st.plotly_chart(fig, use_container_width=True)