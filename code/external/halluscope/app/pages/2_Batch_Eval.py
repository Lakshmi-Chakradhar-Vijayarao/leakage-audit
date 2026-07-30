# ── app/pages/2_Batch_Eval.py ─────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from core.model import generate_responses, load_model
from core.eigenscore import eigenscore
from core.spectral_entropy import spectral_entropy, combined_score
from eval.datasets import load_triviaqa, load_coqa
from eval.correctness import is_correct
from eval.benchmark import auroc, pcc, save_results

st.set_page_config(page_title="Batch Evaluation", layout="wide")
st.title("📦 Batch Evaluation")

with st.sidebar:
    dataset = st.selectbox("Dataset", ["TriviaQA", "CoQA"])
    n_samples = st.slider("Samples", 10, 200, 50)
    apply_clip = st.toggle("Feature Clipping", value=True)

@st.cache_resource(show_spinner="Loading Phi-2...")
def get_model():
    return load_model()
get_model()

if st.button("▶ Run Evaluation", type="primary"):
    loader = load_triviaqa if dataset == "TriviaQA" else load_coqa
    samples = loader(max_samples=n_samples)

    results = []
    prog = st.progress(0, text="Evaluating...")
    status = st.empty()

    for i, sample in enumerate(samples):
        q, refs = sample["question"], sample["answers"]
        status.text(f"[{i+1}/{len(samples)}] {q[:80]}...")

        responses, hidden_states = generate_responses(q)
        e_res = eigenscore(hidden_states, apply_clip=apply_clip)
        s_res = spectral_entropy(hidden_states, apply_clip=apply_clip)
        c_sc  = combined_score(e_res, s_res)
        correct = is_correct(responses[0], refs)

        results.append({
            "question": q,
            "prediction": responses[0],
            "correct": int(correct),
            "eigenscore": e_res["score"],
            "spectral_entropy": s_res["score"],
            "spectral_norm": s_res["normalized_score"],
            "combined": c_sc,
        })
        prog.progress((i + 1) / len(samples))

    status.success("Done!")

    df = pd.DataFrame(results)
    labels = df["correct"].tolist()

    auroc_e = auroc(df["eigenscore"].tolist(), labels)
    auroc_s = auroc(df["spectral_norm"].tolist(), labels)
    auroc_c = auroc(df["combined"].tolist(), labels)

    st.subheader("📈 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("AUROC — EigenScore",      f"{auroc_e:.3f}")
    m2.metric("AUROC — SpectralEntropy", f"{auroc_s:.3f}")
    m3.metric("AUROC — Combined",        f"{auroc_c:.3f}")

    st.dataframe(df, use_container_width=True)

    path = save_results({
        "dataset": dataset, "n_samples": n_samples,
        "auroc_eigen": auroc_e, "auroc_spectral": auroc_s, "auroc_combined": auroc_c,
        "rows": results
    }, name=f"{dataset}_{n_samples}")
    st.success(f"Results saved to `{path}`")