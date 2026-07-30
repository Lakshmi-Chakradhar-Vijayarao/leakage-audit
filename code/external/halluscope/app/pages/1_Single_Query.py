# ── app/pages/1_Single_Query.py ───────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.model import generate_responses, load_model
from core.eigenscore import eigenscore
from core.spectral_entropy import spectral_entropy, combined_score
from utils.visualize import (
    plot_eigenvalues, plot_spectral_probs, hallucination_gauge
)

st.set_page_config(page_title="Single Query", layout="wide")
st.title("🔎 Single Query Analysis")

# Model loading
with st.sidebar:
    st.header("⚙️ Settings")
    apply_clip = st.toggle("Feature Clipping", value=True)
    num_gen = st.slider("Number of Generations (K)", 5, 20, 10)
    from utils.config import CFG
    CFG.sampling.num_generations = num_gen

@st.cache_resource(show_spinner="Loading Phi-2...")
def get_model():
    return load_model()

get_model()

question = st.text_input("Enter your question:", placeholder="Who wrote Romeo and Juliet?")
run = st.button("Analyze", type="primary", disabled=not question)

if run and question:
    with st.spinner(f"Generating {num_gen} responses..."):
        responses, hidden_states = generate_responses(question)

    with st.spinner("Computing metrics..."):
        e_result = eigenscore(hidden_states, apply_clip=apply_clip)
        s_result = spectral_entropy(hidden_states, apply_clip=apply_clip)
        c_score  = combined_score(e_result, s_result)

    # ── Responses
    st.subheader("📝 Generated Responses")
    cols = st.columns(min(5, len(responses)))
    for i, (col, resp) in enumerate(zip(cols * 2, responses)):
        with col:
            st.info(f"**#{i+1}:** {resp}")

    st.divider()

    # ── Gauges
    st.subheader("📊 Hallucination Risk")
    g1, g2, g3 = st.columns(3)
    with g1:
        import numpy as np
        e_norm = float(np.clip(-e_result["score"] / 10.0, 0, 1))
        st.plotly_chart(hallucination_gauge(1 - e_norm, "EigenScore Risk"), use_container_width=True)
        st.metric("EigenScore (raw)", f"{e_result['score']:.4f}")
    with g2:
        st.plotly_chart(hallucination_gauge(s_result["normalized_score"], "SpectralEntropy Risk"), use_container_width=True)
        st.metric("SpectralEntropy (normalized)", f"{s_result['normalized_score']:.4f}")
    with g3:
        st.plotly_chart(hallucination_gauge(c_score, "Combined Risk"), use_container_width=True)
        st.metric("Combined Score", f"{c_score:.4f}")

    st.divider()

    # ── Eigenvalue plots
    st.subheader("🔬 Eigenvalue Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_eigenvalues(e_result["eigenvalues"], "EigenScore — Eigenvalue Spectrum"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            plot_spectral_probs(s_result["probs"]),
            use_container_width=True
        )