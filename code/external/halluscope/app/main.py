import streamlit as st

st.set_page_config(
    page_title="HalluScope",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 HalluScope")
st.subheader("LLM Hallucination Detection via Internal States")
st.markdown("""
**HalluScope** detects hallucinations in LLM outputs by analyzing the model's internal
hidden states — without any external labels or auxiliary models.

### Metrics
| Metric | Source | Description |
|--------|--------|-------------|
| **EigenScore** | INSIDE (2024) | Log-determinant of sentence embedding covariance |
| **SpectralEntropy** | 🆕 This project | Shannon entropy over normalized eigenvalue distribution |
| **Combined** | 🆕 This project | Weighted fusion of both metrics |

### Navigation
Use the sidebar to:
- **Single Query** — Analyze one question interactively
- **Batch Eval** — Run on a dataset and compute AUROC/PCC
- **Metric Compare** — Compare all metrics side by side
""")