import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List


def plot_eigenvalues(eigenvalues: List[float], title: str = "Eigenvalue Spectrum") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(eigenvalues) + 1)),
        y=eigenvalues,
        marker_color=[
            f"rgba(99, 110, 250, {max(0.3, v / max(eigenvalues))})"
            for v in eigenvalues
        ],
        name="Eigenvalues"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_dark",
        height=350,
    )
    return fig


def plot_spectral_probs(probs: List[float]) -> go.Figure:
    """Pie-like distribution of normalized eigenvalues."""
    labels = [f"λ{i+1}" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker_color=px.colors.sequential.Plasma_r[:len(probs)],
    ))
    fig.update_layout(
        title="SpectralEntropy — Eigenvalue Probability Distribution",
        xaxis_title="Eigenvalue",
        yaxis_title="Normalized Weight",
        template="plotly_dark",
        height=350,
    )
    return fig


def plot_metric_comparison(
    methods: List[str],
    auroc_scores: List[float],
    pcc_scores: List[float]
) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(name="AUROC", x=methods, y=auroc_scores, marker_color="#636EFA"),
        go.Bar(name="PCC",   x=methods, y=pcc_scores,   marker_color="#EF553B"),
    ])
    fig.update_layout(
        barmode="group",
        title="Metric Comparison",
        yaxis_title="Score",
        template="plotly_dark",
        height=400,
    )
    return fig


def hallucination_gauge(score: float, label: str = "Hallucination Risk") -> go.Figure:
    """Score in [0, 1], higher = more hallucination."""
    color = "green" if score < 0.4 else ("orange" if score < 0.7 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "rgba(0,200,0,0.15)"},
                {"range": [40, 70], "color": "rgba(255,165,0,0.15)"},
                {"range": [70, 100],"color": "rgba(255,0,0,0.15)"},
            ],
        }
    ))
    fig.update_layout(height=300, template="plotly_dark")
    return fig