from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "microsoft/phi-2"
    device: str = "cuda"          # "cpu" for fallback
    torch_dtype: str = "float16"  # float16 for RTX 5070
    trust_remote_code: bool = True

@dataclass
class SamplingConfig:
    num_generations: int = 10     # K in paper
    temperature: float = 0.5
    top_p: float = 0.99
    top_k: int = 5
    max_new_tokens: int = 64

@dataclass
class EigenScoreConfig:
    alpha: float = 0.001          # Covariance matrix regularization
    layer_index: Optional[int] = None  # None → auto mid layer

@dataclass
class SpectralEntropyConfig:
    """
    New metric config.
    SpectralEntropy normalizes eigenvalues to a probability distribution
    and computes Shannon entropy over them.
    High entropy → diverse eigenvalue spectrum → likely hallucination.
    """
    alpha: float = 0.001
    epsilon: float = 1e-10        # Numerical stability for log

@dataclass
class FeatureClipConfig:
    memory_bank_size: int = 3000  # N in paper
    clip_percentile: float = 0.2  # p in paper (top/bottom %)

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eigenscore: EigenScoreConfig = field(default_factory=EigenScoreConfig)
    spectral: SpectralEntropyConfig = field(default_factory=SpectralEntropyConfig)
    clip: FeatureClipConfig = field(default_factory=FeatureClipConfig)
    results_dir: str = "results"

# Global singleton
CFG = Config()