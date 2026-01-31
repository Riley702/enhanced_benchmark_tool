"""Enhanced Benchmark Tool.

Public API (stable):
- profile_dataset
- benchmark_model
- extract_feature_importance

"""

from .dataset_profiling import profile_dataset
from .model_benchmarking import benchmark_model, extract_feature_importance

__all__ = [
    "profile_dataset",
    "benchmark_model",
    "extract_feature_importance",
]
