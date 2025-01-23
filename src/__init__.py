"""
Source Code Package
====================

This package contains the core modules for the enhanced_benchmark_tool project.

Modules:
    dataset_profiling - Functions for profiling datasets.
    feature_importance - Functions for computing feature importance.
    model_benchmarking - Core benchmarking logic for models.
    visualizations - Utilities for generating data visualizations.
    util - Utility functions for data processing, metrics, logging, and configuration.
"""

# Make key modules accessible at the package level
from . import dataset_profiling
from . import feature_importance
from . import model_benchmarking
from . import visualizations
from . import util

__all__ = [
    "dataset_profiling",
    "feature_importance",
    "model_benchmarking",
    "visualizations",
    "util",
]
