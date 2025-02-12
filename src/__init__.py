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

Package-Level Functions:
    - load_config(): Loads configuration settings for the package.
    - initialize_logging(): Configures logging for debugging and tracking.

Attributes:
    - VERSION: Defines the current version of the package.
"""

import logging
import os
import json
from . import dataset_profiling
from . import feature_importance
from . import model_benchmarking
from . import visualizations
from . import util

# Define the version of the package
VERSION = "1.0.0"

__all__ = [
    "dataset_profiling",
    "feature_importance",
    "model_benchmarking",
    "visualizations",
    "util",
    "load_config",
    "initialize_logging",
]



