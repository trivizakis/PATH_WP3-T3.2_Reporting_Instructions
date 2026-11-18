# -*- coding: utf-8 -*-
"""
PATH WP3 - Task 3.2: AI Fairness & Bias Reporting Toolkit
=========================================================
A standardized Python toolkit to evaluate, quantify, and visualize
algorithmic fairness, discrimination, and subgroup performance disparities
in healthcare and biomedical machine learning models.
"""

from .metrics import FairnessMetricsReporter
from .visualization import FairnessFigureGenerator
from .datasets import generate_synthetic_fairness_dataset

__version__ = "1.0.0"
__all__ = [
    "FairnessMetricsReporter",
    "FairnessFigureGenerator",
    "generate_synthetic_fairness_dataset",
]

