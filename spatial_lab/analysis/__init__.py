"""
Analysis Module for Calibration Experiments

Provides statistical analysis tools for measuring LLM calibration
including ECE, Brier scores, and reliability diagrams.
"""

from .calibration import (
    CalibrationAnalyzer,
    CalibrationMetrics,
    compute_ece,
    compute_brier_score,
    compute_adaptive_ece,
    separate_api_failures,
)

from .visualization import (
    create_reliability_diagram,
    create_calibration_summary,
    export_analysis_report,
)

__all__ = [
    # Calibration Analysis
    "CalibrationAnalyzer",
    "CalibrationMetrics",
    "compute_ece",
    "compute_brier_score",
    "compute_adaptive_ece",
    "separate_api_failures",
    # Visualization
    "create_reliability_diagram",
    "create_calibration_summary",
    "export_analysis_report",
]
