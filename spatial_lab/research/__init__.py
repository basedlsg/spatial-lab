"""
Research validation module for Spatial AI Research Lab.

Provides comprehensive research validation capabilities with proper scientific
methodology and statistical analysis.
"""

from .research_validator import (
    ResearchValidator,
    ExperimentalCondition,
    TrialResult,
    BaselineResult
)

__all__ = [
    'ResearchValidator',
    'ExperimentalCondition', 
    'TrialResult',
    'BaselineResult'
] 