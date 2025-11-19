"""
Research module for Spatial AI Research Lab.

Includes:
- Research validation with statistical analysis
- NL + Code Generation experiment for robot coordination
"""

from .research_validator import (
    ResearchValidator,
    ExperimentalCondition,
    TrialResult,
    BaselineResult
)

from .groq_client import GroqClient, GroqConfig, create_groq_client
from .nl_code_experiment import (
    NLCodeExperiment,
    ExperimentResults,
    ExperimentTrial,
    RobotState,
    NeedType,
    run_quick_test
)

__all__ = [
    # Research validator
    'ResearchValidator',
    'ExperimentalCondition',
    'TrialResult',
    'BaselineResult',
    # NL + Code experiment
    'GroqClient',
    'GroqConfig',
    'create_groq_client',
    'NLCodeExperiment',
    'ExperimentResults',
    'ExperimentTrial',
    'RobotState',
    'NeedType',
    'run_quick_test'
] 