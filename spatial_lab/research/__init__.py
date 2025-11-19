"""
Research module for Spatial AI Research Lab.

Includes:
- Research validation with statistical analysis
- NL + Code Generation experiment for robot coordination
"""

# Import research validator (requires optional dependencies)
try:
    from .research_validator import (
        ResearchValidator,
        ExperimentalCondition,
        TrialResult,
        BaselineResult
    )
except ImportError:
    ResearchValidator = None
    ExperimentalCondition = None
    TrialResult = None
    BaselineResult = None

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
    # Research validator (may be None if deps missing)
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