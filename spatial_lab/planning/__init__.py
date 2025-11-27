"""
Planning Module for Spatial Lab

Provides A*-based path planning with multiple cost functions
for generating candidate navigation plans.
"""

from .astar_planner import (
    AStarPlanner,
    CandidatePlan,
    PlanMetadata,
    CostFunction,
    check_collision,
    interpolate_path,
)

from .plan_generator import (
    MultiPlanGenerator,
    PlanGenerationConfig,
    ScenarioResult,
    create_test_scenario,
)

from .execution_sim import (
    ExecutionSimulator,
    ExecutionResult,
    evaluate_accuracy,
)

# v2 components
from .scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    compute_J,
    compute_regret,
    is_optimal_choice,
    check_plan_diversity,
)

from .plan_generator_v2 import (
    MultiPlanGeneratorV2,
    PlanGenerationConfigV2,
    ScenarioResultV2,
    PlanWithScore,
    create_diverse_scenario,
)

from .execution_sim_v2 import (
    ExecutionSimulatorV2,
    ExecutionConfigV2,
    ExecutionResultV2,
    compute_trial_metrics,
)

__all__ = [
    # A* Planner
    "AStarPlanner",
    "CandidatePlan",
    "PlanMetadata",
    "CostFunction",
    "check_collision",
    "interpolate_path",
    # Plan Generator (v1)
    "MultiPlanGenerator",
    "PlanGenerationConfig",
    "ScenarioResult",
    "create_test_scenario",
    # Execution Simulation (v1)
    "ExecutionSimulator",
    "ExecutionResult",
    "evaluate_accuracy",
    # Scoring (v2)
    "ScoringConfig",
    "DEFAULT_SCORING",
    "compute_J",
    "compute_regret",
    "is_optimal_choice",
    "check_plan_diversity",
    # Plan Generator (v2)
    "MultiPlanGeneratorV2",
    "PlanGenerationConfigV2",
    "ScenarioResultV2",
    "PlanWithScore",
    "create_diverse_scenario",
    # Execution Simulation (v2)
    "ExecutionSimulatorV2",
    "ExecutionConfigV2",
    "ExecutionResultV2",
    "compute_trial_metrics",
]
