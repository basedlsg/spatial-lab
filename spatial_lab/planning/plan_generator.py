"""
Multi-Plan Generator

Generates multiple candidate plans using different cost functions
for LLM-based plan selection experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import random
import hashlib
import json

from .astar_planner import (
    AStarPlanner,
    CandidatePlan,
    CostFunction,
    check_collision,
)


@dataclass
class PlanGenerationConfig:
    """Configuration for plan generation."""
    grid_resolution: float = 0.5
    safety_margin: float = 0.3
    cost_functions: List[CostFunction] = field(
        default_factory=lambda: [
            CostFunction.SHORTEST,
            CostFunction.SAFEST,
            CostFunction.BALANCED,
        ]
    )
    randomize_order: bool = True
    validate_plans: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "grid_resolution": self.grid_resolution,
            "safety_margin": self.safety_margin,
            "cost_functions": [cf.value for cf in self.cost_functions],
            "randomize_order": self.randomize_order,
            "validate_plans": self.validate_plans,
        }


@dataclass
class ScenarioResult:
    """Result of generating plans for a scenario."""
    scenario_id: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    obstacles: List[Dict[str, Any]]
    warehouse_dims: Tuple[float, float]
    candidate_plans: List[CandidatePlan]
    optimal_plan_id: str  # Ground truth best plan
    plan_order: List[str]  # Order presented to LLM (may be randomized)
    generation_successful: bool
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": self.obstacles,
            "warehouse_dims": list(self.warehouse_dims),
            "candidate_plans": [p.to_dict() for p in self.candidate_plans],
            "optimal_plan_id": self.optimal_plan_id,
            "plan_order": self.plan_order,
            "generation_successful": self.generation_successful,
            "failure_reason": self.failure_reason,
        }


class MultiPlanGenerator:
    """
    Generates multiple candidate plans for a navigation scenario.

    Uses different cost functions to create diverse plans that
    trade off between path length and safety.
    """

    def __init__(self, config: Optional[PlanGenerationConfig] = None):
        """
        Initialize the generator.

        Args:
            config: Generation configuration.
        """
        self.config = config or PlanGenerationConfig()
        self.planner = AStarPlanner(
            grid_resolution=self.config.grid_resolution,
            safety_margin=self.config.safety_margin,
        )

    def generate_plans(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Dict[str, Any]],
        warehouse_dims: Tuple[float, float],
        scenario_id: Optional[str] = None,
    ) -> ScenarioResult:
        """
        Generate candidate plans for a scenario.

        Args:
            start: Starting position (x, y).
            goal: Goal position (x, y).
            obstacles: List of obstacles.
            warehouse_dims: Warehouse dimensions (width, height).
            scenario_id: Optional identifier for this scenario.

        Returns:
            ScenarioResult containing all candidate plans.
        """
        if scenario_id is None:
            # Generate deterministic ID from scenario parameters
            scenario_data = json.dumps({
                "start": start,
                "goal": goal,
                "obstacles": obstacles,
                "dims": warehouse_dims,
            }, sort_keys=True)
            scenario_id = hashlib.md5(scenario_data.encode()).hexdigest()[:8]

        candidate_plans = []
        failure_reasons = []

        # Generate plan with each cost function
        for cost_func in self.config.cost_functions:
            try:
                plan = self.planner.plan(
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    warehouse_dims=warehouse_dims,
                    cost_function=cost_func,
                )

                if plan is not None:
                    # Validate plan if configured
                    if self.config.validate_plans:
                        has_collision, _, _ = check_collision(
                            plan.path, obstacles
                        )
                        if has_collision:
                            failure_reasons.append(
                                f"{cost_func.value}: generated path has collision"
                            )
                            continue

                    candidate_plans.append(plan)
                else:
                    failure_reasons.append(
                        f"{cost_func.value}: no path found"
                    )
            except Exception as e:
                failure_reasons.append(
                    f"{cost_func.value}: {str(e)}"
                )

        # Determine optimal plan (ground truth)
        optimal_plan_id = self._determine_optimal_plan(candidate_plans)

        # Create plan order (potentially randomized)
        plan_order = [p.plan_id for p in candidate_plans]
        if self.config.randomize_order and len(plan_order) > 1:
            random.shuffle(plan_order)

        # Determine success
        generation_successful = len(candidate_plans) >= 2
        failure_reason = None
        if not generation_successful:
            failure_reason = "; ".join(failure_reasons) if failure_reasons else "Unknown"

        return ScenarioResult(
            scenario_id=scenario_id,
            start=start,
            goal=goal,
            obstacles=obstacles,
            warehouse_dims=warehouse_dims,
            candidate_plans=candidate_plans,
            optimal_plan_id=optimal_plan_id,
            plan_order=plan_order,
            generation_successful=generation_successful,
            failure_reason=failure_reason,
        )

    def _determine_optimal_plan(
        self,
        plans: List[CandidatePlan]
    ) -> str:
        """
        Determine which plan is objectively optimal.

        Uses a scoring function that balances:
        - Path length (lower is better)
        - Risk score (lower is better)

        Args:
            plans: List of candidate plans.

        Returns:
            plan_id of the optimal plan.
        """
        if not plans:
            return ""

        best_plan = None
        best_score = float('inf')

        for plan in plans:
            # Composite score: normalized length + risk
            # Lower is better
            length_score = plan.metadata.total_length / 10.0  # Normalize
            risk_score = plan.metadata.risk_score

            # Weight risk more heavily (safety is important)
            composite_score = length_score + 1.5 * risk_score

            if composite_score < best_score:
                best_score = composite_score
                best_plan = plan

        return best_plan.plan_id if best_plan else ""

    def get_plan_comparison(
        self,
        scenario: ScenarioResult
    ) -> Dict[str, Any]:
        """
        Get comparison data for all plans in a scenario.

        Useful for analysis and visualization.

        Args:
            scenario: ScenarioResult to analyze.

        Returns:
            Dictionary with comparison metrics.
        """
        if not scenario.candidate_plans:
            return {"error": "No plans to compare"}

        comparison = {
            "scenario_id": scenario.scenario_id,
            "num_plans": len(scenario.candidate_plans),
            "optimal_plan": scenario.optimal_plan_id,
            "plans": {}
        }

        # Compute relative metrics
        lengths = [p.metadata.total_length for p in scenario.candidate_plans]
        risks = [p.metadata.risk_score for p in scenario.candidate_plans]

        min_length = min(lengths)
        min_risk = min(risks)

        for plan in scenario.candidate_plans:
            comparison["plans"][plan.plan_id] = {
                "length": plan.metadata.total_length,
                "length_ratio": plan.metadata.total_length / min_length,
                "risk": plan.metadata.risk_score,
                "risk_ratio": plan.metadata.risk_score / min_risk if min_risk > 0 else 1.0,
                "min_clearance": plan.metadata.min_clearance,
                "estimated_time": plan.metadata.estimated_time,
                "is_optimal": plan.plan_id == scenario.optimal_plan_id,
            }

        return comparison


def create_test_scenario(
    complexity: str = "simple",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a test scenario for plan generation.

    Args:
        complexity: One of 'simple', 'moderate', 'complex', 'challenging'.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with scenario parameters.
    """
    if seed is not None:
        random.seed(seed)

    warehouse_dims = (20.0, 20.0)

    if complexity == "simple":
        # Few obstacles, clear path
        start = (2.0, 2.0)
        goal = (18.0, 18.0)
        obstacles = [
            {"position": [10.0, 5.0], "radius": 1.0},
            {"position": [10.0, 15.0], "radius": 1.0},
        ]

    elif complexity == "moderate":
        # More obstacles, multiple viable routes
        start = (2.0, 10.0)
        goal = (18.0, 10.0)
        obstacles = [
            {"position": [7.0, 10.0], "radius": 1.5},
            {"position": [10.0, 7.0], "radius": 1.2},
            {"position": [10.0, 13.0], "radius": 1.2},
            {"position": [13.0, 10.0], "radius": 1.5},
        ]

    elif complexity == "complex":
        # Dense obstacles, narrow passages
        start = (2.0, 2.0)
        goal = (18.0, 18.0)
        obstacles = [
            {"position": [5.0, 5.0], "radius": 1.5},
            {"position": [5.0, 10.0], "radius": 1.5},
            {"position": [5.0, 15.0], "radius": 1.5},
            {"position": [10.0, 5.0], "radius": 1.5},
            {"position": [10.0, 10.0], "radius": 2.0},
            {"position": [10.0, 15.0], "radius": 1.5},
            {"position": [15.0, 5.0], "radius": 1.5},
            {"position": [15.0, 10.0], "radius": 1.5},
            {"position": [15.0, 15.0], "radius": 1.5},
        ]

    elif complexity == "challenging":
        # Maze-like, requires careful planning
        start = (1.0, 1.0)
        goal = (19.0, 19.0)
        obstacles = []

        # Create wall-like obstacles
        for i in range(3, 17, 4):
            for j in range(0, 16, 2):
                if not (i == 7 and j in [8, 10]):  # Leave gap
                    obstacles.append({
                        "position": [float(i), float(j)],
                        "radius": 0.8
                    })
            for j in range(4, 20, 2):
                if not (i == 11 and j in [10, 12]):  # Leave gap
                    obstacles.append({
                        "position": [float(i + 2), float(j)],
                        "radius": 0.8
                    })

    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    return {
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "warehouse_dims": warehouse_dims,
        "complexity": complexity,
    }
