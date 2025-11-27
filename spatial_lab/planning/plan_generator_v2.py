"""
Multi-Plan Generator v2

Generates multiple candidate plans for LLM-based plan selection experiments.

Key v2 changes:
1. Uses unified scoring function (compute_J) for determining optimal plan
2. Includes potentially unsafe plans (low clearance, may fail under noise)
3. Enforces diversity filtering to reject degenerate trials
4. Computes J scores for all plans transparently
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import random
import hashlib
import json
import math

from .astar_planner import (
    AStarPlanner,
    CandidatePlan,
    CostFunction,
    PlanMetadata,
    check_collision,
    interpolate_path,
)
from .scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    compute_J,
    find_optimal_plan,
    check_plan_diversity,
    compute_plan_scores,
    compute_risk_score_continuous,
)


@dataclass
class PlanGenerationConfigV2:
    """Configuration for v2 plan generation."""
    grid_resolution: float = 0.5
    safety_margin: float = 0.3

    # Whether to include an intentionally risky plan
    include_risky_plan: bool = True
    risky_plan_min_clearance_threshold: float = 0.3  # meters

    # Diversity requirements
    require_diversity: bool = True
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)

    # Plan presentation
    randomize_order: bool = True
    use_letter_ids: bool = True  # Use A, B, C instead of shortest_plan, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_resolution": self.grid_resolution,
            "safety_margin": self.safety_margin,
            "include_risky_plan": self.include_risky_plan,
            "risky_plan_min_clearance_threshold": self.risky_plan_min_clearance_threshold,
            "require_diversity": self.require_diversity,
            "scoring_config": self.scoring_config.to_dict(),
            "randomize_order": self.randomize_order,
            "use_letter_ids": self.use_letter_ids,
        }


@dataclass
class PlanWithScore:
    """A plan with its computed J score and safety status."""
    plan: CandidatePlan
    J_score: float
    is_potentially_unsafe: bool  # True if may fail under noisy execution
    failure_probability: float   # Estimated P(collision) under noise


@dataclass
class ScenarioResultV2:
    """Result of generating plans for a scenario (v2)."""
    scenario_id: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    obstacles: List[Dict[str, Any]]
    warehouse_dims: Tuple[float, float]

    # Plans with scores
    candidate_plans: List[PlanWithScore]
    optimal_plan_id: str
    optimal_J: float

    # Presentation order (potentially randomized, using letter IDs)
    plan_order: List[str]
    plan_id_mapping: Dict[str, str]  # Maps letter ID -> original plan type

    # Diversity info
    diversity_stats: Dict[str, Any]
    is_diverse: bool

    # Generation status
    generation_successful: bool
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": self.obstacles,
            "warehouse_dims": list(self.warehouse_dims),
            "candidate_plans": [
                {
                    "plan_id": p.plan.plan_id,
                    "J_score": round(p.J_score, 3),
                    "is_potentially_unsafe": p.is_potentially_unsafe,
                    "failure_probability": round(p.failure_probability, 3),
                    "metadata": p.plan.metadata.to_dict(),
                    "path": [(round(x, 2), round(y, 2)) for x, y in p.plan.path],
                }
                for p in self.candidate_plans
            ],
            "optimal_plan_id": self.optimal_plan_id,
            "optimal_J": round(self.optimal_J, 3),
            "plan_order": self.plan_order,
            "plan_id_mapping": self.plan_id_mapping,
            "diversity_stats": self.diversity_stats,
            "is_diverse": self.is_diverse,
            "generation_successful": self.generation_successful,
            "failure_reason": self.failure_reason,
        }

    def get_plan_by_id(self, plan_id: str) -> Optional[PlanWithScore]:
        """Get a plan by its ID (supports both letter and original IDs)."""
        # Check if it's a letter ID that needs mapping
        original_id = self.plan_id_mapping.get(plan_id, plan_id)
        for p in self.candidate_plans:
            if p.plan.plan_id == original_id or p.plan.plan_id == plan_id:
                return p
        return None


class MultiPlanGeneratorV2:
    """
    Generates multiple candidate plans for a navigation scenario.

    v2 improvements:
    - Uses unified scoring function
    - Generates risky plans that may fail
    - Enforces diversity requirements
    - Provides transparent J scores
    """

    def __init__(self, config: Optional[PlanGenerationConfigV2] = None):
        self.config = config or PlanGenerationConfigV2()

        # Standard planner for safe paths
        self.safe_planner = AStarPlanner(
            grid_resolution=self.config.grid_resolution,
            safety_margin=self.config.safety_margin,
        )

        # Risky planner with minimal safety margin
        self.risky_planner = AStarPlanner(
            grid_resolution=self.config.grid_resolution,
            safety_margin=0.05,  # Almost no margin
        )

    def generate_plans(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Dict[str, Any]],
        warehouse_dims: Tuple[float, float],
        scenario_id: Optional[str] = None,
    ) -> ScenarioResultV2:
        """Generate candidate plans with diversity and safety analysis."""

        if scenario_id is None:
            scenario_data = json.dumps({
                "start": start, "goal": goal,
                "obstacles": obstacles, "dims": warehouse_dims,
            }, sort_keys=True)
            scenario_id = hashlib.md5(scenario_data.encode()).hexdigest()[:8]

        plans_with_scores: List[PlanWithScore] = []
        failure_reasons = []

        # 1. Generate safe plans (shortest and safest)
        for cost_func in [CostFunction.SHORTEST, CostFunction.SAFEST]:
            try:
                plan = self.safe_planner.plan(
                    start=start, goal=goal,
                    obstacles=obstacles,
                    warehouse_dims=warehouse_dims,
                    cost_function=cost_func,
                )
                if plan:
                    scored = self._score_plan(plan, obstacles, is_risky_variant=False)
                    plans_with_scores.append(scored)
                else:
                    failure_reasons.append(f"{cost_func.value}: no path found")
            except Exception as e:
                failure_reasons.append(f"{cost_func.value}: {e}")

        # 2. Generate risky plan (if configured)
        if self.config.include_risky_plan:
            risky_plan = self._generate_risky_plan(
                start, goal, obstacles, warehouse_dims
            )
            if risky_plan:
                plans_with_scores.append(risky_plan)

        # 3. Check diversity
        plan_dicts = [
            {
                "plan_id": p.plan.plan_id,
                "length": p.plan.metadata.total_length,
                "risk_score": p.plan.metadata.risk_score,
            }
            for p in plans_with_scores
        ]
        is_diverse, diversity_stats = check_plan_diversity(
            plan_dicts, self.config.scoring_config
        )

        # 4. Find optimal plan using unified scoring
        optimal_id, optimal_J = find_optimal_plan(
            plan_dicts, self.config.scoring_config
        )

        # 5. Assign letter IDs and randomize order
        plan_id_mapping = {}
        if self.config.use_letter_ids:
            letters = ['A', 'B', 'C', 'D', 'E']
            original_ids = [p.plan.plan_id for p in plans_with_scores]

            if self.config.randomize_order:
                random.shuffle(original_ids)

            for i, orig_id in enumerate(original_ids):
                if i < len(letters):
                    plan_id_mapping[letters[i]] = orig_id

            plan_order = list(plan_id_mapping.keys())

            # Update optimal_id to letter if applicable
            for letter, orig in plan_id_mapping.items():
                if orig == optimal_id:
                    optimal_id = letter
                    break
        else:
            plan_order = [p.plan.plan_id for p in plans_with_scores]
            if self.config.randomize_order:
                random.shuffle(plan_order)
            plan_id_mapping = {pid: pid for pid in plan_order}

        # 6. Check success criteria
        generation_successful = (
            len(plans_with_scores) >= 2 and
            (is_diverse or not self.config.require_diversity)
        )

        failure_reason = None
        if not generation_successful:
            reasons = failure_reasons.copy()
            if not is_diverse and self.config.require_diversity:
                reasons.append(f"Insufficient diversity: {diversity_stats}")
            failure_reason = "; ".join(reasons) if reasons else "Unknown"

        return ScenarioResultV2(
            scenario_id=scenario_id,
            start=start,
            goal=goal,
            obstacles=obstacles,
            warehouse_dims=warehouse_dims,
            candidate_plans=plans_with_scores,
            optimal_plan_id=optimal_id,
            optimal_J=optimal_J,
            plan_order=plan_order,
            plan_id_mapping=plan_id_mapping,
            diversity_stats=diversity_stats,
            is_diverse=is_diverse,
            generation_successful=generation_successful,
            failure_reason=failure_reason,
        )

    def _generate_risky_plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Dict[str, Any]],
        warehouse_dims: Tuple[float, float],
    ) -> Optional[PlanWithScore]:
        """Generate a plan that passes close to obstacles (risky but shorter)."""

        # Try shortest path with minimal safety margin
        plan = self.risky_planner.plan(
            start=start, goal=goal,
            obstacles=obstacles,
            warehouse_dims=warehouse_dims,
            cost_function=CostFunction.SHORTEST,
        )

        if plan is None:
            return None

        # Check if this plan is actually risky (low clearance)
        if plan.metadata.min_clearance >= self.config.risky_plan_min_clearance_threshold:
            # Not risky enough - environment doesn't support risky paths
            return None

        # Rename to indicate it's the risky variant
        plan.plan_id = "risky_plan"

        return self._score_plan(plan, obstacles, is_risky_variant=True)

    def _score_plan(
        self,
        plan: CandidatePlan,
        obstacles: List[Dict[str, Any]],
        is_risky_variant: bool,
    ) -> PlanWithScore:
        """Score a plan and assess its safety."""

        # Compute J score
        J = compute_J(
            plan.metadata.total_length,
            plan.metadata.risk_score,
            self.config.scoring_config
        )

        # Estimate failure probability based on clearance
        # Plans with very low clearance are likely to fail under noise
        min_clear = plan.metadata.min_clearance
        if min_clear < 0.1:
            failure_prob = 0.8
        elif min_clear < 0.2:
            failure_prob = 0.5
        elif min_clear < 0.3:
            failure_prob = 0.3
        elif min_clear < 0.5:
            failure_prob = 0.1
        else:
            failure_prob = 0.02

        is_unsafe = is_risky_variant or failure_prob > 0.2

        return PlanWithScore(
            plan=plan,
            J_score=J,
            is_potentially_unsafe=is_unsafe,
            failure_probability=failure_prob,
        )


def create_diverse_scenario(
    complexity: str = "moderate",
    seed: Optional[int] = None,
    max_attempts: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Create a scenario that produces diverse plans.

    Tries multiple random configurations until one produces
    sufficiently diverse candidate plans.

    Args:
        complexity: Base complexity level.
        seed: Random seed.
        max_attempts: Max scenarios to try.

    Returns:
        Scenario dict if successful, None if no diverse scenario found.
    """
    if seed is not None:
        random.seed(seed)

    generator = MultiPlanGeneratorV2()

    for attempt in range(max_attempts):
        # Generate scenario with some randomization
        scenario = _create_randomized_scenario(complexity, seed=seed + attempt if seed else None)

        # Try to generate plans
        result = generator.generate_plans(
            start=scenario["start"],
            goal=scenario["goal"],
            obstacles=scenario["obstacles"],
            warehouse_dims=scenario["warehouse_dims"],
        )

        if result.generation_successful and result.is_diverse:
            scenario["_generation_result"] = result
            return scenario

    return None


def _create_randomized_scenario(
    complexity: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a scenario with controlled randomization."""

    if seed is not None:
        random.seed(seed)

    warehouse_dims = (20.0, 20.0)

    # Base configurations with some randomization
    if complexity == "simple":
        start = (2.0 + random.uniform(-1, 1), 2.0 + random.uniform(-1, 1))
        goal = (18.0 + random.uniform(-1, 1), 18.0 + random.uniform(-1, 1))
        num_obstacles = random.randint(3, 5)
        obstacle_radius_range = (0.8, 1.5)

    elif complexity == "moderate":
        start = (2.0 + random.uniform(-1, 1), 10.0 + random.uniform(-3, 3))
        goal = (18.0 + random.uniform(-1, 1), 10.0 + random.uniform(-3, 3))
        num_obstacles = random.randint(5, 8)
        obstacle_radius_range = (1.0, 2.0)

    elif complexity == "complex":
        start = (2.0, 2.0)
        goal = (18.0, 18.0)
        num_obstacles = random.randint(8, 12)
        obstacle_radius_range = (1.0, 2.5)

    elif complexity == "challenging":
        start = (1.0, 1.0)
        goal = (19.0, 19.0)
        num_obstacles = random.randint(12, 18)
        obstacle_radius_range = (0.8, 1.5)

    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    # Generate obstacles that create interesting choices
    obstacles = []
    for _ in range(num_obstacles):
        # Place obstacles along potential paths
        t = random.uniform(0.2, 0.8)  # Along start-goal line
        offset = random.uniform(-5, 5)  # Perpendicular offset

        # Interpolate along diagonal
        base_x = start[0] + t * (goal[0] - start[0])
        base_y = start[1] + t * (goal[1] - start[1])

        # Add perpendicular offset
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 1

        obs_x = base_x + offset * perp_x
        obs_y = base_y + offset * perp_y

        # Clamp to warehouse
        obs_x = max(2, min(18, obs_x))
        obs_y = max(2, min(18, obs_y))

        radius = random.uniform(*obstacle_radius_range)

        obstacles.append({
            "position": [obs_x, obs_y],
            "radius": radius,
        })

    return {
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "warehouse_dims": warehouse_dims,
        "complexity": complexity,
    }
