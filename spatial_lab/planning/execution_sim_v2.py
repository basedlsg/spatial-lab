"""
Execution Simulation v2

Key v2 changes:
1. Noisy execution - robot radius inflated by random noise
2. Separate success_exec (safety) from correct_opt (optimality)
3. Computes regret using unified scoring
4. Supports both deterministic and stochastic execution
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import random

from .astar_planner import CandidatePlan, interpolate_path
from .plan_generator_v2 import ScenarioResultV2, PlanWithScore
from .scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    compute_J,
    compute_regret,
    is_optimal_choice,
)


@dataclass
class ExecutionResultV2:
    """Result of executing a plan (v2)."""
    plan_id: str

    # Primary safety metric
    success_exec: bool        # Did execution complete without collision?

    # Execution details
    collision_occurred: bool
    collision_point: Optional[Tuple[float, float]]
    final_position: Tuple[float, float]
    goal_position: Tuple[float, float]
    reached_goal: bool
    path_completion: float

    # Scoring
    J_score: float
    regret: float            # J_chosen - J_optimal
    correct_opt: bool        # Is regret <= epsilon?

    # Execution parameters
    noise_applied: float     # How much radius inflation was applied
    execution_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "success_exec": self.success_exec,
            "collision_occurred": self.collision_occurred,
            "collision_point": list(self.collision_point) if self.collision_point else None,
            "final_position": list(self.final_position),
            "goal_position": list(self.goal_position),
            "reached_goal": self.reached_goal,
            "path_completion": round(self.path_completion, 3),
            "J_score": round(self.J_score, 3),
            "regret": round(self.regret, 3),
            "correct_opt": self.correct_opt,
            "noise_applied": round(self.noise_applied, 3),
            "execution_time": round(self.execution_time, 2),
        }


@dataclass
class ExecutionConfigV2:
    """Configuration for v2 execution simulation."""
    # Robot parameters
    base_robot_radius: float = 0.5
    goal_tolerance: float = 0.5
    step_size: float = 0.1
    robot_speed: float = 1.0

    # Noise parameters for realistic execution
    enable_noise: bool = True
    noise_std: float = 0.1        # Std dev of radius inflation
    max_noise: float = 0.2        # Max radius inflation

    # Scoring
    scoring_config: ScoringConfig = None

    def __post_init__(self):
        if self.scoring_config is None:
            self.scoring_config = DEFAULT_SCORING


class ExecutionSimulatorV2:
    """
    Simulates execution of navigation plans with noise.

    Key features:
    - Noisy execution (inflated robot radius)
    - Computes both safety (success_exec) and optimality (correct_opt)
    - Uses unified scoring for regret computation
    """

    def __init__(self, config: Optional[ExecutionConfigV2] = None):
        self.config = config or ExecutionConfigV2()

    def execute_plan(
        self,
        plan: PlanWithScore,
        obstacles: List[Dict[str, Any]],
        optimal_J: float,
        deterministic: bool = False,
    ) -> ExecutionResultV2:
        """
        Execute a plan with optional noise.

        Args:
            plan: The plan to execute (with J score).
            obstacles: Obstacle configuration.
            optimal_J: J score of the optimal plan (for regret).
            deterministic: If True, no noise applied.

        Returns:
            ExecutionResultV2 with success and regret metrics.
        """
        # Determine effective robot radius (with or without noise)
        if deterministic or not self.config.enable_noise:
            noise = 0.0
        else:
            noise = random.gauss(0, self.config.noise_std)
            noise = max(-self.config.max_noise, min(self.config.max_noise, noise))

        effective_radius = self.config.base_robot_radius + max(0, noise)

        # Interpolate path for fine-grained checking
        fine_path = interpolate_path(plan.plan.path, step_size=self.config.step_size)

        # Simulate execution
        last_valid_pos = fine_path[0]
        collision_occurred = False
        collision_point = None
        path_completion = 0.0

        for i, pos in enumerate(fine_path):
            if self._check_collision(pos, obstacles, effective_radius):
                collision_occurred = True
                collision_point = pos
                path_completion = i / len(fine_path)
                break

            last_valid_pos = pos
            path_completion = (i + 1) / len(fine_path)

        # Check goal reached
        goal = plan.plan.goal
        dist_to_goal = math.sqrt(
            (last_valid_pos[0] - goal[0]) ** 2 +
            (last_valid_pos[1] - goal[1]) ** 2
        )
        reached_goal = dist_to_goal <= self.config.goal_tolerance

        # Primary safety metric
        success_exec = (not collision_occurred) and reached_goal

        # Compute regret and optimality
        regret = compute_regret(plan.J_score, optimal_J)
        correct_opt = is_optimal_choice(
            plan.J_score, optimal_J, self.config.scoring_config
        )

        # Execution time
        executed_length = self._path_length(
            fine_path[:int(len(fine_path) * path_completion) + 1]
        )
        execution_time = executed_length / self.config.robot_speed

        return ExecutionResultV2(
            plan_id=plan.plan.plan_id,
            success_exec=success_exec,
            collision_occurred=collision_occurred,
            collision_point=collision_point,
            final_position=last_valid_pos,
            goal_position=goal,
            reached_goal=reached_goal,
            path_completion=path_completion,
            J_score=plan.J_score,
            regret=regret,
            correct_opt=correct_opt,
            noise_applied=noise,
            execution_time=execution_time,
        )

    def execute_scenario(
        self,
        scenario: ScenarioResultV2,
        selected_plan_id: str,
        deterministic: bool = False,
    ) -> Tuple[ExecutionResultV2, Dict[str, Any]]:
        """
        Execute a selected plan in a scenario.

        Args:
            scenario: The scenario with candidate plans.
            selected_plan_id: ID of the selected plan (letter or original).
            deterministic: If True, no noise applied.

        Returns:
            Tuple of (ExecutionResultV2, validation_dict).
        """
        # Find the selected plan
        selected = scenario.get_plan_by_id(selected_plan_id)

        if selected is None:
            # Return failure result
            return ExecutionResultV2(
                plan_id=selected_plan_id,
                success_exec=False,
                collision_occurred=False,
                collision_point=None,
                final_position=scenario.start,
                goal_position=scenario.goal,
                reached_goal=False,
                path_completion=0.0,
                J_score=float('inf'),
                regret=float('inf'),
                correct_opt=False,
                noise_applied=0.0,
                execution_time=0.0,
            ), {"valid_selection": False, "error": f"Plan {selected_plan_id} not found"}

        # Execute the plan
        result = self.execute_plan(
            plan=selected,
            obstacles=scenario.obstacles,
            optimal_J=scenario.optimal_J,
            deterministic=deterministic,
        )

        # Build validation dict
        validation = {
            "valid_selection": True,
            "selected_plan_id": selected_plan_id,
            "optimal_plan_id": scenario.optimal_plan_id,
            "success_exec": result.success_exec,
            "correct_opt": result.correct_opt,
            "regret": result.regret,
            "collision_occurred": result.collision_occurred,
        }

        return result, validation

    def _check_collision(
        self,
        pos: Tuple[float, float],
        obstacles: List[Dict[str, Any]],
        robot_radius: float,
    ) -> bool:
        """Check for collision at a position."""
        for obs in obstacles:
            obs_pos = obs.get("position", obs.get("pos", [0, 0]))
            obs_radius = obs.get("radius", 1.0)

            dist = math.sqrt(
                (pos[0] - obs_pos[0]) ** 2 +
                (pos[1] - obs_pos[1]) ** 2
            )

            if dist < (obs_radius + robot_radius):
                return True

        return False

    def _path_length(self, path: List[Tuple[float, float]]) -> float:
        """Compute path length."""
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            length += math.sqrt(dx ** 2 + dy ** 2)

        return length


def compute_trial_metrics(
    confidence: float,
    execution_result: ExecutionResultV2,
) -> Dict[str, Any]:
    """
    Compute all metrics for a single trial.

    Args:
        confidence: LLM's stated confidence (0.0-1.0).
        execution_result: Result of executing the selected plan.

    Returns:
        Dictionary with all trial metrics.
    """
    return {
        # LLM outputs
        "confidence": confidence,

        # Primary metrics (for calibration)
        "success_exec": execution_result.success_exec,
        "correct_opt": execution_result.correct_opt,

        # Secondary metrics
        "regret": execution_result.regret,
        "collision_occurred": execution_result.collision_occurred,
        "reached_goal": execution_result.reached_goal,
        "path_completion": execution_result.path_completion,

        # Derived calibration targets
        "calibration_target_safety": int(execution_result.success_exec),
        "calibration_target_optimality": int(execution_result.correct_opt),
    }
