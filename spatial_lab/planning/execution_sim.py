"""
Execution Simulation

Simulates robot execution of selected plans with:
- Collision detection
- Path following
- Success/failure determination
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

from .astar_planner import CandidatePlan, check_collision, interpolate_path


@dataclass
class ExecutionResult:
    """Result of executing a plan."""
    plan_id: str
    success: bool
    final_position: Tuple[float, float]
    goal_position: Tuple[float, float]
    distance_to_goal: float
    reached_goal: bool
    collision_occurred: bool
    collision_point: Optional[Tuple[float, float]]
    colliding_obstacle: Optional[Dict[str, Any]]
    path_completion: float  # 0.0 to 1.0
    execution_time: float   # simulated seconds
    status: str            # "success", "collision", "timeout", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "final_position": list(self.final_position),
            "goal_position": list(self.goal_position),
            "distance_to_goal": round(self.distance_to_goal, 3),
            "reached_goal": self.reached_goal,
            "collision_occurred": self.collision_occurred,
            "collision_point": list(self.collision_point) if self.collision_point else None,
            "colliding_obstacle": self.colliding_obstacle,
            "path_completion": round(self.path_completion, 3),
            "execution_time": round(self.execution_time, 2),
            "status": self.status,
        }


class ExecutionSimulator:
    """
    Simulates execution of navigation plans.

    Provides deterministic execution results based on plan quality
    and obstacle configuration.
    """

    # Execution parameters
    ROBOT_SPEED = 1.0       # meters per second
    ROBOT_RADIUS = 0.5      # meters
    GOAL_TOLERANCE = 0.5    # meters - within this distance counts as reached
    STEP_SIZE = 0.1         # meters - simulation step size

    def __init__(
        self,
        robot_radius: float = 0.5,
        goal_tolerance: float = 0.5,
    ):
        """
        Initialize the simulator.

        Args:
            robot_radius: Robot collision radius.
            goal_tolerance: Distance threshold for goal reached.
        """
        self.robot_radius = robot_radius
        self.goal_tolerance = goal_tolerance

    def execute_plan(
        self,
        plan: CandidatePlan,
        obstacles: List[Dict[str, Any]],
    ) -> ExecutionResult:
        """
        Simulate execution of a plan.

        Args:
            plan: The plan to execute.
            obstacles: Current obstacle configuration.

        Returns:
            ExecutionResult with success/failure details.
        """
        # Interpolate path for fine-grained collision checking
        fine_path = interpolate_path(plan.path, step_size=self.STEP_SIZE)

        # Simulate following the path
        last_valid_pos = fine_path[0]
        collision_occurred = False
        collision_point = None
        colliding_obstacle = None
        path_completion = 0.0

        for i, pos in enumerate(fine_path):
            # Check for collision at this position
            has_collision = self._check_point_collision(pos, obstacles)

            if has_collision:
                collision_occurred = True
                collision_point = pos
                colliding_obstacle = self._get_colliding_obstacle(pos, obstacles)
                path_completion = i / len(fine_path)
                break

            last_valid_pos = pos
            path_completion = (i + 1) / len(fine_path)

        # Compute final results
        goal = plan.goal
        distance_to_goal = math.sqrt(
            (last_valid_pos[0] - goal[0]) ** 2 +
            (last_valid_pos[1] - goal[1]) ** 2
        )
        reached_goal = distance_to_goal <= self.goal_tolerance

        # Compute execution time
        path_length = self._compute_path_length(fine_path[:int(len(fine_path) * path_completion) + 1])
        execution_time = path_length / self.ROBOT_SPEED

        # Determine success and status
        if collision_occurred:
            success = False
            status = "collision"
        elif reached_goal:
            success = True
            status = "success"
        else:
            success = False
            status = "incomplete"

        return ExecutionResult(
            plan_id=plan.plan_id,
            success=success,
            final_position=last_valid_pos,
            goal_position=goal,
            distance_to_goal=distance_to_goal,
            reached_goal=reached_goal,
            collision_occurred=collision_occurred,
            collision_point=collision_point,
            colliding_obstacle=colliding_obstacle,
            path_completion=path_completion,
            execution_time=execution_time,
            status=status,
        )

    def _check_point_collision(
        self,
        pos: Tuple[float, float],
        obstacles: List[Dict[str, Any]]
    ) -> bool:
        """Check if a point collides with any obstacle."""
        for obs in obstacles:
            obs_pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0)

            dist = math.sqrt(
                (pos[0] - obs_pos[0]) ** 2 +
                (pos[1] - obs_pos[1]) ** 2
            )

            if dist < (radius + self.robot_radius):
                return True

        return False

    def _get_colliding_obstacle(
        self,
        pos: Tuple[float, float],
        obstacles: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get the obstacle that caused a collision."""
        for obs in obstacles:
            obs_pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0)

            dist = math.sqrt(
                (pos[0] - obs_pos[0]) ** 2 +
                (pos[1] - obs_pos[1]) ** 2
            )

            if dist < (radius + self.robot_radius):
                return obs

        return None

    def _compute_path_length(
        self,
        path: List[Tuple[float, float]]
    ) -> float:
        """Compute total path length."""
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            length += math.sqrt(dx ** 2 + dy ** 2)

        return length

    def validate_plan_selection(
        self,
        selected_plan_id: str,
        optimal_plan_id: str,
        plans: List[CandidatePlan],
        obstacles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate an LLM's plan selection.

        Args:
            selected_plan_id: Plan ID selected by LLM.
            optimal_plan_id: Ground truth optimal plan ID.
            plans: All candidate plans.
            obstacles: Obstacle configuration.

        Returns:
            Dictionary with validation results.
        """
        # Find selected plan
        selected_plan = None
        optimal_plan = None

        for plan in plans:
            if plan.plan_id == selected_plan_id:
                selected_plan = plan
            if plan.plan_id == optimal_plan_id:
                optimal_plan = plan

        if selected_plan is None:
            return {
                "valid_selection": False,
                "error": f"Selected plan '{selected_plan_id}' not found in candidates",
                "selection_correct": False,
                "execution_success": False,
            }

        # Execute selected plan
        execution_result = self.execute_plan(selected_plan, obstacles)

        # Determine selection correctness
        selection_correct = selected_plan_id == optimal_plan_id

        # Compute selection quality metrics
        if optimal_plan:
            length_ratio = selected_plan.metadata.total_length / optimal_plan.metadata.total_length
            risk_diff = selected_plan.metadata.risk_score - optimal_plan.metadata.risk_score
        else:
            length_ratio = 1.0
            risk_diff = 0.0

        return {
            "valid_selection": True,
            "selected_plan_id": selected_plan_id,
            "optimal_plan_id": optimal_plan_id,
            "selection_correct": selection_correct,
            "execution_result": execution_result.to_dict(),
            "execution_success": execution_result.success,
            "length_ratio": round(length_ratio, 3),
            "risk_difference": round(risk_diff, 3),
        }


def evaluate_accuracy(
    selected_plan_id: str,
    optimal_plan_id: str,
    execution_result: ExecutionResult,
) -> Dict[str, Any]:
    """
    Compute accuracy metrics for a single trial.

    Args:
        selected_plan_id: LLM's selected plan.
        optimal_plan_id: Ground truth optimal plan.
        execution_result: Result of executing selected plan.

    Returns:
        Dictionary with accuracy metrics.
    """
    # Selection accuracy: did LLM choose the optimal plan?
    selection_correct = selected_plan_id == optimal_plan_id

    # Execution accuracy: did the selected plan succeed?
    execution_success = execution_result.success

    # Combined accuracy: both correct selection AND successful execution
    combined_accuracy = selection_correct and execution_success

    return {
        "selection_correct": selection_correct,
        "execution_success": execution_success,
        "combined_accuracy": combined_accuracy,
        "reached_goal": execution_result.reached_goal,
        "collision_free": not execution_result.collision_occurred,
        "path_completion": execution_result.path_completion,
    }
