"""
A* Path Planner with Multiple Cost Functions

Provides A*-based path planning for warehouse navigation with support
for multiple cost function variants to generate diverse candidate plans.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
import math


class CostFunction(Enum):
    """Cost function variants for path planning."""
    SHORTEST = "shortest"      # Minimize path length
    SAFEST = "safest"          # Maximize obstacle clearance
    BALANCED = "balanced"      # Balance length and safety


@dataclass
class PlanMetadata:
    """Metadata about a candidate plan."""
    cost_function: CostFunction
    total_length: float
    min_clearance: float
    avg_clearance: float
    num_waypoints: int
    estimated_time: float  # seconds at nominal speed
    risk_score: float      # 0.0 (safe) to 1.0 (risky)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cost_function": self.cost_function.value,
            "total_length": round(self.total_length, 2),
            "min_clearance": round(self.min_clearance, 2),
            "avg_clearance": round(self.avg_clearance, 2),
            "num_waypoints": self.num_waypoints,
            "estimated_time": round(self.estimated_time, 2),
            "risk_score": round(self.risk_score, 3),
        }


@dataclass
class CandidatePlan:
    """A candidate navigation plan with path and metadata."""
    plan_id: str
    path: List[Tuple[float, float]]
    metadata: PlanMetadata
    start: Tuple[float, float]
    goal: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "path": [(round(x, 2), round(y, 2)) for x, y in self.path],
            "start": (round(self.start[0], 2), round(self.start[1], 2)),
            "goal": (round(self.goal[0], 2), round(self.goal[1], 2)),
            "metadata": self.metadata.to_dict(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary for LLM prompts."""
        return (
            f"Plan {self.plan_id}: {self.metadata.cost_function.value} approach, "
            f"length={self.metadata.total_length:.1f}m, "
            f"min_clearance={self.metadata.min_clearance:.1f}m, "
            f"risk={self.metadata.risk_score:.2f}, "
            f"time={self.metadata.estimated_time:.1f}s"
        )


@dataclass(order=True)
class AStarNode:
    """Node for A* priority queue."""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['AStarNode'] = field(default=None, compare=False)


class AStarPlanner:
    """
    A* path planner supporting multiple cost functions.

    The planner operates on a discretized grid but returns continuous
    coordinates for the final path.
    """

    # Robot parameters
    ROBOT_RADIUS = 0.5  # meters
    NOMINAL_SPEED = 1.0  # meters per second

    def __init__(
        self,
        grid_resolution: float = 0.5,
        safety_margin: float = 0.3,
    ):
        """
        Initialize the A* planner.

        Args:
            grid_resolution: Size of each grid cell in meters.
            safety_margin: Additional clearance beyond robot radius.
        """
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.min_clearance = self.ROBOT_RADIUS + safety_margin

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Dict[str, Any]],
        warehouse_dims: Tuple[float, float],
        cost_function: CostFunction = CostFunction.SHORTEST,
    ) -> Optional[CandidatePlan]:
        """
        Plan a path from start to goal using A*.

        Args:
            start: Starting position (x, y) in meters.
            goal: Goal position (x, y) in meters.
            obstacles: List of obstacles with 'position' and 'radius' keys.
            warehouse_dims: Warehouse dimensions (width, height) in meters.
            cost_function: Which cost function to use.

        Returns:
            CandidatePlan if path found, None otherwise.
        """
        # Convert to grid coordinates
        start_grid = self._to_grid(start)
        goal_grid = self._to_grid(goal)

        # Build obstacle map
        grid_width = int(warehouse_dims[0] / self.grid_resolution) + 1
        grid_height = int(warehouse_dims[1] / self.grid_resolution) + 1

        obstacle_map = self._build_obstacle_map(
            obstacles, grid_width, grid_height
        )
        clearance_map = self._build_clearance_map(
            obstacles, grid_width, grid_height
        )

        # Check start and goal are valid
        if not self._is_valid(start_grid, obstacle_map, grid_width, grid_height):
            return None
        if not self._is_valid(goal_grid, obstacle_map, grid_width, grid_height):
            return None

        # Run A*
        path_grid = self._astar(
            start_grid, goal_grid,
            obstacle_map, clearance_map,
            grid_width, grid_height,
            cost_function
        )

        if path_grid is None:
            return None

        # Convert back to continuous coordinates and smooth
        path_continuous = [self._to_continuous(p) for p in path_grid]
        path_smoothed = self._smooth_path(path_continuous, obstacles)

        # Compute metadata
        metadata = self._compute_metadata(
            path_smoothed, obstacles, cost_function
        )

        return CandidatePlan(
            plan_id=f"{cost_function.value}_plan",
            path=path_smoothed,
            metadata=metadata,
            start=start,
            goal=goal,
        )

    def _to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert continuous position to grid coordinates."""
        return (
            int(pos[0] / self.grid_resolution),
            int(pos[1] / self.grid_resolution)
        )

    def _to_continuous(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to continuous position."""
        return (
            pos[0] * self.grid_resolution,
            pos[1] * self.grid_resolution
        )

    def _build_obstacle_map(
        self,
        obstacles: List[Dict[str, Any]],
        width: int,
        height: int
    ) -> Set[Tuple[int, int]]:
        """Build set of obstacle grid cells."""
        obstacle_cells = set()

        for obs in obstacles:
            pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0) + self.min_clearance

            # Convert to grid
            cx, cy = pos[0], pos[1]
            grid_radius = int(radius / self.grid_resolution) + 1

            # Mark all cells within radius
            gx, gy = int(cx / self.grid_resolution), int(cy / self.grid_resolution)
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    cell = (gx + dx, gy + dy)
                    if 0 <= cell[0] < width and 0 <= cell[1] < height:
                        # Check actual distance
                        cell_pos = self._to_continuous(cell)
                        dist = math.sqrt(
                            (cell_pos[0] - cx) ** 2 +
                            (cell_pos[1] - cy) ** 2
                        )
                        if dist < radius:
                            obstacle_cells.add(cell)

        return obstacle_cells

    def _build_clearance_map(
        self,
        obstacles: List[Dict[str, Any]],
        width: int,
        height: int
    ) -> Dict[Tuple[int, int], float]:
        """Build map of minimum clearance at each cell."""
        clearance_map = {}

        for x in range(width):
            for y in range(height):
                pos = self._to_continuous((x, y))
                min_clear = float('inf')

                for obs in obstacles:
                    obs_pos = obs.get("position", obs.get("pos", [0, 0]))
                    radius = obs.get("radius", 1.0)

                    dist = math.sqrt(
                        (pos[0] - obs_pos[0]) ** 2 +
                        (pos[1] - obs_pos[1]) ** 2
                    )
                    clearance = dist - radius - self.ROBOT_RADIUS
                    min_clear = min(min_clear, clearance)

                # Also consider walls
                wall_clear = min(pos[0], pos[1],
                                width * self.grid_resolution - pos[0],
                                height * self.grid_resolution - pos[1])
                wall_clear -= self.ROBOT_RADIUS
                min_clear = min(min_clear, wall_clear)

                clearance_map[(x, y)] = max(0, min_clear)

        return clearance_map

    def _is_valid(
        self,
        pos: Tuple[int, int],
        obstacle_map: Set[Tuple[int, int]],
        width: int,
        height: int
    ) -> bool:
        """Check if a grid position is valid."""
        x, y = pos
        return (
            0 <= x < width and
            0 <= y < height and
            pos not in obstacle_map
        )

    def _get_neighbors(
        self,
        pos: Tuple[int, int],
        obstacle_map: Set[Tuple[int, int]],
        width: int,
        height: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring cells with movement cost."""
        x, y = pos
        neighbors = []

        # 8-connected grid
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if self._is_valid(neighbor, obstacle_map, width, height):
                # Diagonal moves cost sqrt(2)
                cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                neighbors.append((neighbor, cost))

        return neighbors

    def _heuristic(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """Octile distance heuristic."""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _compute_edge_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        base_cost: float,
        clearance_map: Dict[Tuple[int, int], float],
        cost_function: CostFunction
    ) -> float:
        """Compute edge cost based on cost function."""
        if cost_function == CostFunction.SHORTEST:
            return base_cost * self.grid_resolution

        elif cost_function == CostFunction.SAFEST:
            # Penalize low clearance heavily
            clearance = clearance_map.get(to_pos, 0)
            if clearance < 0.1:
                safety_penalty = 10.0
            elif clearance < 0.5:
                safety_penalty = 5.0
            elif clearance < 1.0:
                safety_penalty = 2.0
            else:
                safety_penalty = 1.0
            return base_cost * self.grid_resolution * safety_penalty

        elif cost_function == CostFunction.BALANCED:
            # Moderate penalty for low clearance
            clearance = clearance_map.get(to_pos, 0)
            if clearance < 0.5:
                safety_penalty = 2.0
            elif clearance < 1.0:
                safety_penalty = 1.5
            else:
                safety_penalty = 1.0
            return base_cost * self.grid_resolution * safety_penalty

        return base_cost * self.grid_resolution

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: Set[Tuple[int, int]],
        clearance_map: Dict[Tuple[int, int], float],
        width: int,
        height: int,
        cost_function: CostFunction
    ) -> Optional[List[Tuple[int, int]]]:
        """Run A* algorithm."""
        open_set = []
        start_node = AStarNode(
            f_score=self._heuristic(start, goal),
            position=start,
            g_score=0.0
        )
        heapq.heappush(open_set, start_node)

        g_scores = {start: 0.0}
        came_from = {start: None}

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal:
                # Reconstruct path
                path = []
                pos = goal
                while pos is not None:
                    path.append(pos)
                    pos = came_from[pos]
                return list(reversed(path))

            for neighbor, base_cost in self._get_neighbors(
                current.position, obstacle_map, width, height
            ):
                edge_cost = self._compute_edge_cost(
                    current.position, neighbor, base_cost,
                    clearance_map, cost_function
                )
                tentative_g = g_scores[current.position] + edge_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current.position
                    f_score = tentative_g + self._heuristic(neighbor, goal)

                    new_node = AStarNode(
                        f_score=f_score,
                        position=neighbor,
                        g_score=tentative_g
                    )
                    heapq.heappush(open_set, new_node)

        return None  # No path found

    def _smooth_path(
        self,
        path: List[Tuple[float, float]],
        obstacles: List[Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """Smooth path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip waypoints
            best_next = current_idx + 1

            for check_idx in range(len(path) - 1, current_idx, -1):
                if self._line_of_sight(
                    path[current_idx], path[check_idx], obstacles
                ):
                    best_next = check_idx
                    break

            smoothed.append(path[best_next])
            current_idx = best_next

        return smoothed

    def _line_of_sight(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        obstacles: List[Dict[str, Any]]
    ) -> bool:
        """Check if there's a clear line of sight between two points."""
        # Sample points along line
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        num_samples = max(int(dist / (self.grid_resolution / 2)), 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            # Check against obstacles
            for obs in obstacles:
                obs_pos = obs.get("position", obs.get("pos", [0, 0]))
                radius = obs.get("radius", 1.0) + self.min_clearance

                obs_dist = math.sqrt(
                    (x - obs_pos[0]) ** 2 +
                    (y - obs_pos[1]) ** 2
                )
                if obs_dist < radius:
                    return False

        return True

    def _compute_metadata(
        self,
        path: List[Tuple[float, float]],
        obstacles: List[Dict[str, Any]],
        cost_function: CostFunction
    ) -> PlanMetadata:
        """Compute metadata for a path."""
        # Path length
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total_length += math.sqrt(dx ** 2 + dy ** 2)

        # Clearances
        clearances = []
        for x, y in path:
            min_clear = float('inf')
            for obs in obstacles:
                obs_pos = obs.get("position", obs.get("pos", [0, 0]))
                radius = obs.get("radius", 1.0)
                dist = math.sqrt(
                    (x - obs_pos[0]) ** 2 +
                    (y - obs_pos[1]) ** 2
                )
                clearance = dist - radius - self.ROBOT_RADIUS
                min_clear = min(min_clear, clearance)
            clearances.append(max(0, min_clear))

        min_clearance = min(clearances) if clearances else 0.0
        avg_clearance = sum(clearances) / len(clearances) if clearances else 0.0

        # Risk score based on clearance
        if min_clearance > 2.0:
            risk_score = 0.1
        elif min_clearance > 1.0:
            risk_score = 0.3
        elif min_clearance > 0.5:
            risk_score = 0.5
        elif min_clearance > 0.2:
            risk_score = 0.7
        else:
            risk_score = 0.9

        return PlanMetadata(
            cost_function=cost_function,
            total_length=total_length,
            min_clearance=min_clearance,
            avg_clearance=avg_clearance,
            num_waypoints=len(path),
            estimated_time=total_length / self.NOMINAL_SPEED,
            risk_score=risk_score,
        )


def check_collision(
    path: List[Tuple[float, float]],
    obstacles: List[Dict[str, Any]],
    robot_radius: float = 0.5
) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    """
    Check if a path collides with any obstacles.

    Args:
        path: List of waypoints (x, y).
        obstacles: List of obstacles with 'position' and 'radius'.
        robot_radius: Robot collision radius.

    Returns:
        Tuple of (has_collision, collision_waypoint_idx, colliding_obstacle).
    """
    for i, (x, y) in enumerate(path):
        for obs in obstacles:
            obs_pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0)

            dist = math.sqrt(
                (x - obs_pos[0]) ** 2 +
                (y - obs_pos[1]) ** 2
            )

            if dist < (radius + robot_radius):
                return True, i, obs

    return False, None, None


def interpolate_path(
    path: List[Tuple[float, float]],
    step_size: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Interpolate a path with fine-grained waypoints.

    Args:
        path: Original path waypoints.
        step_size: Distance between interpolated points.

    Returns:
        Interpolated path with more waypoints.
    """
    if len(path) < 2:
        return path

    interpolated = []

    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        num_steps = max(int(dist / step_size), 1)

        for j in range(num_steps):
            t = j / num_steps
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            interpolated.append((x, y))

    interpolated.append(path[-1])
    return interpolated
