"""
Spatial Path Planning for Warehouse Robot Navigation

Implements A* pathfinding algorithm with obstacle avoidance for warehouse robots.
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass, field


@dataclass
class PathPoint:
    """Represents a point in the path."""
    x: float
    y: float
    timestamp: float = 0.0


@dataclass(order=True)
class AStarNode:
    """Node for A* algorithm priority queue."""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['AStarNode'] = field(default=None, compare=False)


class SpatialPathPlanner:
    """Path planning system for warehouse robot navigation with A* algorithm."""

    def __init__(self, grid_resolution: float = 0.5, obstacle_radius: float = 0.5):
        """
        Initialize the path planner.

        Args:
            grid_resolution: Size of each grid cell in meters
            obstacle_radius: Safety radius around obstacles
        """
        self.grid_resolution = grid_resolution
        self.obstacle_radius = obstacle_radius

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        return (
            int(world_pos[0] / self.grid_resolution),
            int(world_pos[1] / self.grid_resolution)
        )

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        return (
            grid_pos[0] * self.grid_resolution + self.grid_resolution / 2,
            grid_pos[1] * self.grid_resolution + self.grid_resolution / 2
        )

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean) between two grid positions."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions (8-directional movement)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((pos[0] + dx, pos[1] + dy))
        return neighbors

    def _create_obstacle_grid(
        self,
        obstacles: List[Tuple[float, float]],
        grid_bounds: Tuple[int, int, int, int]
    ) -> Set[Tuple[int, int]]:
        """
        Create a set of blocked grid cells from obstacle positions.

        Args:
            obstacles: List of obstacle positions in world coordinates
            grid_bounds: (min_x, min_y, max_x, max_y) in grid coordinates
        """
        blocked = set()

        # Calculate radius in grid cells
        radius_cells = int(np.ceil(self.obstacle_radius / self.grid_resolution))

        for obs in obstacles:
            obs_grid = self._world_to_grid(obs)

            # Block cells within obstacle radius
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    cell = (obs_grid[0] + dx, obs_grid[1] + dy)
                    # Check if cell is within distance
                    dist = np.sqrt(dx**2 + dy**2) * self.grid_resolution
                    if dist <= self.obstacle_radius:
                        blocked.add(cell)

        return blocked

    async def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float]]] = None
    ) -> List[PathPoint]:
        """
        Plan a path from start to goal avoiding obstacles using A* algorithm.

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            obstacles: List of obstacle positions to avoid

        Returns:
            List of PathPoints from start to goal
        """
        if obstacles is None:
            obstacles = []

        # If no obstacles, use simple straight-line path for efficiency
        if not obstacles:
            return self._straight_line_path(start, goal)

        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # If start equals goal, return single point
        if start_grid == goal_grid:
            return [PathPoint(start[0], start[1], 0.0)]

        # Calculate grid bounds with padding
        padding = 10
        min_x = min(start_grid[0], goal_grid[0]) - padding
        min_y = min(start_grid[1], goal_grid[1]) - padding
        max_x = max(start_grid[0], goal_grid[0]) + padding
        max_y = max(start_grid[1], goal_grid[1]) + padding

        # Create obstacle grid
        blocked = self._create_obstacle_grid(obstacles, (min_x, min_y, max_x, max_y))

        # Check if start or goal is blocked
        if start_grid in blocked or goal_grid in blocked:
            # Fall back to straight line if can't plan around obstacles
            return self._straight_line_path(start, goal)

        # A* algorithm
        open_set: List[AStarNode] = []
        closed_set: Set[Tuple[int, int]] = set()
        g_scores: Dict[Tuple[int, int], float] = {start_grid: 0}

        # Initialize with start node
        start_node = AStarNode(
            f_score=self._heuristic(start_grid, goal_grid),
            position=start_grid,
            g_score=0
        )
        heapq.heappush(open_set, start_node)

        # Track best path to each node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            current = heapq.heappop(open_set)

            # Skip if already processed
            if current.position in closed_set:
                continue

            # Check if reached goal
            if current.position == goal_grid:
                # Reconstruct path
                return self._reconstruct_path(came_from, current.position, start, goal)

            closed_set.add(current.position)

            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current.position):
                # Skip if blocked or already processed
                if neighbor_pos in blocked or neighbor_pos in closed_set:
                    continue

                # Calculate tentative g_score
                # Diagonal movement costs sqrt(2), orthogonal costs 1
                dx = abs(neighbor_pos[0] - current.position[0])
                dy = abs(neighbor_pos[1] - current.position[1])
                move_cost = np.sqrt(2) if (dx + dy) == 2 else 1.0

                tentative_g = current.g_score + move_cost

                # Check if this path is better
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    came_from[neighbor_pos] = current.position

                    f_score = tentative_g + self._heuristic(neighbor_pos, goal_grid)
                    neighbor_node = AStarNode(
                        f_score=f_score,
                        position=neighbor_pos,
                        g_score=tentative_g
                    )
                    heapq.heappush(open_set, neighbor_node)

        # No path found - fall back to straight line
        return self._straight_line_path(start, goal)

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float]
    ) -> List[PathPoint]:
        """Reconstruct the path from A* result."""
        # Build path in reverse
        grid_path = [current]
        while current in came_from:
            current = came_from[current]
            grid_path.append(current)
        grid_path.reverse()

        # Convert to world coordinates
        path = []
        total_length = len(grid_path)

        for i, grid_pos in enumerate(grid_path):
            if i == 0:
                # Use exact start position
                world_pos = start_world
            elif i == total_length - 1:
                # Use exact goal position
                world_pos = goal_world
            else:
                # Use grid center
                world_pos = self._grid_to_world(grid_pos)

            timestamp = i / max(1, total_length - 1)
            path.append(PathPoint(world_pos[0], world_pos[1], timestamp))

        # Smooth path by removing unnecessary waypoints
        return self._smooth_path(path)

    def _smooth_path(self, path: List[PathPoint]) -> List[PathPoint]:
        """Remove unnecessary waypoints from path (simple smoothing)."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            prev = smoothed[-1]
            curr = path[i]
            next_pt = path[i + 1]

            # Calculate direction change
            dir1 = (curr.x - prev.x, curr.y - prev.y)
            dir2 = (next_pt.x - curr.x, next_pt.y - curr.y)

            # Normalize
            len1 = np.sqrt(dir1[0]**2 + dir1[1]**2)
            len2 = np.sqrt(dir2[0]**2 + dir2[1]**2)

            if len1 > 0.001 and len2 > 0.001:
                dir1 = (dir1[0] / len1, dir1[1] / len1)
                dir2 = (dir2[0] / len2, dir2[1] / len2)

                # Calculate dot product (cosine of angle)
                dot = dir1[0] * dir2[0] + dir1[1] * dir2[1]

                # Keep point if direction changes significantly (> 15 degrees)
                if dot < 0.966:  # cos(15°) ≈ 0.966
                    smoothed.append(curr)
            else:
                smoothed.append(curr)

        smoothed.append(path[-1])

        # Update timestamps
        for i, point in enumerate(smoothed):
            point.timestamp = i / max(1, len(smoothed) - 1)

        return smoothed

    def _straight_line_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[PathPoint]:
        """Generate a simple straight-line path (no obstacle avoidance)."""
        distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        num_points = max(2, int(distance / self.grid_resolution))

        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append(PathPoint(x, y, t))

        return path

    def calculate_path_length(self, path: List[PathPoint]) -> float:
        """Calculate total path length in world units."""
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            total_length += np.sqrt(dx*dx + dy*dy)

        return total_length

    def is_path_clear(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]]
    ) -> bool:
        """Check if a straight path between two points is clear of obstacles."""
        if not obstacles:
            return True

        # Sample points along the line
        distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        num_samples = max(2, int(distance / (self.grid_resolution / 2)))

        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])

            # Check distance to each obstacle
            for obs in obstacles:
                dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
                if dist < self.obstacle_radius:
                    return False

        return True
