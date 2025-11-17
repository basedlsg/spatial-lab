"""
Spatial Path Planning for Warehouse Robot Navigation
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PathPoint:
    """Represents a point in the path."""
    x: float
    y: float
    timestamp: float = 0.0


class SpatialPathPlanner:
    """Path planning system for warehouse robot navigation."""
    
    def __init__(self, grid_resolution: float = 0.1):
        self.grid_resolution = grid_resolution
        
    async def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float]]] = None
    ) -> List[PathPoint]:
        """Plan a path from start to goal avoiding obstacles."""
        # Simple straight-line path for basic functionality
        if obstacles is None:
            obstacles = []

        # Generate waypoints along straight line
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
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            total_length += np.sqrt(dx*dx + dy*dy)
            
        return total_length
