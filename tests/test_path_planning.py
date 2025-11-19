"""Unit tests for path planning module."""
import pytest
import numpy as np
from spatial_lab.coordination.path_planning import (
    SpatialPathPlanner,
    PathPoint
)


class TestSpatialPathPlanner:
    """Tests for the SpatialPathPlanner class."""

    @pytest.fixture
    def planner(self):
        """Create a path planner instance."""
        return SpatialPathPlanner(grid_resolution=0.5, obstacle_radius=0.5)

    @pytest.mark.asyncio
    async def test_straight_line_path_no_obstacles(self, planner):
        """Test that straight-line path is generated without obstacles."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)

        path = await planner.plan_path(start, goal)

        assert len(path) > 0
        assert path[0].x == pytest.approx(start[0], abs=0.1)
        assert path[0].y == pytest.approx(start[1], abs=0.1)
        assert path[-1].x == pytest.approx(goal[0], abs=0.1)
        assert path[-1].y == pytest.approx(goal[1], abs=0.1)

    @pytest.mark.asyncio
    async def test_path_with_obstacles(self, planner):
        """Test that path avoids obstacles."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)
        obstacles = [(5.0, 0.0)]  # Obstacle in the middle

        path = await planner.plan_path(start, goal, obstacles)

        assert len(path) > 0
        # Path should be longer than straight line due to obstacle
        path_length = planner.calculate_path_length(path)
        straight_distance = 10.0
        # With obstacle, path should be longer (going around)
        assert path_length >= straight_distance - 0.1

    @pytest.mark.asyncio
    async def test_same_start_and_goal(self, planner):
        """Test path when start equals goal."""
        start = (5.0, 5.0)
        goal = (5.0, 5.0)

        path = await planner.plan_path(start, goal)

        assert len(path) >= 1
        assert path[0].x == pytest.approx(start[0], abs=0.1)

    @pytest.mark.asyncio
    async def test_diagonal_path(self, planner):
        """Test diagonal path generation."""
        start = (0.0, 0.0)
        goal = (10.0, 10.0)

        path = await planner.plan_path(start, goal)

        assert len(path) > 0
        # Diagonal distance should be approximately sqrt(200)
        path_length = planner.calculate_path_length(path)
        expected = np.sqrt(200)
        assert path_length == pytest.approx(expected, rel=0.1)

    def test_calculate_path_length_empty(self, planner):
        """Test path length calculation with empty path."""
        path = []
        length = planner.calculate_path_length(path)
        assert length == 0.0

    def test_calculate_path_length_single_point(self, planner):
        """Test path length calculation with single point."""
        path = [PathPoint(0.0, 0.0)]
        length = planner.calculate_path_length(path)
        assert length == 0.0

    def test_calculate_path_length(self, planner):
        """Test path length calculation."""
        path = [
            PathPoint(0.0, 0.0),
            PathPoint(3.0, 4.0),  # 5 units from start
        ]
        length = planner.calculate_path_length(path)
        assert length == pytest.approx(5.0, abs=0.01)

    def test_is_path_clear_no_obstacles(self, planner):
        """Test path clearance check without obstacles."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)

        is_clear = planner.is_path_clear(start, goal, [])
        assert is_clear is True

    def test_is_path_clear_with_obstacle(self, planner):
        """Test path clearance check with obstacle in path."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)
        obstacles = [(5.0, 0.0)]  # Obstacle in the middle

        is_clear = planner.is_path_clear(start, goal, obstacles)
        assert is_clear is False

    def test_is_path_clear_obstacle_far(self, planner):
        """Test path clearance check with distant obstacle."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)
        obstacles = [(5.0, 10.0)]  # Obstacle far from path

        is_clear = planner.is_path_clear(start, goal, obstacles)
        assert is_clear is True

    def test_world_to_grid_conversion(self, planner):
        """Test world to grid coordinate conversion."""
        world_pos = (5.0, 10.0)
        grid_pos = planner._world_to_grid(world_pos)

        # With grid_resolution=0.5, (5.0, 10.0) -> (10, 20)
        assert grid_pos == (10, 20)

    def test_grid_to_world_conversion(self, planner):
        """Test grid to world coordinate conversion."""
        grid_pos = (10, 20)
        world_pos = planner._grid_to_world(grid_pos)

        # With grid_resolution=0.5, (10, 20) -> (5.25, 10.25)
        # Grid center = grid_pos * resolution + resolution/2
        assert world_pos[0] == pytest.approx(5.25, abs=0.01)
        assert world_pos[1] == pytest.approx(10.25, abs=0.01)

    @pytest.mark.asyncio
    async def test_multiple_obstacles(self, planner):
        """Test path planning with multiple obstacles."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)
        obstacles = [(3.0, 0.0), (5.0, 0.0), (7.0, 0.0)]

        path = await planner.plan_path(start, goal, obstacles)

        assert len(path) > 0
        # Should still reach goal
        assert path[-1].x == pytest.approx(goal[0], abs=0.5)
        assert path[-1].y == pytest.approx(goal[1], abs=0.5)


class TestPathPoint:
    """Tests for the PathPoint dataclass."""

    def test_path_point_creation(self):
        """Test PathPoint creation with default timestamp."""
        point = PathPoint(1.0, 2.0)
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.timestamp == 0.0

    def test_path_point_with_timestamp(self):
        """Test PathPoint creation with custom timestamp."""
        point = PathPoint(1.0, 2.0, 0.5)
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.timestamp == 0.5
