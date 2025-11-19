"""Integration tests for Spatial Lab system."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from spatial_lab.coordination.path_planning import SpatialPathPlanner
from spatial_lab.coordination.multi_agent_coordinator import (
    MultiAgentCoordinator,
    CoordinationStrategy,
    CoordinationTask,
    AgentCapability,
    TaskStatus
)
from spatial_lab.coordination.communication import (
    RobotCommunicationSystem,
    RobotMessage,
    MessageType
)


class TestPathPlanningIntegration:
    """Integration tests for path planning with obstacles."""

    @pytest.fixture
    def planner(self):
        return SpatialPathPlanner(grid_resolution=0.5, obstacle_radius=0.5)

    @pytest.mark.asyncio
    async def test_complex_obstacle_course(self, planner):
        """Test navigation through complex obstacle field."""
        start = (0.0, 0.0)
        goal = (20.0, 20.0)

        # Create obstacle field
        obstacles = [
            (5.0, 5.0), (5.0, 6.0), (5.0, 7.0),  # Wall 1
            (10.0, 10.0), (11.0, 10.0), (12.0, 10.0),  # Wall 2
            (15.0, 15.0), (15.0, 16.0),  # Wall 3
        ]

        path = await planner.plan_path(start, goal, obstacles)

        # Should find a path
        assert len(path) > 0
        assert path[0].x == pytest.approx(start[0], abs=0.5)
        assert path[-1].x == pytest.approx(goal[0], abs=0.5)

        # Path should avoid obstacles
        for point in path:
            for obs in obstacles:
                dist = ((point.x - obs[0])**2 + (point.y - obs[1])**2)**0.5
                # Should maintain safe distance (with some tolerance)
                assert dist >= planner.obstacle_radius * 0.8

    @pytest.mark.asyncio
    async def test_path_efficiency(self, planner):
        """Test that A* produces efficient paths."""
        start = (0.0, 0.0)
        goal = (10.0, 0.0)

        # Single obstacle requiring small detour
        obstacles = [(5.0, 0.0)]

        path = await planner.plan_path(start, goal, obstacles)
        path_length = planner.calculate_path_length(path)

        # Path should be reasonably efficient (not too much longer than direct)
        direct_distance = 10.0
        assert path_length < direct_distance * 1.5  # At most 50% longer


class TestCoordinatorIntegration:
    """Integration tests for multi-agent coordination."""

    @pytest.fixture
    def coordinator(self):
        return MultiAgentCoordinator(strategy=CoordinationStrategy.CENTRALIZED)

    @pytest.fixture
    def fleet_of_agents(self, coordinator):
        """Create a fleet of 5 agents."""
        agents = []
        for i in range(5):
            agent = AgentCapability(
                agent_id=f"robot_{i:03d}",
                max_payload=10.0,
                speed=1.0,
                battery_level=0.9,
                current_location=(i * 5.0, 0.0),
                available=True,
                skill_level=0.8 + i * 0.05
            )
            coordinator.register_agent(agent)
            agents.append(agent)
        return agents

    @pytest.mark.asyncio
    async def test_multiple_task_allocation(self, coordinator, fleet_of_agents):
        """Test allocating multiple tasks to fleet."""
        # Create multiple tasks
        tasks = [
            CoordinationTask(f"task_{i}", priority=float(i),
                           estimated_duration=30.0, required_agents=1,
                           location=(10.0 + i * 5, 10.0))
            for i in range(3)
        ]

        for task in tasks:
            coordinator.submit_task(task)

        # Allocate all tasks
        assignments = await coordinator.allocate_tasks()

        # Should assign all 3 tasks
        assert len(assignments) == 3

        # Each task should be assigned to different agent
        assigned_agents = [a.agent_id for a in assignments]
        assert len(set(assigned_agents)) == 3

    @pytest.mark.asyncio
    async def test_task_completion_workflow(self, coordinator, fleet_of_agents):
        """Test complete task workflow from submission to completion."""
        # Submit task
        task = CoordinationTask(
            task_id="workflow_test",
            priority=1.0,
            estimated_duration=60.0,
            required_agents=1,
            location=(10.0, 10.0)
        )
        coordinator.submit_task(task)

        # Allocate
        assignments = await coordinator.allocate_tasks()
        assert len(assignments) == 1
        assert task.status == TaskStatus.ASSIGNED

        # Execute
        result = await coordinator.execute_task("workflow_test")
        assert result["success"] is True
        assert task.status == TaskStatus.IN_PROGRESS

        # Complete
        success = await coordinator.complete_task("workflow_test", success=True)
        assert success is True
        assert "workflow_test" in coordinator.completed_tasks
        assert len(coordinator.task_history) == 1

        # Agent should be available again
        assigned_agent_id = assignments[0].agent_id
        assert coordinator.agents[assigned_agent_id].available is True

    @pytest.mark.asyncio
    async def test_priority_based_allocation(self, coordinator, fleet_of_agents):
        """Test that higher priority tasks get assigned first."""
        # Submit tasks with different priorities
        low_priority = CoordinationTask("low", 1.0, 60.0, 1, (10.0, 10.0))
        high_priority = CoordinationTask("high", 10.0, 60.0, 1, (15.0, 15.0))

        coordinator.submit_task(low_priority)
        coordinator.submit_task(high_priority)

        # Get pending tasks
        pending = coordinator.get_pending_tasks()

        # High priority should be first
        assert pending[0].task_id == "high"
        assert pending[1].task_id == "low"

    @pytest.mark.asyncio
    async def test_collaborative_task(self, coordinator, fleet_of_agents):
        """Test task requiring multiple agents."""
        # Task requiring 3 agents
        task = CoordinationTask(
            task_id="collaborative",
            priority=5.0,
            estimated_duration=120.0,
            required_agents=3,
            location=(25.0, 25.0)
        )
        coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()

        # Should assign 3 agents
        assert len(assignments) == 3
        assert len(task.assigned_agents) == 3
        assert task.status == TaskStatus.ASSIGNED


class TestCommunicationIntegration:
    """Integration tests for robot communication."""

    @pytest.fixture
    def comm_system(self):
        return RobotCommunicationSystem(max_range=10.0)

    @pytest.mark.asyncio
    async def test_message_delivery(self, comm_system):
        """Test message sending and receiving."""
        # Register robots
        comm_system.register_robot("robot_1", (0.0, 0.0))
        comm_system.register_robot("robot_2", (5.0, 0.0))

        # Send message
        import time
        message = RobotMessage(
            sender_id="robot_1",
            receiver_id="robot_2",
            message_type=MessageType.COORDINATION,
            content={"action": "wait"},
            timestamp=time.time()
        )

        success = await comm_system.send_message(message)
        assert success is True

        # Receive message
        messages = comm_system.receive_messages("robot_2")
        assert len(messages) == 1
        assert messages[0].content["action"] == "wait"

    @pytest.mark.asyncio
    async def test_out_of_range_communication(self, comm_system):
        """Test that out-of-range communication fails."""
        # Register robots far apart
        comm_system.register_robot("robot_1", (0.0, 0.0))
        comm_system.register_robot("robot_2", (100.0, 0.0))  # Far away

        import time
        message = RobotMessage(
            sender_id="robot_1",
            receiver_id="robot_2",
            message_type=MessageType.STATUS_UPDATE,
            content={},
            timestamp=time.time()
        )

        success = await comm_system.send_message(message)
        assert success is False


class TestFullSystemIntegration:
    """Full system integration tests."""

    @pytest.mark.asyncio
    async def test_path_planning_with_coordination(self):
        """Test path planning integrated with coordination."""
        planner = SpatialPathPlanner()
        coordinator = MultiAgentCoordinator()

        # Register agents
        for i in range(3):
            agent = AgentCapability(
                agent_id=f"robot_{i}",
                max_payload=10.0,
                speed=1.0,
                battery_level=0.9,
                current_location=(i * 5.0, 0.0)
            )
            coordinator.register_agent(agent)

        # Submit task
        task = CoordinationTask(
            task_id="integrated_task",
            priority=1.0,
            estimated_duration=60.0,
            required_agents=1,
            location=(20.0, 20.0)
        )
        coordinator.submit_task(task)

        # Allocate task
        assignments = await coordinator.allocate_tasks()
        assert len(assignments) == 1

        # Plan path for assigned agent
        assigned_agent = coordinator.agents[assignments[0].agent_id]
        path = await planner.plan_path(
            assigned_agent.current_location,
            task.location
        )

        assert len(path) > 0
        assert path[-1].x == pytest.approx(task.location[0], abs=0.5)

    @pytest.mark.asyncio
    async def test_multi_agent_pathfinding(self):
        """Test multiple agents planning paths simultaneously."""
        planner = SpatialPathPlanner()

        # Multiple agents with different start/goal pairs
        agent_configs = [
            ((0.0, 0.0), (10.0, 10.0)),
            ((20.0, 0.0), (10.0, 20.0)),
            ((0.0, 20.0), (20.0, 20.0)),
        ]

        # Plan all paths concurrently
        paths = await asyncio.gather(*[
            planner.plan_path(start, goal)
            for start, goal in agent_configs
        ])

        # All paths should be valid
        for i, path in enumerate(paths):
            start, goal = agent_configs[i]
            assert len(path) > 0
            assert path[0].x == pytest.approx(start[0], abs=0.5)
            assert path[-1].x == pytest.approx(goal[0], abs=0.5)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_fleet(self):
        """Test coordinator with no agents."""
        coordinator = MultiAgentCoordinator()
        task = CoordinationTask("lonely_task", 1.0, 60.0, 1, (10.0, 10.0))
        coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()
        assert len(assignments) == 0
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_more_tasks_than_agents(self):
        """Test when there are more tasks than available agents."""
        coordinator = MultiAgentCoordinator()

        # Only 2 agents
        for i in range(2):
            agent = AgentCapability(f"robot_{i}", 10.0, 1.0, 0.9, (i * 5.0, 0.0))
            coordinator.register_agent(agent)

        # 5 tasks
        for i in range(5):
            task = CoordinationTask(f"task_{i}", 1.0, 60.0, 1, (10.0, i * 5.0))
            coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()

        # Should only assign 2 tasks
        assert len(assignments) == 2
        # 3 tasks should remain pending
        assert len(coordinator.get_pending_tasks()) == 3

    @pytest.mark.asyncio
    async def test_zero_length_path(self):
        """Test path from point to itself."""
        planner = SpatialPathPlanner()
        path = await planner.plan_path((5.0, 5.0), (5.0, 5.0))

        assert len(path) >= 1
        assert planner.calculate_path_length(path) == pytest.approx(0.0, abs=0.1)


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_fleet_allocation(self):
        """Test allocation with large fleet."""
        coordinator = MultiAgentCoordinator()

        # 50 agents
        for i in range(50):
            agent = AgentCapability(
                f"robot_{i:03d}", 10.0, 1.0, 0.9,
                (i % 10 * 5.0, i // 10 * 5.0)
            )
            coordinator.register_agent(agent)

        # 30 tasks
        for i in range(30):
            task = CoordinationTask(
                f"task_{i:03d}", float(i % 5), 60.0, 1,
                (i % 10 * 3.0, i // 10 * 3.0)
            )
            coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()

        # Should assign all 30 tasks
        assert len(assignments) == 30

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_path_planning(self):
        """Test path planning over long distances."""
        planner = SpatialPathPlanner(grid_resolution=1.0)

        start = (0.0, 0.0)
        goal = (100.0, 100.0)

        path = await planner.plan_path(start, goal)

        assert len(path) > 0
        path_length = planner.calculate_path_length(path)
        # Should be approximately diagonal distance
        expected = ((100**2 + 100**2)**0.5)
        assert path_length == pytest.approx(expected, rel=0.1)
