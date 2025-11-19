"""Unit tests for multi-agent coordinator module."""
import pytest
import time
from spatial_lab.coordination.multi_agent_coordinator import (
    MultiAgentCoordinator,
    CoordinationStrategy,
    CoordinationTask,
    AgentCapability,
    TaskStatus
)


class TestMultiAgentCoordinator:
    """Tests for the MultiAgentCoordinator class."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator instance."""
        return MultiAgentCoordinator(strategy=CoordinationStrategy.CENTRALIZED)

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent."""
        return AgentCapability(
            agent_id="robot_001",
            max_payload=10.0,
            speed=1.0,
            battery_level=0.9,
            current_location=(0.0, 0.0),
            available=True
        )

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return CoordinationTask(
            task_id="task_001",
            priority=1.0,
            estimated_duration=60.0,
            required_agents=1,
            location=(10.0, 10.0)
        )

    def test_register_agent(self, coordinator, sample_agent):
        """Test agent registration."""
        result = coordinator.register_agent(sample_agent)

        assert result is True
        assert "robot_001" in coordinator.agents
        assert coordinator.agents["robot_001"].battery_level == 0.9

    def test_register_duplicate_agent(self, coordinator, sample_agent):
        """Test registering same agent twice updates it."""
        coordinator.register_agent(sample_agent)

        # Update and re-register
        sample_agent.battery_level = 0.5
        result = coordinator.register_agent(sample_agent)

        assert result is True
        assert coordinator.agents["robot_001"].battery_level == 0.5

    def test_unregister_agent(self, coordinator, sample_agent):
        """Test agent unregistration."""
        coordinator.register_agent(sample_agent)
        result = coordinator.unregister_agent("robot_001")

        assert result is True
        assert "robot_001" not in coordinator.agents

    def test_unregister_nonexistent_agent(self, coordinator):
        """Test unregistering non-existent agent."""
        result = coordinator.unregister_agent("nonexistent")
        assert result is False

    def test_submit_task(self, coordinator, sample_task):
        """Test task submission."""
        result = coordinator.submit_task(sample_task)

        assert result is True
        assert "task_001" in coordinator.active_tasks
        assert coordinator.active_tasks["task_001"].status == TaskStatus.PENDING

    def test_submit_duplicate_task(self, coordinator, sample_task):
        """Test submitting duplicate task."""
        coordinator.submit_task(sample_task)
        result = coordinator.submit_task(sample_task)

        assert result is False

    def test_get_available_agents(self, coordinator):
        """Test getting available agents."""
        # Add multiple agents
        agent1 = AgentCapability("r1", 10.0, 1.0, 0.9, (0.0, 0.0), True)
        agent2 = AgentCapability("r2", 10.0, 1.0, 0.9, (5.0, 5.0), False)  # Not available
        agent3 = AgentCapability("r3", 10.0, 1.0, 0.05, (10.0, 10.0), True)  # Low battery

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        coordinator.register_agent(agent3)

        available = coordinator.get_available_agents()

        # Only agent1 should be available (agent2 not available, agent3 low battery)
        assert len(available) == 1
        assert available[0].agent_id == "r1"

    def test_get_pending_tasks(self, coordinator):
        """Test getting pending tasks sorted by priority."""
        task1 = CoordinationTask("t1", 1.0, 60.0, 1, (0.0, 0.0))
        task2 = CoordinationTask("t2", 3.0, 60.0, 1, (5.0, 5.0))  # Higher priority
        task3 = CoordinationTask("t3", 2.0, 60.0, 1, (10.0, 10.0))

        coordinator.submit_task(task1)
        coordinator.submit_task(task2)
        coordinator.submit_task(task3)

        pending = coordinator.get_pending_tasks()

        # Should be sorted by priority (highest first)
        assert len(pending) == 3
        assert pending[0].task_id == "t2"  # Priority 3.0
        assert pending[1].task_id == "t3"  # Priority 2.0
        assert pending[2].task_id == "t1"  # Priority 1.0

    @pytest.mark.asyncio
    async def test_allocate_tasks_single_agent(self, coordinator, sample_agent, sample_task):
        """Test task allocation with single agent."""
        coordinator.register_agent(sample_agent)
        coordinator.submit_task(sample_task)

        assignments = await coordinator.allocate_tasks()

        assert len(assignments) == 1
        assert assignments[0].agent_id == "robot_001"
        assert assignments[0].task_id == "task_001"
        assert sample_task.status == TaskStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_allocate_tasks_no_available_agents(self, coordinator, sample_task):
        """Test task allocation with no available agents."""
        coordinator.submit_task(sample_task)

        assignments = await coordinator.allocate_tasks()

        assert len(assignments) == 0
        assert sample_task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_allocate_tasks_multiple_required(self, coordinator):
        """Test task requiring multiple agents."""
        # Add 3 agents
        for i in range(3):
            agent = AgentCapability(f"r{i}", 10.0, 1.0, 0.9, (i*5.0, 0.0), True)
            coordinator.register_agent(agent)

        # Task requires 2 agents
        task = CoordinationTask("t1", 1.0, 60.0, 2, (10.0, 10.0))
        coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()

        assert len(assignments) == 2
        assert len(task.assigned_agents) == 2
        assert task.status == TaskStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_complete_task(self, coordinator, sample_agent, sample_task):
        """Test task completion."""
        coordinator.register_agent(sample_agent)
        coordinator.submit_task(sample_task)
        await coordinator.allocate_tasks()

        result = await coordinator.complete_task("task_001", success=True)

        assert result is True
        assert "task_001" not in coordinator.active_tasks
        assert "task_001" in coordinator.completed_tasks
        assert coordinator.agents["robot_001"].available is True

    @pytest.mark.asyncio
    async def test_task_dependencies(self, coordinator, sample_agent):
        """Test task dependency checking."""
        coordinator.register_agent(sample_agent)

        # Create dependent tasks
        task1 = CoordinationTask("t1", 1.0, 60.0, 1, (5.0, 5.0))
        task2 = CoordinationTask("t2", 2.0, 60.0, 1, (10.0, 10.0), dependencies=["t1"])

        coordinator.submit_task(task1)
        coordinator.submit_task(task2)

        # First allocation should only assign task1
        assignments = await coordinator.allocate_tasks()
        assigned_tasks = [a.task_id for a in assignments]

        assert "t1" in assigned_tasks
        # t2 has dependency on t1, should not be assigned
        assert task2.status == TaskStatus.PENDING

    def test_calculate_task_cost(self, coordinator, sample_agent):
        """Test task cost calculation."""
        task = CoordinationTask("t1", 1.0, 60.0, 1, (10.0, 0.0))

        cost = coordinator._calculate_task_cost(sample_agent, task)

        # Cost should include distance (10 units) plus other factors
        assert cost > 0
        assert cost < 100  # Reasonable upper bound

    def test_calculate_task_cost_low_battery(self, coordinator):
        """Test that low battery increases cost."""
        task = CoordinationTask("t1", 1.0, 60.0, 1, (10.0, 0.0))

        agent_high_battery = AgentCapability("r1", 10.0, 1.0, 0.9, (0.0, 0.0))
        agent_low_battery = AgentCapability("r2", 10.0, 1.0, 0.2, (0.0, 0.0))

        cost_high = coordinator._calculate_task_cost(agent_high_battery, task)
        cost_low = coordinator._calculate_task_cost(agent_low_battery, task)

        # Low battery should increase cost
        assert cost_low > cost_high

    def test_get_status(self, coordinator, sample_agent, sample_task):
        """Test coordinator status."""
        coordinator.register_agent(sample_agent)
        coordinator.submit_task(sample_task)

        status = coordinator.get_status()

        assert status["strategy"] == "centralized"
        assert status["total_agents"] == 1
        assert status["active_tasks"] == 1

    def test_update_agent_location(self, coordinator, sample_agent):
        """Test updating agent location."""
        coordinator.register_agent(sample_agent)

        result = coordinator.update_agent_location("robot_001", (50.0, 50.0))

        assert result is True
        assert coordinator.agents["robot_001"].current_location == (50.0, 50.0)

    def test_update_agent_battery(self, coordinator, sample_agent):
        """Test updating agent battery level."""
        coordinator.register_agent(sample_agent)

        result = coordinator.update_agent_battery("robot_001", 0.5)

        assert result is True
        assert coordinator.agents["robot_001"].battery_level == 0.5

    def test_update_agent_battery_clamp(self, coordinator, sample_agent):
        """Test battery level clamping."""
        coordinator.register_agent(sample_agent)

        # Test upper bound
        coordinator.update_agent_battery("robot_001", 1.5)
        assert coordinator.agents["robot_001"].battery_level == 1.0

        # Test lower bound
        coordinator.update_agent_battery("robot_001", -0.5)
        assert coordinator.agents["robot_001"].battery_level == 0.0


class TestCoordinationStrategies:
    """Tests for different coordination strategies."""

    @pytest.mark.asyncio
    async def test_auction_based_allocation(self):
        """Test auction-based task allocation."""
        coordinator = MultiAgentCoordinator(strategy=CoordinationStrategy.AUCTION_BASED)

        # Add agents at different distances
        agent1 = AgentCapability("r1", 10.0, 1.0, 0.9, (0.0, 0.0))
        agent2 = AgentCapability("r2", 10.0, 1.0, 0.9, (5.0, 5.0))  # Closer to task
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        task = CoordinationTask("t1", 1.0, 60.0, 1, (7.0, 7.0))
        coordinator.submit_task(task)

        assignments = await coordinator.allocate_tasks()

        # Agent2 should be assigned (closer)
        assert len(assignments) == 1
        assert assignments[0].agent_id == "r2"

    @pytest.mark.asyncio
    async def test_distributed_allocation(self):
        """Test distributed task allocation."""
        coordinator = MultiAgentCoordinator(strategy=CoordinationStrategy.DISTRIBUTED)

        # Add agents
        for i in range(3):
            agent = AgentCapability(f"r{i}", 10.0, 1.0, 0.9, (i*10.0, 0.0))
            coordinator.register_agent(agent)

        # Add tasks
        task1 = CoordinationTask("t1", 1.0, 60.0, 1, (5.0, 5.0))
        task2 = CoordinationTask("t2", 1.0, 60.0, 1, (15.0, 5.0))
        coordinator.submit_task(task1)
        coordinator.submit_task(task2)

        assignments = await coordinator.allocate_tasks()

        # Should assign both tasks
        assert len(assignments) == 2


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_agent_capability_defaults(self):
        """Test AgentCapability default values."""
        agent = AgentCapability(
            agent_id="r1",
            max_payload=10.0,
            speed=1.0,
            battery_level=0.9,
            current_location=(0.0, 0.0)
        )

        assert agent.available is True
        assert agent.current_task is None
        assert agent.skill_level == 1.0


class TestCoordinationTask:
    """Tests for CoordinationTask dataclass."""

    def test_task_defaults(self):
        """Test CoordinationTask default values."""
        task = CoordinationTask(
            task_id="t1",
            priority=1.0,
            estimated_duration=60.0,
            required_agents=1,
            location=(0.0, 0.0)
        )

        assert task.dependencies == []
        assert task.deadline is None
        assert task.status == TaskStatus.PENDING
        assert task.assigned_agents == []
