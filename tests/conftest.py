"""Shared pytest fixtures for Spatial Lab tests."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

# Configure asyncio event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock LLM Clients
@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API client."""
    client = MagicMock()
    client.robot_coordination_decision = AsyncMock(return_value={
        "action": "move_to",
        "target": (10.0, 20.0, 0.0),
        "confidence": 0.9
    })
    client.analyze_warehouse_layout = AsyncMock(return_value={
        "zones": ["picking", "packing", "storage"],
        "obstacles": []
    })
    return client


@pytest.fixture
def mock_llama_client():
    """Mock Llama API client."""
    client = MagicMock()
    client.robot_coordination_decision = AsyncMock(return_value={
        "action": "wait",
        "confidence": 0.7
    })
    return client


# Sample Environments
@pytest.fixture
def sample_warehouse_config() -> Dict[str, Any]:
    """Sample warehouse configuration."""
    return {
        "warehouse_width": 50,
        "warehouse_height": 50,
        "num_robots": 3,
        "num_shelves": 10,
        "num_items": 20,
        "communication_range": 10.0
    }


@pytest.fixture
def sample_robot_state() -> Dict[str, Any]:
    """Sample robot state."""
    return {
        "robot_id": "robot_001",
        "position": (10.0, 20.0, 0.0),
        "battery_level": 0.8,
        "carrying_item": None,
        "current_task": None,
        "status": "idle"
    }


# Environment Fixtures
@pytest.fixture
async def warehouse_environment(sample_warehouse_config):
    """Create warehouse environment for testing."""
    from spatial_lab.environments.warehouse_environment import WarehouseSpatialEnvironment

    env = WarehouseSpatialEnvironment(**sample_warehouse_config)
    await env.reset()
    yield env
    # Cleanup if needed


@pytest.fixture
def robot_fleet_simulator(sample_warehouse_config):
    """Create robot fleet simulator for testing."""
    from spatial_lab.coordination.robot_fleet import RobotFleetSimulator

    fleet = RobotFleetSimulator(
        num_robots=sample_warehouse_config["num_robots"],
        communication_range=sample_warehouse_config["communication_range"]
    )
    return fleet


# Utility Fixtures
@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
