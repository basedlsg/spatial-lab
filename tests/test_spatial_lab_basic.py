#!/usr/bin/env python3
"""
Basic test for Spatial AI Lab implementation
Tests core functionality without requiring full LLM integration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spatial_lab.config import ExperimentConfig
from spatial_lab.environments.warehouse_environment import WarehouseSpatialEnvironment, WarehouseSpatialEnvironmentConfig
from spatial_lab.environments.warehouse_layout import WarehouseLayoutGenerator
from spatial_lab.coordination.robot_fleet import RobotFleetSimulator
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_warehouse_layout():
    """Test warehouse layout generation"""
    logger.info("Testing warehouse layout generation...")
    
    layout_gen = WarehouseLayoutGenerator(width=30.0, height=20.0, num_shelves=10)
    layout = await layout_gen.generate_layout()
    
    assert layout is not None
    assert len(layout.shelves) == 10
    assert layout.dimensions[0] == 30.0  # width
    assert layout.dimensions[1] == 20.0  # height
    
    logger.info(f"✓ Layout generated: {len(layout.shelves)} shelves, {len(layout.aisles)} aisles")
    return layout


async def test_robot_fleet():
    """Test robot fleet simulation"""
    logger.info("Testing robot fleet simulation...")
    
    # Create simple layout
    layout_gen = WarehouseLayoutGenerator(width=20.0, height=15.0, num_shelves=5)
    layout = await layout_gen.generate_layout()
    
    # Initialize robot fleet
    robot_fleet = RobotFleetSimulator(num_robots=3, communication_range=5.0)
    await robot_fleet.initialize(layout)
    
    robot_states = robot_fleet.get_robot_states()
    assert len(robot_states) == 3
    
    logger.info(f"✓ Robot fleet initialized: {len(robot_states)} robots")
    
    # Test robot movement
    robot_id = robot_states[0].robot_id
    new_position = (5.0, 5.0, 0.0)
    success = await robot_fleet.move_robot(robot_id, new_position)
    
    logger.info(f"✓ Robot movement test: {'success' if success else 'failed'}")
    return robot_fleet


async def test_warehouse_environment():
    """Test warehouse environment setup"""
    logger.info("Testing warehouse environment...")
    
    # Create minimal config
    config = WarehouseSpatialEnvironmentConfig(
        warehouse_width=25.0,
        warehouse_height=15.0,
        num_robots=3,
        num_shelves=8,
        max_task_duration=50,
        items_per_task=5
    )
    
    # Create dummy server config for testing
    dummy_server_config = APIServerConfig(
        model_name="test_model",
        base_url="http://localhost:8000",
        api_key="test_key",
        num_requests_for_eval=1,
        max_tokens=100,
        temperature=0.7
    )
    
    # Create environment with dummy server config
    env = WarehouseSpatialEnvironment(config=config, server_configs=[dummy_server_config])
    await env.setup()
    
    # Test task generation
    task_item = await env.get_next_item()
    assert task_item is not None
    assert "task" in task_item["data"]
    assert "warehouse_layout" in task_item["data"]
    
    logger.info(f"✓ Environment setup complete, task generated: {task_item['item_id']}")
    return env


async def test_basic_coordination():
    """Test basic robot coordination without LLM"""
    logger.info("Testing basic robot coordination...")
    
    # Setup environment
    config = WarehouseSpatialEnvironmentConfig(
        warehouse_width=20.0,
        warehouse_height=15.0,
        num_robots=2,
        num_shelves=5,
        max_task_duration=30,
        items_per_task=3
    )
    
    # Create dummy server config for testing
    dummy_server_config = APIServerConfig(
        model_name="test_model",
        base_url="http://localhost:8000",
        api_key="test_key",
        num_requests_for_eval=1,
        max_tokens=100,
        temperature=0.7
    )
    
    env = WarehouseSpatialEnvironment(config=config, server_configs=[dummy_server_config])
    await env.setup()
    
    # Get robot observations
    observations = await env.get_robot_observations()
    assert len(observations) == 2
    
    logger.info(f"✓ Robot observations collected: {len(observations)} robots")
    
    # Test simple robot actions (without LLM decisions)
    simple_decisions = {}
    for robot_id in observations.keys():
        simple_decisions[robot_id] = {
            "action": "wait",
            "reasoning": "Basic test action",
            "parameters": {}
        }
    
    step_result = await env.execute_robot_actions(simple_decisions)
    assert step_result is not None
    
    logger.info("✓ Basic robot coordination test completed")
    return True


async def run_basic_tests():
    """Run all basic tests"""
    logger.info("Starting Spatial AI Lab basic tests...")
    
    try:
        # Test 1: Warehouse layout
        layout = await test_warehouse_layout()
        
        # Test 2: Robot fleet
        robot_fleet = await test_robot_fleet()
        
        # Test 3: Warehouse environment
        env = await test_warehouse_environment()
        
        # Test 4: Basic coordination
        coordination_success = await test_basic_coordination()
        
        logger.info("🎉 All basic tests passed!")
        
        # Print summary
        print("\n" + "="*60)
        print("SPATIAL AI LAB - BASIC TEST RESULTS")
        print("="*60)
        print(f"✓ Warehouse Layout Generation: PASSED")
        print(f"✓ Robot Fleet Simulation: PASSED") 
        print(f"✓ Environment Setup: PASSED")
        print(f"✓ Basic Coordination: PASSED")
        print("="*60)
        print("🚀 Spatial lab core functionality is working!")
        print("Ready for LLM integration and full experiments.")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_basic_tests())
    sys.exit(0 if success else 1) 