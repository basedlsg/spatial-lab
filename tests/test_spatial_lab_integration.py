#!/usr/bin/env python3
"""
Integration test for Spatial AI Research Lab

Tests the integration between the spatial lab and Nous Atropos infrastructure.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src").absolute()))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_integration():
    """Test basic integration between spatial lab and Atropos"""
    
    logger.info("Testing Spatial AI Research Lab integration...")
    
    try:
        # Test imports
        logger.info("Testing imports...")
        from spatial_lab.config import get_config, ExperimentConfig
        from spatial_lab.environments import WarehouseSpatialEnvironment, WarehouseSpatialEnvironmentConfig
        from spatial_lab.evaluation import SpatialMetricsCalculator
        
        logger.info("✓ All imports successful")
        
        # Test configuration
        logger.info("Testing configuration system...")
        config = get_config("basic_warehouse")
        assert isinstance(config, ExperimentConfig)
        assert config.num_robots > 0
        assert config.warehouse_width > 0
        logger.info("✓ Configuration system working")
        
        # Test environment creation
        logger.info("Testing environment creation...")
        env_config = WarehouseSpatialEnvironmentConfig(
            warehouse_width=20.0,
            warehouse_height=15.0,
            num_robots=2,
            num_shelves=5,
            max_task_duration=50,
            items_per_task=3
        )
        
        # Create minimal server config for testing
        from atroposlib.envs.server_handling.server_baseline import APIServerConfig
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url=None,
                api_key="test-key",  # Test key
                num_requests_for_eval=4,
            )
        ]
        
        environment = WarehouseSpatialEnvironment(
            config=env_config,
            server_configs=server_configs
        )
        
        logger.info("✓ Environment created successfully")
        
        # Test environment setup (without actual LLM calls)
        logger.info("Testing environment setup...")
        await environment.setup()
        logger.info("✓ Environment setup completed")
        
        # Test task generation
        logger.info("Testing task generation...")
        task_item = await environment.get_next_item()
        assert task_item is not None
        assert "task" in task_item.data
        assert "warehouse_layout" in task_item.data
        logger.info("✓ Task generation working")
        
        # Test metrics calculator
        logger.info("Testing metrics calculator...")
        metrics_calc = SpatialMetricsCalculator()
        
        # Create dummy step result for testing
        dummy_step_result = {
            "robot_actions": {
                "robot_001": {"action": "move_to", "success": True},
                "robot_002": {"action": "wait", "success": True}
            },
            "collisions": [],
            "items_collected": ["item_001"],
            "items_delivered": [],
            "coordination_events": [],
            "progress_made": 0.5
        }
        
        step_metrics = await metrics_calc.calculate_step_metrics(dummy_step_result)
        assert "navigation_success_rate" in step_metrics
        assert "collision_count" in step_metrics
        logger.info("✓ Metrics calculation working")
        
        logger.info("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration_variations():
    """Test different configuration variations"""
    
    logger.info("Testing configuration variations...")
    
    try:
        from spatial_lab.config import create_custom_config
        
        # Test custom configuration
        custom_config = create_custom_config(
            base_config="basic_warehouse",
            **{
                "environment.num_robots": 4,
                "tasks.complexity": "medium",
                "training.num_episodes": 10
            }
        )
        
        assert custom_config.num_robots == 4
        assert custom_config.task_complexity == "medium"
        assert custom_config.num_training_episodes == 10
        
        logger.info("✓ Custom configuration working")
        
        # Test configuration validation
        from spatial_lab.config import validate_config
        warnings = validate_config(custom_config)
        logger.info(f"Configuration warnings: {len(warnings)}")
        
        logger.info("✓ Configuration validation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_warehouse_layout():
    """Test warehouse layout generation"""
    
    logger.info("Testing warehouse layout generation...")
    
    try:
        from spatial_lab.environments.warehouse_layout import WarehouseLayoutGenerator
        
        # Create layout generator
        layout_gen = WarehouseLayoutGenerator(
            width=30.0,
            height=20.0,
            num_shelves=8
        )
        
        # Generate layout
        layout = await layout_gen.generate_layout("standard")
        
        assert layout is not None
        assert len(layout.shelves) > 0
        assert len(layout.zones) > 0
        assert layout.dimensions == (30.0, 20.0)
        
        # Test position validation
        valid_pos = layout.is_position_valid((5.0, 5.0))
        assert isinstance(valid_pos, bool)
        
        logger.info("✓ Warehouse layout generation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Warehouse layout test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    
    logger.info("Starting Spatial AI Research Lab Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_basic_integration,
        test_configuration_variations,
        test_warehouse_layout
    ]
    
    results = []
    
    for test in tests:
        logger.info(f"\nRunning {test.__name__}...")
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{i+1}. {test.__name__}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! The Spatial AI Research Lab is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 