"""
Command-line interface for Spatial Lab.

Provides commands for running demos, tests, and diagnostics.
"""

import argparse
import asyncio
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def run_demo():
    """Run a demonstration of the Spatial Lab system."""
    from spatial_lab.coordination.path_planning import SpatialPathPlanner
    from spatial_lab.coordination.multi_agent_coordinator import (
        MultiAgentCoordinator,
        CoordinationStrategy,
        CoordinationTask,
        AgentCapability
    )
    from spatial_lab.coordination.communication import RobotCommunicationSystem

    print("=" * 60)
    print("Spatial Lab Demo")
    print("=" * 60)
    print()

    # Demo 1: Path Planning
    print("1. Path Planning Demo")
    print("-" * 40)

    planner = SpatialPathPlanner(grid_resolution=0.5, obstacle_radius=0.5)

    start = (0.0, 0.0)
    goal = (10.0, 10.0)
    obstacles = [(5.0, 5.0), (5.0, 6.0), (6.0, 5.0)]

    print(f"   Planning path from {start} to {goal}")
    print(f"   Avoiding {len(obstacles)} obstacles")

    path = await planner.plan_path(start, goal, obstacles)
    path_length = planner.calculate_path_length(path)

    print(f"   Path found with {len(path)} waypoints")
    print(f"   Total path length: {path_length:.2f} units")
    print()

    # Demo 2: Multi-Agent Coordination
    print("2. Multi-Agent Coordination Demo")
    print("-" * 40)

    coordinator = MultiAgentCoordinator(strategy=CoordinationStrategy.CENTRALIZED)

    # Register agents
    for i in range(3):
        agent = AgentCapability(
            agent_id=f"robot_{i:03d}",
            max_payload=10.0,
            speed=1.0,
            battery_level=0.9,
            current_location=(i * 5.0, 0.0),
            available=True
        )
        coordinator.register_agent(agent)
        print(f"   Registered {agent.agent_id} at {agent.current_location}")

    # Submit tasks
    tasks = [
        CoordinationTask("task_1", priority=2.0, estimated_duration=60.0,
                        required_agents=1, location=(15.0, 10.0)),
        CoordinationTask("task_2", priority=1.0, estimated_duration=45.0,
                        required_agents=1, location=(10.0, 15.0)),
    ]

    for task in tasks:
        coordinator.submit_task(task)
        print(f"   Submitted {task.task_id} at {task.location} (priority: {task.priority})")

    # Allocate tasks
    assignments = await coordinator.allocate_tasks()
    print(f"\n   Task allocation results:")
    for assignment in assignments:
        print(f"   - {assignment.task_id} -> {assignment.agent_id} (cost: {assignment.cost:.2f})")

    print()

    # Demo 3: Communication System
    print("3. Communication System Demo")
    print("-" * 40)

    comm = RobotCommunicationSystem(max_range=10.0)

    # Register robots
    comm.register_robot("sender", (0.0, 0.0))
    comm.register_robot("receiver", (5.0, 0.0))

    print(f"   Registered sender at (0, 0)")
    print(f"   Registered receiver at (5, 0)")
    print(f"   Communication range: 10.0 units")

    # Check if communication is possible
    from spatial_lab.coordination.communication import RobotMessage, MessageType
    import time

    message = RobotMessage(
        sender_id="sender",
        receiver_id="receiver",
        message_type=MessageType.COORDINATION,
        content={"action": "move_to", "target": (10.0, 10.0)},
        timestamp=time.time()
    )

    success = await comm.send_message(message)
    print(f"   Message sent: {success}")

    if success:
        received = comm.receive_messages("receiver")
        print(f"   Messages received by receiver: {len(received)}")

    print()
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


async def run_diagnostics():
    """Run system diagnostics to verify installation."""
    print("=" * 60)
    print("Spatial Lab Diagnostics")
    print("=" * 60)
    print()

    results = []

    # Check core imports
    print("Checking core modules...")

    try:
        from spatial_lab.coordination.path_planning import SpatialPathPlanner, PathPoint
        print("  [OK] path_planning")
        results.append(("path_planning", True))
    except ImportError as e:
        print(f"  [FAIL] path_planning: {e}")
        results.append(("path_planning", False))

    try:
        from spatial_lab.coordination.multi_agent_coordinator import (
            MultiAgentCoordinator, CoordinationStrategy, CoordinationTask,
            AgentCapability, TaskAssignment, TaskStatus
        )
        print("  [OK] multi_agent_coordinator")
        results.append(("multi_agent_coordinator", True))
    except ImportError as e:
        print(f"  [FAIL] multi_agent_coordinator: {e}")
        results.append(("multi_agent_coordinator", False))

    try:
        from spatial_lab.coordination.communication import (
            RobotCommunicationSystem, RobotMessage, MessageType
        )
        print("  [OK] communication")
        results.append(("communication", True))
    except ImportError as e:
        print(f"  [FAIL] communication: {e}")
        results.append(("communication", False))

    try:
        from spatial_lab.coordination.robot_fleet import RobotFleetSimulator
        print("  [OK] robot_fleet")
        results.append(("robot_fleet", True))
    except ImportError as e:
        print(f"  [FAIL] robot_fleet: {e}")
        results.append(("robot_fleet", False))

    try:
        from spatial_lab.environments.warehouse_environment import WarehouseEnvironment
        print("  [OK] warehouse_environment")
        results.append(("warehouse_environment", True))
    except ImportError as e:
        print(f"  [FAIL] warehouse_environment: {e}")
        results.append(("warehouse_environment", False))

    print()

    # Run basic functionality tests
    print("Running basic functionality tests...")

    try:
        planner = SpatialPathPlanner()
        path = await planner.plan_path((0, 0), (5, 5))
        assert len(path) > 0
        print("  [OK] Path planning works")
        results.append(("path_planning_test", True))
    except Exception as e:
        print(f"  [FAIL] Path planning: {e}")
        results.append(("path_planning_test", False))

    try:
        coordinator = MultiAgentCoordinator()
        agent = AgentCapability("test_agent", 10.0, 1.0, 0.9, (0, 0))
        coordinator.register_agent(agent)
        assert "test_agent" in coordinator.agents
        print("  [OK] Coordinator works")
        results.append(("coordinator_test", True))
    except Exception as e:
        print(f"  [FAIL] Coordinator: {e}")
        results.append(("coordinator_test", False))

    try:
        comm = RobotCommunicationSystem()
        comm.register_robot("test", (0, 0))
        assert "test" in comm.robot_positions
        print("  [OK] Communication works")
        results.append(("communication_test", True))
    except Exception as e:
        print(f"  [FAIL] Communication: {e}")
        results.append(("communication_test", False))

    print()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    print("=" * 60)
    print(f"Diagnostics complete: {passed}/{total} checks passed")

    if passed == total:
        print("Status: All systems operational!")
        return 0
    else:
        print("Status: Some issues detected")
        return 1


def show_info():
    """Show information about Spatial Lab."""
    print("=" * 60)
    print("Spatial Lab - Multi-Agent Warehouse Robotics")
    print("=" * 60)
    print()
    print("Version: 1.0.0")
    print()
    print("Components:")
    print("  - Path Planning: A* algorithm with obstacle avoidance")
    print("  - Multi-Agent Coordinator: Task allocation and scheduling")
    print("  - Communication System: Range-based robot messaging")
    print("  - Robot Fleet: Fleet management and state tracking")
    print("  - Warehouse Environment: Simulation environment")
    print()
    print("Usage:")
    print("  spatial-lab demo        Run interactive demo")
    print("  spatial-lab check       Run system diagnostics")
    print("  spatial-lab info        Show this information")
    print()
    print("Documentation:")
    print("  See README.md for getting started")
    print("  See CODEBASE_ANALYSIS.md for architecture details")
    print("  See PROJECT_STATUS.md for current status")
    print()


def main(args: Optional[list] = None):
    """Main entry point for Spatial Lab CLI."""
    parser = argparse.ArgumentParser(
        description="Spatial Lab - Multi-Agent Warehouse Robotics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    subparsers.add_parser("demo", help="Run interactive demo")

    # Check command
    subparsers.add_parser("check", help="Run system diagnostics")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    parsed_args = parser.parse_args(args)

    setup_logging(parsed_args.verbose)

    if parsed_args.command == "demo":
        asyncio.run(run_demo())
        return 0
    elif parsed_args.command == "check":
        return asyncio.run(run_diagnostics())
    elif parsed_args.command == "info":
        show_info()
        return 0
    else:
        # Default: show info
        show_info()
        return 0


if __name__ == "__main__":
    sys.exit(main())
