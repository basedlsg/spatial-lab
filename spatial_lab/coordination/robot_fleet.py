"""
Robot Fleet Simulator

Manages a fleet of robots in the warehouse environment, handling movement,
collision detection, task assignment, and coordination.
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

from .path_planning import SpatialPathPlanner
from .communication import RobotCommunicationSystem

logger = logging.getLogger(__name__)


class RobotStatus(str, Enum):
    """Robot operational status"""
    IDLE = "idle"
    MOVING = "moving"
    PICKING = "picking"
    DROPPING = "dropping"
    WAITING = "waiting"
    CHARGING = "charging"
    ERROR = "error"


@dataclass
class RobotState:
    """Complete state of a warehouse robot"""
    robot_id: str
    position: Tuple[float, float, float]
    orientation: float  # radians
    target_position: Optional[Tuple[float, float, float]] = None
    
    # Physical state
    battery_level: float = 1.0
    carrying_item: Optional[str] = None
    max_load_capacity: float = 10.0
    current_load: float = 0.0
    
    # Operational state
    status: RobotStatus = RobotStatus.IDLE
    current_task: Optional[str] = None
    assigned_items: List[str] = field(default_factory=list)
    
    # Navigation state
    current_path: List[Tuple[float, float]] = field(default_factory=list)
    path_index: int = 0
    stuck_counter: int = 0
    last_position: Optional[Tuple[float, float, float]] = None
    
    # Communication state
    last_communication: float = 0.0
    pending_messages: List[Dict] = field(default_factory=list)
    
    # Performance metrics
    distance_traveled: float = 0.0
    items_collected: int = 0
    items_delivered: int = 0
    coordination_events: int = 0
    
    def update_position(self, new_position: Tuple[float, float, float]):
        """Update robot position and calculate distance traveled"""
        if self.position:
            distance = np.linalg.norm(
                np.array(new_position[:2]) - np.array(self.position[:2])
            )
            self.distance_traveled += distance
        
        self.last_position = self.position
        self.position = new_position
    
    def can_carry_item(self, item_weight: float) -> bool:
        """Check if robot can carry additional item"""
        return (self.current_load + item_weight) <= self.max_load_capacity
    
    def pick_item(self, item_id: str, item_weight: float) -> bool:
        """Pick up an item if possible"""
        if self.can_carry_item(item_weight) and not self.carrying_item:
            self.carrying_item = item_id
            self.current_load += item_weight
            self.items_collected += 1
            return True
        return False
    
    def drop_item(self) -> Optional[str]:
        """Drop currently carried item"""
        if self.carrying_item:
            dropped_item = self.carrying_item
            self.carrying_item = None
            self.current_load = 0.0  # Simplified - assume one item at a time
            self.items_delivered += 1
            return dropped_item
        return None
    
    def is_stuck(self, threshold: int = 5) -> bool:
        """Check if robot appears to be stuck"""
        return self.stuck_counter >= threshold
    
    def to_dict(self) -> Dict:
        """Convert robot state to dictionary"""
        return {
            "robot_id": self.robot_id,
            "position": self.position,
            "orientation": self.orientation,
            "target_position": self.target_position,
            "battery_level": self.battery_level,
            "carrying_item": self.carrying_item,
            "load": f"{self.current_load:.1f}/{self.max_load_capacity:.1f}",
            "status": self.status.value,
            "current_task": self.current_task,
            "assigned_items": self.assigned_items,
            "path_progress": f"{self.path_index}/{len(self.current_path)}",
            "performance": {
                "distance_traveled": self.distance_traveled,
                "items_collected": self.items_collected,
                "items_delivered": self.items_delivered,
                "coordination_events": self.coordination_events
            }
        }


class RobotFleetSimulator:
    """Simulates and manages a fleet of warehouse robots"""
    
    def __init__(self, num_robots: int = 5, communication_range: float = 10.0):
        self.num_robots = num_robots
        self.communication_range = communication_range
        
        # Fleet state
        self.robots: Dict[str, RobotState] = {}
        self.robot_positions_history: Dict[str, List[Tuple[float, float, float]]] = {}
        
        # Coordination systems
        self.path_planner = SpatialPathPlanner()
        self.communication_system = RobotCommunicationSystem(communication_range)
        
        # Simulation parameters
        self.robot_speed = 2.0  # meters per second
        self.robot_radius = 0.5  # meters
        self.collision_threshold = 1.0  # meters
        self.battery_drain_rate = 0.001  # per step
        
        # Current warehouse layout
        self.warehouse_layout = None
        
        logger.info(f"Initialized robot fleet with {num_robots} robots")
    
    async def initialize(self, warehouse_layout):
        """Initialize robot fleet in the warehouse"""
        self.warehouse_layout = warehouse_layout
        
        # Generate safe starting positions for robots
        safe_positions = self._generate_safe_positions(self.num_robots)
        
        # Create robot instances
        for i in range(self.num_robots):
            robot_id = f"robot_{i:03d}"
            
            if i < len(safe_positions):
                position = (*safe_positions[i], 0.5)  # Add z-coordinate
            else:
                # Fallback position if not enough safe positions
                position = (2.0 + i * 2.0, 2.0, 0.5)
            
            robot = RobotState(
                robot_id=robot_id,
                position=position,
                orientation=np.random.uniform(0, 2 * np.pi),
                max_load_capacity=random.uniform(8.0, 12.0)
            )
            
            self.robots[robot_id] = robot
            self.robot_positions_history[robot_id] = [position]
        
        # Initialize coordination systems
        # Register robots with communication system
        for robot_id, robot in self.robots.items():
            self.communication_system.register_robot(robot_id, robot.position[:2])
        
        logger.info(f"Initialized {len(self.robots)} robots in warehouse")
    
    def _generate_safe_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """Generate safe starting positions for robots"""
        if not self.warehouse_layout:
            # Default positions in a grid
            positions = []
            for i in range(num_positions):
                x = 2.0 + (i % 3) * 3.0
                y = 2.0 + (i // 3) * 3.0
                positions.append((x, y))
            return positions
        
        # Use layout generator's method if available
        if hasattr(self.warehouse_layout, 'get_safe_robot_positions'):
            return self.warehouse_layout.get_safe_robot_positions(num_positions)
        
        # Generate positions manually
        positions = []
        max_attempts = num_positions * 20
        attempts = 0
        
        while len(positions) < num_positions and attempts < max_attempts:
            x = np.random.uniform(2, self.warehouse_layout.dimensions[0] - 2)
            y = np.random.uniform(2, self.warehouse_layout.dimensions[1] - 2)
            
            if self.warehouse_layout.is_position_valid((x, y), self.robot_radius):
                # Check minimum distance from other robots
                min_distance = 2.0
                too_close = False
                
                for existing_pos in positions:
                    distance = np.linalg.norm(np.array([x, y]) - np.array(existing_pos))
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append((x, y))
            
            attempts += 1
        
        return positions
    
    def get_robot_states(self) -> List[RobotState]:
        """Get list of all robot states"""
        return list(self.robots.values())
    
    def get_robot(self, robot_id: str) -> Optional[RobotState]:
        """Get specific robot by ID"""
        return self.robots.get(robot_id)
    
    async def move_robot(self, robot_id: str, target_position: Tuple[float, float, float]) -> bool:
        """Move a robot to a target position (simplified for testing)"""
        robot = self.get_robot(robot_id)
        if robot:
            robot.update_position(target_position)
            robot.target_position = target_position
            self.communication_system.update_robot_position(robot_id, target_position[:2])
            return True
        return False
    
    async def get_robot_observation(self, robot_id: str) -> Dict[str, Any]:
        """Get local observation for a specific robot"""
        robot = self.robots.get(robot_id)
        if not robot:
            return {}
        
        # Get nearby robots
        nearby_robots = []
        for other_id, other_robot in self.robots.items():
            if other_id != robot_id:
                distance = np.linalg.norm(
                    np.array(robot.position[:2]) - np.array(other_robot.position[:2])
                )
                if distance <= self.communication_range:
                    nearby_robots.append({
                        "robot_id": other_id,
                        "position": other_robot.position,
                        "status": other_robot.status.value,
                        "carrying": other_robot.carrying_item,
                        "distance": distance
                    })
        
        # Get local environment information
        local_obstacles = self._get_local_obstacles(robot.position)
        local_items = self._get_local_items(robot.position)
        
        return {
            "robot_state": robot.to_dict(),
            "nearby_robots": nearby_robots,
            "local_obstacles": local_obstacles,
            "local_items": local_items,
            "communication_messages": robot.pending_messages.copy(),
            "timestamp": time.time()
        }
    
    async def execute_actions(self, robot_decisions: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute actions for all robots and return results"""
        
        execution_results = {
            "robot_actions": {},
            "collisions": [],
            "items_collected": [],
            "items_delivered": [],
            "coordination_events": [],
            "progress_made": 0.0
        }
        
        # Execute actions for each robot
        for robot_id, decision in robot_decisions.items():
            if robot_id not in self.robots:
                continue
            
            robot = self.robots[robot_id]
            action_result = await self._execute_robot_action(robot, decision)
            execution_results["robot_actions"][robot_id] = action_result
            
            # Aggregate results
            if action_result.get("item_collected"):
                execution_results["items_collected"].append(action_result["item_collected"])
            
            if action_result.get("item_delivered"):
                execution_results["items_delivered"].append(action_result["item_delivered"])
            
            if action_result.get("coordination_event"):
                execution_results["coordination_events"].append(action_result["coordination_event"])
        
        # Check for collisions after all movements
        collisions = await self._check_collisions()
        execution_results["collisions"] = collisions
        
        # Update robot states
        await self._update_robot_states()
        
        # Calculate progress
        execution_results["progress_made"] = self._calculate_step_progress(execution_results)
        
        return execution_results
    
    async def _execute_robot_action(self, robot: RobotState, decision: Dict) -> Dict[str, Any]:
        """Execute a single robot's action"""
        
        action = decision.get("action", "wait")
        parameters = decision.get("parameters", {})
        result = {
            "robot_id": robot.robot_id,
            "action": action,
            "success": False,
            "message": ""
        }
        
        try:
            if action == "move_to":
                await self._execute_move_to(robot, parameters, result)
            elif action == "pick_item":
                await self._execute_pick_item(robot, parameters, result)
            elif action == "drop_item":
                await self._execute_drop_item(robot, parameters, result)
            elif action == "communicate":
                await self._execute_communicate(robot, parameters, result)
            elif action == "wait":
                await self._execute_wait(robot, parameters, result)
            else:
                result["message"] = f"Unknown action: {action}"
                
        except Exception as e:
            logger.error(f"Error executing action {action} for robot {robot.robot_id}: {e}")
            result["message"] = f"Action failed: {str(e)}"
        
        return result
    
    async def _execute_move_to(self, robot: RobotState, parameters: Dict, result: Dict):
        """Execute move_to action"""
        target_x = parameters.get("x", robot.position[0])
        target_y = parameters.get("y", robot.position[1])
        target_position = (target_x, target_y, robot.position[2])
        
        # Plan path if not already planned or target changed
        if (robot.target_position != target_position or 
            not robot.current_path or 
            robot.path_index >= len(robot.current_path)):
            
            path = await self.path_planner.plan_path(
                robot.position[:2], 
                target_position[:2],
                robot.robot_id
            )
            
            if path:
                robot.current_path = path
                robot.path_index = 0
                robot.target_position = target_position
                robot.status = RobotStatus.MOVING
                result["success"] = True
                result["message"] = f"Started moving to {target_position[:2]}"
            else:
                result["message"] = "Could not plan path to target"
                return
        
        # Execute movement along path
        if robot.current_path and robot.path_index < len(robot.current_path):
            next_waypoint = robot.current_path[robot.path_index]
            
            # Calculate movement step
            current_pos = np.array(robot.position[:2])
            target_pos = np.array(next_waypoint)
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:  # Not yet at waypoint
                # Move towards waypoint
                movement_distance = min(self.robot_speed * 0.1, distance)  # 0.1s time step
                normalized_direction = direction / distance
                new_position = current_pos + normalized_direction * movement_distance
                
                # Check if new position is valid
                if self.warehouse_layout.is_position_valid(new_position, self.robot_radius):
                    robot.update_position((*new_position, robot.position[2]))
                    robot.stuck_counter = 0
                    result["success"] = True
                    result["message"] = f"Moving to waypoint {robot.path_index + 1}/{len(robot.current_path)}"
                else:
                    robot.stuck_counter += 1
                    result["message"] = "Path blocked, replanning needed"
            else:
                # Reached waypoint, move to next
                robot.path_index += 1
                if robot.path_index >= len(robot.current_path):
                    # Reached final destination
                    robot.status = RobotStatus.IDLE
                    robot.current_path = []
                    robot.path_index = 0
                    result["message"] = "Reached destination"
                result["success"] = True
    
    async def _execute_pick_item(self, robot: RobotState, parameters: Dict, result: Dict):
        """Execute pick_item action"""
        item_id = parameters.get("item_id")
        if not item_id:
            result["message"] = "No item_id specified"
            return
        
        # Find the item in nearby shelves
        item_shelf = self._find_item_shelf(item_id, robot.position)
        if not item_shelf:
            result["message"] = f"Item {item_id} not found nearby"
            return
        
        # Check if robot is close enough to shelf
        shelf_pos = np.array(item_shelf.position[:2])
        robot_pos = np.array(robot.position[:2])
        distance = np.linalg.norm(shelf_pos - robot_pos)
        
        if distance > 2.0:  # Too far from shelf
            result["message"] = f"Too far from shelf (distance: {distance:.1f}m)"
            return
        
        # Try to pick the item
        item_weight = parameters.get("weight", 1.0)
        if robot.pick_item(item_id, item_weight):
            # Remove item from shelf
            item_shelf.remove_item(item_id)
            robot.status = RobotStatus.IDLE
            result["success"] = True
            result["message"] = f"Picked up {item_id}"
            result["item_collected"] = item_id
        else:
            result["message"] = "Cannot carry item (capacity exceeded or already carrying)"
    
    async def _execute_drop_item(self, robot: RobotState, parameters: Dict, result: Dict):
        """Execute drop_item action"""
        zone_id = parameters.get("zone_id")
        if not zone_id:
            result["message"] = "No zone_id specified"
            return
        
        if not robot.carrying_item:
            result["message"] = "Not carrying any item"
            return
        
        # Find the delivery zone
        delivery_zone = self._find_delivery_zone(zone_id, robot.position)
        if not delivery_zone:
            result["message"] = f"Delivery zone {zone_id} not found nearby"
            return
        
        # Check if robot is in the delivery zone
        if not delivery_zone.contains_point(robot.position[:2]):
            result["message"] = "Not in delivery zone"
            return
        
        # Drop the item
        dropped_item = robot.drop_item()
        if dropped_item:
            robot.status = RobotStatus.IDLE
            result["success"] = True
            result["message"] = f"Delivered {dropped_item} to {zone_id}"
            result["item_delivered"] = dropped_item
        else:
            result["message"] = "Failed to drop item"
    
    async def _execute_communicate(self, robot: RobotState, parameters: Dict, result: Dict):
        """Execute communicate action"""
        target_robot_id = parameters.get("robot_id")
        message = parameters.get("message", "")
        
        if not target_robot_id or target_robot_id not in self.robots:
            result["message"] = "Invalid target robot"
            return
        
        # Send message through communication system
        success = await self.communication_system.send_message(
            robot.robot_id, target_robot_id, message
        )
        
        if success:
            robot.coordination_events += 1
            result["success"] = True
            result["message"] = f"Sent message to {target_robot_id}"
            result["coordination_event"] = {
                "type": "communication",
                "from": robot.robot_id,
                "to": target_robot_id,
                "message": message
            }
        else:
            result["message"] = "Communication failed (out of range?)"
    
    async def _execute_wait(self, robot: RobotState, parameters: Dict, result: Dict):
        """Execute wait action"""
        robot.status = RobotStatus.WAITING
        result["success"] = True
        result["message"] = "Waiting"
    
    async def _check_collisions(self) -> List[Dict]:
        """Check for collisions between robots"""
        collisions = []
        
        robot_list = list(self.robots.values())
        for i in range(len(robot_list)):
            for j in range(i + 1, len(robot_list)):
                robot1 = robot_list[i]
                robot2 = robot_list[j]
                
                distance = np.linalg.norm(
                    np.array(robot1.position[:2]) - np.array(robot2.position[:2])
                )
                
                if distance < self.collision_threshold:
                    collision = {
                        "robot1": robot1.robot_id,
                        "robot2": robot2.robot_id,
                        "distance": distance,
                        "position1": robot1.position,
                        "position2": robot2.position
                    }
                    collisions.append(collision)
                    
                    # Update robot states
                    robot1.status = RobotStatus.ERROR
                    robot2.status = RobotStatus.ERROR
        
        return collisions
    
    async def _update_robot_states(self):
        """Update robot states after action execution"""
        for robot in self.robots.values():
            # Update battery
            robot.battery_level = max(0.0, robot.battery_level - self.battery_drain_rate)
            
            # Update position history
            self.robot_positions_history[robot.robot_id].append(robot.position)
            
            # Keep only recent history
            if len(self.robot_positions_history[robot.robot_id]) > 100:
                self.robot_positions_history[robot.robot_id] = \
                    self.robot_positions_history[robot.robot_id][-100:]
            
            # Process pending messages
            robot.pending_messages = self.communication_system.receive_messages(robot.robot_id)
            
            # Reset error status if robots are no longer colliding
            if robot.status == RobotStatus.ERROR:
                # Check if still colliding
                still_colliding = False
                for other_robot in self.robots.values():
                    if other_robot.robot_id != robot.robot_id:
                        distance = np.linalg.norm(
                            np.array(robot.position[:2]) - np.array(other_robot.position[:2])
                        )
                        if distance < self.collision_threshold:
                            still_colliding = True
                            break
                
                if not still_colliding:
                    robot.status = RobotStatus.IDLE
    
    def _calculate_step_progress(self, execution_results: Dict) -> float:
        """Calculate progress made in this step"""
        progress = 0.0
        
        # Progress from items collected/delivered
        progress += len(execution_results["items_collected"]) * 0.5
        progress += len(execution_results["items_delivered"]) * 1.0
        
        # Progress from successful coordination
        progress += len(execution_results["coordination_events"]) * 0.2
        
        # Penalty for collisions
        progress -= len(execution_results["collisions"]) * 0.5
        
        return max(0.0, progress)
    
    def _get_local_obstacles(self, position: Tuple[float, float, float]) -> List[Dict]:
        """Get obstacles near robot position"""
        if not self.warehouse_layout:
            return []
        
        local_obstacles = []
        search_radius = 5.0
        
        # Check shelves
        for shelf in self.warehouse_layout.shelves:
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(shelf.position[:2])
            )
            if distance <= search_radius:
                local_obstacles.append({
                    "type": "shelf",
                    "id": shelf.shelf_id,
                    "position": shelf.position,
                    "dimensions": shelf.dimensions,
                    "distance": distance
                })
        
        # Check static obstacles
        for obstacle in self.warehouse_layout.obstacles:
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(obstacle["position"][:2])
            )
            if distance <= search_radius:
                local_obstacles.append({
                    "type": obstacle["type"],
                    "id": obstacle["obstacle_id"],
                    "position": obstacle["position"],
                    "radius": obstacle["radius"],
                    "distance": distance
                })
        
        return local_obstacles
    
    def _get_local_items(self, position: Tuple[float, float, float]) -> List[Dict]:
        """Get items available near robot position"""
        if not self.warehouse_layout:
            return []
        
        local_items = []
        search_radius = 3.0
        
        for shelf in self.warehouse_layout.shelves:
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(shelf.position[:2])
            )
            if distance <= search_radius and shelf.items:
                for item_id in shelf.items:
                    local_items.append({
                        "item_id": item_id,
                        "shelf_id": shelf.shelf_id,
                        "shelf_position": shelf.position,
                        "distance": distance
                    })
        
        return local_items
    
    def _find_item_shelf(self, item_id: str, robot_position: Tuple[float, float, float]):
        """Find shelf containing specified item"""
        if not self.warehouse_layout:
            return None
        
        for shelf in self.warehouse_layout.shelves:
            if item_id in shelf.items:
                return shelf
        
        return None
    
    def _find_delivery_zone(self, zone_id: str, robot_position: Tuple[float, float, float]):
        """Find delivery zone by ID"""
        if not self.warehouse_layout:
            return None
        
        for zone in self.warehouse_layout.zones:
            if zone.zone_id == zone_id:
                return zone
        
        return None
    
    def get_fleet_statistics(self) -> Dict[str, Any]:
        """Get overall fleet performance statistics"""
        stats = {
            "total_robots": len(self.robots),
            "active_robots": len([r for r in self.robots.values() if r.status != RobotStatus.ERROR]),
            "total_distance": sum(r.distance_traveled for r in self.robots.values()),
            "total_items_collected": sum(r.items_collected for r in self.robots.values()),
            "total_items_delivered": sum(r.items_delivered for r in self.robots.values()),
            "total_coordination_events": sum(r.coordination_events for r in self.robots.values()),
            "average_battery": np.mean([r.battery_level for r in self.robots.values()]),
            "robots_carrying_items": len([r for r in self.robots.values() if r.carrying_item])
        }
        
        return stats 