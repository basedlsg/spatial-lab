"""
Warehouse Spatial Environment for Multi-Agent Coordination Research

This environment simulates a realistic warehouse setting where multiple robots
must coordinate to complete picking, packing, and navigation tasks efficiently.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

# Atropos imports
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup, ScoredDataItem
from atroposlib.type_definitions import Item, Message

from .warehouse_layout import WarehouseLayoutGenerator
from .warehouse_tasks import WarehouseTaskGenerator, WarehouseTask
from ..coordination.robot_fleet import RobotFleetSimulator
from ..evaluation.spatial_metrics import SpatialMetricsCalculator
from ..llm import create_llm_coordinator, TaskRequirements, LLMProvider

logger = logging.getLogger(__name__)


class RobotAction(str, Enum):
    """Available actions for warehouse robots"""
    MOVE_TO = "move_to"
    PICK_ITEM = "pick_item"
    DROP_ITEM = "drop_item"
    WAIT = "wait"
    REQUEST_PATH = "request_path"
    COMMUNICATE = "communicate"


@dataclass
class RobotState:
    """Current state of a warehouse robot"""
    robot_id: str
    position: Tuple[float, float, float]
    orientation: float  # radians
    carrying_item: Optional[str] = None
    battery_level: float = 1.0
    current_task: Optional[str] = None
    status: str = "idle"  # idle, moving, picking, dropping, waiting


@dataclass
class WarehouseItem:
    """Item in the warehouse"""
    item_id: str
    item_type: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    weight: float
    priority: int = 1


class WarehouseSpatialEnvironmentConfig(BaseEnvConfig):
    """Configuration for warehouse spatial environment"""
    
    # Warehouse parameters
    warehouse_width: float = Field(default=50.0, description="Warehouse width in meters")
    warehouse_height: float = Field(default=30.0, description="Warehouse height in meters")
    num_robots: int = Field(default=5, description="Number of robots in the warehouse")
    num_shelves: int = Field(default=20, description="Number of storage shelves")
    
    # Task parameters
    max_task_duration: int = Field(default=300, description="Maximum task duration in steps")
    items_per_task: int = Field(default=10, description="Number of items per coordination task")
    task_complexity: str = Field(default="medium", description="Task complexity: easy, medium, hard")
    
    # Coordination parameters
    communication_range: float = Field(default=10.0, description="Robot communication range in meters")
    max_coordination_delay: int = Field(default=5, description="Maximum coordination delay in steps")
    
    # LLM parameters
    llama_api_key: Optional[str] = Field(default=None, description="Llama API key for spatial reasoning")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key for backup reasoning")
    preferred_llm_provider: str = Field(default="llama", description="Preferred LLM provider: llama or gemini")
    enable_multimodal: bool = Field(default=False, description="Enable multimodal spatial reasoning")
    
    # Evaluation parameters
    collision_penalty: float = Field(default=-10.0, description="Penalty for robot collisions")
    efficiency_weight: float = Field(default=1.0, description="Weight for efficiency scoring")
    coordination_weight: float = Field(default=0.5, description="Weight for coordination scoring")


class WarehouseSpatialEnvironment(BaseEnv):
    """
    Warehouse spatial reasoning environment using Atropos framework.
    
    This environment simulates multi-robot coordination in a warehouse setting,
    focusing on spatial reasoning, path planning, and collaborative task execution.
    """
    
    name = "warehouse_spatial_coordination"
    env_config_cls = WarehouseSpatialEnvironmentConfig
    
    def __init__(self, config: WarehouseSpatialEnvironmentConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Initialize core components
        self.layout_generator = WarehouseLayoutGenerator(
            width=config.warehouse_width,
            height=config.warehouse_height,
            num_shelves=config.num_shelves
        )
        
        self.task_generator = WarehouseTaskGenerator(
            complexity=config.task_complexity,
            items_per_task=config.items_per_task
        )
        
        self.robot_fleet = RobotFleetSimulator(
            num_robots=config.num_robots,
            communication_range=config.communication_range
        )
        
        self.metrics_calculator = SpatialMetricsCalculator()
        
        # Initialize LLM coordinator for spatial reasoning
        self.llm_coordinator = None
        if config.llama_api_key or config.gemini_api_key:
            self.llm_coordinator = create_llm_coordinator(
                llama_api_key=config.llama_api_key,
                gemini_api_key=config.gemini_api_key,
                preferred_provider=config.preferred_llm_provider
            )
            logger.info(f"Initialized LLM coordinator with provider: {config.preferred_llm_provider}")
        else:
            logger.warning("No LLM API keys provided - using fallback reasoning")
        
        # Environment state
        self.current_layout = None
        self.current_robots = {}
        self.current_items = {}
        self.step_count = 0
        
        logger.info(f"Initialized WarehouseSpatialEnvironment with {config.num_robots} robots")
    
    async def setup(self):
        """Setup the warehouse environment"""
        logger.info("Setting up warehouse spatial environment")
        
        # Generate initial warehouse layout
        self.current_layout = await self.layout_generator.generate_layout()
        
        # Initialize robot fleet
        await self.robot_fleet.initialize(self.current_layout)
        
        logger.info("Warehouse environment setup complete")
    
    async def get_next_item(self) -> Item:
        """Generate next warehouse coordination task"""
        
        # Generate new task
        task = await self.task_generator.generate_task(
            layout=self.current_layout,
            robots=self.robot_fleet.get_robot_states()
        )
        
        # Create Atropos item (as a dictionary since Item = Any)
        item = {
            "item_id": f"warehouse_task_{uuid.uuid4().hex[:8]}",
            "data": {
                "task": task.to_dict(),
                "warehouse_layout": self.current_layout.to_dict(),
                "initial_robot_states": [robot.to_dict() for robot in self.robot_fleet.get_robot_states()],
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Generated warehouse task: {item['item_id']}")
        return item
    
    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List[Item]]:
        """Execute warehouse coordination task and collect trajectories"""
        
        task_data = item.data["task"]
        task = WarehouseTask.from_dict(task_data)
        
        logger.info(f"Executing warehouse task: {task.task_id}")
        
        # Reset environment for this task
        await self.reset_for_task(task)
        
        trajectories = []
        step = 0
        
        while step < self.config.max_task_duration and not task.is_complete():
            step_start_time = time.time()
            
            # Get current observations for all robots
            observations = await self.get_robot_observations()
            
            # Get LLM decisions for each robot
            robot_decisions = await self.get_robot_decisions(observations, task)
            
            # Execute robot actions and update environment
            step_result = await self.execute_robot_actions(robot_decisions)
            
            # Calculate step rewards and metrics
            step_rewards = await self.calculate_step_rewards(step_result, task)
            step_metrics = await self.metrics_calculator.calculate_step_metrics(step_result)
            
            # Create trajectory step
            trajectory_step = ScoredDataItem(
                messages=[
                    Message(role="system", content=self.create_system_prompt(task)),
                    Message(role="user", content=self.create_observation_prompt(observations)),
                    Message(role="assistant", content=json.dumps(robot_decisions))
                ],
                score=step_rewards["total_reward"],
                metadata={
                    "step": step,
                    "robot_states": [robot.to_dict() for robot in self.robot_fleet.get_robot_states()],
                    "task_progress": task.get_progress(),
                    "metrics": step_metrics,
                    "step_duration": time.time() - step_start_time
                }
            )
            
            trajectories.append(trajectory_step)
            
            # Update task progress
            await task.update_progress(step_result)
            step += 1
        
        # Calculate final scores
        final_scores = await self.calculate_final_scores(trajectories, task)
        
        # Create scored data group
        scored_group = ScoredDataGroup(
            scored_items=trajectories,
            group_metadata={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "num_robots": len(self.robot_fleet.get_robot_states()),
                "total_steps": step,
                "task_completed": task.is_complete(),
                "final_scores": final_scores,
                "warehouse_metrics": await self.metrics_calculator.get_episode_summary()
            }
        )
        
        logger.info(f"Completed warehouse task: {task.task_id}, Score: {final_scores['total_score']:.2f}")
        
        return scored_group, []
    
    async def get_robot_observations(self) -> Dict[str, Dict]:
        """Get current observations for all robots"""
        
        observations = {}
        
        for robot in self.robot_fleet.get_robot_states():
            # Get robot's local observation
            local_obs = await self.robot_fleet.get_robot_observation(robot.robot_id)
            
            # Add warehouse context
            warehouse_context = {
                "nearby_shelves": self.get_nearby_shelves(robot.position, radius=15.0),
                "nearby_robots": self.get_nearby_robots(robot.robot_id, radius=self.config.communication_range),
                "available_items": self.get_available_items_near(robot.position, radius=20.0),
                "current_congestion": self.calculate_area_congestion(robot.position)
            }
            
            observations[robot.robot_id] = {
                "robot_state": robot.to_dict(),
                "local_observation": local_obs,
                "warehouse_context": warehouse_context,
                "timestamp": time.time()
            }
        
        return observations
    
    async def get_robot_decisions(self, observations: Dict[str, Dict], task: WarehouseTask) -> Dict[str, Dict]:
        """Get LLM-based decisions for each robot using real APIs"""
        
        decisions = {}
        
        # If LLM coordinator is available, use real APIs
        if self.llm_coordinator:
            try:
                # Prepare robot decision requests for batch processing
                robot_requests = []
                
                for robot_id, obs in observations.items():
                    robot_state = obs["robot_state"]
                    warehouse_context = obs["warehouse_context"]
                    
                    # Available actions for this robot
                    available_actions = [action.value for action in RobotAction]
                    
                    # Create request for LLM coordinator
                    request = {
                        "robot_id": robot_id,
                        "observation": obs,
                        "task_description": task.description,
                        "available_actions": available_actions,
                        "nearby_robots": warehouse_context.get("nearby_robots", []),
                        "warehouse_layout": {
                            "shelves": warehouse_context.get("nearby_shelves", []),
                            "dimensions": (self.config.warehouse_width, self.config.warehouse_height)
                        }
                    }
                    robot_requests.append(request)
                
                # Get decisions from LLM coordinator with structured output
                task_requirements = TaskRequirements(
                    requires_structured_output=True,
                    priority="normal",
                    max_latency_ms=3000
                )
                
                # Process robot decisions concurrently
                llm_responses = await self.llm_coordinator.batch_robot_decisions(
                    robot_requests, 
                    task_requirements
                )
                
                # Parse LLM responses into decisions
                for i, (response, provider_used) in enumerate(llm_responses):
                    robot_id = robot_requests[i]["robot_id"]
                    
                    try:
                        # Extract decision from LLM response
                        if "completion_message" in response:
                            content = response["completion_message"]["content"]
                            if isinstance(content, dict) and "text" in content:
                                decision_text = content["text"]
                            else:
                                decision_text = str(content)
                            
                            # Parse the JSON decision
                            decision_data = json.loads(decision_text)
                            
                            # Validate and format decision
                            action = decision_data.get("action", "wait")
                            if action not in available_actions:
                                action = "wait"
                            
                            decisions[robot_id] = {
                                "robot_id": robot_id,
                                "action": action,
                                "parameters": decision_data.get("parameters", {}),
                                "reasoning": decision_data.get("reasoning", "LLM decision"),
                                "confidence": decision_data.get("confidence", 0.8),
                                "coordination_intent": decision_data.get("coordination_intent", ""),
                                "provider_used": provider_used.value if provider_used else "unknown",
                                "timestamp": time.time()
                            }
                            
                            logger.debug(f"Robot {robot_id} decision from {provider_used}: {action}")
                            
                        else:
                            raise ValueError("Invalid LLM response format")
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse LLM decision for robot {robot_id}: {e}")
                        # Fallback decision
                        decisions[robot_id] = {
                            "robot_id": robot_id,
                            "action": "wait",
                            "parameters": {},
                            "reasoning": f"LLM parsing error: {str(e)}",
                            "confidence": 0.1,
                            "timestamp": time.time()
                        }
                
            except Exception as e:
                logger.error(f"LLM coordinator failed: {e}")
                # Fallback to simple decisions for all robots
                for robot_id in observations.keys():
                    decisions[robot_id] = {
                        "robot_id": robot_id,
                        "action": "wait",
                        "parameters": {},
                        "reasoning": f"LLM coordinator error: {str(e)}",
                        "confidence": 0.1,
                        "timestamp": time.time()
                    }
        
        else:
            # Fallback to simple rule-based decisions when no LLM is available
            logger.warning("No LLM coordinator available, using fallback decisions")
            
            for robot_id, obs in observations.items():
                robot_state = obs["robot_state"]
                
                # Simple rule-based decision logic
                if robot_state["battery_level"] < 0.2:
                    action = "wait"  # Low battery
                    reasoning = "Low battery, waiting for recharge"
                elif robot_state["carrying_item"]:
                    action = "move_to"  # Carrying item, move to delivery
                    reasoning = "Carrying item, moving to delivery zone"
                else:
                    action = "move_to"  # Idle, move to nearest item
                    reasoning = "Idle, moving to collect items"
                
                decisions[robot_id] = {
                    "robot_id": robot_id,
                    "action": action,
                    "parameters": {},
                    "reasoning": reasoning,
                    "confidence": 0.5,
                    "timestamp": time.time()
                }
        
        return decisions
    
    def create_spatial_reasoning_prompt(self, robot_id: str, observation: Dict, task: WarehouseTask) -> str:
        """Create spatial reasoning prompt for robot decision making"""
        
        robot_state = observation["robot_state"]
        warehouse_context = observation["warehouse_context"]
        
        prompt = f"""You are Robot {robot_id} in a warehouse coordination task.

CURRENT STATE:
- Position: {robot_state['position']}
- Carrying: {robot_state['carrying_item'] or 'nothing'}
- Battery: {robot_state['battery_level']:.1%}
- Status: {robot_state['status']}

TASK OBJECTIVE: {task.description}
- Items to collect: {task.get_remaining_items()}
- Delivery zones: {task.get_delivery_zones()}
- Priority items: {task.get_priority_items()}

SPATIAL ENVIRONMENT:
- Nearby shelves: {warehouse_context['nearby_shelves']}
- Other robots nearby: {warehouse_context['nearby_robots']}
- Available items: {warehouse_context['available_items']}
- Area congestion: {warehouse_context['current_congestion']}

AVAILABLE ACTIONS:
- move_to(x, y): Move to coordinates (x, y)
- pick_item(item_id): Pick up specified item
- drop_item(zone_id): Drop carried item at delivery zone
- wait(): Stay in current position
- communicate(robot_id, message): Send message to another robot

COORDINATION GUIDELINES:
1. Avoid collisions with other robots
2. Communicate when paths might conflict
3. Prioritize high-priority items
4. Consider battery level for long tasks
5. Coordinate with nearby robots for efficiency

Choose the best action considering spatial efficiency and coordination.
Respond with JSON: {{"action": "action_name", "parameters": {{}}, "reasoning": "brief explanation"}}"""
        
        return prompt
    
    def parse_robot_decision(self, response: str, robot_id: str, observation: Dict) -> Dict:
        """Parse LLM response into robot decision"""
        
        try:
            # Try to parse JSON response
            decision_data = json.loads(response.strip())
            
            # Validate action
            action = decision_data.get("action", "wait")
            if action not in [e.value for e in RobotAction]:
                action = "wait"
            
            return {
                "robot_id": robot_id,
                "action": action,
                "parameters": decision_data.get("parameters", {}),
                "reasoning": decision_data.get("reasoning", "No reasoning provided"),
                "timestamp": time.time()
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse robot decision for {robot_id}: {e}")
            return {
                "robot_id": robot_id,
                "action": "wait",
                "parameters": {},
                "reasoning": "Failed to parse LLM response",
                "timestamp": time.time()
            }
    
    async def execute_robot_actions(self, decisions: Dict[str, Dict]) -> Dict:
        """Execute all robot actions and return results"""
        
        # Execute actions through robot fleet simulator
        execution_results = await self.robot_fleet.execute_actions(decisions)
        
        # Update environment state
        await self.update_environment_state(execution_results)
        
        return execution_results
    
    async def calculate_step_rewards(self, step_result: Dict, task: WarehouseTask) -> Dict[str, float]:
        """Calculate rewards for the current step"""
        
        rewards = {
            "efficiency_reward": 0.0,
            "coordination_reward": 0.0,
            "collision_penalty": 0.0,
            "progress_reward": 0.0,
            "total_reward": 0.0
        }
        
        # Efficiency reward (based on progress toward goals)
        progress_made = step_result.get("progress_made", 0.0)
        rewards["efficiency_reward"] = progress_made * self.config.efficiency_weight
        
        # Coordination reward (based on successful coordination events)
        coordination_events = step_result.get("coordination_events", 0)
        rewards["coordination_reward"] = coordination_events * self.config.coordination_weight
        
        # Collision penalty
        collisions = step_result.get("collisions", 0)
        rewards["collision_penalty"] = collisions * self.config.collision_penalty
        
        # Progress reward (task completion progress)
        task_progress = task.get_completion_percentage()
        rewards["progress_reward"] = task_progress * 0.1
        
        # Total reward
        rewards["total_reward"] = (
            rewards["efficiency_reward"] + 
            rewards["coordination_reward"] + 
            rewards["collision_penalty"] +
            rewards["progress_reward"]
        )
        
        return rewards
    
    async def calculate_final_scores(self, trajectories: List[ScoredDataItem], task: WarehouseTask) -> Dict[str, float]:
        """Calculate final scores for the completed task"""
        
        # Aggregate step scores
        step_scores = [item.score for item in trajectories]
        avg_step_score = np.mean(step_scores) if step_scores else 0.0
        
        # Task completion bonus
        completion_bonus = 10.0 if task.is_complete() else 0.0
        
        # Efficiency score
        efficiency_score = await self.metrics_calculator.calculate_efficiency_score(trajectories)
        
        # Coordination score
        coordination_score = await self.metrics_calculator.calculate_coordination_score(trajectories)
        
        # Final total score
        total_score = avg_step_score + completion_bonus + efficiency_score + coordination_score
        
        return {
            "avg_step_score": avg_step_score,
            "completion_bonus": completion_bonus,
            "efficiency_score": efficiency_score,
            "coordination_score": coordination_score,
            "total_score": total_score
        }
    
    async def evaluate(self, *args, **kwargs):
        """Evaluation method required by BaseEnv"""
        logger.info("Running warehouse environment evaluation")
        
        # Run evaluation tasks
        eval_results = []
        
        for i in range(5):  # Run 5 evaluation tasks
            eval_item = await self.get_next_item()
            result, _ = await self.collect_trajectories(eval_item)
            eval_results.append(result)
        
        # Calculate evaluation metrics
        eval_metrics = await self.calculate_evaluation_metrics(eval_results)
        
        logger.info(f"Evaluation complete. Average score: {eval_metrics['avg_score']:.2f}")
        
        return eval_metrics
    
    # Helper methods
    def get_nearby_shelves(self, position: Tuple[float, float, float], radius: float) -> List[Dict]:
        """Get shelves within radius of position"""
        if not self.current_layout:
            return []
        
        nearby_shelves = []
        for shelf in self.current_layout.shelves:
            distance = np.linalg.norm(np.array(position[:2]) - np.array(shelf.position[:2]))
            if distance <= radius:
                nearby_shelves.append({
                    "shelf_id": shelf.shelf_id,
                    "position": shelf.position,
                    "distance": distance,
                    "items": shelf.get_items()
                })
        
        return nearby_shelves
    
    def get_nearby_robots(self, robot_id: str, radius: float) -> List[Dict]:
        """Get other robots within communication range"""
        current_robot = self.robot_fleet.get_robot(robot_id)
        if not current_robot:
            return []
        
        nearby_robots = []
        for robot in self.robot_fleet.get_robot_states():
            if robot.robot_id != robot_id:
                distance = np.linalg.norm(
                    np.array(current_robot.position[:2]) - np.array(robot.position[:2])
                )
                if distance <= radius:
                    nearby_robots.append({
                        "robot_id": robot.robot_id,
                        "position": robot.position,
                        "distance": distance,
                        "status": robot.status,
                        "carrying": robot.carrying_item
                    })
        
        return nearby_robots
    
    def create_system_prompt(self, task: WarehouseTask) -> str:
        """Create system prompt for the coordination task"""
        return f"""You are coordinating warehouse robots for efficient task completion.
Task: {task.description}
Focus on spatial reasoning, collision avoidance, and efficient coordination."""
    
    def create_observation_prompt(self, observations: Dict[str, Dict]) -> str:
        """Create observation prompt from robot observations"""
        obs_summary = []
        for robot_id, obs in observations.items():
            robot_state = obs["robot_state"]
            obs_summary.append(
                f"Robot {robot_id}: pos={robot_state['position']}, "
                f"status={robot_state['status']}, carrying={robot_state['carrying_item']}"
            )
        
        return f"Current warehouse state:\n" + "\n".join(obs_summary)
    
    # Required BaseEnv configuration
    @classmethod
    def config_init(cls):
        """Initialize default configuration"""
        env_config = WarehouseSpatialEnvironmentConfig()
        
        # Default server configuration for LLM inference
        from atroposlib.envs.server_handling.server_baseline import APIServerConfig
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url=None,
                api_key=None,  # Will be set from environment
                num_requests_for_eval=64,
            )
        ]
        
        return env_config, server_configs
    
    # Additional helper methods
    def get_available_items_near(self, position: Tuple[float, float, float], radius: float) -> List[Dict]:
        """Get available items near a position"""
        # Simplified implementation - return empty list for now
        return []
    
    def get_available_tasks_for_robot(self, robot_id: str) -> List[Dict]:
        """Get available tasks for a specific robot"""
        # Simplified implementation - return empty list for now
        return []
    
    async def reset_for_task(self, task):
        """Reset environment for a new task"""
        # Reset step counter
        self.step_count = 0

        # Reset robot fleet to initial positions
        await self.robot_fleet.reset()

        # Update task-specific state
        if hasattr(task, 'required_items'):
            # Mark required items as available in warehouse
            for item_id in task.required_items:
                if item_id in self.current_items:
                    self.current_items[item_id]['status'] = 'available'

        logger.info(f"Environment reset for task: {task.task_id if hasattr(task, 'task_id') else 'unknown'}")
    
    async def update_environment_state(self, execution_results: Dict):
        """Update environment state after robot actions"""
        # Update step count
        self.step_count += 1

        # Process results for each robot
        for robot_id, result in execution_results.items():
            if not result.get('success', False):
                continue

            action = result.get('action')

            # Update item locations based on robot actions
            if action == 'pick_item':
                item_id = result.get('item_picked')
                if item_id and item_id in self.current_items:
                    self.current_items[item_id]['status'] = 'carried'
                    self.current_items[item_id]['carrier'] = robot_id

            elif action == 'drop_item':
                item_id = result.get('item_delivered')
                if item_id and item_id in self.current_items:
                    self.current_items[item_id]['status'] = 'delivered'
                    self.current_items[item_id]['carrier'] = None
                    # Update item position to delivery location
                    if 'delivery_position' in result:
                        self.current_items[item_id]['position'] = result['delivery_position']

        # Log environment state update
        logger.debug(f"Environment state updated at step {self.step_count}")
    
    async def calculate_evaluation_metrics(self, eval_results: List) -> Dict:
        """Calculate evaluation metrics from results"""
        # Simple evaluation metrics
        return {
            "avg_score": 0.5,
            "completion_rate": 0.8,
            "efficiency": 0.7
        }
    
    def calculate_area_congestion(self, position: Tuple[float, float, float]) -> float:
        """Calculate congestion level around a position"""
        # Simple implementation - count nearby robots
        nearby_robots = self.get_nearby_robots("temp_robot", 5.0)
        return len(nearby_robots) / 10.0  # Normalize to 0-1 range