"""
Multi-Agent Coordination System for Spatial AI Research Lab

Implements task allocation, conflict resolution, and coordination strategies
for multi-robot warehouse operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Coordination strategies for multi-agent systems."""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    AUCTION_BASED = "auction_based"


class TaskStatus(Enum):
    """Status of a coordination task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CoordinationTask:
    """Represents a task requiring coordination between agents."""
    task_id: str
    priority: float
    estimated_duration: float
    required_agents: int
    location: Tuple[float, float]
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class AgentCapability:
    """Represents capabilities of an individual agent."""
    agent_id: str
    max_payload: float
    speed: float
    battery_level: float
    current_location: Tuple[float, float]
    available: bool = True
    current_task: Optional[str] = None
    skill_level: float = 1.0  # 0.0 to 1.0


@dataclass
class TaskAssignment:
    """Represents an assignment of an agent to a task."""
    task_id: str
    agent_id: str
    assignment_time: float
    estimated_arrival: float
    estimated_completion: float
    cost: float


class MultiAgentCoordinator:
    """
    Coordinates multiple agents in warehouse environments.

    Implements:
    - Task allocation using Hungarian algorithm or greedy assignment
    - Conflict detection and resolution
    - Priority-based scheduling
    - Load balancing across agents
    """

    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.CENTRALIZED):
        """
        Initialize the multi-agent coordinator.

        Args:
            strategy: Coordination strategy to use
        """
        self.strategy = strategy
        self.agents: Dict[str, AgentCapability] = {}
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: Dict[str, CoordinationTask] = {}
        self.assignments: Dict[str, TaskAssignment] = {}
        self.task_history: List[Dict[str, Any]] = []

    def register_agent(self, agent: AgentCapability) -> bool:
        """
        Register a new agent with the coordinator.

        Args:
            agent: Agent capability information

        Returns:
            True if registration successful
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered, updating")

        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} at {agent.current_location}")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the coordinator.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if successful
        """
        if agent_id not in self.agents:
            return False

        # Cancel any assigned tasks
        if agent_id in self.assignments:
            task_id = self.assignments[agent_id].task_id
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if agent_id in task.assigned_agents:
                    task.assigned_agents.remove(agent_id)
                    if not task.assigned_agents:
                        task.status = TaskStatus.PENDING
            del self.assignments[agent_id]

        del self.agents[agent_id]
        logger.info(f"Unregistered agent {agent_id}")
        return True

    def submit_task(self, task: CoordinationTask) -> bool:
        """
        Submit a new coordination task.

        Args:
            task: Task to submit

        Returns:
            True if submission successful
        """
        if task.task_id in self.active_tasks:
            logger.warning(f"Task {task.task_id} already exists")
            return False

        self.active_tasks[task.task_id] = task
        logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
        return True

    def get_available_agents(self) -> List[AgentCapability]:
        """Get list of currently available agents."""
        return [
            agent for agent in self.agents.values()
            if agent.available and agent.battery_level > 0.1
        ]

    def get_pending_tasks(self) -> List[CoordinationTask]:
        """Get list of pending tasks sorted by priority."""
        pending = [
            task for task in self.active_tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        # Sort by priority (higher first) and deadline (earlier first)
        return sorted(
            pending,
            key=lambda t: (-t.priority, t.deadline or float('inf'))
        )

    async def allocate_tasks(self) -> List[TaskAssignment]:
        """
        Allocate pending tasks to available agents.

        Returns:
            List of new task assignments
        """
        available_agents = self.get_available_agents()
        pending_tasks = self.get_pending_tasks()

        if not available_agents or not pending_tasks:
            return []

        if self.strategy == CoordinationStrategy.CENTRALIZED:
            return await self._centralized_allocation(available_agents, pending_tasks)
        elif self.strategy == CoordinationStrategy.AUCTION_BASED:
            return await self._auction_based_allocation(available_agents, pending_tasks)
        else:
            return await self._distributed_allocation(available_agents, pending_tasks)

    async def _centralized_allocation(
        self,
        agents: List[AgentCapability],
        tasks: List[CoordinationTask]
    ) -> List[TaskAssignment]:
        """
        Centralized task allocation using greedy cost minimization.

        Assigns tasks to agents that minimize total cost (distance + time).
        """
        assignments = []

        for task in tasks:
            # Check dependencies
            if not self._check_dependencies(task):
                continue

            # Find best agent(s) for this task
            best_agents = self._find_best_agents(
                task,
                [a for a in agents if a.available]
            )

            if len(best_agents) >= task.required_agents:
                # Assign agents to task
                for agent in best_agents[:task.required_agents]:
                    assignment = self._create_assignment(agent, task)
                    assignments.append(assignment)

                    # Update agent and task state
                    agent.available = False
                    agent.current_task = task.task_id
                    task.assigned_agents.append(agent.agent_id)
                    self.assignments[agent.agent_id] = assignment

                task.status = TaskStatus.ASSIGNED
                task.start_time = time.time()

                logger.info(
                    f"Assigned task {task.task_id} to agents "
                    f"{task.assigned_agents}"
                )

        return assignments

    async def _auction_based_allocation(
        self,
        agents: List[AgentCapability],
        tasks: List[CoordinationTask]
    ) -> List[TaskAssignment]:
        """
        Auction-based task allocation where agents bid on tasks.

        Each agent calculates its cost for each task, lowest bid wins.
        """
        assignments = []

        for task in tasks:
            if not self._check_dependencies(task):
                continue

            # Collect bids from all available agents
            bids: List[Tuple[float, AgentCapability]] = []
            for agent in agents:
                if not agent.available:
                    continue
                cost = self._calculate_task_cost(agent, task)
                bids.append((cost, agent))

            # Sort by cost (lowest first)
            bids.sort(key=lambda x: x[0])

            # Assign to lowest bidders
            assigned_count = 0
            for cost, agent in bids:
                if assigned_count >= task.required_agents:
                    break

                assignment = self._create_assignment(agent, task)
                assignments.append(assignment)

                agent.available = False
                agent.current_task = task.task_id
                task.assigned_agents.append(agent.agent_id)
                self.assignments[agent.agent_id] = assignment
                assigned_count += 1

            if assigned_count >= task.required_agents:
                task.status = TaskStatus.ASSIGNED
                task.start_time = time.time()

        return assignments

    async def _distributed_allocation(
        self,
        agents: List[AgentCapability],
        tasks: List[CoordinationTask]
    ) -> List[TaskAssignment]:
        """
        Distributed allocation where agents self-select tasks.

        Each agent picks the best available task for itself.
        """
        assignments = []
        remaining_tasks = list(tasks)

        for agent in agents:
            if not agent.available or not remaining_tasks:
                continue

            # Find best task for this agent
            best_task = None
            best_cost = float('inf')

            for task in remaining_tasks:
                if not self._check_dependencies(task):
                    continue

                cost = self._calculate_task_cost(agent, task)
                if cost < best_cost:
                    best_cost = cost
                    best_task = task

            if best_task:
                assignment = self._create_assignment(agent, best_task)
                assignments.append(assignment)

                agent.available = False
                agent.current_task = best_task.task_id
                best_task.assigned_agents.append(agent.agent_id)
                self.assignments[agent.agent_id] = assignment

                # Check if task is fully assigned
                if len(best_task.assigned_agents) >= best_task.required_agents:
                    best_task.status = TaskStatus.ASSIGNED
                    best_task.start_time = time.time()
                    remaining_tasks.remove(best_task)

        return assignments

    def _find_best_agents(
        self,
        task: CoordinationTask,
        agents: List[AgentCapability]
    ) -> List[AgentCapability]:
        """Find the best agents for a task based on cost."""
        if not agents:
            return []

        # Calculate cost for each agent
        agent_costs = [
            (self._calculate_task_cost(agent, task), agent)
            for agent in agents
        ]

        # Sort by cost (lowest first)
        agent_costs.sort(key=lambda x: x[0])

        return [agent for _, agent in agent_costs]

    def _calculate_task_cost(
        self,
        agent: AgentCapability,
        task: CoordinationTask
    ) -> float:
        """
        Calculate the cost for an agent to complete a task.

        Cost includes:
        - Travel distance
        - Estimated time
        - Battery consumption
        - Skill match
        """
        # Distance to task
        dx = task.location[0] - agent.current_location[0]
        dy = task.location[1] - agent.current_location[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # Travel time
        travel_time = distance / max(agent.speed, 0.1)

        # Battery cost (penalize low battery)
        battery_penalty = 0
        if agent.battery_level < 0.3:
            battery_penalty = 10 * (0.3 - agent.battery_level)

        # Skill factor (lower skill = higher cost)
        skill_factor = 1.0 / max(agent.skill_level, 0.1)

        # Total cost
        cost = (
            distance * 1.0 +           # Distance weight
            travel_time * 0.5 +        # Time weight
            battery_penalty * 2.0 +    # Battery weight
            skill_factor * 0.3         # Skill weight
        )

        return cost

    def _create_assignment(
        self,
        agent: AgentCapability,
        task: CoordinationTask
    ) -> TaskAssignment:
        """Create a task assignment for an agent."""
        # Calculate estimates
        dx = task.location[0] - agent.current_location[0]
        dy = task.location[1] - agent.current_location[1]
        distance = np.sqrt(dx*dx + dy*dy)
        travel_time = distance / max(agent.speed, 0.1)

        current_time = time.time()
        estimated_arrival = current_time + travel_time
        estimated_completion = estimated_arrival + task.estimated_duration

        return TaskAssignment(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            assignment_time=current_time,
            estimated_arrival=estimated_arrival,
            estimated_completion=estimated_completion,
            cost=self._calculate_task_cost(agent, task)
        )

    def _check_dependencies(self, task: CoordinationTask) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                dep_task = self.active_tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            elif dep_id not in self.completed_tasks:
                return False
        return True

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute an assigned task.

        Args:
            task_id: ID of task to execute

        Returns:
            Execution result dictionary
        """
        if task_id not in self.active_tasks:
            return {"success": False, "error": "Task not found"}

        task = self.active_tasks[task_id]

        if task.status != TaskStatus.ASSIGNED:
            return {"success": False, "error": f"Task not assigned (status: {task.status})"}

        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Executing task {task_id} with agents {task.assigned_agents}")

        # Simulate task execution
        # In real implementation, this would coordinate actual robot movements
        result = {
            "success": True,
            "task_id": task_id,
            "agents": task.assigned_agents,
            "start_time": task.start_time,
            "execution_time": time.time() - (task.start_time or time.time())
        }

        return result

    async def complete_task(
        self,
        task_id: str,
        success: bool = True,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a task as completed and free assigned agents.

        Args:
            task_id: ID of task to complete
            success: Whether task completed successfully
            result: Optional result data

        Returns:
            True if successful
        """
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]
        task.completion_time = time.time()

        if success:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED

        # Free assigned agents
        for agent_id in task.assigned_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.available = True
                agent.current_task = None
                # Update agent location to task location
                agent.current_location = task.location

            if agent_id in self.assignments:
                del self.assignments[agent_id]

        # Record in history
        self.task_history.append({
            "task_id": task_id,
            "status": task.status.value,
            "duration": (task.completion_time - task.start_time) if task.start_time else 0,
            "agents": task.assigned_agents.copy(),
            "result": result
        })

        # Move to completed tasks
        self.completed_tasks[task_id] = task
        del self.active_tasks[task_id]

        logger.info(f"Task {task_id} completed with status {task.status.value}")
        return True

    def resolve_conflicts(self) -> List[Tuple[str, str]]:
        """
        Detect and resolve conflicts between agents.

        Returns:
            List of (agent_id, resolution) tuples
        """
        resolutions = []

        # Get agents with assignments
        assigned_agents = [
            (agent_id, self.agents[agent_id])
            for agent_id in self.assignments.keys()
            if agent_id in self.agents
        ]

        # Check for spatial conflicts (agents too close)
        for i, (id1, agent1) in enumerate(assigned_agents):
            for id2, agent2 in assigned_agents[i+1:]:
                # Calculate distance
                dx = agent1.current_location[0] - agent2.current_location[0]
                dy = agent1.current_location[1] - agent2.current_location[1]
                distance = np.sqrt(dx*dx + dy*dy)

                # If too close, need resolution
                if distance < 1.0:  # 1 meter threshold
                    # Lower priority agent waits
                    task1 = self.active_tasks.get(agent1.current_task)
                    task2 = self.active_tasks.get(agent2.current_task)

                    if task1 and task2:
                        if task1.priority < task2.priority:
                            resolutions.append((id1, "wait"))
                        else:
                            resolutions.append((id2, "wait"))
                    else:
                        resolutions.append((id1, "wait"))

        return resolutions

    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            "strategy": self.strategy.value,
            "total_agents": len(self.agents),
            "available_agents": len(self.get_available_agents()),
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.get_pending_tasks()),
            "completed_tasks": len(self.completed_tasks),
            "current_assignments": len(self.assignments)
        }

    def get_agent_workload(self) -> Dict[str, int]:
        """Get workload distribution across agents."""
        workload = {}
        for agent_id in self.agents:
            completed = sum(
                1 for entry in self.task_history
                if agent_id in entry.get("agents", [])
            )
            workload[agent_id] = completed
        return workload

    def update_agent_location(
        self,
        agent_id: str,
        location: Tuple[float, float]
    ) -> bool:
        """Update an agent's current location."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].current_location = location
        return True

    def update_agent_battery(self, agent_id: str, battery_level: float) -> bool:
        """Update an agent's battery level."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].battery_level = max(0.0, min(1.0, battery_level))
        return True
