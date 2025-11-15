"""
Warehouse Task Generator

Creates realistic warehouse tasks for multi-agent coordination experiments.
Tasks focus on spatial reasoning, efficient path planning, and collaborative execution.
"""

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np


class TaskType(str, Enum):
    """Types of warehouse tasks"""
    ITEM_PICKING = "item_picking"
    BATCH_COLLECTION = "batch_collection"
    ZONE_REORGANIZATION = "zone_reorganization"
    COLLABORATIVE_TRANSPORT = "collaborative_transport"
    INVENTORY_COUNT = "inventory_count"
    PRIORITY_FULFILLMENT = "priority_fulfillment"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskItem:
    """An item required for a warehouse task"""
    item_id: str
    item_type: str
    source_shelf: str
    destination_zone: str
    priority: TaskPriority = TaskPriority.MEDIUM
    size: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    weight: float = 1.0
    special_handling: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "source_shelf": self.source_shelf,
            "destination_zone": self.destination_zone,
            "priority": self.priority.value,
            "size": self.size,
            "weight": self.weight,
            "special_handling": self.special_handling
        }


@dataclass
class WarehouseTask:
    """A complete warehouse task for multi-agent coordination"""
    task_id: str
    task_type: TaskType
    description: str
    items: List[TaskItem] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Task progress tracking
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    
    def start_task(self):
        """Mark task as started"""
        self.started_at = time.time()
        self.progress = {
            "items_collected": [],
            "items_delivered": [],
            "robots_assigned": [],
            "coordination_events": [],
            "completion_percentage": 0.0
        }
    
    def is_complete(self) -> bool:
        """Check if task is complete"""
        if not self.progress:
            return False
        
        total_items = len(self.items)
        delivered_items = len(self.progress.get("items_delivered", []))
        
        return delivered_items >= total_items
    
    def get_completion_percentage(self) -> float:
        """Get task completion percentage"""
        if not self.progress or not self.items:
            return 0.0
        
        total_items = len(self.items)
        delivered_items = len(self.progress.get("items_delivered", []))
        
        return min(100.0, (delivered_items / total_items) * 100.0)
    
    def get_remaining_items(self) -> List[TaskItem]:
        """Get items that still need to be collected/delivered"""
        if not self.progress:
            return self.items.copy()
        
        delivered_item_ids = set(self.progress.get("items_delivered", []))
        return [item for item in self.items if item.item_id not in delivered_item_ids]
    
    def get_priority_items(self) -> List[TaskItem]:
        """Get high priority items"""
        return [item for item in self.items if item.priority in [TaskPriority.HIGH, TaskPriority.URGENT]]
    
    def get_delivery_zones(self) -> List[str]:
        """Get unique delivery zones for this task"""
        return list(set(item.destination_zone for item in self.items))
    
    def get_source_shelves(self) -> List[str]:
        """Get unique source shelves for this task"""
        return list(set(item.source_shelf for item in self.items))
    
    async def update_progress(self, step_result: Dict):
        """Update task progress based on step results"""
        if not self.progress:
            self.start_task()
        
        # Update collected items
        newly_collected = step_result.get("items_collected", [])
        for item_id in newly_collected:
            if item_id not in self.progress["items_collected"]:
                self.progress["items_collected"].append(item_id)
        
        # Update delivered items
        newly_delivered = step_result.get("items_delivered", [])
        for item_id in newly_delivered:
            if item_id not in self.progress["items_delivered"]:
                self.progress["items_delivered"].append(item_id)
        
        # Update coordination events
        coordination_events = step_result.get("coordination_events", [])
        self.progress["coordination_events"].extend(coordination_events)
        
        # Update completion percentage
        self.progress["completion_percentage"] = self.get_completion_percentage()
        
        # Mark as complete if all items delivered
        if self.is_complete() and not self.completed_at:
            self.completed_at = time.time()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return self.progress.copy() if self.progress else {}
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary representation"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "items": [item.to_dict() for item in self.items],
            "constraints": self.constraints,
            "deadline": self.deadline,
            "priority": self.priority.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WarehouseTask':
        """Create task from dictionary representation"""
        task = cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            description=data["description"],
            constraints=data.get("constraints", {}),
            deadline=data.get("deadline"),
            priority=TaskPriority(data.get("priority", "medium")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            progress=data.get("progress", {})
        )
        
        # Reconstruct items
        for item_data in data.get("items", []):
            item = TaskItem(
                item_id=item_data["item_id"],
                item_type=item_data["item_type"],
                source_shelf=item_data["source_shelf"],
                destination_zone=item_data["destination_zone"],
                priority=TaskPriority(item_data.get("priority", "medium")),
                size=tuple(item_data.get("size", (0.3, 0.3, 0.3))),
                weight=item_data.get("weight", 1.0),
                special_handling=item_data.get("special_handling", False)
            )
            task.items.append(item)
        
        return task


class WarehouseTaskGenerator:
    """Generates realistic warehouse tasks for coordination experiments"""
    
    def __init__(self, complexity: str = "medium", items_per_task: int = 10):
        self.complexity = complexity
        self.items_per_task = items_per_task
        
        # Task templates based on complexity
        self.task_templates = {
            "easy": {
                "max_items": 5,
                "max_zones": 2,
                "priority_distribution": {"low": 0.7, "medium": 0.3, "high": 0.0},
                "special_handling_chance": 0.1
            },
            "medium": {
                "max_items": 10,
                "max_zones": 3,
                "priority_distribution": {"low": 0.4, "medium": 0.5, "high": 0.1},
                "special_handling_chance": 0.2
            },
            "hard": {
                "max_items": 20,
                "max_zones": 5,
                "priority_distribution": {"low": 0.2, "medium": 0.5, "high": 0.3},
                "special_handling_chance": 0.3
            }
        }
        
        self.item_types = [
            "electronics", "clothing", "books", "tools", "food",
            "toys", "furniture", "sports", "automotive", "medical",
            "hardware", "beauty", "home", "garden", "office"
        ]
    
    async def generate_task(self, layout, robots: List) -> WarehouseTask:
        """Generate a warehouse task based on current layout and robot states"""
        
        task_type = self._select_task_type()
        
        if task_type == TaskType.ITEM_PICKING:
            return await self._generate_item_picking_task(layout, robots)
        elif task_type == TaskType.BATCH_COLLECTION:
            return await self._generate_batch_collection_task(layout, robots)
        elif task_type == TaskType.COLLABORATIVE_TRANSPORT:
            return await self._generate_collaborative_transport_task(layout, robots)
        elif task_type == TaskType.PRIORITY_FULFILLMENT:
            return await self._generate_priority_fulfillment_task(layout, robots)
        else:
            # Default to item picking
            return await self._generate_item_picking_task(layout, robots)
    
    def _select_task_type(self) -> TaskType:
        """Select task type based on complexity settings"""
        
        if self.complexity == "easy":
            # Simpler tasks for easy mode
            task_types = [TaskType.ITEM_PICKING, TaskType.BATCH_COLLECTION]
            weights = [0.7, 0.3]
        elif self.complexity == "medium":
            task_types = [
                TaskType.ITEM_PICKING, 
                TaskType.BATCH_COLLECTION,
                TaskType.COLLABORATIVE_TRANSPORT,
                TaskType.PRIORITY_FULFILLMENT
            ]
            weights = [0.4, 0.3, 0.2, 0.1]
        else:  # hard
            task_types = list(TaskType)
            weights = [0.2, 0.25, 0.15, 0.2, 0.1, 0.1]
        
        return np.random.choice(task_types, p=weights)
    
    async def _generate_item_picking_task(self, layout, robots: List) -> WarehouseTask:
        """Generate a standard item picking task"""
        
        task_id = f"pick_{uuid.uuid4().hex[:8]}"
        template = self.task_templates[self.complexity]
        
        # Select number of items
        num_items = min(
            random.randint(3, template["max_items"]),
            self.items_per_task
        )
        
        # Select shelves with items
        available_shelves = [shelf for shelf in layout.shelves if shelf.items]
        if len(available_shelves) < num_items:
            available_shelves = layout.shelves  # Use all shelves if needed
        
        selected_shelves = random.sample(available_shelves, min(num_items, len(available_shelves)))
        
        # Select delivery zones
        delivery_zones = layout.get_zone_by_type("shipping") + layout.get_zone_by_type("picking")
        if not delivery_zones:
            # Create default delivery zones if none exist
            delivery_zones = [{"zone_id": "default_delivery", "zone_type": "delivery"}]
        
        # Generate task items
        items = []
        for i, shelf in enumerate(selected_shelves):
            # Select item from shelf
            if shelf.items:
                item_id = random.choice(shelf.items)
            else:
                item_id = f"item_{i:03d}"
            
            # Assign priority based on distribution
            priority = self._select_priority(template["priority_distribution"])
            
            # Select destination zone
            dest_zone = random.choice(delivery_zones)
            
            task_item = TaskItem(
                item_id=item_id,
                item_type=random.choice(self.item_types),
                source_shelf=shelf.shelf_id,
                destination_zone=dest_zone.zone_id if hasattr(dest_zone, 'zone_id') else dest_zone["zone_id"],
                priority=priority,
                size=self._generate_item_size(),
                weight=random.uniform(0.5, 5.0),
                special_handling=random.random() < template["special_handling_chance"]
            )
            
            items.append(task_item)
        
        # Create task
        task = WarehouseTask(
            task_id=task_id,
            task_type=TaskType.ITEM_PICKING,
            description=f"Collect {num_items} items from various shelves and deliver to designated zones",
            items=items,
            constraints={
                "max_robots": len(robots),
                "time_limit": 300,  # 5 minutes
                "coordination_required": num_items > 5
            },
            deadline=time.time() + 300,  # 5 minutes from now
            priority=self._get_overall_task_priority(items)
        )
        
        return task
    
    async def _generate_batch_collection_task(self, layout, robots: List) -> WarehouseTask:
        """Generate a batch collection task where items must be collected in groups"""
        
        task_id = f"batch_{uuid.uuid4().hex[:8]}"
        template = self.task_templates[self.complexity]
        
        # Create batches of related items
        batch_types = random.sample(self.item_types, min(3, len(self.item_types)))
        items = []
        
        for batch_type in batch_types:
            batch_size = random.randint(2, 5)
            
            # Find shelves that might have this item type
            candidate_shelves = random.sample(layout.shelves, min(batch_size, len(layout.shelves)))
            
            for i, shelf in enumerate(candidate_shelves):
                item_id = f"{batch_type}_{i:03d}"
                priority = self._select_priority(template["priority_distribution"])
                
                # All items in batch go to same zone
                delivery_zones = layout.get_zone_by_type("shipping")
                if not delivery_zones:
                    delivery_zones = [{"zone_id": "batch_delivery", "zone_type": "delivery"}]
                
                dest_zone = delivery_zones[0]
                
                task_item = TaskItem(
                    item_id=item_id,
                    item_type=batch_type,
                    source_shelf=shelf.shelf_id,
                    destination_zone=dest_zone.zone_id if hasattr(dest_zone, 'zone_id') else dest_zone["zone_id"],
                    priority=priority,
                    size=self._generate_item_size(),
                    weight=random.uniform(0.5, 3.0),
                    special_handling=random.random() < template["special_handling_chance"]
                )
                
                items.append(task_item)
        
        task = WarehouseTask(
            task_id=task_id,
            task_type=TaskType.BATCH_COLLECTION,
            description=f"Collect batches of related items: {', '.join(batch_types)}",
            items=items,
            constraints={
                "batch_types": batch_types,
                "coordination_required": True,
                "batch_delivery": True,
                "time_limit": 400
            },
            deadline=time.time() + 400,
            priority=self._get_overall_task_priority(items)
        )
        
        return task
    
    async def _generate_collaborative_transport_task(self, layout, robots: List) -> WarehouseTask:
        """Generate a task requiring multiple robots to transport large items"""
        
        task_id = f"collab_{uuid.uuid4().hex[:8]}"
        
        # Generate fewer, larger items that require collaboration
        num_items = random.randint(2, 5)
        items = []
        
        available_shelves = random.sample(layout.shelves, min(num_items, len(layout.shelves)))
        delivery_zones = layout.get_zone_by_type("shipping")
        if not delivery_zones:
            delivery_zones = [{"zone_id": "collab_delivery", "zone_type": "delivery"}]
        
        for i, shelf in enumerate(available_shelves):
            item_id = f"large_item_{i:03d}"
            
            task_item = TaskItem(
                item_id=item_id,
                item_type="furniture",  # Large items
                source_shelf=shelf.shelf_id,
                destination_zone=delivery_zones[0].zone_id if hasattr(delivery_zones[0], 'zone_id') else delivery_zones[0]["zone_id"],
                priority=TaskPriority.HIGH,  # Collaborative tasks are usually high priority
                size=(1.5, 1.0, 0.8),  # Large size
                weight=random.uniform(10.0, 50.0),  # Heavy items
                special_handling=True  # Requires special handling
            )
            
            items.append(task_item)
        
        task = WarehouseTask(
            task_id=task_id,
            task_type=TaskType.COLLABORATIVE_TRANSPORT,
            description=f"Collaborate to transport {num_items} large items requiring multiple robots",
            items=items,
            constraints={
                "min_robots_per_item": 2,
                "coordination_required": True,
                "special_equipment": True,
                "time_limit": 600
            },
            deadline=time.time() + 600,
            priority=TaskPriority.HIGH
        )
        
        return task
    
    async def _generate_priority_fulfillment_task(self, layout, robots: List) -> WarehouseTask:
        """Generate a task with mixed priorities requiring smart coordination"""
        
        task_id = f"priority_{uuid.uuid4().hex[:8]}"
        template = self.task_templates[self.complexity]
        
        num_items = min(random.randint(8, template["max_items"]), self.items_per_task)
        items = []
        
        # Ensure we have a good mix of priorities
        priority_counts = {
            TaskPriority.URGENT: max(1, num_items // 5),
            TaskPriority.HIGH: max(1, num_items // 3),
            TaskPriority.MEDIUM: num_items // 2,
            TaskPriority.LOW: num_items // 4
        }
        
        # Adjust counts to match total items
        total_assigned = sum(priority_counts.values())
        if total_assigned > num_items:
            priority_counts[TaskPriority.MEDIUM] -= (total_assigned - num_items)
        elif total_assigned < num_items:
            priority_counts[TaskPriority.MEDIUM] += (num_items - total_assigned)
        
        available_shelves = layout.shelves
        delivery_zones = layout.get_zone_by_type("shipping") + layout.get_zone_by_type("picking")
        if not delivery_zones:
            delivery_zones = [{"zone_id": "priority_delivery", "zone_type": "delivery"}]
        
        item_count = 0
        for priority, count in priority_counts.items():
            for _ in range(count):
                if item_count >= num_items:
                    break
                
                shelf = random.choice(available_shelves)
                dest_zone = random.choice(delivery_zones)
                
                task_item = TaskItem(
                    item_id=f"priority_item_{item_count:03d}",
                    item_type=random.choice(self.item_types),
                    source_shelf=shelf.shelf_id,
                    destination_zone=dest_zone.zone_id if hasattr(dest_zone, 'zone_id') else dest_zone["zone_id"],
                    priority=priority,
                    size=self._generate_item_size(),
                    weight=random.uniform(0.5, 8.0),
                    special_handling=priority in [TaskPriority.URGENT, TaskPriority.HIGH]
                )
                
                items.append(task_item)
                item_count += 1
        
        task = WarehouseTask(
            task_id=task_id,
            task_type=TaskType.PRIORITY_FULFILLMENT,
            description=f"Fulfill mixed priority order with {num_items} items - prioritize urgent items first",
            items=items,
            constraints={
                "priority_order": ["urgent", "high", "medium", "low"],
                "coordination_required": True,
                "time_penalties": True,
                "time_limit": 450
            },
            deadline=time.time() + 450,
            priority=TaskPriority.HIGH
        )
        
        return task
    
    def _select_priority(self, priority_distribution: Dict[str, float]) -> TaskPriority:
        """Select priority based on distribution"""
        priorities = list(priority_distribution.keys())
        weights = list(priority_distribution.values())
        
        selected = np.random.choice(priorities, p=weights)
        return TaskPriority(selected)
    
    def _generate_item_size(self) -> Tuple[float, float, float]:
        """Generate realistic item size"""
        size_types = {
            "small": (0.2, 0.2, 0.2),
            "medium": (0.4, 0.3, 0.3),
            "large": (0.8, 0.6, 0.4),
            "extra_large": (1.2, 0.8, 0.6)
        }
        
        size_type = np.random.choice(
            list(size_types.keys()),
            p=[0.4, 0.4, 0.15, 0.05]
        )
        
        base_size = size_types[size_type]
        
        # Add some variation
        return tuple(
            base_size[i] * random.uniform(0.8, 1.2)
            for i in range(3)
        )
    
    def _get_overall_task_priority(self, items: List[TaskItem]) -> TaskPriority:
        """Determine overall task priority based on item priorities"""
        
        priorities = [item.priority for item in items]
        
        if TaskPriority.URGENT in priorities:
            return TaskPriority.URGENT
        elif TaskPriority.HIGH in priorities:
            return TaskPriority.HIGH
        elif priorities.count(TaskPriority.MEDIUM) > len(priorities) // 2:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    async def generate_evaluation_tasks(self, layout, robots: List, num_tasks: int = 10) -> List[WarehouseTask]:
        """Generate a set of evaluation tasks for benchmarking"""
        
        tasks = []
        
        # Generate diverse task types for comprehensive evaluation
        task_types = [
            TaskType.ITEM_PICKING,
            TaskType.BATCH_COLLECTION,
            TaskType.COLLABORATIVE_TRANSPORT,
            TaskType.PRIORITY_FULFILLMENT
        ]
        
        for i in range(num_tasks):
            task_type = task_types[i % len(task_types)]
            
            if task_type == TaskType.ITEM_PICKING:
                task = await self._generate_item_picking_task(layout, robots)
            elif task_type == TaskType.BATCH_COLLECTION:
                task = await self._generate_batch_collection_task(layout, robots)
            elif task_type == TaskType.COLLABORATIVE_TRANSPORT:
                task = await self._generate_collaborative_transport_task(layout, robots)
            else:
                task = await self._generate_priority_fulfillment_task(layout, robots)
            
            tasks.append(task)
        
        return tasks 