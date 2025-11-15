"""
Spatial Reasoning Metrics

Comprehensive metrics for evaluating spatial reasoning performance in warehouse
coordination tasks, following rigorous scientific evaluation standards.
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


@dataclass
class SpatialMetrics:
    """Container for spatial reasoning evaluation metrics"""
    
    # Path efficiency metrics
    path_efficiency: float = 0.0  # actual_path_length / optimal_path_length
    path_smoothness: float = 0.0  # measure of path smoothness (lower is better)
    path_deviation: float = 0.0  # deviation from optimal path
    
    # Navigation performance
    navigation_success_rate: float = 0.0  # percentage of successful navigations
    collision_rate: float = 0.0  # collisions per unit time
    obstacle_avoidance_success: float = 0.0  # successful obstacle avoidances
    
    # Spatial memory and learning
    spatial_memory_accuracy: float = 0.0  # accuracy of spatial representations
    exploration_coverage: float = 0.0  # percentage of space explored
    revisit_efficiency: float = 0.0  # efficiency of returning to known locations
    
    # Multi-agent coordination
    coordination_efficiency: float = 0.0  # efficiency of multi-agent coordination
    space_utilization: float = 0.0  # how well agents utilize available space
    conflict_resolution_time: float = 0.0  # time to resolve spatial conflicts
    
    # Task completion metrics
    task_completion_rate: float = 0.0  # percentage of tasks completed
    task_completion_time: float = 0.0  # average time to complete tasks
    resource_utilization: float = 0.0  # efficiency of resource usage

    # Shape vs Metadata
    shape_priority_rate: float = 0.0 # percentage of times the LLM prioritizes shape over metadata
    
    # Statistical measures
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation"""
        return {
            "path_efficiency": {
                "value": self.path_efficiency,
                "ci": self.confidence_intervals.get("path_efficiency", (0, 0)),
                "effect_size": self.effect_sizes.get("path_efficiency", 0),
                "p_value": self.p_values.get("path_efficiency", 1.0)
            },
            "navigation_success_rate": {
                "value": self.navigation_success_rate,
                "ci": self.confidence_intervals.get("navigation_success_rate", (0, 0)),
                "effect_size": self.effect_sizes.get("navigation_success_rate", 0),
                "p_value": self.p_values.get("navigation_success_rate", 1.0)
            },
            "collision_rate": {
                "value": self.collision_rate,
                "ci": self.confidence_intervals.get("collision_rate", (0, 0)),
                "effect_size": self.effect_sizes.get("collision_rate", 0),
                "p_value": self.p_values.get("collision_rate", 1.0)
            },
            "coordination_efficiency": {
                "value": self.coordination_efficiency,
                "ci": self.confidence_intervals.get("coordination_efficiency", (0, 0)),
                "effect_size": self.effect_sizes.get("coordination_efficiency", 0),
                "p_value": self.p_values.get("coordination_efficiency", 1.0)
            },
            "task_completion_rate": {
                "value": self.task_completion_rate,
                "ci": self.confidence_intervals.get("task_completion_rate", (0, 0)),
                "effect_size": self.effect_sizes.get("task_completion_rate", 0),
                "p_value": self.p_values.get("task_completion_rate", 1.0)
            },
            "shape_priority_rate": {
                "value": self.shape_priority_rate,
                "ci": self.confidence_intervals.get("shape_priority_rate", (0, 0)),
                "effect_size": self.effect_sizes.get("shape_priority_rate", 0),
                "p_value": self.p_values.get("shape_priority_rate", 1.0)
            },
        }


class SpatialMetricsCalculator:
    """Calculator for spatial reasoning metrics with statistical rigor"""
    
    def __init__(self):
        self.episode_data = []
        self.baseline_data = None
        self.current_episode_metrics = {}
        
        # Baseline performance for comparison
        self.baseline_metrics = {
            "random_agent_path_efficiency": 0.3,
            "random_agent_collision_rate": 0.8,
            "random_agent_completion_rate": 0.1,
            "rule_based_path_efficiency": 0.6,
            "rule_based_collision_rate": 0.2,
            "rule_based_completion_rate": 0.4
        }
    
    async def calculate_step_metrics(self, step_result: Dict) -> Dict[str, float]:
        """Calculate metrics for a single step"""
        
        step_metrics = {}
        
        # Navigation metrics
        robot_actions = step_result.get("robot_actions", {})
        successful_moves = sum(1 for action in robot_actions.values() 
                             if action.get("action") == "move_to" and action.get("success"))
        total_move_attempts = sum(1 for action in robot_actions.values() 
                                if action.get("action") == "move_to")
        
        step_metrics["navigation_success_rate"] = (
            successful_moves / max(1, total_move_attempts)
        )
        
        # Collision metrics
        collisions = step_result.get("collisions", [])
        step_metrics["collision_count"] = len(collisions)
        step_metrics["collision_rate"] = len(collisions) / max(1, len(robot_actions))
        
        # Coordination metrics
        coordination_events = step_result.get("coordination_events", [])
        step_metrics["coordination_events"] = len(coordination_events)
        
        # Progress metrics
        step_metrics["progress_made"] = step_result.get("progress_made", 0.0)
        step_metrics["items_collected"] = len(step_result.get("items_collected", []))
        step_metrics["items_delivered"] = len(step_result.get("items_delivered", []))
        
        # Store for episode aggregation
        self.current_episode_metrics.setdefault("step_metrics", []).append(step_metrics)
        
        return step_metrics
    
    async def calculate_episode_metrics(self, trajectories: List, task) -> SpatialMetrics:
        """Calculate comprehensive metrics for a complete episode"""
        
        if not trajectories:
            return SpatialMetrics()
        
        # Extract step metrics from trajectories
        step_metrics_list = []
        robot_paths = {}
        
        # Track shape priority
        shape_priority_count = 0
        total_tasks = 0

        for trajectory in trajectories:
            step_data = trajectory.metadata
            step_metrics_list.append(step_data.get("metrics", {}))
            
            # Extract robot paths for path analysis
            robot_states = step_data.get("robot_states", [])
            for robot_state in robot_states:
                robot_id = robot_state["robot_id"]
                position = robot_state["position"]
                
                if robot_id not in robot_paths:
                    robot_paths[robot_id] = []
                robot_paths[robot_id].append(position[:2])  # x, y coordinates

            # Check if the LLM prioritized shape over metadata
            closest_object_id = step_data.get("closest_object_id")
            if closest_object_id:
                total_tasks += 1
                reference_shape = task.reference_object_id.split('_')[-1]
                if reference_shape in closest_object_id:
                    shape_priority_count += 1
        
        # Calculate path efficiency metrics
        path_metrics = await self._calculate_path_metrics(robot_paths, task)
        
        # Calculate navigation performance
        navigation_metrics = await self._calculate_navigation_metrics(step_metrics_list)
        
        # Calculate coordination metrics
        coordination_metrics = await self._calculate_coordination_metrics(step_metrics_list)
        
        # Calculate task completion metrics
        completion_metrics = await self._calculate_completion_metrics(trajectories, task)
        
        # Calculate shape priority rate
        shape_priority_rate = (shape_priority_count / total_tasks) if total_tasks else 0.0

        # Combine all metrics
        metrics = SpatialMetrics(
            path_efficiency=path_metrics["efficiency"],
            path_smoothness=path_metrics["smoothness"],
            path_deviation=path_metrics["deviation"],
            navigation_success_rate=navigation_metrics["success_rate"],
            collision_rate=navigation_metrics["collision_rate"],
            obstacle_avoidance_success=navigation_metrics["obstacle_avoidance"],
            coordination_efficiency=coordination_metrics["efficiency"],
            space_utilization=coordination_metrics["space_utilization"],
            conflict_resolution_time=coordination_metrics["conflict_resolution_time"],
            task_completion_rate=completion_metrics["completion_rate"],
            task_completion_time=completion_metrics["completion_time"],
            resource_utilization=completion_metrics["resource_utilization"],
            shape_priority_rate=shape_priority_rate
        )
        
        # Calculate statistical measures
        await self._calculate_statistical_measures(metrics)
        
        # Store episode data
        self.episode_data.append(metrics)
        
        return metrics
    
    async def _calculate_path_metrics(self, robot_paths: Dict[str, List], task) -> Dict[str, float]:
        """Calculate path efficiency and quality metrics"""
        
        if not robot_paths:
            return {"efficiency": 0.0, "smoothness": 0.0, "deviation": 0.0}
        
        path_efficiencies = []
        path_smoothness_values = []
        path_deviations = []
        
        for robot_id, path in robot_paths.items():
            if len(path) < 2:
                continue
            
            # Calculate actual path length
            actual_length = 0.0
            for i in range(1, len(path)):
                actual_length += np.linalg.norm(
                    np.array(path[i]) - np.array(path[i-1])
                )
            
            # Calculate optimal path length (straight line from start to end)
            optimal_length = np.linalg.norm(
                np.array(path[-1]) - np.array(path[0])
            )
            
            if optimal_length > 0:
                efficiency = optimal_length / actual_length
                path_efficiencies.append(min(1.0, efficiency))  # Cap at 1.0
            
            # Calculate path smoothness (sum of direction changes)
            if len(path) >= 3:
                direction_changes = 0.0
                for i in range(1, len(path) - 1):
                    v1 = np.array(path[i]) - np.array(path[i-1])
                    v2 = np.array(path[i+1]) - np.array(path[i])
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        v1_norm = v1 / np.linalg.norm(v1)
                        v2_norm = v2 / np.linalg.norm(v2)
                        
                        # Calculate angle between vectors
                        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                        angle_change = np.arccos(cos_angle)
                        direction_changes += angle_change
                
                # Normalize by path length
                smoothness = 1.0 / (1.0 + direction_changes / len(path))
                path_smoothness_values.append(smoothness)
            
            # Calculate path deviation from optimal
            if optimal_length > 0:
                deviation = (actual_length - optimal_length) / optimal_length
                path_deviations.append(deviation)
        
        return {
            "efficiency": np.mean(path_efficiencies) if path_efficiencies else 0.0,
            "smoothness": np.mean(path_smoothness_values) if path_smoothness_values else 0.0,
            "deviation": np.mean(path_deviations) if path_deviations else 0.0
        }
    
    async def _calculate_navigation_metrics(self, step_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate navigation performance metrics"""
        
        if not step_metrics:
            return {"success_rate": 0.0, "collision_rate": 0.0, "obstacle_avoidance": 0.0}
        
        # Navigation success rate
        success_rates = [m.get("navigation_success_rate", 0.0) for m in step_metrics]
        avg_success_rate = np.mean(success_rates)
        
        # Collision rate
        collision_counts = [m.get("collision_count", 0) for m in step_metrics]
        total_collisions = sum(collision_counts)
        collision_rate = total_collisions / len(step_metrics)
        
        # Obstacle avoidance (inverse of collision rate, adjusted)
        obstacle_avoidance = max(0.0, 1.0 - collision_rate / 2.0)
        
        return {
            "success_rate": avg_success_rate,
            "collision_rate": collision_rate,
            "obstacle_avoidance": obstacle_avoidance
        }
    
    async def _calculate_coordination_metrics(self, step_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate multi-agent coordination metrics"""
        
        if not step_metrics:
            return {"efficiency": 0.0, "space_utilization": 0.0, "conflict_resolution_time": 0.0}
        
        # Coordination efficiency based on coordination events
        coordination_events = [m.get("coordination_events", 0) for m in step_metrics]
        total_events = sum(coordination_events)
        
        # Higher coordination events generally indicate better coordination
        # but normalize by episode length
        coordination_efficiency = min(1.0, total_events / max(1, len(step_metrics)))
        
        # Space utilization (simplified - based on progress made)
        progress_values = [m.get("progress_made", 0.0) for m in step_metrics]
        space_utilization = np.mean(progress_values) if progress_values else 0.0
        
        # Conflict resolution time (simplified - based on collision recovery)
        collision_counts = [m.get("collision_count", 0) for m in step_metrics]
        
        # Calculate average time to resolve conflicts
        conflict_resolution_time = 0.0
        if total_events > 0:
            # Estimate based on coordination events and collisions
            conflict_resolution_time = sum(collision_counts) / max(1, total_events)
        
        return {
            "efficiency": coordination_efficiency,
            "space_utilization": min(1.0, space_utilization),
            "conflict_resolution_time": conflict_resolution_time
        }
    
    async def _calculate_completion_metrics(self, trajectories: List, task) -> Dict[str, float]:
        """Calculate task completion metrics"""
        
        if not trajectories:
            return {"completion_rate": 0.0, "completion_time": 0.0, "resource_utilization": 0.0}
        
        # Task completion rate
        final_trajectory = trajectories[-1]
        task_progress = final_trajectory.metadata.get("task_progress", {})
        completion_percentage = task_progress.get("completion_percentage", 0.0)
        completion_rate = completion_percentage / 100.0
        
        # Task completion time
        completion_time = len(trajectories)  # Number of steps
        
        # Resource utilization (items collected/delivered vs. total items)
        total_items = len(task.items) if task else 1
        items_delivered = len(task_progress.get("items_delivered", [])) if task_progress else 0
        resource_utilization = items_delivered / total_items
        
        return {
            "completion_rate": completion_rate,
            "completion_time": completion_time,
            "resource_utilization": resource_utilization
        }
    
    async def _calculate_statistical_measures(self, metrics: SpatialMetrics):
        """Calculate statistical measures including confidence intervals and effect sizes"""
        
        if len(self.episode_data) < 2:
            return  # Need at least 2 episodes for statistical analysis
        
        # Extract values for each metric from episode history
        metric_values = {
            "path_efficiency": [ep.path_efficiency for ep in self.episode_data],
            "navigation_success_rate": [ep.navigation_success_rate for ep in self.episode_data],
            "collision_rate": [ep.collision_rate for ep in self.episode_data],
            "coordination_efficiency": [ep.coordination_efficiency for ep in self.episode_data],
            "task_completion_rate": [ep.task_completion_rate for ep in self.episode_data],
            "shape_priority_rate": [ep.shape_priority_rate for ep in self.episode_data]
        }
        
        # Calculate confidence intervals and effect sizes
        for metric_name, values in metric_values.items():
            if len(values) >= 3:  # Need at least 3 values for meaningful statistics
                
                # 95% confidence interval
                mean_val = np.mean(values)
                sem = stats.sem(values)
                ci = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=sem)
                metrics.confidence_intervals[metric_name] = ci
                
                # Effect size (Cohen's d) compared to baseline
                baseline_key = f"random_agent_{metric_name.split('_')[0]}_{'_'.join(metric_name.split('_')[1:])}"
                if baseline_key in self.baseline_metrics:
                    baseline_mean = self.baseline_metrics[baseline_key]
                    pooled_std = np.std(values + [baseline_mean] * len(values))
                    if pooled_std > 0:
                        cohens_d = (mean_val - baseline_mean) / pooled_std
                        metrics.effect_sizes[metric_name] = cohens_d
                
                # Statistical significance test (t-test against baseline)
                if baseline_key in self.baseline_metrics:
                    baseline_mean = self.baseline_metrics[baseline_key]
                    t_stat, p_value = stats.ttest_1samp(values, baseline_mean)
                    metrics.p_values[metric_name] = p_value
    
    async def calculate_efficiency_score(self, trajectories: List) -> float:
        """Calculate overall efficiency score for trajectory scoring"""
        
        if not trajectories:
            return 0.0
        
        # Extract efficiency-related metrics
        efficiency_scores = []
        
        for trajectory in trajectories:
            step_data = trajectory.metadata
            metrics = step_data.get("metrics", {})
            
            # Combine multiple efficiency indicators
            navigation_success = metrics.get("navigation_success_rate", 0.0)
            progress_made = metrics.get("progress_made", 0.0)
            collision_penalty = metrics.get("collision_count", 0) * -0.1
            
            step_efficiency = navigation_success + progress_made + collision_penalty
            efficiency_scores.append(max(0.0, step_efficiency))
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    async def calculate_coordination_score(self, trajectories: List) -> float:
        """Calculate coordination score for trajectory scoring"""
        
        if not trajectories:
            return 0.0
        
        coordination_scores = []
        
        for trajectory in trajectories:
            step_data = trajectory.metadata
            metrics = step_data.get("metrics", {})
            
            # Coordination indicators
            coordination_events = metrics.get("coordination_events", 0)
            collision_penalty = metrics.get("collision_count", 0) * -0.2
            
            step_coordination = coordination_events * 0.5 + collision_penalty
            coordination_scores.append(max(0.0, step_coordination))
        
        return np.mean(coordination_scores) if coordination_scores else 0.0
    
    async def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode metrics"""
        
        if not self.current_episode_metrics:
            return {}
        
        step_metrics = self.current_episode_metrics.get("step_metrics", [])
        
        if not step_metrics:
            return {}
        
        summary = {
            "total_steps": len(step_metrics),
            "avg_navigation_success": np.mean([m.get("navigation_success_rate", 0) for m in step_metrics]),
            "total_collisions": sum([m.get("collision_count", 0) for m in step_metrics]),
            "total_coordination_events": sum([m.get("coordination_events", 0) for m in step_metrics]),
            "total_progress": sum([m.get("progress_made", 0) for m in step_metrics]),
            "items_collected": sum([m.get("items_collected", 0) for m in step_metrics]),
            "items_delivered": sum([m.get("items_delivered", 0) for m in step_metrics])
        }
        
        # Clear current episode data
        self.current_episode_metrics = {}
        
        return summary
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative analysis against baselines"""
        
        if not self.episode_data:
            return {}
        
        # Calculate current performance
        current_metrics = {
            "path_efficiency": np.mean([ep.path_efficiency for ep in self.episode_data]),
            "collision_rate": np.mean([ep.collision_rate for ep in self.episode_data]),
            "completion_rate": np.mean([ep.task_completion_rate for ep in self.episode_data])
        }
        
        # Compare to baselines
        comparisons = {}
        
        # vs Random Agent
        comparisons["vs_random_agent"] = {
            "path_efficiency_improvement": (
                current_metrics["path_efficiency"] - self.baseline_metrics["random_agent_path_efficiency"]
            ) / self.baseline_metrics["random_agent_path_efficiency"] * 100,
            "collision_reduction": (
                self.baseline_metrics["random_agent_collision_rate"] - current_metrics["collision_rate"]
            ) / self.baseline_metrics["random_agent_collision_rate"] * 100,
            "completion_improvement": (
                current_metrics["completion_rate"] - self.baseline_metrics["random_agent_completion_rate"]
            ) / self.baseline_metrics["random_agent_completion_rate"] * 100
        }
        
        # vs Rule-based Agent
        comparisons["vs_rule_based"] = {
            "path_efficiency_improvement": (
                current_metrics["path_efficiency"] - self.baseline_metrics["rule_based_path_efficiency"]
            ) / self.baseline_metrics["rule_based_path_efficiency"] * 100,
            "collision_reduction": (
                self.baseline_metrics["rule_based_collision_rate"] - current_metrics["collision_rate"]
            ) / self.baseline_metrics["rule_based_collision_rate"] * 100,
            "completion_improvement": (
                current_metrics["completion_rate"] - self.baseline_metrics["rule_based_completion_rate"]
            ) / self.baseline_metrics["rule_based_completion_rate"] * 100
        }
        
        return {
            "current_performance": current_metrics,
            "baseline_comparisons": comparisons,
            "sample_size": len(self.episode_data)
        }