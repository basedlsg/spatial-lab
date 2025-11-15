"""
Coordination Metrics for Multi-Agent Performance Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoordinationMetric:
    """Represents a coordination performance metric."""
    name: str
    value: float
    unit: str
    confidence_interval: Optional[tuple] = None
    sample_size: int = 0


class CoordinationMetrics:
    """Calculates and tracks coordination performance metrics."""
    
    def __init__(self):
        self.metrics_data: Dict[str, List[float]] = {}
        
    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics_data:
            self.metrics_data[metric_name] = []
        self.metrics_data[metric_name].append(value)
        
    def calculate_coordination_efficiency(self, task_completion_times: List[float]) -> CoordinationMetric:
        """Calculate coordination efficiency metric."""
        if not task_completion_times:
            return CoordinationMetric("coordination_efficiency", 0.0, "ratio")
            
        mean_time = np.mean(task_completion_times)
        efficiency = 1.0 / mean_time if mean_time > 0 else 0.0
        
        return CoordinationMetric(
            name="coordination_efficiency",
            value=efficiency,
            unit="tasks/second",
            sample_size=len(task_completion_times)
        )
        
    def calculate_resource_utilization(self, utilization_data: List[float]) -> CoordinationMetric:
        """Calculate resource utilization metric."""
        if not utilization_data:
            return CoordinationMetric("resource_utilization", 0.0, "ratio")
            
        mean_utilization = np.mean(utilization_data)
        
        return CoordinationMetric(
            name="resource_utilization",
            value=mean_utilization,
            unit="ratio",
            sample_size=len(utilization_data)
        )
        
    def get_all_metrics(self) -> Dict[str, CoordinationMetric]:
        """Get all calculated metrics."""
        metrics = {}
        
        for metric_name, values in self.metrics_data.items():
            if values:
                metrics[metric_name] = CoordinationMetric(
                    name=metric_name,
                    value=np.mean(values),
                    unit="various",
                    sample_size=len(values)
                )
                
        return metrics
