"""
Performance Analysis System for Spatial AI Research Lab
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Represents a performance analysis result."""
    metric_name: str
    value: float
    baseline_value: Optional[float] = None
    improvement_percentage: Optional[float] = None
    statistical_significance: Optional[bool] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class PerformanceAnalyzer:
    """Analyzes performance metrics with statistical validation."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.performance_data: Dict[str, List[float]] = {}
        self.baseline_data: Dict[str, List[float]] = {}
        
    def record_performance(self, metric_name: str, value: float, is_baseline: bool = False) -> None:
        """Record a performance measurement."""
        if is_baseline:
            if metric_name not in self.baseline_data:
                self.baseline_data[metric_name] = []
            self.baseline_data[metric_name].append(value)
        else:
            if metric_name not in self.performance_data:
                self.performance_data[metric_name] = []
            self.performance_data[metric_name].append(value)
            
    def analyze_metric(self, metric_name: str) -> Optional[PerformanceResult]:
        """Analyze a specific performance metric."""
        if metric_name not in self.performance_data:
            return None
            
        values = self.performance_data[metric_name]
        if not values:
            return None
            
        mean_value = np.mean(values)
        
        result = PerformanceResult(
            metric_name=metric_name,
            value=mean_value
        )
        
        # Compare with baseline if available
        if metric_name in self.baseline_data and self.baseline_data[metric_name]:
            baseline_values = self.baseline_data[metric_name]
            baseline_mean = np.mean(baseline_values)
            result.baseline_value = baseline_mean
            
            if baseline_mean != 0:
                improvement = ((mean_value - baseline_mean) / baseline_mean) * 100
                result.improvement_percentage = improvement
                
            # Simple statistical significance test (t-test approximation)
            if len(values) > 1 and len(baseline_values) > 1:
                result.statistical_significance = self._is_significantly_different(
                    values, baseline_values
                )
                
        # Calculate confidence interval
        if len(values) > 1:
            std_error = np.std(values) / np.sqrt(len(values))
            margin = 1.96 * std_error  # 95% CI approximation
            result.confidence_interval = (mean_value - margin, mean_value + margin)
            
        return result
        
    def _is_significantly_different(self, values1: List[float], values2: List[float]) -> bool:
        """Simple statistical significance test."""
        if len(values1) < 2 or len(values2) < 2:
            return False
            
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1), np.std(values2)
        n1, n2 = len(values1), len(values2)
        
        # Pooled standard error
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        if pooled_se == 0:
            return False
            
        # Simple t-statistic
        t_stat = abs(mean1 - mean2) / pooled_se
        
        # Rough significance threshold (t > 2 for p < 0.05)
        return t_stat > 2.0
        
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance summary."""
        report = {
            'metrics_analyzed': len(self.performance_data),
            'total_measurements': sum(len(values) for values in self.performance_data.values()),
            'results': {}
        }
        
        for metric_name in self.performance_data.keys():
            result = self.analyze_metric(metric_name)
            if result:
                report['results'][metric_name] = {
                    'value': result.value,
                    'baseline_value': result.baseline_value,
                    'improvement_percentage': result.improvement_percentage,
                    'statistical_significance': result.statistical_significance,
                    'confidence_interval': result.confidence_interval
                }
                
        return report
        
    def clear_data(self) -> None:
        """Clear all performance data."""
        self.performance_data.clear()
        self.baseline_data.clear()
