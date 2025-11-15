"""
Statistical Analysis System for Spatial AI Research Lab
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Represents a statistical analysis result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    significant: bool = False
    interpretation: str = ""


class StatisticalAnalyzer:
    """Performs statistical analysis with scientific rigor."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results: List[StatisticalResult] = []
        
    def t_test(self, group1: List[float], group2: List[float], 
               test_name: str = "t_test") -> StatisticalResult:
        """Perform independent samples t-test."""
        try:
            statistic, p_value = stats.ttest_ind(group1, group2)
            
            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0
            
            result = StatisticalResult(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                significant=p_value < self.alpha,
                interpretation=self._interpret_effect_size(effect_size)
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in t-test: {e}")
            return StatisticalResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                interpretation="Error in calculation"
            )
            
    def confidence_interval(self, data: List[float], 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        if len(data) < 2:
            return (0.0, 0.0)
            
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)
        
        return (mean - h, mean + h)
        
    def effect_size_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0
            
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (np.mean(group1) - np.mean(group2)) / pooled_std
        
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible effect"
        elif abs_effect < 0.5:
            return "small effect"
        elif abs_effect < 0.8:
            return "medium effect"
        else:
            return "large effect"
            
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = "bonferroni") -> List[float]:
        """Apply multiple comparisons correction."""
        if method == "bonferroni":
            return [min(p * len(p_values), 1.0) for p in p_values]
        else:
            # Default to no correction
            return p_values
            
    def descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics."""
        if not data:
            return {}
            
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'count': len(data)
        }
        
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all statistical analyses performed."""
        return {
            'total_tests': len(self.results),
            'significant_results': len([r for r in self.results if r.significant]),
            'results': [
                {
                    'test_name': r.test_name,
                    'p_value': r.p_value,
                    'effect_size': r.effect_size,
                    'significant': r.significant,
                    'interpretation': r.interpretation
                }
                for r in self.results
            ]
        }
        
    def clear_results(self) -> None:
        """Clear all analysis results."""
        self.results.clear()
