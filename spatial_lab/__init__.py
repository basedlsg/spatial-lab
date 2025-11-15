"""
Focused Spatial AI Research Lab
A practical implementation of spatial reasoning using Nous Atropos infrastructure.

Core focus: Warehouse robot coordination and multi-agent spatial reasoning.
"""

__version__ = "0.1.0"
__author__ = "Spatial AI Research Lab"

from .environments import WarehouseSpatialEnvironment
from .coordination import MultiAgentCoordinator
from .evaluation import SpatialMetrics

__all__ = [
    "WarehouseSpatialEnvironment",
    "MultiAgentCoordinator", 
    "SpatialMetrics"
] 