"""
Focused Spatial AI Research Lab
A practical implementation of spatial reasoning using Nous Atropos infrastructure.

Core focus: Warehouse robot coordination and multi-agent spatial reasoning.
"""

__version__ = "1.0.0"
__author__ = "Spatial AI Research Lab"

__all__ = []

# Import coordination (core functionality - no heavy dependencies)
try:
    from .coordination import MultiAgentCoordinator
    __all__.append("MultiAgentCoordinator")
except ImportError:
    MultiAgentCoordinator = None

# Import environments (requires atroposlib)
try:
    from .environments import WarehouseSpatialEnvironment
    __all__.append("WarehouseSpatialEnvironment")
except ImportError:
    WarehouseSpatialEnvironment = None

# Import evaluation (may have optional dependencies)
try:
    from .evaluation import SpatialMetrics
    __all__.append("SpatialMetrics")
except ImportError:
    SpatialMetrics = None 