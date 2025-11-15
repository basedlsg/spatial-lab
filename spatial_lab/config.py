"""
Configuration Management for Spatial AI Research Lab

Centralized configuration for experiments, environments, and evaluation.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for spatial reasoning experiments"""
    
    # Experiment identification
    experiment_name: str = "warehouse_coordination"
    experiment_id: str = ""
    description: str = "Multi-agent warehouse coordination experiment"
    
    # Environment configuration
    warehouse_width: float = 50.0
    warehouse_height: float = 30.0
    num_robots: int = 5
    num_shelves: int = 20
    
    # Task configuration
    task_complexity: str = "medium"  # easy, medium, hard
    max_task_duration: int = 300
    items_per_task: int = 10
    num_evaluation_tasks: int = 20
    
    # Training configuration
    num_training_episodes: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Evaluation configuration
    evaluation_frequency: int = 10  # episodes
    num_evaluation_runs: int = 5
    baseline_comparisons: List[str] = field(default_factory=lambda: ["random", "rule_based"])
    
    # Statistical analysis
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.5
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 256
    temperature: float = 0.7
    
    # Infrastructure
    use_wandb: bool = True
    save_trajectories: bool = True
    save_frequency: int = 50
    output_dir: str = "results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "environment": {
                "warehouse_width": self.warehouse_width,
                "warehouse_height": self.warehouse_height,
                "num_robots": self.num_robots,
                "num_shelves": self.num_shelves
            },
            "tasks": {
                "complexity": self.task_complexity,
                "max_duration": self.max_task_duration,
                "items_per_task": self.items_per_task,
                "num_evaluation_tasks": self.num_evaluation_tasks
            },
            "training": {
                "num_episodes": self.num_training_episodes,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate
            },
            "evaluation": {
                "frequency": self.evaluation_frequency,
                "num_runs": self.num_evaluation_runs,
                "baselines": self.baseline_comparisons
            },
            "statistics": {
                "confidence_level": self.confidence_level,
                "significance_threshold": self.significance_threshold,
                "effect_size_threshold": self.effect_size_threshold
            },
            "model": {
                "name": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            },
            "infrastructure": {
                "use_wandb": self.use_wandb,
                "save_trajectories": self.save_trajectories,
                "save_frequency": self.save_frequency,
                "output_dir": self.output_dir
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        config = cls()
        
        # Basic info
        config.experiment_name = data.get("experiment_name", config.experiment_name)
        config.experiment_id = data.get("experiment_id", config.experiment_id)
        config.description = data.get("description", config.description)
        
        # Environment
        env_config = data.get("environment", {})
        config.warehouse_width = env_config.get("warehouse_width", config.warehouse_width)
        config.warehouse_height = env_config.get("warehouse_height", config.warehouse_height)
        config.num_robots = env_config.get("num_robots", config.num_robots)
        config.num_shelves = env_config.get("num_shelves", config.num_shelves)
        
        # Tasks
        task_config = data.get("tasks", {})
        config.task_complexity = task_config.get("complexity", config.task_complexity)
        config.max_task_duration = task_config.get("max_duration", config.max_task_duration)
        config.items_per_task = task_config.get("items_per_task", config.items_per_task)
        config.num_evaluation_tasks = task_config.get("num_evaluation_tasks", config.num_evaluation_tasks)
        
        # Training
        train_config = data.get("training", {})
        config.num_training_episodes = train_config.get("num_episodes", config.num_training_episodes)
        config.batch_size = train_config.get("batch_size", config.batch_size)
        config.learning_rate = train_config.get("learning_rate", config.learning_rate)
        
        # Evaluation
        eval_config = data.get("evaluation", {})
        config.evaluation_frequency = eval_config.get("frequency", config.evaluation_frequency)
        config.num_evaluation_runs = eval_config.get("num_runs", config.num_evaluation_runs)
        config.baseline_comparisons = eval_config.get("baselines", config.baseline_comparisons)
        
        # Statistics
        stats_config = data.get("statistics", {})
        config.confidence_level = stats_config.get("confidence_level", config.confidence_level)
        config.significance_threshold = stats_config.get("significance_threshold", config.significance_threshold)
        config.effect_size_threshold = stats_config.get("effect_size_threshold", config.effect_size_threshold)
        
        # Model
        model_config = data.get("model", {})
        config.model_name = model_config.get("name", config.model_name)
        config.max_tokens = model_config.get("max_tokens", config.max_tokens)
        config.temperature = model_config.get("temperature", config.temperature)
        
        # Infrastructure
        infra_config = data.get("infrastructure", {})
        config.use_wandb = infra_config.get("use_wandb", config.use_wandb)
        config.save_trajectories = infra_config.get("save_trajectories", config.save_trajectories)
        config.save_frequency = infra_config.get("save_frequency", config.save_frequency)
        config.output_dir = infra_config.get("output_dir", config.output_dir)
        
        return config


class ConfigManager:
    """Manages experiment configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "basic_warehouse": self._create_basic_warehouse_config(),
            "complex_warehouse": self._create_complex_warehouse_config(),
            "evaluation_benchmark": self._create_evaluation_config()
        }
    
    def _create_basic_warehouse_config(self) -> ExperimentConfig:
        """Create basic warehouse experiment configuration"""
        return ExperimentConfig(
            experiment_name="basic_warehouse_coordination",
            description="Basic warehouse coordination with 3 robots and simple tasks",
            warehouse_width=30.0,
            warehouse_height=20.0,
            num_robots=3,
            num_shelves=10,
            task_complexity="easy",
            max_task_duration=200,
            items_per_task=5,
            num_training_episodes=50,
            num_evaluation_tasks=10
        )
    
    def _create_complex_warehouse_config(self) -> ExperimentConfig:
        """Create complex warehouse experiment configuration"""
        return ExperimentConfig(
            experiment_name="complex_warehouse_coordination",
            description="Complex warehouse coordination with 8 robots and challenging tasks",
            warehouse_width=80.0,
            warehouse_height=50.0,
            num_robots=8,
            num_shelves=40,
            task_complexity="hard",
            max_task_duration=500,
            items_per_task=20,
            num_training_episodes=200,
            num_evaluation_tasks=30
        )
    
    def _create_evaluation_config(self) -> ExperimentConfig:
        """Create evaluation benchmark configuration"""
        return ExperimentConfig(
            experiment_name="evaluation_benchmark",
            description="Standardized evaluation benchmark for spatial reasoning",
            warehouse_width=50.0,
            warehouse_height=30.0,
            num_robots=5,
            num_shelves=20,
            task_complexity="medium",
            max_task_duration=300,
            items_per_task=10,
            num_training_episodes=0,  # Evaluation only
            num_evaluation_tasks=50,
            num_evaluation_runs=10
        )
    
    def get_config(self, config_name: str) -> Optional[ExperimentConfig]:
        """Get configuration by name"""
        
        # Check default configurations
        if config_name in self.default_configs:
            return self.default_configs[config_name]
        
        # Check saved configurations
        config_file = self.config_dir / f"{config_name}.json"
        if config_file.exists():
            return self.load_config(config_file)
        
        logger.warning(f"Configuration '{config_name}' not found")
        return None
    
    def save_config(self, config: ExperimentConfig, filename: str):
        """Save configuration to file"""
        config_file = self.config_dir / f"{filename}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")
    
    def load_config(self, config_file: Path) -> ExperimentConfig:
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        return ExperimentConfig.from_dict(data)
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = list(self.default_configs.keys())
        
        # Add saved configurations
        for config_file in self.config_dir.glob("*.json"):
            configs.append(config_file.stem)
        
        return sorted(configs)
    
    def create_experiment_config(
        self,
        base_config: str = "basic_warehouse",
        **overrides
    ) -> ExperimentConfig:
        """Create experiment configuration with overrides"""
        
        base = self.get_config(base_config)
        if not base:
            base = self.default_configs["basic_warehouse"]
        
        # Apply overrides
        config_dict = base.to_dict()
        
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "environment.num_robots"
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return ExperimentConfig.from_dict(config_dict)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(config_name: str = "basic_warehouse") -> ExperimentConfig:
    """Get experiment configuration"""
    config = config_manager.get_config(config_name)
    if not config:
        logger.warning(f"Using default configuration instead of '{config_name}'")
        config = config_manager.default_configs["basic_warehouse"]
    return config


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create custom configuration with overrides"""
    return config_manager.create_experiment_config(**kwargs)


def setup_logging(config: ExperimentConfig):
    """Setup logging for experiment"""
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"{config.experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging setup for experiment: {config.experiment_name}")
    logger.info(f"Log file: {log_file}")


def setup_wandb(config: ExperimentConfig):
    """Setup Weights & Biases tracking"""
    
    if not config.use_wandb:
        return
    
    try:
        import wandb
        
        wandb.init(
            project="spatial-ai-research-lab",
            name=config.experiment_name,
            config=config.to_dict(),
            tags=["spatial-reasoning", "multi-agent", "warehouse"]
        )
        
        logger.info("Initialized Weights & Biases tracking")
        
    except ImportError:
        logger.warning("Weights & Biases not available, skipping setup")
    except Exception as e:
        logger.warning(f"Failed to setup Weights & Biases: {e}")


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate experiment configuration"""
    
    warnings = []
    
    # Check environment parameters
    if config.num_robots > 10:
        warnings.append("Large number of robots may cause performance issues")
    
    if config.warehouse_width * config.warehouse_height < config.num_shelves * 4:
        warnings.append("Warehouse may be too small for number of shelves")
    
    # Check task parameters
    if config.items_per_task > config.num_shelves * 5:
        warnings.append("Items per task may exceed available inventory")
    
    # Check training parameters
    if config.num_training_episodes > 1000:
        warnings.append("Large number of training episodes may take very long")
    
    # Check evaluation parameters
    if config.num_evaluation_runs < 5:
        warnings.append("Few evaluation runs may not provide reliable statistics")
    
    # Check statistical parameters
    if config.confidence_level < 0.9:
        warnings.append("Low confidence level may not provide reliable results")
    
    return warnings 