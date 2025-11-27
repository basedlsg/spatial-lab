# Spatial Lab

> Multi-agent warehouse robotics coordination with LLM-driven spatial reasoning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Spatial Lab is a research framework for multi-agent robotics that combines traditional robotic planning with LLM-driven spatial reasoning. The system coordinates warehouse robots for tasks like navigation, object manipulation, and collaborative task execution.

### Key Features

- **LLM Integration**: Groq (Llama), Google Gemini, and OpenAI for spatial reasoning
- **Multi-Robot Coordination**: Fleet management with path planning
- **Confidence Calibration**: Research tools for measuring LLM calibration
- **Experiment Framework**: Reproducible experiments with statistical analysis
- **Performance Monitoring**: Metrics collection and analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/basedlsg/spatial-lab.git
cd spatial-lab

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[all]"

# Or install from requirements.txt
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` with your API keys:

```bash
# LLM API Keys (at least one required)
GROQ_API_KEY=gsk_...            # Groq/Llama (recommended)
GOOGLE_API_KEY=AIza...          # Google Gemini (fallback)

# Optional
WANDB_API_KEY=your_key          # Weights & Biases tracking
DEFAULT_LLM_MODEL=llama-3.3-70b-versatile
```

### Run Your First Experiment

```bash
# Test API connectivity
PYTHONPATH=. python scripts/test_llm_apis.py

# Run calibration experiment (~5 minutes, 144 trials)
PYTHONPATH=. python scripts/calibration_experiment.py

# Results saved to experiment_results/
```

## Usage

### LLM Spatial Reasoning

```python
import asyncio
from spatial_lab.llm import GroqAPIConfig, GroqAPIClient

async def main():
    config = GroqAPIConfig(
        api_key="your_groq_key",
        model="llama-3.3-70b-versatile"
    )

    async with GroqAPIClient(config) as client:
        response = await client.robot_coordination_decision(
            robot_id="robot_001",
            observation={
                "position": [5.0, 5.0, 0.0],
                "battery_level": 0.95,
                "status": "idle"
            },
            task_description="Navigate to position (10, 8) and pick up item_box_A",
            available_actions=["move_to", "pick_item", "wait"],
            warehouse_layout={
                "dimensions": [20, 20],
                "obstacles": [{"position": [7, 6], "radius": 1.5}]
            }
        )
        print(response)

asyncio.run(main())
```

### Robot Fleet Management

```python
from spatial_lab.coordination import RobotFleetSimulator

# Create fleet with 5 robots
fleet = RobotFleetSimulator(num_robots=5)

# Get robot status
status = fleet.get_robot_status("robot_0")
print(f"Position: {status['position']}")
print(f"Status: {status['status']}")
```

### Path Planning

```python
from spatial_lab.coordination import SpatialPathPlanner

planner = SpatialPathPlanner()

# Plan path avoiding obstacles
path = planner.plan_path(
    start=(0, 0),
    goal=(10, 10),
    obstacles=[{"position": (5, 5), "radius": 2.0}]
)
```

### Configuration Management

```python
from spatial_lab.config import ExperimentConfig, get_preset_config

# Use preset configuration
config = get_preset_config("basic_warehouse")

# Or create custom config
config = ExperimentConfig(
    experiment_name="my_experiment",
    num_robots=5,
    warehouse_width=50.0,
    warehouse_height=30.0,
    task_complexity="medium"
)
```

## Project Structure

```
spatial-lab/
├── spatial_lab/              # Main package
│   ├── llm/                  # LLM provider clients
│   │   ├── groq_client.py    # Groq/Llama API
│   │   ├── gemini_client.py  # Google Gemini API
│   │   └── llm_coordinator.py # Multi-provider coordinator
│   ├── coordination/         # Robot coordination
│   │   ├── robot_fleet.py    # Fleet simulation
│   │   ├── path_planning.py  # Path algorithms
│   │   └── communication.py  # Robot messaging
│   ├── environments/         # Simulation environments
│   ├── evaluation/           # Metrics & analysis
│   ├── config.py             # Configuration system
│   └── experiment_runner.py  # Experiment orchestration
├── scripts/                  # Executable experiments
│   ├── calibration_experiment.py
│   ├── run_experiment.py
│   └── test_llm_apis.py
├── tests/                    # Test suite
├── docs/                     # Documentation
├── experiment_results/       # Output directory
├── pyproject.toml           # Package configuration
└── requirements.txt         # Dependencies
```

## Experiments

### Calibration Experiment

Measures LLM confidence calibration in spatial reasoning tasks:

```bash
PYTHONPATH=. python scripts/calibration_experiment.py
```

**Design:** 4 complexity levels × 3 prompt conditions × 3 distances × 4 reps = 144 trials

**Key Findings (v0.1.0):**
- LLMs show significant overconfidence (ECE=0.209)
- Simple tasks: good calibration (50% accuracy, 47% confidence)
- Complex tasks: poor calibration (8% accuracy, 32% confidence)
- Uncertainty-aware prompts reduce calibration error by 32%

See `experiment_results/SCIENTIFIC_REPORT.md` for full analysis.

### Custom Experiments

```bash
# Run with specific parameters
PYTHONPATH=. python scripts/run_experiment.py --trials 50 --robots 5

# Run Groq-specific experiments
PYTHONPATH=. python scripts/run_groq_experiment.py
```

## Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black spatial_lab/ tests/ scripts/
isort spatial_lab/ tests/ scripts/

# Lint
ruff check spatial_lab/
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=spatial_lab --cov-report=html

# Specific test
pytest tests/test_spatial_lab_basic.py -v
```

## Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)** - Production roadmap
- **[scripts/README.md](scripts/README.md)** - Experiment scripts guide

## Known Limitations

This is a research prototype. See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for details.

- Path planning uses simplified algorithms
- Multi-agent coordinator is partially implemented
- Test coverage needs improvement
- Some API inconsistencies exist

## Research Output

### Calibration Study Results

| Metric | Value |
|--------|-------|
| Expected Calibration Error | 0.209 |
| Brier Score | 0.202 |
| Overconfidence Rate | 20.1% |
| Mean Accuracy | 17.4% |

**Statistical Significance:**
- Calibration error > 0: p < 0.0001, Cohen's d = 1.04
- Complexity effect: F(3,140) = 15.73, p < 0.0001, η² = 0.252

## Citation

```bibtex
@software{spatial_lab_2025,
  title = {Spatial Lab: Multi-Agent Warehouse Robotics with LLM Coordination},
  author = {Spatial Lab Contributors},
  year = {2025},
  url = {https://github.com/basedlsg/spatial-lab}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Version:** 0.1.0 | **Status:** Research Prototype | **Python:** 3.11+
