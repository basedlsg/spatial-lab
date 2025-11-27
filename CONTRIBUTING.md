# Contributing to Spatial Lab

Thank you for your interest in contributing to Spatial Lab! This document provides guidelines for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- API keys for LLM providers (Groq, Google Gemini)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/basedlsg/spatial-lab.git
   cd spatial-lab
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install package in editable mode with dev dependencies
   pip install -e ".[dev,all]"

   # Or install from requirements.txt
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Verify installation:**
   ```bash
   pytest tests/ -v
   python scripts/test_llm_apis.py
   ```

## Code Style

We follow PEP 8 with some modifications. Use the following tools to ensure consistency:

### Formatting

```bash
# Format code with Black
black spatial_lab/ tests/ scripts/

# Sort imports with isort
isort spatial_lab/ tests/ scripts/

# Lint with Ruff
ruff check spatial_lab/ tests/ scripts/
```

### Style Guidelines

- **Line length:** 100 characters maximum
- **Quotes:** Double quotes for strings
- **Imports:** Group in order: stdlib, third-party, local
- **Type hints:** Required for all public functions
- **Docstrings:** Required for all public modules, classes, and functions

### Example Function

```python
from typing import Dict, List, Optional

def process_robot_decision(
    robot_id: str,
    observation: Dict[str, Any],
    available_actions: List[str],
    timeout_ms: int = 5000,
) -> Optional[Dict[str, Any]]:
    """
    Process a robot's decision based on current observation.

    Args:
        robot_id: Unique identifier for the robot.
        observation: Current state observation dictionary.
        available_actions: List of valid actions the robot can take.
        timeout_ms: Maximum time to wait for decision in milliseconds.

    Returns:
        Decision dictionary with 'action' and 'parameters' keys,
        or None if decision could not be made.

    Raises:
        ValueError: If robot_id is empty or observation is invalid.

    Example:
        >>> decision = process_robot_decision(
        ...     robot_id="robot_001",
        ...     observation={"position": [5.0, 5.0, 0.0]},
        ...     available_actions=["move_to", "wait"]
        ... )
        >>> print(decision["action"])
        'move_to'
    """
    if not robot_id:
        raise ValueError("robot_id cannot be empty")

    # Implementation here
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spatial_lab --cov-report=html

# Run specific test file
pytest tests/test_spatial_lab_basic.py -v

# Run tests matching a pattern
pytest tests/ -k "test_robot"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for >80% coverage on new code

```python
import pytest
from spatial_lab.coordination import RobotFleetSimulator

class TestRobotFleet:
    @pytest.fixture
    def fleet(self):
        """Create a test robot fleet."""
        return RobotFleetSimulator(num_robots=3)

    def test_fleet_initialization(self, fleet):
        """Test that fleet initializes with correct number of robots."""
        assert len(fleet.robots) == 3

    @pytest.mark.asyncio
    async def test_robot_movement(self, fleet):
        """Test that robots can move to target positions."""
        result = await fleet.move_robot("robot_0", [10.0, 10.0])
        assert result["success"] is True
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

3. **Run quality checks:**
   ```bash
   # Format code
   black spatial_lab/ tests/ scripts/
   isort spatial_lab/ tests/ scripts/

   # Run linter
   ruff check spatial_lab/ tests/ scripts/

   # Run tests
   pytest tests/ -v

   # Type check (optional but recommended)
   mypy spatial_lab/
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional longer description>

<optional footer>
```

**Types:**
- `Add`: New feature
- `Fix`: Bug fix
- `Update`: Enhancement to existing feature
- `Refactor`: Code refactoring (no functional change)
- `Docs`: Documentation changes
- `Test`: Adding or updating tests
- `Chore`: Maintenance tasks

**Examples:**
```
Add: LLM confidence calibration experiment

Implements a 144-trial factorial experiment measuring LLM
calibration across complexity levels and prompt conditions.

Closes #42
```

```
Fix: Rate limiting in Groq API client

Added exponential backoff when rate limits are hit.
```

### PR Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No sensitive data (API keys, credentials) in code
- [ ] Commit messages are clear

## Project Structure

```
spatial-lab/
├── spatial_lab/              # Main package
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   ├── experiment_runner.py # Experiment orchestration
│   ├── llm/                 # LLM provider clients
│   │   ├── groq_client.py   # Groq/Llama API
│   │   ├── gemini_client.py # Google Gemini API
│   │   └── llm_coordinator.py
│   ├── coordination/        # Multi-agent coordination
│   │   ├── robot_fleet.py   # Robot fleet simulation
│   │   └── path_planning.py # Spatial path planning
│   ├── environments/        # Simulation environments
│   ├── evaluation/          # Metrics and analysis
│   └── research/            # Research validation
├── scripts/                 # Executable experiment scripts
│   ├── calibration_experiment.py
│   ├── run_experiment.py
│   └── test_llm_apis.py
├── tests/                   # Test suite
├── docs/                    # Documentation
├── experiment_results/      # Output directory (gitignored)
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── README.md               # Project overview
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `spatial_lab.llm` | LLM API clients (Groq, Gemini, Llama) |
| `spatial_lab.coordination` | Robot fleet and path planning |
| `spatial_lab.environments` | Warehouse simulation |
| `spatial_lab.evaluation` | Metrics and statistical analysis |
| `spatial_lab.config` | Experiment configuration |

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Use discussions for general questions

Thank you for contributing!
