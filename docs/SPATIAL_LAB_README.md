# Spatial AI Research Lab

A focused research platform for evaluating multi-agent spatial reasoning capabilities using the Nous Atropos infrastructure. This lab conducts rigorous scientific experiments on warehouse robot coordination, emphasizing measurable improvements in spatial cognition and collaborative task execution.

## Overview

The Spatial AI Research Lab addresses a critical gap in spatial reasoning evaluation for multi-agent systems. Rather than making broad claims about "breakthrough" capabilities, we focus on **measurable, statistically significant improvements** in specific spatial coordination tasks.

### Core Research Questions

1. **Path Efficiency**: Can LLM-based agents achieve statistically significant improvements in path planning efficiency compared to rule-based baselines?
2. **Collision Avoidance**: How do multi-agent coordination strategies affect collision rates in constrained environments?
3. **Task Completion**: What factors most influence successful task completion in collaborative spatial reasoning scenarios?

### Key Features

- **Rigorous Evaluation**: Statistical significance testing, confidence intervals, and effect size calculations
- **Baseline Comparisons**: Systematic comparison against random and rule-based agents
- **Reproducible Experiments**: Standardized evaluation protocols and documented methodologies
- **Integration with Atropos**: Leverages existing Nous Research infrastructure for LLM evaluation

## Architecture

```
src/spatial_lab/
├── environments/           # Warehouse simulation environments
│   ├── warehouse_environment.py    # Main Atropos-integrated environment
│   ├── warehouse_layout.py         # Layout generation and management
│   └── warehouse_tasks.py          # Task generation and coordination
├── coordination/          # Multi-agent coordination systems
│   ├── robot_fleet.py             # Robot fleet simulation
│   ├── path_planning.py           # Spatial path planning
│   └── communication.py           # Inter-robot communication
├── evaluation/           # Metrics and statistical analysis
│   ├── spatial_metrics.py        # Comprehensive spatial metrics
│   ├── statistical_analysis.py   # Statistical significance testing
│   └── performance_analyzer.py   # Performance analysis tools
├── config.py            # Configuration management
└── experiment_runner.py  # Main experiment orchestration
```

## Quick Start

### 1. Setup

```bash
# Clone and setup the lab
python setup_spatial_lab.py

# Install dependencies
pip install -r requirements_spatial_lab.txt

# Configure environment variables
cp .env.example .env
# Edit .env to add your OpenAI API key
```

### 2. Run a Basic Experiment

```bash
# Run a simple 3-robot warehouse coordination experiment
python -m src.spatial_lab.experiment_runner --config basic_warehouse

# Run with custom parameters
python -m src.spatial_lab.experiment_runner \
    --config basic_warehouse \
    --robots 5 \
    --episodes 50 \
    --eval-tasks 20
```

### 3. View Results

Results are saved in the `results/` directory with:
- Detailed metrics and statistical analysis
- Human-readable experiment summaries
- Raw trajectory data for further analysis

## Experimental Design

### Environment Configuration

**Warehouse Setup:**
- Configurable dimensions (default: 50m × 30m)
- Realistic shelf layouts with aisles
- Multiple task types: picking, batch collection, collaborative transport
- Obstacle placement and navigation constraints

**Robot Fleet:**
- 3-8 robots with realistic movement constraints
- Battery limitations and load capacity
- Communication range limitations
- Collision detection and avoidance

### Evaluation Metrics

Following scientific evaluation standards, we measure:

#### Primary Metrics
- **Path Efficiency**: `optimal_path_length / actual_path_length`
- **Task Completion Rate**: Percentage of successfully completed tasks
- **Collision Rate**: Collisions per episode
- **Coordination Efficiency**: Successful coordination events per task

#### Statistical Analysis
- **Confidence Intervals**: 95% CI for all primary metrics
- **Effect Sizes**: Cohen's d for practical significance
- **Significance Testing**: t-tests against baseline performance
- **Multiple Comparison Correction**: Bonferroni adjustment when appropriate

### Baseline Comparisons

All experiments compare against established baselines:

1. **Random Agent**: Random movement and action selection
2. **Rule-Based Agent**: Hand-coded heuristics for navigation and coordination
3. **Human Performance**: Where applicable, human operator benchmarks

## Configuration

### Experiment Configurations

The lab provides several pre-configured experiments:

```python
# Basic warehouse (3 robots, simple tasks)
config = get_config("basic_warehouse")

# Complex warehouse (8 robots, challenging tasks)  
config = get_config("complex_warehouse")

# Evaluation benchmark (standardized evaluation)
config = get_config("evaluation_benchmark")
```

### Custom Configuration

```python
from src.spatial_lab.config import create_custom_config

config = create_custom_config(
    base_config="basic_warehouse",
    **{
        "environment.num_robots": 6,
        "tasks.complexity": "hard",
        "training.num_episodes": 100,
        "evaluation.num_evaluation_tasks": 30
    }
)
```

## Results and Analysis

### Statistical Reporting

All results include:

```json
{
  "path_efficiency": {
    "value": 0.73,
    "ci": [0.68, 0.78],
    "effect_size": 1.2,
    "p_value": 0.003
  },
  "vs_random_agent": {
    "improvement": "+140%",
    "significance": "p < 0.001"
  },
  "vs_rule_based": {
    "improvement": "+23%", 
    "significance": "p = 0.012"
  }
}
```

### Performance Visualization

The lab generates comprehensive visualizations:
- Learning curves with confidence bands
- Comparative performance charts
- Spatial heatmaps of robot movement
- Task completion analysis

## Integration with Nous Atropos

The lab seamlessly integrates with the Atropos framework:

```python
# Environment extends Atropos BaseEnv
class WarehouseSpatialEnvironment(BaseEnv):
    name = "warehouse_spatial_coordination"
    
    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List[Item]]:
        # Execute coordination task and collect LLM trajectories
        # Calculate spatial reasoning metrics
        # Return scored trajectories for Atropos processing
```

### LLM Integration

Robots use structured prompts for spatial reasoning:

```python
prompt = f"""You are Robot {robot_id} in a warehouse coordination task.

CURRENT STATE:
- Position: {robot_state['position']}
- Carrying: {robot_state['carrying_item'] or 'nothing'}

SPATIAL ENVIRONMENT:
- Nearby shelves: {warehouse_context['nearby_shelves']}
- Other robots nearby: {warehouse_context['nearby_robots']}
- Area congestion: {warehouse_context['current_congestion']}

Choose the best action considering spatial efficiency and coordination.
Respond with JSON: {{"action": "action_name", "parameters": {{}}, "reasoning": "brief explanation"}}"""
```

## Scientific Standards Compliance

This lab follows rigorous scientific evaluation standards:

### Experimental Design
- ✅ Clear hypotheses and success criteria
- ✅ Appropriate control groups and baselines
- ✅ Statistical power analysis for sample sizes
- ✅ Reproducible experimental protocols

### Statistical Analysis
- ✅ Effect size calculations (Cohen's d)
- ✅ Confidence intervals (95% CI)
- ✅ Multiple comparison corrections
- ✅ Significance testing with appropriate thresholds

### Reporting Standards
- ✅ Honest reporting of null results
- ✅ Clear limitations and scope
- ✅ Appropriate language (no hyperbolic claims)
- ✅ Detailed methodology documentation

## Use Cases

### 1. Research Applications

**Academic Research:**
- Spatial reasoning evaluation for multi-agent systems
- Coordination strategy comparison studies
- Benchmark development for warehouse robotics

**Industry Applications:**
- Warehouse automation system evaluation
- Robot fleet coordination optimization
- Spatial AI capability assessment

### 2. Practical Deployment

**Proof of Concept:**
- Demonstrate measurable improvements in specific metrics
- Validate coordination strategies before real-world deployment
- Risk assessment for autonomous robot fleets

**Performance Optimization:**
- Identify bottlenecks in multi-agent coordination
- Optimize task allocation strategies
- Improve spatial reasoning prompts and approaches

## Limitations and Scope

### Current Limitations

1. **Simulation Environment**: Results limited to simulated warehouse settings
2. **Task Specificity**: Evaluation focused on warehouse coordination tasks
3. **LLM Dependency**: Performance tied to underlying language model capabilities
4. **Computational Requirements**: Extensive evaluation requires significant compute resources

### Future Extensions

1. **Environment Diversity**: Additional spatial reasoning domains
2. **Real-World Transfer**: Sim-to-real validation studies
3. **Scalability Analysis**: Performance with larger robot fleets
4. **Human-Robot Collaboration**: Mixed human-robot team coordination

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_spatial_lab.txt

# Run tests
pytest src/spatial_lab/tests/

# Code formatting
black src/spatial_lab/
flake8 src/spatial_lab/
```

### Adding New Environments

1. Extend `BaseEnv` from Atropos
2. Implement required methods: `get_next_item()`, `collect_trajectories()`
3. Add comprehensive metrics calculation
4. Include statistical analysis capabilities

### Adding New Metrics

1. Extend `SpatialMetrics` dataclass
2. Implement calculation in `SpatialMetricsCalculator`
3. Add statistical significance testing
4. Update baseline comparisons

## License and Citation

This research lab is part of the broader Nous Research ecosystem. When using this lab for research, please cite:

```bibtex
@software{spatial_ai_research_lab,
  title={Spatial AI Research Lab: Rigorous Evaluation of Multi-Agent Spatial Reasoning},
  author={Nous Research},
  year={2024},
  url={https://github.com/NousResearch/spatial-ai-lab}
}
```

## Contact and Support

For questions, issues, or collaboration opportunities:

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for research questions
- **Email**: [research@nousresearch.com](mailto:research@nousresearch.com)

---

**Note**: This lab prioritizes **scientific rigor over marketing claims**. All results are reported with appropriate statistical context, limitations are clearly stated, and claims are backed by empirical evidence with proper significance testing. 