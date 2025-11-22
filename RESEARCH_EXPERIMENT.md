# Natural Language + Code Generation Robot Coordination Research

## Research Question

**Can robots effectively communicate needs in natural language and have those needs fulfilled through LLM-generated code?**

This experiment explores the intersection of:
- Natural language communication between robots
- LLM-generated code for robot actions
- Safety verification before execution
- Comparison with traditional algorithmic approaches

## What This Measures

1. **Communication Success Rate**: How often robots correctly interpret each other's needs
2. **Code Generation Accuracy**: Success rate of LLM-generated code passing verification
3. **Latency Analysis**: Breakdown of interpretation vs code generation time
4. **Algorithmic Comparison**: When LLM beats rule-based approaches and vice versa

## Setup Instructions

### 1. Prerequisites

```bash
# Python 3.11+ required
python --version

# Install dependencies
pip install scipy aiohttp numpy pydantic
```

### 2. Set Your API Key

You have two options:

**Option A: Environment Variable (Recommended)**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Option B: Command Line Flag**
```bash
python run_experiment.py --api-key "your_groq_api_key_here" --test
```

### 3. Navigate to Project Directory

```bash
cd /path/to/spatial-lab
```

## Running the Experiments

### Quick Test (1 trial)

Verifies everything works - takes ~5 seconds:

```bash
python run_experiment.py --test
```

**Expected Output:**
```
Quick Test Results:
  Message: Need assistance with item pickup at position (10.0, 10.0)...
  Interpretation: {'need': 'help_carry', 'suggested_action': 'move to sender and assist', ...}
  Code Generated: 145 chars
  Verification: PASS
  Success: True
  Total Time: 3847ms

Generated Code:
await robot.move_to((10.0, 10.0))
await robot.help_carry()
```

### Full Experiment (50 trials)

Runs 50 trials across all 5 need types (~5-10 minutes):

```bash
python run_experiment.py --trials 50
```

**Expected Output:**
```
================================================================================
EXPERIMENT RESULTS: Natural Language + Code Generation Robot Coordination
================================================================================

Overall Performance:
  Total Trials:     50
  Successful:       42 (84.0%)

Failure Breakdown:
  Interpretation:   3
  Code Generation:  2
  Verification:     3
  Execution:        0

Latency (successful trials):
  Interpretation:   1847.3 ms
  Code Generation:  2134.7 ms
  Total:            3982.0 ms

Results by Need Type:
  help_carry          : 9/10 (90%) - 4125ms avg
  clear_path          : 8/10 (80%) - 3654ms avg
  share_location      : 10/10 (100%) - 3201ms avg
  coordinate_timing   : 8/10 (80%) - 4532ms avg
  battery_low         : 7/10 (70%) - 4412ms avg

Detailed results saved to: experiment_results.json
```

### Baseline Comparison (30 trials)

Compares LLM vs pure algorithmic approach:

```bash
python run_experiment.py --compare --trials 30
```

**Expected Output:**
```
================================================================================
BASELINE COMPARISON: Algorithmic vs LLM Robot Coordination
================================================================================

Overall Results:
  Total Trials:      30

  Algorithmic Wins:  12 (40.0%)
  LLM Wins:          15 (50.0%)
  Ties:              3 (10.0%)

Success Rates:
  Algorithmic:       93.3%
  LLM:               80.0%

Average Response Time:
  Algorithmic:       0.12 ms
  LLM:               4127 ms  (34392x slower)

Results by Need Type:
  help_carry          : Algo 2 | LLM 4 | Tie 0
  clear_path          : Algo 3 | LLM 2 | Tie 1
  share_location      : Algo 1 | LLM 5 | Tie 0
  coordinate_timing   : Algo 4 | LLM 2 | Tie 0
  battery_low         : Algo 2 | LLM 2 | Tie 2

Key Insights:
  - LLM approach wins on contextual understanding
  - LLM excels at: share_location
  - Algorithmic excels at: coordinate_timing
```

### Custom Configuration

```bash
# Run 100 trials with 10 robots
python run_experiment.py --trials 100 --robots 10

# Save to custom file
python run_experiment.py --trials 50 --output my_results.json

# Use different model
python run_experiment.py --model llama-3.1-8b-instant --trials 30

# Verbose logging
python run_experiment.py --test --verbose
```

## Understanding the Results

### Need Types Tested

1. **HELP_CARRY**: Robot requests assistance with heavy item
   - Tests: collaboration, spatial coordination

2. **CLEAR_PATH**: Robot asks another to move out of the way
   - Tests: spatial awareness, conflict resolution

3. **SHARE_LOCATION**: Robot requests position information
   - Tests: information sharing, simple responses

4. **COORDINATE_TIMING**: Robots sync arrival times
   - Tests: temporal reasoning, planning

5. **BATTERY_LOW**: Robot needs task handoff
   - Tests: urgency recognition, task transfer

### Output Files

**experiment_results.json** contains:
```json
{
  "summary": {
    "total_trials": 50,
    "successful_trials": 42,
    "success_rate": 0.84,
    "avg_interpretation_latency_ms": 1847.3,
    "avg_code_gen_latency_ms": 2134.7
  },
  "by_type": {
    "help_carry": {"total": 10, "success": 9, "avg_latency_ms": 4125},
    ...
  },
  "trials": [
    {
      "trial_id": 0,
      "sender_id": "robot_001",
      "receiver_id": "robot_003",
      "need_type": "help_carry",
      "message": "I have a heavy item at (10.5, 23.1)...",
      "success": true,
      "interpretation_latency_ms": 1923.4,
      "code_gen_latency_ms": 2156.8,
      "generated_code": "await robot.move_to((10.5, 23.1))\n..."
    },
    ...
  ]
}
```

## Analyzing the Data

### Success Rate Analysis

Look for patterns in failure types:
- High interpretation failures → LLM struggles with specific need types
- High verification failures → Generated code is unsafe
- High code gen failures → LLM can't produce valid Python

### Latency Analysis

- Interpretation: ~2 seconds (message understanding)
- Code generation: ~2 seconds (producing executable code)
- Total: ~4 seconds per coordination

### Baseline Comparison Insights

The comparison reveals:
- **Algorithmic wins**: Speed, simple spatial tasks
- **LLM wins**: Contextual understanding, complex coordination
- **Use case**: Hybrid approach may be optimal

## Troubleshooting

### "No API key provided"
```bash
# Make sure to export the key:
export GROQ_API_KEY="your_key_here"

# Or use the flag:
python run_experiment.py --api-key "your_key_here" --test
```

### "Cannot connect to host api.groq.com"
- Check internet connection
- Verify firewall isn't blocking api.groq.com
- Try: `curl -I https://api.groq.com`

### Import errors
```bash
# Install missing dependencies:
pip install scipy aiohttp numpy pydantic
```

### Slow performance
- Use faster model: `--model llama-3.1-8b-instant`
- Reduce trials: `--trials 10`
- The 70B model is more accurate but slower

## Advanced Usage

### Programmatic Access

```python
import asyncio
from spatial_lab.research import NLCodeExperiment, RobotState, NeedType

async def main():
    # Initialize experiment
    api_key = "your_key_here"
    experiment = NLCodeExperiment(api_key)

    # Create robots
    sender = RobotState("robot_1", (10, 10), battery=0.5, carrying="box")
    receiver = RobotState("robot_2", (15, 10), battery=0.9)

    # Run single trial
    trial = await experiment.run_trial(
        trial_id=0,
        sender=sender,
        receiver=receiver,
        need_type=NeedType.HELP_CARRY
    )

    print(f"Success: {trial.success}")
    print(f"Generated code:\n{trial.generated_code}")

asyncio.run(main())
```

### Export for Analysis

```python
import json
import pandas as pd

# Load results
with open('experiment_results.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['trials'])

# Analyze
print(df.groupby('need_type')['success'].mean())
print(df[df['success']]['total_time_ms'].describe())
```

## Research Questions to Explore

1. **Which need types benefit most from LLM understanding?**
   - Hypothesis: Complex coordination tasks (battery_low, coordinate_timing)

2. **What's the latency/accuracy tradeoff?**
   - Hypothesis: Smaller models faster but less accurate

3. **Can we hybrid approaches beat both?**
   - Idea: Use algorithmic for simple, LLM for complex

4. **Does generated code quality improve with examples?**
   - Experiment: Add few-shot examples to prompts

## Citation

If you use this experiment in research:

```
Natural Language + Code Generation Robot Coordination Experiment
Spatial Lab v1.0
November 2025
```

## Support

For issues or questions, check the main README or open an issue.
