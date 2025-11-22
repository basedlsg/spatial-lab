# Run Research Experiment Locally - Quick Guide

## One-Time Setup (30 seconds)

```bash
# 1. Navigate to project
cd /path/to/spatial-lab

# 2. Install dependencies
pip install scipy aiohttp

# 3. Set your API key (contact user for the key)
export GROQ_API_KEY="your_groq_api_key_here"
```

## Method 1: Interactive Script (Easiest)

```bash
./quick_start.sh
```

Then choose from menu:
- `1` = Quick test (5 seconds)
- `2` = Full experiment (5-10 minutes)
- `3` = Compare LLM vs algorithmic
- `4` = Custom settings

## Method 2: Direct Commands

```bash
# Quick test
python run_experiment.py --test

# Full experiment (50 trials)
python run_experiment.py --trials 50

# Compare LLM vs algorithmic
python run_experiment.py --compare --trials 30

# Custom
python run_experiment.py --trials 100 --robots 10 --output my_results.json
```

## What You'll See

### Quick Test Output:
```
Quick Test Results:
  Message: Need assistance with item pickup at position (10.0, 10.0)...
  Interpretation: {'need': 'help_carry', 'can_help': True, ...}
  Code Generated: 145 chars
  Verification: PASS
  Success: True
  Total Time: 3847ms

Generated Code:
await robot.move_to((10.0, 10.0))
await robot.help_carry()
```

### Full Experiment Output:
```
================================================================================
EXPERIMENT RESULTS
================================================================================

Overall Performance:
  Total Trials:     50
  Successful:       42 (84.0%)

Latency (successful trials):
  Interpretation:   1847 ms
  Code Generation:  2135 ms
  Total:            3982 ms

Results by Need Type:
  help_carry          : 9/10 (90%)
  clear_path          : 8/10 (80%)
  share_location      : 10/10 (100%)
  coordinate_timing   : 8/10 (80%)
  battery_low         : 7/10 (70%)

Detailed results saved to: experiment_results.json
```

### Comparison Output:
```
================================================================================
BASELINE COMPARISON
================================================================================

Overall Results:
  Algorithmic Wins:  12 (40%)
  LLM Wins:          15 (50%)
  Ties:              3 (10%)

Average Response Time:
  Algorithmic:       0.12 ms
  LLM:               4127 ms  (34392x slower)

Key Insights:
  - LLM excels at: share_location
  - Algorithmic excels at: coordinate_timing
```

## Troubleshooting

### "No API key provided"
```bash
export GROQ_API_KEY="your_key_here"
```

### "Cannot connect to api.groq.com"
- Check internet connection
- Test: `curl -I https://api.groq.com`

### Import errors
```bash
pip install scipy aiohttp numpy pydantic
```

## Next Steps

1. Run quick test to verify everything works
2. Run full experiment (50 trials)
3. Check `experiment_results.json` for detailed data
4. Read `RESEARCH_EXPERIMENT.md` for full documentation

## API Key

The Groq API key is required to run the experiments. Contact the project owner for access.
