# Experiment Scripts

This directory contains executable scripts for running experiments with Spatial Lab.

## Quick Start

```bash
# From the repository root:
cd /path/to/spatial-lab

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run calibration experiment (~5 minutes)
PYTHONPATH=. python scripts/calibration_experiment.py

# Run basic experiment
PYTHONPATH=. python scripts/run_experiment.py --trials 10
```

## Available Scripts

### `calibration_experiment.py`
**Purpose:** LLM Confidence Calibration Study
**Runtime:** ~5 minutes (144 trials)
**API Required:** Groq (GROQ_API_KEY)

Investigates whether LLM-reported confidence scores are well-calibrated predictors of spatial reasoning accuracy.

```bash
PYTHONPATH=. python scripts/calibration_experiment.py
```

**Output:** `experiment_results/calibration_*_{results,metrics,analysis}.json`

### `run_experiment.py`
**Purpose:** Main experiment runner with CLI interface
**Runtime:** Variable (depends on configuration)
**API Required:** Groq or Gemini

```bash
# Quick test (10 trials)
PYTHONPATH=. python scripts/run_experiment.py --trials 10

# Full experiment
PYTHONPATH=. python scripts/run_experiment.py --trials 50 --robots 5
```

### `run_groq_experiment.py`
**Purpose:** Groq/Llama-specific spatial reasoning experiments
**Runtime:** ~2-3 minutes (30 trials)
**API Required:** Groq (GROQ_API_KEY)

```bash
PYTHONPATH=. python scripts/run_groq_experiment.py
```

### `run_real_experiment.py`
**Purpose:** Full integration experiments with multiple LLM providers
**Runtime:** ~5-10 minutes
**API Required:** Groq and/or Gemini

```bash
PYTHONPATH=. python scripts/run_real_experiment.py
```

### `test_llm_apis.py`
**Purpose:** Validate API connectivity before running experiments
**Runtime:** ~30 seconds
**API Required:** Groq and Gemini

```bash
PYTHONPATH=. python scripts/test_llm_apis.py
```

## Environment Variables

All scripts require API keys set in `.env`:

```bash
GROQ_API_KEY=gsk_...          # Required for most experiments
GOOGLE_API_KEY=AIza...        # Required for Gemini fallback
DEFAULT_LLM_MODEL=llama-3.3-70b-versatile
```

## Output

All experiment results are saved to `experiment_results/` with timestamped filenames:

```
experiment_results/
├── calibration_YYYYMMDD_HHMMSS_results.json   # Raw trial data
├── calibration_YYYYMMDD_HHMMSS_metrics.json   # Computed metrics
├── calibration_YYYYMMDD_HHMMSS_analysis.json  # Statistical tests
└── SCIENTIFIC_REPORT.md                        # Human-readable report
```

## Troubleshooting

### "GROQ_API_KEY not found"
```bash
cp .env.example .env
# Edit .env with your Groq API key from https://console.groq.com
```

### "Rate limit exceeded"
The scripts include rate limiting (0.25s between requests). If you still hit limits:
- Wait 1-2 minutes and retry
- Upgrade your Groq tier at https://console.groq.com/settings/billing

### "Module not found"
Ensure you're running from the repository root with PYTHONPATH:
```bash
cd /path/to/spatial-lab
PYTHONPATH=. python scripts/your_script.py
```
