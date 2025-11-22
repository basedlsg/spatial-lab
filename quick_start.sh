#!/bin/bash
# Quick Start Script for Research Experiment

set -e

echo "========================================================================"
echo "Spatial Lab: Natural Language + Code Generation Research Experiment"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check/install dependencies
echo ""
echo "Checking dependencies..."
if ! python -c "import scipy" 2>/dev/null; then
    echo "Installing scipy..."
    pip install scipy --quiet
fi

if ! python -c "import aiohttp" 2>/dev/null; then
    echo "Installing aiohttp..."
    pip install aiohttp --quiet
fi

echo "Dependencies OK!"

# Check API key
echo ""
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY environment variable not set"
    echo ""
    echo "Please set it first:"
    echo "  export GROQ_API_KEY=\"your_groq_api_key_here\""
    echo ""
    exit 1
else
    echo "Using GROQ_API_KEY environment variable"
fi

# Show menu
echo ""
echo "========================================================================"
echo "What would you like to run?"
echo "========================================================================"
echo ""
echo "  1) Quick Test (1 trial, ~5 seconds)"
echo "  2) Full Experiment (50 trials, ~5-10 minutes)"
echo "  3) Baseline Comparison (30 trials, LLM vs Algorithmic)"
echo "  4) Custom Configuration"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        python run_experiment.py --test
        ;;
    2)
        echo ""
        echo "Running full experiment (50 trials)..."
        python run_experiment.py --trials 50
        ;;
    3)
        echo ""
        echo "Running baseline comparison..."
        python run_experiment.py --compare --trials 30
        ;;
    4)
        echo ""
        read -p "Number of trials [50]: " trials
        trials=${trials:-50}
        read -p "Number of robots [5]: " robots
        robots=${robots:-5}
        read -p "Output file [experiment_results.json]: " output
        output=${output:-experiment_results.json}

        echo ""
        echo "Running custom experiment..."
        python run_experiment.py --trials $trials --robots $robots --output $output
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Experiment complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  - Check experiment_results.json for detailed data"
echo "  - Read RESEARCH_EXPERIMENT.md for analysis tips"
echo "  - Run again with different parameters to compare"
echo ""
