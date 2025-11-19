#!/usr/bin/env python3
"""
Runner script for NL + Code Generation Robot Coordination Experiment.

Research Question: Can robots effectively communicate needs in natural language
and have those needs fulfilled through LLM-generated code?

Usage:
    # Quick test (1 trial)
    python run_experiment.py --test

    # Run full experiment (50 trials)
    python run_experiment.py --trials 50

    # Run with custom settings
    python run_experiment.py --trials 100 --robots 10 --output results.json

    # Compare LLM vs algorithmic approach
    python run_experiment.py --compare --trials 30
"""

import asyncio
import argparse
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_lab.research import (
    NLCodeExperiment,
    run_quick_test,
    RobotState,
    NeedType
)
from spatial_lab.research.baseline_comparison import BaselineComparison


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Run NL + Code Generation Robot Coordination Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test (single trial)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison: LLM vs algorithmic approach"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials to run (default: 50)"
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=5,
        help="Number of robots in experiment (default: 5)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b-versatile",
        help="Groq model to use"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.json",
        help="Output file for detailed results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Groq API key (or set GROQ_API_KEY env var)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Get API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("Error: No API key provided.")
        print("Set GROQ_API_KEY environment variable or use --api-key flag")
        return 1

    print("=" * 70)
    print("NL + Code Generation Robot Coordination Experiment")
    print("=" * 70)
    print()
    print("Research Question:")
    print("  Can robots communicate needs in natural language and have")
    print("  those needs fulfilled through LLM-generated code?")
    print()

    if args.test:
        # Run quick test
        print("Running quick test...")
        print("-" * 70)
        success = await run_quick_test(api_key)
        print("-" * 70)
        print(f"Quick test: {'PASSED' if success else 'FAILED'}")
        return 0 if success else 1

    if args.compare:
        # Run baseline comparison
        print("Running baseline comparison: Algorithmic vs LLM")
        print("-" * 70)

        comparison = BaselineComparison(api_key, model=args.model)
        results = await comparison.run_comparison(num_trials=args.trials)
        print(comparison.get_summary())

        return 0

    # Run full experiment
    print(f"Configuration:")
    print(f"  Trials:  {args.trials}")
    print(f"  Robots:  {args.robots}")
    print(f"  Model:   {args.model}")
    print(f"  Output:  {args.output}")
    print()
    print("Starting experiment...")
    print("-" * 70)

    experiment = NLCodeExperiment(api_key, model=args.model)

    # Generate robots
    import random
    robots = []
    for i in range(args.robots):
        robots.append(RobotState(
            robot_id=f"robot_{i:03d}",
            position=(random.uniform(0, 50), random.uniform(0, 50)),
            battery=random.uniform(0.3, 1.0),
            carrying=random.choice([None, f"item_{random.randint(1,100)}"]),
            current_task=random.choice([None, "pickup", "delivery", "patrol"])
        ))

    # Run experiment
    results = await experiment.run_experiment(
        num_trials=args.trials,
        robots=robots
    )

    # Print summary
    print(experiment.get_summary())

    # Export detailed results
    experiment.export_results(args.output)
    print(f"\nDetailed results saved to: {args.output}")

    # Return success based on results
    success_rate = results.successful_trials / results.total_trials if results.total_trials > 0 else 0
    return 0 if success_rate > 0.5 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
