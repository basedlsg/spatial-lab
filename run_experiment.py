#!/usr/bin/env python3
"""
Spatial Lab - Quick Experiment Runner
Wrapper script for easy experiment execution
"""

import argparse
import asyncio
import os
import sys
from spatial_lab.experiment_runner import run_spatial_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run Spatial AI Research Lab experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_experiment.py --test
  
  # Full experiment with 50 trials
  python run_experiment.py --trials 50
  
  # Compare LLM vs algorithmic baseline
  python run_experiment.py --compare --trials 30
  
  # Custom configuration
  python run_experiment.py --robots 5 --episodes 100
        """
    )
    
    parser.add_argument("--test", action="store_true", 
                       help="Run quick test with minimal configuration")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of evaluation trials (default: 10)")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison between LLM and algorithmic baseline")
    parser.add_argument("--robots", type=int, default=3,
                       help="Number of robots in warehouse (default: 3)")
    parser.add_argument("--episodes", type=int, default=0,
                       help="Number of training episodes (default: 0)")
    parser.add_argument("--config", default="basic_warehouse",
                       help="Configuration preset to use")
    
    args = parser.parse_args()
    
    # Setup Groq API if GROQ_API_KEY is set
    if "GROQ_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        print("🔑 Using Groq API")
    
    # Build configuration overrides
    overrides = {}
    
    if args.test:
        print("🧪 Running quick test configuration...")
        overrides = {
            "num_robots": 2,
            "num_evaluation_tasks": 3,
            "num_evaluation_runs": 1,
            "num_training_episodes": 0,
            "warehouse_width": 20,
            "warehouse_height": 20,
            "num_shelves": 10
        }
    else:
        overrides["num_robots"] = args.robots
        overrides["num_evaluation_tasks"] = args.trials
        overrides["num_training_episodes"] = args.episodes
        
        if args.compare:
            print("📊 Running comparative analysis...")
            overrides["compare_baselines"] = True
    
    # Run experiment
    try:
        print(f"\n🚀 Starting Spatial Lab Experiment")
        print(f"   Configuration: {args.config}")
        print(f"   Robots: {overrides.get('num_robots', args.robots)}")
        print(f"   Evaluation Tasks: {overrides.get('num_evaluation_tasks', args.trials)}")
        print(f"   Training Episodes: {overrides.get('num_training_episodes', args.episodes)}\n")
        
        asyncio.run(run_spatial_experiment(args.config, **overrides))
        
        print("\n✅ Experiment completed successfully!")
        print("📁 Results saved to experiment output directory")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
