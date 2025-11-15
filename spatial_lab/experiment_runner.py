"""
Spatial AI Research Lab - Experiment Runner

Main experiment orchestration for warehouse spatial reasoning experiments.
Integrates with Nous Atropos infrastructure for LLM-based agent evaluation.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Atropos imports
from atroposlib.envs.base import BaseEnv
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

# Spatial lab imports
from .environments import WarehouseSpatialEnvironment, WarehouseSpatialEnvironmentConfig
from .config import ExperimentConfig, setup_logging, setup_wandb, validate_config
from .evaluation import SpatialMetricsCalculator, PerformanceAnalyzer, StatisticalAnalyzer

logger = logging.getLogger(__name__)


class SpatialReasoningExperiment:
    """Main experiment orchestrator for spatial reasoning research"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = f"{config.experiment_name}_{int(time.time())}"
        
        # Initialize components
        self.environment: Optional[WarehouseSpatialEnvironment] = None
        self.metrics_calculator = SpatialMetricsCalculator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Experiment state
        self.results = []
        self.current_episode = 0
        self.start_time = None
        self.output_dir = None
        
        logger.info(f"Initialized experiment: {self.experiment_id}")
    
    async def setup(self):
        """Setup experiment environment and infrastructure"""
        logger.info("Setting up spatial reasoning experiment")
        
        # Setup logging and output directory
        self.output_dir = Path(self.config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging(self.config)
        setup_wandb(self.config)
        
        # Validate configuration
        warnings = validate_config(self.config)
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # Create environment configuration
        env_config = WarehouseSpatialEnvironmentConfig(
            warehouse_width=self.config.warehouse_width,
            warehouse_height=self.config.warehouse_height,
            num_robots=self.config.num_robots,
            num_shelves=self.config.num_shelves,
            max_task_duration=self.config.max_task_duration,
            items_per_task=self.config.items_per_task,
            task_complexity=self.config.task_complexity
        )
        
        # Create server configuration for LLM inference
        server_configs = [
            APIServerConfig(
                model_name=self.config.model_name,
                base_url=None,
                api_key=None,  # Will use environment variable
                num_requests_for_eval=self.config.batch_size,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        ]
        
        # Initialize environment
        self.environment = WarehouseSpatialEnvironment(
            config=env_config,
            server_configs=server_configs
        )
        
        await self.environment.setup()
        
        logger.info("Experiment setup complete")
    
    async def run_experiment(self):
        """Run the complete spatial reasoning experiment"""
        logger.info(f"Starting experiment: {self.experiment_id}")
        self.start_time = time.time()
        
        try:
            # Run training episodes if configured
            if self.config.num_training_episodes > 0:
                await self._run_training_phase()
            
            # Run evaluation
            await self._run_evaluation_phase()
            
            # Generate final analysis
            await self._generate_final_analysis()
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            await self._cleanup()
        
        total_time = time.time() - self.start_time
        logger.info(f"Experiment completed in {total_time:.2f} seconds")
    
    async def _run_training_phase(self):
        """Run training episodes"""
        logger.info(f"Starting training phase: {self.config.num_training_episodes} episodes")
        
        for episode in range(self.config.num_training_episodes):
            self.current_episode = episode
            
            logger.info(f"Training episode {episode + 1}/{self.config.num_training_episodes}")
            
            # Generate task
            task_item = await self.environment.get_next_item()
            
            # Execute episode
            episode_result, _ = await self.environment.collect_trajectories(task_item)
            
            # Calculate metrics
            episode_metrics = await self.metrics_calculator.calculate_episode_metrics(
                episode_result.scored_items, 
                task_item.data["task"]
            )
            
            # Store results
            self.results.append({
                "episode": episode,
                "phase": "training",
                "task_id": task_item.data["task"]["task_id"],
                "metrics": episode_metrics,
                "episode_result": episode_result
            })
            
            # Log progress
            if (episode + 1) % 10 == 0:
                await self._log_progress(episode + 1, "training")
            
            # Run evaluation periodically
            if (episode + 1) % self.config.evaluation_frequency == 0:
                await self._run_periodic_evaluation(episode + 1)
            
            # Save checkpoints
            if (episode + 1) % self.config.save_frequency == 0:
                await self._save_checkpoint(episode + 1)
        
        logger.info("Training phase completed")
    
    async def _run_evaluation_phase(self):
        """Run comprehensive evaluation"""
        logger.info(f"Starting evaluation phase: {self.config.num_evaluation_tasks} tasks")
        
        evaluation_results = []
        
        for run in range(self.config.num_evaluation_runs):
            logger.info(f"Evaluation run {run + 1}/{self.config.num_evaluation_runs}")
            
            run_results = []
            
            for task_num in range(self.config.num_evaluation_tasks):
                logger.info(f"Evaluation task {task_num + 1}/{self.config.num_evaluation_tasks}")
                
                # Generate evaluation task
                task_item = await self.environment.get_next_item()
                
                # Execute task
                episode_result, _ = await self.environment.collect_trajectories(task_item)
                
                # Calculate metrics
                episode_metrics = await self.metrics_calculator.calculate_episode_metrics(
                    episode_result.scored_items,
                    task_item.data["task"]
                )
                
                run_results.append({
                    "task": task_num,
                    "task_id": task_item.data["task"]["task_id"],
                    "metrics": episode_metrics,
                    "episode_result": episode_result
                })
            
            evaluation_results.append({
                "run": run,
                "results": run_results
            })
        
        # Store evaluation results
        self.evaluation_results = evaluation_results
        
        logger.info("Evaluation phase completed")
    
    async def _run_periodic_evaluation(self, episode: int):
        """Run periodic evaluation during training"""
        logger.info(f"Running periodic evaluation at episode {episode}")
        
        eval_results = []
        
        for i in range(5):  # Quick evaluation with 5 tasks
            task_item = await self.environment.get_next_item()
            episode_result, _ = await self.environment.collect_trajectories(task_item)
            
            episode_metrics = await self.metrics_calculator.calculate_episode_metrics(
                episode_result.scored_items,
                task_item.data["task"]
            )
            
            eval_results.append(episode_metrics)
        
        # Calculate average performance
        avg_metrics = self._calculate_average_metrics(eval_results)
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log({
                    "episode": episode,
                    "eval_path_efficiency": avg_metrics.path_efficiency,
                    "eval_navigation_success": avg_metrics.navigation_success_rate,
                    "eval_collision_rate": avg_metrics.collision_rate,
                    "eval_coordination_efficiency": avg_metrics.coordination_efficiency,
                    "eval_task_completion": avg_metrics.task_completion_rate
                })
            except ImportError:
                pass
        
        logger.info(f"Periodic evaluation complete - Task completion: {avg_metrics.task_completion_rate:.2%}")
    
    async def _generate_final_analysis(self):
        """Generate comprehensive final analysis"""
        logger.info("Generating final analysis")
        
        # Analyze training progression if available
        training_analysis = None
        if self.results:
            training_results = [r for r in self.results if r["phase"] == "training"]
            if training_results:
                training_analysis = await self.performance_analyzer.analyze_training_progression(
                    training_results
                )
        
        # Analyze evaluation results
        evaluation_analysis = await self.performance_analyzer.analyze_evaluation_results(
            self.evaluation_results
        )
        
        # Statistical analysis
        statistical_analysis = await self.statistical_analyzer.analyze_results(
            self.evaluation_results,
            confidence_level=self.config.confidence_level
        )
        
        # Comparative analysis
        comparative_analysis = self.metrics_calculator.get_comparative_analysis()
        
        # Compile final report
        final_report = {
            "experiment_info": {
                "experiment_id": self.experiment_id,
                "config": self.config.to_dict(),
                "start_time": self.start_time,
                "duration": time.time() - self.start_time if self.start_time else 0
            },
            "training_analysis": training_analysis,
            "evaluation_analysis": evaluation_analysis,
            "statistical_analysis": statistical_analysis,
            "comparative_analysis": comparative_analysis,
            "conclusions": await self._generate_conclusions(
                evaluation_analysis, statistical_analysis, comparative_analysis
            )
        }
        
        # Save final report
        await self._save_final_report(final_report)
        
        logger.info("Final analysis completed")
    
    async def _generate_conclusions(
        self, 
        evaluation_analysis: Dict, 
        statistical_analysis: Dict, 
        comparative_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate scientific conclusions from analysis"""
        
        conclusions = {
            "key_findings": [],
            "statistical_significance": {},
            "effect_sizes": {},
            "performance_vs_baselines": {},
            "limitations": [],
            "future_work": []
        }
        
        # Key findings based on evaluation
        if evaluation_analysis:
            avg_completion = evaluation_analysis.get("average_completion_rate", 0)
            avg_efficiency = evaluation_analysis.get("average_path_efficiency", 0)
            
            if avg_completion > 0.8:
                conclusions["key_findings"].append(
                    f"High task completion rate observed ({avg_completion:.1%})"
                )
            
            if avg_efficiency > 0.7:
                conclusions["key_findings"].append(
                    f"Efficient path planning demonstrated ({avg_efficiency:.1%} efficiency)"
                )
        
        # Statistical significance
        if statistical_analysis:
            for metric, p_value in statistical_analysis.get("p_values", {}).items():
                is_significant = p_value < self.config.significance_threshold
                conclusions["statistical_significance"][metric] = {
                    "p_value": p_value,
                    "significant": is_significant,
                    "threshold": self.config.significance_threshold
                }
        
        # Effect sizes
        if statistical_analysis:
            for metric, effect_size in statistical_analysis.get("effect_sizes", {}).items():
                magnitude = "small" if abs(effect_size) < 0.3 else \
                           "medium" if abs(effect_size) < 0.8 else "large"
                conclusions["effect_sizes"][metric] = {
                    "value": effect_size,
                    "magnitude": magnitude,
                    "threshold": self.config.effect_size_threshold
                }
        
        # Performance vs baselines
        if comparative_analysis:
            baselines = comparative_analysis.get("baseline_comparisons", {})
            for baseline, improvements in baselines.items():
                conclusions["performance_vs_baselines"][baseline] = improvements
        
        # Standard limitations
        conclusions["limitations"] = [
            "Evaluation limited to simulated warehouse environment",
            "Performance may vary with different warehouse layouts",
            "Results specific to current task complexity settings",
            f"Sample size: {self.config.num_evaluation_tasks * self.config.num_evaluation_runs} episodes"
        ]
        
        # Future work suggestions
        conclusions["future_work"] = [
            "Evaluate performance in more diverse environments",
            "Test scalability with larger robot fleets",
            "Investigate transfer learning to real-world scenarios",
            "Compare with additional baseline algorithms"
        ]
        
        return conclusions
    
    async def _log_progress(self, episode: int, phase: str):
        """Log experiment progress"""
        
        if not self.results:
            return
        
        recent_results = [r for r in self.results[-10:] if r["phase"] == phase]
        if not recent_results:
            return
        
        # Calculate recent performance
        recent_metrics = [r["metrics"] for r in recent_results]
        avg_completion = sum(m.task_completion_rate for m in recent_metrics) / len(recent_metrics)
        avg_efficiency = sum(m.path_efficiency for m in recent_metrics) / len(recent_metrics)
        avg_collisions = sum(m.collision_rate for m in recent_metrics) / len(recent_metrics)
        
        logger.info(
            f"{phase.title()} progress (episode {episode}): "
            f"Completion: {avg_completion:.1%}, "
            f"Efficiency: {avg_efficiency:.1%}, "
            f"Collisions: {avg_collisions:.2f}/episode"
        )
        
        # Log to wandb
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log({
                    "episode": episode,
                    f"{phase}_completion_rate": avg_completion,
                    f"{phase}_path_efficiency": avg_efficiency,
                    f"{phase}_collision_rate": avg_collisions
                })
            except ImportError:
                pass
    
    async def _save_checkpoint(self, episode: int):
        """Save experiment checkpoint"""
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{episode}.json"
        
        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "episode": episode,
            "config": self.config.to_dict(),
            "results": [
                {
                    "episode": r["episode"],
                    "phase": r["phase"],
                    "task_id": r["task_id"],
                    "metrics": r["metrics"].to_dict()
                }
                for r in self.results
            ],
            "timestamp": time.time()
        }
        
        import json
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint at episode {episode}")
    
    async def _save_final_report(self, report: Dict[str, Any]):
        """Save final experiment report"""
        
        report_file = self.output_dir / "final_report.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save a human-readable summary
        summary_file = self.output_dir / "experiment_summary.md"
        await self._generate_markdown_summary(report, summary_file)
        
        logger.info(f"Saved final report: {report_file}")
        logger.info(f"Saved summary: {summary_file}")
    
    async def _generate_markdown_summary(self, report: Dict, output_file: Path):
        """Generate human-readable markdown summary"""
        
        with open(output_file, 'w') as f:
            f.write(f"# Spatial AI Research Lab - Experiment Report\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Experimental Configuration\n\n")
            config = report["experiment_info"]["config"]
            f.write(f"- **Environment:** {config['environment']['warehouse_width']}x{config['environment']['warehouse_height']}m warehouse\n")
            f.write(f"- **Robots:** {config['environment']['num_robots']} agents\n")
            f.write(f"- **Task Complexity:** {config['tasks']['complexity']}\n")
            f.write(f"- **Evaluation Tasks:** {config['tasks']['num_evaluation_tasks']}\n\n")
            
            # Key findings
            conclusions = report.get("conclusions", {})
            if conclusions.get("key_findings"):
                f.write("## Key Findings\n\n")
                for finding in conclusions["key_findings"]:
                    f.write(f"- {finding}\n")
                f.write("\n")
            
            # Statistical results
            if conclusions.get("statistical_significance"):
                f.write("## Statistical Analysis\n\n")
                f.write("| Metric | p-value | Significant | Effect Size |\n")
                f.write("|--------|---------|-------------|-------------|\n")
                
                significance = conclusions["statistical_significance"]
                effect_sizes = conclusions.get("effect_sizes", {})
                
                for metric, stats in significance.items():
                    p_val = stats["p_value"]
                    significant = "Yes" if stats["significant"] else "No"
                    effect_size = effect_sizes.get(metric, {}).get("value", "N/A")
                    f.write(f"| {metric} | {p_val:.3f} | {significant} | {effect_size:.3f} |\n")
                f.write("\n")
            
            # Performance comparison
            if conclusions.get("performance_vs_baselines"):
                f.write("## Performance vs Baselines\n\n")
                baselines = conclusions["performance_vs_baselines"]
                for baseline, improvements in baselines.items():
                    f.write(f"### {baseline.replace('_', ' ').title()}\n\n")
                    for metric, improvement in improvements.items():
                        f.write(f"- **{metric.replace('_', ' ').title()}:** {improvement:+.1f}%\n")
                    f.write("\n")
            
            # Limitations
            if conclusions.get("limitations"):
                f.write("## Limitations\n\n")
                for limitation in conclusions["limitations"]:
                    f.write(f"- {limitation}\n")
                f.write("\n")
            
            # Future work
            if conclusions.get("future_work"):
                f.write("## Future Work\n\n")
                for work in conclusions["future_work"]:
                    f.write(f"- {work}\n")
                f.write("\n")
    
    def _calculate_average_metrics(self, metrics_list: List) -> Any:
        """Calculate average metrics from a list"""
        if not metrics_list:
            return None
        
        # Create average metrics object
        from .evaluation.spatial_metrics import SpatialMetrics
        
        avg_metrics = SpatialMetrics(
            path_efficiency=sum(m.path_efficiency for m in metrics_list) / len(metrics_list),
            navigation_success_rate=sum(m.navigation_success_rate for m in metrics_list) / len(metrics_list),
            collision_rate=sum(m.collision_rate for m in metrics_list) / len(metrics_list),
            coordination_efficiency=sum(m.coordination_efficiency for m in metrics_list) / len(metrics_list),
            task_completion_rate=sum(m.task_completion_rate for m in metrics_list) / len(metrics_list)
        )
        
        return avg_metrics
    
    async def _cleanup(self):
        """Cleanup experiment resources"""
        logger.info("Cleaning up experiment resources")
        
        # Close wandb if used
        if self.config.use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        
        # Any other cleanup tasks
        logger.info("Cleanup completed")


async def run_spatial_experiment(config_name: str = "basic_warehouse", **config_overrides):
    """Run a spatial reasoning experiment with specified configuration"""
    
    from .config import get_config, create_custom_config
    
    # Get configuration
    if config_overrides:
        config = create_custom_config(base_config=config_name, **config_overrides)
    else:
        config = get_config(config_name)
    
    # Create and run experiment
    experiment = SpatialReasoningExperiment(config)
    await experiment.setup()
    await experiment.run_experiment()
    
    return experiment


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run spatial reasoning experiments")
    parser.add_argument("--config", default="basic_warehouse", help="Configuration name")
    parser.add_argument("--robots", type=int, help="Number of robots")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--eval-tasks", type=int, help="Number of evaluation tasks")
    
    args = parser.parse_args()
    
    # Build config overrides
    overrides = {}
    if args.robots:
        overrides["environment.num_robots"] = args.robots
    if args.episodes:
        overrides["training.num_episodes"] = args.episodes
    if args.eval_tasks:
        overrides["evaluation.num_evaluation_tasks"] = args.eval_tasks
    
    # Run experiment
    asyncio.run(run_spatial_experiment(args.config, **overrides)) 