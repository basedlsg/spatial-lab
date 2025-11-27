#!/usr/bin/env python3
"""
Spatial Lab - Groq-focused Experiment

Runs spatial reasoning experiments using Groq's fast Llama inference.
Optimized for collecting high-quality scientific data.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import numpy as np

sys.path.insert(0, '.')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Single trial result"""
    trial_id: int
    task_type: str
    model: str

    # Decision
    action: str
    destination: List[float]
    confidence: float
    reasoning: str

    # Metrics
    latency_ms: int
    success: bool
    path_efficiency: float
    spatial_correct: bool

    # Context
    robot_position: List[float]
    target_position: List[float]
    timestamp: str


class GroqExperiment:
    """Groq-focused spatial reasoning experiment"""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.output_dir = Path("experiment_results")
        self.output_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[TrialResult] = []

        from spatial_lab.llm.groq_client import GroqAPIConfig, GroqAPIClient
        self.GroqAPIConfig = GroqAPIConfig
        self.GroqAPIClient = GroqAPIClient

        logger.info(f"Experiment {self.experiment_id} initialized")

    def generate_scenarios(self, n: int = 30) -> List[Dict]:
        """Generate diverse spatial reasoning scenarios"""
        np.random.seed(42)
        scenarios = []

        task_types = [
            ("pickup", "Pick up item from shelf"),
            ("delivery", "Deliver item to shipping zone"),
            ("navigate", "Navigate around obstacle"),
            ("multi_stop", "Visit multiple locations"),
            ("coordination", "Coordinate with nearby robot")
        ]

        for i in range(n):
            task_name, task_base = task_types[i % len(task_types)]

            # Robot position
            rx, ry = np.random.uniform(2, 18, 2)

            # Target position
            tx, ty = np.random.uniform(2, 18, 2)
            while np.sqrt((tx-rx)**2 + (ty-ry)**2) < 3:  # Ensure some distance
                tx, ty = np.random.uniform(2, 18, 2)

            # Build scenario
            scenario = {
                "id": i + 1,
                "task_type": task_name,
                "robot_id": f"robot_{i+1:03d}",
                "robot_position": [round(rx, 1), round(ry, 1), 0.0],
                "target_position": [round(tx, 1), round(ty, 1)],
                "task_description": f"{task_base} at position ({tx:.1f}, {ty:.1f})",
                "battery": round(np.random.uniform(0.6, 1.0), 2),
                "carrying": task_name == "delivery",
                "nearby_robot": task_name == "coordination",
                "obstacle": task_name == "navigate"
            }
            scenarios.append(scenario)

        return scenarios

    async def run_trial(self, scenario: Dict, model: str) -> TrialResult:
        """Run single trial"""

        start = time.time()

        config = self.GroqAPIConfig(
            api_key=self.api_key,
            model=model,
            temperature=0.7
        )

        # Build observation
        observation = {
            "position": scenario["robot_position"],
            "battery_level": scenario["battery"],
            "carrying_item": "item_A" if scenario["carrying"] else None,
            "status": "carrying" if scenario["carrying"] else "idle"
        }

        # Build layout
        layout = {
            "dimensions": [20, 20],
            "target": {"position": scenario["target_position"]}
        }

        if scenario["obstacle"]:
            # Place obstacle between robot and target
            ox = (scenario["robot_position"][0] + scenario["target_position"][0]) / 2
            oy = (scenario["robot_position"][1] + scenario["target_position"][1]) / 2
            layout["obstacles"] = [{"position": [round(ox, 1), round(oy, 1)], "radius": 1.5}]

        nearby = []
        if scenario["nearby_robot"]:
            nx = scenario["robot_position"][0] + np.random.uniform(-2, 2)
            ny = scenario["robot_position"][1] + np.random.uniform(-2, 2)
            nearby = [{"robot_id": "robot_nearby", "position": [round(nx, 1), round(ny, 1), 0.0]}]

        try:
            async with self.GroqAPIClient(config) as client:
                response = await client.robot_coordination_decision(
                    robot_id=scenario["robot_id"],
                    observation=observation,
                    task_description=scenario["task_description"],
                    available_actions=["move_to", "pick_item", "drop_item", "wait"],
                    nearby_robots=nearby,
                    warehouse_layout=layout
                )

            latency = int((time.time() - start) * 1000)

            # Parse response
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
                decision = json.loads(content)
            else:
                decision = {"action": "error", "confidence": 0}

            action = decision.get("action", "unknown")
            params = decision.get("parameters", {})
            dest = params.get("destination", [0, 0])
            confidence = decision.get("confidence", 0.5)
            reasoning = decision.get("reasoning", "")[:200]

            # Evaluate
            success = action == "move_to"

            # Path efficiency
            if isinstance(dest, list) and len(dest) >= 2:
                actual_dist = np.sqrt(
                    (dest[0] - scenario["robot_position"][0])**2 +
                    (dest[1] - scenario["robot_position"][1])**2
                )
                optimal_dist = np.sqrt(
                    (scenario["target_position"][0] - scenario["robot_position"][0])**2 +
                    (scenario["target_position"][1] - scenario["robot_position"][1])**2
                )
                # Check if destination is close to target
                target_dist = np.sqrt(
                    (dest[0] - scenario["target_position"][0])**2 +
                    (dest[1] - scenario["target_position"][1])**2
                )
                spatial_correct = target_dist < 2.0  # Within 2 units of target
                path_efficiency = min(1.0, optimal_dist / max(actual_dist, 0.1)) if spatial_correct else 0.3
            else:
                spatial_correct = False
                path_efficiency = 0.0
                dest = [0, 0]

            return TrialResult(
                trial_id=scenario["id"],
                task_type=scenario["task_type"],
                model=model,
                action=action,
                destination=dest if isinstance(dest, list) else [0, 0],
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency,
                success=success,
                path_efficiency=round(path_efficiency, 3),
                spatial_correct=spatial_correct,
                robot_position=scenario["robot_position"],
                target_position=scenario["target_position"],
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Trial {scenario['id']} failed: {e}")
            return TrialResult(
                trial_id=scenario["id"],
                task_type=scenario["task_type"],
                model=model,
                action="error",
                destination=[0, 0],
                confidence=0.0,
                reasoning=str(e),
                latency_ms=int((time.time() - start) * 1000),
                success=False,
                path_efficiency=0.0,
                spatial_correct=False,
                robot_position=scenario["robot_position"],
                target_position=scenario["target_position"],
                timestamp=datetime.now().isoformat()
            )

    async def run_experiment(self, n_trials: int = 30) -> Dict:
        """Run full experiment"""

        logger.info(f"\n{'='*60}")
        logger.info("SPATIAL REASONING EXPERIMENT - GROQ/LLAMA")
        logger.info(f"{'='*60}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Model: llama-3.3-70b-versatile")
        logger.info(f"{'='*60}\n")

        scenarios = self.generate_scenarios(n_trials)

        for i, scenario in enumerate(scenarios):
            logger.info(f"Trial {i+1}/{n_trials}: {scenario['task_type']}")

            result = await self.run_trial(scenario, "llama-3.3-70b-versatile")
            self.results.append(result)

            status = "OK" if result.success and result.spatial_correct else "PARTIAL" if result.success else "FAIL"
            logger.info(f"  -> {status} | Action: {result.action} | "
                       f"Confidence: {result.confidence:.2f} | "
                       f"Spatial: {'correct' if result.spatial_correct else 'wrong'} | "
                       f"Latency: {result.latency_ms}ms")

            await asyncio.sleep(0.2)  # Rate limiting

        # Calculate statistics
        stats = self._calculate_stats()

        # Save results
        self._save_results(stats)

        return stats

    def _calculate_stats(self) -> Dict:
        """Calculate experiment statistics"""

        if not self.results:
            return {}

        successes = [r for r in self.results if r.success]
        spatial_correct = [r for r in self.results if r.spatial_correct]

        # By task type
        task_stats = {}
        for task_type in set(r.task_type for r in self.results):
            task_results = [r for r in self.results if r.task_type == task_type]
            task_stats[task_type] = {
                "count": len(task_results),
                "success_rate": len([r for r in task_results if r.success]) / len(task_results),
                "spatial_accuracy": len([r for r in task_results if r.spatial_correct]) / len(task_results),
                "avg_confidence": np.mean([r.confidence for r in task_results]),
                "avg_latency_ms": np.mean([r.latency_ms for r in task_results])
            }

        stats = {
            "experiment_id": self.experiment_id,
            "total_trials": len(self.results),
            "successful_actions": len(successes),
            "action_success_rate": len(successes) / len(self.results),
            "spatially_correct": len(spatial_correct),
            "spatial_accuracy": len(spatial_correct) / len(self.results),
            "avg_confidence": np.mean([r.confidence for r in self.results]),
            "std_confidence": np.std([r.confidence for r in self.results]),
            "avg_latency_ms": np.mean([r.latency_ms for r in self.results]),
            "std_latency_ms": np.std([r.latency_ms for r in self.results]),
            "avg_path_efficiency": np.mean([r.path_efficiency for r in self.results]),
            "task_breakdown": task_stats
        }

        return stats

    def _save_results(self, stats: Dict):
        """Save results to files"""

        # Detailed results
        results_file = self.output_dir / f"groq_experiment_{self.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Summary
        summary_file = self.output_dir / f"groq_experiment_{self.experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENT RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trials: {stats['total_trials']}")
        logger.info(f"Action Success Rate: {stats['action_success_rate']*100:.1f}%")
        logger.info(f"Spatial Accuracy: {stats['spatial_accuracy']*100:.1f}%")
        logger.info(f"Avg Confidence: {stats['avg_confidence']:.2f} (+/- {stats['std_confidence']:.2f})")
        logger.info(f"Avg Latency: {stats['avg_latency_ms']:.0f}ms (+/- {stats['std_latency_ms']:.0f}ms)")
        logger.info(f"Avg Path Efficiency: {stats['avg_path_efficiency']:.2f}")

        logger.info(f"\nBy Task Type:")
        for task, task_stat in stats['task_breakdown'].items():
            logger.info(f"  {task}: {task_stat['success_rate']*100:.0f}% success, "
                       f"{task_stat['spatial_accuracy']*100:.0f}% spatial accuracy")

        logger.info(f"\nResults: {results_file}")
        logger.info(f"Summary: {summary_file}")
        logger.info(f"{'='*60}\n")


async def main():
    print("\n" + "="*60)
    print("SPATIAL LAB - GROQ EXPERIMENT")
    print("="*60 + "\n")

    experiment = GroqExperiment()
    stats = await experiment.run_experiment(n_trials=30)

    print(f"\nExperiment Complete!")
    print(f"Spatial Accuracy: {stats['spatial_accuracy']*100:.1f}%")

    return stats['spatial_accuracy'] >= 0.6


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
