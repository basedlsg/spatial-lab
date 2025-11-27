#!/usr/bin/env python3
"""
Spatial Lab - Real Experiment Runner

Runs actual spatial reasoning experiments using Groq (primary) and Gemini (fallback) APIs.
Collects real data for scientific analysis.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import numpy as np

# Add project to path
sys.path.insert(0, '.')

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Single experiment trial result"""
    trial_id: int
    robot_id: str
    task_type: str
    llm_provider: str
    llm_model: str

    # Decision metrics
    action_chosen: str
    decision_confidence: float
    reasoning_quality: str  # 'good', 'acceptable', 'poor'

    # Performance metrics
    decision_latency_ms: int
    task_completed: bool
    steps_to_complete: int

    # Spatial reasoning metrics
    path_efficiency: float  # 1.0 = optimal, <1.0 = suboptimal
    collision_avoided: bool
    spatial_understanding: str  # 'correct', 'partial', 'incorrect'

    # Raw data
    observation: Dict
    response: Dict
    timestamp: str


@dataclass
class ExperimentSummary:
    """Summary statistics for experiment"""
    total_trials: int
    successful_trials: int
    success_rate: float

    avg_decision_latency_ms: float
    avg_confidence: float
    avg_path_efficiency: float

    provider_breakdown: Dict[str, int]
    action_distribution: Dict[str, int]

    spatial_understanding_accuracy: float
    collision_avoidance_rate: float


class SpatialReasoningExperiment:
    """
    Runs spatial reasoning experiments with real LLM APIs.
    """

    def __init__(
        self,
        groq_api_key: str = None,
        gemini_api_key: str = None,
        output_dir: str = "experiment_results"
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: List[ExperimentResult] = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Import LLM clients
        from spatial_lab.llm.groq_client import GroqAPIConfig, GroqAPIClient
        from spatial_lab.llm.gemini_client import GeminiAPIConfig, GeminiAPIClient

        self.GroqAPIConfig = GroqAPIConfig
        self.GroqAPIClient = GroqAPIClient
        self.GeminiAPIConfig = GeminiAPIConfig
        self.GeminiAPIClient = GeminiAPIClient

        logger.info(f"Initialized experiment {self.experiment_id}")

    def generate_warehouse_scenarios(self, num_scenarios: int = 10) -> List[Dict]:
        """Generate diverse warehouse task scenarios"""
        scenarios = []

        task_types = [
            "single_pickup",
            "multi_pickup",
            "delivery",
            "collision_avoidance",
            "path_planning"
        ]

        np.random.seed(42)  # Reproducibility

        for i in range(num_scenarios):
            task_type = task_types[i % len(task_types)]

            # Base robot position
            robot_x = np.random.uniform(1, 18)
            robot_y = np.random.uniform(1, 18)

            # Target positions
            shelf_x = np.random.uniform(5, 15)
            shelf_y = np.random.uniform(5, 15)
            delivery_x = np.random.uniform(1, 5)
            delivery_y = np.random.uniform(15, 19)

            scenario = {
                "scenario_id": i + 1,
                "task_type": task_type,
                "robot_id": f"robot_{i+1:03d}",
                "observation": {
                    "position": [round(robot_x, 1), round(robot_y, 1), 0.0],
                    "battery_level": round(np.random.uniform(0.5, 1.0), 2),
                    "carrying_item": None if task_type != "delivery" else f"item_{i}",
                    "status": "idle" if task_type != "delivery" else "carrying"
                },
                "task_description": self._generate_task_description(
                    task_type, shelf_x, shelf_y, delivery_x, delivery_y, i
                ),
                "warehouse_layout": {
                    "dimensions": [20, 20],
                    "shelves": [
                        {"id": f"shelf_{i}", "position": [round(shelf_x, 1), round(shelf_y, 1)]}
                    ],
                    "obstacles": self._generate_obstacles(task_type, robot_x, robot_y, shelf_x, shelf_y),
                    "zones": {
                        "shipping": {"position": [round(delivery_x, 1), round(delivery_y, 1)]}
                    }
                },
                "nearby_robots": self._generate_nearby_robots(task_type, robot_x, robot_y),
                "expected_action": self._get_expected_action(task_type),
                "optimal_path_length": self._calculate_optimal_path(
                    robot_x, robot_y, shelf_x, shelf_y, delivery_x, delivery_y, task_type
                )
            }

            scenarios.append(scenario)

        return scenarios

    def _generate_task_description(
        self, task_type: str, shelf_x: float, shelf_y: float,
        delivery_x: float, delivery_y: float, idx: int
    ) -> str:
        """Generate task description based on type"""
        descriptions = {
            "single_pickup": f"Pick up item_{idx} from shelf at position ({shelf_x:.1f}, {shelf_y:.1f})",
            "multi_pickup": f"Pick up item_{idx}_A and item_{idx}_B from shelf at ({shelf_x:.1f}, {shelf_y:.1f}), then deliver to shipping at ({delivery_x:.1f}, {delivery_y:.1f})",
            "delivery": f"You are carrying item_{idx}. Deliver it to shipping zone at ({delivery_x:.1f}, {delivery_y:.1f})",
            "collision_avoidance": f"Navigate to shelf at ({shelf_x:.1f}, {shelf_y:.1f}) while avoiding nearby robot",
            "path_planning": f"Find efficient path from current position to shelf at ({shelf_x:.1f}, {shelf_y:.1f}) avoiding obstacles"
        }
        return descriptions.get(task_type, descriptions["single_pickup"])

    def _generate_obstacles(
        self, task_type: str, robot_x: float, robot_y: float,
        shelf_x: float, shelf_y: float
    ) -> List[Dict]:
        """Generate obstacles for path planning scenarios"""
        if task_type != "path_planning":
            return []

        # Place obstacle between robot and target
        mid_x = (robot_x + shelf_x) / 2
        mid_y = (robot_y + shelf_y) / 2

        return [
            {"id": "obstacle_1", "position": [round(mid_x, 1), round(mid_y, 1)], "radius": 1.5}
        ]

    def _generate_nearby_robots(
        self, task_type: str, robot_x: float, robot_y: float
    ) -> List[Dict]:
        """Generate nearby robots for collision scenarios"""
        if task_type != "collision_avoidance":
            return []

        # Place robot nearby
        nearby_x = robot_x + np.random.uniform(-2, 2)
        nearby_y = robot_y + np.random.uniform(-2, 2)

        return [
            {
                "robot_id": "robot_nearby",
                "position": [round(nearby_x, 1), round(nearby_y, 1), 0.0],
                "status": "moving",
                "heading": "towards_target"
            }
        ]

    def _get_expected_action(self, task_type: str) -> str:
        """Get expected action for task type"""
        expected = {
            "single_pickup": "move_to",
            "multi_pickup": "move_to",
            "delivery": "move_to",
            "collision_avoidance": "move_to",
            "path_planning": "move_to"
        }
        return expected.get(task_type, "move_to")

    def _calculate_optimal_path(
        self, robot_x: float, robot_y: float,
        shelf_x: float, shelf_y: float,
        delivery_x: float, delivery_y: float,
        task_type: str
    ) -> float:
        """Calculate optimal path length for efficiency scoring"""
        if task_type == "delivery":
            return np.sqrt((delivery_x - robot_x)**2 + (delivery_y - robot_y)**2)
        else:
            return np.sqrt((shelf_x - robot_x)**2 + (shelf_y - robot_y)**2)

    async def run_trial(
        self,
        scenario: Dict,
        use_groq: bool = True
    ) -> ExperimentResult:
        """Run a single experiment trial"""

        start_time = time.time()

        # Select provider
        if use_groq and self.groq_api_key:
            config = self.GroqAPIConfig(
                api_key=self.groq_api_key,
                model="llama-3.3-70b-versatile"
            )
            client_class = self.GroqAPIClient
            provider = "groq"
            model = "llama-3.3-70b-versatile"
        elif self.gemini_api_key:
            config = self.GeminiAPIConfig(
                api_key=self.gemini_api_key,
                model="gemini-2.0-flash-exp"
            )
            client_class = self.GeminiAPIClient
            provider = "gemini"
            model = "gemini-2.0-flash-exp"
        else:
            raise ValueError("No API keys available")

        try:
            async with client_class(config) as client:
                response = await client.robot_coordination_decision(
                    robot_id=scenario["robot_id"],
                    observation=scenario["observation"],
                    task_description=scenario["task_description"],
                    available_actions=["move_to", "pick_item", "drop_item", "wait", "communicate"],
                    nearby_robots=scenario.get("nearby_robots", []),
                    warehouse_layout=scenario["warehouse_layout"]
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            decision = self._parse_response(response, provider)

            # Evaluate decision quality
            action_correct = decision.get("action") == scenario["expected_action"]
            confidence = decision.get("confidence", 0.5)

            # Calculate path efficiency
            if "destination" in decision.get("parameters", {}):
                dest = decision["parameters"]["destination"]
                if isinstance(dest, list) and len(dest) >= 2:
                    actual_dist = np.sqrt(
                        (dest[0] - scenario["observation"]["position"][0])**2 +
                        (dest[1] - scenario["observation"]["position"][1])**2
                    )
                    optimal = scenario["optimal_path_length"]
                    # Path efficiency: how close is actual to optimal
                    path_efficiency = min(1.0, optimal / max(actual_dist, 0.1))
                else:
                    path_efficiency = 0.5
            else:
                path_efficiency = 0.5 if action_correct else 0.3

            # Evaluate spatial understanding
            spatial_understanding = "correct" if action_correct and confidence > 0.7 else \
                                   "partial" if action_correct else "incorrect"

            # Reasoning quality
            reasoning = decision.get("reasoning", "")
            reasoning_quality = "good" if len(reasoning) > 50 and action_correct else \
                               "acceptable" if len(reasoning) > 20 else "poor"

            return ExperimentResult(
                trial_id=scenario["scenario_id"],
                robot_id=scenario["robot_id"],
                task_type=scenario["task_type"],
                llm_provider=provider,
                llm_model=model,
                action_chosen=decision.get("action", "unknown"),
                decision_confidence=confidence,
                reasoning_quality=reasoning_quality,
                decision_latency_ms=latency_ms,
                task_completed=action_correct,
                steps_to_complete=1,
                path_efficiency=round(path_efficiency, 3),
                collision_avoided=True,  # Assumed for single-step
                spatial_understanding=spatial_understanding,
                observation=scenario["observation"],
                response=decision,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Trial {scenario['scenario_id']} failed: {e}")
            latency_ms = int((time.time() - start_time) * 1000)

            return ExperimentResult(
                trial_id=scenario["scenario_id"],
                robot_id=scenario["robot_id"],
                task_type=scenario["task_type"],
                llm_provider=provider,
                llm_model=model,
                action_chosen="error",
                decision_confidence=0.0,
                reasoning_quality="poor",
                decision_latency_ms=latency_ms,
                task_completed=False,
                steps_to_complete=0,
                path_efficiency=0.0,
                collision_avoided=True,
                spatial_understanding="incorrect",
                observation=scenario["observation"],
                response={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _parse_response(self, response: Dict, provider: str) -> Dict:
        """Parse LLM response based on provider format"""
        try:
            if provider == "groq":
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    return json.loads(content)
            elif provider == "gemini":
                if "candidates" in response and len(response["candidates"]) > 0:
                    content = response["candidates"][0]["content"]["parts"][0]["text"]
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    return {"action": "wait", "reasoning": content, "confidence": 0.5}
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse response: {e}")

        return {"action": "unknown", "parameters": {}, "reasoning": "Parse error", "confidence": 0.0}

    async def run_experiment(
        self,
        num_trials: int = 20,
        use_both_providers: bool = True
    ) -> ExperimentSummary:
        """Run full experiment with multiple trials"""

        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING SPATIAL REASONING EXPERIMENT")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Number of trials: {num_trials}")
        logger.info(f"{'='*60}\n")

        # Generate scenarios
        scenarios = self.generate_warehouse_scenarios(num_trials)

        # Run trials
        for i, scenario in enumerate(scenarios):
            # Alternate providers if using both
            use_groq = True
            if use_both_providers and i % 2 == 1 and self.gemini_api_key:
                use_groq = False

            logger.info(f"Trial {i+1}/{num_trials}: {scenario['task_type']} "
                       f"(Provider: {'Groq' if use_groq else 'Gemini'})")

            result = await self.run_trial(scenario, use_groq=use_groq)
            self.results.append(result)

            # Log result
            status = "OK" if result.task_completed else "FAIL"
            logger.info(f"  -> {status} | Action: {result.action_chosen} | "
                       f"Confidence: {result.decision_confidence:.2f} | "
                       f"Latency: {result.decision_latency_ms}ms")

            # Rate limiting pause
            await asyncio.sleep(0.3)

        # Generate summary
        summary = self._generate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _generate_summary(self) -> ExperimentSummary:
        """Generate experiment summary statistics"""

        if not self.results:
            return ExperimentSummary(
                total_trials=0, successful_trials=0, success_rate=0.0,
                avg_decision_latency_ms=0.0, avg_confidence=0.0, avg_path_efficiency=0.0,
                provider_breakdown={}, action_distribution={},
                spatial_understanding_accuracy=0.0, collision_avoidance_rate=0.0
            )

        successful = [r for r in self.results if r.task_completed]

        # Provider breakdown
        provider_counts = {}
        for r in self.results:
            provider_counts[r.llm_provider] = provider_counts.get(r.llm_provider, 0) + 1

        # Action distribution
        action_counts = {}
        for r in self.results:
            action_counts[r.action_chosen] = action_counts.get(r.action_chosen, 0) + 1

        # Spatial understanding accuracy
        correct_understanding = len([r for r in self.results if r.spatial_understanding == "correct"])

        return ExperimentSummary(
            total_trials=len(self.results),
            successful_trials=len(successful),
            success_rate=len(successful) / len(self.results),
            avg_decision_latency_ms=np.mean([r.decision_latency_ms for r in self.results]),
            avg_confidence=np.mean([r.decision_confidence for r in self.results]),
            avg_path_efficiency=np.mean([r.path_efficiency for r in self.results]),
            provider_breakdown=provider_counts,
            action_distribution=action_counts,
            spatial_understanding_accuracy=correct_understanding / len(self.results),
            collision_avoidance_rate=len([r for r in self.results if r.collision_avoided]) / len(self.results)
        )

    def _save_results(self, summary: ExperimentSummary):
        """Save experiment results to files"""

        # Save detailed results
        results_file = self.output_dir / f"experiment_{self.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Save summary
        summary_file = self.output_dir / f"experiment_{self.experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trials: {summary.total_trials}")
        logger.info(f"Successful: {summary.successful_trials} ({summary.success_rate*100:.1f}%)")
        logger.info(f"Avg Latency: {summary.avg_decision_latency_ms:.0f}ms")
        logger.info(f"Avg Confidence: {summary.avg_confidence:.2f}")
        logger.info(f"Avg Path Efficiency: {summary.avg_path_efficiency:.2f}")
        logger.info(f"Spatial Understanding Accuracy: {summary.spatial_understanding_accuracy*100:.1f}%")
        logger.info(f"\nProvider Breakdown: {summary.provider_breakdown}")
        logger.info(f"Action Distribution: {summary.action_distribution}")
        logger.info(f"\nResults saved to: {results_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"{'='*60}\n")


async def main():
    """Main experiment runner"""

    print("\n" + "="*60)
    print("SPATIAL LAB - REAL SPATIAL REASONING EXPERIMENT")
    print("="*60 + "\n")

    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if not groq_key and not gemini_key:
        print("ERROR: No API keys found. Set GROQ_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    print(f"Groq API: {'Available' if groq_key else 'Not available'}")
    print(f"Gemini API: {'Available' if gemini_key else 'Not available'}")
    print()

    # Run experiment
    experiment = SpatialReasoningExperiment()

    # Run with 20 trials (10 per provider if both available)
    num_trials = 20
    use_both = bool(groq_key and gemini_key)

    summary = await experiment.run_experiment(
        num_trials=num_trials,
        use_both_providers=use_both
    )

    print("\nExperiment completed!")
    print(f"Success Rate: {summary.success_rate*100:.1f}%")
    print(f"Results saved to: experiment_results/")

    return summary.success_rate >= 0.5  # Success if >50% correct


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
