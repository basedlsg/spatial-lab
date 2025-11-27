#!/usr/bin/env python3
"""
Spatial Lab - LLM Confidence Calibration Experiment

Research Question: Are LLM-reported confidence scores well-calibrated
predictors of spatial reasoning accuracy?

Hypothesis: LLMs exhibit overconfidence - reporting high confidence even
when spatial accuracy is low, especially for obstacle avoidance tasks.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import numpy as np
from scipy import stats

sys.path.insert(0, '.')
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

COMPLEXITY_LEVELS = ["simple", "moderate", "complex", "very_complex"]
PROMPT_CONDITIONS = ["standard", "uncertainty_aware", "self_critique"]
DISTANCE_CATEGORIES = ["short", "medium", "long"]

PROMPT_TEMPLATES = {
    "standard": """You are a robot coordination specialist. Make an optimal decision.
Respond with JSON containing: action, parameters, reasoning, confidence (0-1).""",

    "uncertainty_aware": """You are a robot coordination specialist. Make an optimal decision.

IMPORTANT: Be HONEST about your uncertainty.
- If you're unsure about obstacle avoidance or complex navigation, report LOW confidence (<0.5)
- Only report high confidence (>0.8) if you're CERTAIN the path is clear
- Consider what could go wrong before rating confidence

Respond with JSON containing: action, parameters, reasoning, confidence (0-1).""",

    "self_critique": """You are a robot coordination specialist.

BEFORE deciding, identify potential failure modes:
1. Are there obstacles in the direct path?
2. Could I collide with other robots?
3. Is the target position clearly defined?

Then make your decision and rate confidence based on the likelihood of these failures.

Respond with JSON containing: action, parameters, reasoning, confidence (0-1)."""
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CalibrationTrial:
    """Single trial result for calibration study"""
    trial_id: int
    complexity: str
    prompt_condition: str
    distance_category: str

    # Scenario
    robot_position: List[float]
    target_position: List[float]
    obstacles: List[Dict]
    nearby_robots: List[Dict]
    optimal_distance: float

    # LLM Response
    action: str
    destination: List[float]
    confidence: float
    reasoning: str

    # Evaluation
    spatial_accuracy: bool
    destination_error: float
    path_efficiency: float
    mentions_obstacles: bool

    # Metadata
    latency_ms: int
    timestamp: str
    model: str = "llama-3.3-70b-versatile"


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics"""
    expected_calibration_error: float
    brier_score: float
    overconfidence_rate: float
    mean_confidence: float
    mean_accuracy: float
    confidence_accuracy_correlation: float
    n_trials: int


# ============================================================================
# EXPERIMENT CLASS
# ============================================================================

class CalibrationExperiment:
    """LLM Confidence Calibration Experiment"""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.output_dir = Path("experiment_results")
        self.output_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[CalibrationTrial] = []

        from spatial_lab.llm.groq_client import GroqAPIConfig, GroqAPIClient
        self.GroqAPIConfig = GroqAPIConfig
        self.GroqAPIClient = GroqAPIClient

        logger.info(f"Calibration Experiment {self.experiment_id} initialized")

    def generate_scenarios(self, trials_per_condition: int = 4) -> List[Dict]:
        """Generate experimental scenarios with factorial design"""

        np.random.seed(42)
        scenarios = []
        trial_id = 0

        # Generate all combinations
        for complexity in COMPLEXITY_LEVELS:
            for prompt_cond in PROMPT_CONDITIONS:
                for distance_cat in DISTANCE_CATEGORIES:
                    for rep in range(trials_per_condition):
                        trial_id += 1
                        scenario = self._create_scenario(
                            trial_id, complexity, prompt_cond, distance_cat
                        )
                        scenarios.append(scenario)

        # Randomize order
        np.random.shuffle(scenarios)
        return scenarios

    def _create_scenario(
        self,
        trial_id: int,
        complexity: str,
        prompt_cond: str,
        distance_cat: str
    ) -> Dict:
        """Create a single experimental scenario"""

        # Distance ranges
        distance_ranges = {
            "short": (3, 6),
            "medium": (7, 12),
            "long": (13, 18)
        }
        min_dist, max_dist = distance_ranges[distance_cat]

        # Robot position (avoid edges)
        rx = np.random.uniform(2, 18)
        ry = np.random.uniform(2, 18)

        # Target at specified distance
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(min_dist, max_dist)
        tx = rx + dist * np.cos(angle)
        ty = ry + dist * np.sin(angle)

        # Clamp to bounds
        tx = np.clip(tx, 1, 19)
        ty = np.clip(ty, 1, 19)

        # Recalculate actual distance
        optimal_dist = np.sqrt((tx - rx)**2 + (ty - ry)**2)

        # Generate obstacles based on complexity
        obstacles = []
        nearby_robots = []

        if complexity == "moderate":
            # Add a waypoint constraint (described in task, no physical obstacle)
            pass

        elif complexity == "complex":
            # Single obstacle between robot and target
            ox = (rx + tx) / 2 + np.random.uniform(-1, 1)
            oy = (ry + ty) / 2 + np.random.uniform(-1, 1)
            obstacles = [{"position": [round(ox, 1), round(oy, 1)], "radius": 1.5}]

        elif complexity == "very_complex":
            # Multiple obstacles + nearby robot
            ox1 = (rx + tx) / 2 + np.random.uniform(-2, 0)
            oy1 = (ry + ty) / 2 + np.random.uniform(-1, 1)
            ox2 = (rx + tx) / 2 + np.random.uniform(0, 2)
            oy2 = (ry + ty) / 2 + np.random.uniform(-1, 1)
            obstacles = [
                {"position": [round(ox1, 1), round(oy1, 1)], "radius": 1.2},
                {"position": [round(ox2, 1), round(oy2, 1)], "radius": 1.2}
            ]
            nearby_robots = [{
                "robot_id": "robot_nearby",
                "position": [round(rx + np.random.uniform(-3, 3), 1),
                            round(ry + np.random.uniform(-3, 3), 1), 0.0],
                "status": "moving"
            }]

        # Build task description
        task_desc = self._build_task_description(
            complexity, tx, ty, obstacles, nearby_robots
        )

        return {
            "trial_id": trial_id,
            "complexity": complexity,
            "prompt_condition": prompt_cond,
            "distance_category": distance_cat,
            "robot_position": [round(rx, 1), round(ry, 1), 0.0],
            "target_position": [round(tx, 1), round(ty, 1)],
            "obstacles": obstacles,
            "nearby_robots": nearby_robots,
            "optimal_distance": round(optimal_dist, 2),
            "task_description": task_desc
        }

    def _build_task_description(
        self,
        complexity: str,
        tx: float,
        ty: float,
        obstacles: List[Dict],
        nearby_robots: List[Dict]
    ) -> str:
        """Build task description based on complexity"""

        base = f"Navigate to target position ({tx:.1f}, {ty:.1f})."

        if complexity == "simple":
            return base

        elif complexity == "moderate":
            waypoint_x = np.random.uniform(3, 17)
            waypoint_y = np.random.uniform(3, 17)
            return f"{base} First pass through waypoint ({waypoint_x:.1f}, {waypoint_y:.1f})."

        elif complexity == "complex":
            obs = obstacles[0]
            return f"{base} WARNING: There is an obstacle at ({obs['position'][0]}, {obs['position'][1]}) with radius {obs['radius']}. You must navigate AROUND it."

        elif complexity == "very_complex":
            obs_desc = ", ".join([f"({o['position'][0]}, {o['position'][1]})" for o in obstacles])
            robot_pos = nearby_robots[0]["position"]
            return f"{base} WARNING: Obstacles at {obs_desc}. Another robot is at ({robot_pos[0]}, {robot_pos[1]}) and moving. Avoid collisions."

        return base

    async def run_trial(self, scenario: Dict) -> CalibrationTrial:
        """Run a single calibration trial"""

        start_time = time.time()

        config = self.GroqAPIConfig(
            api_key=self.api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            timeout=30
        )

        # Get prompt template for this condition
        system_prompt = PROMPT_TEMPLATES[scenario["prompt_condition"]]

        # Build observation
        observation = {
            "position": scenario["robot_position"],
            "battery_level": 0.95,
            "carrying_item": None,
            "status": "idle"
        }

        # Build layout
        layout = {
            "dimensions": [20, 20],
            "target": {"position": scenario["target_position"]},
            "obstacles": scenario["obstacles"]
        }

        try:
            async with self.GroqAPIClient(config) as client:
                # Custom prompt injection
                original_prompt = client._get_robot_coordination_system_prompt
                client._get_robot_coordination_system_prompt = lambda: system_prompt

                response = await client.robot_coordination_decision(
                    robot_id=f"robot_{scenario['trial_id']:03d}",
                    observation=observation,
                    task_description=scenario["task_description"],
                    available_actions=["move_to", "wait"],
                    nearby_robots=scenario["nearby_robots"],
                    warehouse_layout=layout
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            decision = self._parse_response(response)

            # Extract values
            action = decision.get("action", "unknown")
            params = decision.get("parameters", {})
            dest = params.get("destination", [0, 0])
            confidence = float(decision.get("confidence", 0.5))
            reasoning = decision.get("reasoning", "")[:300]

            # Ensure dest is a list
            if not isinstance(dest, list) or len(dest) < 2:
                dest = [0, 0]

            # Calculate spatial accuracy
            target = scenario["target_position"]
            dest_error = np.sqrt((dest[0] - target[0])**2 + (dest[1] - target[1])**2)
            spatial_accuracy = dest_error < 2.0  # Within 2 units = correct

            # Path efficiency
            actual_dist = np.sqrt(
                (dest[0] - scenario["robot_position"][0])**2 +
                (dest[1] - scenario["robot_position"][1])**2
            )
            path_efficiency = min(1.0, scenario["optimal_distance"] / max(actual_dist, 0.1))

            # Check if reasoning mentions obstacles
            mentions_obstacles = any(word in reasoning.lower()
                                    for word in ["obstacle", "avoid", "around", "blocked", "collision"])

            return CalibrationTrial(
                trial_id=scenario["trial_id"],
                complexity=scenario["complexity"],
                prompt_condition=scenario["prompt_condition"],
                distance_category=scenario["distance_category"],
                robot_position=scenario["robot_position"],
                target_position=scenario["target_position"],
                obstacles=scenario["obstacles"],
                nearby_robots=scenario["nearby_robots"],
                optimal_distance=scenario["optimal_distance"],
                action=action,
                destination=dest,
                confidence=confidence,
                reasoning=reasoning,
                spatial_accuracy=spatial_accuracy,
                destination_error=round(dest_error, 2),
                path_efficiency=round(path_efficiency, 3),
                mentions_obstacles=mentions_obstacles,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Trial {scenario['trial_id']} failed: {e}")
            return CalibrationTrial(
                trial_id=scenario["trial_id"],
                complexity=scenario["complexity"],
                prompt_condition=scenario["prompt_condition"],
                distance_category=scenario["distance_category"],
                robot_position=scenario["robot_position"],
                target_position=scenario["target_position"],
                obstacles=scenario["obstacles"],
                nearby_robots=scenario["nearby_robots"],
                optimal_distance=scenario["optimal_distance"],
                action="error",
                destination=[0, 0],
                confidence=0.0,
                reasoning=str(e),
                spatial_accuracy=False,
                destination_error=999.0,
                path_efficiency=0.0,
                mentions_obstacles=False,
                latency_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.now().isoformat()
            )

    def _parse_response(self, response: Dict) -> Dict:
        """Parse LLM response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        return {"action": "unknown", "confidence": 0.5}

    async def run_experiment(self, trials_per_condition: int = 4) -> Dict:
        """Run full calibration experiment"""

        # Generate scenarios
        scenarios = self.generate_scenarios(trials_per_condition)
        total_trials = len(scenarios)

        logger.info(f"\n{'='*70}")
        logger.info("LLM CONFIDENCE CALIBRATION EXPERIMENT")
        logger.info(f"{'='*70}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Total Trials: {total_trials}")
        logger.info(f"Design: {len(COMPLEXITY_LEVELS)} complexity × {len(PROMPT_CONDITIONS)} prompts × {len(DISTANCE_CATEGORIES)} distances × {trials_per_condition} reps")
        logger.info(f"{'='*70}\n")

        # Run trials
        for i, scenario in enumerate(scenarios):
            logger.info(f"Trial {i+1}/{total_trials}: {scenario['complexity']} | {scenario['prompt_condition']} | {scenario['distance_category']}")

            result = await self.run_trial(scenario)
            self.results.append(result)

            status = "CORRECT" if result.spatial_accuracy else "WRONG"
            logger.info(f"  -> {status} | Conf: {result.confidence:.2f} | Error: {result.destination_error:.1f}u | Latency: {result.latency_ms}ms")

            await asyncio.sleep(0.25)  # Rate limiting

        # Calculate metrics
        metrics = self._calculate_calibration_metrics()
        analysis = self._statistical_analysis()

        # Save results
        self._save_results(metrics, analysis)

        return {"metrics": metrics, "analysis": analysis}

    def _calculate_calibration_metrics(self) -> Dict[str, CalibrationMetrics]:
        """Calculate calibration metrics for each condition"""

        metrics = {}

        # Overall metrics
        metrics["overall"] = self._compute_metrics(self.results)

        # By complexity
        for complexity in COMPLEXITY_LEVELS:
            trials = [r for r in self.results if r.complexity == complexity]
            metrics[f"complexity_{complexity}"] = self._compute_metrics(trials)

        # By prompt condition
        for prompt in PROMPT_CONDITIONS:
            trials = [r for r in self.results if r.prompt_condition == prompt]
            metrics[f"prompt_{prompt}"] = self._compute_metrics(trials)

        return metrics

    def _compute_metrics(self, trials: List[CalibrationTrial]) -> CalibrationMetrics:
        """Compute calibration metrics for a set of trials"""

        if not trials:
            return CalibrationMetrics(0, 0, 0, 0, 0, 0, 0)

        confidences = [t.confidence for t in trials]
        accuracies = [1.0 if t.spatial_accuracy else 0.0 for t in trials]

        # Expected Calibration Error (ECE)
        # Bin into 5 bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        ece = 0.0
        for i in range(len(bins) - 1):
            mask = [(bins[i] <= c < bins[i+1]) for c in confidences]
            bin_trials = [(c, a) for c, a, m in zip(confidences, accuracies, mask) if m]
            if bin_trials:
                bin_conf = np.mean([c for c, a in bin_trials])
                bin_acc = np.mean([a for c, a in bin_trials])
                bin_weight = len(bin_trials) / len(trials)
                ece += abs(bin_conf - bin_acc) * bin_weight

        # Brier Score
        brier = np.mean([(c - a)**2 for c, a in zip(confidences, accuracies)])

        # Overconfidence Rate
        overconfident = sum(1 for c, a in zip(confidences, accuracies) if c > 0.8 and a == 0)
        overconf_rate = overconfident / len(trials)

        # Correlation
        if len(set(confidences)) > 1 and len(set(accuracies)) > 1:
            corr, _ = stats.pearsonr(confidences, accuracies)
        else:
            corr = 0.0

        return CalibrationMetrics(
            expected_calibration_error=round(ece, 3),
            brier_score=round(brier, 3),
            overconfidence_rate=round(overconf_rate, 3),
            mean_confidence=round(np.mean(confidences), 3),
            mean_accuracy=round(np.mean(accuracies), 3),
            confidence_accuracy_correlation=round(corr, 3),
            n_trials=len(trials)
        )

    def _statistical_analysis(self) -> Dict:
        """Perform statistical tests"""

        analysis = {}

        # Test 1: Is calibration error significantly > 0?
        calibration_errors = [abs(t.confidence - (1.0 if t.spatial_accuracy else 0.0))
                             for t in self.results]
        t_stat, p_val = stats.ttest_1samp(calibration_errors, 0)
        cohens_d = np.mean(calibration_errors) / np.std(calibration_errors) if np.std(calibration_errors) > 0 else 0

        analysis["calibration_vs_zero"] = {
            "test": "one-sample t-test",
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 6),
            "cohens_d": round(cohens_d, 3),
            "mean_error": round(np.mean(calibration_errors), 3),
            "significant": p_val < 0.05
        }

        # Test 2: ANOVA across complexity levels
        complexity_groups = {
            level: [abs(t.confidence - (1.0 if t.spatial_accuracy else 0.0))
                   for t in self.results if t.complexity == level]
            for level in COMPLEXITY_LEVELS
        }

        f_stat, p_val = stats.f_oneway(*complexity_groups.values())

        # Effect size (eta-squared)
        all_errors = [e for errors in complexity_groups.values() for e in errors]
        ss_total = sum((e - np.mean(all_errors))**2 for e in all_errors)
        ss_between = sum(len(errors) * (np.mean(errors) - np.mean(all_errors))**2
                        for errors in complexity_groups.values())
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        analysis["complexity_effect"] = {
            "test": "one-way ANOVA",
            "f_statistic": round(f_stat, 3),
            "p_value": round(p_val, 6),
            "eta_squared": round(eta_squared, 3),
            "significant": p_val < 0.05,
            "group_means": {k: round(np.mean(v), 3) for k, v in complexity_groups.items()}
        }

        # Test 3: Prompt condition comparison
        prompt_groups = {
            prompt: [abs(t.confidence - (1.0 if t.spatial_accuracy else 0.0))
                    for t in self.results if t.prompt_condition == prompt]
            for prompt in PROMPT_CONDITIONS
        }

        f_stat, p_val = stats.f_oneway(*prompt_groups.values())

        analysis["prompt_effect"] = {
            "test": "one-way ANOVA",
            "f_statistic": round(f_stat, 3),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "group_means": {k: round(np.mean(v), 3) for k, v in prompt_groups.items()}
        }

        # Test 4: Confidence-accuracy correlation by complexity
        for level in COMPLEXITY_LEVELS:
            trials = [t for t in self.results if t.complexity == level]
            confs = [t.confidence for t in trials]
            accs = [1.0 if t.spatial_accuracy else 0.0 for t in trials]

            if len(set(confs)) > 1 and len(set(accs)) > 1:
                corr, p_val = stats.pearsonr(confs, accs)
            else:
                corr, p_val = 0, 1

            analysis[f"correlation_{level}"] = {
                "pearson_r": round(corr, 3),
                "p_value": round(p_val, 6),
                "n": len(trials)
            }

        return analysis

    def _save_results(self, metrics: Dict, analysis: Dict):
        """Save all results and generate report"""

        # Save raw results
        results_file = self.output_dir / f"calibration_{self.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)

        # Save metrics
        metrics_file = self.output_dir / f"calibration_{self.experiment_id}_metrics.json"
        metrics_serializable = {
            k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
            for k, v in metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        # Save analysis
        analysis_file = self.output_dir / f"calibration_{self.experiment_id}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Print comprehensive report
        self._print_report(metrics, analysis)

    def _print_report(self, metrics: Dict, analysis: Dict):
        """Print formatted experiment report"""

        print("\n" + "="*70)
        print("LLM CONFIDENCE CALIBRATION - EXPERIMENT RESULTS")
        print("="*70)

        overall = metrics["overall"]
        print(f"\nOVERALL CALIBRATION METRICS (N={overall.n_trials})")
        print("-"*50)
        print(f"  Expected Calibration Error (ECE): {overall.expected_calibration_error:.3f}")
        print(f"  Brier Score:                      {overall.brier_score:.3f}")
        print(f"  Overconfidence Rate:              {overall.overconfidence_rate*100:.1f}%")
        print(f"  Mean Confidence:                  {overall.mean_confidence:.3f}")
        print(f"  Mean Accuracy:                    {overall.mean_accuracy*100:.1f}%")
        print(f"  Confidence-Accuracy Correlation:  {overall.confidence_accuracy_correlation:.3f}")

        print(f"\nCALIBRATION BY TASK COMPLEXITY")
        print("-"*50)
        print(f"{'Complexity':<15} {'ECE':>8} {'Accuracy':>10} {'Confidence':>12} {'Overconf':>10}")
        for level in COMPLEXITY_LEVELS:
            m = metrics[f"complexity_{level}"]
            print(f"{level:<15} {m.expected_calibration_error:>8.3f} {m.mean_accuracy*100:>9.1f}% {m.mean_confidence:>12.3f} {m.overconfidence_rate*100:>9.1f}%")

        print(f"\nCALIBRATION BY PROMPT CONDITION")
        print("-"*50)
        print(f"{'Prompt':<20} {'ECE':>8} {'Accuracy':>10} {'Confidence':>12}")
        for prompt in PROMPT_CONDITIONS:
            m = metrics[f"prompt_{prompt}"]
            print(f"{prompt:<20} {m.expected_calibration_error:>8.3f} {m.mean_accuracy*100:>9.1f}% {m.mean_confidence:>12.3f}")

        print(f"\nSTATISTICAL ANALYSIS")
        print("-"*50)

        cal_test = analysis["calibration_vs_zero"]
        sig = "*" if cal_test["significant"] else ""
        print(f"H1: Calibration Error > 0")
        print(f"    t({overall.n_trials-1}) = {cal_test['t_statistic']:.2f}, p = {cal_test['p_value']:.4f}{sig}")
        print(f"    Cohen's d = {cal_test['cohens_d']:.2f} ({'large' if abs(cal_test['cohens_d']) > 0.8 else 'medium' if abs(cal_test['cohens_d']) > 0.5 else 'small'})")

        comp_test = analysis["complexity_effect"]
        sig = "*" if comp_test["significant"] else ""
        print(f"\nH2: Complexity affects calibration")
        print(f"    F(3, {overall.n_trials-4}) = {comp_test['f_statistic']:.2f}, p = {comp_test['p_value']:.4f}{sig}")
        print(f"    η² = {comp_test['eta_squared']:.3f}")

        prompt_test = analysis["prompt_effect"]
        sig = "*" if prompt_test["significant"] else ""
        print(f"\nH3: Prompt condition affects calibration")
        print(f"    F(2, {overall.n_trials-3}) = {prompt_test['f_statistic']:.2f}, p = {prompt_test['p_value']:.4f}{sig}")

        print(f"\nCORRELATION BY COMPLEXITY")
        print("-"*50)
        for level in COMPLEXITY_LEVELS:
            corr_data = analysis[f"correlation_{level}"]
            sig = "*" if corr_data["p_value"] < 0.05 else ""
            print(f"  {level}: r = {corr_data['pearson_r']:.3f}, p = {corr_data['p_value']:.4f}{sig}")

        print(f"\nKEY FINDINGS")
        print("-"*50)

        # Determine key findings
        simple_acc = metrics["complexity_simple"].mean_accuracy
        complex_acc = metrics["complexity_complex"].mean_accuracy
        simple_conf = metrics["complexity_simple"].mean_confidence
        complex_conf = metrics["complexity_complex"].mean_confidence

        if complex_conf > 0.7 and complex_acc < 0.3:
            print(f"  1. SEVERE OVERCONFIDENCE in complex tasks:")
            print(f"     - Confidence: {complex_conf:.2f}, Accuracy: {complex_acc*100:.0f}%")
            print(f"     - LLM reports high confidence despite low accuracy")

        if simple_acc > 0.8:
            print(f"  2. GOOD CALIBRATION for simple tasks:")
            print(f"     - Confidence: {simple_conf:.2f}, Accuracy: {simple_acc*100:.0f}%")

        # Compare prompt conditions
        standard_ece = metrics["prompt_standard"].expected_calibration_error
        uncertainty_ece = metrics["prompt_uncertainty_aware"].expected_calibration_error
        if uncertainty_ece < standard_ece:
            improvement = (standard_ece - uncertainty_ece) / standard_ece * 100
            print(f"  3. UNCERTAINTY-AWARE prompts reduce calibration error by {improvement:.0f}%")

        print(f"\n{'='*70}")
        print(f"Results saved to: {self.output_dir}/calibration_{self.experiment_id}_*")
        print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "="*70)
    print("SPATIAL LAB - LLM CONFIDENCE CALIBRATION EXPERIMENT")
    print("="*70)
    print("\nResearch Question: Are LLM confidence scores well-calibrated")
    print("                   predictors of spatial reasoning accuracy?")
    print("="*70 + "\n")

    experiment = CalibrationExperiment()

    # Run with 4 trials per condition = 144 total trials
    # Takes ~3-5 minutes with rate limiting
    results = await experiment.run_experiment(trials_per_condition=4)

    return results


if __name__ == "__main__":
    asyncio.run(main())
