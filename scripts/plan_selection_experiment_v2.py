#!/usr/bin/env python3
"""
Plan Selection Calibration Experiment v2

Key improvements over v1:
1. Unified scoring function (J) exposed to model
2. Potentially unsafe plans that can fail
3. Diversity filtering to remove degenerate trials
4. Calibrated confidence elicitation (0-1 with anchors)
5. Separate success_exec (safety) and correct_opt (optimality) metrics
6. Scaled to 360 trials for statistical power

Design:
- 3 information conditions × 4 complexity levels × 30 repetitions = 360 trials
- 120 trials per condition, sufficient for meaningful calibration analysis
"""

import asyncio
import json
import os
import sys
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_lab.planning.plan_generator_v2 import (
    MultiPlanGeneratorV2,
    PlanGenerationConfigV2,
    ScenarioResultV2,
    create_diverse_scenario,
    _create_randomized_scenario,
)
from spatial_lab.planning.execution_sim_v2 import (
    ExecutionSimulatorV2,
    ExecutionConfigV2,
    ExecutionResultV2,
    compute_trial_metrics,
)
from spatial_lab.planning.scoring import ScoringConfig, DEFAULT_SCORING
from spatial_lab.prompts.plan_selection_v2 import (
    PromptConditionV2,
    build_v2_prompt,
)
from spatial_lab.analysis import (
    CalibrationAnalyzer,
    CalibrationMetrics,
    create_reliability_diagram,
    create_calibration_summary,
    export_analysis_report,
)


@dataclass
class ExperimentConfigV2:
    """Configuration for v2 experiment."""
    name: str = "plan_selection_calibration_v2"

    # Experimental design
    complexity_levels: Tuple[str, ...] = ("simple", "moderate", "complex", "challenging")
    prompt_conditions: Tuple[str, ...] = ("minimal", "numeric", "failure_augmented")
    repetitions: int = 30  # Per cell → 4 × 3 × 30 = 360 total

    # Randomization
    random_seed: int = 42

    # LLM settings
    llm_provider: str = "groq"  # "groq" or "gemini"
    llm_model: str = "llama-3.3-70b-versatile"  # or "gemini-2.0-flash-exp"
    max_retries: int = 2
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.5  # Slightly longer to avoid rate limits

    # Execution settings
    enable_noisy_execution: bool = True

    # Scoring (exposed to model)
    scoring_config: ScoringConfig = None

    def __post_init__(self):
        if self.scoring_config is None:
            self.scoring_config = DEFAULT_SCORING

    @property
    def total_trials(self) -> int:
        return (
            len(self.complexity_levels) *
            len(self.prompt_conditions) *
            self.repetitions
        )


class PlanSelectionExperimentV2:
    """
    Main experiment runner for v2 plan selection calibration.
    """

    def __init__(self, config: ExperimentConfigV2):
        self.config = config

        # Initialize components
        self.plan_generator = MultiPlanGeneratorV2(
            PlanGenerationConfigV2(
                include_risky_plan=True,
                require_diversity=True,
                scoring_config=config.scoring_config,
                use_letter_ids=True,
                randomize_order=True,
            )
        )

        self.executor = ExecutionSimulatorV2(
            ExecutionConfigV2(
                enable_noise=config.enable_noisy_execution,
                scoring_config=config.scoring_config,
            )
        )

        self.analyzer = CalibrationAnalyzer(n_bins=10, use_adaptive_binning=True)

        # Results storage
        self.trials: List[Dict[str, Any]] = []
        self.api_failures: List[Dict[str, Any]] = []
        self.scenario_failures: List[Dict[str, Any]] = []

        random.seed(config.random_seed)

    async def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        print(f"\n{'='*70}")
        print(f"Plan Selection Calibration Experiment v2")
        print(f"{'='*70}")
        print(f"Design: {len(self.config.complexity_levels)} complexity × "
              f"{len(self.config.prompt_conditions)} conditions × "
              f"{self.config.repetitions} reps = {self.config.total_trials} trials")
        print(f"Scoring: {self.config.scoring_config.get_formula_string()}")
        print(f"LLM: {self.config.llm_provider} / {self.config.llm_model}")
        print(f"{'='*70}\n")

        # Initialize LLM client based on provider
        try:
            if self.config.llm_provider == "gemini":
                from spatial_lab.llm import GeminiAPIConfig, GeminiAPIClient
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                llm_config = GeminiAPIConfig(
                    api_key=api_key,
                    model=self.config.llm_model,
                )
                client_class = GeminiAPIClient
            else:  # groq
                from spatial_lab.llm import GroqAPIConfig, GroqAPIClient
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                llm_config = GroqAPIConfig(
                    api_key=api_key,
                    model=self.config.llm_model,
                )
                client_class = GroqAPIClient
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return {"error": str(e)}

        # Run trials
        trial_num = 0
        async with client_class(llm_config) as client:
            for complexity in self.config.complexity_levels:
                for prompt_cond in self.config.prompt_conditions:
                    for rep in range(self.config.repetitions):
                        trial_num += 1
                        seed = self.config.random_seed + trial_num

                        # Progress indicator (every 10 trials)
                        if trial_num % 10 == 0 or trial_num == 1:
                            print(f"\nTrial {trial_num}/{self.config.total_trials}: "
                                  f"{complexity}/{prompt_cond}")

                        try:
                            result = await self._run_single_trial(
                                client=client,
                                complexity=complexity,
                                prompt_condition=prompt_cond,
                                repetition=rep,
                                seed=seed,
                            )

                            if result.get("scenario_failure"):
                                self.scenario_failures.append(result)
                            elif result.get("api_error"):
                                self.api_failures.append(result)
                                print(f"  [!] API: {result.get('error_message', 'Unknown')[:50]}")
                            else:
                                self.trials.append(result)
                                # Brief status
                                s = "S" if result.get("success_exec") else "F"
                                o = "O" if result.get("correct_opt") else "X"
                                c = result.get("confidence", 0)
                                if trial_num % 10 == 0:
                                    print(f"  [{s}{o}] conf={c:.2f}")

                        except Exception as e:
                            error_result = {
                                "trial_num": trial_num,
                                "complexity": complexity,
                                "prompt_condition": prompt_cond,
                                "repetition": rep,
                                "api_error": True,
                                "error_message": str(e),
                            }
                            self.api_failures.append(error_result)

                        # Rate limiting
                        await asyncio.sleep(self.config.rate_limit_delay)

        # Analyze results
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print(f"{'='*70}")

        analysis = self._analyze_results()

        # Save results
        output_dir = Path("experiment_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f"plan_selection_v2_{timestamp}_results.json"
        report_path = output_dir / f"plan_selection_v2_{timestamp}_report.md"

        full_results = {
            "config": self._config_to_dict(),
            "timestamp": timestamp,
            "trials": self.trials,
            "api_failures": self.api_failures,
            "scenario_failures": self.scenario_failures,
            "analysis": analysis,
        }

        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

        # Generate report
        if self.trials:
            self._generate_report(report_path, analysis)
            print(f"Report saved to: {report_path}")

        return full_results

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dict."""
        return {
            "name": self.config.name,
            "complexity_levels": list(self.config.complexity_levels),
            "prompt_conditions": list(self.config.prompt_conditions),
            "repetitions": self.config.repetitions,
            "random_seed": self.config.random_seed,
            "llm_model": self.config.llm_model,
            "enable_noisy_execution": self.config.enable_noisy_execution,
            "scoring_config": self.config.scoring_config.to_dict(),
            "total_trials": self.config.total_trials,
        }

    async def _run_single_trial(
        self,
        client: Any,
        complexity: str,
        prompt_condition: str,
        repetition: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single trial."""

        trial_id = f"{complexity}_{prompt_condition}_{repetition}"

        # Generate diverse scenario (may take multiple attempts)
        scenario_params = _create_randomized_scenario(complexity, seed=seed)

        scenario = self.plan_generator.generate_plans(
            start=scenario_params["start"],
            goal=scenario_params["goal"],
            obstacles=scenario_params["obstacles"],
            warehouse_dims=scenario_params["warehouse_dims"],
            scenario_id=f"{complexity}_{seed}",
        )

        if not scenario.generation_successful:
            return {
                "trial_id": trial_id,
                "complexity": complexity,
                "prompt_condition": prompt_condition,
                "repetition": repetition,
                "scenario_failure": True,
                "failure_reason": scenario.failure_reason,
            }

        # Build prompt
        prompt_cond_enum = PromptConditionV2(prompt_condition)
        prompt = build_v2_prompt(
            scenario=scenario,
            condition=prompt_cond_enum,
            scoring_config=self.config.scoring_config,
        )

        # Call LLM
        try:
            response = await self._call_llm_with_retry(
                client=client,
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
            )
        except Exception as e:
            return {
                "trial_id": trial_id,
                "complexity": complexity,
                "prompt_condition": prompt_condition,
                "repetition": repetition,
                "api_error": True,
                "error_message": str(e),
            }

        # Parse response
        parsed = self._parse_llm_response(response)

        if parsed is None:
            return {
                "trial_id": trial_id,
                "complexity": complexity,
                "prompt_condition": prompt_condition,
                "repetition": repetition,
                "api_error": True,
                "error_message": "Failed to parse LLM response",
                "raw_response": response[:500],
            }

        chosen_plan = parsed.get("chosen_plan", "")
        confidence = parsed.get("confidence", 0.5)
        rationale = parsed.get("rationale", "")

        # Normalize confidence to [0, 1]
        if confidence > 1:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))

        # Validate plan selection
        valid_ids = scenario.plan_order + list(scenario.plan_id_mapping.values())
        if chosen_plan not in valid_ids:
            # Try to match
            matched = None
            for vid in valid_ids:
                if vid.lower() == chosen_plan.lower():
                    matched = vid
                    break
            if matched:
                chosen_plan = matched
            else:
                return {
                    "trial_id": trial_id,
                    "complexity": complexity,
                    "prompt_condition": prompt_condition,
                    "repetition": repetition,
                    "api_error": True,
                    "error_message": f"Invalid plan: {chosen_plan}",
                    "valid_plans": valid_ids,
                }

        # Execute selected plan
        exec_result, validation = self.executor.execute_scenario(
            scenario=scenario,
            selected_plan_id=chosen_plan,
            deterministic=not self.config.enable_noisy_execution,
        )

        # Compute metrics
        metrics = compute_trial_metrics(confidence, exec_result)

        return {
            "trial_id": trial_id,
            "complexity": complexity,
            "prompt_condition": prompt_condition,
            "repetition": repetition,
            "scenario_id": scenario.scenario_id,
            # Selection
            "chosen_plan": chosen_plan,
            "optimal_plan": scenario.optimal_plan_id,
            "confidence": confidence,
            "rationale": rationale,
            # Primary metrics
            "success_exec": exec_result.success_exec,
            "correct_opt": exec_result.correct_opt,
            "regret": exec_result.regret,
            # Execution details
            "collision_occurred": exec_result.collision_occurred,
            "reached_goal": exec_result.reached_goal,
            "path_completion": exec_result.path_completion,
            "J_score": exec_result.J_score,
            "optimal_J": scenario.optimal_J,
            # Diversity
            "is_diverse": scenario.is_diverse,
            "diversity_stats": scenario.diversity_stats,
            # Flags
            "api_error": False,
            "scenario_failure": False,
        }

    async def _call_llm_with_retry(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call LLM with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Gemini doesn't support json_mode, so pass it conditionally
                if self.config.llm_provider == "gemini":
                    response = await client.spatial_reasoning_completion(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                    )
                else:
                    response = await client.spatial_reasoning_completion(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                    )

                if isinstance(response, dict):
                    # Gemini format: candidates[0].content.parts[0].text
                    candidates = response.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            return parts[0].get("text", "")

                    # Groq format: choices[0].message.content
                    choices = response.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")

                return str(response)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        import re

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        match = re.search(r'\{[^{}]*"chosen_plan"[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        if not self.trials:
            return {"error": "No successful trials"}

        # Extract calibration data
        confidences = [t["confidence"] for t in self.trials]
        success_exec = [int(t["success_exec"]) for t in self.trials]
        correct_opt = [int(t["correct_opt"]) for t in self.trials]

        # Overall metrics - Safety calibration
        safety_metrics = self.analyzer.analyze(confidences, success_exec)
        print("\n" + "="*50)
        print("SAFETY CALIBRATION (success_exec)")
        print("="*50)
        print(create_calibration_summary(safety_metrics, include_interpretation=False))

        # Overall metrics - Optimality calibration
        opt_metrics = self.analyzer.analyze(confidences, correct_opt)
        print("\n" + "="*50)
        print("OPTIMALITY CALIBRATION (correct_opt)")
        print("="*50)
        print(create_calibration_summary(opt_metrics, include_interpretation=False))

        # Per-condition analysis
        print("\n" + "-"*50)
        print("BY INFORMATION CONDITION")
        condition_results = {}
        for cond in self.config.prompt_conditions:
            cond_trials = [t for t in self.trials if t["prompt_condition"] == cond]
            if cond_trials:
                confs = [t["confidence"] for t in cond_trials]
                safety = [int(t["success_exec"]) for t in cond_trials]
                opt = [int(t["correct_opt"]) for t in cond_trials]

                safety_m = self.analyzer.analyze(confs, safety)
                opt_m = self.analyzer.analyze(confs, opt)

                print(f"  {cond}:")
                print(f"    Safety:  ECE={safety_m.ece:.3f}, Acc={safety_m.mean_accuracy:.1%}")
                print(f"    Optimal: ECE={opt_m.ece:.3f}, Acc={opt_m.mean_accuracy:.1%}")
                print(f"    Confidence: mean={safety_m.mean_confidence:.2f}")

                condition_results[cond] = {
                    "safety": safety_m.to_dict(),
                    "optimality": opt_m.to_dict(),
                    "n_trials": len(cond_trials),
                }

        # By complexity
        print("\n" + "-"*50)
        print("BY COMPLEXITY LEVEL")
        complexity_results = {}
        for comp in self.config.complexity_levels:
            comp_trials = [t for t in self.trials if t["complexity"] == comp]
            if comp_trials:
                confs = [t["confidence"] for t in comp_trials]
                safety = [int(t["success_exec"]) for t in comp_trials]

                safety_m = self.analyzer.analyze(confs, safety)
                print(f"  {comp}: ECE={safety_m.ece:.3f}, "
                      f"Acc={safety_m.mean_accuracy:.1%}, "
                      f"Conf={safety_m.mean_confidence:.2f}")

                complexity_results[comp] = safety_m.to_dict()

        # Confidence distribution
        print("\n" + "-"*50)
        print("CONFIDENCE DISTRIBUTION")
        conf_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
        for i in range(len(conf_bins) - 1):
            count = sum(1 for c in confidences if conf_bins[i] <= c < conf_bins[i+1])
            pct = count / len(confidences) * 100
            print(f"  [{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}): {count:3d} ({pct:5.1f}%)")

        # Summary stats
        n_success = sum(success_exec)
        n_optimal = sum(correct_opt)
        api_rate = len(self.trials) / (len(self.trials) + len(self.api_failures))

        print(f"\n" + "-"*50)
        print(f"SUMMARY")
        print(f"  Successful trials: {len(self.trials)}")
        print(f"  API failures: {len(self.api_failures)}")
        print(f"  Scenario failures: {len(self.scenario_failures)}")
        print(f"  API success rate: {api_rate:.1%}")
        print(f"  Execution success rate: {n_success/len(self.trials):.1%}")
        print(f"  Optimal selection rate: {n_optimal/len(self.trials):.1%}")

        return {
            "safety_calibration": safety_metrics.to_dict(),
            "optimality_calibration": opt_metrics.to_dict(),
            "by_condition": condition_results,
            "by_complexity": complexity_results,
            "api_success_rate": api_rate,
            "execution_success_rate": n_success / len(self.trials),
            "optimal_selection_rate": n_optimal / len(self.trials),
            "n_trials": len(self.trials),
            "n_api_failures": len(self.api_failures),
            "n_scenario_failures": len(self.scenario_failures),
        }

    def _generate_report(self, path: Path, analysis: Dict[str, Any]):
        """Generate markdown report."""
        safety = analysis.get("safety_calibration", {})
        opt = analysis.get("optimality_calibration", {})

        report = f"""# Plan Selection Calibration Experiment v2

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

| Metric | Safety (success_exec) | Optimality (correct_opt) |
|--------|----------------------|--------------------------|
| ECE | {safety.get('ece', 'N/A'):.4f} | {opt.get('ece', 'N/A'):.4f} |
| Accuracy | {safety.get('mean_accuracy', 0):.1%} | {opt.get('mean_accuracy', 0):.1%} |
| Mean Confidence | {safety.get('mean_confidence', 0):.2f} | {opt.get('mean_confidence', 0):.2f} |
| Overconfidence | {safety.get('overconfidence_rate', 0):.1%} | {opt.get('overconfidence_rate', 0):.1%} |

## Experiment Design

- **Total trials:** {analysis.get('n_trials', 0)}
- **Conditions:** {list(self.config.prompt_conditions)}
- **Complexity levels:** {list(self.config.complexity_levels)}
- **Scoring:** {self.config.scoring_config.get_formula_string()}

## Key Findings

### Information Condition Effects

"""
        by_cond = analysis.get("by_condition", {})
        for cond, data in by_cond.items():
            safety_data = data.get("safety", {})
            report += f"**{cond}:** ECE={safety_data.get('ece', 0):.3f}, "
            report += f"Acc={safety_data.get('mean_accuracy', 0):.1%}\n\n"

        report += """
## Methodology

- Unified scoring function J exposed to model
- Potentially unsafe plans included
- Diversity filtering applied
- Calibrated confidence elicitation with anchors
- Noisy execution simulation

---
*Generated by Spatial Lab v2*
"""

        with open(path, "w") as f:
            f.write(report)


async def main():
    """Run the v2 experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run v2 plan selection calibration experiment")
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq",
                        help="LLM provider (default: groq)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default depends on provider)")
    parser.add_argument("--reps", type=int, default=30,
                        help="Repetitions per cell (default: 30 for 360 total)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--complexity", type=str, default="simple,moderate,complex,challenging",
                        help="Comma-separated complexity levels")
    parser.add_argument("--conditions", type=str, default="minimal,numeric,failure_augmented",
                        help="Comma-separated prompt conditions")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Rate limit delay in seconds (default: 1.5, use 7.0 for Gemini free tier)")

    args = parser.parse_args()

    # Set default model based on provider
    if args.model is None:
        if args.provider == "gemini":
            model = "gemini-2.0-flash-exp"
        else:
            model = "llama-3.3-70b-versatile"
    else:
        model = args.model

    complexity_levels = tuple(args.complexity.split(","))
    prompt_conditions = tuple(args.conditions.split(","))

    config = ExperimentConfigV2(
        name="plan_selection_calibration_v2",
        complexity_levels=complexity_levels,
        prompt_conditions=prompt_conditions,
        repetitions=args.reps,
        random_seed=args.seed,
        enable_noisy_execution=True,
        llm_provider=args.provider,
        llm_model=model,
        rate_limit_delay=args.delay,
    )

    experiment = PlanSelectionExperimentV2(config)
    results = await experiment.run()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
