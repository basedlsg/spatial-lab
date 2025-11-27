#!/usr/bin/env python3
"""
Plan Selection Calibration Experiment

Redesigned experiment measuring LLM calibration for navigation plan selection.
The LLM chooses among pre-generated candidate plans (not generating coordinates).

Experimental Design:
- 4 complexity levels x 3 prompt conditions x 3 repetitions = 36 base trials
- Separated API failures from calibration analysis
- Proper collision checking and execution simulation

Author: Spatial Lab Team
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

from spatial_lab.planning import (
    MultiPlanGenerator,
    PlanGenerationConfig,
    CostFunction,
    ExecutionSimulator,
    create_test_scenario,
    evaluate_accuracy,
)
from spatial_lab.prompts import (
    PromptCondition,
    ObservabilityCondition,
    build_plan_selection_prompt,
)
from spatial_lab.analysis import (
    CalibrationAnalyzer,
    create_reliability_diagram,
    create_calibration_summary,
    export_analysis_report,
    separate_api_failures,
)


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    name: str = "plan_selection_calibration"
    complexity_levels: Tuple[str, ...] = ("simple", "moderate", "complex", "challenging")
    prompt_conditions: Tuple[str, ...] = ("minimal", "annotated", "failure_mode")
    observability: str = "full"
    repetitions: int = 3
    random_seed: int = 42
    llm_model: str = "llama-3.3-70b-versatile"
    max_retries: int = 2
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0


class PlanSelectionExperiment:
    """
    Main experiment runner for plan selection calibration.

    The LLM is presented with pre-generated candidate plans and must
    select the best one, providing confidence in its selection.
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment."""
        self.config = config
        self.plan_generator = MultiPlanGenerator(
            PlanGenerationConfig(
                grid_resolution=0.5,
                safety_margin=0.3,
                randomize_order=True,
                validate_plans=True,
            )
        )
        self.executor = ExecutionSimulator()
        self.analyzer = CalibrationAnalyzer(n_bins=10, use_adaptive_binning=True)

        # Results storage
        self.trials: List[Dict[str, Any]] = []
        self.api_failures: List[Dict[str, Any]] = []

        # Set random seed
        random.seed(config.random_seed)

    async def run(self) -> Dict[str, Any]:
        """
        Run the full experiment.

        Returns:
            Dictionary with results and analysis.
        """
        print(f"\n{'='*60}")
        print(f"Plan Selection Calibration Experiment")
        print(f"{'='*60}")
        print(f"Config: {self.config.name}")
        print(f"Complexity levels: {self.config.complexity_levels}")
        print(f"Prompt conditions: {self.config.prompt_conditions}")
        print(f"Repetitions per cell: {self.config.repetitions}")

        total_trials = (
            len(self.config.complexity_levels) *
            len(self.config.prompt_conditions) *
            self.config.repetitions
        )
        print(f"Total trials: {total_trials}")
        print(f"{'='*60}\n")

        # Initialize LLM client
        try:
            from spatial_lab.llm import GroqAPIConfig, GroqAPIClient
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("GROQ_API_KEY not set")

            llm_config = GroqAPIConfig(
                api_key=groq_key,
                model=self.config.llm_model,
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return {"error": str(e)}

        # Run trials
        trial_num = 0
        async with GroqAPIClient(llm_config) as client:
            for complexity in self.config.complexity_levels:
                for prompt_cond in self.config.prompt_conditions:
                    for rep in range(self.config.repetitions):
                        trial_num += 1
                        seed = self.config.random_seed + trial_num

                        print(f"\nTrial {trial_num}/{total_trials}: "
                              f"{complexity}/{prompt_cond}/rep{rep+1}")

                        try:
                            result = await self._run_single_trial(
                                client=client,
                                complexity=complexity,
                                prompt_condition=prompt_cond,
                                repetition=rep,
                                seed=seed,
                            )

                            if result.get("api_error"):
                                self.api_failures.append(result)
                                print(f"  API Error: {result.get('error_message', 'Unknown')}")
                            else:
                                self.trials.append(result)
                                correct = "Y" if result.get("selection_correct") else "N"
                                conf = result.get("confidence", 0)
                                print(f"  Correct: {correct}, Confidence: {conf:.0f}%")

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
                            print(f"  Exception: {e}")

                        # Rate limiting
                        await asyncio.sleep(self.config.rate_limit_delay)

        # Analyze results
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")

        analysis = self._analyze_results()

        # Save results
        output_dir = Path("experiment_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f"plan_selection_{timestamp}_results.json"
        report_path = output_dir / f"plan_selection_{timestamp}_report.md"

        full_results = {
            "config": asdict(self.config),
            "timestamp": timestamp,
            "trials": self.trials,
            "api_failures": self.api_failures,
            "analysis": analysis,
        }

        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

        # Generate report
        if self.trials:
            metrics = self.analyzer.analyze(
                confidences=[t["confidence"] for t in self.trials if "confidence" in t],
                accuracies=[int(t["selection_correct"]) for t in self.trials if "selection_correct" in t],
            )
            report = export_analysis_report(
                metrics=metrics,
                experiment_config=asdict(self.config),
                trials=self.trials,
                output_path=str(report_path),
            )
            print(f"Report saved to: {report_path}")

        return full_results

    async def _run_single_trial(
        self,
        client: Any,
        complexity: str,
        prompt_condition: str,
        repetition: int,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run a single experimental trial.

        Args:
            client: LLM client.
            complexity: Scenario complexity level.
            prompt_condition: Prompt information condition.
            repetition: Repetition number.
            seed: Random seed for scenario generation.

        Returns:
            Trial result dictionary.
        """
        # Generate scenario
        scenario_params = create_test_scenario(complexity=complexity, seed=seed)

        # Generate candidate plans
        scenario = self.plan_generator.generate_plans(
            start=scenario_params["start"],
            goal=scenario_params["goal"],
            obstacles=scenario_params["obstacles"],
            warehouse_dims=scenario_params["warehouse_dims"],
            scenario_id=f"{complexity}_{seed}",
        )

        if not scenario.generation_successful:
            return {
                "trial_id": f"{complexity}_{prompt_condition}_{repetition}",
                "complexity": complexity,
                "prompt_condition": prompt_condition,
                "repetition": repetition,
                "api_error": True,
                "error_message": f"Plan generation failed: {scenario.failure_reason}",
            }

        # Build prompt
        prompt_cond_enum = PromptCondition(prompt_condition)
        obs_enum = ObservabilityCondition(self.config.observability)

        prompt = build_plan_selection_prompt(
            scenario=scenario,
            prompt_condition=prompt_cond_enum,
            observability=obs_enum,
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
                "trial_id": f"{complexity}_{prompt_condition}_{repetition}",
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
                "trial_id": f"{complexity}_{prompt_condition}_{repetition}",
                "complexity": complexity,
                "prompt_condition": prompt_condition,
                "repetition": repetition,
                "api_error": True,
                "error_message": "Failed to parse LLM response",
                "raw_response": response,
            }

        selected_plan_id = parsed.get("selected_plan", "")
        confidence = parsed.get("confidence", 50)
        reasoning = parsed.get("reasoning", "")

        # Validate selection
        valid_plan_ids = [p.plan_id for p in scenario.candidate_plans]
        if selected_plan_id not in valid_plan_ids:
            # Try to match partial
            matched = None
            for pid in valid_plan_ids:
                if pid in selected_plan_id or selected_plan_id in pid:
                    matched = pid
                    break
            if matched:
                selected_plan_id = matched
            else:
                return {
                    "trial_id": f"{complexity}_{prompt_condition}_{repetition}",
                    "complexity": complexity,
                    "prompt_condition": prompt_condition,
                    "repetition": repetition,
                    "api_error": True,
                    "error_message": f"Invalid plan selection: {selected_plan_id}",
                    "valid_plans": valid_plan_ids,
                    "raw_response": response,
                }

        # Execute selected plan
        selected_plan = next(
            p for p in scenario.candidate_plans
            if p.plan_id == selected_plan_id
        )
        execution_result = self.executor.execute_plan(
            plan=selected_plan,
            obstacles=scenario_params["obstacles"],
        )

        # Compute accuracy metrics
        accuracy_metrics = evaluate_accuracy(
            selected_plan_id=selected_plan_id,
            optimal_plan_id=scenario.optimal_plan_id,
            execution_result=execution_result,
        )

        # Normalize confidence to 0-1
        if confidence > 1:
            confidence = confidence / 100.0

        return {
            "trial_id": f"{complexity}_{prompt_condition}_{repetition}",
            "complexity": complexity,
            "prompt_condition": prompt_condition,
            "repetition": repetition,
            "scenario_id": scenario.scenario_id,
            # Selection
            "selected_plan_id": selected_plan_id,
            "optimal_plan_id": scenario.optimal_plan_id,
            "selection_correct": accuracy_metrics["selection_correct"],
            # Confidence
            "confidence": confidence,
            "reasoning": reasoning,
            # Execution
            "execution_success": execution_result.success,
            "execution_status": execution_result.status,
            "reached_goal": execution_result.reached_goal,
            "collision_free": not execution_result.collision_occurred,
            "path_completion": execution_result.path_completion,
            # Combined
            "combined_accuracy": accuracy_metrics["combined_accuracy"],
            # Metadata
            "num_candidate_plans": len(scenario.candidate_plans),
            "plan_order": scenario.plan_order,
            "api_error": False,
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
                response = await client.spatial_reasoning_completion(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                )
                # Extract content from response
                if isinstance(response, dict):
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
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object anywhere
        json_match = re.search(r'\{[^{}]*"selected_plan"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        if not self.trials:
            return {"error": "No successful trials"}

        # Extract data for analysis
        confidences = [t["confidence"] for t in self.trials if "confidence" in t]
        accuracies = [int(t["selection_correct"]) for t in self.trials if "selection_correct" in t]

        # Overall metrics
        overall_metrics = self.analyzer.analyze(confidences, accuracies)

        print("\n" + create_calibration_summary(overall_metrics))
        print(create_reliability_diagram(overall_metrics))

        # Per-condition analysis
        condition_results = {}

        # By complexity
        complexity_metrics = self.analyzer.analyze_by_condition(
            self.trials, "complexity"
        )
        print("\n" + "-" * 40)
        print("BY COMPLEXITY LEVEL")
        for level, metrics in sorted(complexity_metrics.items()):
            print(f"  {level}: ECE={metrics.ece:.3f}, Acc={metrics.mean_accuracy:.1%}, "
                  f"Conf={metrics.mean_confidence:.1%}")
        condition_results["by_complexity"] = {
            k: v.to_dict() for k, v in complexity_metrics.items()
        }

        # By prompt condition
        prompt_metrics = self.analyzer.analyze_by_condition(
            self.trials, "prompt_condition"
        )
        print("\n" + "-" * 40)
        print("BY PROMPT CONDITION")
        for cond, metrics in sorted(prompt_metrics.items()):
            print(f"  {cond}: ECE={metrics.ece:.3f}, Acc={metrics.mean_accuracy:.1%}, "
                  f"Conf={metrics.mean_confidence:.1%}")
        condition_results["by_prompt"] = {
            k: v.to_dict() for k, v in prompt_metrics.items()
        }

        # API failure summary
        success_rate = len(self.trials) / (len(self.trials) + len(self.api_failures))
        print(f"\n" + "-" * 40)
        print(f"API Success Rate: {success_rate:.1%} "
              f"({len(self.trials)} successful, {len(self.api_failures)} failed)")

        return {
            "overall": overall_metrics.to_dict(),
            "by_condition": condition_results,
            "api_success_rate": success_rate,
            "n_successful_trials": len(self.trials),
            "n_api_failures": len(self.api_failures),
        }


async def main():
    """Run the experiment."""
    config = ExperimentConfig(
        name="plan_selection_calibration_v1",
        complexity_levels=("simple", "moderate", "complex", "challenging"),
        prompt_conditions=("minimal", "annotated", "failure_mode"),
        repetitions=3,
        random_seed=42,
    )

    experiment = PlanSelectionExperiment(config)
    results = await experiment.run()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    if "error" not in results:
        analysis = results.get("analysis", {})
        overall = analysis.get("overall", {})
        print(f"\nFinal ECE: {overall.get('ece', 'N/A')}")
        print(f"Final Accuracy: {overall.get('mean_accuracy', 'N/A')}")
        api_rate = analysis.get('api_success_rate')
        if api_rate is not None:
            print(f"API Success Rate: {api_rate:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
