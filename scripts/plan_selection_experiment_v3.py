#!/usr/bin/env python3
"""
Plan Selection Calibration Experiment v3: Contextual Reasoning

Key changes from v2:
1. NO J formula given to LLM - only natural language mission briefs
2. Loops through 3 Mission Contexts per scenario (same plans, different briefs)
3. Tests semantic reasoning: Can the LLM infer correct cost trade-offs?
4. New metrics:
   - Contextual Alignment Score: Did model pick optimal for HIDDEN scoring?
   - Semantic Sensitivity: JS divergence of confidence across contexts

Design:
- For each generated scenario, present it under all 3 mission contexts
- Compare how model's choices and confidences shift with context
- This tests genuine spatial reasoning vs arithmetic calculation
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
    _create_randomized_scenario,
)
from spatial_lab.planning.scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    MISSION_CONTEXTS,
    MissionContext,
    get_mission_context,
    compute_J,
)
from spatial_lab.prompts.plan_selection_v3 import (
    InfoLevel,
    build_v3_prompt,
    compute_contextual_alignment,
    compute_semantic_sensitivity,
)
from spatial_lab.analysis import (
    CalibrationAnalyzer,
    CalibrationMetrics,
    create_calibration_summary,
)


@dataclass
class ExperimentConfigV3:
    """Configuration for v3 experiment."""
    name: str = "plan_selection_calibration_v3_contextual"

    # Experimental design
    complexity_levels: Tuple[str, ...] = ("simple", "moderate", "complex", "challenging")
    mission_contexts: Tuple[str, ...] = ("safety_critical", "time_critical", "balanced")
    info_level: str = "quantitative"  # quantitative or qualitative
    repetitions: int = 30  # Per complexity level

    # Randomization
    random_seed: int = 42

    # LLM settings
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    max_retries: int = 2
    retry_delay: float = 2.0
    rate_limit_delay: float = 7.0  # For Gemini free tier

    # Use DEFAULT_SCORING for plan generation only (diverse plans)
    generation_scoring: ScoringConfig = None

    def __post_init__(self):
        if self.generation_scoring is None:
            self.generation_scoring = DEFAULT_SCORING

    @property
    def total_scenarios(self) -> int:
        """Number of unique scenarios (before mission context multiplication)."""
        return len(self.complexity_levels) * self.repetitions

    @property
    def total_trials(self) -> int:
        """Total trials = scenarios × contexts."""
        return self.total_scenarios * len(self.mission_contexts)


class PlanSelectionExperimentV3:
    """
    V3 Experiment: Contextual Reasoning without explicit formula.

    For each scenario, we test all 3 mission contexts to measure:
    1. Does the model pick the optimal plan for each context's hidden scoring?
    2. Does the model's confidence distribution shift appropriately?
    """

    def __init__(self, config: ExperimentConfigV3):
        self.config = config

        # Plan generator uses balanced scoring for diversity
        self.plan_generator = MultiPlanGeneratorV2(
            PlanGenerationConfigV2(
                include_risky_plan=True,
                require_diversity=True,
                scoring_config=config.generation_scoring,
                use_letter_ids=True,
                randomize_order=True,
            )
        )

        self.analyzer = CalibrationAnalyzer(n_bins=10, use_adaptive_binning=True)

        # Results storage
        self.trials: List[Dict[str, Any]] = []
        self.api_failures: List[Dict[str, Any]] = []
        self.scenario_failures: List[Dict[str, Any]] = []

        # For semantic sensitivity analysis
        self.confidence_by_context: Dict[str, List[float]] = {
            ctx: [] for ctx in config.mission_contexts
        }

        random.seed(config.random_seed)

    async def run(self) -> Dict[str, Any]:
        """Run the full v3 experiment."""
        print(f"\n{'='*70}")
        print(f"Plan Selection Calibration Experiment v3: CONTEXTUAL REASONING")
        print(f"{'='*70}")
        print(f"Design: {len(self.config.complexity_levels)} complexity × "
              f"{self.config.repetitions} reps = {self.config.total_scenarios} scenarios")
        print(f"Each scenario tested with {len(self.config.mission_contexts)} mission contexts")
        print(f"Total trials: {self.config.total_trials}")
        print(f"LLM: {self.config.llm_provider} / {self.config.llm_model}")
        print(f"Info Level: {self.config.info_level}")
        print(f"\nMission Contexts (HIDDEN from LLM):")
        for ctx_name in self.config.mission_contexts:
            ctx = get_mission_context(ctx_name)
            sc = ctx.scoring_config
            print(f"  - {ctx_name}: J = {sc.alpha}×Length + {sc.beta}×Risk")
        print(f"{'='*70}\n")

        # Initialize LLM client
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
        scenario_num = 0
        trial_num = 0

        async with client_class(llm_config) as client:
            for complexity in self.config.complexity_levels:
                for rep in range(self.config.repetitions):
                    scenario_num += 1
                    seed = self.config.random_seed + scenario_num

                    # Generate scenario once
                    scenario_params = _create_randomized_scenario(complexity, seed=seed)
                    scenario = self.plan_generator.generate_plans(
                        start=scenario_params["start"],
                        goal=scenario_params["goal"],
                        obstacles=scenario_params["obstacles"],
                        warehouse_dims=scenario_params["warehouse_dims"],
                        scenario_id=f"{complexity}_{seed}",
                    )

                    if not scenario.generation_successful:
                        self.scenario_failures.append({
                            "scenario_num": scenario_num,
                            "complexity": complexity,
                            "repetition": rep,
                            "failure_reason": scenario.failure_reason,
                        })
                        continue

                    # Progress indicator
                    if scenario_num % 10 == 0 or scenario_num == 1:
                        print(f"\nScenario {scenario_num}/{self.config.total_scenarios}: {complexity}")

                    # Test this scenario with all mission contexts
                    scenario_results = []
                    for ctx_name in self.config.mission_contexts:
                        trial_num += 1
                        mission_ctx = get_mission_context(ctx_name)

                        try:
                            result = await self._run_context_trial(
                                client=client,
                                scenario=scenario,
                                mission_context=mission_ctx,
                                complexity=complexity,
                                repetition=rep,
                                scenario_num=scenario_num,
                            )

                            if result.get("api_error"):
                                self.api_failures.append(result)
                                print(f"  [{ctx_name[:4]}] API Error")
                            else:
                                self.trials.append(result)
                                scenario_results.append(result)

                                # Track confidence for semantic sensitivity
                                self.confidence_by_context[ctx_name].append(
                                    result.get("confidence", 0.5)
                                )

                                # Brief status
                                aligned = "A" if result.get("aligned") else "X"
                                conf = result.get("confidence", 0)
                                chosen = result.get("chosen_plan", "?")
                                optimal = result.get("optimal_plan", "?")
                                if scenario_num % 10 == 0:
                                    print(f"  [{ctx_name[:4]}] {aligned} chose={chosen} "
                                          f"opt={optimal} conf={conf:.2f}")

                        except Exception as e:
                            self.api_failures.append({
                                "scenario_num": scenario_num,
                                "context": ctx_name,
                                "api_error": True,
                                "error_message": str(e),
                            })

                        # Rate limiting between context calls
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
        results_path = output_dir / f"plan_selection_v3_{timestamp}_results.json"
        report_path = output_dir / f"plan_selection_v3_{timestamp}_report.md"

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
            "mission_contexts": list(self.config.mission_contexts),
            "info_level": self.config.info_level,
            "repetitions": self.config.repetitions,
            "random_seed": self.config.random_seed,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "total_scenarios": self.config.total_scenarios,
            "total_trials": self.config.total_trials,
            "generation_scoring": self.config.generation_scoring.to_dict(),
        }

    async def _run_context_trial(
        self,
        client: Any,
        scenario: ScenarioResultV2,
        mission_context: MissionContext,
        complexity: str,
        repetition: int,
        scenario_num: int,
    ) -> Dict[str, Any]:
        """Run a single trial with a specific mission context."""

        trial_id = f"{scenario_num}_{mission_context.name}"

        # Get plans from scenario
        plans = []
        for plan_id in scenario.plan_order:
            # Use get_plan_by_id to handle letter ID to original ID mapping
            plan_with_score = scenario.get_plan_by_id(plan_id)
            if plan_with_score:
                metadata = plan_with_score.plan.metadata
                plans.append({
                    "plan_id": plan_id,
                    "length": metadata.total_length,
                    "risk_score": metadata.risk_score,
                    "min_clearance": metadata.min_clearance,
                })

        # Build V3 prompt (NO FORMULA - only mission brief)
        info_level = InfoLevel(self.config.info_level)
        prompt = build_v3_prompt(
            mission_context=mission_context,
            plans=plans,
            plan_order=scenario.plan_order,
            info_level=info_level,
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
                "context": mission_context.name,
                "complexity": complexity,
                "api_error": True,
                "error_message": str(e),
            }

        # Parse response
        parsed = self._parse_llm_response(response)

        if parsed is None:
            return {
                "trial_id": trial_id,
                "context": mission_context.name,
                "complexity": complexity,
                "api_error": True,
                "error_message": "Failed to parse LLM response",
                "raw_response": response[:500] if response else "",
            }

        chosen_plan = parsed.get("chosen_plan", "")
        confidence = parsed.get("confidence", 0.5)
        rationale = parsed.get("rationale", "")

        # Normalize confidence
        if confidence > 1:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))

        # Validate plan selection
        valid_ids = scenario.plan_order
        if chosen_plan not in valid_ids:
            # Try case-insensitive match
            for vid in valid_ids:
                if vid.lower() == chosen_plan.lower():
                    chosen_plan = vid
                    break
            else:
                return {
                    "trial_id": trial_id,
                    "context": mission_context.name,
                    "complexity": complexity,
                    "api_error": True,
                    "error_message": f"Invalid plan: {chosen_plan}",
                    "valid_plans": valid_ids,
                }

        # Compute contextual alignment (vs HIDDEN scoring function)
        alignment = compute_contextual_alignment(
            chosen_plan_id=chosen_plan,
            plans=plans,
            mission_context=mission_context,
        )

        return {
            "trial_id": trial_id,
            "scenario_num": scenario_num,
            "context": mission_context.name,
            "complexity": complexity,
            "repetition": repetition,
            # LLM response
            "chosen_plan": chosen_plan,
            "confidence": confidence,
            "rationale": rationale,
            # Alignment metrics
            "aligned": alignment["aligned"],
            "optimal_plan": alignment["optimal_plan"],
            "chosen_J": alignment["chosen_J"],
            "optimal_J": alignment["optimal_J"],
            "regret": alignment["regret"],
            # Plan details for analysis
            "plans": plans,
            "plan_order": scenario.plan_order,
            # Flags
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
                )

                if isinstance(response, dict):
                    # Gemini format
                    candidates = response.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            return parts[0].get("text", "")

                    # Groq format
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

        if not response:
            return None

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
        """Analyze v3 experiment results."""
        if not self.trials:
            return {"error": "No successful trials"}

        # === CONTEXTUAL ALIGNMENT BY CONTEXT ===
        print("\n" + "="*50)
        print("CONTEXTUAL ALIGNMENT BY MISSION CONTEXT")
        print("="*50)
        print("(Did the LLM pick the optimal plan for each context's HIDDEN scoring?)\n")

        context_results = {}
        for ctx_name in self.config.mission_contexts:
            ctx_trials = [t for t in self.trials if t["context"] == ctx_name]
            if ctx_trials:
                n_aligned = sum(1 for t in ctx_trials if t.get("aligned", False))
                alignment_rate = n_aligned / len(ctx_trials)
                mean_regret = sum(t.get("regret", 0) for t in ctx_trials) / len(ctx_trials)
                mean_conf = sum(t.get("confidence", 0) for t in ctx_trials) / len(ctx_trials)

                ctx = get_mission_context(ctx_name)
                sc = ctx.scoring_config
                print(f"{ctx_name}:")
                print(f"  Hidden J: {sc.alpha}×Length + {sc.beta}×Risk")
                print(f"  Alignment: {alignment_rate:.1%} ({n_aligned}/{len(ctx_trials)})")
                print(f"  Mean Regret: {mean_regret:.3f}")
                print(f"  Mean Confidence: {mean_conf:.2f}")
                print()

                context_results[ctx_name] = {
                    "alignment_rate": alignment_rate,
                    "n_aligned": n_aligned,
                    "n_trials": len(ctx_trials),
                    "mean_regret": round(mean_regret, 4),
                    "mean_confidence": round(mean_conf, 3),
                    "hidden_alpha": sc.alpha,
                    "hidden_beta": sc.beta,
                }

        # === SEMANTIC SENSITIVITY ===
        print("="*50)
        print("SEMANTIC SENSITIVITY (Jensen-Shannon Divergence)")
        print("="*50)
        print("(Do confidence distributions shift when mission context changes?)\n")

        sensitivity = compute_semantic_sensitivity(self.confidence_by_context)
        for key, value in sensitivity.items():
            print(f"  {key}: {value:.4f}")
        print()

        # === CALIBRATION PER CONTEXT ===
        print("="*50)
        print("CALIBRATION BY CONTEXT")
        print("="*50)

        calibration_results = {}
        for ctx_name in self.config.mission_contexts:
            ctx_trials = [t for t in self.trials if t["context"] == ctx_name]
            if ctx_trials:
                confs = [t["confidence"] for t in ctx_trials]
                aligned = [int(t.get("aligned", False)) for t in ctx_trials]

                metrics = self.analyzer.analyze(confs, aligned)
                print(f"\n{ctx_name}:")
                print(f"  ECE: {metrics.ece:.4f}")
                print(f"  Brier: {metrics.brier_score:.4f}")
                print(f"  Mean Acc (aligned): {metrics.mean_accuracy:.1%}")
                print(f"  Mean Conf: {metrics.mean_confidence:.2f}")
                print(f"  Overconfidence: {metrics.overconfidence_rate:.1%}")

                calibration_results[ctx_name] = metrics.to_dict()

        # === CHOICE CONSISTENCY ===
        print("\n" + "="*50)
        print("CHOICE CONSISTENCY ACROSS CONTEXTS")
        print("="*50)
        print("(Does the model appropriately change its choice based on context?)\n")

        # Group trials by scenario
        scenarios = {}
        for trial in self.trials:
            snum = trial.get("scenario_num")
            if snum not in scenarios:
                scenarios[snum] = {}
            scenarios[snum][trial["context"]] = trial["chosen_plan"]

        # Count how often choices differ between contexts
        n_complete = 0
        n_all_same = 0
        n_all_different = 0
        choice_patterns = {}

        for snum, choices in scenarios.items():
            if len(choices) == len(self.config.mission_contexts):
                n_complete += 1
                unique_choices = set(choices.values())

                if len(unique_choices) == 1:
                    n_all_same += 1
                elif len(unique_choices) == len(self.config.mission_contexts):
                    n_all_different += 1

                # Track pattern
                pattern = tuple(choices.get(ctx, "?") for ctx in self.config.mission_contexts)
                choice_patterns[pattern] = choice_patterns.get(pattern, 0) + 1

        if n_complete > 0:
            print(f"Scenarios with all 3 contexts: {n_complete}")
            print(f"  Same choice for all contexts: {n_all_same} ({n_all_same/n_complete:.1%})")
            print(f"  Different choice for all contexts: {n_all_different} ({n_all_different/n_complete:.1%})")
            print(f"\nTop choice patterns (safety, time, balanced):")
            sorted_patterns = sorted(choice_patterns.items(), key=lambda x: -x[1])[:5]
            for pattern, count in sorted_patterns:
                print(f"  {pattern}: {count}")

        consistency_stats = {
            "n_complete_scenarios": n_complete,
            "n_all_same_choice": n_all_same,
            "n_all_different_choice": n_all_different,
            "same_choice_rate": n_all_same / n_complete if n_complete > 0 else 0,
        }

        # === OVERALL SUMMARY ===
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)

        total_aligned = sum(1 for t in self.trials if t.get("aligned", False))
        overall_alignment = total_aligned / len(self.trials) if self.trials else 0
        overall_regret = sum(t.get("regret", 0) for t in self.trials) / len(self.trials)
        overall_conf = sum(t.get("confidence", 0) for t in self.trials) / len(self.trials)

        print(f"\nTotal trials: {len(self.trials)}")
        print(f"API failures: {len(self.api_failures)}")
        print(f"Scenario failures: {len(self.scenario_failures)}")
        print(f"\nOverall Contextual Alignment: {overall_alignment:.1%}")
        print(f"Overall Mean Regret: {overall_regret:.3f}")
        print(f"Overall Mean Confidence: {overall_conf:.2f}")
        print(f"Semantic Sensitivity (mean JS divergence): {sensitivity.get('mean_sensitivity', 0):.4f}")

        return {
            "by_context": context_results,
            "semantic_sensitivity": sensitivity,
            "calibration_by_context": calibration_results,
            "choice_consistency": consistency_stats,
            "overall_alignment": overall_alignment,
            "overall_mean_regret": round(overall_regret, 4),
            "overall_mean_confidence": round(overall_conf, 3),
            "n_trials": len(self.trials),
            "n_api_failures": len(self.api_failures),
            "n_scenario_failures": len(self.scenario_failures),
        }

    def _generate_report(self, path: Path, analysis: Dict[str, Any]):
        """Generate markdown report for v3 experiment."""
        by_ctx = analysis.get("by_context", {})
        sensitivity = analysis.get("semantic_sensitivity", {})
        consistency = analysis.get("choice_consistency", {})

        report = f"""# Plan Selection Calibration Experiment v3: Contextual Reasoning

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {self.config.llm_model}

## Executive Summary

This experiment tests whether the LLM can infer correct cost trade-offs from
**natural language mission briefs** without being given an explicit formula.

### Key Finding

| Mission Context | Alignment Rate | Mean Regret | Mean Confidence |
|----------------|---------------|-------------|-----------------|
"""
        for ctx_name, data in by_ctx.items():
            report += f"| {ctx_name} | {data.get('alignment_rate', 0):.1%} | "
            report += f"{data.get('mean_regret', 0):.3f} | "
            report += f"{data.get('mean_confidence', 0):.2f} |\n"

        report += f"""
**Overall Contextual Alignment:** {analysis.get('overall_alignment', 0):.1%}
**Semantic Sensitivity (mean JS divergence):** {sensitivity.get('mean_sensitivity', 0):.4f}

## Experimental Design

### The "Semantics-to-Cost Gap"

In V2, the LLM was given the explicit formula `J = α×Length + β×Risk`.
This made plan selection an arithmetic task, not a reasoning task.

In V3, the LLM only sees a **natural language mission brief** that describes
the priorities (e.g., "transporting nitroglycerin - avoid ALL collisions").
The model must **infer** the correct trade-off from semantic context.

### Mission Contexts (Hidden Scoring)

| Context | Brief Summary | Hidden Scoring |
|---------|--------------|----------------|
| safety_critical | Nitroglycerin transport | J = 0.0×L + 5.0×R |
| time_critical | Medical emergency | J = 1.0×L + 0.1×R |
| balanced | Standard delivery | J = 0.1×L + 1.5×R |

### Design Parameters

- **Complexity levels:** {list(self.config.complexity_levels)}
- **Repetitions per level:** {self.config.repetitions}
- **Total scenarios:** {self.config.total_scenarios}
- **Total trials:** {self.config.total_trials} (scenarios × 3 contexts)

## Semantic Sensitivity Analysis

Do the model's confidence distributions shift when the mission context changes?

| Context Pair | JS Divergence |
|-------------|---------------|
"""
        for key, value in sensitivity.items():
            if key != "mean_sensitivity":
                report += f"| {key} | {value:.4f} |\n"

        report += f"""
**Mean Sensitivity:** {sensitivity.get('mean_sensitivity', 0):.4f}

Higher JS divergence indicates the model appropriately adjusts its confidence
based on mission context. Low divergence suggests context-insensitivity.

## Choice Consistency Analysis

"""
        if consistency:
            report += f"""
- Scenarios with all 3 contexts completed: {consistency.get('n_complete_scenarios', 0)}
- Same choice across all contexts: {consistency.get('n_all_same_choice', 0)} ({consistency.get('same_choice_rate', 0):.1%})

If the model always picks the same plan regardless of context, it's not
performing contextual reasoning - it's defaulting to a fixed heuristic.
"""

        report += """
## Interpretation

### What "Alignment" Means

A trial is "aligned" if the model's chosen plan matches the optimal plan
according to that context's **hidden** scoring function. This is the key
metric for V3 - it tests whether the model can translate natural language
priorities into correct cost trade-offs.

### Expected Behavior

An ideal model would show:
1. **High alignment** across all contexts (correctly inferring trade-offs)
2. **High semantic sensitivity** (confidence shifts with context)
3. **Low same-choice rate** (appropriately changing selections)

### Potential Failure Modes

1. **Context-blind**: Same choice regardless of brief → semantic reasoning failure
2. **Overconfidence in ambiguous contexts**: High confidence even when alignment is low
3. **Asymmetric reasoning**: Good at "prioritize safety" but poor at "prioritize speed"

---
*Generated by Spatial Lab v3 - Contextual Reasoning Experiment*
"""

        with open(path, "w") as f:
            f.write(report)


async def main():
    """Run the v3 experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run v3 plan selection calibration experiment (contextual reasoning)"
    )
    parser.add_argument("--provider", choices=["groq", "gemini"], default="gemini",
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default depends on provider)")
    parser.add_argument("--reps", type=int, default=30,
                        help="Repetitions per complexity level (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--complexity", type=str, default="simple,moderate,complex,challenging",
                        help="Comma-separated complexity levels")
    parser.add_argument("--contexts", type=str, default="safety_critical,time_critical,balanced",
                        help="Comma-separated mission contexts")
    parser.add_argument("--info-level", choices=["quantitative", "qualitative"],
                        default="quantitative",
                        help="How much numeric info to give (default: quantitative)")
    parser.add_argument("--delay", type=float, default=7.0,
                        help="Rate limit delay in seconds (default: 7.0 for Gemini free tier)")

    args = parser.parse_args()

    # Set default model based on provider
    if args.model is None:
        if args.provider == "gemini":
            model = "gemini-2.0-flash"
        else:
            model = "llama-3.3-70b-versatile"
    else:
        model = args.model

    complexity_levels = tuple(args.complexity.split(","))
    mission_contexts = tuple(args.contexts.split(","))

    config = ExperimentConfigV3(
        name="plan_selection_calibration_v3_contextual",
        complexity_levels=complexity_levels,
        mission_contexts=mission_contexts,
        info_level=args.info_level,
        repetitions=args.reps,
        random_seed=args.seed,
        llm_provider=args.provider,
        llm_model=model,
        rate_limit_delay=args.delay,
    )

    experiment = PlanSelectionExperimentV3(config)
    results = await experiment.run()

    print("\n" + "="*70)
    print("V3 EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
