"""
Plan Selection Prompt Templates v2

Key v2 changes:
1. Identical core instruction across all conditions
2. Only information content varies (not tone/framing)
3. Explicit J formula exposed to model
4. Calibrated confidence elicitation with anchors
5. Uses letter IDs (A, B, C) to prevent heuristics on names
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import json

from ..planning.plan_generator_v2 import ScenarioResultV2, PlanWithScore
from ..planning.scoring import ScoringConfig, DEFAULT_SCORING


class PromptConditionV2(Enum):
    """Information conditions for v2 experiment."""
    MINIMAL = "minimal"           # Qualitative only, no numbers except length
    NUMERIC = "numeric"           # Exact fields used in J (length, risk_score)
    FAILURE_AUGMENTED = "failure_augmented"  # Numeric + failure hints


# Unified response schema for all conditions
RESPONSE_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "chosen_plan": {
            "type": "string",
            "description": "The letter ID of your chosen plan (A, B, or C)"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Probability (0.0-1.0) that this is the best plan"
        },
        "rationale": {
            "type": "string",
            "description": "Brief explanation of your choice"
        }
    },
    "required": ["chosen_plan", "confidence", "rationale"]
}


# Core instruction - IDENTICAL across all conditions
CORE_INSTRUCTION = """You are selecting a navigation plan for a warehouse robot.

OBJECTIVE: Choose the plan that minimizes the score J = {alpha} × length + {beta} × risk_score
Lower J is better. The plan with the lowest J is the best choice.

Additionally, avoid plans that might collide with obstacles. Some plans pass very close to obstacles and may fail.

OUTPUT FORMAT:
Return a JSON object with:
- "chosen_plan": The letter (A, B, or C) of your chosen plan
- "confidence": Your confidence (0.0 to 1.0) that this is the best plan
- "rationale": Brief explanation

CONFIDENCE GUIDELINES:
- Use 0.5 when plans seem roughly equal or you're uncertain
- Use 0.3-0.4 when you have a slight preference but aren't sure
- Use 0.6-0.7 when one plan seems clearly better
- Use 0.8+ only when one plan is obviously superior (rare)
- Values above 0.9 should be very rare"""


# Calibration examples to anchor confidence
CALIBRATION_EXAMPLES = """
EXAMPLE 1 - Low confidence (similar plans):
Plans have lengths 25.0m, 25.5m, 26.0m and risks 0.3, 0.35, 0.3
These are very similar. Confidence should be around 0.4-0.5

EXAMPLE 2 - Medium confidence (one clearly better):
Plan A: length=30m, risk=0.2 → J = 3.0 + 0.3 = 3.3
Plan B: length=25m, risk=0.5 → J = 2.5 + 0.75 = 3.25
Plan B is slightly better. Confidence around 0.6

EXAMPLE 3 - High confidence (obvious winner):
Plan A: length=35m, risk=0.8 → J = 3.5 + 1.2 = 4.7
Plan B: length=22m, risk=0.2 → J = 2.2 + 0.3 = 2.5
Plan B is much better. Confidence around 0.85"""


@dataclass
class PromptConfigV2:
    """Configuration for v2 prompt generation."""
    condition: PromptConditionV2
    scoring_config: ScoringConfig = None
    include_calibration_examples: bool = True

    def __post_init__(self):
        if self.scoring_config is None:
            self.scoring_config = DEFAULT_SCORING


class PlanSelectionPromptBuilderV2:
    """Builds prompts for v2 plan selection experiment."""

    def __init__(self, config: Optional[PromptConfigV2] = None):
        self.config = config or PromptConfigV2(
            condition=PromptConditionV2.NUMERIC
        )

    def build_prompt(
        self,
        scenario: ScenarioResultV2,
    ) -> Dict[str, Any]:
        """
        Build a complete prompt for plan selection.

        Args:
            scenario: The navigation scenario with plans.

        Returns:
            Dictionary with 'system', 'user', and 'schema' keys.
        """
        # Build system prompt (same for all conditions)
        system = self._build_system_prompt()

        # Build user prompt (varies by condition)
        if self.config.condition == PromptConditionV2.MINIMAL:
            user = self._build_minimal_user_prompt(scenario)
        elif self.config.condition == PromptConditionV2.NUMERIC:
            user = self._build_numeric_user_prompt(scenario)
        elif self.config.condition == PromptConditionV2.FAILURE_AUGMENTED:
            user = self._build_failure_augmented_user_prompt(scenario)
        else:
            raise ValueError(f"Unknown condition: {self.config.condition}")

        return {
            "system": system,
            "user": user,
            "schema": RESPONSE_SCHEMA_V2,
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt with core instruction."""
        sc = self.config.scoring_config

        system = CORE_INSTRUCTION.format(
            alpha=sc.alpha,
            beta=sc.beta,
        )

        if self.config.include_calibration_examples:
            system += "\n\n" + CALIBRATION_EXAMPLES

        return system

    def _build_minimal_user_prompt(self, scenario: ScenarioResultV2) -> str:
        """Build minimal prompt - qualitative descriptions only."""

        lines = [
            "## Navigation Task",
            f"Navigate from ({scenario.start[0]:.1f}, {scenario.start[1]:.1f}) "
            f"to ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})",
            "",
            "## Available Plans",
        ]

        # Get plans in presentation order
        for letter_id in scenario.plan_order:
            plan_scored = scenario.get_plan_by_id(letter_id)
            if not plan_scored:
                continue

            meta = plan_scored.plan.metadata

            # Qualitative description only
            length_desc = self._describe_length(meta.total_length, scenario)
            safety_desc = self._describe_safety(meta.min_clearance)

            lines.append(f"**Plan {letter_id}**: {length_desc}, {safety_desc}")

        lines.extend([
            "",
            "Choose the best plan and provide your confidence (0.0-1.0).",
            "",
            "Respond with JSON: {\"chosen_plan\": \"X\", \"confidence\": 0.X, \"rationale\": \"...\"}",
        ])

        return "\n".join(lines)

    def _build_numeric_user_prompt(self, scenario: ScenarioResultV2) -> str:
        """Build numeric prompt - exact J components provided."""

        sc = self.config.scoring_config
        lines = [
            "## Navigation Task",
            f"Start: ({scenario.start[0]:.1f}, {scenario.start[1]:.1f})",
            f"Goal: ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})",
            "",
            f"Scoring: J = {sc.alpha} × length + {sc.beta} × risk_score (lower is better)",
            "",
            "## Candidate Plans",
            "",
            "| Plan | Length (m) | Risk Score | J Score |",
            "|------|------------|------------|---------|",
        ]

        for letter_id in scenario.plan_order:
            plan_scored = scenario.get_plan_by_id(letter_id)
            if not plan_scored:
                continue

            meta = plan_scored.plan.metadata
            J = plan_scored.J_score

            lines.append(
                f"| {letter_id} | {meta.total_length:.1f} | "
                f"{meta.risk_score:.2f} | {J:.2f} |"
            )

        lines.extend([
            "",
            "Choose the plan with the lowest J score while considering collision safety.",
            "",
            "Respond with JSON: {\"chosen_plan\": \"X\", \"confidence\": 0.X, \"rationale\": \"...\"}",
        ])

        return "\n".join(lines)

    def _build_failure_augmented_user_prompt(self, scenario: ScenarioResultV2) -> str:
        """Build failure-augmented prompt - numeric + failure hints."""

        sc = self.config.scoring_config
        lines = [
            "## Navigation Task",
            f"Start: ({scenario.start[0]:.1f}, {scenario.start[1]:.1f})",
            f"Goal: ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})",
            "",
            f"Scoring: J = {sc.alpha} × length + {sc.beta} × risk_score (lower is better)",
            "",
            "## Candidate Plans",
        ]

        for letter_id in scenario.plan_order:
            plan_scored = scenario.get_plan_by_id(letter_id)
            if not plan_scored:
                continue

            meta = plan_scored.plan.metadata
            J = plan_scored.J_score

            lines.append(f"")
            lines.append(f"**Plan {letter_id}**")
            lines.append(f"- Length: {meta.total_length:.1f}m")
            lines.append(f"- Risk score: {meta.risk_score:.2f}")
            lines.append(f"- J score: {J:.2f}")
            lines.append(f"- Min clearance: {meta.min_clearance:.2f}m")

            # Add failure hints (neutral tone)
            failure_hints = self._get_failure_hints(plan_scored)
            if failure_hints:
                lines.append(f"- Notes: {failure_hints}")

        lines.extend([
            "",
            "Choose the plan with the lowest J score while considering collision safety.",
            "Plans with very low clearance may fail during execution.",
            "",
            "Respond with JSON: {\"chosen_plan\": \"X\", \"confidence\": 0.X, \"rationale\": \"...\"}",
        ])

        return "\n".join(lines)

    def _describe_length(self, length: float, scenario: ScenarioResultV2) -> str:
        """Qualitative description of path length."""
        lengths = [p.plan.metadata.total_length for p in scenario.candidate_plans]
        min_len, max_len = min(lengths), max(lengths)

        if max_len - min_len < 2:
            return "moderate length"

        if length <= min_len + (max_len - min_len) * 0.33:
            return "shorter path"
        elif length >= min_len + (max_len - min_len) * 0.67:
            return "longer path"
        else:
            return "medium length"

    def _describe_safety(self, min_clearance: float) -> str:
        """Qualitative description of safety."""
        if min_clearance > 1.5:
            return "very safe (far from obstacles)"
        elif min_clearance > 1.0:
            return "safe path"
        elif min_clearance > 0.5:
            return "moderately safe"
        elif min_clearance > 0.3:
            return "passes near obstacles"
        else:
            return "risky (very close to obstacles)"

    def _get_failure_hints(self, plan: PlanWithScore) -> str:
        """Get neutral failure hints for a plan."""
        hints = []

        if plan.failure_probability > 0.3:
            hints.append("passes close to multiple obstacles")
        elif plan.failure_probability > 0.1:
            hints.append("some tight clearances")

        if plan.plan.metadata.min_clearance < 0.3:
            hints.append(f"narrowest gap is {plan.plan.metadata.min_clearance:.2f}m")

        if plan.is_potentially_unsafe:
            hints.append("may have difficulty in narrow sections")

        return "; ".join(hints) if hints else "standard clearances"


def build_v2_prompt(
    scenario: ScenarioResultV2,
    condition: PromptConditionV2 = PromptConditionV2.NUMERIC,
    scoring_config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to build a v2 prompt.

    Args:
        scenario: The navigation scenario.
        condition: Information condition.
        scoring_config: Scoring configuration.

    Returns:
        Dictionary with 'system', 'user', and 'schema' keys.
    """
    config = PromptConfigV2(
        condition=condition,
        scoring_config=scoring_config or DEFAULT_SCORING,
    )
    builder = PlanSelectionPromptBuilderV2(config)
    return builder.build_prompt(scenario)
