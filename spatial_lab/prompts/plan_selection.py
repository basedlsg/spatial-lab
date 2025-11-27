"""
Plan Selection Prompt Templates

Implements three prompt conditions for the plan selection experiment:
1. Minimal: Basic task description only
2. Annotated: Full metadata for each plan
3. Failure-mode: Explicit warnings about collision risks
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import json
import random

from ..planning import CandidatePlan, ScenarioResult


class PromptCondition(Enum):
    """Prompt information density conditions."""
    MINIMAL = "minimal"
    ANNOTATED = "annotated"
    FAILURE_MODE = "failure_mode"


class ObservabilityCondition(Enum):
    """Information observability conditions."""
    FULL = "full"           # All obstacle positions known
    PARTIAL = "partial"     # Some obstacles hidden
    NOISY = "noisy"         # Positions have noise added
    DYNAMIC = "dynamic"     # Some obstacles may move


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    prompt_condition: PromptCondition
    observability: ObservabilityCondition
    include_confidence_request: bool = True
    confidence_scale: str = "0-100"  # or "0.0-1.0"


# JSON schema for structured output
PLAN_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_plan": {
            "type": "string",
            "description": "The plan_id of your selected plan"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Your confidence in this selection (0-100)"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of your choice"
        }
    },
    "required": ["selected_plan", "confidence", "reasoning"]
}


class PlanSelectionPromptBuilder:
    """Builds prompts for plan selection tasks."""

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize the builder.

        Args:
            config: Prompt configuration.
        """
        self.config = config or PromptConfig(
            prompt_condition=PromptCondition.ANNOTATED,
            observability=ObservabilityCondition.FULL,
        )

    def build_prompt(
        self,
        scenario: ScenarioResult,
        plan_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete prompt for plan selection.

        Args:
            scenario: The navigation scenario.
            plan_order: Order to present plans (uses scenario order if None).

        Returns:
            Dictionary with 'system', 'user', and 'schema' keys.
        """
        # Get plans in presentation order
        if plan_order is None:
            plan_order = scenario.plan_order

        ordered_plans = []
        plan_map = {p.plan_id: p for p in scenario.candidate_plans}
        for plan_id in plan_order:
            if plan_id in plan_map:
                ordered_plans.append(plan_map[plan_id])

        # Apply observability transformations
        visible_obstacles = self._apply_observability(
            scenario.obstacles,
            scenario.warehouse_dims
        )

        # Build prompt based on condition
        if self.config.prompt_condition == PromptCondition.MINIMAL:
            return self._build_minimal_prompt(
                scenario, ordered_plans, visible_obstacles
            )
        elif self.config.prompt_condition == PromptCondition.ANNOTATED:
            return self._build_annotated_prompt(
                scenario, ordered_plans, visible_obstacles
            )
        elif self.config.prompt_condition == PromptCondition.FAILURE_MODE:
            return self._build_failure_mode_prompt(
                scenario, ordered_plans, visible_obstacles
            )

    def _apply_observability(
        self,
        obstacles: List[Dict[str, Any]],
        warehouse_dims: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Apply observability condition to obstacles."""
        if self.config.observability == ObservabilityCondition.FULL:
            return obstacles

        elif self.config.observability == ObservabilityCondition.PARTIAL:
            # Hide 30% of obstacles
            visible = []
            for i, obs in enumerate(obstacles):
                if i % 3 != 0:  # Keep 2/3 visible
                    visible.append(obs)
            return visible

        elif self.config.observability == ObservabilityCondition.NOISY:
            # Add position noise
            noisy = []
            for obs in obstacles:
                pos = obs.get("position", obs.get("pos", [0, 0]))
                noisy_pos = [
                    pos[0] + random.gauss(0, 0.5),
                    pos[1] + random.gauss(0, 0.5)
                ]
                noisy_obs = obs.copy()
                noisy_obs["position"] = noisy_pos
                noisy.append(noisy_obs)
            return noisy

        elif self.config.observability == ObservabilityCondition.DYNAMIC:
            # Mark some obstacles as potentially moving
            dynamic = []
            for i, obs in enumerate(obstacles):
                obs_copy = obs.copy()
                if i % 4 == 0:  # 25% are dynamic
                    obs_copy["dynamic"] = True
                    obs_copy["velocity_estimate"] = [
                        random.uniform(-0.2, 0.2),
                        random.uniform(-0.2, 0.2)
                    ]
                dynamic.append(obs_copy)
            return dynamic

        return obstacles

    def _build_minimal_prompt(
        self,
        scenario: ScenarioResult,
        plans: List[CandidatePlan],
        obstacles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build minimal information prompt."""
        system = """You are a warehouse robot navigation system.
Select the best plan from the given options.
Respond with valid JSON only."""

        user = f"""Navigate from {scenario.start} to {scenario.goal}.

Available plans:
"""
        for i, plan in enumerate(plans, 1):
            user += f"\n{i}. Plan ID: {plan.plan_id}"
            # Minimal: only show plan ID and basic path
            user += f"\n   Path: {len(plan.path)} waypoints"

        user += """

Select the best plan and provide your confidence (0-100).

Respond with JSON:
{"selected_plan": "<plan_id>", "confidence": <0-100>, "reasoning": "<brief explanation>"}"""

        return {
            "system": system,
            "user": user,
            "schema": PLAN_SELECTION_SCHEMA,
        }

    def _build_annotated_prompt(
        self,
        scenario: ScenarioResult,
        plans: List[CandidatePlan],
        obstacles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build fully annotated prompt with metadata."""
        system = """You are a warehouse robot navigation system.
Your task is to select the optimal navigation plan based on safety and efficiency.

Consider:
- Path length (shorter is generally better)
- Clearance from obstacles (more clearance is safer)
- Risk score (lower is better)
- Estimated time

Respond with valid JSON only."""

        user = f"""## Navigation Task

**Start Position:** ({scenario.start[0]:.1f}, {scenario.start[1]:.1f})
**Goal Position:** ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})
**Warehouse Size:** {scenario.warehouse_dims[0]:.0f}m x {scenario.warehouse_dims[1]:.0f}m

### Known Obstacles ({len(obstacles)} total):
"""
        for obs in obstacles[:5]:  # Limit to first 5 for readability
            pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0)
            user += f"- Position: ({pos[0]:.1f}, {pos[1]:.1f}), Radius: {radius:.1f}m\n"
        if len(obstacles) > 5:
            user += f"- ... and {len(obstacles) - 5} more obstacles\n"

        user += "\n### Candidate Plans:\n"
        for i, plan in enumerate(plans, 1):
            meta = plan.metadata
            user += f"""
**Plan {i}: {plan.plan_id}**
- Strategy: {meta.cost_function.value}
- Total length: {meta.total_length:.1f}m
- Minimum clearance: {meta.min_clearance:.1f}m
- Average clearance: {meta.avg_clearance:.1f}m
- Risk score: {meta.risk_score:.2f} (0=safe, 1=risky)
- Estimated time: {meta.estimated_time:.1f}s
- Waypoints: {meta.num_waypoints}
"""

        user += """
### Your Task
Select the best plan considering both safety and efficiency.
Provide your confidence (0-100) in this being the optimal choice.

Respond with JSON:
{"selected_plan": "<plan_id>", "confidence": <0-100>, "reasoning": "<brief explanation>"}"""

        return {
            "system": system,
            "user": user,
            "schema": PLAN_SELECTION_SCHEMA,
        }

    def _build_failure_mode_prompt(
        self,
        scenario: ScenarioResult,
        plans: List[CandidatePlan],
        obstacles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build prompt with explicit failure mode warnings."""
        system = """You are a warehouse robot navigation system.
CRITICAL: Collisions cause costly damage and operational delays.

Your primary objective is SAFE navigation. A slightly longer but safer path
is strongly preferred over a short risky path.

IMPORTANT WARNINGS:
- Plans with min_clearance < 0.5m have HIGH collision risk
- Risk scores > 0.5 indicate dangerous paths
- Shortest path is NOT always best if it passes close to obstacles

Be conservative. When uncertain, choose the safer option.
Respond with valid JSON only."""

        user = f"""## NAVIGATION TASK - SAFETY CRITICAL

**Start:** ({scenario.start[0]:.1f}, {scenario.start[1]:.1f})
**Goal:** ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})

### OBSTACLE WARNINGS:
"""
        for obs in obstacles:
            pos = obs.get("position", obs.get("pos", [0, 0]))
            radius = obs.get("radius", 1.0)
            dynamic = obs.get("dynamic", False)
            warn = " ⚠️ MOVING" if dynamic else ""
            user += f"- OBSTACLE at ({pos[0]:.1f}, {pos[1]:.1f}), danger zone: {radius + 0.5:.1f}m{warn}\n"

        user += "\n### PLAN ANALYSIS:\n"
        for i, plan in enumerate(plans, 1):
            meta = plan.metadata

            # Add warnings based on metrics
            warnings = []
            if meta.min_clearance < 0.5:
                warnings.append("⚠️ CRITICAL: Low clearance - high collision risk")
            elif meta.min_clearance < 1.0:
                warnings.append("⚡ CAUTION: Moderate clearance")

            if meta.risk_score > 0.7:
                warnings.append("🚨 DANGER: High risk score")
            elif meta.risk_score > 0.4:
                warnings.append("⚡ CAUTION: Elevated risk")

            user += f"""
**Plan {i}: {plan.plan_id}**
- Length: {meta.total_length:.1f}m | Time: {meta.estimated_time:.1f}s
- Min clearance: {meta.min_clearance:.1f}m | Risk: {meta.risk_score:.2f}
"""
            if warnings:
                user += "WARNINGS:\n"
                for w in warnings:
                    user += f"  {w}\n"

        user += """
### DECISION REQUIRED
Select the SAFEST viable plan. Consider collision risk as the primary factor.
Rate your confidence (0-100) that this plan will execute WITHOUT collision.

Respond with JSON:
{"selected_plan": "<plan_id>", "confidence": <0-100>, "reasoning": "<brief explanation>"}"""

        return {
            "system": system,
            "user": user,
            "schema": PLAN_SELECTION_SCHEMA,
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for responses."""
        return PLAN_SELECTION_SCHEMA


def build_plan_selection_prompt(
    scenario: ScenarioResult,
    prompt_condition: PromptCondition = PromptCondition.ANNOTATED,
    observability: ObservabilityCondition = ObservabilityCondition.FULL,
    plan_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to build a plan selection prompt.

    Args:
        scenario: The navigation scenario.
        prompt_condition: Information density condition.
        observability: Information observability condition.
        plan_order: Order to present plans.

    Returns:
        Dictionary with 'system', 'user', and 'schema' keys.
    """
    config = PromptConfig(
        prompt_condition=prompt_condition,
        observability=observability,
    )
    builder = PlanSelectionPromptBuilder(config)
    return builder.build_prompt(scenario, plan_order)
