"""
V3 Plan Selection Prompts: Contextual Reasoning

Key Change from V2:
- V2 gave the LLM the explicit J formula (arithmetic task)
- V3 gives only a natural language "mission brief" (reasoning task)

The LLM must infer the correct cost trade-off from semantic context,
not compute it from a formula. This tests genuine spatial reasoning.
"""

from typing import Dict, Any, List
from enum import Enum

from spatial_lab.planning.scoring import MissionContext, MISSION_CONTEXTS


class InfoLevel(Enum):
    """How much numeric information to provide about plans."""
    QUALITATIVE = "qualitative"      # Only descriptions
    QUANTITATIVE = "quantitative"    # Include numbers


# Core instruction - same for all conditions
# NO FORMULA - model must infer trade-offs from mission brief
CORE_INSTRUCTION_V3 = """You are a warehouse robot path planner.

Given the MISSION BRIEF and CANDIDATE PLANS below, select the best plan
for the current mission. Your selection should reflect the priorities
described in the mission brief.

After analyzing all plans, respond with ONLY a JSON object:
{
    "chosen_plan": "A",
    "confidence": 0.XX,
    "rationale": "Brief explanation of why this plan fits the mission"
}

IMPORTANT:
- "confidence" is your probability (0.0 to 1.0) that your choice is correct
  for THIS SPECIFIC MISSION context.
- Different missions have different priorities - consider what matters most.
"""


# Confidence calibration guidance (no anchors referencing J scores)
CALIBRATION_GUIDANCE_V3 = """
CONFIDENCE CALIBRATION:

Your confidence should reflect how well the plans match the mission needs:

LOW CONFIDENCE (0.3-0.5):
- Plans are all similar in the dimension that matters for this mission
- Hard to distinguish which is truly best for the mission

MEDIUM CONFIDENCE (0.5-0.7):
- One plan seems better but others could work
- Trade-offs are present

HIGH CONFIDENCE (0.8-0.9):
- One plan clearly matches the mission requirements
- Other plans would be obviously wrong for this mission
"""


def format_plan_qualitative(
    plan: Dict[str, Any],
    plan_id: str
) -> str:
    """Format plan with qualitative descriptions only."""
    length = plan.get('length', plan.get('total_length', 0))
    risk = plan.get('risk_score', 0)
    clearance = plan.get('min_clearance', 0.5)

    # Length descriptions
    if length < 20:
        length_desc = "short"
    elif length < 30:
        length_desc = "medium-length"
    else:
        length_desc = "long"

    # Risk descriptions
    if risk < 0.3:
        risk_desc = "very safe (wide clearances)"
    elif risk < 0.5:
        risk_desc = "moderately safe"
    elif risk < 0.7:
        risk_desc = "somewhat risky (tight clearances)"
    else:
        risk_desc = "risky (narrow passages)"

    return f"""Plan {plan_id}:
  - Path: {length_desc} route
  - Safety: {risk_desc}"""


def format_plan_quantitative(
    plan: Dict[str, Any],
    plan_id: str
) -> str:
    """Format plan with exact numbers (but NO J score formula)."""
    length = plan.get('length', plan.get('total_length', 0))
    risk = plan.get('risk_score', 0)
    clearance = plan.get('min_clearance', 0.5)

    return f"""Plan {plan_id}:
  - Path length: {length:.1f} meters
  - Risk score: {risk:.2f} (0=safest, 1=riskiest)
  - Minimum clearance: {clearance:.2f} meters"""


def build_v3_prompt(
    mission_context: MissionContext,
    plans: List[Dict[str, Any]],
    plan_order: List[str],
    info_level: InfoLevel = InfoLevel.QUANTITATIVE,
) -> Dict[str, str]:
    """
    Build V3 prompt with mission brief (no formula).

    Args:
        mission_context: The mission context with brief and hidden scoring.
        plans: List of candidate plans with stats.
        plan_order: Order of plan IDs (e.g., ['A', 'B', 'C']).
        info_level: How much numeric detail to include.

    Returns:
        Dict with 'system' and 'user' prompts.
    """
    # System prompt: core instruction + calibration
    system_prompt = CORE_INSTRUCTION_V3 + "\n" + CALIBRATION_GUIDANCE_V3

    # Format plans based on info level
    if info_level == InfoLevel.QUALITATIVE:
        formatter = format_plan_qualitative
    else:
        formatter = format_plan_quantitative

    # Build plan descriptions
    plan_descriptions = []
    plan_map = {p.get('plan_id', ''): p for p in plans}

    for pid in plan_order:
        if pid in plan_map:
            plan_descriptions.append(formatter(plan_map[pid], pid))

    plans_text = "\n\n".join(plan_descriptions)

    # User prompt: mission brief + plans
    user_prompt = f"""{mission_context.brief}

---

CANDIDATE PLANS:

{plans_text}

---

Based on the mission requirements above, which plan should you select?
Remember: respond with ONLY a valid JSON object."""

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def compute_contextual_alignment(
    chosen_plan_id: str,
    plans: List[Dict[str, Any]],
    mission_context: MissionContext
) -> Dict[str, Any]:
    """
    Compute contextual alignment metrics.

    Did the model pick the plan that's optimal for THIS mission's
    hidden scoring function?

    Args:
        chosen_plan_id: The plan ID the model selected.
        plans: All candidate plans.
        mission_context: The mission context with hidden scoring.

    Returns:
        Dict with alignment metrics.
    """
    # Find optimal plan for this mission's hidden J function
    optimal_id, optimal_J = mission_context.get_optimal_plan(plans)

    # Get chosen plan's J score under this mission's scoring
    chosen_plan = None
    for p in plans:
        if p.get('plan_id', '') == chosen_plan_id:
            chosen_plan = p
            break

    if chosen_plan is None:
        return {
            "aligned": False,
            "error": f"Chosen plan {chosen_plan_id} not found",
        }

    from spatial_lab.planning.scoring import compute_J, compute_regret

    config = mission_context.scoring_config
    chosen_J = compute_J(
        chosen_plan.get('length', chosen_plan.get('total_length', 0)),
        chosen_plan.get('risk_score', 0),
        config
    )
    regret = compute_regret(chosen_J, optimal_J)

    # Alignment = chose the optimal plan (or within epsilon)
    aligned = regret <= config.regret_epsilon

    return {
        "aligned": aligned,
        "chosen_plan": chosen_plan_id,
        "optimal_plan": optimal_id,
        "chosen_J": round(chosen_J, 3),
        "optimal_J": round(optimal_J, 3),
        "regret": round(regret, 3),
        "mission_context": mission_context.name,
    }


def compute_semantic_sensitivity(
    confidence_distributions: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Compute semantic sensitivity via Jensen-Shannon divergence.

    High divergence = model's confidence distribution shifts significantly
    when the mission context changes (good semantic sensitivity).

    Args:
        confidence_distributions: Dict mapping context_name -> list of confidences.

    Returns:
        Dict of pairwise JS divergence values.
    """
    import math
    from collections import Counter

    def histogram(values: List[float], bins: int = 10) -> List[float]:
        """Convert values to probability histogram."""
        if not values:
            return [1.0 / bins] * bins  # Uniform if no data

        counts = [0] * bins
        for v in values:
            bin_idx = min(int(v * bins), bins - 1)
            counts[bin_idx] += 1

        total = sum(counts)
        return [c / total if total > 0 else 1.0/bins for c in counts]

    def kl_divergence(p: List[float], q: List[float]) -> float:
        """KL divergence with smoothing."""
        eps = 1e-10
        return sum(pi * math.log((pi + eps) / (qi + eps))
                   for pi, qi in zip(p, q) if pi > 0)

    def js_divergence(p: List[float], q: List[float]) -> float:
        """Jensen-Shannon divergence."""
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    # Compute pairwise JS divergence
    contexts = list(confidence_distributions.keys())
    results = {}

    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            hist1 = histogram(confidence_distributions[ctx1])
            hist2 = histogram(confidence_distributions[ctx2])
            jsd = js_divergence(hist1, hist2)
            results[f"{ctx1}_vs_{ctx2}"] = round(jsd, 4)

    # Mean sensitivity across all pairs
    if results:
        results["mean_sensitivity"] = round(
            sum(results.values()) / len(results), 4
        )

    return results
