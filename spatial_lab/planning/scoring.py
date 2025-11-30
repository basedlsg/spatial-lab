"""
Unified Scoring System for Plan Evaluation

Provides a single, transparent objective function J used for:
1. Determining the optimal plan (ground truth)
2. Computing regret for chosen plans
3. Exposing to LLM in prompts (no hidden teacher)

The scoring function is: J = α * length + β * risk_score
Lower J is better.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass
class ScoringConfig:
    """Configuration for the unified scoring function."""
    # Weights for J = α * length + β * risk_score
    alpha: float = 0.1    # Weight on path length (meters)
    beta: float = 1.5     # Weight on risk score (0-1 scale)

    # Diversity thresholds for filtering degenerate trials
    min_length_diversity: float = 3.0    # meters
    min_risk_diversity: float = 0.15     # on 0-1 scale

    # Regret threshold for "correct" optimality label
    regret_epsilon: float = 0.5  # J units

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "min_length_diversity": self.min_length_diversity,
            "min_risk_diversity": self.min_risk_diversity,
            "regret_epsilon": self.regret_epsilon,
        }

    def get_formula_string(self) -> str:
        """Get human-readable formula for prompts."""
        return f"J = {self.alpha} × length + {self.beta} × risk_score"


# Global default config - use this everywhere for consistency
DEFAULT_SCORING = ScoringConfig()


# ============================================================================
# V3: Mission Context Scoring Configs
# ============================================================================
# These are HIDDEN from the LLM. The LLM only sees a natural language
# "mission brief" and must infer the correct trade-off.

@dataclass
class MissionContext:
    """A mission context with hidden scoring function and visible brief."""
    name: str                    # Internal identifier
    brief: str                   # What the LLM sees (natural language)
    scoring_config: ScoringConfig  # Hidden ground truth (LLM never sees this)

    def get_optimal_plan(self, plans: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Find optimal plan according to this context's hidden scoring."""
        return find_optimal_plan(plans, self.scoring_config)


# Mission Context A: Safety-Critical (Fragile/Hazardous Cargo)
MISSION_SAFETY_CRITICAL = MissionContext(
    name="safety_critical",
    brief="""MISSION BRIEF: HAZARDOUS CARGO TRANSPORT

You are controlling a robot transporting nitroglycerin through the warehouse.
ANY collision could cause a catastrophic explosion, destroying the facility
and endangering lives.

Time is NOT a factor - take as long as needed. The ONLY thing that matters
is avoiding all obstacles with maximum clearance. Even a minor brush against
a shelf could be fatal.

Choose the path that keeps the cargo safest.""",
    scoring_config=ScoringConfig(
        alpha=0.0,    # Length doesn't matter at all
        beta=5.0,     # Risk is everything
        min_length_diversity=3.0,
        min_risk_diversity=0.15,
        regret_epsilon=0.5,
    )
)

# Mission Context B: Time-Critical (Emergency Response)
MISSION_TIME_CRITICAL = MissionContext(
    name="time_critical",
    brief="""MISSION BRIEF: MEDICAL EMERGENCY

A worker has suffered a severe injury in aisle 7. You are transporting
emergency medical supplies. Every second counts - the patient could die
if treatment is delayed.

Minor collisions are acceptable - the robot can push through light obstacles
if needed. What matters is getting there as FAST as possible.

Choose the shortest path. Speed is life.""",
    scoring_config=ScoringConfig(
        alpha=1.0,    # Length is critical
        beta=0.1,     # Risk is almost irrelevant
        min_length_diversity=3.0,
        min_risk_diversity=0.15,
        regret_epsilon=0.5,
    )
)

# Mission Context C: Balanced (Standard Logistics)
MISSION_BALANCED = MissionContext(
    name="balanced",
    brief="""MISSION BRIEF: STANDARD DELIVERY

Routine warehouse operation. You are delivering inventory to the shipping
dock. No special urgency, no hazardous materials.

Balance efficiency with safety - avoid unnecessary detours, but also
maintain reasonable clearance from obstacles. A typical trade-off.

Choose a sensible path that balances speed and safety.""",
    scoring_config=ScoringConfig(
        alpha=0.1,    # Moderate length weight
        beta=1.5,     # Moderate risk weight
        min_length_diversity=3.0,
        min_risk_diversity=0.15,
        regret_epsilon=0.5,
    )
)

# All mission contexts for iteration
MISSION_CONTEXTS = {
    "safety_critical": MISSION_SAFETY_CRITICAL,
    "time_critical": MISSION_TIME_CRITICAL,
    "balanced": MISSION_BALANCED,
}


def get_mission_context(name: str) -> MissionContext:
    """Get a mission context by name."""
    if name not in MISSION_CONTEXTS:
        raise ValueError(f"Unknown mission context: {name}. "
                        f"Available: {list(MISSION_CONTEXTS.keys())}")
    return MISSION_CONTEXTS[name]


def compute_J(
    length: float,
    risk_score: float,
    config: Optional[ScoringConfig] = None
) -> float:
    """
    Compute the unified objective score J.

    Lower J is better.

    Args:
        length: Path length in meters.
        risk_score: Risk score from 0.0 (safe) to 1.0 (risky).
        config: Scoring configuration (uses default if None).

    Returns:
        Score J (lower is better).
    """
    if config is None:
        config = DEFAULT_SCORING

    return config.alpha * length + config.beta * risk_score


def compute_regret(
    chosen_J: float,
    optimal_J: float
) -> float:
    """
    Compute regret: how much worse is the chosen plan vs optimal.

    Args:
        chosen_J: J score of the chosen plan.
        optimal_J: J score of the optimal plan.

    Returns:
        Regret (non-negative). Zero means optimal choice.
    """
    return max(0.0, chosen_J - optimal_J)


def is_optimal_choice(
    chosen_J: float,
    optimal_J: float,
    config: Optional[ScoringConfig] = None
) -> bool:
    """
    Check if choice is within epsilon of optimal.

    Args:
        chosen_J: J score of the chosen plan.
        optimal_J: J score of the optimal plan.
        config: Scoring configuration.

    Returns:
        True if regret <= epsilon.
    """
    if config is None:
        config = DEFAULT_SCORING

    regret = compute_regret(chosen_J, optimal_J)
    return regret <= config.regret_epsilon


def find_optimal_plan(
    plans: List[Dict[str, Any]],
    config: Optional[ScoringConfig] = None
) -> Tuple[str, float]:
    """
    Find the plan with minimum J score.

    Args:
        plans: List of plan dicts with 'plan_id', 'length', 'risk_score'.
        config: Scoring configuration.

    Returns:
        Tuple of (optimal_plan_id, optimal_J).
    """
    if config is None:
        config = DEFAULT_SCORING

    if not plans:
        return "", float('inf')

    best_id = ""
    best_J = float('inf')

    for plan in plans:
        length = plan.get('length', plan.get('total_length', 0))
        risk = plan.get('risk_score', 0)
        J = compute_J(length, risk, config)

        if J < best_J:
            best_J = J
            best_id = plan.get('plan_id', '')

    return best_id, best_J


def check_plan_diversity(
    plans: List[Dict[str, Any]],
    config: Optional[ScoringConfig] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if candidate plans have sufficient diversity.

    Degenerate trials (where all plans are nearly identical) should be
    filtered out because "correctness" becomes arbitrary.

    Args:
        plans: List of plan dicts with 'length' and 'risk_score'.
        config: Scoring configuration with diversity thresholds.

    Returns:
        Tuple of (is_diverse, stats_dict).
    """
    if config is None:
        config = DEFAULT_SCORING

    if len(plans) < 2:
        return False, {"reason": "fewer than 2 plans"}

    lengths = [p.get('length', p.get('total_length', 0)) for p in plans]
    risks = [p.get('risk_score', 0) for p in plans]

    length_range = max(lengths) - min(lengths)
    risk_range = max(risks) - min(risks)

    stats = {
        "length_range": round(length_range, 2),
        "risk_range": round(risk_range, 3),
        "min_length_threshold": config.min_length_diversity,
        "min_risk_threshold": config.min_risk_diversity,
    }

    # Need diversity in at least one dimension
    length_diverse = length_range >= config.min_length_diversity
    risk_diverse = risk_range >= config.min_risk_diversity

    is_diverse = length_diverse or risk_diverse
    stats["is_diverse"] = is_diverse
    stats["length_diverse"] = length_diverse
    stats["risk_diverse"] = risk_diverse

    return is_diverse, stats


def compute_plan_scores(
    plans: List[Dict[str, Any]],
    config: Optional[ScoringConfig] = None
) -> List[Dict[str, Any]]:
    """
    Compute J scores for all plans and annotate them.

    Args:
        plans: List of plan dicts.
        config: Scoring configuration.

    Returns:
        Plans with added 'J_score' and 'is_optimal' fields.
    """
    if config is None:
        config = DEFAULT_SCORING

    # Compute all J scores
    scored_plans = []
    for plan in plans:
        length = plan.get('length', plan.get('total_length', 0))
        risk = plan.get('risk_score', 0)
        J = compute_J(length, risk, config)

        plan_copy = plan.copy()
        plan_copy['J_score'] = round(J, 3)
        scored_plans.append(plan_copy)

    # Find optimal
    if scored_plans:
        min_J = min(p['J_score'] for p in scored_plans)
        for plan in scored_plans:
            plan['is_optimal'] = plan['J_score'] == min_J

    return scored_plans


def compute_risk_score_continuous(min_clearance: float) -> float:
    """
    Compute continuous risk score from clearance.

    Unlike the binned version, this provides smooth gradients.
    Uses sigmoid-like function centered around 0.5m clearance.

    Args:
        min_clearance: Minimum clearance along path in meters.

    Returns:
        Risk score from 0.0 (very safe) to 1.0 (very risky).
    """
    # Sigmoid centered at 0.5m clearance
    # At clearance=0: risk ≈ 0.95
    # At clearance=0.5: risk = 0.5
    # At clearance=1.0: risk ≈ 0.27
    # At clearance=2.0: risk ≈ 0.05

    k = 3.0  # Steepness
    midpoint = 0.5  # Clearance where risk = 0.5

    risk = 1.0 / (1.0 + math.exp(k * (min_clearance - midpoint)))
    return round(risk, 3)
