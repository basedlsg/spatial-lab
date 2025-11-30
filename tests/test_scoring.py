#!/usr/bin/env python3
"""
Unit tests for Plan Scoring System.

Tests the unified scoring function J = α×Length + β×Risk
and mission context functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_lab.planning.scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    compute_J,
    find_optimal_plan,
    check_plan_diversity,
    compute_plan_scores,
    compute_risk_score_continuous,
    MissionContext,
    MISSION_CONTEXTS,
    MISSION_SAFETY_CRITICAL,
    MISSION_TIME_CRITICAL,
    MISSION_BALANCED,
    get_mission_context,
)


class TestScoringConfig:
    """Test ScoringConfig dataclass."""

    def test_default_scoring_values(self):
        """Test default scoring configuration."""
        config = ScoringConfig()
        assert config.alpha == 0.1
        assert config.beta == 1.5

    def test_custom_scoring_config(self):
        """Test custom scoring configuration."""
        config = ScoringConfig(alpha=1.0, beta=0.5)
        assert config.alpha == 1.0
        assert config.beta == 0.5

    def test_formula_string(self):
        """Test formula string generation."""
        config = ScoringConfig(alpha=0.5, beta=2.0)
        formula = config.get_formula_string()
        assert "0.5" in formula
        assert "2.0" in formula
        assert "length" in formula.lower() or "J" in formula

    def test_config_to_dict(self):
        """Test serialization to dict."""
        config = ScoringConfig(alpha=0.3, beta=1.2)
        d = config.to_dict()
        assert d["alpha"] == 0.3
        assert d["beta"] == 1.2


class TestComputeJ:
    """Test the unified scoring function J = α×Length + β×Risk."""

    def test_basic_computation(self):
        """Test basic J score computation."""
        config = ScoringConfig(alpha=1.0, beta=1.0)
        J = compute_J(length=10.0, risk_score=0.5, config=config)
        assert J == 10.5  # 1.0*10 + 1.0*0.5

    def test_length_only(self):
        """Test scoring when only length matters."""
        config = ScoringConfig(alpha=1.0, beta=0.0)
        J = compute_J(length=15.0, risk_score=0.9, config=config)
        assert J == 15.0  # Only length contributes

    def test_risk_only(self):
        """Test scoring when only risk matters."""
        config = ScoringConfig(alpha=0.0, beta=5.0)
        J = compute_J(length=100.0, risk_score=0.5, config=config)
        assert J == 2.5  # Only risk contributes

    def test_default_scoring(self):
        """Test with default scoring config."""
        J = compute_J(length=20.0, risk_score=0.4, config=DEFAULT_SCORING)
        expected = 0.1 * 20.0 + 1.5 * 0.4  # 2.0 + 0.6 = 2.6
        assert abs(J - expected) < 0.001


class TestFindOptimalPlan:
    """Test optimal plan selection."""

    def setup_method(self):
        """Set up test plans."""
        self.plans = [
            {"plan_id": "A", "length": 20.0, "risk_score": 0.3},
            {"plan_id": "B", "length": 25.0, "risk_score": 0.1},
            {"plan_id": "C", "length": 18.0, "risk_score": 0.5},
        ]

    def test_find_optimal_balanced(self):
        """Test finding optimal with balanced scoring."""
        config = ScoringConfig(alpha=0.1, beta=1.5)
        optimal_id, optimal_J = find_optimal_plan(self.plans, config)

        # Compute J for each:
        # A: 0.1*20 + 1.5*0.3 = 2.0 + 0.45 = 2.45
        # B: 0.1*25 + 1.5*0.1 = 2.5 + 0.15 = 2.65
        # C: 0.1*18 + 1.5*0.5 = 1.8 + 0.75 = 2.55
        assert optimal_id == "A"
        assert abs(optimal_J - 2.45) < 0.01

    def test_find_optimal_speed_focused(self):
        """Test finding optimal when speed is prioritized."""
        config = ScoringConfig(alpha=1.0, beta=0.1)
        optimal_id, optimal_J = find_optimal_plan(self.plans, config)

        # Plan C has shortest length (18.0)
        assert optimal_id == "C"

    def test_find_optimal_safety_focused(self):
        """Test finding optimal when safety is prioritized."""
        config = ScoringConfig(alpha=0.0, beta=5.0)
        optimal_id, optimal_J = find_optimal_plan(self.plans, config)

        # Plan B has lowest risk (0.1)
        assert optimal_id == "B"


class TestPlanDiversity:
    """Test plan diversity checking."""

    def test_diverse_plans(self):
        """Test plans that are sufficiently diverse."""
        plans = [
            {"plan_id": "A", "length": 20.0, "risk_score": 0.3},
            {"plan_id": "B", "length": 30.0, "risk_score": 0.1},  # 10 units longer, 0.2 safer
            {"plan_id": "C", "length": 18.0, "risk_score": 0.6},  # 2 units shorter, 0.3 riskier
        ]

        # Use config with diversity thresholds
        config = ScoringConfig(min_length_diversity=3.0, min_risk_diversity=0.15)
        is_diverse, stats = check_plan_diversity(plans, config)

        assert is_diverse is True

    def test_non_diverse_plans(self):
        """Test plans that are too similar."""
        plans = [
            {"plan_id": "A", "length": 20.0, "risk_score": 0.3},
            {"plan_id": "B", "length": 20.5, "risk_score": 0.32},  # Too similar
            {"plan_id": "C", "length": 21.0, "risk_score": 0.31},  # Too similar
        ]

        # Use config with diversity thresholds
        config = ScoringConfig(min_length_diversity=3.0, min_risk_diversity=0.15)
        is_diverse, stats = check_plan_diversity(plans, config)

        assert is_diverse is False


class TestRiskScoreComputation:
    """Test continuous risk score computation."""

    def test_high_clearance_low_risk(self):
        """Test that high clearance results in low risk."""
        risk = compute_risk_score_continuous(min_clearance=2.0)
        assert risk < 0.3

    def test_low_clearance_high_risk(self):
        """Test that low clearance results in high risk."""
        risk = compute_risk_score_continuous(min_clearance=0.1)
        assert risk > 0.7

    def test_risk_in_valid_range(self):
        """Test that risk is always in [0, 1]."""
        for min_c in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
            risk = compute_risk_score_continuous(min_clearance=min_c)
            assert 0.0 <= risk <= 1.0


class TestMissionContexts:
    """Test V3 Mission Context functionality."""

    def test_safety_critical_context(self):
        """Test safety-critical mission context."""
        ctx = MISSION_SAFETY_CRITICAL
        assert ctx.name == "safety_critical"
        assert ctx.scoring_config.alpha == 0.0  # Length doesn't matter
        assert ctx.scoring_config.beta == 5.0   # Risk is everything
        assert "nitroglycerin" in ctx.brief.lower() or "hazardous" in ctx.brief.lower()

    def test_time_critical_context(self):
        """Test time-critical mission context."""
        ctx = MISSION_TIME_CRITICAL
        assert ctx.name == "time_critical"
        assert ctx.scoring_config.alpha == 1.0  # Length matters
        assert ctx.scoring_config.beta == 0.1   # Risk is minimal
        assert "emergency" in ctx.brief.lower() or "fast" in ctx.brief.lower()

    def test_balanced_context(self):
        """Test balanced mission context."""
        ctx = MISSION_BALANCED
        assert ctx.name == "balanced"
        assert ctx.scoring_config.alpha == 0.1
        assert ctx.scoring_config.beta == 1.5
        assert "balance" in ctx.brief.lower() or "routine" in ctx.brief.lower()

    def test_get_mission_context(self):
        """Test mission context retrieval."""
        ctx = get_mission_context("safety_critical")
        assert ctx is MISSION_SAFETY_CRITICAL

        ctx = get_mission_context("time_critical")
        assert ctx is MISSION_TIME_CRITICAL

        ctx = get_mission_context("balanced")
        assert ctx is MISSION_BALANCED

    def test_mission_contexts_dict(self):
        """Test all contexts are in the dictionary."""
        assert len(MISSION_CONTEXTS) == 3
        assert "safety_critical" in MISSION_CONTEXTS
        assert "time_critical" in MISSION_CONTEXTS
        assert "balanced" in MISSION_CONTEXTS

    def test_context_optimal_plan_selection(self):
        """Test that different contexts select different optimal plans."""
        plans = [
            {"plan_id": "A", "length": 15.0, "risk_score": 0.8},  # Short but risky
            {"plan_id": "B", "length": 30.0, "risk_score": 0.1},  # Long but safe
            {"plan_id": "C", "length": 20.0, "risk_score": 0.4},  # Balanced
        ]

        # Safety context should prefer B (lowest risk)
        safety_opt, _ = MISSION_SAFETY_CRITICAL.get_optimal_plan(plans)
        assert safety_opt == "B"

        # Time context should prefer A (shortest)
        time_opt, _ = MISSION_TIME_CRITICAL.get_optimal_plan(plans)
        assert time_opt == "A"


class TestComputePlanScores:
    """Test batch plan score computation."""

    def test_compute_all_scores(self):
        """Test computing J scores for multiple plans."""
        plans = [
            {"plan_id": "A", "length": 20.0, "risk_score": 0.3},
            {"plan_id": "B", "length": 25.0, "risk_score": 0.1},
        ]
        config = ScoringConfig(alpha=0.1, beta=1.5)

        scored = compute_plan_scores(plans, config)

        assert len(scored) == 2
        assert "J_score" in scored[0]
        assert "J_score" in scored[1]

    def test_scores_preserve_plan_data(self):
        """Test that original plan data is preserved."""
        plans = [
            {"plan_id": "X", "length": 10.0, "risk_score": 0.5, "extra": "data"},
        ]
        config = ScoringConfig()

        scored = compute_plan_scores(plans, config)

        assert scored[0]["plan_id"] == "X"
        assert scored[0]["length"] == 10.0
        assert scored[0]["extra"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
