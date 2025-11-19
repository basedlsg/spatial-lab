"""
Baseline Comparison: Algorithmic vs LLM Robot Coordination

Compares:
1. Pure algorithmic approach (rule-based response to needs)
2. LLM approach (natural language interpretation + code generation)

Measures where each approach excels and fails.
"""

import asyncio
import json
import logging
import time
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .nl_code_experiment import RobotState, NeedType, ExperimentTrial
from .groq_client import GroqClient, GroqConfig

logger = logging.getLogger(__name__)


@dataclass
class ComparisonTrial:
    """Single comparison trial"""
    trial_id: int
    sender: RobotState
    receiver: RobotState
    need_type: NeedType
    message: str

    # Algorithmic results
    algo_response: Optional[str] = None
    algo_success: bool = False
    algo_time_ms: float = 0.0
    algo_appropriate: bool = False  # Was the response contextually appropriate?

    # LLM results
    llm_interpretation: Optional[Dict] = None
    llm_code: Optional[str] = None
    llm_success: bool = False
    llm_time_ms: float = 0.0
    llm_appropriate: bool = False

    # Comparison
    winner: Optional[str] = None  # "algo", "llm", or "tie"
    notes: str = ""


@dataclass
class ComparisonResults:
    """Aggregated comparison results"""
    total_trials: int = 0

    algo_wins: int = 0
    llm_wins: int = 0
    ties: int = 0

    algo_success_rate: float = 0.0
    llm_success_rate: float = 0.0

    algo_avg_time_ms: float = 0.0
    llm_avg_time_ms: float = 0.0

    # Breakdown by need type
    by_type: Dict[str, Dict] = field(default_factory=dict)

    trials: List[ComparisonTrial] = field(default_factory=list)


class AlgorithmicResponder:
    """
    Rule-based algorithmic response system.

    Uses simple heuristics to respond to robot needs.
    No LLM involved - pure pattern matching and distance calculations.
    """

    def respond(
        self,
        need_type: NeedType,
        message: str,
        sender: RobotState,
        receiver: RobotState
    ) -> Tuple[str, bool]:
        """
        Generate algorithmic response to a need.

        Returns:
            (response_action, success)
        """
        start = time.time()

        # Calculate distance between robots
        distance = math.dist(sender.position, receiver.position)

        if need_type == NeedType.HELP_CARRY:
            # Check if receiver can help (not carrying, has battery)
            if receiver.carrying is None and receiver.battery > 0.3:
                if distance < 20:
                    response = f"move_to({sender.position}); help_carry()"
                    success = True
                else:
                    response = f"move_to({sender.position}); help_carry()  # far"
                    success = True
            else:
                response = "cannot_help(busy)"
                success = False

        elif need_type == NeedType.CLEAR_PATH:
            # Simple: move away from sender's position
            new_pos = (
                receiver.position[0] + 5,
                receiver.position[1] + 5
            )
            response = f"move_to({new_pos})"
            success = True

        elif need_type == NeedType.SHARE_LOCATION:
            # Always can share location
            response = f"send_position({receiver.position})"
            success = True

        elif need_type == NeedType.COORDINATE_TIMING:
            # Calculate estimated arrival time
            if distance > 0:
                eta = distance / receiver.speed
                response = f"acknowledge_timing(eta={eta:.1f}s)"
                success = True
            else:
                response = "already_at_location()"
                success = True

        elif need_type == NeedType.BATTERY_LOW:
            # Check if receiver can take over
            if receiver.current_task is None and receiver.battery > 0.5:
                response = f"move_to({sender.position}); take_over_task()"
                success = True
            else:
                response = "cannot_take_over(busy_or_low_battery)"
                success = False

        else:
            response = "unknown_need_type()"
            success = False

        elapsed = (time.time() - start) * 1000
        return response, success, elapsed

    def is_contextually_appropriate(
        self,
        response: str,
        need_type: NeedType,
        sender: RobotState,
        receiver: RobotState
    ) -> bool:
        """
        Check if the algorithmic response is contextually appropriate.

        The algorithmic approach often gives technically correct but
        contextually tone-deaf responses.
        """
        # Algorithmic responses are always "appropriate" in their simplicity
        # But they miss nuance - this is what we want to compare
        distance = math.dist(sender.position, receiver.position)

        # Some cases where algorithmic response might be inappropriate
        if need_type == NeedType.BATTERY_LOW:
            # If sender is very low battery, urgency matters
            if sender.battery < 0.2 and "cannot" in response:
                return False  # Should try harder to help

        if need_type == NeedType.HELP_CARRY:
            # If very far away, should maybe suggest another robot
            if distance > 30 and "far" not in response:
                return False

        return True


class BaselineComparison:
    """
    Run comparison between algorithmic and LLM approaches.
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.algo = AlgorithmicResponder()
        self.config = GroqConfig(
            api_key=groq_api_key,
            model=model,
            temperature=0.3
        )
        self.results = ComparisonResults()

        logger.info("Initialized baseline comparison")

    def generate_message(self, need_type: NeedType, sender: RobotState) -> str:
        """Generate test message"""
        templates = {
            NeedType.HELP_CARRY: f"I have a heavy item at {sender.position} that exceeds my capacity. Can you help?",
            NeedType.CLEAR_PATH: f"My path is blocked near {sender.position}. Can you clear it?",
            NeedType.SHARE_LOCATION: f"Where are you? I'm at {sender.position} planning a route.",
            NeedType.COORDINATE_TIMING: f"I'll arrive in 30s from {sender.position}. Be ready?",
            NeedType.BATTERY_LOW: f"Battery at {sender.battery:.0%}. Need task handoff at {sender.position}.",
        }
        return templates[need_type]

    async def run_trial(
        self,
        trial_id: int,
        sender: RobotState,
        receiver: RobotState,
        need_type: NeedType
    ) -> ComparisonTrial:
        """Run single comparison trial"""

        message = self.generate_message(need_type, sender)

        trial = ComparisonTrial(
            trial_id=trial_id,
            sender=sender,
            receiver=receiver,
            need_type=need_type,
            message=message
        )

        # Run algorithmic approach
        algo_response, algo_success, algo_time = self.algo.respond(
            need_type, message, sender, receiver
        )
        trial.algo_response = algo_response
        trial.algo_success = algo_success
        trial.algo_time_ms = algo_time
        trial.algo_appropriate = self.algo.is_contextually_appropriate(
            algo_response, need_type, sender, receiver
        )

        # Run LLM approach
        async with GroqClient(self.config) as client:
            start = time.time()

            try:
                sender_ctx = {
                    "robot_id": sender.robot_id,
                    "position": sender.position,
                    "battery": sender.battery,
                    "carrying": sender.carrying
                }
                receiver_ctx = {
                    "robot_id": receiver.robot_id,
                    "position": receiver.position,
                    "battery": receiver.battery,
                    "carrying": receiver.carrying
                }

                # Interpret message
                interpretation = await client.interpret_message(
                    message, sender_ctx, receiver_ctx
                )
                trial.llm_interpretation = interpretation

                # Generate code
                if interpretation.get("can_help", True):
                    code_result = await client.generate_code(
                        task_description=interpretation.get("suggested_action", "Help sender"),
                        available_apis=["move_to()", "help_carry()", "send_message()"],
                        context={"my_pos": receiver.position, "sender_pos": sender.position}
                    )
                    trial.llm_code = code_result["code"]
                    trial.llm_success = True
                else:
                    trial.llm_code = "# Cannot help"
                    trial.llm_success = False

                trial.llm_time_ms = (time.time() - start) * 1000

                # Check if LLM response is appropriate
                trial.llm_appropriate = self._check_llm_appropriate(
                    interpretation, need_type, sender, receiver
                )

            except Exception as e:
                trial.llm_success = False
                trial.llm_time_ms = (time.time() - start) * 1000
                trial.notes = f"LLM error: {str(e)}"

        # Determine winner
        trial.winner = self._determine_winner(trial)

        return trial

    def _check_llm_appropriate(
        self,
        interpretation: Dict,
        need_type: NeedType,
        sender: RobotState,
        receiver: RobotState
    ) -> bool:
        """Check if LLM response is contextually appropriate"""
        if not interpretation:
            return False

        priority = interpretation.get("priority", "medium")

        # Check if priority matches urgency
        if need_type == NeedType.BATTERY_LOW and sender.battery < 0.2:
            if priority != "high":
                return False

        # Check if spatial understanding is correct
        spatial = interpretation.get("spatial_info")
        if spatial and isinstance(spatial, dict):
            # LLM should understand positions
            pass

        return True

    def _determine_winner(self, trial: ComparisonTrial) -> str:
        """Determine which approach won the trial"""

        # Scoring:
        # - Success: 2 points
        # - Appropriate: 1 point
        # - Speed bonus if > 10x faster: 0.5 points

        algo_score = 0
        llm_score = 0

        if trial.algo_success:
            algo_score += 2
        if trial.algo_appropriate:
            algo_score += 1

        if trial.llm_success:
            llm_score += 2
        if trial.llm_appropriate:
            llm_score += 1

        # Speed comparison (algorithmic is always faster, but we give small bonus)
        if trial.algo_time_ms > 0 and trial.llm_time_ms > 0:
            if trial.algo_time_ms * 10 < trial.llm_time_ms:
                algo_score += 0.5

        if algo_score > llm_score:
            return "algo"
        elif llm_score > algo_score:
            return "llm"
        else:
            return "tie"

    async def run_comparison(
        self,
        num_trials: int = 30,
        robots: List[RobotState] = None
    ) -> ComparisonResults:
        """Run full comparison experiment"""

        if robots is None:
            robots = self._generate_robots(5)

        logger.info(f"Running {num_trials} comparison trials")

        # Initialize by-type tracking
        for need_type in NeedType:
            self.results.by_type[need_type.value] = {
                "algo_wins": 0,
                "llm_wins": 0,
                "ties": 0
            }

        # Run trials
        for i in range(num_trials):
            sender = random.choice(robots)
            receiver = random.choice([r for r in robots if r.robot_id != sender.robot_id])
            need_type = random.choice(list(NeedType))

            trial = await self.run_trial(i, sender, receiver, need_type)
            self.results.trials.append(trial)

            # Update by-type results
            type_results = self.results.by_type[need_type.value]
            if trial.winner == "algo":
                type_results["algo_wins"] += 1
            elif trial.winner == "llm":
                type_results["llm_wins"] += 1
            else:
                type_results["ties"] += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_trials} comparison trials")

        # Calculate aggregated metrics
        self._calculate_metrics()

        return self.results

    def _generate_robots(self, count: int) -> List[RobotState]:
        """Generate test robots"""
        robots = []
        for i in range(count):
            robots.append(RobotState(
                robot_id=f"robot_{i:03d}",
                position=(random.uniform(0, 50), random.uniform(0, 50)),
                battery=random.uniform(0.2, 1.0),
                carrying=random.choice([None, f"item_{i}"]),
                current_task=random.choice([None, "pickup", "delivery"])
            ))
        return robots

    def _calculate_metrics(self):
        """Calculate aggregated metrics"""
        self.results.total_trials = len(self.results.trials)

        self.results.algo_wins = sum(1 for t in self.results.trials if t.winner == "algo")
        self.results.llm_wins = sum(1 for t in self.results.trials if t.winner == "llm")
        self.results.ties = sum(1 for t in self.results.trials if t.winner == "tie")

        algo_successes = [t for t in self.results.trials if t.algo_success]
        llm_successes = [t for t in self.results.trials if t.llm_success]

        if self.results.total_trials > 0:
            self.results.algo_success_rate = len(algo_successes) / self.results.total_trials
            self.results.llm_success_rate = len(llm_successes) / self.results.total_trials

        if algo_successes:
            self.results.algo_avg_time_ms = sum(t.algo_time_ms for t in algo_successes) / len(algo_successes)
        if llm_successes:
            self.results.llm_avg_time_ms = sum(t.llm_time_ms for t in llm_successes) / len(llm_successes)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        r = self.results

        summary = f"""
================================================================================
BASELINE COMPARISON: Algorithmic vs LLM Robot Coordination
================================================================================

Overall Results:
  Total Trials:      {r.total_trials}

  Algorithmic Wins:  {r.algo_wins} ({r.algo_wins/r.total_trials*100:.1f}%)
  LLM Wins:          {r.llm_wins} ({r.llm_wins/r.total_trials*100:.1f}%)
  Ties:              {r.ties} ({r.ties/r.total_trials*100:.1f}%)

Success Rates:
  Algorithmic:       {r.algo_success_rate*100:.1f}%
  LLM:               {r.llm_success_rate*100:.1f}%

Average Response Time:
  Algorithmic:       {r.algo_avg_time_ms:.2f} ms
  LLM:               {r.llm_avg_time_ms:.0f} ms  ({r.llm_avg_time_ms/max(r.algo_avg_time_ms, 0.001):.0f}x slower)

Results by Need Type:
"""
        for need_type, data in r.by_type.items():
            total = data["algo_wins"] + data["llm_wins"] + data["ties"]
            if total > 0:
                summary += f"  {need_type:20s}: Algo {data['algo_wins']} | LLM {data['llm_wins']} | Tie {data['ties']}\n"

        summary += """
Key Insights:
"""
        # Generate insights based on results
        if r.algo_wins > r.llm_wins:
            summary += "  - Algorithmic approach wins on speed and simple tasks\n"
        elif r.llm_wins > r.algo_wins:
            summary += "  - LLM approach wins on contextual understanding\n"
        else:
            summary += "  - Both approaches perform similarly overall\n"

        # Find where LLM excels
        llm_best = max(r.by_type.items(), key=lambda x: x[1]["llm_wins"])
        if llm_best[1]["llm_wins"] > 0:
            summary += f"  - LLM excels at: {llm_best[0]}\n"

        # Find where Algo excels
        algo_best = max(r.by_type.items(), key=lambda x: x[1]["algo_wins"])
        if algo_best[1]["algo_wins"] > 0:
            summary += f"  - Algorithmic excels at: {algo_best[0]}\n"

        summary += "================================================================================"

        return summary


async def run_baseline_comparison(groq_api_key: str, num_trials: int = 30):
    """Convenience function to run comparison"""
    comparison = BaselineComparison(groq_api_key)
    results = await comparison.run_comparison(num_trials)
    print(comparison.get_summary())
    return results
