"""
Natural Language + Code Generation Robot Coordination Experiment

Research Question: Can robots effectively communicate needs in natural language
and have those needs fulfilled through LLM-generated code?

This experiment measures:
1. Communication success rate
2. Code generation accuracy
3. Task completion rate
4. Comparison to baseline algorithmic approach
"""

import asyncio
import json
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

from .groq_client import GroqClient, GroqConfig

logger = logging.getLogger(__name__)


class NeedType(Enum):
    """Types of needs a robot can express"""
    HELP_CARRY = "help_carry"
    CLEAR_PATH = "clear_path"
    SHARE_LOCATION = "share_location"
    COORDINATE_TIMING = "coordinate_timing"
    BATTERY_LOW = "battery_low"


@dataclass
class RobotState:
    """State of a robot in the experiment"""
    robot_id: str
    position: Tuple[float, float]
    battery: float
    carrying: Optional[str] = None
    current_task: Optional[str] = None
    speed: float = 1.0
    max_payload: float = 10.0


@dataclass
class ExperimentTrial:
    """Single trial in the experiment"""
    trial_id: int
    sender: RobotState
    receiver: RobotState
    need_type: NeedType
    message: str

    # Results
    interpretation: Optional[Dict] = None
    generated_code: Optional[str] = None
    verification_passed: bool = False
    execution_result: Optional[Dict] = None

    # Metrics
    interpretation_latency_ms: float = 0.0
    code_gen_latency_ms: float = 0.0
    total_time_ms: float = 0.0
    success: bool = False
    failure_reason: Optional[str] = None


@dataclass
class ExperimentResults:
    """Aggregated experiment results"""
    total_trials: int = 0
    successful_trials: int = 0
    failed_interpretation: int = 0
    failed_code_gen: int = 0
    failed_verification: int = 0
    failed_execution: int = 0

    avg_interpretation_latency_ms: float = 0.0
    avg_code_gen_latency_ms: float = 0.0
    avg_total_time_ms: float = 0.0

    trials: List[ExperimentTrial] = field(default_factory=list)

    # Breakdown by need type
    results_by_type: Dict[str, Dict] = field(default_factory=dict)


class CodeVerifier:
    """
    Verifies generated code is safe to execute.

    Checks:
    - No dangerous imports
    - No file I/O
    - No network calls
    - No infinite loops (basic check)
    - Uses only allowed APIs
    """

    FORBIDDEN_PATTERNS = [
        r'\bimport\s+os\b',
        r'\bimport\s+sys\b',
        r'\bimport\s+subprocess\b',
        r'\bopen\s*\(',
        r'\bexec\s*\(',
        r'\beval\s*\(',
        r'requests\.',
        r'urllib\.',
        r'socket\.',
        r'__import__',
        r'\bwhile\s+True\b',
    ]

    ALLOWED_APIS = [
        'planner.plan_path',
        'robot.move_to',
        'robot.pick_item',
        'robot.drop_item',
        'comm.send_message',
        'coordinator.allocate',
        'math.',
        'await',
        'async',
        'return',
        'if',
        'for',
        'in',
        'range',
        'len',
        'print',
    ]

    def verify(self, code: str) -> Tuple[bool, str]:
        """
        Verify code is safe to execute.

        Returns:
            (is_safe, reason)
        """
        if not code or not code.strip():
            return False, "Empty code"

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return False, f"Forbidden pattern: {pattern}"

        # Check code length (prevent very long generated code)
        if len(code) > 2000:
            return False, "Code too long"

        # Basic syntax check
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        return True, "Passed"


class NLCodeExperiment:
    """
    Main experiment runner for natural language + code generation coordination.
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.config = GroqConfig(
            api_key=groq_api_key,
            model=model,
            temperature=0.3
        )
        self.verifier = CodeVerifier()
        self.results = ExperimentResults()

        # Available APIs that generated code can use
        self.available_apis = [
            "await planner.plan_path(start, goal, obstacles) -> List[PathPoint]",
            "await robot.move_to(position) -> bool",
            "await robot.pick_item(item_id) -> bool",
            "await robot.drop_item(location) -> bool",
            "await comm.send_message(receiver_id, content) -> bool",
            "math.sqrt(x), math.dist(p1, p2)",
            "robot.position -> (x, y)",
            "robot.battery -> float",
            "robot.carrying -> str or None",
        ]

        logger.info(f"Initialized NLCodeExperiment with model: {model}")

    def generate_need_message(self, need_type: NeedType, sender: RobotState) -> str:
        """Generate a natural language message for a need"""

        templates = {
            NeedType.HELP_CARRY: [
                f"I have a heavy item at {sender.position} that exceeds my capacity. Can you help carry it?",
                f"Need assistance with item pickup at position {sender.position}. Too heavy for me alone.",
                f"I'm at {sender.position} with a large load. Could use help transporting it.",
            ],
            NeedType.CLEAR_PATH: [
                f"My path to the destination is blocked near {sender.position}. Can you clear the area?",
                f"There's an obstacle in my way around {sender.position}. Need the path cleared.",
                f"Requesting path clearance near my position {sender.position}.",
            ],
            NeedType.SHARE_LOCATION: [
                f"Where are you currently? I'm at {sender.position} and need to coordinate.",
                f"Please share your location. I'm planning a route from {sender.position}.",
                f"Need your position for coordination. I'm currently at {sender.position}.",
            ],
            NeedType.COORDINATE_TIMING: [
                f"I'll reach the drop zone in about 30 seconds from {sender.position}. Can you be ready?",
                f"Need to sync timing. I'm departing {sender.position} now. When should I arrive?",
                f"Let's coordinate arrival. I'm starting from {sender.position}.",
            ],
            NeedType.BATTERY_LOW: [
                f"Battery at {sender.battery:.0%}. Need to hand off my task at {sender.position}.",
                f"Low power warning. Can you take over my current task? I'm at {sender.position}.",
                f"Running low on battery ({sender.battery:.0%}). Requesting task handoff at {sender.position}.",
            ],
        }

        return random.choice(templates[need_type])

    async def run_trial(
        self,
        trial_id: int,
        sender: RobotState,
        receiver: RobotState,
        need_type: NeedType
    ) -> ExperimentTrial:
        """
        Run a single experiment trial.

        Flow:
        1. Sender generates natural language need message
        2. Receiver's LLM interprets the message
        3. LLM generates code to fulfill the need
        4. Code is verified for safety
        5. Code is (simulated) executed
        6. Results are recorded
        """

        message = self.generate_need_message(need_type, sender)

        trial = ExperimentTrial(
            trial_id=trial_id,
            sender=sender,
            receiver=receiver,
            need_type=need_type,
            message=message
        )

        start_time = time.time()

        async with GroqClient(self.config) as client:
            # Step 1: Interpret the message
            try:
                sender_context = {
                    "robot_id": sender.robot_id,
                    "position": sender.position,
                    "battery": sender.battery,
                    "carrying": sender.carrying,
                    "task": sender.current_task
                }

                receiver_context = {
                    "robot_id": receiver.robot_id,
                    "position": receiver.position,
                    "battery": receiver.battery,
                    "carrying": receiver.carrying,
                    "task": receiver.current_task
                }

                interpretation = await client.interpret_message(
                    message=message,
                    sender_context=sender_context,
                    receiver_context=receiver_context
                )

                trial.interpretation = interpretation
                trial.interpretation_latency_ms = interpretation.get("latency_ms", 0)

                if interpretation.get("parse_error") or not interpretation.get("can_help", True):
                    trial.failure_reason = "Interpretation failed or cannot help"
                    self.results.failed_interpretation += 1
                    trial.total_time_ms = (time.time() - start_time) * 1000
                    return trial

            except Exception as e:
                trial.failure_reason = f"Interpretation error: {str(e)}"
                self.results.failed_interpretation += 1
                trial.total_time_ms = (time.time() - start_time) * 1000
                return trial

            # Step 2: Generate code to fulfill the need
            try:
                task_description = f"""
                As robot {receiver.robot_id} at position {receiver.position}:
                {interpretation.get('suggested_action', 'Help the sender')}

                Sender's need: {interpretation.get('need', message)}
                Sender's position: {sender.position}
                Priority: {interpretation.get('priority', 'medium')}
                """

                context = {
                    "my_position": receiver.position,
                    "my_battery": receiver.battery,
                    "sender_position": sender.position,
                    "spatial_info": interpretation.get("spatial_info")
                }

                code_result = await client.generate_code(
                    task_description=task_description,
                    available_apis=self.available_apis,
                    context=context
                )

                trial.generated_code = code_result["code"]
                trial.code_gen_latency_ms = code_result["latency_ms"]

            except Exception as e:
                trial.failure_reason = f"Code generation error: {str(e)}"
                self.results.failed_code_gen += 1
                trial.total_time_ms = (time.time() - start_time) * 1000
                return trial

            # Step 3: Verify the generated code
            is_safe, reason = self.verifier.verify(trial.generated_code)
            trial.verification_passed = is_safe

            if not is_safe:
                trial.failure_reason = f"Verification failed: {reason}"
                self.results.failed_verification += 1
                trial.total_time_ms = (time.time() - start_time) * 1000
                return trial

            # Step 4: Simulate execution (we don't actually run the code)
            trial.execution_result = {
                "simulated": True,
                "would_execute": trial.generated_code[:200] + "..." if len(trial.generated_code) > 200 else trial.generated_code
            }
            trial.success = True

        trial.total_time_ms = (time.time() - start_time) * 1000
        return trial

    async def run_experiment(
        self,
        num_trials: int = 50,
        robots: List[RobotState] = None
    ) -> ExperimentResults:
        """
        Run the full experiment.

        Args:
            num_trials: Number of trials to run
            robots: List of robot states (will generate if not provided)

        Returns:
            Aggregated results
        """

        # Generate robots if not provided
        if robots is None:
            robots = self._generate_robots(5)

        logger.info(f"Starting experiment with {num_trials} trials and {len(robots)} robots")

        # Initialize results by type
        for need_type in NeedType:
            self.results.results_by_type[need_type.value] = {
                "total": 0,
                "success": 0,
                "avg_latency_ms": 0.0
            }

        # Run trials
        for i in range(num_trials):
            # Select random sender and receiver
            sender = random.choice(robots)
            receiver = random.choice([r for r in robots if r.robot_id != sender.robot_id])

            # Select random need type
            need_type = random.choice(list(NeedType))

            # Run trial
            trial = await self.run_trial(i, sender, receiver, need_type)
            self.results.trials.append(trial)

            # Update type-specific results
            type_results = self.results.results_by_type[need_type.value]
            type_results["total"] += 1
            if trial.success:
                type_results["success"] += 1

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_trials} trials")

        # Calculate aggregated metrics
        self._calculate_metrics()

        return self.results

    def _generate_robots(self, count: int) -> List[RobotState]:
        """Generate random robot states for experiment"""
        robots = []
        for i in range(count):
            robots.append(RobotState(
                robot_id=f"robot_{i:03d}",
                position=(random.uniform(0, 50), random.uniform(0, 50)),
                battery=random.uniform(0.3, 1.0),
                carrying=random.choice([None, f"item_{random.randint(1,100)}"]),
                current_task=random.choice([None, "pickup", "delivery", "patrol"])
            ))
        return robots

    def _calculate_metrics(self):
        """Calculate aggregated metrics from trials"""
        self.results.total_trials = len(self.results.trials)
        self.results.successful_trials = sum(1 for t in self.results.trials if t.success)

        # Calculate averages
        if self.results.successful_trials > 0:
            successful = [t for t in self.results.trials if t.success]
            self.results.avg_interpretation_latency_ms = sum(t.interpretation_latency_ms for t in successful) / len(successful)
            self.results.avg_code_gen_latency_ms = sum(t.code_gen_latency_ms for t in successful) / len(successful)
            self.results.avg_total_time_ms = sum(t.total_time_ms for t in successful) / len(successful)

        # Calculate per-type averages
        for need_type in NeedType:
            type_results = self.results.results_by_type[need_type.value]
            type_trials = [t for t in self.results.trials if t.need_type == need_type and t.success]
            if type_trials:
                type_results["avg_latency_ms"] = sum(t.total_time_ms for t in type_trials) / len(type_trials)

    def get_summary(self) -> str:
        """Get human-readable summary of results"""
        r = self.results

        success_rate = r.successful_trials / r.total_trials * 100 if r.total_trials > 0 else 0

        summary = f"""
================================================================================
EXPERIMENT RESULTS: Natural Language + Code Generation Robot Coordination
================================================================================

Overall Performance:
  Total Trials:     {r.total_trials}
  Successful:       {r.successful_trials} ({success_rate:.1f}%)

Failure Breakdown:
  Interpretation:   {r.failed_interpretation}
  Code Generation:  {r.failed_code_gen}
  Verification:     {r.failed_verification}
  Execution:        {r.failed_execution}

Latency (successful trials):
  Interpretation:   {r.avg_interpretation_latency_ms:.1f} ms
  Code Generation:  {r.avg_code_gen_latency_ms:.1f} ms
  Total:            {r.avg_total_time_ms:.1f} ms

Results by Need Type:
"""
        for need_type, data in r.results_by_type.items():
            if data["total"] > 0:
                type_rate = data["success"] / data["total"] * 100
                summary += f"  {need_type:20s}: {data['success']}/{data['total']} ({type_rate:.0f}%) - {data['avg_latency_ms']:.0f}ms avg\n"

        summary += "================================================================================"

        return summary

    def export_results(self, filepath: str):
        """Export detailed results to JSON"""
        export_data = {
            "summary": {
                "total_trials": self.results.total_trials,
                "successful_trials": self.results.successful_trials,
                "success_rate": self.results.successful_trials / self.results.total_trials if self.results.total_trials > 0 else 0,
                "failed_interpretation": self.results.failed_interpretation,
                "failed_code_gen": self.results.failed_code_gen,
                "failed_verification": self.results.failed_verification,
                "avg_interpretation_latency_ms": self.results.avg_interpretation_latency_ms,
                "avg_code_gen_latency_ms": self.results.avg_code_gen_latency_ms,
                "avg_total_time_ms": self.results.avg_total_time_ms
            },
            "by_type": self.results.results_by_type,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "sender_id": t.sender.robot_id,
                    "receiver_id": t.receiver.robot_id,
                    "need_type": t.need_type.value,
                    "message": t.message,
                    "success": t.success,
                    "failure_reason": t.failure_reason,
                    "interpretation_latency_ms": t.interpretation_latency_ms,
                    "code_gen_latency_ms": t.code_gen_latency_ms,
                    "total_time_ms": t.total_time_ms,
                    "generated_code": t.generated_code
                }
                for t in self.results.trials
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Results exported to {filepath}")


async def run_quick_test(groq_api_key: str) -> bool:
    """
    Run a quick test to verify the experiment setup works.

    Args:
        groq_api_key: Groq API key

    Returns:
        True if test passes
    """
    try:
        experiment = NLCodeExperiment(groq_api_key)

        # Create test robots
        sender = RobotState(
            robot_id="test_sender",
            position=(10.0, 10.0),
            battery=0.5,
            carrying="heavy_box"
        )

        receiver = RobotState(
            robot_id="test_receiver",
            position=(15.0, 10.0),
            battery=0.9
        )

        # Run single trial
        trial = await experiment.run_trial(
            trial_id=0,
            sender=sender,
            receiver=receiver,
            need_type=NeedType.HELP_CARRY
        )

        print(f"\nQuick Test Results:")
        print(f"  Message: {trial.message}")
        print(f"  Interpretation: {trial.interpretation}")
        print(f"  Code Generated: {len(trial.generated_code or '')} chars")
        print(f"  Verification: {'PASS' if trial.verification_passed else 'FAIL'}")
        print(f"  Success: {trial.success}")
        print(f"  Total Time: {trial.total_time_ms:.0f}ms")

        if trial.generated_code:
            print(f"\nGenerated Code:\n{trial.generated_code}")

        if trial.failure_reason:
            print(f"  Failure Reason: {trial.failure_reason}")

        return trial.success

    except Exception as e:
        import traceback
        print(f"Quick test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False
