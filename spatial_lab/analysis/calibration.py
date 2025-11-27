"""
Calibration Analysis

Computes calibration metrics for LLM confidence assessments.
Implements Expected Calibration Error (ECE), Brier scores,
and adaptive binning methods.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import defaultdict


@dataclass
class CalibrationMetrics:
    """Complete calibration metrics for an experiment."""
    ece: float                          # Expected Calibration Error
    ece_stderr: float                   # Standard error of ECE estimate
    brier_score: float                  # Brier score (mean squared error)
    overconfidence_rate: float          # Fraction of overconfident predictions
    underconfidence_rate: float         # Fraction of underconfident predictions
    mean_confidence: float              # Average confidence
    mean_accuracy: float                # Average accuracy
    calibration_slope: float            # Regression slope (1.0 = perfect)
    calibration_intercept: float        # Regression intercept (0.0 = perfect)
    n_samples: int                      # Total samples
    n_bins_used: int                    # Number of non-empty bins
    bin_data: List[Dict[str, Any]]      # Per-bin statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ece": round(self.ece, 4),
            "ece_stderr": round(self.ece_stderr, 4),
            "brier_score": round(self.brier_score, 4),
            "overconfidence_rate": round(self.overconfidence_rate, 4),
            "underconfidence_rate": round(self.underconfidence_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "calibration_slope": round(self.calibration_slope, 4),
            "calibration_intercept": round(self.calibration_intercept, 4),
            "n_samples": self.n_samples,
            "n_bins_used": self.n_bins_used,
            "bin_data": self.bin_data,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return f"""Calibration Summary (n={self.n_samples}):
  ECE: {self.ece:.3f} (±{self.ece_stderr:.3f})
  Brier Score: {self.brier_score:.3f}
  Mean Confidence: {self.mean_confidence:.1%}
  Mean Accuracy: {self.mean_accuracy:.1%}
  Overconfidence Rate: {self.overconfidence_rate:.1%}
  Calibration Slope: {self.calibration_slope:.2f}
  Non-empty Bins: {self.n_bins_used}"""


def compute_ece(
    confidences: List[float],
    accuracies: List[int],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error with equal-width binning.

    Args:
        confidences: List of confidence scores (0.0 to 1.0).
        accuracies: List of binary outcomes (0 or 1).
        n_bins: Number of bins.

    Returns:
        Tuple of (ECE value, list of bin statistics).
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidence and accuracy lists must have same length")

    if len(confidences) == 0:
        return 0.0, []

    # Initialize bins
    bins = [[] for _ in range(n_bins)]

    # Assign samples to bins
    for conf, acc in zip(confidences, accuracies):
        # Clamp confidence to [0, 1]
        conf = max(0.0, min(1.0, conf))
        # Determine bin index
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, acc))

    # Compute ECE
    total_samples = len(confidences)
    ece = 0.0
    bin_data = []

    for i, bin_samples in enumerate(bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        if len(bin_samples) == 0:
            bin_data.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "count": 0,
                "mean_confidence": None,
                "mean_accuracy": None,
                "gap": None,
            })
            continue

        bin_confs = [s[0] for s in bin_samples]
        bin_accs = [s[1] for s in bin_samples]

        mean_conf = sum(bin_confs) / len(bin_confs)
        mean_acc = sum(bin_accs) / len(bin_accs)
        gap = abs(mean_acc - mean_conf)

        # Weight by bin size
        weight = len(bin_samples) / total_samples
        ece += weight * gap

        bin_data.append({
            "bin_lower": round(bin_lower, 2),
            "bin_upper": round(bin_upper, 2),
            "count": len(bin_samples),
            "mean_confidence": round(mean_conf, 4),
            "mean_accuracy": round(mean_acc, 4),
            "gap": round(gap, 4),
        })

    return ece, bin_data


def compute_adaptive_ece(
    confidences: List[float],
    accuracies: List[int],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute ECE with equal-mass (adaptive) binning.

    Bins are created to have approximately equal numbers of samples,
    which is more robust when confidence distribution is skewed.

    Args:
        confidences: List of confidence scores (0.0 to 1.0).
        accuracies: List of binary outcomes (0 or 1).
        n_bins: Target number of bins.

    Returns:
        Tuple of (ECE value, list of bin statistics).
    """
    if len(confidences) == 0:
        return 0.0, []

    # Sort by confidence
    paired = sorted(zip(confidences, accuracies), key=lambda x: x[0])

    # Create equal-mass bins
    n = len(paired)
    bin_size = max(1, n // n_bins)

    bins = []
    for i in range(0, n, bin_size):
        end = min(i + bin_size, n)
        bins.append(paired[i:end])

    # Merge small final bin
    if len(bins) > 1 and len(bins[-1]) < bin_size // 2:
        bins[-2].extend(bins[-1])
        bins = bins[:-1]

    # Compute ECE
    total_samples = len(confidences)
    ece = 0.0
    bin_data = []

    for bin_samples in bins:
        confs = [s[0] for s in bin_samples]
        accs = [s[1] for s in bin_samples]

        mean_conf = sum(confs) / len(confs)
        mean_acc = sum(accs) / len(accs)
        gap = abs(mean_acc - mean_conf)

        weight = len(bin_samples) / total_samples
        ece += weight * gap

        bin_data.append({
            "bin_lower": round(min(confs), 4),
            "bin_upper": round(max(confs), 4),
            "count": len(bin_samples),
            "mean_confidence": round(mean_conf, 4),
            "mean_accuracy": round(mean_acc, 4),
            "gap": round(gap, 4),
        })

    return ece, bin_data


def compute_brier_score(
    confidences: List[float],
    accuracies: List[int],
) -> float:
    """
    Compute Brier score (mean squared error of confidence).

    Args:
        confidences: List of confidence scores (0.0 to 1.0).
        accuracies: List of binary outcomes (0 or 1).

    Returns:
        Brier score (lower is better, 0.0 is perfect).
    """
    if len(confidences) == 0:
        return 0.0

    mse = sum(
        (conf - acc) ** 2
        for conf, acc in zip(confidences, accuracies)
    ) / len(confidences)

    return mse


class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis.

    Computes multiple calibration metrics and provides
    statistical summaries.
    """

    def __init__(
        self,
        n_bins: int = 10,
        use_adaptive_binning: bool = True,
    ):
        """
        Initialize the analyzer.

        Args:
            n_bins: Number of bins for ECE computation.
            use_adaptive_binning: Use equal-mass binning.
        """
        self.n_bins = n_bins
        self.use_adaptive_binning = use_adaptive_binning

    def analyze(
        self,
        confidences: List[float],
        accuracies: List[int],
    ) -> CalibrationMetrics:
        """
        Perform complete calibration analysis.

        Args:
            confidences: List of confidence scores (0.0 to 1.0).
            accuracies: List of binary outcomes (0 or 1).

        Returns:
            CalibrationMetrics with all computed metrics.
        """
        if len(confidences) == 0:
            return self._empty_metrics()

        # Normalize confidences to [0, 1]
        confidences = [
            c / 100.0 if c > 1.0 else c
            for c in confidences
        ]

        # Compute ECE
        if self.use_adaptive_binning:
            ece, bin_data = compute_adaptive_ece(
                confidences, accuracies, self.n_bins
            )
        else:
            ece, bin_data = compute_ece(
                confidences, accuracies, self.n_bins
            )

        # Compute standard error via bootstrap
        ece_stderr = self._bootstrap_ece_stderr(confidences, accuracies)

        # Compute Brier score
        brier = compute_brier_score(confidences, accuracies)

        # Compute over/underconfidence rates
        overconf_count = 0
        underconf_count = 0
        for conf, acc in zip(confidences, accuracies):
            if conf > acc:  # Overconfident if confidence > actual outcome
                overconf_count += 1
            elif conf < acc:
                underconf_count += 1

        n = len(confidences)
        overconf_rate = overconf_count / n
        underconf_rate = underconf_count / n

        # Compute calibration regression
        slope, intercept = self._fit_calibration_line(confidences, accuracies)

        # Count non-empty bins
        n_bins_used = sum(1 for b in bin_data if b["count"] > 0)

        return CalibrationMetrics(
            ece=ece,
            ece_stderr=ece_stderr,
            brier_score=brier,
            overconfidence_rate=overconf_rate,
            underconfidence_rate=underconf_rate,
            mean_confidence=sum(confidences) / n,
            mean_accuracy=sum(accuracies) / n,
            calibration_slope=slope,
            calibration_intercept=intercept,
            n_samples=n,
            n_bins_used=n_bins_used,
            bin_data=bin_data,
        )

    def analyze_by_condition(
        self,
        trials: List[Dict[str, Any]],
        condition_key: str,
    ) -> Dict[str, CalibrationMetrics]:
        """
        Analyze calibration separately for each condition.

        Args:
            trials: List of trial dictionaries.
            condition_key: Key to group trials by.

        Returns:
            Dictionary mapping condition values to metrics.
        """
        # Group trials by condition
        groups = defaultdict(list)
        for trial in trials:
            condition_value = trial.get(condition_key, "unknown")
            groups[condition_value].append(trial)

        # Analyze each group
        results = {}
        for condition_value, group_trials in groups.items():
            confidences = []
            accuracies = []

            for trial in group_trials:
                conf = trial.get("confidence", trial.get("llm_confidence"))
                acc = trial.get("accuracy", trial.get("selection_correct", 0))

                if conf is not None:
                    confidences.append(conf)
                    accuracies.append(int(acc) if acc else 0)

            if confidences:
                results[str(condition_value)] = self.analyze(confidences, accuracies)

        return results

    def _bootstrap_ece_stderr(
        self,
        confidences: List[float],
        accuracies: List[int],
        n_bootstrap: int = 100,
    ) -> float:
        """Estimate ECE standard error via bootstrap."""
        import random

        if len(confidences) < 10:
            return 0.0

        n = len(confidences)
        ece_values = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = [random.randint(0, n - 1) for _ in range(n)]
            boot_conf = [confidences[i] for i in indices]
            boot_acc = [accuracies[i] for i in indices]

            if self.use_adaptive_binning:
                ece, _ = compute_adaptive_ece(boot_conf, boot_acc, self.n_bins)
            else:
                ece, _ = compute_ece(boot_conf, boot_acc, self.n_bins)

            ece_values.append(ece)

        # Compute standard deviation
        mean_ece = sum(ece_values) / len(ece_values)
        variance = sum((e - mean_ece) ** 2 for e in ece_values) / len(ece_values)
        return math.sqrt(variance)

    def _fit_calibration_line(
        self,
        confidences: List[float],
        accuracies: List[int],
    ) -> Tuple[float, float]:
        """Fit linear regression line to calibration data."""
        n = len(confidences)
        if n < 2:
            return 1.0, 0.0

        # Simple linear regression
        mean_x = sum(confidences) / n
        mean_y = sum(accuracies) / n

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(confidences, accuracies)
        )
        denominator = sum((x - mean_x) ** 2 for x in confidences)

        if abs(denominator) < 1e-10:
            slope = 0.0
        else:
            slope = numerator / denominator

        intercept = mean_y - slope * mean_x

        return slope, intercept

    def _empty_metrics(self) -> CalibrationMetrics:
        """Return empty metrics when no data."""
        return CalibrationMetrics(
            ece=0.0,
            ece_stderr=0.0,
            brier_score=0.0,
            overconfidence_rate=0.0,
            underconfidence_rate=0.0,
            mean_confidence=0.0,
            mean_accuracy=0.0,
            calibration_slope=1.0,
            calibration_intercept=0.0,
            n_samples=0,
            n_bins_used=0,
            bin_data=[],
        )


def separate_api_failures(
    trials: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate successful trials from API failures.

    IMPORTANT: API failures should NOT be included in calibration analysis
    as they don't represent model performance.

    Args:
        trials: All trial data.

    Returns:
        Tuple of (successful_trials, failed_trials).
    """
    successful = []
    failed = []

    for trial in trials:
        # Check various failure indicators
        is_failure = (
            trial.get("api_error") or
            trial.get("api_failure") or
            trial.get("status") == "api_error" or
            trial.get("error") is not None or
            trial.get("confidence") is None
        )

        if is_failure:
            failed.append(trial)
        else:
            successful.append(trial)

    return successful, failed
