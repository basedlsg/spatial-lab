"""
Visualization and Reporting

Generates reliability diagrams and calibration reports
in text and markdown formats.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .calibration import CalibrationMetrics, CalibrationAnalyzer


def create_reliability_diagram(
    metrics: CalibrationMetrics,
    title: str = "Reliability Diagram",
    width: int = 50,
) -> str:
    """
    Create ASCII reliability diagram.

    Args:
        metrics: Calibration metrics with bin data.
        title: Diagram title.
        width: Character width for bars.

    Returns:
        ASCII diagram string.
    """
    lines = [
        f"\n{title}",
        "=" * len(title),
        "",
        "Confidence vs Accuracy by Bin",
        "-" * 40,
    ]

    if not metrics.bin_data:
        lines.append("No data available")
        return "\n".join(lines)

    # Header
    lines.append(f"{'Bin':<12} {'Count':>6} {'Conf':>6} {'Acc':>6} {'Gap':>6}")
    lines.append("-" * 40)

    for bin_info in metrics.bin_data:
        if bin_info["count"] == 0:
            continue

        bin_label = f"{bin_info['bin_lower']:.1f}-{bin_info['bin_upper']:.1f}"
        count = bin_info["count"]
        conf = bin_info["mean_confidence"]
        acc = bin_info["mean_accuracy"]
        gap = bin_info["gap"]

        lines.append(
            f"{bin_label:<12} {count:>6} {conf:>6.2f} {acc:>6.2f} {gap:>6.3f}"
        )

    lines.append("-" * 40)

    # Visual bar chart
    lines.append("")
    lines.append("Visual Calibration (| = confidence, * = accuracy)")
    lines.append("-" * 60)

    for bin_info in metrics.bin_data:
        if bin_info["count"] == 0:
            continue

        bin_label = f"{bin_info['bin_lower']:.1f}-{bin_info['bin_upper']:.1f}"
        conf = bin_info["mean_confidence"]
        acc = bin_info["mean_accuracy"]

        # Create bar
        conf_pos = int(conf * width)
        acc_pos = int(acc * width)

        bar = [" "] * (width + 1)

        # Mark confidence with |
        if 0 <= conf_pos <= width:
            bar[conf_pos] = "|"

        # Mark accuracy with *
        if 0 <= acc_pos <= width:
            if bar[acc_pos] == "|":
                bar[acc_pos] = "X"  # Overlap
            else:
                bar[acc_pos] = "*"

        bar_str = "".join(bar)
        lines.append(f"{bin_label:<8} [{bar_str}]")

    lines.append("")
    lines.append("Legend: | = mean confidence, * = mean accuracy, X = overlap")

    # Perfect calibration line
    lines.append("")
    lines.append("Perfect calibration: confidence = accuracy (diagonal line)")
    lines.append(f"ECE = {metrics.ece:.3f} (lower is better, 0 = perfect)")

    return "\n".join(lines)


def create_calibration_summary(
    metrics: CalibrationMetrics,
    experiment_name: str = "Calibration Experiment",
    include_interpretation: bool = True,
) -> str:
    """
    Create comprehensive calibration summary report.

    Args:
        metrics: Calibration metrics.
        experiment_name: Name for the report header.
        include_interpretation: Include interpretation guidelines.

    Returns:
        Formatted summary string.
    """
    lines = [
        f"\n{'=' * 60}",
        f" {experiment_name}",
        f"{'=' * 60}",
        "",
        "CALIBRATION METRICS",
        "-" * 30,
        f"  Expected Calibration Error (ECE): {metrics.ece:.4f}",
        f"  ECE Standard Error:               {metrics.ece_stderr:.4f}",
        f"  Brier Score:                      {metrics.brier_score:.4f}",
        "",
        "CONFIDENCE STATISTICS",
        "-" * 30,
        f"  Mean Confidence: {metrics.mean_confidence:.1%}",
        f"  Mean Accuracy:   {metrics.mean_accuracy:.1%}",
        f"  Gap:             {abs(metrics.mean_confidence - metrics.mean_accuracy):.1%}",
        "",
        "BIAS ANALYSIS",
        "-" * 30,
        f"  Overconfidence Rate:  {metrics.overconfidence_rate:.1%}",
        f"  Underconfidence Rate: {metrics.underconfidence_rate:.1%}",
        "",
        "CALIBRATION FIT",
        "-" * 30,
        f"  Slope:     {metrics.calibration_slope:.3f} (ideal = 1.0)",
        f"  Intercept: {metrics.calibration_intercept:.3f} (ideal = 0.0)",
        "",
        "DATA SUMMARY",
        "-" * 30,
        f"  Total Samples:   {metrics.n_samples}",
        f"  Non-empty Bins:  {metrics.n_bins_used}",
    ]

    if include_interpretation:
        lines.extend([
            "",
            "INTERPRETATION GUIDE",
            "-" * 30,
            "",
            "ECE (Expected Calibration Error):",
            "  < 0.05: Excellent calibration",
            "  0.05 - 0.10: Good calibration",
            "  0.10 - 0.20: Moderate miscalibration",
            "  > 0.20: Poor calibration",
            "",
            "Calibration Slope:",
            "  = 1.0: Perfect calibration slope",
            "  < 1.0: Overconfident (confidence > accuracy)",
            "  > 1.0: Underconfident (confidence < accuracy)",
            "",
            f"Assessment: ",
        ])

        # Add assessment
        if metrics.ece < 0.05:
            assessment = "WELL CALIBRATED"
        elif metrics.ece < 0.10:
            assessment = "REASONABLY CALIBRATED"
        elif metrics.ece < 0.20:
            assessment = "MODERATELY MISCALIBRATED"
        else:
            assessment = "POORLY CALIBRATED"

        if metrics.overconfidence_rate > 0.6:
            assessment += " - OVERCONFIDENT"
        elif metrics.underconfidence_rate > 0.6:
            assessment += " - UNDERCONFIDENT"

        lines.append(f"  >> {assessment} <<")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def export_analysis_report(
    metrics: CalibrationMetrics,
    experiment_config: Dict[str, Any],
    trials: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Export comprehensive analysis report in Markdown format.

    Args:
        metrics: Overall calibration metrics.
        experiment_config: Experiment configuration.
        trials: Raw trial data.
        output_path: Optional path to save report.

    Returns:
        Markdown report string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Calibration Experiment Report

**Generated:** {timestamp}

## Executive Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ECE | {metrics.ece:.4f} | {'Good' if metrics.ece < 0.1 else 'Needs Improvement'} |
| Brier Score | {metrics.brier_score:.4f} | {'Low' if metrics.brier_score < 0.25 else 'High'} |
| Mean Confidence | {metrics.mean_confidence:.1%} | |
| Mean Accuracy | {metrics.mean_accuracy:.1%} | |
| Overconfidence Rate | {metrics.overconfidence_rate:.1%} | {'Significant' if metrics.overconfidence_rate > 0.5 else 'Acceptable'} |

## Experiment Configuration

```json
{_format_config(experiment_config)}
```

## Bin-Level Analysis

| Bin Range | Count | Mean Conf | Mean Acc | Gap |
|-----------|-------|-----------|----------|-----|
"""

    for bin_info in metrics.bin_data:
        if bin_info["count"] > 0:
            report += (
                f"| {bin_info['bin_lower']:.2f}-{bin_info['bin_upper']:.2f} | "
                f"{bin_info['count']} | "
                f"{bin_info['mean_confidence']:.3f} | "
                f"{bin_info['mean_accuracy']:.3f} | "
                f"{bin_info['gap']:.3f} |\n"
            )

    report += f"""
## Reliability Diagram (ASCII)

```
{create_reliability_diagram(metrics, title="", width=40)}
```

## Statistical Summary

- **Calibration Slope:** {metrics.calibration_slope:.3f}
  - Interpretation: {'Well-calibrated' if 0.8 < metrics.calibration_slope < 1.2 else 'Miscalibrated'}

- **ECE Standard Error:** {metrics.ece_stderr:.4f}
  - 95% CI: [{metrics.ece - 1.96*metrics.ece_stderr:.4f}, {metrics.ece + 1.96*metrics.ece_stderr:.4f}]

## Key Findings

"""

    # Generate key findings
    findings = []

    if metrics.ece > 0.15:
        findings.append(
            f"1. **High miscalibration detected** (ECE = {metrics.ece:.3f}). "
            "The model's confidence does not align well with actual accuracy."
        )

    if metrics.overconfidence_rate > 0.6:
        findings.append(
            f"2. **Systematic overconfidence** ({metrics.overconfidence_rate:.1%} of predictions). "
            "The model tends to be more confident than warranted."
        )

    if metrics.mean_accuracy < 0.3:
        findings.append(
            f"3. **Low task accuracy** ({metrics.mean_accuracy:.1%}). "
            "Consider if the task is too difficult or if there are design issues."
        )

    gap = abs(metrics.mean_confidence - metrics.mean_accuracy)
    if gap > 0.2:
        findings.append(
            f"4. **Large confidence-accuracy gap** ({gap:.1%}). "
            f"Mean confidence ({metrics.mean_confidence:.1%}) differs significantly "
            f"from mean accuracy ({metrics.mean_accuracy:.1%})."
        )

    if not findings:
        findings.append("No significant calibration issues detected.")

    report += "\n".join(findings)

    report += f"""

## Methodology Notes

- ECE computed using {'adaptive (equal-mass)' if len(metrics.bin_data) > 0 else 'equal-width'} binning
- Total samples: {metrics.n_samples}
- Non-empty bins: {metrics.n_bins_used}

---

*Report generated by Spatial Lab Calibration Analysis*
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def _format_config(config: Dict[str, Any]) -> str:
    """Format config dict as JSON-like string."""
    import json
    return json.dumps(config, indent=2, default=str)


def create_condition_comparison(
    condition_metrics: Dict[str, CalibrationMetrics],
    condition_name: str = "Condition",
) -> str:
    """
    Create comparison table across conditions.

    Args:
        condition_metrics: Metrics per condition.
        condition_name: Name of the condition variable.

    Returns:
        Formatted comparison string.
    """
    lines = [
        f"\n{condition_name} Comparison",
        "=" * 50,
        "",
        f"{'Condition':<15} {'ECE':>8} {'Brier':>8} {'Acc':>8} {'Conf':>8}",
        "-" * 50,
    ]

    for condition, metrics in sorted(condition_metrics.items()):
        lines.append(
            f"{condition:<15} {metrics.ece:>8.3f} {metrics.brier_score:>8.3f} "
            f"{metrics.mean_accuracy:>8.1%} {metrics.mean_confidence:>8.1%}"
        )

    lines.append("-" * 50)

    return "\n".join(lines)
