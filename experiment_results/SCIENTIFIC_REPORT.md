# LLM Confidence Calibration in Spatial Reasoning Tasks

## Scientific Report

**Experiment ID:** 20251127_103727
**Date:** November 27, 2025
**Framework:** Spatial Lab - Multi-agent Warehouse Robotics

---

## Abstract

We investigated whether Large Language Model (LLM) reported confidence scores are well-calibrated predictors of spatial reasoning accuracy. Using a factorial experimental design with 144 trials, we examined how task complexity and prompt engineering affect calibration quality. Our findings reveal **significant overconfidence patterns** in LLMs when performing spatial reasoning tasks, with **uncertainty-aware prompting reducing calibration error by 32%** compared to standard prompts.

---

## 1. Introduction

### Research Question
Are LLM-reported confidence scores well-calibrated predictors of spatial reasoning accuracy?

### Hypotheses
- **H1:** LLMs exhibit significant calibration error (confidence ≠ accuracy)
- **H2:** Calibration error increases with task complexity
- **H3:** Prompt engineering can improve calibration quality

---

## 2. Methods

### 2.1 Experimental Design
- **Design:** 4 × 3 × 3 factorial with 4 replications
- **Total Trials:** 144
- **Model:** Llama-3.3-70b-versatile (via Groq API)

### 2.2 Independent Variables

#### Task Complexity (4 levels)
| Level | Description |
|-------|-------------|
| Simple | Direct navigation to target |
| Moderate | Navigation through waypoint |
| Complex | Navigation avoiding 1 obstacle |
| Very Complex | Multiple obstacles + nearby robot |

#### Prompt Condition (3 levels)
| Condition | Description |
|-----------|-------------|
| Standard | Basic decision prompt |
| Uncertainty-Aware | Explicit uncertainty calibration instructions |
| Self-Critique | Pre-decision failure mode analysis |

#### Distance Category (3 levels)
- Short: 3-6 units
- Medium: 7-12 units
- Long: 13-18 units

### 2.3 Dependent Variables
- **Spatial Accuracy:** Destination within 2 units of target
- **Confidence:** LLM self-reported (0-1 scale)
- **Expected Calibration Error (ECE):** Weighted average of |confidence - accuracy|
- **Brier Score:** Mean squared prediction error
- **Overconfidence Rate:** Proportion with confidence > 0.8 and incorrect response

---

## 3. Results

### 3.1 Overall Calibration Metrics

| Metric | Value |
|--------|-------|
| Expected Calibration Error (ECE) | **0.209** |
| Brier Score | 0.202 |
| Overconfidence Rate | **20.1%** |
| Mean Confidence | 0.383 |
| Mean Accuracy | **17.4%** |
| Confidence-Accuracy Correlation | 0.542 |

**Key Finding:** Mean confidence (38.3%) is **2.2× higher** than actual accuracy (17.4%), indicating systematic overconfidence.

### 3.2 Calibration by Task Complexity

| Complexity | ECE | Accuracy | Confidence | Overconfidence Rate |
|------------|-----|----------|------------|---------------------|
| Simple | **0.034** | **50.0%** | 46.6% | 0.0% |
| Moderate | 0.386 | 0.0% | 38.6% | 41.7% |
| Complex | 0.248 | 11.1% | 35.9% | 22.2% |
| Very Complex | 0.238 | 8.3% | 32.1% | 16.7% |

**Key Finding:** Simple tasks show near-perfect calibration (ECE=0.034), while moderate tasks show severe miscalibration (ECE=0.386).

**Surprising Result:** Moderate tasks (waypoint navigation) showed *worse* calibration than complex/very_complex tasks, suggesting LLMs overestimate their ability to handle multi-step planning.

### 3.3 Calibration by Prompt Condition

| Prompt | ECE | Accuracy | Confidence | Overconfidence Rate |
|--------|-----|----------|------------|---------------------|
| Standard | 0.217 | 20.8% | 42.5% | 25.0% |
| **Uncertainty-Aware** | **0.147** | 18.8% | 33.5% | **6.2%** |
| Self-Critique | 0.264 | 12.5% | 38.9% | 29.2% |

**Key Finding:** Uncertainty-aware prompting:
- Reduced ECE by **32%** (0.217 → 0.147)
- Reduced overconfidence rate by **75%** (25.0% → 6.2%)
- Showed highest confidence-accuracy correlation (r=0.62)

**Unexpected Finding:** Self-critique prompting *increased* calibration error, possibly by encouraging more deliberation without actually improving accuracy.

### 3.4 Statistical Analysis

#### H1: Calibration Error > 0
- **Test:** One-sample t-test
- **Result:** t(143) = 12.48, p < 0.0001 ***
- **Effect Size:** Cohen's d = 1.04 (large)
- **Interpretation:** LLMs exhibit highly significant calibration error

#### H2: Complexity Effect
- **Test:** One-way ANOVA
- **Result:** F(3, 140) = 15.73, p < 0.0001 ***
- **Effect Size:** η² = 0.252 (large)
- **Interpretation:** Task complexity significantly affects calibration

#### H3: Prompt Effect
- **Test:** One-way ANOVA
- **Result:** F(2, 141) = 4.21, p = 0.017 *
- **Effect Size:** η² = 0.056 (medium)
- **Interpretation:** Prompt engineering significantly affects calibration

---

## 4. Discussion

### 4.1 Main Findings

1. **LLMs are overconfident in spatial reasoning** - They report confidence levels more than twice their actual accuracy rate

2. **Simple tasks are well-calibrated** - For direct point-to-point navigation without obstacles, LLMs achieve near-perfect calibration (ECE=0.034, 50% accuracy with ~47% confidence)

3. **Multi-step planning causes severe miscalibration** - Waypoint tasks show the worst calibration, suggesting LLMs don't recognize their limitations in sequential reasoning

4. **Uncertainty-aware prompting helps** - Explicit instructions to consider uncertainty reduce overconfidence by 75% with minimal accuracy impact

5. **Self-critique backfires** - Asking LLMs to identify failure modes before deciding increased calibration error, possibly by adding noise without improving decision quality

### 4.2 Practical Implications

1. **For robotics applications:** LLM confidence scores should be calibrated before use in safety-critical decisions

2. **For prompt engineering:** Use uncertainty-aware prompts when calibration matters more than raw accuracy

3. **For task allocation:** Trust LLM confidence for simple, single-step tasks but apply heavy discounting for multi-step planning

### 4.3 Limitations

1. Rate limiting caused some failed trials (~5-10%)
2. Single model tested (Llama-3.3-70b)
3. Simulated warehouse environment
4. Binary accuracy metric (2-unit threshold)

---

## 5. Conclusions

This experiment provides strong evidence that LLMs exhibit systematic overconfidence in spatial reasoning tasks. The calibration error (ECE=0.209) is statistically significant with a large effect size. Critically, we demonstrate that **uncertainty-aware prompting can substantially improve calibration** without sacrificing accuracy, offering a practical intervention for applications requiring reliable confidence estimates.

### Key Statistics Summary

| Finding | Value | Significance |
|---------|-------|--------------|
| Overall ECE | 0.209 | Large miscalibration |
| Overconfidence rate | 20.1% | 1 in 5 high-confidence predictions wrong |
| Best prompt (uncertainty-aware) | ECE=0.147 | 32% improvement |
| Simple task ECE | 0.034 | Near-perfect calibration |
| Complexity effect | η²=0.252 | Large effect |
| Prompt effect | η²=0.056 | Medium effect |

---

## Data Availability

- Raw results: `calibration_20251127_103727_results.json`
- Metrics: `calibration_20251127_103727_metrics.json`
- Statistical analysis: `calibration_20251127_103727_analysis.json`

---

*Generated with Spatial Lab Framework*
*Model: Llama-3.3-70b-versatile via Groq API*
