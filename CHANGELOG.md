# Changelog

All notable changes to Spatial Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository cleanup and organization improvements
- `pyproject.toml` for modern Python packaging
- `CONTRIBUTING.md` with development guidelines
- `scripts/` directory for experiment runners
- `py.typed` marker for PEP 561 compliance
- Comprehensive `.gitignore` patterns

### Changed
- Moved experiment scripts from root to `scripts/` directory
- Reorganized project structure for better maintainability

## [0.1.0] - 2025-11-27

### Added
- **Core Framework**
  - Multi-agent warehouse robotics coordination system
  - LLM-driven spatial reasoning capabilities
  - Configurable experiment runner

- **LLM Integration**
  - Groq API client (`spatial_lab.llm.groq_client`)
  - Google Gemini API client (`spatial_lab.llm.gemini_client`)
  - Llama API client (`spatial_lab.llm.llama_client`)
  - LLM Coordinator with automatic fallback (`spatial_lab.llm.llm_coordinator`)

- **Spatial Reasoning**
  - Warehouse environment simulation
  - Robot fleet simulator with path planning
  - Multi-agent coordination (stub implementation)
  - Spatial metrics collection

- **Experiments**
  - LLM Confidence Calibration experiment (144-trial factorial design)
  - Basic spatial reasoning experiments
  - API connectivity tests

- **Evaluation**
  - Expected Calibration Error (ECE) metric
  - Brier score calculation
  - Statistical analysis (t-tests, ANOVA, correlations)
  - Performance metrics collection

- **Documentation**
  - Comprehensive README with installation and usage
  - API documentation in docstrings
  - Scientific report generation

### Research Findings (v0.1.0)
- LLMs exhibit significant overconfidence in spatial reasoning (ECE=0.209)
- Simple navigation tasks show good calibration (ECE=0.034)
- Uncertainty-aware prompting reduces calibration error by 32%
- Multi-step planning tasks show worst calibration

## [0.0.1] - 2025-11-27

### Added
- Initial project structure
- Basic warehouse environment
- Robot fleet simulation
- Configuration system

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-11-27 | Full LLM integration, calibration experiments |
| 0.0.1 | 2025-11-27 | Initial commit |

## Upgrade Notes

### Upgrading to 0.1.0

1. **New Dependencies**: Install LLM provider packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**: Create `.env` file with API keys:
   ```bash
   cp .env.example .env
   # Add GROQ_API_KEY and GOOGLE_API_KEY
   ```

3. **Script Location**: Experiment scripts moved to `scripts/`:
   ```bash
   # Old: python run_experiment.py
   # New: PYTHONPATH=. python scripts/run_experiment.py
   ```
