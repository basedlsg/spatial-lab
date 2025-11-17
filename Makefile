.PHONY: help install install-dev test test-fast lint format clean demo docs

# Default target
help:
	@echo "Spatial Lab Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-fast     - Run tests without slow tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code with black"
	@echo ""
	@echo "Running:"
	@echo "  make demo          - Run example experiment"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Generate documentation"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run with docker-compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --cov=spatial_lab --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow" --cov=spatial_lab

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

# Code quality
lint:
	@echo "Running ruff..."
	ruff check spatial_lab/ tests/
	@echo "Running mypy..."
	mypy spatial_lab/ --ignore-missing-imports

format:
	@echo "Formatting with black..."
	black spatial_lab/ tests/
	@echo "Sorting imports..."
	ruff check --select I --fix spatial_lab/ tests/

# Running experiments
demo:
	python -m spatial_lab.experiment_runner --help

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation generation requires Sphinx setup"
	@echo "Run: pip install sphinx sphinx-rtd-theme"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Docker
docker-build:
	docker build -t spatial-lab:latest .

docker-run:
	docker-compose up

# Git helpers
git-status:
	@git status

git-push:
	@git add .
	@git commit -m "Update"
	@git push
