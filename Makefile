setup:
	pip install -e .

setup-dev:
	pip install -e ".[dev]"

# Synthetic experiments
synthetic-terse:
	python -m src.pipeline.run_synthetic --config configs/synthetic/terse.yaml

synthetic-verbose:
	python -m src.pipeline.run_synthetic --config configs/synthetic/verbose.yaml

# Software experiments
software-humaneval:
	python -m src.pipeline.run_software --config configs/software/humaneval_incremental.yaml

# Dialog experiments
dialog-qa-clean:
	python -m src.pipeline.run_dialog --config configs/dialog/qa_clean.yaml

# Analysis
figures:
	python -m src.reporting.figures

# Development
test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true

.PHONY: setup setup-dev test lint format typecheck clean figures