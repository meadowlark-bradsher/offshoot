# Offshoot: Survival Modeling of LLM Context Degradation

This repository implements survival modeling research for studying how Large Language Models (LLMs) degrade as dialog context grows. The research focuses on three experimental tracks to understand failure patterns and develop statistical models for predicting when recontextualization or summarization interventions should occur.

## Research Overview

- **Synthetic tasks**: Function composition (arithmetic chains) with verifiable ground truth
- **Software tasks**: Incremental coding with staged specifications and test validation
- **Dialog tasks**: Multi-turn Q&A with qualitative degradation assessment

The core hypothesis: survival dynamics from synthetic tasks can proxy natural dialog survival, with redundancy explaining why real dialogs survive longer than synthetic ones.

## Quick Start

```bash
# Install the package and dependencies
pip install -e .
```
```bash
# Run a test synthetic experiment
make synthetic-terse
```

```bash
# Or run with specific config
python -m src.pipeline.run_synthetic --config configs/synthetic/test.yaml
```

```bash
# Analyze results
python analyze_test_results.py
```

## Project Structure

- `src/common/` - Shared utilities (I/O, config, logging, tokenization, RNG)
- `src/modeling/` - LLM interface and generation API with streaming token capture
- `src/scoring/` - Task-specific evaluation (exact match, unit tests, QA scoring)
- `src/survival/` - Survival analysis toolkit (Kaplan-Meier, Cox, plotting)
- `src/tasks/` - Experiment implementations by track (synthetic/software/dialog)
- `src/pipeline/` - Main execution scripts per track
- `configs/` - YAML configurations split by experiment track
- `results/` - Raw logs, processed data, and figures organized by track

## Key Concepts

### Steps vs Tokens Analysis
- **Steps**: Pure sequential depth (composition steps, dialog turns)
- **Tokens**: Context size including redundancy and clarifications
- **Hypothesis**: Token-based survival should be more forgiving than step-based survival in natural tasks (but not synthetic ones)

### Data Schema
All experiments emit standardized data:
- Raw logs (JSONL): `task_family`, `condition`, `depth_step`, `tokens_in/out/total`, `correct`, `failure_reason`
- Survival tables (CSV/Parquet): `time_var` (step|tokens_total), `event`, grouping variables

### Survival Analysis
- Uses Kaplan-Meier curves and Cox models to quantify degradation patterns
- Supports both step-based and token-based time axes
- Statistical tests for comparing conditions (terse vs verbose vs redundant)

## Example: Synthetic Arithmetic Chains

The synthetic experiment generates arithmetic chains like:
```
Start with 1, then add 1, then multiply by 3, then subtract 2
```

Three conditions test redundancy effects:
- **Terse**: Minimal token usage
- **Verbose**: Detailed explanations
- **Redundant**: Step-by-step breakdowns with verification requests

Expected finding: In synthetic tasks, redundancy doesn't help survival (tokens ≈ steps). In natural tasks, redundancy provides robustness (tokens > steps survival).

## Development

```bash
# Run tests
make test

# Format code
make format

# Check types
make typecheck

# Generate figures
make figures
```

## Configuration

Experiments use YAML configs with inheritance from `configs/global.yaml`. Override any parameters:

```bash
python -m src.pipeline.run_synthetic \
  --config configs/synthetic/terse.yaml \
  --overrides data.n_instances=200 model.name=gpt2
```

## Based on Research Papers

This implements the experimental framework described in:
- **Paper 1**: Survival Modeling of LLM Context Degradation (synthetic → software → dialog evidence ladder)
- **Paper 2**: Fine-Gray Competing Risks for Strategic Dialog Management (future work)