# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements survival modeling research for LLM context degradation, focusing on three experimental tracks:

1. **Synthetic tasks** - Function composition (pointer-chasing, arithmetic) with verifiable ground truth
2. **Software tasks** - Incremental coding with staged specs and test validation
3. **Dialog tasks** - Multi-turn Q&A with qualitative degradation assessment

The core research hypothesis: survival dynamics from synthetic tasks can proxy natural dialog survival, with redundancy explaining why real dialogs survive longer than synthetic ones.

## Architecture

Based on the planned structure in `docs/seed-repo.md`, the codebase will follow this organization:

- `src/common/` - Shared utilities (I/O, config, logging, tokenization, RNG)
- `src/modeling/` - LLM interface and generation API with streaming token capture
- `src/scoring/` - Task-specific evaluation (exact match, unit tests, QA scoring)
- `src/survival/` - Survival analysis toolkit (Kaplan-Meier, Cox, plotting)
- `src/tasks/` - Experiment implementations by track (synthetic/software/dialog)
- `src/pipeline/` - Main execution scripts per track
- `src/reporting/` - Paper-ready figures and tables generation
- `configs/` - YAML configurations split by experiment track
- `results/` - Raw logs, processed data, and figures organized by track

## Data Schema

**Raw experimental logs** (JSONL format):
- `task_family`: "synthetic|software|dialog"
- `condition`: experimental condition (e.g., "terse|verbose|redundant")
- `depth_step`: composition/turn index
- `tokens_in/out/total`: token consumption tracking
- `correct`: boolean success/failure
- `failure_reason`: categorized failure modes

**Survival analysis tables** (CSV/Parquet):
- `time_var`: "step" (sequential depth) or "tokens_total" (context size)
- `event`: binary failure indicator
- Grouping variables: `condition`, `task_family`, `seed`, `model`

## Key Research Concepts

- **Steps vs Tokens**: Steps measure pure sequential depth, tokens capture depth + redundancy
- **Survival Analysis**: Uses Kaplan-Meier curves and Cox models to quantify degradation patterns
- **Redundancy Hypothesis**: Token-based survival should be more forgiving than step-based survival in natural tasks (but not synthetic ones)
- **Three Evidence Levels**: Synthetic (objective) → Software (semi-objective) → Dialog (subjective)

## Recommended Models

**For all three experimental tracks (synthetic/software/dialog):**
- **Primary**: `Qwen/Qwen2.5-3B-Instruct` - Fast, capable generalist model
- **Alternative**: `Qwen/Qwen2.5-7B-Instruct` - More capable but slower
- **Not recommended**: Specialized models (Qwen2.5-Math, Qwen2.5-Coder) as they don't generalize across all tasks

**Expected failure depths:**
- Synthetic arithmetic chains: 16-30 steps
- Software incremental tasks: Variable based on complexity
- Dialog multi-turn: Variable based on context accumulation

## Development Guidelines

- All experiments should emit data conforming to the common schemas above
- Survival analysis should support both step-based and token-based time axes
- Configuration files should enable reproducible experiment runs
- Results should be organized by track for systematic comparison
- The modeling layer should abstract LLM choice from task implementations
- Use high token limits (2048+) to avoid artificial failures