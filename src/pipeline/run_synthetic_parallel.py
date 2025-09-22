"""Parallel pipeline for synthetic experiments."""

import argparse
from pathlib import Path
from omegaconf import OmegaConf

from ..modeling.run import GenerationRunner
from ..modeling.llm import LLMManager
from ..tasks.synthetic.parallel_runner import ParallelSyntheticRunner
from ..common.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Run parallel synthetic survival experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    logger = setup_logging()

    # Setup experiment directory
    experiment_name = config.experiment_name
    out_dir = Path(config.logging.out_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info(f"Starting parallel synthetic experiment: {experiment_name}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Workers: {args.workers}")

    # Initialize model
    llm_manager = LLMManager(
        model_name=config.model.name,
        **config.model.get("model_kwargs", {})
    )
    generation_runner = GenerationRunner(llm_manager)

    logger.info(f"Loaded model: {config.model.name}")

    # Run experiment with parallel processing
    synthetic_runner = ParallelSyntheticRunner(
        generation_runner=generation_runner,
        config=config,
        logger=logger,
        num_workers=args.workers
    )

    # Output file for incremental writes
    raw_output_file = raw_dir / "results.jsonl"

    results = synthetic_runner.run_experiment(output_file=str(raw_output_file))

    logger.info(f"Saved {len(results)} records to {raw_output_file}")
    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()