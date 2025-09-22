"""Main pipeline for synthetic experiments with vLLM support."""

import argparse
from pathlib import Path

from ..common.config import load_config, get_output_dir, save_config
from ..common.logging import setup_logging
from ..common.rng import set_seed
from ..common.io import save_jsonl
from ..modeling.llm import LLMManager
from ..modeling.run import GenerationRunner
from ..modeling.vllm_manager import VLLMGenerationRunner
from ..tasks.synthetic.runner import SyntheticRunner


def main():
    parser = argparse.ArgumentParser(description="Run synthetic survival experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Configuration overrides in dotlist format"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)

    output_dir = get_output_dir(config)
    logger = setup_logging(
        log_file=str(output_dir / "experiment.log"),
        level=config.get("logging", {}).get("level", "INFO")
    )

    set_seed(config["seed"])

    logger.info(f"Starting synthetic experiment: {config.experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    save_config(config, str(output_dir / "config.yaml"))

    # Detect backend and create appropriate generation runner
    backend = config.model.get("backend", "huggingface").lower()

    if backend == "vllm":
        logger.info("Attempting to use vLLM backend for high-performance inference")

        try:
            # Try to import and use vLLM
            from ..modeling.vllm_manager import VLLMGenerationRunner

            # Extract vLLM-specific parameters
            vllm_kwargs = config.model.get("vllm", {})

            generation_runner = VLLMGenerationRunner(
                model_name=config.model.name,
                **vllm_kwargs
            )
            logger.info(f"Successfully initialized vLLM backend")

        except ImportError as e:
            logger.warning(f"vLLM not available ({e}), falling back to HuggingFace backend")
            backend = "huggingface"

        except Exception as e:
            logger.warning(f"vLLM initialization failed ({e}), falling back to HuggingFace backend")
            backend = "huggingface"

    if backend == "huggingface":
        logger.info("Using HuggingFace backend")

        llm_manager = LLMManager(
            model_name=config.model.name,
            **config.model.get("model_kwargs", {})
        )
        generation_runner = GenerationRunner(llm_manager)

    logger.info(f"Loaded model: {config.model.name} (backend: {backend})")

    synthetic_runner = SyntheticRunner(
        generation_runner=generation_runner,
        config=config,
        logger=logger
    )

    # Set up incremental output file
    raw_output_file = output_dir / "raw" / "results.jsonl"
    raw_output_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing file if it exists
    if raw_output_file.exists():
        raw_output_file.unlink()

    logger.info(f"Results will be written incrementally to: {raw_output_file}")

    results = synthetic_runner.run_experiment(output_file=str(raw_output_file))

    logger.info(f"Saved {len(results)} records to {raw_output_file}")
    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()