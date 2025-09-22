"""Parallel synthetic task runner for GPU optimization."""

import uuid
import json
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

from .gen_arith import generate_chain_instance, calculate_chain_answer
from ...modeling.run import GenerationRunner
from ...scoring.synthetic import score_synthetic_response, determine_failure_reason
from ...common.rng import SeededRNG
from ...common.logging import setup_logging


class ParallelSyntheticRunner:
    """Runs synthetic survival experiments with parallel processing."""

    def __init__(
        self,
        generation_runner: GenerationRunner,
        config: Dict[str, Any],
        logger=None,
        num_workers: int = 3
    ):
        self.generation_runner = generation_runner
        self.config = config
        self.logger = logger or setup_logging()
        self.num_workers = num_workers

    def _run_single_instance(self, run_idx: int, seed_offset: int) -> List[Dict[str, Any]]:
        """Run a single survival chain instance."""
        data_config = self.config["data"]
        model_config = self.config["model"]

        max_depth = data_config["max_depth"]
        condition = data_config["condition"]
        initial_value = data_config.get("initial_value", 1)
        base_seed = data_config.get("seed", 42)

        # Create instance-specific RNG
        instance_seed = base_seed + seed_offset + run_idx
        rng = SeededRNG(instance_seed)
        instance_id = uuid.uuid4().hex[:8]

        instance_results = []
        current_value = initial_value

        for depth in range(1, max_depth + 1):
            prompt, expected, metadata = generate_chain_instance(
                initial_value=current_value,
                depth=depth,
                condition=condition,
                instance_id=instance_id,
                rng=rng
            )

            response, tokens_out, gen_info = self.generation_runner.generate(
                prompt=prompt,
                max_new_tokens=model_config.get("max_new_tokens", 256),
                temperature=model_config.get("temperature", 0.0)
            )

            scores = score_synthetic_response(
                response, expected, metadata
            )

            result = {
                "task_family": "synthetic",
                "condition": condition,
                "seed": instance_seed,
                "instance_id": instance_id,
                "depth_step": depth,
                "depth_max": max_depth,
                "tokens_in": self.generation_runner.count_tokens(prompt),
                "tokens_out": tokens_out,
                "tokens_total": self.generation_runner.count_tokens(prompt) + tokens_out,
                "answer": scores.get("extracted_answer"),
                "gold": expected,
                "correct": scores["exact_match"],
                "failure_reason": determine_failure_reason(scores),
                "time_iso": datetime.utcnow().isoformat() + "Z",
                "model": self.generation_runner.llm_manager.model_name,
                "model_rev": "hf-default",
                "raw_response": response,
                "used_code": False,
                "value_present": scores.get("value_present", True)
            }

            instance_results.append(result)

            # Stop on first failure (natural termination)
            if not scores["exact_match"]:
                break

            # Update current value for next step
            current_value = expected

        return instance_results

    def run_experiment(self, output_file=None) -> List[Dict[str, Any]]:
        """
        Run synthetic survival experiment with parallel processing.

        Args:
            output_file: Optional path to write results incrementally

        Returns list of experimental records in the common schema.
        """
        data_config = self.config["data"]
        n_instances = data_config["n_instances"]
        condition = data_config["condition"]

        self.logger.info(f"Starting parallel synthetic experiment: {condition}")
        self.logger.info(f"Instances: {n_instances}, Workers: {self.num_workers}")

        all_results = []

        # Prepare output file if specified
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Results will be written incrementally to: {output_file}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all instances
            future_to_idx = {
                executor.submit(self._run_single_instance, i, i * 1000): i
                for i in range(n_instances)
            }

            # Process completed instances
            completed_instances = 0
            progress_bar = tqdm(total=n_instances, desc="Instances")

            for future in concurrent.futures.as_completed(future_to_idx):
                instance_idx = future_to_idx[future]
                try:
                    instance_results = future.result()
                    all_results.extend(instance_results)

                    # Write results incrementally
                    if output_file:
                        with open(output_file, "a") as f:
                            for result in instance_results:
                                f.write(json.dumps(result) + "\n")
                                f.flush()

                    completed_instances += 1
                    final_depth = len(instance_results)

                    self.logger.info(f"Instance {completed_instances}/{n_instances} completed: {final_depth} steps, final depth: {final_depth}")
                    progress_bar.update(1)

                except Exception as e:
                    self.logger.error(f"Instance {instance_idx} failed: {e}")

            progress_bar.close()

        self.logger.info(f"Completed experiment with {len(all_results)} records")
        return all_results