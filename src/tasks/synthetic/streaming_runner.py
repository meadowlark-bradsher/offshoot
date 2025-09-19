"""Streaming synthetic task runner that saves results after each instance."""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .gen_arith import generate_chain_instance, calculate_chain_answer
from ...modeling.run import GenerationRunner
from ...scoring.synthetic import score_synthetic_response, determine_failure_reason
from ...common.rng import SeededRNG
from ...common.logging import setup_logging
from ...common.io import save_jsonl
from pathlib import Path


class StreamingSyntheticRunner:
    """Runs synthetic survival experiments with streaming results."""

    def __init__(
        self,
        generation_runner: GenerationRunner,
        config: Dict[str, Any],
        logger = None
    ):
        self.generation_runner = generation_runner
        self.config = config
        self.logger = logger or setup_logging()

    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run synthetic survival experiment with streaming saves.

        Saves results after each instance completion.
        """
        data_config = self.config["data"]
        model_config = self.config["model"]

        max_depth = data_config["max_depth"]
        n_instances = data_config["n_instances"]
        condition = data_config["condition"]
        initial_value = data_config.get("initial_value", 1)

        # Create output directory
        output_dir = Path(self.config["logging"]["out_dir"]) / self.config["experiment_name"]
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        streaming_file = raw_dir / "streaming_results.jsonl"

        all_results = []

        self.logger.info(f"Starting synthetic experiment: {condition}")
        self.logger.info(f"Max depth: {max_depth}, Instances: {n_instances}")
        self.logger.info(f"Streaming to: {streaming_file}")

        for run_idx in tqdm(range(n_instances), desc="Instances"):
            rng = SeededRNG(self.config["seed"] + run_idx)
            instance_id = str(uuid.uuid4())[:8]
            instance_results = []

            self.logger.info(f"Processing instance {run_idx + 1}/{n_instances} (ID: {instance_id})")

            for depth in range(1, max_depth + 1):
                prompt, expected, metadata = generate_chain_instance(
                    initial_value=initial_value,
                    depth=depth,
                    condition=condition,
                    instance_id=instance_id,
                    rng=rng
                )

                tokens_in = self.generation_runner.count_tokens(prompt)

                try:
                    response, tokens_out, gen_info = self.generation_runner.generate(
                        prompt=prompt,
                        max_new_tokens=model_config.get("max_new_tokens", 256),
                        temperature=model_config.get("temperature", 0.0)
                    )

                    scores = score_synthetic_response(response, expected, metadata)
                    failure_reason = determine_failure_reason(scores)

                    record = {
                        "task_family": "synthetic",
                        "condition": condition,
                        "seed": self.config["seed"] + run_idx,
                        "instance_id": instance_id,
                        "depth_step": depth,
                        "depth_max": max_depth,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "tokens_total": tokens_in + tokens_out,
                        "answer": scores.get("predicted_boxed"),
                        "gold": expected,
                        "correct": scores["exact_match"],
                        "failure_reason": failure_reason,
                        "time_iso": datetime.utcnow().isoformat() + "Z",
                        "model": self.generation_runner.llm_manager.model_name,
                        "model_rev": "hf-default",
                        "raw_response": response.replace("\n", " ").strip(),
                        "used_code": scores["used_code"],
                        "value_present": scores["value_present"],
                    }

                    instance_results.append(record)
                    all_results.append(record)

                    self.logger.info(f"Depth {depth}: {'✓' if scores['exact_match'] else '✗'} "
                                   f"Expected: {expected}, Got: {scores.get('predicted_boxed', 'None')}")

                    # Stop on incorrect answers
                    if not scores["exact_match"]:
                        break

                except Exception as e:
                    self.logger.error(f"Error processing depth {depth}: {e}")
                    break

            # Save after each instance
            save_jsonl(all_results, streaming_file)
            self.logger.info(f"Instance {run_idx + 1} completed at depth {len(instance_results)}. "
                           f"Saved {len(all_results)} total records.")

        self.logger.info(f"Completed experiment with {len(all_results)} records")
        return all_results